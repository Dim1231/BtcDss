from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
import joblib
import os
import json
import io
import csv
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)


# ─── EMA Calculation ─────────────────────────────────────────────────────────
# Implementasi iteratif sesuai rumus Bab II:
#   α = 2 / (n + 1)
#   EMA₁ = Close₁
#   EMAₜ = α × Closeₜ + (1 − α) × EMAₜ₋₁

def calculate_ema(prices, period):
    """Hitung EMA secara iteratif (tanpa pandas ewm) sesuai Persamaan 2.1–2.2."""
    alpha = 2 / (period + 1)
    ema = [prices[0]]          # EMA₁ = Close₁ (inisialisasi)
    for i in range(1, len(prices)):
        ema_t = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        ema.append(ema_t)
    return ema


# ─── Preprocessing & Feature Engineering ─────────────────────────────────────

def prepare_dataset(df):
    """
    Pipeline pra-pemrosesan sesuai Sub-bab 3.3 dan 3.4:
    1. Urutkan kronologis berdasarkan kolom waktu (timeOpen / date / timestamp)
    2. Tangani missing values pada close price (forward fill)
    3. Hitung EMA-50 dan EMA-200 secara iteratif
    4. Bentuk label tren prediktif: Label(t) = bullish jika Close(t+1) > Close(t)
    5. Hapus baris terakhir (tidak memiliki Close(t+1))
    Catatan: warm-up EMA-200 (200 baris pertama) dikecualikan dari TRAINING,
             bukan dari labeling. Total berlabel = 671 baris.
    """
    # ── 1. Identifikasi kolom tanggal ─────────────────────────────────────────
    date_col = None
    for col in ['timeopen', 'date', 'timestamp', 'time', 'timeOpen']:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
        df = df.sort_values(date_col).reset_index(drop=True)

    # ── 2. Identifikasi kolom harga penutupan ─────────────────────────────────
    close_col = None
    for col in ['close', 'price']:
        if col in df.columns:
            close_col = col
            break
    if close_col is None:
        raise ValueError("Kolom harga penutupan (close) tidak ditemukan.")

    # Bersihkan format angka (hapus koma ribuan, tanda kutip)
    df[close_col] = (
        df[close_col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('"', '', regex=False)
        .str.strip()
    )
    df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

    # ── 3. Tangani missing values (Sub-bab 3.3 – forward fill) ───────────────
    df[close_col] = df[close_col].ffill()
    df = df.dropna(subset=[close_col]).reset_index(drop=True)

    # Hapus duplikat timestamp (Sub-bab 3.3 – sub-tahap keempat)
    if date_col:
        df = df.drop_duplicates(subset=[date_col]).reset_index(drop=True)

    prices = df[close_col].tolist()

    # ── 4. Hitung EMA-50 dan EMA-200 (Sub-bab 3.4) ───────────────────────────
    #   Fitur input model: hanya ema50 dan ema200 (2 fitur, sesuai Sub-bab 3.5)
    df['ema50']  = calculate_ema(prices, 50)
    df['ema200'] = calculate_ema(prices, 200)

    # ── 5. Pelabelan tren prediktif (Sub-bab 3.4.3) ───────────────────────────
    #   Label(t) = bullish  jika Close(t+1) > Close(t)
    #   Label(t) = bearish  jika Close(t+1) < Close(t)
    #   Baris ke-672 (Close(t+1) = NaN) → tidak berlabel → dibuang
    labels = []
    for i in range(len(df) - 1):
        if prices[i + 1] > prices[i]:
            labels.append('bullish')
        else:
            labels.append('bearish')
    labels.append(None)          # baris terakhir tidak memiliki Close(t+1)

    df['label'] = labels
    df = df.iloc[:-1].copy()     # buang baris terakhir (671 berlabel tersisa)

    # ── 6. Normalisasi nama kolom untuk output ────────────────────────────────
    rename_map = {close_col: 'close'}
    if date_col:
        rename_map[date_col] = 'date'
    df = df.rename(columns=rename_map)

    return df, close_col, date_col


# ─── Model Training ───────────────────────────────────────────────────────────

def train_model(df, n_estimators=100, warm_up=200):
    """
    Latih Random Forest sesuai Sub-bab 3.5 dan 4.5:
    - Lewati 200 baris warm-up EMA-200 (baris ke-201 dst. dipakai)
    - Fitur X: ema50, ema200  (2 fitur — sesuai Tabel 3.6)
    - Target y: label (bullish / bearish)
    - Stratified split 80:20, random_state=42
    - n_estimators=100, class_weight='balanced', oob_score=True
    """
    # Lewati periode warm-up EMA-200 (Sub-bab 3.4.1 dan Lampiran A)
    df_train_pool = df.iloc[warm_up:].copy() if len(df) > warm_up else df.copy()
    df_train_pool = df_train_pool.dropna(subset=['ema50', 'ema200', 'label'])

    if len(df_train_pool) < 20:
        raise ValueError("Data tidak cukup setelah warm-up. Butuh minimal 220 baris.")

    # Fitur: hanya ema50 dan ema200 (Tabel 3.6 — 2 fitur numerik kontinu)
    X = df_train_pool[['ema50', 'ema200']].values
    y = df_train_pool['label'].values

    # Stratified split 80:20 (Sub-bab 3.5, Tabel 3.6)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Inisialisasi dan latih model (Sub-bab 4.5)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        oob_score=True
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    classes = model.classes_.tolist()

    cm = confusion_matrix(y_test, y_pred, labels=['bullish', 'bearish'])

    metrics = {
        'accuracy':            round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision_bullish':   round(precision_score(y_test, y_pred, pos_label='bullish',  zero_division=0) * 100, 2),
        'recall_bullish':      round(recall_score   (y_test, y_pred, pos_label='bullish',  zero_division=0) * 100, 2),
        'f1_bullish':          round(f1_score       (y_test, y_pred, pos_label='bullish',  zero_division=0) * 100, 2),
        'precision_bearish':   round(precision_score(y_test, y_pred, pos_label='bearish',  zero_division=0) * 100, 2),
        'recall_bearish':      round(recall_score   (y_test, y_pred, pos_label='bearish',  zero_division=0) * 100, 2),
        'f1_bearish':          round(f1_score       (y_test, y_pred, pos_label='bearish',  zero_division=0) * 100, 2),
        'train_size':          len(X_train),
        'test_size':           len(X_test),
        'total_labeled':       len(df_train_pool),
        'oob_score':           round(model.oob_score_ * 100, 2),
        'confusion_matrix':    cm.tolist(),
        'cm_labels':           ['bullish', 'bearish'],
    }

    return model, metrics, df_train_pool


# ─── Prediction (seluruh baris untuk tampilan) ────────────────────────────────

def predict_all(model, df):
    """
    Prediksi label dan probabilitas untuk seluruh baris (termasuk warm-up,
    untuk keperluan visualisasi tabel). Menggunakan predict() dan predict_proba()
    sesuai Sub-bab 4.5 dan Tabel 3.14.
    Fitur: ema50, ema200 (2 fitur — konsisten dengan train_model)
    """
    X_all = df[['ema50', 'ema200']].values
    probs  = model.predict_proba(X_all)
    preds  = model.predict(X_all)
    classes = model.classes_.tolist()
    bull_idx = classes.index('bullish') if 'bullish' in classes else 0
    bear_idx = 1 - bull_idx

    df = df.copy()
    df['predicted_label'] = preds
    df['prob_bullish']     = [round(p[bull_idx] * 100, 1) for p in probs]
    df['prob_bearish']     = [round(p[bear_idx] * 100, 1) for p in probs]
    df['is_correct']       = (df['predicted_label'] == df['label']).astype(int)
    return df


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'File tidak ditemukan.'}), 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Harap upload file CSV.'}), 400

    n_estimators = int(request.form.get('n_estimators', 100))

    try:
        content = file.read().decode('utf-8')

        # ── Auto-detect separator (comma atau semicolon) ──────────────────────
        first_line = content.split('\n')[0]
        sep = ';' if first_line.count(';') > first_line.count(',') else ','

        df_raw = pd.read_csv(io.StringIO(content), sep=sep)
        df_raw.columns = df_raw.columns.str.lower().str.strip()

        df, close_col, date_col = prepare_dataset(df_raw)
        model, metrics, df_labeled = train_model(df, n_estimators=n_estimators)
        df_result = predict_all(model, df_labeled)

        # ── Prediksi minggu berikutnya (future prediction) ────────────────────
        # Menggunakan baris terakhir dari df_result (data berlabel setelah warm-up)
        # sesuai dengan distribusi fitur yang digunakan saat pelatihan
        latest_row = df_result.iloc[-1]
        future_features = [[latest_row['ema50'], latest_row['ema200']]]
        future_pred  = model.predict(future_features)[0]
        future_prob  = model.predict_proba(future_features)[0]
        classes      = model.classes_.tolist()
        bull_idx     = classes.index('bullish')
        bear_idx     = classes.index('bearish')
        future_bull  = round(future_prob[bull_idx] * 100, 2)
        future_bear  = round(future_prob[bear_idx] * 100, 2)

        # Simpan model
        joblib.dump(model, 'models/rf_model.pkl')

        # ── Data grafik (seluruh baris df) ────────────────────────────────────
        df_chart = df.copy()
        if 'date' in df_chart.columns:
            dates = df_chart['date'].dt.strftime('%Y-%m-%d').tolist()
        else:
            dates = [f"Minggu {i+1}" for i in range(len(df_chart))]

        chart_data = {
            'dates':  dates,
            'close':  df_chart['close'].round(2).tolist(),
            'ema50':  df_chart['ema50'].round(2).tolist(),
            'ema200': df_chart['ema200'].round(2).tolist(),
        }

        # ── Data tabel (200 baris terakhir dari df_result) ────────────────────
        table_df = df_result.tail(200).copy()
        if 'date' in table_df.columns:
            table_dates = table_df['date'].dt.strftime('%Y-%m-%d').tolist()
        else:
            table_dates = [f"Minggu {i+1}" for i in range(len(table_df))]

        table_data = []
        for i, (_, row) in enumerate(table_df.iterrows()):
            table_data.append({
                'date':            table_dates[i],
                'close':           round(row['close'], 2),
                'ema50':           round(row['ema50'], 2),
                'ema200':          round(row['ema200'], 2),
                'label':           row.get('label', '-'),
                'predicted_label': row.get('predicted_label', '-'),
                'prob_bullish':    row.get('prob_bullish', 0),
                'prob_bearish':    row.get('prob_bearish', 0),
                'is_correct':      int(row.get('is_correct', 0)),
            })

        # ── Info baris terkini ─────────────────────────────────────────────────
        last      = df_result.iloc[-1]
        last_date = dates[-1] if dates else "N/A"
        latest = {
            'date':         last_date,
            'close':        round(last['close'], 2),
            'ema50':        round(last['ema50'], 2),
            'ema200':       round(last['ema200'], 2),
            'trend':        str(last.get('predicted_label', '-')),
            'prob_bullish': float(last.get('prob_bullish', 0)),
            'prob_bearish': float(last.get('prob_bearish', 0)),
            'signal':       'Golden Cross' if last['ema50'] > last['ema200'] else 'Death Cross',
        }

        # ── Distribusi label ──────────────────────────────────────────────────
        label_counts = df_labeled['label'].value_counts().to_dict()
        dist = {
            'bullish': int(label_counts.get('bullish', 0)),
            'bearish': int(label_counts.get('bearish', 0)),
        }

        return jsonify({
            'success':           True,
            'metrics':           metrics,
            'chart_data':        chart_data,
            'table_data':        table_data,
            'latest':            latest,
            'distribution':      dist,
            'total_rows':        len(df_raw),
            'labeled_rows':      len(df_labeled),
            'future_prediction': {
                'trend':        future_pred,
                'prob_bullish': future_bull,
                'prob_bearish': future_bear,
            },
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export', methods=['POST'])
def export():
    data  = request.get_json()
    table = data.get('table_data', [])
    if not table:
        return jsonify({'error': 'Tidak ada data.'}), 400

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=table[0].keys())
    writer.writeheader()
    writer.writerows(table)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'bitcoin_dss_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
