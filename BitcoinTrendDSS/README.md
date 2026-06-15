# Sistem Pendukung Keputusan Analisis Tren Harga Bitcoin
## EMA + Random Forest

**Penulis:** Farhan Adhi Widiarsono  
**NIM:** 535220228  
**Program Studi:** Teknik Informatika — Universitas Tarumanagara  
**Pembimbing:** Tri Sutrisno, S.Si., M.Sc.

---

## Deskripsi

Aplikasi web berbasis Python/Flask yang mengimplementasikan Sistem Pendukung Keputusan (DSS)
untuk menganalisis tren harga Bitcoin menggunakan:
- **Exponential Moving Average (EMA-50 & EMA-200)**
- **Algoritma klasifikasi Random Forest** (scikit-learn)

---

## Fitur Utama

- Upload dataset CSV dari CoinMarketCap
- Perhitungan EMA-50 dan EMA-200 secara iteratif
- Pelabelan tren prediktif (bullish/bearish) tanpa label leakage
- Pelatihan model Random Forest dengan stratified split 80:20
- Evaluasi model: Akurasi, Presisi, Recall, F1-Score, Confusion Matrix
- Visualisasi grafik EMA interaktif (Plotly.js)
- Tabel hasil klasifikasi per periode
- Export hasil ke CSV

---

## Instalasi

```bash
# 1. Buat virtual environment (opsional)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan aplikasi
python app.py
```

Buka browser: **http://localhost:5000**

---

## Cara Penggunaan

1. **Upload CSV** — Unduh data historis Bitcoin dari CoinMarketCap (weekly), lalu upload.
2. **Atur parameter** — Tentukan jumlah pohon (n_estimators, default: 100).
3. **Jalankan Analisis** — Klik tombol "Jalankan Analisis".
4. **Lihat hasil** — Tren terkini, grafik EMA, metrik model, confusion matrix, tabel prediksi.
5. **Export** — Unduh hasil dalam format CSV.

---

## Format Dataset

File CSV dari CoinMarketCap dengan kolom minimal:
- `timeOpen` (atau `Date`) — tanggal periode
- `close` — harga penutupan mingguan (USD)

---

## Teknologi

| Komponen | Teknologi |
|---------|-----------|
| Backend | Python 3.9+, Flask 2.x |
| ML | scikit-learn (Random Forest) |
| Data | pandas, NumPy |
| Frontend | HTML5, Bootstrap 5, Plotly.js |
| Model persistence | joblib |

---

## Struktur Proyek

```
bitcoin_dss/
├── app.py              # Backend Flask + ML pipeline
├── requirements.txt    # Dependensi Python
├── README.md           # Dokumentasi
├── templates/
│   └── index.html      # Antarmuka web
├── models/             # Model Random Forest tersimpan
└── uploads/            # File CSV upload sementara
```
