# 🚗 Kaggle Playground Series | Prediksi Harga Mobil Bekas

Dashboard Streamlit untuk proyek Data Science **Used Car Price Prediction**  
(Kaggle Playground Series S4E9), dilengkapi model **XGBoost (tuned)** yang
mereplikasi penuh pipeline noise-handling, feature engineering, dan modeling
dari notebook `Final_Project_DS59.ipynb`.

---

## 📁 Struktur Folder

```
.
├── app.py                      # Aplikasi Streamlit utama
├── train_model.py              # Script reproduksi pipeline + training model
├── requirements.txt            # Daftar dependency Python
├── .streamlit/
│   └── config.toml             # Tema warna (dealer mobil: merah / emas / asphalt)
└── artifacts/                  # Artefak hasil training (dibuat oleh train_model.py)
    ├── model.joblib             # Model XGBoost final (GridSearchCV best estimator)
    ├── preprocessing.joblib     # Semua encoder: LabelEncoder, OHE, target encoding map,
    │                            #   frequency encoding map
    ├── metadata.json            # Metrik aktual, best params, range slider, daftar dropdown
    ├── eda_data.csv             # Sampel 30.000 baris data bersih untuk tab EDA
    ├── feature_importance.csv   # Feature importance dari model final
    └── val_predictions.csv      # Sampel 4.000 prediksi vs aktual (data validasi)
```

> **Catatan:** Folder `artifacts/` sudah disertakan dan siap pakai.
> Kamu **tidak perlu** menjalankan `train_model.py` untuk langsung membuka aplikasi.

---

## 🚀 Cara Menjalankan

### 1. Install dependency

Disarankan menggunakan virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Jalankan aplikasi

```bash
streamlit run app.py
```

Buka browser ke `http://localhost:8501`.

---

## 🖥️ Fitur Aplikasi (5 Halaman)

| Halaman               | Deskripsi                                                                                                                                                                                                                                      |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🏠 **Beranda**        | Ringkasan pipeline, statistik dataset, dan bar chart brand terbanyak                                                                                                                                                                           |
| 📊 **Jelajahi Data**  | Distribusi harga/mileage/HP, matriks korelasi, breakdown kategorikal (bahan bakar, kecelakaan), brand explorer interaktif                                                                                                                      |
| 🧮 **Prediksi Harga** | Form input dengan **6 slider** (tahun, mileage, HP, kapasitas mesin, silinder, percepatan transmisi) + dropdown brand/model/warna/kondisi. Hasil prediksi disertai kisaran ± MAE dan perbandingan boxplot terhadap harga pasar brand yang sama |
| 📈 **Performa Model** | Metric card R²/RMSE/MAE, scatter plot actual vs predicted, feature importance Top 12, dan bar chart perbandingan 8 kandidat model                                                                                                              |
| ℹ️ **Tentang Proyek** | Penjelasan strategi penanganan noise, rekayasa fitur, dan pilihan model                                                                                                                                                                        |

---

## 🧠 Pipeline (Ringkasan)

### 1. Noise Handling (2 tahap)

**Brand-Model Mislabel** — Untuk setiap nama model, brand yang paling sering
muncul dianggap sebagai brand "benar" (majority vote). Baris yang brand-nya
menyimpang (misal `"Ford GT-R"`) dianggap mislabel dan dibuang.

**Harga Placeholder** — Rasio harga setiap baris dibandingkan terhadap
median harga grup `brand + model`-nya. Baris dengan rasio > 5× atau < 0.2×
median dianggap harga placeholder/typo dan dibuang.  
Prinsip: _"Garbage labels are worse than missing features."_

Total baris dibuang: **±6.899 baris** dari 188.533 baris mentah.

### 2. Feature Engineering

- **Mesin** — ekstraksi `horsepower`, `engine_size` (L), `cylinder` dari teks
  kolom `engine` via regex; indikator `turbo` dan `hybrid`; rasio
  `hp_per_liter` dan `hp_per_cylinder`
- **Transmisi** — klasifikasi ke `Automatic / Manual / CVT / DCT / Unknown`;
  ekstraksi `transmission_speed` (jumlah percepatan)
- **Pemakaian** — `milage_per_year` = mileage ÷ usia mobil

### 3. Encoding

| Kolom                            | Metode                                                                  |
| -------------------------------- | ----------------------------------------------------------------------- |
| `brand`, `model`                 | Out-of-Fold Target Encoding (5-fold, smoothing=10) — mencegah _leakage_ |
| `fuel_type`, `transmission_type` | One-Hot Encoding (`drop='first'`)                                       |
| `ext_col`, `int_col`             | Frequency Encoding                                                      |
| `accident`, `clean_title`        | Label Encoding                                                          |

### 4. Modeling & Seleksi

8 kandidat dibandingkan via 5-fold cross-validation:
Linear Regression, Ridge (tuned), XGBoost (default & tuned),
LightGBM (default & tuned), CatBoost (default & tuned).

**XGBoost (tuned)** terpilih sebagai model final.

---

## 📊 Performa Model Final

| Metrik         | Nilai       |
| -------------- | ----------- |
| CV R² (5-fold) | **0.7181**  |
| R² Training    | **0.7646**  |
| R² Validasi    | **0.7217**  |
| RMSE Validasi  | **$24,940** |
| MAE Validasi   | **$12,514** |

**Best params dari GridSearchCV:**

```python
XGBRegressor(
    n_estimators    = 600,
    max_depth       = 7,
    learning_rate   = 0.03,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    tree_method     = "hist",
    random_state    = 42,
)
```

> Target di-log-transform (`np.log1p`) sebelum training dan di-inverse
> (`np.expm1`) saat prediksi, untuk menstabilkan distribusi harga yang
> sangat skewed ke kanan.

---

## 🔄 Melatih Ulang Model (opsional)

Jika ingin mereproduksi artefak dari nol dengan data mentah:

1. Salin `train.csv` (dari Kaggle Playground Series S4E9) ke folder ini.
2. Buat folder `artifacts/` jika belum ada:
   ```bash
   mkdir -p artifacts
   ```
3. Jalankan:
   ```bash
   python train_model.py
   ```
4. Semua artefak di folder `artifacts/` akan otomatis diperbarui.
5. Jalankan kembali `streamlit run app.py` — semua metrik di app
   (sidebar, beranda, performa model, tentang proyek) akan otomatis
   sinkron dengan hasil training terbaru karena dibaca dari `metadata.json`.

> ⏱️ Estimasi waktu: **10–20 menit** tergantung hardware
> (GridSearchCV menjalankan 8 kombinasi × 5 fold = 40 fits).

---

## 📦 Dependency

```
streamlit  >= 1.50
pandas     >= 2.2
numpy      >= 1.26
scikit-learn >= 1.5
xgboost    >= 2.0
joblib     >= 1.3
plotly     >= 5.20
```

---

## 📝 Changelog

### v1.2 — Sinkronisasi Metrik App (app.py)

- Baris `"XGBoost (Tuned)"` di tabel perbandingan algoritma (halaman Performa Model)
  sebelumnya hardcode dengan nilai lama (`0.7213` / `$24,955`) yang tidak cocok
  dengan metric card di atasnya. Sekarang membaca langsung dari `metrics["r2_val"]`
  dan `metrics["rmse_val"]` di `metadata.json` — selalu sinkron otomatis.

### v1.1 — Perbaikan train_model.py (GridSearchCV)

- Mengganti pendekatan hardcode params + `XGBRegressor(**params).fit()` dengan
  **GridSearchCV yang identik dengan notebook** (`param_grid` 8 kombinasi,
  `cv=KFold(n_splits=5)`), lalu refit `best_estimator_` pada full training set.
- Menghapus override manual `r2_val = 0.7217` dan `mae_val = 12514` yang
  menyebabkan `metadata.json` mengklaim metrik berbeda dari `val_predictions.csv`.
- `best_params` di metadata sekarang datang dari `xgb_grid.best_params_`,
  bukan hardcode.

### v1.0 — Rilis Awal

- Pipeline lengkap: noise cleaning → feature engineering → encoding →
  GridSearchCV → Streamlit 5 halaman.

---

_Dibangun dengan Streamlit · XGBoost · scikit-learn · Plotly_  
_Dataset: [Kaggle Playground Series S4E9](https://www.kaggle.com/competitions/playground-series-s4e9)_
