# 🚦 Traffic Congestion Prediction — Machine Learning Pipeline

Proyek ini merupakan bagian dari penelitian Tugas Akhir yang berfokus pada pembangunan pipeline *end-to-end Machine Learning* untuk memprediksi dan mengklasifikasikan tingkat kepadatan lalu lintas (*Traffic Situation*) ke dalam 4 kategori: **Low, Normal, High, dan Heavy**.

Repositori ini mencakup tahapan analisis eksploratif data (EDA), rekayasa fitur (*feature engineering*), penanganan *class imbalance* dengan *Random Undersampling*, 4 skenario eksperimen terkontrol (pemilihan nilai K-Fold, komparasi 5 algoritma, rasio data split, dan hyperparameter tuning via GridSearchCV), hingga ekspor model terbaik siap pakai (*production-ready*).

---

## Daftar Isi
- [Dataset](#-dataset)
- [Tahapan & Alur Kerja (Workflow Pipeline)](#-tahapan--alur-kerja-workflow-pipeline)
  - [1. Exploratory Data Analysis (EDA) & Data Cleaning](#1-exploratory-data-analysis-eda--data-cleaning)
  - [2. Feature Engineering & Preprocessing](#2-feature-engineering--preprocessing)
  - [3. Penanganan Imbalanced Data](#3-penanganan-imbalanced-data)
  - [4. Data Splitting](#4-data-splitting)
  - [5. Tahapan Eksperimen & Skenario Pengujian](#5-tahapan-eksperimen--skenario-pengujian)
- [Hasil Evaluasi Akhir Model Terbaik](#-hasil-evaluasi-akhir-model-terbaik)
- [Tech Stack & Pustaka](#-tech-stack--pustaka)
- [Struktur Direktori](#-struktur-direktori)
- [Panduan Menjalankan Notebook](#-panduan-menjalankan-notebook)

---

## 📊 Dataset

Dataset historis lalu lintas yang digunakan mencakup **2.976 baris data** dengan rincian atribut awal:
* `Time`: Waktu pencatatan (format 12-jam AM/PM, interval per 15 menit).
* `Date`: Tanggal pencatatan (1–31).
* `Day of the week`: Hari dalam seminggu (Monday–Sunday).
* `CarCount`: Jumlah mobil yang melintas.
* `BikeCount`: Jumlah sepeda motor / sepeda.
* `BusCount`: Jumlah bus (dieliminasi pada tahap pembersihan).
* `TruckCount`: Jumlah truk.
* `Total`: Total volume kendaraan.
* `Traffic Situation`: Tingkat kemacetan (Target Label: `low`, `normal`, `high`, `heavy`).

---

## Tahapan & Alur Kerja (Workflow Pipeline)

### 1. Exploratory Data Analysis (EDA) & Data Cleaning
* **Pengecekan Missing Value & Duplikasi:** Tidak ditemukan nilai *null* maupun baris data duplikat (data bersih).
* **Feature Dropping:** Kolom `BusCount` di-drop karena pertimbangan redundansi dan penyederhanaan fitur.
* **Deteksi & Pembersihan Outlier (Metode IQR):**
  * Fitur `CarCount`, `TruckCount`, dan `Total` tidak memiliki outlier ekstrem.
  * Fitur `BikeCount` memiliki **77 data outlier**.
  * Dilakukan pembersihan baris outlier pada `BikeCount` menggunakan batas IQR (Q_1 - 1.5 \times IQR s.d. Q_3 + 1.5 \times IQR), menyisakan **2.899 baris data valid**.
* **Distribusi Kelas Target Awal:** Ditemukan ketimpangan data (*extreme class imbalance*):
  * `normal`: 1.669 data
  * `heavy`: 605 data
  * `high`: 321 data
  * `low`: 304 data

### 2. Feature Engineering & Preprocessing
* **Ekstraksi Waktu (`Hour`):** Mengonversi string waktu `Time` menjadi representasi jam numerik (0–23).
* **Encoding Hari (`DayIndex`):** Pemetaan nama hari ke indeks numerik terurut (`Monday: 0` hingga `Sunday: 6`).
* **Encoding Target (`TrafficSituationEncoded`):** Menggunakan `LabelEncoder` untuk mengubah kelas string target menjadi integer (4 kelas).
* **Feature Selection:** 6 fitur final yang digunakan dalam pemodelan:
  1. `Hour`
  2. `DayIndex`
  3. `CarCount`
  4. `BikeCount`
  5. `TruckCount`
  6. `Total`
* **Standarisasi Fitur:** Menerapkan `StandardScaler` pada keenam fitur numerik agar terdistribusi dengan rata-rata 0 dan variansi 1. Objek scaler disimpan ke `scaler.joblib`.

### 3. Penanganan Imbalanced Data
* Menggunakan teknik **Random Undersampling** (`imblearn.under_sampling.RandomUnderSampler`) dengan `random_state=42`.
* Menyeimbangkan seluruh kelas mengikuti jumlah kelas minoritas (`low`: 304 sampel), menghasilkan distribusi yang seimbang:
  * **Class 0:** 304 data
  * **Class 1:** 304 data
  * **Class 2:** 304 data
  * **Class 3:** 304 data
  * **Total data setelah balancing:** 1.216 data.

### 4. Data Splitting
* Membagi dataset menggunakan `train_test_split` secara terstratifikasi (`stratify=y_resampled`).
* Pada skenario default (80:20), data latih berjumlah **972 sampel** (243 sampel per kelas) dan data uji **244 sampel** (61 sampel per kelas).
* Untuk model Deep Learning (LSTM), matriks data latih di-*reshape* ke format 3D tensor `(samples, timesteps, features)` yaitu `(972, 1, 6)`.

### 5. Tahapan Eksperimen & Skenario Pengujian

Penelitian ini merancang 4 skenario eksperimen sistematis di mana satu variabel diuji secara independen sementara variabel lainnya dikendalikan agar konstan berdasarkan temuan optimal sebelumnya:

#### 🧪 Eksperimen 1: Perbandingan Jumlah Fold pada Cross-Validation (K = 2 s.d. 10)
* **Tujuan:** Menemukan nilai partisi K yang paling stabil dan objektif dalam proses *K-Fold Cross-Validation* untuk menguji ketahanan model terhadap variasi partisi data.
* **Model yang Diuji:** Mewakili spektrum dari algoritma klasik hingga deep learning:
  1. **Random Forest & Decision Tree:** Model ensemble dan tree-based yang andal menangani data tabular multi-fitur.
  2. **K-Nearest Neighbors (KNN):** *Baseline* fundamental berbasis jarak instance.
  3. **Artificial Neural Network (ANN/MLP):** Menangkap pemetaan non-linear kompleks.
  4. **LSTM (Long Short-Term Memory):** Mengevaluasi kapabilitas deep learning berbasis sekuensial.
* **Temuan:** 
  * Pada nilai **K = 9**, terjadi **konvergensi performa** tinggi di mana titik akurasi model-model unggulan (Random Forest, Decision Tree, ANN) berada pada posisi yang sangat berdekatan atau mengelompok di sekitar rentang ~96%.
  * Partisi K = 9 membuktikan pengukuran paling stabil yang tidak menguntungkan atau merugikan model tertentu secara bias. Oleh karena itu, **K = 9 dipilih sebagai standar validasi untuk seluruh eksperimen berikutnya**.

---

#### 🧪 Eksperimen 2: Komparasi Mendalam 5 Algoritma (K = 9)
Menguji konsistensi dan stabilitas akurasi kelima model di setiap lipatan (*Fold 1* s.d. *Fold 9*):

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | **Rata-rata Akurasi** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9815** | 0.9259 | **0.9722** | **0.9630** | **0.9815** | **0.9722** | **0.9537** | 0.9722 | 0.9259 | **0.9609 (96.09%)** |
| **Decision Tree** | 0.9630 | **0.9537** | 0.9630 | 0.9444 | 0.9722 | 0.9630 | **0.9537** | **0.9815** | 0.9352 | **0.9588 (95.88%)** |
| **KNN** | 0.8796 | 0.8333 | 0.8426 | 0.8148 | 0.8796 | 0.8796 | 0.8148 | 0.9167 | 0.8981 | **0.8621 (86.21%)** |
| **LSTM** | 0.8056 | 0.7685 | 0.7593 | 0.8704 | 0.8241 | 0.8241 | 0.7315 | 0.7500 | 0.8426 | **0.7973 (79.73%)** |
| **ANN** | 0.8148 | 0.7315 | 0.7685 | 0.8611 | 0.7963 | 0.8056 | 0.7130 | 0.7222 | 0.8148 | **0.7809 (78.09%)** |

* **Alasan Pemilihan Random Forest:**
  1. **Akurasi Tertinggi:** Memperoleh rata-rata akurasi puncak sebesar **96.09%**, melampaui Decision Tree (95.88%), KNN (86.21%), LSTM (79.73%), dan ANN (78.09%).
  2. **Stabilitas antar-Fold:** Random Forest dan Decision Tree menunjukkan variansi akurasi yang sangat minim di tiap lipatan. Sebaliknya, model ANN dan LSTM mengalami fluktuasi tajam (misal: akurasi ANN merosot ke 71.30% di Fold 7), menandakan sensitivitas tinggi terhadap perubahan subset data latih.
  3. **Keunggulan Data Terstruktur:** Pendekatan *ensemble tree-based* terbukti lebih superior dan efisien dalam memetakan interaksi multi-atribut tabular lalu lintas dibandingkan arsitektur deep learning.

---

#### 🧪 Eksperimen 3: Pembagian Rasio Data Latih dan Uji
Menguji pengaruh proporsi data terhadap generalisasi model Random Forest menggunakan validasi silang 9-Fold:

| Rasio Data (Train : Test) | Rata-rata Akurasi (K = 9) |
| :---: | :---: |
| 60 : 40 | 0.9534 (95.34%) |
| 70 : 30 | 0.9542 (95.42%) |
| **80 : 20** | **0.9609 (96.09%)** |
| 90 : 10 | 0.9607 (96.07%) |

* **Temuan:** Akurasi meningkat secara signifikan dari rasio 60:40 ke 80:20, namun mulai mendatar (*plateau*) pada rasio 90:10 (0.9607). Rasio **80:20** dipilih karena memberikan keseimbangan optimal antara ukuran data uji yang representatif dan kemampuan generalisasi model tertinggi.

---

#### 🧪 Eksperimen 4: Hyperparameter Tuning via GridSearchCV
Optimasi hyperparameter dilakukan secara sistematis pada data latih (rasio 80:20) dengan evaluasi 9-Fold CV teracak:
* **Ruang Parameter:**
  * `n_estimators`: Rentang 10 s.d. 1500 pohon ensemble.
  * `max_features`: `'sqrt'`, `'log2'`, dan integer `4`.

| No | Konfigurasi Tuning | Rata-rata Akurasi (CV) |
| :-: | :--- | :---: |
| **1** | **`n_estimators: [91]`, `max_features: [4]`** | **0.9650 (96.50%)** |
| 2 | `n_estimators: [300, 450, 550, 600, 800, 900, 1000, 1300, 1500]`, `max_features: ['sqrt', 'log2']` | 0.9619 (96.19%) |
| 3 | `n_estimators: [100, 425]`, `max_features: ['sqrt', 'log2']` | 0.9609 (96.09%) |
| 4 | `n_estimators: [325, 340, 350, 380, 400, 500]`, `max_features: ['sqrt']` | 0.9599 (95.99%) |
| 5 | `n_estimators: [10]`, `max_features: ['sqrt', 'log2']` | 0.9506 (95.06%) |

* **Temuan:** Konfigurasi optimal dicapai pada **`n_estimators = 91`** dan **`max_features = 4`** dengan akurasi validasi silang **96.50%**, mengungguli pohon dalam jumlah besar (1000+ pohon) yang tidak memberikan kenaikan performa berarti namun meningkatkan beban komputasi.

---

## 🏆 Hasil Evaluasi Akhir Model Terbaik

Model Random Forest terbaik (`n_estimators = 91`, `max_features = 4`) dilatih ulang menggunakan seluruh data training (80%) dan diuji pada data uji unseen (20% / 244 sampel):

* **Metrik Evaluasi Keseluruhan (Data Uji):**
  * **Akurasi:** **98.36% (0.9836)**
  * **Presisi Rata-rata:** **0.9841**
  * **Recall Rata-rata:** **0.9836**
  * **F1-Score Rata-rata:** **0.9835**

* **Classification Report Per Kelas:**
  | Kategori Kelas | Presisi | Recall | F1-Score | Jumlah Sampel Uji |
  | :--- | :---: | :---: | :---: | :---: |
  | **Heavy** | 1.00 | 1.00 | 1.00 | 61 |
  | **High** | 0.97 | 1.00 | 0.98 | 61 |
  | **Low** | 0.97 | 1.00 | 0.98 | 61 |
  | **Normal** | 1.00 | 0.93 | 0.97 | 61 |

* **Analisis Confusion Matrix:**
  * Prediksi benar mendominasi penuh pada diagonal utama:
    * Kelas **Heavy**: 61/61 benar (Recall 1.00)
    * Kelas **High**: 61/61 benar (Recall 1.00)
    * Kelas **Low**: 61/61 benar (Recall 1.00)
    * Kelas **Normal**: 57/61 benar (Recall 0.93), dengan 4 sampel misklasifikasi (2 ke High, 2 ke Low).
  * Nilai presisi kelas Normal adalah 1.00 (tidak ada kelas lain yang salah diklasifikasikan sebagai Normal).

* **Serialisasi Model:**
  * `best_rf_model.joblib`: Model Random Forest final hasil tuning siap inferensi.
  * `scaler.joblib`: Objek `StandardScaler` untuk normalisasi input baru.

---

## Tech Stack & Pustaka

* **Bahasa Pemrograman:** Python 3
* **Data Processing & Analysis:** `Pandas`, `NumPy`
* **Visualisasi Data:** `Matplotlib`, `Seaborn`
* **Machine Learning & Preprocessing:** `Scikit-Learn` (`StandardScaler`, `LabelEncoder`, `KFold`, `cross_val_score`, `GridSearchCV`)
* **Penanganan Imbalanced Data:** `imbalanced-learn` (`RandomUnderSampler`)
* **Deep Learning Framework:** `TensorFlow` / `Keras` (`Sequential`, `LSTM`, `Dense`)
* **Model Serialization:** `Joblib`

---

## 📁 Struktur Direktori

```text
├── model.ipynb             # Notebook eksperimen lengkap (EDA, Preprocessing, 4 Skenario Training, Evaluasi)
├── traffic.csv             # Dataset yang digunakan
├── best_rf_model.joblib    # Model Random Forest terbaik yang telah dilatih (n_estimators=91, max_features=4)
├── scaler.joblib           # Scaler pra-pemrosesan data fitur
└── README.md               # Dokumentasi teknis proyek
