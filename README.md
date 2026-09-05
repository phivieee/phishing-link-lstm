# Phishing URL Detector — Pure Character-Level LSTM

## Project Overview

Project ini membangun sistem **deteksi URL phishing berbasis deep learning** menggunakan **Pure Character-Level LSTM**. Model menerima URL dalam bentuk teks dan mempelajari pola urutan karakter secara langsung untuk mengklasifikasikan URL menjadi dua kelas:

- `0` = Legitimate
- `1` = Phishing

Pendekatan character-level digunakan agar model dapat mempelajari pola karakter, susunan domain, path, simbol, angka, dan struktur URL secara langsung tanpa memerlukan feature engineering manual.

---

## Data

Dataset telah melalui proses cleaning dan dipisahkan berdasarkan **hostname** sebelum tahap pemodelan. Pemisahan berdasarkan hostname dilakukan untuk mengurangi risiko data leakage, sehingga hostname yang sama tidak muncul pada train, validation, dan test set.

| Dataset | Total URL | Legitimate | Phishing |
|---|---:|---:|---:|
| Training | 255,282 | 179,006 | 76,276 |
| Validation | 32,097 | 22,635 | 9,462 |
| Internal Test | 32,021 | 22,368 | 9,653 |
| External PhishTank | 10,000 | 0 | 10,000 |

External test menggunakan data PhishTank yang seluruhnya berisi URL phishing dan digunakan untuk melihat kemampuan generalisasi model pada data eksternal.

---

## Tahapan Project

### 1. Data Validation dan Cleaning

Sebelum pemodelan dilakukan validasi tambahan pada setiap dataset, meliputi:

- memastikan kolom `url` dan `label` tersedia;
- menghapus data dengan URL atau label kosong;
- mengubah URL menjadi tipe data string;
- menghapus whitespace pada awal dan akhir URL;
- mengubah label menjadi numerik;
- mempertahankan hanya label `0` dan `1`;
- mengubah tipe label menjadi `int8`;
- melakukan reset index setelah filtering.

### 2. Analisis Data

Analisis dilakukan terhadap distribusi kelas dan panjang URL pada masing-masing dataset.

Pada training set:

- mean panjang URL: `49.56`;
- median panjang URL: `40`;
- percentile 99: `210`;
- maksimum panjang URL: `2,175`.

Berdasarkan distribusi panjang URL tersebut, panjang sequence dibatasi menjadi:

```text
MAX_LEN = 180
```

URL yang lebih pendek akan memperoleh padding, sedangkan URL yang lebih panjang akan dipotong.

### 3. Character-Level Vectorization

URL diproses pada level karakter menggunakan `TextVectorization`.

```text
split = character
MAX_LEN = 180
Vocabulary size = 194
```

Contoh:

```text
paypal-login.com
↓
p → a → y → p → a → l → - → l → o → g → i → n → . → c → o → m
↓
integer token sequence
```

Vocabulary hanya dipelajari dari training set agar informasi dari validation dan test set tidak masuk ke proses training.

### 4. Pembentukan Dataset

Data dikonversi menjadi `tf.data.Dataset`.

- Training set menggunakan `shuffle`.
- Batch size yang digunakan adalah `256`.
- Pipeline menggunakan `prefetch` untuk meningkatkan efisiensi proses training.

### 5. Training dan Threshold Selection

Model dilatih maksimal selama `15 epochs` menggunakan validation set untuk memantau performa model.

Callback yang digunakan:

- `ModelCheckpoint`;
- `EarlyStopping`;
- `ReduceLROnPlateau`;
- `TerminateOnNaN`.

Training berhenti pada epoch ke-11 dan model terbaik berasal dari **epoch ke-9** berdasarkan nilai `val_loss`.

Setelah training, threshold klasifikasi ditentukan menggunakan validation set.

- **Balanced threshold = 0.5104**
- **Strict threshold = 0.8673**

Balanced threshold digunakan untuk menyeimbangkan precision dan recall, sedangkan strict threshold digunakan untuk menekan false positive.

---

## Arsitektur Model

Arsitektur Pure Character-Level LSTM yang digunakan:

```text
Raw URL
   ↓
Character TextVectorization
   ↓
Embedding (24 dimensions)
   ↓
SpatialDropout1D (0.35)
   ↓
LSTM (32 units)
   ↓
Dense (16, ReLU)
   ↓
Dropout (0.50)
   ↓
Dense (1, Sigmoid)
   ↓
Phishing Probability
```

Konfigurasi utama model:

| Komponen | Konfigurasi |
|---|---|
| Maximum sequence length | 180 |
| Vocabulary size | 194 |
| Embedding dimension | 24 |
| LSTM units | 32 |
| Dense units | 16 |
| SpatialDropout1D | 0.35 |
| Final Dropout | 0.50 |
| Output activation | Sigmoid |
| Optimizer | AdamW |
| Initial learning rate | 0.0003 |
| Batch size | 256 |

Total parameter model:

```text
12,497 trainable parameters
```

---

## Hasil

### Validation pada Model Terbaik

| Metric | Score |
|---|---:|
| Accuracy | 85.49% |
| Precision | 74.04% |
| Recall | 78.18% |
| ROC-AUC | 91.17% |
| PR-AUC | 85.10% |
| Loss | 0.3799 |

### Internal Test — Balanced Threshold

| Metric | Score |
|---|---:|
| Accuracy | **85.01%** |
| Precision | **73.94%** |
| Recall | **77.64%** |
| F1-score | **75.75%** |
| ROC-AUC | **90.46%** |
| PR-AUC | **84.00%** |

### Internal Test — Strict Threshold

| Metric | Score |
|---|---:|
| Accuracy | **81.89%** |
| Precision | **93.86%** |
| Recall | **42.73%** |
| F1-score | **58.73%** |
| ROC-AUC | **90.46%** |
| PR-AUC | **84.00%** |

Pada strict threshold diperoleh:

```text
False Positive = 270
False Negative = 5,528
```

### External Test — PhishTank

| Threshold | Phishing Recall |
|---|---:|
| Balanced | **65.95%** |
| Strict | **29.47%** |

Hasil menunjukkan bahwa balanced threshold memberikan keseimbangan yang lebih baik antara precision dan recall, sedangkan strict threshold menghasilkan precision yang lebih tinggi dengan konsekuensi recall yang lebih rendah. Pada external PhishTank, recall balanced sebesar **65.95%**, yang menunjukkan bahwa performa pada data eksternal masih lebih rendah dibanding internal test.
