# Laporan Proyek: Automatic License Plate Recognition (ALPR)

> Sistem cerdas pendeteksi kendaraan dan pengenalan plat nomor menggunakan **YOLOv8** dan **PaddleOCR** yang dirancang untuk performa tinggi dan akurasi *real-world*.

---

## Development Phases (Siklus Pengembangan)

Proyek ini dibangun menggunakan metodologi MLDLC (*Machine Learning Development Life Cycle*) yang komprehensif, mencakup 7 fase utama yang telah berhasil diimplementasikan:

### 1. Dataset Collection
> **Status:** Selesai

- Dataset plat nomor Indonesia dikumpulkan secara komprehensif dan spesifik melalui platform **Roboflow**.
- Konfigurasi struktur dataset (lokasi folder, daftar kelas) diatur dan disimpan rapi dalam file `dataset/data_indo.yaml` dan `dataset/data.yaml`.

### 2. Image Preprocessing (Class Imbalance & Augmentation)
> **Status:** Selesai

- **Exploratory Data Analysis (EDA):** Analisis distribusi data dan validasi kualitas *bounding box* divisualisasikan melalui *script* `eda.py`.
- **Training Augmentation:** Teknik augmentasi canggih bawaan YOLO (*Mosaic*, *MixUp*, dan *HSV manipulation*) dimanfaatkan secara otomatis pada fase *training* untuk mencegah *overfitting* dan mengatasi ketidakseimbangan variasi data.
- **Inference Enhancement:** Di sisi aplikasi, diterapkan **Unsharp Masking** (kombinasi *Gaussian Blur* dan *Weighted Addition*) pada area plat yang terpotong untuk menajamkan kontur huruf sebelum dibaca oleh OCR.

### 3. Research & Selection
> **Status:** Selesai

Pemilihan arsitektur model AI dilakukan melalui riset perbandingan:
- **Model Kendaraan:** Menggunakan `YOLOv8s-seg` (*Instance Segmentation*) untuk memisahkan bodi kendaraan (Mobil, Motor, Bus, Truk) dari *background* secara piksel-sempurna.
- **Model Plat Nomor:** Memanfaatkan `YOLOv8n` versi kustom yang telah di-*fine-tune* khusus untuk plat nomor, memberikan rasio FPS dan Akurasi (mAP) terbaik (`best.pt`).
- **Sistem OCR:** Memilih `PaddleOCR` (mengalahkan Tesseract) karena jauh lebih tangguh menghadapi kemiringan, teks tipis, dan *noise* lingkungan liar (*in-the-wild*).

### 4. Training & Evaluation
> **Status:** Selesai

- **Pipeline:** Pelatihan model object detection dijalankan secara modular menggunakan `train.py` pada resolusi gambar 640px.
- **Metrik:** Hasil dari sesi pelatihan menunjukkan tingkat *Mean Average Precision* (mAP) yang sangat tinggi. Bobot terbaik diekstraksi dan dipakai untuk basis sistem produksi.

### 5. Testing
> **Status:** Selesai

- **Uji Coba Deteksi:** Diuji melalui `detect.py` untuk memastikan model dapat menemukan plat secara lokal dengan presisi.
- **Koreksi Teks Logis:** Diciptakan algoritma khusus dengan **Regex** (`format_indo_plate`) yang bertugas mengoreksi salah baca OCR (seperti huruf 'O' terbaca angka '0') dan merapikan hasil akhir (Spasi antara Kode Wilayah - Nomor - Akhiran).

### 6. Repo & Deploy
> **Status:** Selesai

- **Front-End:** Dibangun antarmuka aplikasi web modern dan dinamis menggunakan **Streamlit** via `app.py`.
- **Fitur Lengkap:** Mendukung dua mode *input* (Unggah Gambar dan Kamera *Realtime*) berbasis *OpenCV*.
- **Kesiapan Rilis:** Lingkungan dependensi terekam sempurna di `requirements.txt`. Proyek siap untuk didorong ke repositori GitHub dan di-*deploy* instan ke platform *cloud* AI seperti **Streamlit Community Cloud** atau **Hugging Face Spaces**.

### 7. Reporting
> **Status:** Selesai

- Laporan Markdown ini mendokumentasikan secara rinci perjalanan proyek teknis dari pengumpulan data gambar mentah hingga siap di-*deploy* ke produksi.

---

## Kesimpulan Akhir
Proyek deteksi plat nomor ini secara konsisten mengikuti standar industri *Software Engineering* dan *Machine Learning*. Seluruh siklus dari hulu ke hilir berhasil dieksekusi dengan baik, menghasilkan aplikasi akhir yang elegan, responsif, dan siap guna.
