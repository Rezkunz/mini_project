# Smart Automatic License Plate Recognition (ALPR)

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-FF1493?style=for-the-badge&logo=yolo&logoColor=white" />
  <img src="https://img.shields.io/badge/PaddleOCR-007AFF?style=for-the-badge&logo=paddlepaddle&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
</div>

<br>

**Smart ALPR System** adalah sistem cerdas *end-to-end* untuk mendeteksi kendaraan, memotong (*crop*) area plat nomor secara akurat, dan membaca teks plat nomor kendaraan Indonesia secara dinamis di lingkungan *real-world*.

Proyek ini dibangun dengan memadukan kekuatan **Instance Segmentation** dari YOLOv8 dan keakuratan pembacaan teks **PaddleOCR**, dibungkus dalam antarmuka web modern berbasi Streamlit (menggunakan gaya *Glassmorphism*).

---

## Fitur Utama

- **Deteksi Presisi Tinggi:** Mampu mendeteksi berbagai jenis kendaraan (Mobil, Motor, Truk, Bus) sekaligus memotong plat nomornya.
- **Optical Character Recognition (OCR):** Menggunakan PaddleOCR yang telah di-tuning agresif (Binarization & Box Thresh rendah) agar tangguh membaca font plat nomor yang tipis atau miring.
- **Logika Format Indonesia:** Hasil bacaan mentah dari OCR tidak langsung ditelan mentah-mentah. Sistem memiliki fungsi Regex pintar yang akan mengoreksi salah baca (contoh: huruf 'O' terbaca angka '0') dan merapikannya ke format resmi (misal: `B 1234 ABC`).
- **UI/UX Premium:** Antarmuka bergaya *Glassmorphism* (efek tembus pandang/kaca) dengan tata letak hasil dalam bentuk Tabel, memberikan impresi profesional seperti aplikasi tingkat *Enterprise*.
- **Super Cepat:** Menggunakan varian `YOLOv8s-seg` yang jauh lebih ringan dibanding model konvensional, sangat cocok untuk inferensi dengan CPU maupun GPU.

---

## Teknologi yang Digunakan

*   **Computer Vision Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Detection & Segmentation).
*   **OCR Engine:** [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Jauh lebih unggul dibanding Tesseract untuk bacaan *in-the-wild*).
*   **Web Framework:** [Streamlit](https://streamlit.io/) (Untuk Frontend UI).
*   **Image Processing:** OpenCV (`cv2`) & NumPy (Penerapan *Unsharp Masking* pada plat sebelum di-OCR).

---

## Instalasi

Ikuti langkah-langkah di bawah ini untuk menjalankan *project* ini di komputer/laptop Anda secara lokal:

1. **Clone Repository**
   ```bash
   git clone https://github.com/Rezkunz/mini_project.git
   cd mini_project
   ```

2. **Buat Virtual Environment (Opsional tapi Direkomendasikan)**
   ```bash
   python -m venv venv
   # Untuk Windows:
   venv\Scripts\activate
   # Untuk Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Pastikan koneksi internet stabil karena PaddleOCR dan PyTorch membutuhkan ukuran download yang cukup besar).*

---

## Cara Pakai

Setelah semua dependensi terinstall, Anda hanya perlu menjalankan satu perintah sederhana:

```bash
python -m streamlit run app.py
```

**Panduan Penggunaan di Browser:**
1. Halaman web akan otomatis terbuka di `http://localhost:8501`.
2. Pada panel navigasi (Sidebar), pilih opsi unggah gambar.
3. Upload gambar kendaraan (bisa mobil, motor, dll) dari komputer Anda.
4. Klik tombol **Proses**.
5. AI akan melakukan *scanning* dan memproses deteksi dalam beberapa detik.
6. Hasil pembacaan plat (bersama dengan gambarnya) akan disajikan dalam bentuk Tabel.

---

## Hasil Analisis Model

Pelatihan model Plat Nomor dilakukan pada arsitektur `YOLOv8n` dengan iterasi pada dataset plat nomor Indonesia. Berikut adalah grafik metrik performa (*Training Results*):

### 1. Training Metrics (Loss & mAP)
Grafik di bawah ini menunjukkan penurunan tingkat kesalahan (*Loss*) secara konsisten selama iterasi *training*, serta nilai presisi rata-rata (mAP50) yang mendekati **0.9** ke atas. Ini membuktikan bahwa model tidak mengalami *overfitting*.

![Training Results](runs/detect/car_plate_detection/yolov8_plate_indo/results.png)

### 2. Confusion Matrix
Matrix kebingungan (*Confusion Matrix*) menunjukkan betapa akuratnya model dalam memprediksi kelas target (Plat Nomor) tanpa banyak melakukan *False Positive* pada latar belakang (*background*).

![Confusion Matrix](runs/detect/car_plate_detection/yolov8_plate_indo/confusion_matrix.png)

---

<p align="center">
  <i>Dibuat untuk menunjang pengembangan Intelligent Transportation System (ITS) di Indonesia.</i>
</p>
