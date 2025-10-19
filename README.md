# Face Gesture Recognition

Program Python untuk mendeteksi wajah dan gesture tangan menggunakan kamera, dengan menampilkan gambar berbeda berdasarkan posisi jari dan ekspresi wajah.

## Fitur

- **Standby**: Menampilkan gambar `img/standby.jpg` saat wajah terdeteksi tanpa gesture khusus
- **Confuse**: Menampilkan gambar `img/confused.jpg` saat jari berada di area mulut
- **Know**: Menampilkan gambar `img/know.jpg` saat jari lurus (seperti gesture "I know!") di area sebelah kanan kepala
- **Shock**: Menampilkan gambar `img/shock.jpg` saat kedua tangan di area dada (5+ jari terdeteksi) DAN mulut terbuka

## Instalasi

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Siapkan gambar-gambar yang diperlukan di folder `img/`:
   - `img/standby.jpg` - Gambar untuk mode standby
   - `img/confused.jpg` - Gambar untuk gesture confuse (jari di mulut)
   - `img/know.jpg` - Gambar untuk gesture know (jari lurus di samping kepala kanan)
   - `img/shock.jpg` - Gambar untuk gesture shock (kedua tangan di dada + mulut terbuka)

## Cara Menggunakan

1. Jalankan program:
```bash
python face_gesture_recognition.py
```

2. Program akan membuka kamera dan menampilkan:
   - Feed kamera di sebelah kiri dengan visualisasi:
     - Kotak hijau di kanan kepala (area deteksi "Know")
     - Kotak biru di bawah kepala (area deteksi "Shock")
   - Gambar status di sebelah kanan

3. Gesture yang dikenali:
   - **Standby**: Hanya wajah terlihat → Tampilkan gambar standby
   - **Confuse**: Tunjuk jari ke area mulut → Tampilkan gambar confuse
   - **Know**: Angkat jari telunjuk lurus di samping kanan kepala (seperti gesture "Aha! I know!") → Tampilkan gambar know
   - **Shock**: Letakkan kedua tangan di dada (minimal 5 jari terdeteksi) DAN buka mulut → Tampilkan gambar shock

4. Tekan 'q' untuk keluar

## Teknologi yang Digunakan

- **OpenCV**: Untuk capture kamera dan display video
- **MediaPipe Face Detection**: Untuk mendeteksi wajah
- **MediaPipe Face Mesh**: Untuk mendeteksi mulut terbuka
- **MediaPipe Hands**: Untuk mendeteksi tangan dan jari

## Catatan

- Program menggunakan MediaPipe untuk deteksi wajah, mulut, dan tangan
- Kamera akan di-mirror (flip horizontal) untuk pengalaman lebih natural
- Jika gambar tidak ditemukan, akan ditampilkan placeholder hitam dengan teks
- Kotak visualisasi (hijau dan biru) membantu melihat area deteksi gesture
- Gesture "Shock" dirancang untuk menangani tangan yang tumpang tindih dengan menghitung total jari yang terdeteksi

## Requirements

- Python 3.7+
- Webcam
- opencv-python
- mediapipe
- numpy
