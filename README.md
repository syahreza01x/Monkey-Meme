# Face Gesture Recognition

Program Python untuk mendeteksi wajah dan gesture tangan menggunakan kamera, dengan menampilkan gambar berbeda berdasarkan posisi jari.

## Fitur

- **Standby**: Menampilkan gambar `img/standby.png` saat wajah terdeteksi tanpa gesture khusus
- **Confuse**: Menampilkan gambar `img/confuse.png` saat jari berada di area mulut
- **Know**: Menampilkan gambar `img/know.png` saat jari berada di area sebelah kanan kepala

## Instalasi

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Siapkan gambar-gambar yang diperlukan di folder `img/`:
   - `img/standby.png` - Gambar untuk mode standby
   - `img/confuse.png` - Gambar untuk gesture confuse (jari di mulut)
   - `img/know.png` - Gambar untuk gesture know (jari di samping kepala kanan)

## Cara Menggunakan

1. Jalankan program:
```bash
python face_gesture_recognition.py
```

2. Program akan membuka kamera dan menampilkan:
   - Feed kamera di sebelah kiri
   - Gambar status di sebelah kanan

3. Gesture yang dikenali:
   - **Normal**: Hanya wajah terlihat → Tampilkan gambar standby
   - **Confuse**: Tunjuk jari ke area mulut → Tampilkan gambar confuse
   - **Know**: Tunjuk jari ke area samping kanan kepala → Tampilkan gambar know

4. Tekan 'q' untuk keluar

## Catatan

- Program menggunakan MediaPipe untuk deteksi wajah dan tangan
- Kamera akan di-mirror (flip horizontal) untuk pengalaman lebih natural
- Jika gambar tidak ditemukan, akan ditampilkan placeholder hitam dengan teks

## Requirements

- Python 3.7+
- Webcam
- opencv-python
- mediapipe
- numpy
