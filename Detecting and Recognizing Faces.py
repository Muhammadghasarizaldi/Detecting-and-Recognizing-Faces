import cv2
import os

# File cascade yang digunakan
CASCADE_FILENAME = 'haarcascade_frontalface_default.xml'

# 1. Muat Cascade Classifier
if not os.path.exists(CASCADE_FILENAME):
    print(f"Error: File cascade tidak ditemukan: {CASCADE_FILENAME}")
    print("Pastikan file XML berada di direktori yang sama dengan skrip ini.")
    exit()

face_cascade = cv2.CascadeClassifier(CASCADE_FILENAME)

# 2. Inisialisasi Video Capture (Webcam default, indeks 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Gagal membuka webcam. Pastikan kamera berfungsi.")
    exit()

print("Deteksi Wajah Real-Time Aktif. Tekan tombol 'q' untuk keluar.")

while True:
    # Ambil frame demi frame
    ret, frame = cap.read()
    
    if not ret:
        print("Gagal mengambil frame.")
        break

    # 3. Konversi ke Grayscale untuk pemrosesan yang cepat
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. Deteksi Wajah (menggunakan parameter yang disempurnakan)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60) 
    )

    # 5. Gambarkan persegi panjang (bounding box) di sekitar wajah
    for (x, y, w, h) in faces:
        # Menggambar kotak berwarna biru (BGR: 255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        # Menambahkan label teks
        cv2.putText(frame, 'WAJAH', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 6. Tampilkan Frame
    cv2.imshow('OpenCV Face Detection (Webcam)', frame)

    # 7. Mekanisme Keluar (Tekan 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Pembersihan sumber daya
cap.release()
cv2.destroyAllWindows()