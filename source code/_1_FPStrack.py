import cv2
import numpy as np
from pathlib import Path
import torch
import time

# Inisialisasi model YOLOv5
model_path = Path("C:/Users/afafi/Documents/PT. SMART/_Image Processing/YOLOv5")  # Sesuaikan dengan lokasi penyimpanan YOLOv5
model_weights = model_path / '_dsetface.pt'
model = torch.hub.load(model_path, 'custom', path=model_weights, source='local')  
model.eval()

# Inisialisasi webcam dengan resolusi HD (1280x720)
cap = cv2.VideoCapture(0)  # Gunakan 0 untuk kamera default, atau ganti dengan alamat IP jika menggunakan IP camera
cap.set(3, 1280)  # Lebar frame
cap.set(4, 720)   # Tinggi frame

# Inisialisasi untuk menghitung FPS
prev_time = 0

while True:
    ret, frame = cap.read()
    
    # Deteksi objek dengan YOLOv5
    results = model(frame)
    
    # Menampilkan hasil deteksi
    results.print()
    
    # Menggambar kotak persegi panjang di sekitar objek yang terdeteksi
    img_with_boxes = results.render()[0]

    # Menghitung FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Menampilkan FPS pada pojok kiri atas layar
    cv2.putText(img_with_boxes, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Menampilkan frame dengan objek yang terdeteksi
    cv2.imshow('Object Detection', img_with_boxes)
    
    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
