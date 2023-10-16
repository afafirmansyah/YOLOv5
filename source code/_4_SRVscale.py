import cv2
import numpy as np
from pathlib import Path
import torch
import time
import serial

# Serial port configuration for COM2 (untuk membaca data dari sensor)
serial_port_com2 = 'COM2'  # Ganti dengan nama port serial yang sesuai untuk COM2
baud_rate_com2 = 9600  # Sesuaikan dengan kecepatan baud yang digunakan oleh sensor

try:
    # Buka koneksi ke port serial COM2
    ser_com2 = serial.Serial(serial_port_com2, baud_rate_com2)
except serial.SerialException as e:
    print(f"Error opening serial port COM2: {e}")
    exit(1)

# Serial port configuration for COM3 (untuk mengirim data servoPos[0] dan servoPos[1])
serial_port_com3 = 'COM3'  # Ganti dengan nama port serial yang sesuai untuk COM3
baud_rate_com3 = 115200  # Sesuaikan dengan kecepatan baud yang digunakan oleh perangkat eksternal di COM3

try:
    # Buka koneksi ke port serial COM3
    ser_com3 = serial.Serial(serial_port_com3, baud_rate_com3)
except serial.SerialException as e:
    print(f"Error opening serial port COM3: {e}")
    exit(1)

# Inisialisasi model YOLOv5
model_path = Path("C:/Users/afafi/Documents/PT. SMART/_Image Processing/YOLOv5")  # Sesuaikan dengan lokasi penyimpanan YOLOv5
model_weights = model_path / '_dsetface.pt'
model = torch.hub.load(model_path, 'custom', path=model_weights, source='local')
model.eval()

# Inisialisasi webcam dengan resolusi HD (1280x720)
cap = cv2.VideoCapture(01)  # Gunakan 0 untuk kamera default, atau ganti dengan alamat IP jika menggunakan IP camera
cap.set(3, 1280)  # Lebar frame
cap.set(4, 720)   # Tinggi frame

# Inisialisasi untuk menghitung FPS
prev_time = 0

# Initialize the initial data_serial variable
distance_data = "N/A"

# Inisialisasi posisi servo awal
servoX = 90
servoY = 90

# Faktor pengubah sudut servo (sesuaikan dengan pergerakan yang diinginkan)
servoX_scale = 10.0
servoY_scale = 10.0

while True:
    ret, frame = cap.read()
    
    # Deteksi objek dengan YOLOv5
    results = model(frame)
    
    # Mengambil hasil deteksi wajah
    faces = results.pred[0]  # Menggunakan prediksi dari indeks pertama
    
    if len(faces) > 0:
        # Mengambil koordinat tengah wajah pertama yang terdeteksi
        fx, fy = (faces[0, :2] + faces[0, 2:4]) // 2
        
        # Menggambar kotak persegi panjang di sekitar wajah yang terdeteksi
        x1, y1, x2, y2 = faces[0, :4].to(torch.int)  # Mengubah tipe data tensor menjadi int
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()  # Mengambil nilai int dari tensor
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Menghitung FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Menampilkan FPS pada pojok kanan atas layar
        cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Menggambar lensa sniper
        lens_radius = 80
        lens_color = (0, 0, 255)
        cv2.circle(frame, (int(fx), int(fy)), lens_radius, lens_color, 2)
        cv2.circle(frame, (int(fx), int(fy)), 3, lens_color, cv2.FILLED)
        cv2.line(frame, (int(fx), int(fy) - lens_radius), (int(fx), int(fy) - 2 * lens_radius), lens_color, 2)
        cv2.line(frame, (int(fx), int(fy) + lens_radius), (int(fx), int(fy) + 2 * lens_radius), lens_color, 2)
        cv2.line(frame, (int(fx) - lens_radius, int(fy)), (int(fx) - 2 * lens_radius, int(fy)), lens_color, 2)
        cv2.line(frame, (int(fx) + lens_radius, int(fy)), (int(fx) + 2 * lens_radius, int(fy)), lens_color, 2)

        # Menampilkan teks "TARGET LOCKED" di atas wajah terdeteksi
        text = "TARGET LOCKED"
        text_color = (0, 255, 0)
        cv2.putText(frame, text, (int(fx) - 100, int(fy) - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        # Menghitung perubahan posisi servo berdasarkan koordinat wajah
        ws, hs = 1280, 720  # Resolusi frame
        servoX = servoX + (fx - ws / 2) / (ws / 2) * servoX_scale
        servoY = servoY + (fy - hs / 2) / (hs / 2) * servoY_scale

        if servoX < 0:
            servoX = 0
        elif servoX > 180:
            servoX = 180
        if servoY < 0:
            servoY = 0
        elif servoY > 180:
            servoY = 180

        # Mengirim nilai servoX dan servoY ke COM3
        ser_com3.write(f'{int(servoX)}\n'.encode('utf-8'))
        #ser_com3.write(f'{int(servoY)}\n'.encode('utf-8'))

    else:
        # Menampilkan teks "NO TARGET" jika tidak ada wajah yang terdeteksi
        text = "NO TARGET"
        text_color = (0, 255, 0)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (1280 - text_size[0]) // 2  # Posisi horizontal tengah
        text_y = (720 - text_size[1]) // 2  # Posisi vertikal tengah
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        

    # Menerima data serial dari COM2
    if ser_com2.in_waiting > 0:
        distance_data = ser_com2.readline().decode("utf-8").strip()

    cv2.putText(frame, f'Servo X: {int(servoX)} deg', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(frame, f'Servo Y: {int(servoY)} deg', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(frame, f'Distance: {distance_data} cm', (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Object Detection", frame)
    
    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam dan koneksi serial
cap.release()
ser_com2.close()
ser_com3.close()
cv2.destroyAllWindows()
