import sys
import cv2
import threading
import serial
import time
from collections import Counter
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class PillDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("藥品瑕疵檢測系統")
        self.setGeometry(100, 100, 1280, 720)

        # 左側：影像顯示
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 360)  # 顯示為原解析度 1/2

        # 右側：數據區
        self.data_label = QLabel("目前尚未辨識")
        self.data_label.setStyleSheet("font-size: 20px;")

        # 佈局
        layout = QHBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.data_label)
        self.setLayout(layout)

        # 初始化模型與攝影機
        self.model = YOLO(r"C:/Yolov8/ultralytics/segment/train1/weights/best.pt")
        self.cap = cv2.VideoCapture("http://10.22.54.143:8080/video")

        # 初始化 Serial
        self.ser = serial.Serial("COM7", 9600, timeout=1)
        time.sleep(2)

        # 狀態變數
        self.cooldown_frames = 30
        self.cooldown_counter = 0
        self.frame_counter = 0
        self.pill_detected = False
        self.label_counts = []

        # 定時更新畫面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # 背景執行序讀取 Arduino
        self.serial_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.serial_thread.start()

    def read_serial(self):
        while True:
            if self.ser.in_waiting > 0:
                response = self.ser.readline().decode().strip()
                print(f"[Arduino 回應] {response}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        results = self.model(frame, conf=0.9, verbose=False)
        detected_label = None
        detected_something = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                color = (255, 0, 0) if label == "Intact Pill" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                detected_something = True
                detected_label = "1" if label == "Chipped Pill" else "0"

        if self.cooldown_counter == 0:
            if detected_something and not self.pill_detected:
                self.frame_counter = 0
                self.label_counts = []
                self.pill_detected = True

            if detected_label and self.frame_counter < 3:
                self.label_counts.append(detected_label)
                self.frame_counter += 1

            if self.frame_counter == 3 and self.label_counts:
                final_label = Counter(self.label_counts).most_common(1)[0][0]
                self.ser.write((final_label + "\n").encode())
                print(f"發送至 Arduino: {final_label}")
                self.data_label.setText(f"目前辨識結果：{'瑕疵' if final_label == '1' else '良好'}")
                self.cooldown_counter = self.cooldown_frames
                self.pill_detected = False

        if not detected_something:
            self.pill_detected = False

        # 縮小顯示用影像
        resized = cv2.resize(frame, (640, 360))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()
        self.ser.close()

# 執行程式
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PillDetectionApp()
    win.show()
    sys.exit(app.exec_())
