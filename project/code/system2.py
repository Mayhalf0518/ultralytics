import sys
import cv2
import serial
import time
import threading
from collections import Counter
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# UART 初始化
serial_port = "COM7"
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate, timeout=1)
time.sleep(2)

# 相機來源
ip_address = "10.22.54.143"
port = "8080"
camera_link = f"http://{ip_address}:{port}/video"

# 載入模型
model_path = r"C:/Yolov8/ultralytics/segment/train1/weights/best.pt"
model = YOLO(model_path)

# 顏色設定
colors = {
    "Intact Pill": (255, 0, 0),
    "Chipped Pill": (0, 0, 255)
}

class PillDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智慧藥檢")
        self.setGeometry(100, 100, 1280, 720)

        main_layout = QHBoxLayout()

        # 左側影像顯示區
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(960, 720)
        main_layout.addWidget(self.camera_label)

        # 右側數據顯示（先顯示靜態文字）
        right_layout = QVBoxLayout()
        self.total_label = QLabel("總數: 0", self)
        self.good_label = QLabel("良品: 0", self)
        self.yield_label = QLabel("良率: 0.00%", self)
        for label in [self.total_label, self.good_label, self.yield_label]:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 24px; font-weight: bold; color: black;")
            right_layout.addWidget(label)

        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # 啟動攝影機執行緒
        self.camera_thread = CameraThread()
        self.camera_thread.image_signal.connect(self.update_camera)
        self.camera_thread.start()

        # 啟動串口接收執行緒
        serial_thread = threading.Thread(target=self.read_serial, daemon=True)
        serial_thread.start()

    def update_camera(self, qimage):
        self.camera_label.setPixmap(QPixmap.fromImage(qimage))

    def read_serial(self):
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode().strip()
                print(f"[Arduino 回應] {data}")

class CameraThread(QThread):
    image_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.cooldown_frames = 30
        self.cooldown_counter = 0
        self.frame_counter = 0
        self.pill_detected = False
        self.label_counts = []

    def run(self):
        cap = cv2.VideoCapture(camera_link)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 推論
            results = model(frame, conf=0.9, verbose=False)
            detected_label = None
            detected_something = False

            for result in results:
                for mask, box in zip(result.masks.data, result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    color = colors.get(label, (255, 255, 255))

                    # 畫多邊形輪廓
                    mask_np = mask.cpu().numpy()
                    contours, _ = cv2.findContours((mask_np * 255).astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, color, 2)

                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    detected_something = True
                    if label == "Chipped Pill":
                        detected_label = "1"
                    elif label == "Intact Pill":
                        detected_label = "0"

            # 冷卻邏輯
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

            if self.cooldown_counter == 0:
                if detected_something and not self.pill_detected:
                    self.frame_counter = 0
                    self.label_counts = []
                    self.pill_detected = True

                if detected_label is not None and self.frame_counter < 3:
                    self.label_counts.append(detected_label)
                    self.frame_counter += 1

                if self.frame_counter == 3 and self.label_counts:
                    most_common_label = Counter(self.label_counts).most_common(1)[0][0]
                    ser.write((most_common_label + "\n").encode())
                    print(f"發送至 Arduino: {most_common_label}")
                    self.pill_detected = False
                    self.frame_counter = 0
                    self.label_counts = []
                    self.cooldown_counter = self.cooldown_frames

            if not detected_something:
                self.pill_detected = False

            # ➤ 優化顯示流程（縮小處理後畫面）
            resized = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_LINEAR)
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_signal.emit(qimage)

        cap.release()

# 啟動應用程式
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PillDetectionApp()
    window.show()
    sys.exit(app.exec_())
