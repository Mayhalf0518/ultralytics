import sys
import cv2
import serial
import time
import threading
from collections import Counter
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy, QPushButton
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
        self.setGeometry(100, 100, 960, 540)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)  # 外距
        main_layout.setSpacing(30)  # 左右區域的間距

        # 左側影像顯示區
        self.camera_label = QLabel(self)
        self.camera_label.setStyleSheet("background-color: #ccc;")  # 灰底方便排版時測試
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # ⭐自動撐滿左側空間
        self.camera_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.camera_label, stretch=3)  # ⭐給它更多伸展空間

        # 右側數據顯示（先顯示靜態文字）
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 靠左上對齊
        right_layout.setSpacing(20)

        self.pill_label = QLabel("藥物種類: 藥丸")
        self.total_label = QLabel("總數: 0", self)
        self.good_label = QLabel("良品: 0", self)
        self.yield_label = QLabel("良率: 0.00%", self)

        for label in [self.pill_label, self.total_label, self.good_label, self.yield_label]:
            label.setAlignment(Qt.AlignLeft)
            label.setStyleSheet("font-size: 24px; font-weight: bold; color: black;")
            right_layout.addWidget(label)

        # === 重設按鈕 ===
        self.reset_button = QPushButton("重設計數", self)
        self.reset_button.setStyleSheet("font-size: 18px; padding: 10px;")
        self.reset_button.clicked.connect(self.reset_counts)
        right_layout.addStretch(1)  # 將按鈕推到底部
        right_layout.addWidget(self.reset_button)

        main_layout.addLayout(right_layout, stretch=1)  # ⭐右側占比較少空間
        self.setLayout(main_layout)

        # 啟動攝影機執行緒
        self.camera_thread = CameraThread()
        self.camera_thread.image_signal.connect(self.update_camera)
        self.camera_thread.stats_signal.connect(self.update_stats)
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

    def update_stats(self, total, good):
        self.total_label.setText(f"總數: {total}")
        self.good_label.setText(f"良品: {good}")
        yield_rate = (good / total) * 100 if total > 0 else 0.0
        self.yield_label.setText(f"良率: {yield_rate:.2f}%")

    def reset_counts(self):
        self.camera_thread.total_count = 0
        self.camera_thread.good_count = 0
        self.update_stats(0, 0)
        print("[系統] 計數已重設")
# === 影像處理執行緒（含 YOLO 與 GUI 顯示） ===
class CameraThread(QThread):
    image_signal = pyqtSignal(QImage)

    stats_signal = pyqtSignal(int, int)  # 傳送總數、良品數

    def __init__(self):
        super().__init__()
        self.cooldown_frames = 40
        self.cooldown_counter = 0
        self.frame_counter = 0
        self.pill_detected = False
        self.label_counts = []
        self.total_count = 0
        self.good_count = 0

    def run(self):
        cap = cv2.VideoCapture(camera_link)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue


            frame = cv2.resize(frame, (640, 480))
            # YOLO 推論（含 Segmentation）
            results = model(frame, conf=0.7, verbose=False)
            result = results[0]

            # 自動繪製包含遮罩的圖像
            annotated_frame = frame.copy()
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = model.names[class_id]

                    color = colors.get(label, (255, 255, 255))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 轉成 QImage 給 PyQt 顯示
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            qimage = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.image_signal.emit(qimage)

            # 更新冷卻計數
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

            # 判斷是否有偵測到藥丸
            detected_label = None
            detected_something = False

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    detected_something = True
                    if label == "Chipped Pill":
                        detected_label = "1"
                    elif label == "Intact Pill":
                        detected_label = "0"

            # 執行判斷與發送邏輯
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

                    self.total_count += 1
                    if most_common_label == "0":
                        self.good_count += 1
                    
                    self.pill_detected = False
                    self.frame_counter = 0
                    self.label_counts = []
                    self.cooldown_counter = self.cooldown_frames

                    self.stats_signal.emit(self.total_count, self.good_count)

            if not detected_something:
                self.pill_detected = False

        cap.release()

# 啟動應用程式
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PillDetectionApp()
    window.show()
    sys.exit(app.exec_())
