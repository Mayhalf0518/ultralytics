import sys
import cv2
import serial
import time
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt


# === 1️⃣ 初始化 Arduino 串口 ===
serial_port = "COM7"
baud_rate = 4800
ser = serial.Serial(serial_port, baud_rate, timeout=1)
time.sleep(2)  # 等待 Arduino 初始化

# === 2️⃣ 設定攝影機串流來源 ===
ip_address = "10.22.54.143"
port = "8080"
camera_link = f"http://{ip_address}:{port}/video"

# === 3️⃣ 建立 PyQt 介面 ===
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("藥品缺陷檢測系統")
        self.setGeometry(100, 100, 1280, 720)

        # 設定主畫面布局
        main_layout = QHBoxLayout()

        # 左側影像顯示區域
        self.camera_label = QLabel(self)
        self.camera_label.setFixedSize(960, 720)
        main_layout.addWidget(self.camera_label)

        # 右側計數資訊
        right_layout = QVBoxLayout()
        self.total_label = QLabel("總數: 0", self)
        self.good_label = QLabel("良品: 0", self)
        self.yield_label = QLabel("良率: 0.00%", self)
        
        # **調整文字對齊**
        for label in [self.total_label, self.good_label, self.yield_label]:
            label.setAlignment(Qt.AlignCenter)  # **文字置中**
            label.setStyleSheet("font-size: 24px; font-weight: bold; color: black;")  # **調整大小、加粗**
            right_layout.addWidget(label)

        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

        # 啟動相機串流
        self.camera_thread = CameraThread()
        self.camera_thread.image_signal.connect(self.update_camera)
        self.camera_thread.start()

        # 啟動 Arduino 串口讀取執行緒
        self.total_count = 0
        self.good_count = 0
        serial_thread = threading.Thread(target=self.read_serial, daemon=True)
        serial_thread.start()

    def update_camera(self, qimage):
        """更新 GUI 上的攝影機畫面"""
        self.camera_label.setPixmap(QPixmap.fromImage(qimage))

    def read_serial(self):
        """從 Arduino 接收數據並更新計數"""
        while True:
            if ser.in_waiting > 0:
                received_data = ser.readline().decode().strip()
                print(f"{received_data}")

                if received_data.isdigit():
                    num = int(received_data)
                    self.total_count += 1
                    if num == 0:
                        self.good_count += 1

                    # 計算良率
                    rate = (self.good_count / self.total_count) * 100 if self.total_count > 0 else 0
                    
                    # 更新 GUI
                    self.total_label.setText(f"總數: {self.total_count}")
                    self.good_label.setText(f"良品: {self.good_count}")
                    self.yield_label.setText(f"良率: {rate:.2f}%")

# === 4️⃣ 影像處理執行緒 (QThread) ===
class CameraThread(QThread):
    image_signal = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(camera_link)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 轉換影像格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimage = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 傳送影像至 GUI
            self.image_signal.emit(qimage)

        cap.release()

# === 5️⃣ 啟動應用程式 ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
