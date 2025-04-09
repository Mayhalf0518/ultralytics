import cv2
import serial
import time
import threading
from collections import Counter
from ultralytics import YOLO

# 初始化 UART
serial_port = "COM7"
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate, timeout=1)
time.sleep(2)

# 載入模型
model_path = r"C:/Yolov8/ultralytics/segment/train1/weights/best.pt"
model = YOLO(model_path)

# 相機來源
ip_address = "10.22.54.143"
port = "8080"
camera_link = f"http://{ip_address}:{port}/video"
cap = cv2.VideoCapture(camera_link)

# 顏色設定
colors = {
    "Intact Pill": (255, 0, 0),  # 藍
    "Chipped Pill": (0, 0, 255)  # 紅
}

# 狀態變數
frame_counter = 0
pill_detected = False
label_counts = []

cooldown_frames = 30  # 傳送後冷卻幀數
cooldown_counter = 0  # 當前冷卻倒數

# 背景接收 Arduino 回應
def read_serial():
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode().strip()
            print(f"[Arduino 回應] {data}")

serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

# 主迴圈
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 減少冷卻幀數
    if cooldown_counter > 0:
        cooldown_counter -= 1

    # YOLO 偵測
    results = model(frame, conf=0.9, verbose=False)
    detected_label = None
    detected_something = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]

            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detected_something = True
            if label == "Chipped Pill":
                detected_label = "1"
            elif label == "Intact Pill":
                detected_label = "0"

    # 執行偵測與傳送的邏輯（若未在冷卻中）
    if cooldown_counter == 0:

        if detected_something and not pill_detected:
            frame_counter = 0
            label_counts = []
            pill_detected = True

        if detected_label is not None and frame_counter < 3:
            label_counts.append(detected_label)
            frame_counter += 1

        if frame_counter == 3 and label_counts:
            most_common_label = Counter(label_counts).most_common(1)[0][0]
            ser.write((most_common_label + "\n").encode())
            print(f"發送至 Arduino: {most_common_label}")
            pill_detected = False
            frame_counter = 0
            label_counts = []
            cooldown_counter = cooldown_frames  # 啟動冷卻

    # 若藥丸離開畫面，重設
    if not detected_something:
        pill_detected = False

    frame_resized = cv2.resize(frame, (960, 540))
    cv2.imshow("YOLOv8 Detection - Real-Time", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 收尾
cap.release()
cv2.destroyAllWindows()
ser.close()
