import cv2
import serial
import time
import threading
from ultralytics import YOLO

# 初始化 UART 連線
serial_port = "COM7"
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate, timeout=1)
time.sleep(2)  # 等待 Arduino 初始化

# 載入 YOLOv8 訓練好的模型
model_path = r"C:/Yolov8/ultralytics/segment/train1/weights/best.pt"
model = YOLO(model_path)

# 設定 WebCam 影像串流來源
ip_address = "10.22.54.143"
port = "8080"
camera_link = f"http://{ip_address}:{port}/video"

cap = cv2.VideoCapture(camera_link)

# 定義顏色（BGR 格式）
colors = {
    "Intact Pill": (255, 0, 0),  # 藍色
    "Chipped Pill": (0, 0, 255)  # 紅色
}

last_sent = None  # 記錄上一次傳送的結果

# **讀取 Arduino 回傳的數據 (執行緒)**
def read_serial():
    while True:
        if ser.in_waiting > 0:
            received_data = ser.readline().decode().strip()
            print(f"[Arduino 回應] {received_data}")

# 啟動執行緒來讀取 Arduino 資料
serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

# 影像辨識 + UART 傳輸
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 進行 YOLOv8 物件偵測 (信心度 >= 0.9)
    results = model(frame, conf=0.9, verbose=False)

    detected_label = None  # 紀錄目前偵測到的結果

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = float(box.conf[0])  
            class_id = int(box.cls[0])  
            label = model.names[class_id]  
            
            color = colors.get(label, (255, 255, 255))  
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 偵測到藥丸時，立即傳送結果
            if label == "Chipped Pill":
                detected_label = "1"  # 瑕疵
            elif label == "Intact Pill":
                detected_label = "0"  # 良好

    # **如果有偵測結果，立即發送**
    if detected_label is not None and detected_label != last_sent:
        ser.write((detected_label + "\n").encode())  
        print(f"發送至 Arduino: {detected_label}")
        last_sent = detected_label  # 記錄最後發送的結果
    
    # 影像縮小後顯示
    frame_resized = cv2.resize(frame, (960, 540))
    cv2.imshow("YOLOv8 Detection - Real-Time", frame_resized)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
ser.close()
