import cv2
import serial
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# === 1️⃣ 初始化 UART 連線 ===
serial_port = "COM7"
baud_rate = 4800
ser = serial.Serial(serial_port, baud_rate, timeout=1)
time.sleep(2)  # 等待 Arduino 初始化

# === 2️⃣ 載入 YOLOv8 模型 ===
model_path = r"C:/Yolov8/ultralytics/segment/train1/weights/best.pt"
model = YOLO(model_path)

# === 3️⃣ 設定 WebCam 影像串流 ===
ip_address = "10.22.54.143"
port = "8080"
camera_link = f"http://{ip_address}:{port}/video"
cap = cv2.VideoCapture(camera_link)

# === 4️⃣ 設定 Tkinter GUI ===
root = tk.Tk()
root.title("藥品缺陷檢測系統")
root.geometry("1280x720")

# === 5️⃣ 建立 GUI 變數 (數字會即時更新) ===
total_count = tk.IntVar(value=0)  # 總數
good_count = tk.IntVar(value=0)   # 良品數
yield_rate = tk.StringVar(value="0.00%")  # 良率

# === 6️⃣ 建立 UI 元件 ===
frame_left = tk.Frame(root, width=960, height=720)
frame_left.pack(side="left", fill="both", expand=True)

frame_right = tk.Frame(root, width=320, height=720, bg="white")
frame_right.pack(side="right", fill="both")

# 影像顯示 Label
camera_label = tk.Label(frame_left)
camera_label.pack(fill="both", expand=True)

# 文字顯示區域 (總數、良品、良率)
total_label = tk.Label(frame_right, text="總數: 0", font=("Arial", 16), bg="white")
total_label.pack(pady=20)

good_label = tk.Label(frame_right, text="良品: 0", font=("Arial", 16), bg="white")
good_label.pack(pady=20)

yield_label = tk.Label(frame_right, text="良率: 0.00%", font=("Arial", 16), bg="white")
yield_label.pack(pady=20)

# === 7️⃣ 讀取 Arduino 數據 (執行緒) ===
def read_serial():
    global total_count, good_count, yield_rate
    while True:
        if ser.in_waiting > 0:
            received_data = ser.readline().decode().strip()
            print(f"{received_data}")  

            if received_data.isdigit():
                num = int(received_data)
                
                # 計數更新
                total_count.set(total_count.get() + 1)
                if num == 0:
                    good_count.set(good_count.get() + 1)
                
                # 計算良率
                total = total_count.get()
                good = good_count.get()
                rate = (good / total) * 100 if total > 0 else 0
                yield_rate.set(f"{rate:.2f}%")
                
                # 更新 GUI
                total_label.config(text=f"總數: {total}")
                good_label.config(text=f"良品: {good}")
                yield_label.config(text=f"良率: {rate:.2f}%")

# 啟動執行緒讀取 Arduino 資料
serial_thread = threading.Thread(target=read_serial, daemon=True)
serial_thread.start()

# === 8️⃣ 顯示相機影像 & 旋轉 90 度 ===
def update_camera():
    ret, frame = cap.read()
    if ret:
        # 旋轉影像 90 度
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # 轉換格式以顯示在 Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((960, 720))  # 確保大小符合
        imgtk = ImageTk.PhotoImage(image=img)
        
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)

    root.after(10, update_camera)  # 10ms 更新一次

# 啟動相機更新
update_camera()

# === 9️⃣ 啟動 GUI ===
root.mainloop()

# === 🔟 釋放資源 ===
cap.release()
ser.close()
cv2.destroyAllWindows()
