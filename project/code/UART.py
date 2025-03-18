import serial
import time
import random

# 設定 UART 連線參數
serial_port = "COM7"  # 請確認你的 Arduino 端口
baud_rate = 115200    

# 開啟 UART 連線
ser = serial.Serial(serial_port, baud_rate, timeout=1)
time.sleep(2)  # 等待 Arduino 重置

try:
    while True:
        test_value = str(random.choice([0, 1]))  # 隨機選擇 0 或 1
        print(f"發送測試值: {test_value}")
        ser.write((test_value + "\n").encode())  # 發送訊號

        # 讀取 Arduino 回傳的數據
        arduino_response = ser.readline().decode().strip()
        if arduino_response:
            print(f"Arduino 回應: {arduino_response}")

        time.sleep(2)  # 每 2 秒發送一次
except KeyboardInterrupt:
    print("測試結束")
    ser.close()  # 關閉 UART
