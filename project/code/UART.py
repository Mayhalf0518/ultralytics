import serial
import time
import random

# 設定 UART 連線參數
serial_port = "COM7"  # Windows 通常是 COMx，Linux/Mac 可能是 /dev/ttyUSB0 或 /dev/ttyACM0
baud_rate = 115200     # 必須與 Arduino `Serial.begin(115200);` 相同

# 開啟 UART 連線
ser = serial.Serial(serial_port, baud_rate, timeout=1)
time.sleep(2)  # 等待 Arduino 重置

try:
    while True:
        test_value = str(random.choice([0, 1]))  # 隨機選擇 0 或 1
        print(f"發送測試值: {test_value}")
        ser.write((test_value + "\n").encode())  # 發送訊號
        time.sleep(2)  # 每 2 秒發送一次
except KeyboardInterrupt:
    print("測試結束")
    ser.close()  # 關閉 UART
