import serial
import time

# 設定 UART 連線
serial_port = "COM7"  # 根據你的實際情況修改
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)
time.sleep(2)  # 等待 Arduino 初始化

# 手動發送 "0" 和 "1"
ser.write(b"0\n")  # 發送 "0"
time.sleep(1)  # 等待 1 秒
ser.write(b"1\n")  # 發送 "1"

print("數據已發送")
ser.close()
