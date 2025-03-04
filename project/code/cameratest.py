import cv2

# 設定 DroidCamX IP 和 Port
ip_address = "10.22.54.143"
port = "8080"
camera_link = f"http://{ip_address}:{port}/video"

# 啟動 OpenCV 讀取影像串流
cap = cv2.VideoCapture(camera_link)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 無法讀取影像，請檢查 DroidCamX 連線")
        break

    frame = cv2.resize(frame, (600, 400))
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理資源
cap.release()
cv2.destroyAllWindows()
