import cv2
from ultralytics import YOLO

# 載入已訓練的模型
model_path = r"C:/Yolov8/ultralytics/segment/train1/weights/best.pt"
model = YOLO(model_path)

# 設定 WebCam IP 和 Port
ip_address = "10.22.54.143"
port = "8080"
camera_link = f"http://{ip_address}:{port}/video"

# 啟動 OpenCV 讀取影像串流
cap = cv2.VideoCapture(camera_link)

# 取得攝影機資訊
fps = 30  # 設定FPS，通常30即可
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 即時顯示影像
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 進行物件偵測，只保留信心度 >= 0.9
    results = model(frame, conf=0.9)

    # 確保結果有效
    if results and hasattr(results[0], "plot"):
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame  # 若無結果，保持原始影像

    # 顯示影像 (即時顯示)
    cv2.imshow("YOLOv8 Detection - Real-Time", annotated_frame)

    # 按下 'q' 退出即時影像辨識
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
