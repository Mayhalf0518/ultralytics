import cv2

from ultralytics import YOLO

# 載入已訓練的模型
model_path = r"C:/Yolov8/ultralytics/segment/train2/weights/best.pt"
model = YOLO(model_path)

# 設定 WebCam IP 和 Port
ip_address = "10.22.54.143"
port = "8080"
camera_link = f"http://{ip_address}:{port}/video"

# 啟動 OpenCV 讀取影像串流
cap = cv2.VideoCapture(camera_link)

# 定義顏色（BGR 格式）
colors = {
    #"Intact Pill": (255, 0, 0),  # 藍色
    #"Chipped Pill": (0, 0, 255)  # 紅色
    "Intact Capsule": (255, 0, 0),  # 藍色
}

# 取得攝影機資訊
fps = 30  # 設定FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 即時顯示影像
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 進行物件偵測（信心度 >= 0.9）
    results = model(frame, conf=0.9)

    # 解析 YOLOv8 的偵測結果
    for result in results:
        for box in result.boxes:  # 逐個讀取偵測到的物件
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 取得邊界框座標
            conf = float(box.conf[0])  # 取得信心度
            class_id = int(box.cls[0])  # 取得類別索引
            label = model.names[class_id]  # 取得類別名稱
            
            # 取得對應顏色（若類別不在字典中，則預設為白色）
            color = colors.get(label, (255, 255, 255))

            # 繪製邊界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # 顯示類別名稱與信心度
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 顯示影像
    cv2.imshow("YOLOv8 Detection - Real-Time", frame)

    # 按下 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()