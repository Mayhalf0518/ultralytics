import cv2
import time
import numpy as np

from ultralytics import YOLO
    
# 設定 YOLOv8 模型
model_path = r"C:/Users/owner/Downloads/YOLOv8/ultralytics/segment/train1/weights/best.pt"
model = YOLO(model_path)

# 設定 WebCam IP 和 Port
ip_address = "10.22.54.143"
port = "8080"
camera_link = f"http://{ip_address}:{port}/video"

# 啟動 OpenCV 讀取影像串流
cap = cv2.VideoCapture(camera_link)

# 設定分類顏色
class_colors = {
    "Intact Pill": (255, 0, 0),  # Blue
    "Chipped Pill": (0, 0, 255)  # Red
}

# 計算 FPS
prev_time = time.time()
frame_count = 0

# 開始即時影像辨識
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        continue  # 再次嘗試擷取影像
    
    cv2.imshow("YOLOv8 Segmentation Detection", frame)

    # 按下 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # YOLOv8 進行辨識
    results = model(frame)

    # 繪製結果
    for result in results:
        '''
        for box, conf, cls, mask in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls, result.masks.xy):
            x1, y1, x2, y2 = map(int, box)  # 邊界框
            class_name = model.names[int(cls)]
            color = class_colors.get(class_name, (0, 255, 0))  # 預設綠色
            label = f"{class_name} {conf:.2f}"

            mask_pts = np.array(mask, dtype=np.int32)
            cv2.fillPoly(frame, [mask_pts], color)

            #繪製邊界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 紀錄文字
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        '''

        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)  # 邊界框座標
            class_id = int(cls)
            label = f"{model.names[class_id]} {conf:.2f}"
            
            # 取得對應顏色，若無則預設為藍色
            color = class_colors.get(class_id, (255, 0, 0))

            # 繪製邊界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 標註文字
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame_count += 1
    if frame_count >= 10:
        curr_time = time.time()
        fps = frame_count / (curr_time - prev_time)
        prev_time = curr_time
        frame_count = 0 

# 結束程式，關閉視窗與瀏覽器
cap.release()
cv2.destroyAllWindows()