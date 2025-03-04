import cv2
import os

from ultralytics import YOLO

# 載入已訓練的模型
model_path = r"C:/Users/owner/Downloads/YOLOv8/ultralytics/segment/train1/weights/best.pt"
model = YOLO(model_path)

# 讀取影片
video_path = r"C:/Users/owner/Downloads/YOLOv8/ultralytics/project/videos/v1.mp4"
cap = cv2.VideoCapture(video_path)

# 取得影片資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 設定輸出資料夾
output_folder = r"C:/Users/owner/Downloads/YOLOv8/ultralytics/project/results_VD"
os.makedirs(output_folder, exist_ok=True)  # 確保資料夾存在

# 方式 2：使用數字編號避免覆蓋
base_filename = "output_video"
index = 1
output_path = os.path.join(output_folder, f"{base_filename}_{index}.mp4")

# 找到未使用的檔名
while os.path.exists(output_path):
    index += 1
    output_path = os.path.join(output_folder, f"{base_filename}_{index}.mp4")

# 設定輸出影片
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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

    # 顯示影像 (可選)
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # 儲存影片
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"影片已儲存至: {output_path}")
