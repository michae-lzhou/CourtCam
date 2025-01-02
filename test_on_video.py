import torch
from ultralytics import YOLO
import cv2
from tqdm import tqdm

input_video_path = "priory_vs_lghs.mp4"
custom_model_path = "best.pt"
output_video_path = "output.mp4"
threshold = 0.1

# custom_model = YOLO(custom_model_path)

custom_model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_model_path, force_reload=True)

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Cannot open the video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

frame_skip = 1
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    if frame_count % frame_skip == 0:
        # results = custom_model(frame)
        results = custom_model(cv2.resize(frame, (640, 640)))
        
        # Process detected objects
        for det in results.xyxy[0]:  # Access first batch's detections
        # for det in results.boxes.data.tolist():  # Access first batch's detections
            if float(det[4]) > threshold:  # Check confidence
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                conf = float(det[4])
                class_id = int(det[5])
                label = custom_model.names[class_id]
                
                # Set color based on class
                if label == "Basketball":
                    color = (0, 165, 255)  # Orange
                elif label == "Hoop":
                    color = (0, 0, 255)  # Red
                elif label == "Player":
                    color = (255, 0, 0)  # Blue
                
                # Draw bounding box and label
                x1 = int(x1 * 1920 / 640)
                x2 = int(x2 * 1920 / 640)
                y1 = int(y1 * 1080 / 640)
                y2 = int(y2 * 1080 / 640)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
        # cv2.imshow("YOLO Detection", frame)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
            
        progress_bar.update(1)

cap.release()
out.release()
progress_bar.close()
cv2.destroyAllWindows()
print(f"Video with rectangles saved to: {output_video_path}")
