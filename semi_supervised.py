import cv2
import os
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np

def generate_labels(custom_model_path, threshold, video_name, video_path):
    model = YOLO(custom_model_path)

    os.makedirs("_yolo_dataset_complete/images", exist_ok=True)
    os.makedirs("_yolo_dataset_complete/labels", exist_ok=True)

    img_folder = "_yolo_dataset_complete/images"
    label_folder = "_yolo_dataset_complete/labels"


    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open video")
        exit()

    # Color mapping
    color_map = {
        "Basketball": (0, 165, 255),  # Orange
        "Hoop": (0, 0, 255),         # Red
        "Player": (255, 0, 0)        # Blue
    }

    frame_count = 0
    while True:
        ret, img = cap.read()

        if not ret:
            cap.release()
            exit()

        h, w, _ = img.shape

        resized_img = cv2.resize(img, (640, 640))

        if frame_count % 50 == 1:

            results = model(resized_img)[0]

            x1, y1, x2, y2 = 0, 0, 0, 0

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2 = int(result[0]), int(result[1]), int(result[2]), int(result[3])
                conf = float(result[4])
                class_id = int(result[5])
                label = model.model.names[class_id]

                if conf < threshold:
                    continue

                # Draw bounding box and label
                color = color_map.get(label, (0, 255, 0))  # Default to green if label not in map
                x1_ = int(x1 * w / 640)
                x2_ = int(x2 * w / 640)
                y1_ = int(y1 * h / 640)
                y2_ = int(y2 * h / 640)
                cv2.rectangle(img, (x1_, y1_), (x2_, y2_), color, 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1_, y1_ - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            img_file = f"{video_name}_frame_{frame_count}.jpg"
            label_file = f"{video_name}_frame_{frame_count}.txt"

            display_img = cv2.resize(img, (960, 510))
            cv2.imshow(f'{img_file}', display_img)
            key = cv2.waitKey(0)

            if key == ord('q'):
                break
            elif key == ord(' '):
                img_save_path = os.path.join(img_folder, img_file)
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                cv2.imwrite(img_save_path, resized_img)
                print(f"NOTE: {img_file} saved as correct bounding box!")

                label_save_path = os.path.join(label_folder, label_file)
                os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                with open(label_save_path, "w") as file:
                    x1 = x1 / 640
                    y1 = y1 / 640
                    x2 = x2 / 640
                    y2 = y2 / 640
                    file.write(f"0 {(x1 + x2) / 2} {(y1 + y2) / 2} {x2 - x1} {y2 - y1}")
            else:
                print(f'NOTE: {img_file} rejected')

            cv2.destroyAllWindows()

        frame_count += 1

    cv2.destroyAllWindows()

custom_model_path = "yolo_dataset/checkpoints/medium-Model/weights/best.pt"
threshold = 0.10

video_name = "fever_vs_ice"
video_name_ = video_name + ".mp4"
video_path = os.path.join("_videos/", video_name_)

generate_labels(custom_model_path, threshold, video_name, video_path)
