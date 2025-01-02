import cv2
import os
from ultralytics import YOLO

def generate_labels():
    model = YOLO("best.pt")

    os.makedirs("_transferred/images", exist_ok=True)
    os.makedirs("_transferred/labels", exist_ok=True)

    img_folder = "_semi_supervised/images"
    label_folder = "_semi_supervised/labels"

    video_name = "basketball_game"
    video_name_ = video_name + ".mp4"
    video_path = os.path.join("_videos/", video_name_)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Couldn't open video")
        exit()

    frame_count = 0
    while True:
        ret, img = cap.read()

        if not ret:
            cap.release()
            exit()

        img = cv2.resize(img, (640, 640))

        if frame_count % 50 == 1:

            results = model.predict(img)

            x1, y1, y2, y2 = 0, 0, 0, 0

            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf
                class_ids = result.boxes.cls

                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(img, 'Basketball', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            img_file = f"{video_name}_frame_{frame_count}.jpg"
            label_file = f"{video_name}_frame_{frame_count}.txt"

            cv2.imshow(f'{img_file}', cv2.resize(img, (1920, 1080)))
            key = cv2.waitKey(0)

            if key == ord('q'):
                break
            elif key == ord(' '):
                img_save_path = os.path.join(img_folder, img_file)
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                cv2.imwrite(img_save_path, img)
                print(f"NOTE: {img_file} saved as correct bounding box!")

                label_save_path = os.path.join(label_folder, label_file)
                os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                with open(label_save_path, "w") as file:
                    x1 = x1 / 640
                    y1 = y1 / 640
                    x2 = x2 / 640
                    y2 = y2 / 640
                    # lines = [0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                    # file.write(" ".join(lines))
                    file.write(f"0 {(x1 + x2) / 2} {(y1 + y2) / 2} {x2 - x1} {y2 - y1}")
            else:
                print(f'NOTE: {img_file} rejected')

            cv2.destroyAllWindows()

        frame_count += 1

    cv2.destroyAllWindows()

generate_labels()

    # for img_file in sorted(os.listdir(img_folder)):
    #     # if img_file == "tufts_v_brandeis_frame_7461.jpg": begin = True
    #     if not begin: continue
    #     img_path = os.path.join(img_folder, img_file)
    #     img = cv2.imread(img_path)
    #     # img = cv2.resize(img, (1920, 1080))

    #     results = model.predict(img)
    #     
    #     label_file = img_file.replace('.jpg', '.txt')
    #     label_path = os.path.join(label_folder, label_file)
    #     # with open(label_path, 'r') as f:
    #     #     lines = f.readlines()

    #     h, w, _ = img.shape

    #     for line in lines:
    #         elements = line.strip().split()
    #         class_id = int(elements[0])

    #         if class_id == 0:
    #             x_c, y_c, b_w, b_h = map(float, elements[1:])
    #             x1 = int((x_c - b_w / 2) * w)
    #             y1 = int((y_c - b_h / 2) * h)
    #             x2 = int((x_c + b_w / 2) * w)
    #             y2 = int((y_c + b_h / 2) * h)

    #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
    #             cv2.putText(img, 'Basketball', (x1, y1 - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    #         
    #     cv2.imshow(f'{img_file}', img)
    #     key = cv2.waitKey(0)
    #     if key == ord('q'):
    #         break
    #     elif key == ord(' '):
    #         print(f'{img_file} successfully transferred as good training data!')
    #         img_save_path = os.path.join("_transferred/images", f"{img_file}")
    #         cv2.imwrite(img_save_path, img)
    #         label_save_path = os.path.join("_transferred/labels",
    #                                        f"{label_file}")
    #         with open(label_save_path, "w") as file:
    #             file.write("\n".join(lines))
    #     else:
    #         print(f'{img_file} rejected')

    #     cv2.destroyAllWindows()


    # cv2.destroyAllWindows()

