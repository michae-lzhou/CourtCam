import cv2
import os
import numpy as np

def draw_basketball_labels(data_folder, validate):
    if validate:
        print("BEGINNING DATA VALIDATION")
    else:
        print("BEGINNING DATA INSPECTION")
    subsets = ['train', 'val']
    classes = ['Basketball', 'Hoop', 'Player']
    
    for subset in subsets:
        images_path = os.path.join(data_folder, subset, 'images')
        labels_path = os.path.join(data_folder, subset, 'labels')
        
        for label_file in os.listdir(labels_path):
            if label_file.endswith('.txt'):
                # Open the label file and read its contents
                label_path = os.path.join(labels_path, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Open the corresponding image
                img_file = label_file.replace('.txt', '.jpg')
                img_path = os.path.join(images_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Unable to open image {img_path}")
                    continue
                
                resized_img = cv2.resize(img, (960, 540))
                h, w, _ = resized_img.shape
                
                # Store all bounding boxes and their validity status
                boxes = []
                valid_boxes = []
                selected_box = -1
                
                # Parse all bounding boxes
                for line in lines:
                    elements = line.strip().split()
                    class_id = int(elements[0])
                    x_center, y_center, box_width, box_height = map(float, elements[1:])
                    
                    # Convert to pixel coordinates
                    x1 = int((x_center - box_width / 2) * w)
                    y1 = int((y_center - box_height / 2) * h)
                    x2 = int((x_center + box_width / 2) * w)
                    y2 = int((y_center + box_height / 2) * h)
                    
                    boxes.append({
                        'coords': (x1, y1, x2, y2),
                        'class_id': class_id,
                        'original_line': line
                    })
                    valid_boxes.append(True)  # Initially mark all boxes as valid

                def mouse_callback(event, x, y, flags, param):
                    nonlocal selected_box
                    if event == cv2.EVENT_LBUTTONDOWN:
                        # Check if click is inside any box
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box['coords']
                            if x1 <= x <= x2 and y1 <= y <= y2:
                                selected_box = i
                                valid_boxes[i] = not valid_boxes[i]  # Toggle validity
                                break

                # Create window and set mouse callback
                window_name = 'Image with Basketball Labels'
                cv2.namedWindow(window_name)
                cv2.setMouseCallback(window_name, mouse_callback)

                while True:
                    display_img = resized_img.copy()
                    
                    # Draw all boxes with their current status
                    for i, (box, is_valid) in enumerate(zip(boxes, valid_boxes)):
                        x1, y1, x2, y2 = box['coords']
                        class_id = box['class_id']
                        
                        # Green for valid boxes, red for invalid
                        color = (0, 255, 0) if is_valid else (0, 0, 255)
                        
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_img, f"{classes[class_id]} {'(Valid)' if is_valid else '(Invalid)'}", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Display instructions
                    cv2.putText(display_img, "Click boxes to toggle valid/invalid. Press ENTER to confirm, Q to skip, ESC to exit", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    cv2.imshow(window_name, display_img)
                    key = cv2.waitKey(1)

                    if key == ord(' '):
                        # display the original image
                        cv2.imshow(img_file, cv2.resize(img, (960, 540)))

                    if key == 13:  # Enter key
                        # Save only valid boxes to new label file
                        if validate:
                            valid_lines = [box['original_line'] for box, is_valid in zip(boxes, valid_boxes) if is_valid]
                            if valid_lines:
                                with open(label_path, 'w') as f:
                                    f.writelines(valid_lines)
                                print(f"Saved {len(valid_lines)} valid boxes for {img_file}")
                            else:
                                print(f"Removing {img_file} - no valid boxes")
                                os.remove(label_path)
                                if os.path.exists(img_path):
                                    os.remove(img_path)
                        cv2.destroyAllWindows()
                        break
                    elif key == ord('q'):  # Skip this image
                        cv2.destroyAllWindows()
                        break
                    elif key == 27:  # ESC to exit completely
                        cv2.destroyAllWindows()
                        return

        cv2.destroyAllWindows()

# Define the path to the dataset folder
dataset_path = "_annotations"
validate = False
draw_basketball_labels(dataset_path, validate)
