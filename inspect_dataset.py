import cv2
import os

def draw_basketball_labels(data_folder):
    subsets = ['train', 'val']
    
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
                img_file = label_file.replace('.txt', '.jpg')  # Modify extension as needed
                img_path = os.path.join(images_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (1920, 1080))
                
                if img is None:
                    print(f"Warning: Unable to open image {img_path}")
                    continue
                
                h, w, _ = img.shape

                for line in lines:
                    elements = line.strip().split()
                    class_id = int(elements[0])
                    
                    if class_id == 0:  # Basketball class
                        # Extract YOLO format bounding box
                        x_center, y_center, box_width, box_height = map(float, elements[1:])
                        
                        # Convert to pixel coordinates
                        x1 = int((x_center - box_width / 2) * w)
                        y1 = int((y_center - box_height / 2) * h)
                        x2 = int((x_center + box_width / 2) * w)
                        y2 = int((y_center + box_height / 2) * h)
                        
                        # Draw the bounding box and label
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, 'Basketball', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display the image
                cv2.imshow('Image with Basketball Labels', img)
                key = cv2.waitKey(0)  # Press any key to continue
                if key == ord('q'):  # Press ESC to exit
                    break
        
        cv2.destroyAllWindows()

# Define the path to the dataset folder
dataset_path = "dataset"
draw_basketball_labels(dataset_path)

