import torch
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)

def process_video(input_video_path, custom_model_path, output_video_path, threshold):
    # Redirect YOLO's output to a file or suppress it
    # os.environ['YOLO_VERBOSE'] = 'False'  # Suppress YOLO's verbose output
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLO model
    custom_model = YOLO(custom_model_path)
    
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open the video file.")
        return
        
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate resize dimensions
    new_width = 640
    new_height = 640
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(f"Frame Rate: {fps}")
    
    # Configure progress bar to update less frequently
    progress_bar = tqdm(total=total_frames, 
                       desc="Processing Video",
                       unit="frame",
                       mininterval=0.01,  # Minimum time between updates in seconds
                       maxinterval=5.0,  # Maximum time between updates in seconds
                       smoothing=0.3)    # Smoothing factor for speed estimates
    
    # Color mapping
    color_map = {
        "Basketball": (0, 165, 255),
        "Hoop": (0, 0, 255),
        "Player": (255, 0, 0)
    }

    frame_count = 0
    # update_interval = 10  # Update progress bar every 10 frames
    update_interval = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Run inference with verbose=False to suppress per-frame output
        results = custom_model(resized_frame, verbose=False)[0]
        
        # Scale factors
        scale_x = frame_width / new_width
        scale_y = frame_height / new_height
        
        # Process detections
        for det in results.boxes.data.tolist():
            conf = float(det[4])
            if conf < threshold:
                continue
                
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            class_id = int(det[5])
            label = custom_model.model.names[class_id]
            
            # Scale coordinates
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            # Draw bounding box and label
            color = color_map.get(label, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
        frame_count += 1
        
        # Update progress bar less frequently
        if frame_count % update_interval == 0:
            progress_bar.update(update_interval)
    
    # Update any remaining frames
    remaining_frames = frame_count % update_interval
    if remaining_frames > 0:
        progress_bar.update(remaining_frames)
    
    # Cleanup
    cap.release()
    out.release()
    progress_bar.close()
    cv2.destroyAllWindows()
    print(f"\nVideo with detections saved to: {output_video_path}")

if __name__ == "__main__":
    input_video_path = "_videos/RHSR9844.mp4"
    custom_model_path = "dataset/checkpoints/bhp_medium_model/weights/best.pt"
    output_video_path = "output.mp4"
    threshold = 0.10
    process_video(input_video_path, custom_model_path, output_video_path,
            threshold)
