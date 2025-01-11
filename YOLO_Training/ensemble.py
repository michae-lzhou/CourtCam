import cv2
from ultralytics import YOLO
from tqdm import tqdm  # Import tqdm for the progress bar

color_map = {
    "Basketball": (0, 165, 255),
    "Hoop": (0, 0, 255),
    "Player": (255, 0, 0)
}

def run_tracker(filename, output_file=None):
    # Load your models into a list
    model_paths = ['runs/hp_nano_model/weights/best.pt', 'runs/b_nano_model/weights/best.pt']
    models = [YOLO(model_path) for model_path in model_paths]
    
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        print(f"Error: Unable to open video file {filename}")
        return

    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the VideoWriter if output file is provided
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 output
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Initialize the progress bar
    with tqdm(total=total_frames, desc=f"Processing {filename}", unit="frame") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break  # Exit if no more frames are available
            
            resized_frame = cv2.resize(frame, (640, 640))
            all_detections = []

            # Run each model on the frame and collect detections
            for model in models:
                results = model(resized_frame, verbose=False)
                detections = results[0].boxes.data.tolist()
                all_detections.extend(detections)

            # Process detections and draw them
            threshold = 0.25
            frame_height, frame_width, _ = frame.shape
            scale_x = frame_width / 640
            scale_y = frame_height / 640

            for det in all_detections:
                conf = float(det[4])
                if conf < threshold:
                    continue
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                class_id = int(det[5])
                label = models[0].model.names[class_id]
                
                # Scale coordinates
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                base_color = color_map.get(label, (0, 255, 0))
                color = base_color
                
                # Draw bounding boxes and labels
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Write the processed frame to the output video file
            if output_file:
                out.write(frame)
            
            # Update the progress bar
            pbar.update(1)
    
    # Release the video objects
    video.release()
    if output_file:
        out.release()
    cv2.destroyAllWindows()

# Input video files
img_file1 = '_videos/jack_brandeis.mp4'
img_file2 = '_videos/svbc_vs_ninja.mp4'

# Output video file
output_file1 = 'output.mp4'
output_file2 = 'output.mp4'

# Run the tracker for both video files and save the outputs
run_tracker(img_file1, output_file1)
# run_tracker(img_file2, output_file2)
