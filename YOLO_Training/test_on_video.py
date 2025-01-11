import torch
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import warnings
import os
import numpy as np
from collections import defaultdict, deque
warnings.filterwarnings("ignore", category=FutureWarning)

class EllipseTracker:
    def __init__(self, smoothing_factor=0.2):
        self.smoothing_factor = smoothing_factor
        self.current_center_x = None
        self.current_center_y = None
        self.current_width = None
        self.current_height = None

    def update(self, new_center_x, new_center_y, new_width, new_height):
        # Initialize if first update
        if self.current_center_x is None:
            self.current_center_x = new_center_x
            self.current_center_y = new_center_y
            self.current_width = new_width
            self.current_height = new_height
            return new_center_x, new_center_y, new_width, new_height

        # Apply exponential smoothing
        self.current_center_x = int(self.smoothing_factor * new_center_x + (1 - self.smoothing_factor) * self.current_center_x)
        self.current_center_y = int(self.smoothing_factor * new_center_y + (1 - self.smoothing_factor) * self.current_center_y)
        self.current_width = int(self.smoothing_factor * new_width + (1 - self.smoothing_factor) * self.current_width)
        self.current_height = int(self.smoothing_factor * new_height + (1 - self.smoothing_factor) * self.current_height)

        return (self.current_center_x, self.current_center_y,
                self.current_width, self.current_height)

# History of player positions: key = object_id, value = deque of positions
position_history = defaultdict(lambda: deque(maxlen=100))
# player_positions_history = deque(maxlen=50)  # Track all player positions for each frame
frame_threshold = 100
stationary_threshold = 20
position_tolerance = 3000

def is_stationary(history, current_position, tolerance, stationary_frames):
    """
    Check if a detection is stationary based on its position history.
    Returns True if the object hasn't moved significantly in the last stationary_frames frames.
    """
    if len(history) < stationary_frames:
        return False

    recent_positions = list(history)[-stationary_frames:]
    max_deviation = 0
    for pos in recent_positions:
        deviation = ((pos[0] - current_position[0])**2 +
                    (pos[1] - current_position[1])**2)**0.5
        max_deviation = max(max_deviation, deviation)

    return max_deviation <= tolerance

def calculate_player_bounds(frame_positions):
    """Optimized function to calculate player bounds for a single frame."""
    if not frame_positions or len(frame_positions) < 2:
        return None

    # Convert to numpy array for faster computation
    positions = np.array(frame_positions)
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]

    # Quick outlier removal using percentiles
    x_25, x_75 = np.percentile(x_coords, [25, 75])
    y_25, y_75 = np.percentile(y_coords, [25, 75])

    x_iqr = x_75 - x_25
    y_iqr = y_75 - y_25

    x_mask = (x_coords >= x_25 - 1.5 * x_iqr) & (x_coords <= x_75 + 1.5 * x_iqr)
    y_mask = (y_coords >= y_25 - 1.5 * y_iqr) & (y_coords <= y_75 + 1.5 * y_iqr)
    mask = x_mask & y_mask

    if not np.any(mask):
        return None

    filtered_x = x_coords[mask]
    filtered_y = y_coords[mask]

    padding = 200
    center_x = int(np.mean(filtered_x))
    center_y = int(np.mean(filtered_y))
    width = max(400, int(np.max(filtered_x) - np.min(filtered_x))) + padding
    height = max(300, int(np.max(filtered_y) - np.min(filtered_y)))

    return center_x, center_y, width, height

def process_video(input_video_path, custom_model_path, output_video_path, threshold):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    custom_model = YOLO(custom_model_path)

    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open the video file.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

    color_map = {
        "Basketball": (0, 165, 255),
        "Hoop": (0, 0, 255),
        "Player": (255, 0, 0)
    }

    frame_count = 0
    object_tracker = {}

    ellipse_tracker = EllipseTracker(smoothing_factor=0.1)  # Adjust smoothing_factor as needed
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 640))
        results = custom_model(resized_frame, verbose=False, conf=threshold, iou=0.45)[0]

        scale_x = frame_width / 640
        scale_y = frame_height / 640

        # Track player positions for this frame
        current_frame_player_positions = []

        # Sort detections by confidence
        # sorted_detections = sorted(results.boxes.data.tolist(), key=lambda det: det[4], reverse=True)
        sorted_detections = results.boxes.data.tolist()

        for det in sorted_detections:
            conf = float(det[4])
            if conf < threshold:
                continue

            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            class_id = int(det[5])
            label = custom_model.model.names[class_id]

            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_position = (center_x, center_y)

            # Track player positions
            if label == "Player":
                current_frame_player_positions.append(current_position)

            # Rounding the coordinates to the nearest 10 pixels (or adjust based on your needs)
            rounded_center_x = round(center_x / 5) * 5
            rounded_center_y = round(center_y / 5) * 5
            
            # Use the rounded coordinates to generate the object_id
            object_id = f"{label}_{rounded_center_x}_{rounded_center_y}"
            if object_id not in object_tracker:
                object_tracker[object_id] = len(object_tracker)
            
            tracked_id = object_tracker[object_id]
            position_history[tracked_id].append(current_position)

            base_color = color_map.get(label, (0, 255, 0))
            if label == "Player" and is_stationary(position_history[tracked_id], current_position, 
                           position_tolerance, stationary_threshold):
                continue
                color = (255, 255, 255)
            else:
                color = base_color

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update player positions history
        # player_positions_history.append(current_frame_player_positions)

        # Calculate and draw ellipse
        try:
            bounds = calculate_player_bounds(current_frame_player_positions)
            if bounds is not None:
                # Apply smoothing through the EllipseTracker
                center_x, center_y, width, height = ellipse_tracker.update(*bounds)
                cv2.ellipse(frame,
                          (center_x, center_y),
                          (width // 2, height // 2),
                          0, 0, 360,
                          (0, 255, 0), 2)
        except Exception as e:
            print(f"Error drawing ellipse: {e}")

        out.write(frame)
        frame_count += 1
        progress_bar.update(1)


    cap.release()
    out.release()
    progress_bar.close()
    cv2.destroyAllWindows()
    print(f"\nVideo with detections saved to: {output_video_path}")

if __name__ == "__main__":
    # input_video_path = "_videos/RHSR9844.mp4"
    input_video_path = "_videos/fullcourt.mp4"
    custom_model_path = "runs/hp_nano_model/weights/best.pt"
    output_video_path = "output.mp4"
    threshold = 0.25
    process_video(input_video_path, custom_model_path, output_video_path, threshold)
