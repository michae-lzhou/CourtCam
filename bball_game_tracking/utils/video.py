import cv2
import math
from tqdm import tqdm
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict, deque
from ..constants import *
from ..detection.detection import Detection
from ..tracking.basketball import BasketballTracker
from ..tracking.ellipse import EllipseTracker

# History of player positions: key = object_id, value = deque of positions
position_history = defaultdict(lambda: deque(maxlen=DET_FPS * 3))
stationary_threshold = int(DET_FPS * 3 / 5)
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

def calculate_frame_skip(fps: int, target_fps: int = DET_FPS) -> int:
    """Calculate frames to skip for target FPS"""
    if fps <= target_fps:
        return 0
    return max(0, round(fps / target_fps) - 1)

def is_within_ellipse(x, y, ellipse):
    """
    Check if a point (x, y) is within an ellipse defined by center_x, center_y, width, and height.
    """
    center_x, center_y, width, height = ellipse
    return ((x - center_x) ** 2) / (width / 2) ** 2 + ((y - center_y) ** 2) / (height / 2) ** 2 <= 1

def distance_to_ellipse_edge(x, y, ellipse):
    """
    Calculate the Euclidean distance from a point (x, y) to the center of the ellipse.
    """
    center_x, center_y, width, height = ellipse
    return math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

def find_game_ball(ellipse, detections):
    """
    Determine the game ball based on the given criteria.
    """
    center_x, center_y, width, height = ellipse
    ellipse_as_ball = (center_x, center_y, center_x, center_y, 0)
    if not detections:  # Case 3: No balls detected
        return ellipse_as_ball

    # Split detections into within-ellipse and outside-ellipse
    within_ellipse = [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2, conf) in detections if is_within_ellipse((x1 + x2) / 2, (y1 + y2) / 2, ellipse)]
    outside_ellipse = [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2, conf) in detections if not is_within_ellipse((x1 + x2) / 2, (y1 + y2) / 2, ellipse)]

    # Case 1: Balls within the ellipse
    if within_ellipse:
        highest_conf_overall = detections[0]
        highest_conf_within = max(within_ellipse, key=lambda d: d[4])

        if highest_conf_overall in within_ellipse:
            return highest_conf_overall  # 1a: Highest confidence ball is within ellipse
        else:
            # 1b: Compare confidence levels
            conf_highest_overall = highest_conf_overall[4]
            conf_highest_within = highest_conf_within[4]
            if conf_highest_overall >= max(0.8, 1.5 * conf_highest_within):
                return highest_conf_overall  # Highest confidence overall is the game ball
            else:
                return highest_conf_within  # Highest confidence within ellipse is the game ball

    # Case 2: No balls within the ellipse
    valid_balls = [d for d in detections if d[4] > 0.3]
    if valid_balls:
        highest_conf_overall = detections[0]
        if highest_conf_overall[4] > 0.7:
            return highest_conf_overall  # Highest confidence overall is the game ball
        else:
            # Find closest ball to the ellipse's edges
            closest_ball = min(detections, key=lambda d: distance_to_ellipse_edge((d[0] + d[2]) / 2, (d[1] + d[3]) / 2, ellipse))
            return closest_ball
    else:
        # Return the center of the ellipse
        return ellipse_as_ball


def create_output_video(video_path: str, all_detections: List[Detection], model_names: List[str], 
                       output_path: str, confidence_threshold: float = 0.25, force_30fps: bool = False):
    """
    Create output video with merged detections
    
    Parameters:
    force_30fps (bool): If True, output video will be 30 FPS (unless original is lower)
    """
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(video.get(cv2.CAP_PROP_FPS))
    rounded_fps = round(original_fps / 5) * 5
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = BasketballTracker(width, height)
    
    # Determine output FPS and frame sampling
    output_fps = min(30, rounded_fps) if force_30fps else rounded_fps
    frame_sampling = max(1, rounded_fps // output_fps) if force_30fps else 1
    
    frame_skip = calculate_frame_skip(rounded_fps)
    
    # Efficiently group detections by frame
    detections_by_frame = defaultdict(list)
    for det in all_detections:
        if det.confidence >= confidence_threshold:
            detections_by_frame[det.frame_idx].append(det)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    # Process frames
    scale_x = width / 640
    scale_y = height / 640
    
    with tqdm(total=total_frames, desc="Creating output video") as pbar:
        frame_idx = 0
        frame_count = 0
        object_tracker = {}
        ellipse_tracker = EllipseTracker(smoothing_factor=0.1)
        last_detections = []
        bballs = []
        ellipse_info = ()

        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_height, frame_width, _ = frame.shape
                
            # Skip frames based on frame_sampling when force_30fps is True
            if force_30fps and frame_count % frame_sampling != 0:
                frame_count += 1
                pbar.update(1)
                frame_idx += 1
                continue

            current_frame_player_positions = []

            # Use either current frame detections or last known detections
            current_detections = []
            if frame_idx in detections_by_frame:
                current_detections = detections_by_frame[frame_idx]
                last_detections = current_detections  # Update last known detections
                bballs = []
                ellipse_info = ()
            elif last_detections and frame_idx % (frame_skip + 1) != 0:
                # For skipped frames, use the last known detections
                current_detections = last_detections

            # Process detections for current frame
            sorted_detections = sorted(current_detections,
                    key=lambda det: det.confidence, reverse=True)
            for det in sorted_detections:
                label = det.class_name
                x1, y1, x2, y2 = det.bbox
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_position = (center_x, center_y)

                rounded_center_x = round(center_x / 5) * 5
                rounded_center_y = round(center_y / 5) * 5

                object_id = f"{label}_{rounded_center_x}_{rounded_center_y}"
                if object_id not in object_tracker:
                    object_tracker[object_id] = len(object_tracker)
                
                tracked_id = object_tracker[object_id]
                position_history[tracked_id].append(current_position)

                base_color = color_map.get(label, (0, 255, 0))
                if (label == "Player" or label == "Basketball") and is_stationary(position_history[tracked_id],
                        current_position, position_tolerance, stationary_threshold):
                    continue
                    color = (255, 255, 255)
                else:
                    color = base_color

                if label == "Player":
                    current_frame_player_positions.append(current_position)

                if label == "Basketball":
                    bballs.append((x1, y1, x2, y2, det.confidence))

                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # cv2.putText(frame, f"{label} ({model_names[det.model_id]}) {det.confidence:.2f}", 
                #           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Calculate and draw ellipse
            try:
                bounds = calculate_player_bounds(current_frame_player_positions)
                if bounds is not None:
                    ellipse_info = ellipse_tracker.update(*bounds)
                    center_x, center_y, width, height = ellipse_info
                    # cv2.ellipse(frame,
                    #           (center_x, center_y),
                    #           (width // 2, height // 2),
                    #           0, 0, 360,
                    #           (0, 255, 0), 2)
            except Exception as e:
                print(f"Error drawing ellipse: {e}")

            # Process game ball
            game_ball = ()
            if ellipse_info:
                game_ball = find_game_ball(ellipse_info, bballs)
            elif bballs:
                game_ball = bballs[0]

            if game_ball:
                x1, y1, x2, y2, conf = game_ball
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                offset = 10
                top_left = (center_x - offset, center_y - offset)
                bottom_right = (center_x + offset, center_y + offset)
                # cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)

            # Get ellipse center and ball position
            ellipse_center = (ellipse_info[0], ellipse_info[1]) if ellipse_info else None
            ball_pos = ((game_ball[0] + game_ball[2])//2, 
                       (game_ball[1] + game_ball[3])//2) if game_ball and game_ball[4] != 0 else None

            tracker.update(ellipse_info, ball_pos)

            cropped_frame = tracker.get_cropped_frame(frame)
            out.write(cropped_frame)
            frame_idx += 1
            frame_count += 1
            pbar.update(1)
    
    video.release()
    out.release()
