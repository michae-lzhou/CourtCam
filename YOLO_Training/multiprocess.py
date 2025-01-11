import cv2
import math
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import torch
import torch.cuda
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import tkinter as tk
from tkinter import messagebox, ttk
import sys
import multiprocessing
from sklearn.cluster import DBSCAN

DET_FPS = 15

@dataclass
class CropWindow:
    x: int
    y: int
    width: int
    height: int
    
    def get_corners(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return ((self.x, self.y), (self.x + self.width, self.y + self.height))
        
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

class BasketballTracker:
    def __init__(self, frame_width: int, frame_height: int):
        # Frame dimensions
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Base dimensions (half of target resolution)
        self.base_width = 1920 // 2
        self.base_height = 1080 // 2
        
        # Maximum scaling adjustments
        self.max_width_adjustment = 16
        self.max_height_adjustment = 9
        
        # History tracking
        self.ellipse_history = deque(maxlen=15)
        self.ball_history = deque(maxlen=15)
        self.area_history = deque(maxlen=15)

        # Out-of-frame tracking
        self.out_of_frame_history = deque(maxlen=30)  # Track last 30 frames
        self.aggressive_scale_history = deque(maxlen=45)  # Track aggressive scaling states
        self.OUT_OF_FRAME_THRESHOLD = 0.08  # 8% of frames need to be out of frame
        self.AGGRESSIVE_EXIT_THRESHOLD = 0.0  # Must have NO aggressive frames in recent history to exit
        self.aggressive_scaling = False
        self.SCALE_TRANSITION_FACTOR = 0.1  # Control transition speed between scaling modes
        
        # State tracking
        self.last_window = None
        self.current_velocity = np.array([0.0, 0.0])
        self.current_scale = 1.0
        self.target_scale = 1.0
        
        # Smoothing factors
        self.HORIZONTAL_SMOOTH_FACTOR = 0.2
        self.VERTICAL_SMOOTH_FACTOR = 0.005
        self.SCALE_SMOOTH_FACTOR = 0.1
        
        # Stickiness parameters
        self.BASE_INERTIA = 0.85
        self.area_threshold = None  # Will be set based on initial detections
        self.max_observed_area = 0
        self.min_observed_area = float('inf')

    def _is_ball_out_of_frame(self, ball_pos, current_window) -> bool:
        """Check if ball is outside the current window"""
        if not (ball_pos and current_window):
            return False
            
        (x1, y1), (x2, y2) = current_window.get_corners()
        ball_x, ball_y = ball_pos
        
        margin = 50  # Add a small margin to prevent flickering at the edges
        return (ball_x < x1 + margin or 
                ball_x > x2 - margin or 
                ball_y < y1 + margin or 
                ball_y > y2 - margin)

    def _update_out_of_frame_status(self, ball_pos, current_window):
        """Update the out-of-frame history and determine if aggressive scaling is needed"""
        # Update out-of-frame history
        is_out = self._is_ball_out_of_frame(ball_pos, current_window)
        self.out_of_frame_history.append(is_out)
        
        # Calculate if we need aggressive scaling based on recent out-of-frame events
        if len(self.out_of_frame_history) >= 10:
            out_of_frame_ratio = sum(self.out_of_frame_history) / len(self.out_of_frame_history)
            needs_aggressive = out_of_frame_ratio > self.OUT_OF_FRAME_THRESHOLD
            
            # Update aggressive scaling history
            self.aggressive_scale_history.append(needs_aggressive)
            
            # Check if we should enter aggressive scaling mode
            if needs_aggressive:
                self.aggressive_scaling = True
            else:
                # Only exit aggressive scaling if we've had NO aggressive frames recently
                recent_aggressive_ratio = sum(self.aggressive_scale_history) / len(self.aggressive_scale_history)
                if recent_aggressive_ratio <= self.AGGRESSIVE_EXIT_THRESHOLD:
                    self.aggressive_scaling = False

    def _detect_clusters(self, ellipse_positions):
        """Use DBSCAN for clustering with proper error handling"""
        if len(ellipse_positions) < 5:  # Not enough points for meaningful clustering
            return None
            
        # Use DBSCAN for clustering
        dbscan = DBSCAN(eps=30, min_samples=5)
        dbscan.fit(ellipse_positions)
        
        # Get cluster labels (-1 means noise)
        cluster_labels = dbscan.labels_
        
        # Find unique cluster labels, excluding noise (-1)
        unique_labels = set(cluster_labels) - {-1}
        
        if not unique_labels:  # No clusters found
            return None
            
        # Extract the center of each cluster
        cluster_centers = []
        for label in unique_labels:
            # Get points belonging to the current cluster
            cluster_points = ellipse_positions[cluster_labels == label]
            
            if len(cluster_points) > 0:  # Ensure cluster has points
                # Calculate the centroid of the cluster (mean of points)
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)
        
        return np.array(cluster_centers) if cluster_centers else None

    def _detect_erratic_ellipse(self) -> bool:
        """Detect if ellipses are erratic based on standard deviation or sudden jumps."""
        if len(self.ellipse_history) < 5:  # Ensure enough history
            return False
    
        # Calculate standard deviation of ellipse width and height
        center_x = [e[0] for e in self.ellipse_history]
        center_y = [e[1] for e in self.ellipse_history]
        widths = [e[2] for e in self.ellipse_history]
        heights = [e[3] for e in self.ellipse_history]
        center_x_std = np.std(center_x)
        center_y_std = np.std(center_y)
        width_std = np.std(widths)
        height_std = np.std(heights)
    
        # Define thresholds for erratic detection
        std_threshold = 15  # Adjust as needed
        return width_std > std_threshold or height_std > std_threshold or \
    center_x_std > std_threshold or center_y_std > std_threshold
        
    def _calculate_weighted_position(self, positions, weights=None):
        """Calculate weighted average position with optional weights"""
        if len(positions) == 0:  # Changed from if not positions
            return None
            
        if weights is None:
            # Default to exponential weights favoring recent positions
            weights = np.exp(np.linspace(-2, 0, len(positions)))
        weights = weights / weights.sum()
        
        avg_pos = np.average(positions, weights=weights, axis=0)
        return tuple(int(x) for x in avg_pos)
    
    def _get_average_positions(self):
        """Get smoothed positions for ellipse and ball"""
        # Get ellipse centers and areas
        if len(self.ellipse_history) > 0:  # Changed from if self.ellipse_history
            ellipse_positions = np.array([(x, y) for x, y, w, h in self.ellipse_history])
            ellipse_pos = self._calculate_weighted_position(ellipse_positions)
            
            # Update area tracking
            current_ellipse = self.ellipse_history[-1]
            area = np.pi * (current_ellipse[2]/2) * (current_ellipse[3]/2)
            self.area_history.append(area)
            
            # Update area bounds
            self.max_observed_area = max(self.max_observed_area, area)
            self.min_observed_area = min(self.min_observed_area, area)
            
            # Set area threshold if not yet set
            if self.area_threshold is None:
                self.area_threshold = area
        else:
            ellipse_pos = None
            
        # Get ball position
        if len(self.ball_history) > 0:  # Changed from if self.ball_history
            ball_positions = np.array(self.ball_history)
            ball_pos = self._calculate_weighted_position(ball_positions)
        else:
            ball_pos = None
            
        return ellipse_pos, ball_pos
    
    def _calculate_inertia_factor(self, area):
        """Calculate inertia factor based on ellipse area"""
        if not self.area_history:
            return self.BASE_INERTIA
            
        # Calculate how close current area is to max observed
        area_range = self.max_observed_area - self.min_observed_area
        if area_range == 0:
            return self.BASE_INERTIA
            
        normalized_area = (area - self.min_observed_area) / area_range
        # More inertia (stickiness) for larger areas
        return self.BASE_INERTIA + (0.1 * normalized_area)
    
    def _update_velocity(self, target_pos, current_pos, current_area):
        """Update velocity with area-based inertia"""
        if target_pos is None or current_pos is None:
            return np.array([0.0, 0.0])
            
        # Calculate desired movement
        movement = np.array(target_pos) - np.array(current_pos)
        
        # Split into components
        horizontal_movement = np.array([movement[0], 0])
        vertical_movement = np.array([0, movement[1]])
        
        # Calculate smooth factors
        inertia = self._calculate_inertia_factor(current_area)
        
        # Calculate target velocity
        target_velocity = (horizontal_movement * self.HORIZONTAL_SMOOTH_FACTOR +
                         vertical_movement * self.VERTICAL_SMOOTH_FACTOR)
        
        # Apply inertia
        self.current_velocity = (self.current_velocity * inertia +
                               target_velocity * (1 - inertia))
        
        return self.current_velocity
    
    def _calculate_window_dimensions(self, ellipse_area, ball_pos, current_window):
        """Calculate window dimensions with persistent scaling modes"""
        # Base scale on ellipse area
        base_area = np.pi * (self.base_width / 4) * (self.base_height / 4)
        area_scale = 1.0 + 0.1 * (ellipse_area - base_area) / base_area
        area_scale = np.clip(area_scale, 0.95, 1.05)
        
        # Initialize ball scaling
        ball_scale = 1.0
    
        if ball_pos and current_window:
            (x1, y1), (x2, y2) = current_window.get_corners()
            ball_x, ball_y = ball_pos
            
            # Calculate how far the ball is outside the window
            x_distance = max(0, ball_x - x2, x1 - ball_x)
            y_distance = max(0, ball_y - y2, y1 - ball_y)
    
            if x_distance > 0 or y_distance > 0:
                margin = 100
                current_size = max(current_window.width, current_window.height)
                required_size = current_size + max(x_distance, y_distance) * 2 + margin * 2
                ball_scale = min(required_size / current_size, 2.0)
        
        # Update out-of-frame tracking
        self._update_out_of_frame_status(ball_pos, current_window)
        
        # Determine target scale based on current mode
        if self.aggressive_scaling and ball_scale > 1.0:
            self.target_scale = ball_scale
        else:
            if self.aggressive_scaling:
                # Keep a slightly larger scale even when ball is in frame
                # to prevent rapid changes if ball moves out again
                self.target_scale = max(area_scale, 1.2)
            else:
                self.target_scale = area_scale
            
        # Smooth transition between scales
        if self.aggressive_scaling:
            # Faster transition when in aggressive mode
            self.current_scale = (self.current_scale * (1 - self.SCALE_TRANSITION_FACTOR) +
                                self.target_scale * self.SCALE_TRANSITION_FACTOR)
        else:
            # Slower, smoother transition when returning to normal mode
            transition_factor = self.SCALE_SMOOTH_FACTOR * 0.5  # Even smoother transition out
            self.current_scale = (self.current_scale * (1 - transition_factor) +
                                self.target_scale * transition_factor)
    
        return (int(self.base_width * self.current_scale),
                int(self.base_height * self.current_scale))
    
    def update(self, ellipse: Optional[Tuple], 
               ball_pos: Optional[Tuple]):
        """Update tracking history"""
        if ellipse:
            self.ellipse_history.append(ellipse)
        if ball_pos:
            self.ball_history.append(ball_pos)
    
    def get_crop_window(self) -> CropWindow:
        """Calculate crop window with smooth transitions"""
        # Handle no detection case
        if not self.ellipse_history:
            if self.last_window:
                # Smoothly zoom out from last position
                width = int(self.last_window.width * 1.1)
                height = int(self.last_window.height * 1.1)
                return CropWindow(
                    self.last_window.x,
                    self.last_window.y,
                    min(width, self.frame_width),
                    min(height, self.frame_height)
                )
            return CropWindow(0, 0, self.frame_width, self.frame_height)

        # Detect erratic behavior
        erratic_behavior = self._detect_erratic_ellipse()
        
        # Get averaged positions
        ellipse_pos, ball_pos = self._get_average_positions()
        if not ellipse_pos:
            return self.last_window or CropWindow(0, 0, self.frame_width, self.frame_height)
        
        # Get current area for calculations
        current_ellipse = self.ellipse_history[-1]
        current_area = np.pi * (current_ellipse[2]/2) * (current_ellipse[3]/2)

        # Adjust based on erratic behavior
        if erratic_behavior:
            # Focus on the largest detected ellipse for stability
            largest_ellipse = max(self.ellipse_history, key=lambda e: np.pi * (e[2] / 2) * (e[3] / 2))
            ellipse_pos = (largest_ellipse[0], largest_ellipse[1])
            current_area = np.pi * (largest_ellipse[2] / 2) * (largest_ellipse[3] / 2)
        else:
            # Perform cluster-based positional adjustment
            ellipse_positions = np.array([(x, y) for x, y, w, h in self.ellipse_history])
            cluster_centers = self._detect_clusters(ellipse_positions)

            if cluster_centers is not None and len(cluster_centers) > 0:
                # Take the weighted average of the cluster centers
                cluster_center = np.mean(cluster_centers, axis=0)
                ellipse_pos = tuple(int(x) for x in cluster_center)
        
        # Calculate window dimensions
        width, height = self._calculate_window_dimensions(
            current_area, 
            ball_pos,
            self.last_window
        )
        
        # Update position with velocity
        if self.last_window:
            velocity = self._update_velocity(
                ellipse_pos,
                self.last_window.center,
                current_area
            )
            center = np.array(self.last_window.center) + velocity
        else:
            center = np.array(ellipse_pos)
            
        # Calculate window position
        x = int(center[0] - width // 2)
        y = int(center[1] - height // 2)
        
        # Ensure window stays within frame
        x = np.clip(x, 0, self.frame_width - width)
        y = np.clip(y, 0, self.frame_height - height)
        
        # Create and store new window
        self.last_window = CropWindow(x, y, width, height)
        return self.last_window
    
    def get_cropped_frame(self, frame: np.ndarray) -> np.ndarray:
        """Get cropped frame based on current window"""
        window = self.get_crop_window()
        (x1, y1), (x2, y2) = window.get_corners()
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (1920, 1080))

@dataclass
class Detection:
    frame_idx: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    model_id: int

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

color_map = {
    "Basketball": (0, 165, 255),
    "Hoop": (0, 0, 255),
    "Player": (255, 0, 0)
}

# History of player positions: key = object_id, value = deque of positions
position_history = defaultdict(lambda: deque(maxlen=DET_FPS * 3))
stationary_threshold = int(DET_FPS * 3 / 5)
position_tolerance = 3000

class ConfigurationMenu:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ball Tracking Configuration")
        self.root.geometry("500x600")  # Width x Height
        self.root.resizable(False, False)
        
        # Set default values
        self.processing_device = tk.StringVar(value="GPU")
        self.quality = tk.StringVar(value="Original")
        self.start_program = False
        
        self._create_widgets()
        self._center_window()

    def on_closing(self):
        """Handle window close button (X) event"""
        self.config = None  # Or set to default configuration if needed
        try:
            self.root.quit()  # Stop the mainloop
            self.root.destroy()  # Destroy the window
        except:
            pass  # Suppress any destruction-related errors
        
    def _center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def _create_widgets(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Basketball Tracking Configuration",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Processing Device Selection
        device_label = ttk.Label(
            main_frame,
            text="Processing Device:",
            font=('Helvetica', 10)
        )
        device_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        device_combo = ttk.Combobox(
            main_frame,
            textvariable=self.processing_device,
            values=["GPU", "CPU"],
            state="readonly",
            width=30
        )
        device_combo.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        
        # Quality Selection
        quality_label = ttk.Label(
            main_frame,
            text="Processing Quality:",
            font=('Helvetica', 10)
        )
        quality_label.grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        quality_combo = ttk.Combobox(
            main_frame,
            textvariable=self.quality,
            values=["Original", "Fast"],
            state="readonly",
            width=30
        )
        quality_combo.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        
        # Information Text
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=5, column=0, columnspan=2, pady=(0, 20))
        
        info_text = """
Basketball Game Tracker:

This deep-learning program lets parents focus on their child’s performance by simulating dynamic camera panning from a wide-angle recording at half-court.
Disclaimer: Performance may vary due to limited training data.

Processing Device:
• GPU (~7x speed) - Recommended for faster processing (requires CUDA)
• CPU (~0.85x speed) - Slower but universally compatible

Quality Settings:
• Original - Full quality processing; runtime scales with original FPS
• Fast (30 FPS) - Faster processing with fixed video quality.

Note: If GPU processing fails, the program will automatically
fall back to CPU processing.
        """
        
        info_label = ttk.Label(
            info_frame,
            text=info_text,
            justify=tk.LEFT,
            wraplength=450
        )
        info_label.pack(fill=tk.X, padx=10)
        
        # Buttons Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=(20, 0))
        
        # Cancel Button
        cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
            width=15
        )
        cancel_btn.pack(side=tk.LEFT, padx=10)
        
        # Start Button
        start_btn = ttk.Button(
            button_frame,
            text="Start",
            command=self._on_start,
            width=15,
            style="Accent.TButton"
        )
        start_btn.pack(side=tk.LEFT, padx=10)
        
        # Create accent style for the Start button
        style = ttk.Style()
        style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'))
        
    def _on_start(self):
        self.start_program = True
        self.root.quit()
        
    def _on_cancel(self):
        self.start_program = False
        self.root.quit()
        
    def get_configuration(self):
        """Run the menu and return the configuration"""
        try:
            self.root.mainloop()
        except:
            # Handle any mainloop exceptions
            self.start_program = False
            return None
            
        if not self.start_program:
            # User clicked X or cancelled
            try:
                self.root.destroy()
            except:
                pass  # Suppress destroy errors
            sys.exit(0)
            
        # Get configuration if program should start
        try:
            config = {
                'processing_device': self.processing_device.get(),
                'quality': self.quality.get()
            }
            self.root.destroy()
            return config
        except Exception as e:
            print(f"Error getting configuration: {e}")
            try:
                self.root.destroy()
            except:
                pass
            sys.exit(1)

def show_configuration_menu():
    """Shows the configuration menu and returns the user's choices"""
    menu = ConfigurationMenu()
    config = menu.get_configuration()
    return config

def process_video_with_model_cpu(model_path: str, video_path: str, model_id: int,
                               num_cores: int) -> List[Detection]:
    """
    Process video with model using CPU cores
    """
    # Force model to use CPU
    model = YOLO(model_path)
    model.cpu()

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    rounded_fps = round(fps / 5) * 5
    frame_skip = calculate_frame_skip(rounded_fps)

    all_detections = []
    model_names = ["  Player  ", "Basketball"]

    with tqdm(total=total_frames, desc=f"Processing {model_names[model_id]}", position=model_id) as pbar:
        frame_idx = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_idx % (frame_skip + 1) != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            resized_frame = cv2.resize(frame, (640, 640))

            # Process frame on CPU
            results = model(resized_frame, verbose=False)

            for det in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = det
                class_name = model.model.names[int(class_id)]
                detection = Detection(
                    frame_idx=frame_idx,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=int(class_id),
                    class_name=class_name,
                    model_id=model_id
                )
                all_detections.append(detection)

            frame_idx += 1
            pbar.update(1)

    video.release()
    return all_detections

def setup_gpu_partition(gpu_id: int, max_memory: float):
    """
    Set up GPU memory partition
    max_memory: fraction of total GPU memory to use (e.g., 0.3 for 30%)
    """
    torch.cuda.set_device(gpu_id)
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    max_allocated = int(total_memory * max_memory)
    # Set maximum memory allocation for this process
    torch.cuda.set_per_process_memory_fraction(max_memory)
    return torch.cuda.Stream()

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
    """
    Calculate how many frames to skip to achieve target FPS
    Returns the number of frames to skip (e.g., 3 means process every 4th frame)
    """
    if fps <= target_fps:
        return 0
    return max(0, round(fps / target_fps) - 1)

def process_video_with_model(model_path: str, video_path: str, model_id: int,
                           max_memory: float, stream: torch.cuda.Stream) -> List[Detection]:
    """
    Process video with model using specified GPU memory partition and CUDA stream
    """
    with torch.cuda.stream(stream):
        model = YOLO(model_path)
        # Force model to use only allocated memory
        for param in model.parameters():
            param.pin_memory()

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        rounded_fps = round(fps / 5) * 5
        frame_skip = calculate_frame_skip(rounded_fps)

        all_detections = []

        model_names = ["  Player  ", "Basketball"]  # Assign custom model names here

        processed_frames = total_frames // (frame_skip + 1)

        with tqdm(total=total_frames, desc=f"Processing {model_names[model_id]}", position=model_id) as pbar:
            frame_idx = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                # Skip frames based on calculated frame_skip
                if frame_idx % (frame_skip + 1) != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # Process frame
                resized_frame = cv2.resize(frame, (640, 640))

                # Ensure processing happens in the assigned stream
                # with torch.cuda.stream(stream):
                results = model(resized_frame, verbose=False)

                # Convert detections to our format
                for det in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = det
                    class_name = model.model.names[int(class_id)]
                    detection = Detection(
                        frame_idx=frame_idx,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=class_name,
                        model_id=model_id
                    )
                    all_detections.append(detection)

                frame_idx += 1
                pbar.update(1)

                # Synchronize stream to prevent memory overflow
                # stream.synchronize()

        video.release()
        return all_detections

def setup_processing_device(model_paths: List[str], memory_fractions: List[float]):
    """
    Sets up either GPU streams or CPU cores based on user choice
    Returns: tuple (use_cuda: bool, streams_or_cores)
    """
    use_cuda = False  # Default to CPU

    if memory_fractions:  # If memory_fractions is provided, we're using GPU
        try:
            if not torch.cuda.is_available():
                messagebox.showerror(
                    "CUDA Error",
                    "CUDA is not available despite selecting GPU processing.\n" +
                    "Falling back to CPU processing."
                )
            else:
                use_cuda = True
                # Setup GPU streams
                streams = [setup_gpu_partition(0, mf) for mf in memory_fractions]
                return True, streams
        except Exception as e:
            messagebox.showerror(
                "CUDA Error",
                f"Error setting up CUDA: {str(e)}\nFalling back to CPU processing."
            )

    # Setup CPU cores if not using GPU
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    cores_per_model = max(1, num_cores // len(model_paths))
    return False, cores_per_model

def run_partitioned_detection(video_path: str, model_paths: List[str],
                            memory_fractions: List[float], output_path: str):
    """
    Run models with either GPU or CPU processing based on user choice
    """
    use_cuda, processing_resources = setup_processing_device(model_paths, memory_fractions)
    
    if use_cuda:
        # Original GPU processing code
        with ThreadPoolExecutor(max_workers=len(model_paths)) as executor:
            futures = [
                executor.submit(
                    process_video_with_model,
                    model_path,
                    video_path,
                    model_id,
                    memory_fractions[model_id],
                    processing_resources[model_id]
                )
                for model_id, model_path in enumerate(model_paths)
            ]
            
            all_detections = []
            for future in futures:
                all_detections.extend(future.result())
    else:
        # CPU processing code
        with ThreadPoolExecutor(max_workers=len(model_paths)) as executor:
            futures = [
                executor.submit(
                    process_video_with_model_cpu,
                    model_path,
                    video_path,
                    model_id,
                    processing_resources
                )
                for model_id, model_path in enumerate(model_paths)
            ]
            
            all_detections = []
            for future in futures:
                all_detections.extend(future.result())
    
    return all_detections

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

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT

def crop_video(frame, ellipse_info):
    # Extract the bounding box of the ellipse
    cx, cy, w, h = ellipse_info

    # Calculate dimensions of the crop box that match the target aspect ratio
    if w / h > TARGET_ASPECT_RATIO:
        # Width is too large, adjust it
        new_h = int(w / TARGET_ASPECT_RATIO)
        new_w = w
    else:
        # Height is too large, adjust it
        new_w = int(h * TARGET_ASPECT_RATIO)
        new_h = h

    # Compute the top-left corner of the crop box
    x1 = max(0, cx - new_w // 2)
    y1 = max(0, cy - new_h // 2)
    x2 = min(frame.shape[1], cx + new_w // 2)
    y2 = min(frame.shape[0], cy + new_h // 2)

    # Ensure the crop box matches the target aspect ratio
    cropped_frame = frame[y1:y2, x1:x2]

    # Resize to the target resolution
    resized_frame = cv2.resize(cropped_frame, (TARGET_WIDTH, TARGET_HEIGHT))
    return resized_frame

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

if __name__ == '__main__':
    # Show configuration menu
    config = show_configuration_menu()

    # Set up processing based on configuration
    use_cuda = config['processing_device'] == 'GPU'
    fast_mode = config['quality'] == 'Fast'

    model_paths = [
        'runs/hp_nano_model/weights/best.pt',  # Player model
        'runs/b_nano_model/weights/best.pt'    # Basketball model
    ]

    # Allocate 40% of GPU memory to each model if using GPU, otherwise None
    memory_fractions = [0.4, 0.4] if use_cuda else None

    video_file = '_videos/fullcourt.mp4'
    output_file = 'output.mp4'

    video = cv2.VideoCapture(video_file)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    rounded_fps = round(fps / 5) * 5
    print(f"Processing {video_file}")
    print(f"Video FPS: {rounded_fps}")
    video.release()

    print()
    print("\033[1mStep 1: Information Extraction\033[0m")
    # Run detection with partitioned GPU memory
    detections = run_partitioned_detection(video_file, model_paths, memory_fractions, output_file)

    # Create output video with custom model names
    model_names = ["Player_Model", "Basketball_Model"]
    print()
    print("\033[1mStep 2: Creating Output Video\033[0m")
    print(f"Output FPS: {30 if fast_mode else rounded_fps}")
    create_output_video(video_file, detections, model_names, output_file,
                       force_30fps=fast_mode)
