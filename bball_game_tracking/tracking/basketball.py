import cv2
from collections import deque
from typing import Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from ..tracking.window import CropWindow

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
