from dataclasses import dataclass
from typing import Tuple

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
