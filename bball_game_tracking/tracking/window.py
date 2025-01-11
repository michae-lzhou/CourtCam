from dataclasses import dataclass
from typing import Tuple

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
