from dataclasses import dataclass
from typing import Tuple

@dataclass
class Detection:
    frame_idx: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    model_id: int
