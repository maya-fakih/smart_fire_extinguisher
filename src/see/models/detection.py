"""
Detection Module - Single YOLO Bounding Box.

Container for one single YOLO detection (one bounding box).
No logic here, just structure.
"""

from dataclasses import dataclass


# ── Detection ─────────────────────────────────────────────────────────────────
# container for one single YOLO detection — one bounding box
@dataclass
class Detection:
    """
    Single YOLO detection representing one bounding box.

    Attributes:
        label: Class name ("fire" or "smoke")
        confidence: Confidence score from YOLO (0.0 to 1.0)
        bbox: Bounding box as (x_center, y_center, width, height) in pixels
        area_ratio: Fraction of frame area covered by this box
    """
    label: str                              # "fire" or "smoke"
    confidence: float                       # how sure YOLO is, between 0 and 1
    bbox: tuple[int, int, int, int]         # (x, y, w, h) in pixels — x,y is CENTER
    area_ratio: float                       # box area ÷ frame area