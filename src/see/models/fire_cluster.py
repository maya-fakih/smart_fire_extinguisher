"""
FireCluster Module - Group of Connected Fire/Smoke Detections.

Container for one fire cluster — a group of connected fire/smoke boxes.
Uses connected components clustering: boxes are grouped if they intersect.
A fire spreading across multiple disconnected areas = multiple clusters.
Dangerous as it suggests fire is not localized.
"""

from dataclasses import dataclass
from see.models.detection import Detection


# ── FireCluster ───────────────────────────────────────────────────────────────
# container for one fire cluster — a group of connected fire/smoke boxes
@dataclass
class FireCluster:
    """
    Cluster of connected fire/smoke detections.

    Uses connected components clustering: boxes are grouped if they intersect.
    A fire spreading across multiple disconnected areas = multiple clusters.
    Dangerous as it suggests fire is not localized.

    Attributes:
        cluster_id: Unique identifier for this cluster
        has_fire: Whether cluster contains fire detections
        has_smoke: Whether cluster contains smoke detections
        composite_label: Overall label ("fire", "smoke", or "fire_smoke")
        fire_boxes: List of fire Detection objects in cluster
        smoke_boxes: List of smoke Detection objects in cluster
        box_count: Total number of boxes in cluster
        origin_x: Normalized [0, 1] X coordinate of cluster center (0.5 = image center)
        origin_y: Normalized [0, 1] Y coordinate of cluster center (0.5 = image center)
        primary_bbox: Bounding box of most confident detection
        total_area_ratio: Fraction of frame covered by cluster
        fire_area_ratio: Fraction of frame covered by fire in cluster
        smoke_area_ratio: Fraction of frame covered by smoke in cluster
        primary_label: Label of most confident detection
        primary_confidence: Average confidence across cluster
    """

    # ── Identification ────────────────────────────────────────────────────────
    cluster_id: int                         # unique number for this cluster

    # ── Composition flags ─────────────────────────────────────────────────────
    has_fire: bool                          # does this cluster have any fire boxes?
    has_smoke: bool                         # does this cluster have any smoke boxes?
    composite_label: str                    # "fire_smoke", "fire", or "smoke"

    # ── Raw boxes ─────────────────────────────────────────────────────────────
    fire_boxes: list[Detection]             # all fire Detection objects in this cluster
    smoke_boxes: list[Detection]            # all smoke Detection objects in this cluster
    box_count: int                          # total number of boxes (fire + smoke)

    # ── Spatial info ──────────────────────────────────────────────────────────
    # origin_x, origin_y are NORMALIZED [0, 1] — hardware-agnostic, model-portable
    # (0.5, 0.5) = image center, which is what ACT expects as feedback signal
    origin_x: float                         # normalized [0, 1] center x of cluster
    origin_y: float                         # normalized [0, 1] center y of cluster
    primary_bbox: tuple[int, int, int, int] # bbox of the most confident box (still in pixels)

    # ── Area ratios (all as fraction of frame area) ───────────────────────────
    total_area_ratio: float                 # union area of ALL boxes ÷ frame area
    fire_area_ratio: float                  # union area of fire boxes only ÷ frame area
    smoke_area_ratio: float                 # union area of smoke boxes only ÷ frame area

    # ── Confidence ────────────────────────────────────────────────────────────
    primary_label: str                      # label of the most confident box
    primary_confidence: float               # average confidence score in this cluster

    # ── Note ──────────────────────────────────────────────────────────────────
    # danger_score = primary_confidence × total_area_ratio
    # this is computed by THINK, not stored here
    # index 0 = most dangerous cluster