from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class VisionSnapshot:
    """
    Output contract of the SEE layer.
    Emitted to SystemState.see_queue after a threshold crossing.

    Built by VisionFuser from the latest readings across all active cameras.
    Labels are dynamic and set in config.json
    ThinkEngine extracts flat values from this to build ThinkSnapshot.
    """

    # all strings will be transformed to enums at the think layer
    # since xgboost works only with numeric data
    # all labels are defined in config.json so all models know what to expect

    timestamp: datetime = field(default_factory=datetime.now)

    # scene understanding
    scene_label: str
    scene_confidence: str

    # fire assesment:
    composite_label: str    # fire_smoke fire smoke none

    # scene analysis
    glimpsed_fire: bool     # true if any fire box got detected by yolo even if low accuracy
    human_near_fire: bool   # if human label and fire whatsoever co-exist

    # vision statistics to guide xgboost for a larger scale understanding
    fire_count: int         # nb of bounding boxes with fire label
    smoke_count: int        # nb of bounding boxes with smoke label
    fire_union_area: float  # overlap_corrected union of fire bounding boxes
    smoke_union_area: float # overlap_corrected union of smoke bounding boxes

    # fire clusters
    # a fire cluster is calculated after merging connected bounding boxes together to form a cluster
    # having more than one cluster is a huge risk indicator
    # model is expected to learn that higher cluster count means fire is speading
    # dominant cluster is of index 0
    cluster_count: int      # risk indicator
    fire_clusters: list     # of type FireClusters ordered by danger score

    image_url: str          # url to the captured frame saved to google dirve

    raw_detections: list    # list of type Detection includes everything yolo returns for all selected bbxs
    


    



