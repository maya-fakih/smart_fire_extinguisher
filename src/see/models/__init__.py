"""
Vision Models Module.

Contains model interfaces and implementations for fire detection and scene analysis.

Classes:
    VisionModel: Abstract base class for all vision models
    Detection: Single YOLO detection (one bounding box)
    FireCluster: Group of connected fire/smoke detections
    FireDetector: YOLO fire/smoke detection analyzer
    SceneClassifier: Scene understanding (currently unused)
"""

from see.models.vision_model_base import VisionModel
from see.models.detection import Detection
from see.models.fire_cluster import FireCluster
from see.models.fire_detector import FireDetector
from see.models.scene_classifier import SceneClassifier