"""
SEE Layer - Vision/Perception Module.

This module handles all visual perception tasks for the fire robot system.
It manages camera hardware (IMX500), runs YOLO fire detection, processes results,
and outputs structured VisionSnapshot objects to the THINK layer.

Core Components:
    - VisionFuser: Orchestrates the entire SEE layer (capture → analyze → emit)
    - IMX500Camera: Manages camera hardware and frame capture
    - FireDetector: Parses YOLO inference results and clusters detections
    - VisionSnapshot: Output contract - structured perception data for THINK layer
"""

from see.vision_fuser import VisionFuser
from see.camera import IMX500Camera
from see.snapshot import VisionSnapshot
from see.models.detection import Detection
from see.models.fire_cluster import FireCluster
