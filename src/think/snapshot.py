from dataclasses import dataclass
from datetime import datetime
from sense import SensorSnapshot
from see import VisionSnapshot

@dataclass
class ThinkSnapshot():
    # system variables
    timestamp: datetime

    # system snapshots
    sensor_snapshot: SensorSnapshot
    vision_snapshot: VisionSnapshot

