import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SensorSnapshot:
    """
    Output contract of the SENSE layer.
    Emitted to SystemState.sense_queue after a threshold crossing.

    Built by SensorFuser from the latest readings across all active sensors.
    Sensor names are dynamic — defined in config.json.
    ThinkEngine extracts flat values from this to build ThinkSnapshot.

    Note: field names sensor_readings and sensor_normalized are intentional
    renames from the spec's readings and normalized for clarity.
    """

    # Unix timestamp in seconds — use time.time() when constructing
    timestamp: float = field(default_factory=time.time)

    sensor_readings: Dict[str, float] = field(default_factory=dict)
    # sensor_name → physical value in real-world units
    # e.g. {"smoke": 342.1, "temp": 67.4, "co": 12.0}

    sensor_normalized: Dict[str, float] = field(default_factory=dict)
    # sensor_name → normalized 0.0–1.0
    # e.g. {"smoke": 0.34, "temp": 0.45, "co": 0.12}

    triggered_sensors: List[str] = field(default_factory=list)
    # names of sensors that crossed their threshold this reading
    # e.g. ["smoke", "temp"]

    disabled_sensors: List[str] = field(default_factory=list)
    # names of sensors currently faulted and removed from active pool
    # e.g. ["co"] — SensorFuser continues with remaining healthy sensors

    raw_matrices: Dict[str, List[float]] = field(default_factory=dict)
    # grid data for matrix sensors like heat map
    # always present — empty dict {} if no matrix sensors exist
    # e.g. {"heat_matrix": [23.1, 24.5, ...]} — flat list, shape in config