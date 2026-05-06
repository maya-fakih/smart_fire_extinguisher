from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SensorSnapshot:
    """
    Output contract of the SENSE layer.
    Emitted to SystemState.sense_queue after a threshold crossing.

    Built by SensorFuser from the latest readings across all active sensors.
    Sensor names are dynamic — defined in config.json.
    ThinkEngine extracts flat values from this to build ThinkSnapshot.
    """

    timestamp: datetime = field(default_factory=datetime.now)

    sensor_readings: dict[str, float] = field(default_factory=dict)
    # sensor_name → physical value in real-world units
    # e.g. {"smoke": 342.1, "temp": 67.4, "co": 12.0}

    sensor_normalized: dict[str, float] = field(default_factory=dict)
    # sensor_name → normalized 0.0–1.0
    # e.g. {"smoke": 0.34, "temp": 0.45, "co": 0.12}

    enabled_sensors: list[str] = field(default_factory=list)
    # names of sensors set as enabled in config.json

    triggered_sensors: list[str] = field(default_factory=list)
    # names of sensors that crossed their threshold this reading
    # e.g. ["smoke", "temp"]

    faulty_sensors: list[str] = field(default_factory=list)
    # names of sensors currently faulted and removed from active pool
    # e.g. ["co"] — SensorFuser continues with remaining healthy sensors