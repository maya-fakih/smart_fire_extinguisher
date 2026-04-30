from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SensorSnapshot:

    # WHEN was this snapshot taken
    timestamp: float

    # the actual sensor values example: {"smoke": 250.0, "temperature": 45.0}
    readings: Dict[str, float] = field(default_factory=dict)

    # same values but between 0.0 and 1.0 example: {"smoke": 0.25, "temperature": 0.45}
    normalized: Dict[str, float] = field(default_factory=dict)

    # which sensors went above the danger limit example: ["smoke", "temperature"]
    triggered_sensors: List[str] = field(default_factory=list)

    # which sensors are broken or turned off example: ["gas_sensor"]
    disabled_sensors: List[str] = field(default_factory=list)

    # grid data for matrix sensors like a heat map
    # stays empty {} if no matrix sensors exist
    raw_matrices: Dict[str, List[float]] = field(default_factory=dict)