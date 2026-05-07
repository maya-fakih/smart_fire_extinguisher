# src/sense/sensors/adc_sensor.py

from __future__ import annotations
import logging
from sense.sensor_base import Sensor

logger = logging.getLogger(__name__)

try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    _ADS_AVAILABLE = True
except ImportError:
    _ADS_AVAILABLE = False
    logger.warning("adafruit-circuitpython-ads1x15 not installed. ADCSensor unavailable.")

_CHANNEL_MAP = {0: "P0", 1: "P1", 2: "P2", 3: "P3"}
_GAIN_MAP = {1: 2/3, 2: 2, 4: 4, 8: 8, 16: 16}


class ADCSensor(Sensor):
    """
    Reads an analogue channel from an ADS1115 ADC over I2C.
    Config keys: pin (channel 0–3), gain (default 1).
    """

    def __init__(self, pin: int, ads_gain: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.pin = pin
        self.ads_gain = ads_gain
        self._i2c = None
        self._ads = None
        self._channel = None

    def _ping(self) -> None:
        if not _ADS_AVAILABLE:
            raise RuntimeError("adafruit-circuitpython-ads1x15 not installed")
        if self.pin not in _CHANNEL_MAP:
            raise ValueError(f"ADCSensor '{self.name}': invalid pin {self.pin}, must be 0–3")
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1115(i2c)
        if self.ads_gain in _GAIN_MAP:
            ads.gain = _GAIN_MAP[self.ads_gain]
        self._i2c = i2c
        self._ads = ads
        self._channel = AnalogIn(ads, getattr(ADS, _CHANNEL_MAP[self.pin]))
        logger.info("ADCSensor '%s': initialised on channel %d gain %d", self.name, self.pin, self.ads_gain)

    def read(self) -> float:
        if not _ADS_AVAILABLE:
            raise RuntimeError("adafruit-circuitpython-ads1x15 not installed")
        if self._channel is None:
            raise IOError(f"ADCSensor '{self.name}': hardware not initialised — ping() must succeed first")
        raw = self._channel.value
        if raw is None or not (0 <= raw <= 32767):
            raise IOError(f"ADCSensor '{self.name}': unexpected raw value {raw}")
        return float(raw)

    def read_matrix(self) -> list[float]:
        return []