"""
adc_sensor.py — Concrete sensor for ADS1115 ADC over I2C.

Reads an analogue value from a specified channel (pin) of an ADS1115
16-bit analogue-to-digital converter.  Typical use: MQ-series smoke/gas
sensors whose output is a 0–3.3 V analogue voltage.

Hardware dependency: adafruit-circuitpython-ads1x15
  pip install adafruit-circuitpython-ads1x15

The import is guarded so the module can be imported on a development
machine without the Adafruit library installed; a clear RuntimeError is
raised only when read() is actually called.
"""

from __future__ import annotations

import logging

from sense.sensor import Sensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional hardware imports — gracefully absent on non-Pi machines
# ---------------------------------------------------------------------------
try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn

    _ADS_AVAILABLE = True
except ImportError:
    _ADS_AVAILABLE = False
    logger.warning(
        "adafruit-circuitpython-ads1x15 not installed. "
        "ADCSensor will raise RuntimeError on read()."
    )

# Map integer channel index → ADS channel constant
_CHANNEL_MAP = {
    0: "P0",
    1: "P1",
    2: "P2",
    3: "P3",
}

# ADS1115 gain options (±voltage range)
_GAIN_MAP = {
    1:    2 / 3,   # ±6.144 V  (Adafruit uses 2/3 for GAIN_1)
    2:    2,        # ±2.048 V  — most common for 3.3 V systems
    4:    4,
    8:    8,
    16:   16,
}


class ADCSensor(Sensor):
    """
    Reads a single analogue channel from an ADS1115 ADC.

    Args:
        pin:      ADS1115 channel index (0–3).
        ads_gain: ADS1115 programmable gain amplifier setting.
                  Typical values: 1 (±6.144 V), 2 (±2.048 V).
        **kwargs: Passed directly to Sensor.__init__.
    """

    def __init__(self, pin: int, ads_gain: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.pin = pin
        self.ads_gain = ads_gain

        # Lazy-initialised hardware handles
        self._i2c = None
        self._ads = None
        self._channel = None

    # ------------------------------------------------------------------
    # Hardware initialisation (called once on first read)
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        """
        Initialise the I2C bus and ADS1115 chip.

        Deferred until the first read() call so that the object can be
        constructed safely during config parsing on any machine.
        """
        if not _ADS_AVAILABLE:
            raise RuntimeError(
                "adafruit-circuitpython-ads1x15 is not installed. "
                "Cannot read from ADCSensor."
            )

        if self._ads is not None:
            return  # Already initialised

        try:
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._ads = ADS.ADS1115(self._i2c)

            if self.ads_gain not in _GAIN_MAP:
                logger.warning(
                    "ADCSensor '%s': unknown gain %d — defaulting to 1.",
                    self.name,
                    self.ads_gain,
                )
                self.ads_gain = 1

            self._ads.gain = _GAIN_MAP[self.ads_gain]

            if self.pin not in _CHANNEL_MAP:
                raise ValueError(
                    f"ADCSensor '{self.name}': invalid pin {self.pin}. "
                    f"Must be 0–3."
                )

            channel_attr = _CHANNEL_MAP[self.pin]
            self._channel = AnalogIn(self._ads, getattr(ADS, channel_attr))

            logger.info(
                "ADCSensor '%s': initialised on channel %d with gain %d.",
                self.name,
                self.pin,
                self.ads_gain,
            )
        except Exception as exc:
            logger.error(
                "ADCSensor '%s': hardware init failed — %s", self.name, exc
            )
            raise

    # ------------------------------------------------------------------
    # Sensor ABC implementation
    # ------------------------------------------------------------------

    def read(self) -> float:
        """
        Read the raw ADC value (0–32767 for ADS1115 in single-ended mode).

        Returns the integer value as a float for consistency with the
        Sensor base class interface.

        Raises:
            RuntimeError: if the Adafruit library is not installed.
            Exception:    on any hardware communication failure.
        """
        self._init_hardware()
        raw_value = self._channel.value  # 16-bit signed integer, 0–32767
        return float(raw_value)

    # ------------------------------------------------------------------
    # Matrix support (not applicable for ADC — returns empty)
    # ------------------------------------------------------------------

    def read_matrix(self) -> list[float]:
        """ADC sensors are scalar — always returns an empty list."""
        return []