# src/sense/sensors/gpio_sensor.py

from __future__ import annotations
import logging
from sense.sensor_base import Sensor

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
except (ImportError, RuntimeError):
    _GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available. GPIOSensor unavailable.")

_PULL_MAP = {"up": None, "down": None, "none": None}


class GPIOSensor(Sensor):
    """
    Reads a digital HIGH/LOW state from a BCM GPIO pin.
    Returns 1.0 (HIGH) or 0.0 (LOW).
    Config keys: pin (BCM number), pull_up_down ("up"/"down"/"none", default "none").
    """

    def __init__(self, pin: int, pull: str = "none", **kwargs):
        super().__init__(**kwargs)
        self.pin = pin
        self.pull = pull.lower()
        if self.pull not in ("up", "down", "none"):
            logger.warning("GPIOSensor '%s': unknown pull '%s', defaulting to 'none'", self.name, self.pull)
            self.pull = "none"
        self._initialised = False

    def _ping(self) -> None:
        if not _GPIO_AVAILABLE:
            raise RuntimeError("RPi.GPIO not available")
        pud = {"up": GPIO.PUD_UP, "down": GPIO.PUD_DOWN, "none": GPIO.PUD_OFF}[self.pull]
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=pud)
        GPIO.input(self.pin)
        self._initialised = True
        logger.info("GPIOSensor '%s': BCM pin %d configured pull_%s", self.name, self.pin, self.pull)

    def read(self) -> float:
        if not _GPIO_AVAILABLE:
            raise RuntimeError("RPi.GPIO not available")
        if not self._initialised:
            raise IOError(f"GPIOSensor '{self.name}': pin {self.pin} not initialised — ping() must succeed first")
        return float(GPIO.input(self.pin))

    def read_matrix(self) -> list[float]:
        return []

    def stop(self) -> None:
        if _GPIO_AVAILABLE and self._initialised:
            GPIO.cleanup(self.pin)
            logger.info("GPIOSensor '%s': BCM pin %d cleaned up", self.name, self.pin)