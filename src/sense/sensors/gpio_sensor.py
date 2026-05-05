"""
gpio_sensor.py — Concrete sensor for digital GPIO pins.

Reads a digital HIGH/LOW state from a Raspberry Pi GPIO pin and returns
it as 1.0 (HIGH) or 0.0 (LOW).

Typical use cases:
  • Flame sensor modules with digital output (e.g. KY-026)
  • PIR motion detectors
  • Reed switches, door/window sensors
  • Any binary presence/detection sensor

Hardware dependency: RPi.GPIO (Raspberry Pi only)
  pip install RPi.GPIO

Gracefully absent on non-Pi machines — RuntimeError raised only on read().
The pull resistor direction is configurable via the "pull" field in config.json:
  "up"   → GPIO.PUD_UP   (pin reads HIGH when open, LOW when activated)
  "down" → GPIO.PUD_DOWN (pin reads LOW when open, HIGH when activated)
  "none" → no pull resistor
"""

from __future__ import annotations

import logging

from sense.sensor import Sensor

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO  # type: ignore

    _GPIO_AVAILABLE = True
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
except (ImportError, RuntimeError):
    _GPIO_AVAILABLE = False
    logger.warning(
        "RPi.GPIO not available. GPIOSensor will raise RuntimeError on read()."
    )

_PULL_MAP = {
    "up":   "GPIO.PUD_UP",
    "down": "GPIO.PUD_DOWN",
    "none": "GPIO.PUD_OFF",
}


class GPIOSensor(Sensor):
    """
    Reads the digital state of a BCM-numbered GPIO pin.

    Args:
        pin:  BCM GPIO pin number (e.g. 17).
        pull: Pull resistor direction: "up", "down", or "none".
        **kwargs: Passed to Sensor.__init__.
    """

    def __init__(self, pin: int, pull: str = "up", **kwargs):
        super().__init__(**kwargs)
        self.pin = pin
        self.pull = pull.lower()

        # Validate pull direction
        if self.pull not in _PULL_MAP:
            logger.warning(
                "GPIOSensor '%s': unknown pull direction '%s' — defaulting to 'up'.",
                self.name,
                self.pull,
            )
            self.pull = "up"

        self._initialised = False

    # ------------------------------------------------------------------
    # Hardware initialisation
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        """Configure the GPIO pin as input with the specified pull resistor."""
        if self._initialised:
            return

        if not _GPIO_AVAILABLE:
            raise RuntimeError(
                "RPi.GPIO is not installed or not running on a Raspberry Pi. "
                "Cannot read from GPIOSensor."
            )

        pull_const = {
            "up":   GPIO.PUD_UP,
            "down": GPIO.PUD_DOWN,
            "none": GPIO.PUD_OFF,
        }[self.pull]

        try:
            GPIO.setup(self.pin, GPIO.IN, pull_up_down=pull_const)
            self._initialised = True
            logger.info(
                "GPIOSensor '%s': pin BCM %d configured as INPUT with pull_%s.",
                self.name,
                self.pin,
                self.pull,
            )
        except Exception as exc:
            logger.error(
                "GPIOSensor '%s': GPIO.setup failed for pin %d — %s",
                self.name,
                self.pin,
                exc,
            )
            raise

    # ------------------------------------------------------------------
    # Sensor ABC implementation
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """
        Test if the GPIO pin is accessible.

        Attempts to configure the pin and read its current state.
        If the pin is available and not in use, this succeeds.

        Returns:
            True if pin is accessible, False otherwise.
        """
        if not _GPIO_AVAILABLE:
            return False

        try:
            # Try to setup and read the pin
            pull_const = {
                "up":   GPIO.PUD_UP,
                "down": GPIO.PUD_DOWN,
                "none": GPIO.PUD_OFF,
            }[self.pull]

            GPIO.setup(self.pin, GPIO.IN, pull_up_down=pull_const)
            _ = GPIO.input(self.pin)  # Try to read the pin state
            return True
        except Exception:
            return False

    def read(self) -> float:
        """
        Read the digital state of the GPIO pin.

        Returns:
            1.0 if the pin is HIGH, 0.0 if LOW.

        Raises:
            RuntimeError: if RPi.GPIO is not available.
            IOError:      on any GPIO read failure.
        """
        if not _GPIO_AVAILABLE:
            raise RuntimeError(
                "RPi.GPIO is not installed or not running on a Raspberry Pi. "
                "Cannot read from GPIOSensor."
            )

        try:
            self._init_hardware()
        except Exception as exc:
            raise IOError(
                f"GPIOSensor '{self.name}': hardware initialization failed — {exc}"
            ) from exc

        if not self._initialised:
            raise IOError(
                f"GPIOSensor '{self.name}': pin BCM {self.pin} not initialized"
            )

        try:
            state = GPIO.input(self.pin)
        except Exception as exc:
            raise IOError(
                f"GPIOSensor '{self.name}': GPIO.input({self.pin}) failed — {exc}"
            ) from exc

        # Validate state is either 0 or 1
        if state not in (0, 1, True, False):
            raise IOError(
                f"GPIOSensor '{self.name}': GPIO.input returned invalid state: {state}"
            )

        return float(state)

    # ------------------------------------------------------------------
    # Matrix support (not applicable for GPIO)
    # ------------------------------------------------------------------

    def read_matrix(self) -> list[float]:
        """GPIO sensors are scalar — always returns an empty list."""
        return []

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Clean up the GPIO pin before stopping the thread."""
        super().stop()
        if _GPIO_AVAILABLE and self._initialised:
            try:
                GPIO.cleanup(self.pin)
                logger.info(
                    "GPIOSensor '%s': GPIO pin %d cleaned up.",
                    self.name,
                    self.pin,
                )
            except Exception as exc:
                logger.warning(
                    "GPIOSensor '%s': GPIO.cleanup error — %s", self.name, exc
                )