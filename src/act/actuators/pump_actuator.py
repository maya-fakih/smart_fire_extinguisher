# src/act/actuators/pump_actuator.py

import threading
import logging
from typing import Optional

import RPi.GPIO as GPIO

from act.actuators.actuator_base import Actuator
from exceptions import ActuatorFaultError

logger = logging.getLogger(__name__)


class PumpActuator(Actuator):
    """
    GPIO-controlled water pump (or solenoid valve) with a hard safety cutoff.

    Engaged by driving its GPIO pin HIGH, disengaged by driving it LOW.
    A safety timer auto-disengages after `max_duration_s` regardless of
    further activate() calls — prevents the pump running indefinitely if
    ACT crashes or a stale prediction holds it on. To re-engage after a
    safety cutoff, ActEngine must wait for a clear cycle and call
    activate() again.

    Expected config:
        name              injected by ActuatorParser
        enabled           bool
        interface         "gpio"
        pin               int — BCM pin number, e.g. 17
        max_duration_s    float — safety cutoff in seconds, e.g. 30
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._pin = int(config['pin'])
        self._max_duration_s = float(config['max_duration_s'])

        # Set BCM mode (idempotent if already set to BCM; raises if BOARD).
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        try:
            GPIO.setup(self._pin, GPIO.OUT, initial=GPIO.LOW)
        except Exception as e:
            raise ActuatorFaultError(
                f"{self.name}: failed to init GPIO pin {self._pin}: {e}"
            )

        self._safety_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """
        Drive pin HIGH and arm the safety cutoff timer. Idempotent — repeated
        calls while already active are no-ops and do NOT extend the safety
        timer (hard cap from first activation).
        """
        with self._lock:
            if self._is_active:
                logger.debug(f"PumpActuator {self.name}: already active, ignoring activate")
                return

            try:
                GPIO.output(self._pin, GPIO.HIGH)
            except Exception as e:
                logger.error(
                    f"PumpActuator {self.name}: activation failed - "
                    f"{type(e).__name__}: {e}",
                    exc_info=True
                )
                raise ActuatorFaultError(f"{self.name}: activate failed: {e}")

            self._is_active = True
            self._safety_timer = threading.Timer(
                self._max_duration_s, self._safety_cutoff
            )
            self._safety_timer.daemon = True
            self._safety_timer.start()

            logger.info(
                f"PumpActuator {self.name}: activated | "
                f"pin={self._pin} | safety_cutoff_s={self._max_duration_s}"
            )

    def deactivate(self) -> None:
        """Drive pin LOW and cancel the safety timer. Idempotent."""
        with self._lock:
            if self._safety_timer is not None:
                self._safety_timer.cancel()
                self._safety_timer = None

            try:
                GPIO.output(self._pin, GPIO.LOW)
            except Exception as e:
                logger.warning(
                    f"PumpActuator {self.name}: deactivate write failed - "
                    f"{type(e).__name__}: {e}"
                )

            if self._is_active:
                logger.info(f"PumpActuator {self.name}: deactivated")
                self._is_active = False

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def _safety_cutoff(self) -> None:
        """Called by the safety timer thread if max_duration_s is exceeded."""
        logger.warning(
            f"PumpActuator {self.name}: SAFETY CUTOFF triggered after "
            f"{self._max_duration_s}s — forcing deactivate"
        )
        self.deactivate()

    # ------------------------------------------------------------------
    # Health & lifecycle
    # ------------------------------------------------------------------

    def _ping(self) -> None:
        """Verify the GPIO pin is still configured as an output."""
        function = GPIO.gpio_function(self._pin)
        if function != GPIO.OUT:
            raise IOError(
                f"{self.name}: GPIO pin {self._pin} is no longer configured as OUT "
                f"(current function code: {function})"
            )

    def cleanup(self) -> None:
        """Deactivate and release the GPIO pin."""
        super().cleanup()  # calls deactivate()
        try:
            GPIO.cleanup(self._pin)
        except Exception as e:
            logger.warning(
                f"PumpActuator {self.name}: GPIO cleanup error - "
                f"{type(e).__name__}: {e}"
            )