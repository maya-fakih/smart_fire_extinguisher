# src/act/actuators/pump_actuator.py

import threading
import logging
from typing import Optional

import lgpio

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

    Uses lgpio instead of RPi.GPIO — required for Pi 5 (RP1 chip).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._pin = int(config['pin'])
        self._max_duration_s = float(config['max_duration_s'])

        # Open the GPIO chip and claim the pin as output (LOW initially).
        # lgpio uses the Linux character device (/dev/gpiochip0), which
        # works on all Pi models including Pi 5.
        try:
            self._chip = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._chip, self._pin, 0)
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
                lgpio.gpio_write(self._chip, self._pin, 1)
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
                lgpio.gpio_write(self._chip, self._pin, 0)
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
        """Verify the GPIO pin is still claimed and readable."""
        # lgpio.gpio_get_mode returns: 0 = input, 1 = output
        mode = lgpio.gpio_get_mode(self._chip, self._pin)
        if mode != 1:
            raise IOError(
                f"{self.name}: GPIO pin {self._pin} is no longer configured as OUT "
                f"(current mode: {mode})"
            )

    def cleanup(self) -> None:
        """Deactivate and release the GPIO chip handle."""
        super().cleanup()  # calls deactivate()
        try:
            lgpio.gpiochip_close(self._chip)
        except Exception as e:
            logger.warning(
                f"PumpActuator {self.name}: GPIO cleanup error - "
                f"{type(e).__name__}: {e}"
            )