# src/act/actuators/actuator_base.py

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Actuator(ABC):
    """
    Abstract base class for all actuator types (pump, alarm, arm).

    Mirrors the Sensor ABC pattern from the SENSE layer: subclasses implement
    activate(), deactivate(), and _ping(); the base provides health checking
    via ping(), fault tracking, and a cleanup hook.

    Actuators are dispatched by ActEngine based on the recommended_actions
    written to SystemState by THINK. ArmController is a special case — it
    overlays a continuous tracking loop on top of activate/deactivate.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Actuator configuration dict. Must include 'name',
                    injected by ActuatorParser before construction.
        """
        self.name      = config['name']
        self.enabled   = config.get('enabled', True)
        self.interface = config['interface']

        self._is_active = False
        self._faulted   = False

    @property
    def faulted(self) -> bool:
        return self._faulted

    @property
    def is_active(self) -> bool:
        return self._is_active

    # ============================================
    # ABSTRACT METHODS (subclass must implement)
    # ============================================

    @abstractmethod
    def activate(self) -> None:
        """
        Engage the actuator (open valve, sound alarm, enable servos).
        Subclass is responsible for retry logic and for setting
        self._is_active = True on success.

        Raises:
            ActuatorFaultError: if the hardware fails to engage.
        """
        pass

    @abstractmethod
    def deactivate(self) -> None:
        """
        Disengage the actuator. Must be idempotent — safe to call when
        already inactive (e.g. on cleanup paths).
        Subclass sets self._is_active = False on success.
        """
        pass

    @abstractmethod
    def _ping(self) -> None:
        """
        Verify the hardware is reachable and correctly configured.
        Subclass-specific check — GPIO pin function, I2C scan, servo response.

        Raises:
            Exception: on any hardware failure (caught by ping()).
        """
        pass

    # ============================================
    # HEALTH CHECK
    # ============================================

    def ping(self) -> bool:
        """
        Returns True if the actuator is enabled, not faulted, and responds
        to a hardware ping. ActEngine calls this before dispatching an action.
        A failed ping marks the actuator as permanently faulted until the
        layer is restarted.
        """
        if not self.enabled or self._faulted:
            return False
        try:
            self._ping()
            logger.debug(f"Actuator {self.name}: ping successful")
            return True
        except Exception as e:
            logger.error(
                f"Actuator {self.name}: ping failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            self._faulted = True
            return False

    # ============================================
    # LIFECYCLE
    # ============================================

    def cleanup(self) -> None:
        """
        Release any held resources. Default: deactivate.
        Subclasses override if they hold extra state (e.g. servo channels,
        GPIO pins) that needs explicit release.
        """
        try:
            self.deactivate()
        except Exception as e:
            logger.warning(
                f"Actuator {self.name}: cleanup error - "
                f"{type(e).__name__}: {e}"
            )