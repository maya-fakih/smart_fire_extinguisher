# src/act/actuators/arm_controller.py

import threading
import time
import logging
from typing import Optional

import numpy as np
from gpiozero import Servo

from core.system_state import SystemState
from act.actuators.actuator_base import Actuator
from exceptions import ActuatorFaultError

logger = logging.getLogger(__name__)


class ArmController(Actuator):
    """
    Pan/tilt 2-DOF arm with closed-loop visual servoing.

    Special among actuators: takes a SystemState reference because its
    tracking thread reads `latest_heat_matrix`, `latest_fire_x/y`, and
    `sensor_triggered` continuously. ActuatorParser passes state only
    when building this class.

    Feedback hierarchy:
      1. If heat matrix has a peak above heat_use_threshold_c → heat error
      2. Else if camera has a fire detection (latest_fire_x not None) → camera error
      3. Else → no movement this cycle

    Errors are normalized to [-1, +1] from image/grid center. Each joint
    steps by step_deg in the error's sign direction (dead-band controller).
    No IK — pan and tilt are independent loops.
    """

    def __init__(self, config: dict, state: SystemState):
        super().__init__(config)
        self._state = state

        joints = config["joints"]
        self._pan_cfg  = joints["pan"]
        self._tilt_cfg = joints["tilt"]

        self._pan_servo  = Servo(pin=int(self._pan_cfg["pin"]))
        self._tilt_servo = Servo(pin=int(self._tilt_cfg["pin"]))

        # Current commanded angles (start centered)
        self._pan_angle  = 0.0
        self._tilt_angle = 0.0

        feedback = config.get("feedback", {})
        self._heat_use_threshold = float(feedback.get("heat_use_threshold_c", 35.0))
        self._tolerance          = float(feedback.get("tolerance_normalized", 0.1))

        self._cycle_active_ms = int(config.get("cycle_active_ms", 100))
        self._cycle_idle_ms   = int(config.get("cycle_idle_ms",   500))

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Center the servos at construction so they don't sit at a random angle
        self._command_pan(0.0)
        self._command_tilt(0.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Start the tracking thread. Idempotent."""
        if self._is_active:
            logger.debug(f"ArmController {self.name}: already active")
            return
        try:
            self._running = True
            self._thread = threading.Thread(
                target=self._tracking_loop,
                name=f"ArmTracker-{self.name}",
                daemon=True,
            )
            self._thread.start()
            self._is_active = True
            logger.info(f"ArmController {self.name}: tracking thread started")
        except Exception as e:
            self._running = False
            logger.error(
                f"ArmController {self.name}: failed to start - "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise ActuatorFaultError(f"{self.name}: activate failed: {e}")

    def deactivate(self) -> None:
        """Stop the tracking thread and neutralize servos. Idempotent."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        try:
            self._command_pan(0.0)
            self._command_tilt(0.0)
        except Exception as e:
            logger.warning(
                f"ArmController {self.name}: neutralize failed - "
                f"{type(e).__name__}: {e}"
            )
        if self._is_active:
            logger.info(f"ArmController {self.name}: tracking stopped")
            self._is_active = False

    def _ping(self) -> None:
        """gpiozero servos don't have a real ping — verify objects exist and value is readable."""
        if self._pan_servo is None or self._tilt_servo is None:
            raise IOError(f"{self.name}: servo object is None")
        _ = self._pan_servo.value   # raises if pin not configured
        _ = self._tilt_servo.value

    def cleanup(self) -> None:
        """Stop tracking and release GPIO pins."""
        super().cleanup()  # calls deactivate()
        try:
            self._pan_servo.close()
            self._tilt_servo.close()
        except Exception as e:
            logger.warning(
                f"ArmController {self.name}: servo close failed - "
                f"{type(e).__name__}: {e}"
            )

    # ------------------------------------------------------------------
    # Tracking loop
    # ------------------------------------------------------------------

    def _tracking_loop(self) -> None:
        logger.debug(f"ArmController {self.name}: tracking loop entered")
        while self._running and self._state.system_running:
            try:
                triggered = self._state.sensor_triggered

                err = self._compute_error()
                if err is None:
                    self._sleep_cycle(triggered)
                    continue

                err_x, err_y = err
                err_mag = (err_x ** 2 + err_y ** 2) ** 0.5

                if err_mag > self._tolerance:
                    self._step_joints(err_x, err_y)

                self._sleep_cycle(triggered)

            except Exception as e:
                logger.error(
                    f"ArmController {self.name}: tracking loop error - "
                    f"{type(e).__name__}: {e}",
                    exc_info=True,
                )
                time.sleep(1)

        logger.debug(f"ArmController {self.name}: tracking loop exited")

    def _sleep_cycle(self, triggered: bool) -> None:
        interval_ms = self._cycle_active_ms if triggered else self._cycle_idle_ms
        time.sleep(interval_ms / 1000.0)

    # ------------------------------------------------------------------
    # Feedback fusion
    # ------------------------------------------------------------------

    def _compute_error(self) -> Optional[tuple]:
        """
        Hierarchical: heat first if it has a real peak, else camera, else None.
        Returns (err_x, err_y) normalized to [-1, +1] or None.
        """
        heat_err = self._heat_error(self._state.latest_heat_matrix)
        if heat_err is not None:
            return heat_err

        cam_err = self._camera_error(
            self._state.latest_fire_x, self._state.latest_fire_y
        )
        return cam_err

    def _heat_error(self, heat) -> Optional[tuple]:
        if heat is None or len(heat) == 0:
            return None
        arr = np.array(heat, dtype=float)
        if arr.size == 0 or float(arr.max()) < self._heat_use_threshold:
            return None
        flat_idx = int(arr.argmax())
        rows, cols = arr.shape
        r, c = divmod(flat_idx, cols)
        cx = (cols - 1) / 2.0
        cy = (rows - 1) / 2.0
        err_x = (c - cx) / cx if cx > 0 else 0.0
        err_y = (r - cy) / cy if cy > 0 else 0.0
        return err_x, err_y

    def _camera_error(self, fx, fy) -> Optional[tuple]:
        if fx is None or fy is None:
            return None
        # Camera coords are [0, 1] with (0.5, 0.5) as center.
        err_x = (fx - 0.5) * 2.0
        err_y = (fy - 0.5) * 2.0
        return err_x, err_y

    # ------------------------------------------------------------------
    # Joint stepping
    # ------------------------------------------------------------------

    def _step_joints(self, err_x: float, err_y: float) -> None:
        # Pan responds to horizontal error.
        if abs(err_x) > self._tolerance / 2:
            direction = 1 if err_x > 0 else -1
            if self._pan_cfg.get("invert", False):
                direction = -direction
            self._pan_angle = self._clip(
                self._pan_angle + direction * self._pan_cfg["step_deg"],
                self._pan_cfg["limit_min_deg"],
                self._pan_cfg["limit_max_deg"],
            )
            self._command_pan(self._pan_angle)

        # Tilt responds to vertical error.
        if abs(err_y) > self._tolerance / 2:
            direction = 1 if err_y > 0 else -1
            if self._tilt_cfg.get("invert", False):
                direction = -direction
            self._tilt_angle = self._clip(
                self._tilt_angle + direction * self._tilt_cfg["step_deg"],
                self._tilt_cfg["limit_min_deg"],
                self._tilt_cfg["limit_max_deg"],
            )
            self._command_tilt(self._tilt_angle)

    def _command_pan(self, deg: float) -> None:
        self._pan_servo.value = self._deg_to_value(
            deg, self._pan_cfg["servo_min_deg"], self._pan_cfg["servo_max_deg"]
        )

    def _command_tilt(self, deg: float) -> None:
        self._tilt_servo.value = self._deg_to_value(
            deg, self._tilt_cfg["servo_min_deg"], self._tilt_cfg["servo_max_deg"]
        )

    @staticmethod
    def _deg_to_value(deg: float, servo_min_deg: float, servo_max_deg: float) -> float:
        """Map degrees → gpiozero.Servo.value in [-1, +1]."""
        span = servo_max_deg - servo_min_deg
        if span <= 0:
            return 0.0
        v = 2.0 * (deg - servo_min_deg) / span - 1.0
        return max(-1.0, min(1.0, v))

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))