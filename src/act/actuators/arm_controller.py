# src/act/actuators/arm_controller.py

import threading
import time
import logging
from typing import Optional

import numpy as np
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory
_PIN_FACTORY = LGPIOFactory()

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

        self._pan_servo  = Servo(pin=int(self._pan_cfg["pin"]), pin_factory=_PIN_FACTORY)
        self._tilt_servo = Servo(pin=int(self._tilt_cfg["pin"]), pin_factory=_PIN_FACTORY)
        
        # Current commanded angles (start centered)
        self._pan_angle  = 0.0
        self._tilt_angle = 0.0

        feedback = config.get("feedback", {})
        self._heat_use_threshold = float(feedback.get("heat_use_threshold_c", 35.0))
        self._tolerance          = float(feedback.get("tolerance_normalized", 0.1))

        # Per-sensor bias in normalized space (added to that sensor's error
        # vector before fusion). Calibrate by trial and error — these are
        # corrections, not physical offsets.
        offsets = feedback.get("sensor_offsets", {}) or {}
        heat_off   = offsets.get("heat",   {}) or {}
        camera_off = offsets.get("camera", {}) or {}
        self._heat_bias_x   = float(heat_off.get("x_bias",   0.0))
        self._heat_bias_y   = float(heat_off.get("y_bias",   0.0))
        self._camera_bias_x = float(camera_off.get("x_bias", 0.0))
        self._camera_bias_y = float(camera_off.get("y_bias", 0.0))

        self._cycle_active_ms = int(config.get("cycle_active_ms", 100))
        self._cycle_idle_ms   = int(config.get("cycle_idle_ms",   500))

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Adaptive direction state — tracks last error and learned step sign
        # per axis so the arm self-corrects if the servo is physically inverted.
        # After one wrong step the sign flips and stays correct for the session.
        self._pan_last_err  = 0.0
        self._pan_step_sign = 1
        self._tilt_last_err = 0.0
        self._tilt_step_sign = 1

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
                # are we sure we should make it sleap instead of in the main loop checking the state? like if we are in manual state only we turn it on?
                if time.time() < self._state.arm_manual_mode_until:
                    time.sleep(0.1)
                    continue

                triggered = self._state.sensor_triggered

                err = self._compute_error()
                if err is None:
                    self._sleep_cycle(triggered)
                    continue

                err_x, err_y = err
                err_mag = (err_x ** 2 + err_y ** 2) ** 0.5

                if err_mag > self._tolerance:
                    logger.debug(
                        f"ArmController {self.name}: stepping | "
                        f"err_x={err_x:.3f} err_y={err_y:.3f} mag={err_mag:.3f} | "
                        f"pan={self._pan_angle:.1f}° tilt={self._tilt_angle:.1f}°"
                    )
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
        Three-mode fusion of heat and camera signals:
          - both available    → average of (heat + heat_bias) and (camera + camera_bias)
          - heat only         → heat + heat_bias
          - camera only       → camera + camera_bias
          - neither           → None

        "Available" = the sensor's error function returned non-None.
        For heat that means a peak above heat_use_threshold_c.
        For camera that means latest_fire_x/y are both not None (SEE is
        running AND a fire cluster was detected).

        Returns (err_x, err_y) normalized to [-1, +1] or None.
        """
        heat_raw   = self._heat_error(self._state.latest_heat_matrix)
        camera_raw = self._camera_error(
            self._state.latest_fire_x, self._state.latest_fire_y
        )

        if heat_raw is not None and camera_raw is not None:
            heat_corr   = (heat_raw[0]   + self._heat_bias_x,   heat_raw[1]   + self._heat_bias_y)
            camera_corr = (camera_raw[0] + self._camera_bias_x, camera_raw[1] + self._camera_bias_y)
            result = (
                (heat_corr[0] + camera_corr[0]) / 2.0,
                (heat_corr[1] + camera_corr[1]) / 2.0,
            )
            logger.debug(f"ArmController {self.name}: source=heat+camera err=({result[0]:.3f}, {result[1]:.3f})")
            return result

        if heat_raw is not None:
            result = (heat_raw[0] + self._heat_bias_x, heat_raw[1] + self._heat_bias_y)
            logger.debug(f"ArmController {self.name}: source=heat err=({result[0]:.3f}, {result[1]:.3f})")
            return result

        if camera_raw is not None:
            result = (camera_raw[0] + self._camera_bias_x, camera_raw[1] + self._camera_bias_y)
            logger.debug(f"ArmController {self.name}: source=camera err=({result[0]:.3f}, {result[1]:.3f})")
            return result

        return None

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
        """
        Adaptive step: move one step_deg increment toward the target, but
        learn the correct direction from feedback rather than assuming it.

        On the first step (or after centering) we guess: positive error →
        positive direction, adjusted by the invert flag. On every subsequent
        step we compare the new error magnitude to the previous one:
          - smaller  → we're converging, keep going
          - larger   → we overshot or went the wrong way, flip direction

        This makes the arm self-correcting regardless of how the servo is
        physically mounted — one wrong step and it auto-corrects.
        Step size of 0.2–0.5° (set via config) keeps corrections smooth.
        """
        # ── Pan axis ──────────────────────────────────────────────────
        if abs(err_x) > self._tolerance / 2:
            if self._pan_last_err == 0.0:
                # First step: guess direction from error sign + invert flag
                self._pan_step_sign = 1 if err_x > 0 else -1
                if self._pan_cfg.get("invert", False):
                    self._pan_step_sign *= -1
            elif abs(err_x) > abs(self._pan_last_err):
                # Error got worse → flip direction, self-correct
                self._pan_step_sign *= -1
                logger.info(
                    f"ArmController {self.name}: pan direction flipped "
                    f"(err {self._pan_last_err:+.3f} → {err_x:+.3f})"
                )

            self._pan_angle = self._clip(
                self._pan_angle + self._pan_step_sign * self._pan_cfg["step_deg"],
                self._pan_cfg["limit_min_deg"],
                self._pan_cfg["limit_max_deg"],
            )
            self._command_pan(self._pan_angle)
            self._pan_last_err = err_x
        else:
            # Within tolerance — reset so next activation starts fresh
            self._pan_last_err  = 0.0
            self._pan_step_sign = 1

        # ── Tilt axis ─────────────────────────────────────────────────
        if abs(err_y) > self._tolerance / 2:
            if self._tilt_last_err == 0.0:
                self._tilt_step_sign = 1 if err_y > 0 else -1
                if self._tilt_cfg.get("invert", False):
                    self._tilt_step_sign *= -1
            elif abs(err_y) > abs(self._tilt_last_err):
                self._tilt_step_sign *= -1
                logger.info(
                    f"ArmController {self.name}: tilt direction flipped "
                    f"(err {self._tilt_last_err:+.3f} → {err_y:+.3f})"
                )

            self._tilt_angle = self._clip(
                self._tilt_angle + self._tilt_step_sign * self._tilt_cfg["step_deg"],
                self._tilt_cfg["limit_min_deg"],
                self._tilt_cfg["limit_max_deg"],
            )
            self._command_tilt(self._tilt_angle)
            self._tilt_last_err = err_y
        else:
            self._tilt_last_err  = 0.0
            self._tilt_step_sign = 1

    def _command_pan(self, deg: float) -> None:
        val = self._deg_to_value(deg, self._pan_cfg["servo_min_deg"], self._pan_cfg["servo_max_deg"])
        self._pan_servo.value = val
        logger.debug(f"ArmController {self.name}: pan → {deg:.1f}° (servo_val={val:.3f})")

    def _command_tilt(self, deg: float) -> None:
        val = self._deg_to_value(deg, self._tilt_cfg["servo_min_deg"], self._tilt_cfg["servo_max_deg"])
        self._tilt_servo.value = val
        logger.debug(f"ArmController {self.name}: tilt → {deg:.1f}° (servo_val={val:.3f})")

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
    
    # ------------------------------------------------------------------
    # Manual control (called by ActEngine from manual_commands queue)
    # ------------------------------------------------------------------

    def nudge(self, direction: str) -> None:
        """
        Step the arm one increment in the given direction.
        direction: 'pan_left' | 'pan_right' | 'tilt_up' | 'tilt_down'
        """
        if direction == "pan_left":
            self._pan_angle = self._clip(
                self._pan_angle - self._pan_cfg["step_deg"],
                self._pan_cfg["limit_min_deg"], self._pan_cfg["limit_max_deg"]
            )
            self._command_pan(self._pan_angle)
        elif direction == "pan_right":
            self._pan_angle = self._clip(
                self._pan_angle + self._pan_cfg["step_deg"],
                self._pan_cfg["limit_min_deg"], self._pan_cfg["limit_max_deg"]
            )
            self._command_pan(self._pan_angle)
        elif direction == "tilt_up":
            self._tilt_angle = self._clip(
                self._tilt_angle - self._tilt_cfg["step_deg"],
                self._tilt_cfg["limit_min_deg"], self._tilt_cfg["limit_max_deg"]
            )
            self._command_tilt(self._tilt_angle)
        elif direction == "tilt_down":
            self._tilt_angle = self._clip(
                self._tilt_angle + self._tilt_cfg["step_deg"],
                self._tilt_cfg["limit_min_deg"], self._tilt_cfg["limit_max_deg"]
            )
            self._command_tilt(self._tilt_angle)
        else:
            logger.warning(f"ArmController.nudge: unknown direction '{direction}'")