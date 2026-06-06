# src/act/act_engine.py

import time
import logging

from core.system_state import SystemState
from core.enums import SystemMode
from act.actuator_parser import ActuatorParser
from exceptions import ActuatorFaultError

logger = logging.getLogger(__name__)


class ActEngine:
    """
    Main ACT layer engine. Dispatches actions based on system_mode using
    plain if-branches (no ActMode classes — modes are too thin to justify).

    Loop behaviour each tick:
      1. Reconcile actuator state with current intent (pump on iff
         suppress is active and mode allows).
      2. On a new prediction_id, dispatch by mode:
           - autopilot     → fire actuators + notify
           - copilot       → request approval, fire on 'approved'
           - surveillance  → notify only, no autonomous firing
           - training      → no autonomous firing (THINK handles labelling)

    The arm tracking thread runs continuously across all modes — mode only
    gates pump/alarm-type actuators.
    """

    def __init__(self, config: dict, state: SystemState, notifier=None):
        self._config = config
        self._state = state
        self._notifier = notifier

        act_cfg = config.get("act", {})
        sys_cfg = act_cfg.get("system", {})
        self._cycle_s              = sys_cfg.get("cycle_ms", 100) / 1000.0
        self._danger_threshold     = int(sys_cfg.get("danger_threshold_to_act", 3))
        self._copilot_timeout_s    = float(sys_cfg.get("copilot_timeout_s", 60))

        actions_cfg = act_cfg.get("actions", {})
        self._requires_actuator = actions_cfg.get("requires_actuator", {})

        self._actuators: dict = {}   # name → Actuator instance
        self._running = False

        # Dedup
        self._last_handled_pred_id = 0

        # Copilot wait state
        self._copilot_waiting = False
        self._copilot_waiting_since = 0.0
        self._copilot_pending_actions: list = []
        self._copilot_pending_pred_id = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        from core.child_logging import setup_child_logging
        setup_child_logging()
        logger.info("ActEngine: starting")
        try:
            # Child process: ignore SIGINT — Ctrl+C is handled by the parent only.
            # SIGTERM is still handled (see below) so orchestrator.stop() works cleanly.
            import signal as _signal
            _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
            # Install SIGTERM handler so process.terminate() from the orchestrator
            # still runs cleanup (neutralizes arm, stops pump) before exiting.
            def _on_term(sig, frame):
                logger.info("ActEngine: SIGTERM received — running cleanup")
                self.stop()
                raise SystemExit(0)
            _signal.signal(_signal.SIGTERM, _on_term)

            built = ActuatorParser.build_actuators(self._config, self._state)
            self._actuators = {a.name: a for a in built}
            logger.info(f"ActEngine: built actuators {list(self._actuators)}")

            # Arm runs always — activate at boot in every mode
            if "arm" in self._actuators:
                try:
                    self._actuators["arm"].activate()
                except ActuatorFaultError as e:
                    logger.error(f"ActEngine: arm activation failed - {e}")

            self._running = True
            self._state.act_running = True
            self._run_loop()
            # system_running went False — clean up actuators (neutralize arm, stop pump)
            self.stop()
        except Exception as e:
            logger.error(
                f"ActEngine: start failed - {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    def stop(self) -> None:
        logger.info("ActEngine: stop requested")
        self._running = False
        self._state.act_running = False
        for actuator in self._actuators.values():
            try:
                actuator.cleanup()
            except Exception as e:
                logger.warning(
                    f"ActEngine: cleanup error on {actuator.name} - "
                    f"{type(e).__name__}: {e}"
                )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while self._running and self._state.system_running:
            try:
                self._tick()
            except Exception as e:
                logger.error(
                    f"ActEngine: tick error - {type(e).__name__}: {e}",
                    exc_info=True,
                )
            time.sleep(self._cycle_s)

    def _tick(self) -> None:
        # we should dispatch manual commands only if mode is anything but autopilot
        self._dispatch_manual_commands()
        sensor_active = self._state.sensor_triggered
        danger        = self._state.danger_level
        actions       = list(self._state.recommended_actions)
        mode          = self._state.system_mode
        pred_id       = self._state.prediction_id

        # 1. Reconcile pump state every tick (continuous-action discipline).
        self._reconcile_pump(sensor_active, danger, actions, mode)

        # 2. On new prediction, do one-shot work (notifications, copilot req).
        if pred_id != self._last_handled_pred_id and danger >= self._danger_threshold:
            self._handle_new_prediction(pred_id, danger, actions, mode)

        # 3. While in copilot wait, check for decision or timeout.
        if self._copilot_waiting:
            self._check_copilot_resolution()

    # ------------------------------------------------------------------
    # Reconciliation (continuous actions)
    # ------------------------------------------------------------------

    def _reconcile_pump(self, sensor_active, danger, actions, mode) -> None:
        pump = self._actuators.get("pump")
        if pump is None:
            return

        suppress_intent = (
            sensor_active
            and danger >= self._danger_threshold
            and "suppress" in actions
        )

        if mode == SystemMode.AUTOPILOT:
            should_run = suppress_intent
        elif mode == SystemMode.COPILOT:
            # Pump runs only after explicit approval (set in _check_copilot_resolution)
            should_run = suppress_intent and getattr(self, "_copilot_approved_pump", False)
        else:
            # surveillance / training — never auto-run pump
            should_run = False

        if should_run and not pump.is_active:
            try:
                pump.activate()
            except ActuatorFaultError as e:
                logger.error(f"ActEngine: pump activation failed - {e}")
        elif not should_run and pump.is_active:
            pump.deactivate()
            if mode == SystemMode.COPILOT:
                self._copilot_approved_pump = False

    # ------------------------------------------------------------------
    # New prediction dispatch (one-shot)
    # ------------------------------------------------------------------

    def _handle_new_prediction(self, pred_id, danger, actions, mode) -> None:
        logger.info(
            f"ActEngine: new_prediction | id={pred_id} | "
            f"danger={danger} | actions={actions} | mode={mode.value}"
        )

        if mode == SystemMode.AUTOPILOT:
            self._notify(danger, actions, "auto-executing")
            self._last_handled_pred_id = pred_id

        elif mode == SystemMode.COPILOT:
            if self._copilot_waiting:
                # Already waiting — refresh the proposal with newest actions
                self._copilot_pending_actions = actions
                self._copilot_pending_pred_id = pred_id
                logger.info("ActEngine: copilot proposal refreshed mid-wait")
            else:
                self._copilot_waiting = True
                self._copilot_waiting_since = time.monotonic()
                self._copilot_pending_actions = actions
                self._copilot_pending_pred_id = pred_id
                self._state.copilot_decision = None  # clear any stale
                self._notify(danger, actions, "awaiting copilot approval")
            self._last_handled_pred_id = pred_id

        elif mode in (SystemMode.SURVEILLANCE, SystemMode.TRAINING):
            self._notify(danger, actions, f"{mode.value} — manual action required")
            self._last_handled_pred_id = pred_id

    # ------------------------------------------------------------------
    # Copilot resolution
    # ------------------------------------------------------------------

    def _check_copilot_resolution(self) -> None:
        decision = self._state.copilot_decision
        elapsed = time.monotonic() - self._copilot_waiting_since

        if decision == "approved":
            logger.info(
                f"ActEngine: copilot APPROVED | "
                f"pred_id={self._copilot_pending_pred_id} | "
                f"actions={self._copilot_pending_actions}"
            )
            if "suppress" in self._copilot_pending_actions:
                self._copilot_approved_pump = True
            self._reset_copilot_wait()

        elif decision == "rejected":
            logger.info(
                f"ActEngine: copilot REJECTED | "
                f"pred_id={self._copilot_pending_pred_id}"
            )
            self._copilot_approved_pump = False
            self._reset_copilot_wait()

        elif elapsed > self._copilot_timeout_s:
            logger.warning(
                f"ActEngine: copilot TIMEOUT after {self._copilot_timeout_s}s | "
                f"pred_id={self._copilot_pending_pred_id}"
            )
            self._copilot_approved_pump = False
            if self._notifier is not None:
                from notify import EventType
                self._notifier.notify(
                    EventType.COPILOT_TIMEOUT,
                    payload={
                        "pred_id": self._copilot_pending_pred_id,
                        "elapsed_s": elapsed,
                    },
                    source_layer="act",
                )
            self._reset_copilot_wait()

    def _reset_copilot_wait(self) -> None:
        self._copilot_waiting = False
        self._copilot_waiting_since = 0.0
        self._copilot_pending_actions = []
        self._copilot_pending_pred_id = 0
        self._state.copilot_decision = None

    # ------------------------------------------------------------------
    # Notification (placeholder — NotificationService is still a stub)
    # ------------------------------------------------------------------

    def _notify(self, danger: int, actions: list, context: str) -> None:
        """
        Fire a notification for a fire-related event. The exact EventType
        depends on the context string set by _handle_new_prediction.
        """
        from notify import EventType

        # Map context → event type
        if "auto-executing" in context:
            event_type = EventType.PREDICTION_AUTO_EXECUTING
        elif "copilot approval" in context:
            event_type = EventType.COPILOT_APPROVAL_REQUESTED
        elif "manual action required" in context:
            event_type = EventType.SURVEILLANCE_MANUAL_NEEDED
        else:
            event_type = EventType.PREDICTION_AUTO_EXECUTING  # safe default

        logger.info(
            f"ActEngine: NOTIFY | danger={danger} | "
            f"actions={actions} | context={context}"
        )

        if self._notifier is not None:
            self._notifier.notify(
                event_type,
                payload={
                    "danger_level": danger,
                    "actions": actions,
                    "context": context,
                    "mode": self._state.system_mode.value,
                },
                source_layer="act",
            )
    # ------------------------------------------------------------------
    # Manual command dispatch (website → ACT via SystemState queue)
    # ------------------------------------------------------------------

    def _dispatch_manual_commands(self) -> None:
        """
        Drain the manual_commands queue written by the Flask API.
        Each command: {"action": str, "params": dict}

        Actions:
          - "pump_fire"  → activate pump
          - "pump_stop"  → deactivate pump
          - "arm_nudge"  → nudge arm in params["direction"]
        """
        import time as _time
        try:
            while not self._state.manual_commands.empty():
                cmd = self._state.manual_commands.get_nowait()
                action = cmd.get("action")
                params = cmd.get("params", {})
                logger.info(f"ActEngine: manual cmd | action={action} | params={params}")

                if action == "pump_fire":
                    pump = self._actuators.get("pump")
                    if pump:
                        try:
                            pump.activate()
                        except Exception as e:
                            logger.error(f"manual pump_fire failed: {e}")

                elif action == "pump_stop":
                    pump = self._actuators.get("pump")
                    if pump:
                        try:
                            pump.deactivate()
                        except Exception as e:
                            logger.error(f"manual pump_stop failed: {e}")

                elif action == "arm_nudge":
                    arm = self._actuators.get("arm")
                    direction = params.get("direction")
                    if arm and direction:
                        self._state.arm_manual_mode_until = _time.time() + 3.0
                        try:
                            arm.nudge(direction)
                        except Exception as e:
                            logger.error(f"arm_nudge failed: {e}")
                else:
                    logger.warning(f"ActEngine: unknown manual command '{action}'")

        except Exception as e:
            logger.error(f"_dispatch_manual_commands error: {e}")