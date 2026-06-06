import json
import time
import logging
import multiprocessing
from multiprocessing import Manager

from core import SystemState
from exceptions import ConfigError, StateInitError, ModeError

from sense import SensorFuser
from see import VisionFuser
from think import ThinkEngine
from act import ActEngine
from notify import NotificationService, EventType

logger = logging.getLogger(__name__)


class SystemOrchestrator:
    """
    Entry point for the entire system.
    - Loads config.json
    - Creates SystemState (shared across all processes)
    - Constructs the shared NotificationService and passes it to every layer
    - Spawns each layer as an independent OS process
    - Provides API for dashboard (set_mode, set_camera_feed, get_state_summary, etc.)
    """

    def __init__(self, config_path: str):
        self._config_path = config_path
        self._config = self._load_config(config_path)

        self._manager = None
        self._state = None

        # Shared notifier — constructed once, passed to every layer.
        self._notifier = None

        self._sensor_fuser = None
        self._vision_fuser = None
        self._think_engine = None
        self._act_engine = None

        self._sense_process = None
        self._see_process = None
        self._think_process = None
        self._act_process = None

        self._init_manager()
        self._init_state()
        self._init_notifier()
        self._init_layers()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> dict:
        try:
            logger.debug(f"Orchestrator: loading config | path={config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Orchestrator: config loaded successfully")
            return config
        except FileNotFoundError as e:
            logger.error(f"Orchestrator: config file not found - {config_path}", exc_info=True)
            raise ConfigError(f"Config file not found: {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Orchestrator: invalid JSON in config - {e}", exc_info=True)
            raise ConfigError(f"Config file is not valid JSON: {e}")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_manager(self) -> None:
        try:
            logger.debug("Orchestrator: initializing multiprocessing manager")
            self._manager = Manager()
        except Exception as e:
            logger.error(
                f"Orchestrator: failed to start manager - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise StateInitError(f"Failed to start multiprocessing manager: {e}")

    def _init_state(self) -> None:
        try:
            system_mode = self._config.get("system", {}).get("system_mode", "surveillance")
            logger.debug(f"Orchestrator: initializing SystemState | mode={system_mode}")
            self._state = SystemState(self._manager, system_mode)
        except Exception as e:
            logger.error(
                f"Orchestrator: failed to initialize SystemState - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise StateInitError(f"Failed to initialize SystemState: {e}")

    def _init_notifier(self) -> None:
        """
        Construct the single NotificationService that all layers share.
        Does not open the DB connection — that happens lazily on first notify().
        """
        self._notifier = NotificationService(self._config)
        logger.info("Orchestrator: NotificationService constructed")

    def _init_layers(self) -> None:
        # Each layer accepts the notifier so it can fire notifications
        # directly at the source of any fault.
        self._sensor_fuser = SensorFuser(self._config, self._state, self._notifier)
        self._vision_fuser = VisionFuser(self._config, self._state, self._notifier)
        self._think_engine = ThinkEngine(self._config, self._state, self._notifier)
        self._act_engine   = ActEngine(self._config, self._state, self._notifier)

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("Orchestrator: starting system | spawning all processes")
        self._state.system_running = True
        self._notifier.notify(
            EventType.SYSTEM_STARTED,
            payload={"mode": self._state.system_mode.value},
            source_layer="orchestrator",
        )

        self._sense_process = multiprocessing.Process(
            target=self._sensor_fuser.start,
            name="SenseProcess"
        )
        self._see_process = multiprocessing.Process(
            target=self._vision_fuser.start,
            name="SeeProcess"
        )
        self._think_process = multiprocessing.Process(
            target=self._think_engine.start,
            name="ThinkProcess"
        )
        self._act_process = multiprocessing.Process(
            target=self._act_engine.start,
            name="ActProcess"
        )

        self._sense_process.start()
        logger.info("Orchestrator: SenseProcess started")
        self._see_process.start()
        logger.info("Orchestrator: SeeProcess started")
        self._think_process.start()
        logger.info("Orchestrator: ThinkProcess started")
        self._act_process.start()
        logger.info("Orchestrator: ActProcess started")

    def stop(self) -> None:
        logger.info("Orchestrator: stopping system | signaling all processes")
        self._state.system_running = False
        time.sleep(0.5)

        all_processes = [
            self._sense_process,
            self._see_process,
            self._think_process,
            self._act_process,
        ]

        for process in all_processes:
            if process and process.is_alive():
                logger.debug(f"Orchestrator: terminating {process.name}")
                process.terminate()
                process.join(timeout=2)

        self._notifier.notify(
            EventType.SYSTEM_STOPPED,
            source_layer="orchestrator",
        )
        logger.info("Orchestrator: all processes stopped")

    def shutdown(self) -> None:
        """Full teardown — call once on system exit, not on restarts."""
        logger.info("Orchestrator: full shutdown initiated")
        self.stop()
        if self._notifier:
            self._notifier.close()
        if self._manager:
            self._manager.shutdown()
            logger.info("Orchestrator: multiprocessing manager shutdown")

    def restart_all(self) -> None:
        logger.info("Orchestrator: restarting all layers")
        self.stop()
        if self._sensor_fuser:
            self._sensor_fuser.cleanup()
            logger.debug("Orchestrator: sensor cleanup completed")
        self._init_layers()
        logger.debug("Orchestrator: layers reinitialized")
        self.start()

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        logger.debug(f"Orchestrator: set_mode requested | mode={mode}")
        try:
            old_mode = self._state.system_mode.value
            self._state.system_mode = mode
            logger.info(f"Orchestrator: mode changed | mode={mode}")
            self._notifier.notify(
                EventType.MODE_CHANGED,
                payload={"from": old_mode, "to": mode},
                source_layer="orchestrator",
            )
        except ValueError as e:
            logger.error(f"Orchestrator: invalid mode - {type(e).__name__}: {e}", exc_info=True)
            raise ModeError(f"Invalid mode: {e}")

    def set_camera_feed(self, active: bool) -> None:
        self._state.camera_feed_active = active

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        return self._config

    def get_config_section(self, section: str) -> dict:
        return self._config.get(section, {})

    def update_config(self, changes: dict) -> None:
        """
        Apply user changes to the existing config.
        'changes' is a flat dict with dot-separated paths as keys.
        Example: {"sensors.smoke.threshold_physical": 500, "system.system_mode": "autopilot"}
        """
        # Apply each change to the current config
        for path, new_value in changes.items():
            keys = path.split(".")
            target = self._config

            # Walk down to the nested key
            for key in keys[:-1]:
                if key not in target:
                    raise ConfigError(f"Invalid config path: {path}")
                target = target[key]

            # Validate the final key exists
            if keys[-1] not in target:
                raise ConfigError(f"Invalid config key: {path}")

            # Update the value
            target[keys[-1]] = new_value

        # Write the updated config to disk
        with open(self._config_path, 'w') as f:
            json.dump(self._config, f, indent=2)

        self._notifier.notify(
            EventType.CONFIG_UPDATED,
            payload={"changes": list(changes.keys())},
            source_layer="orchestrator",
        )

        # Restart so layers pick up the new values
        self.restart_all()

    # ------------------------------------------------------------------
    # Manual hardware control (website → hardware via SystemState queue)
    # ------------------------------------------------------------------

    def manual_pump_fire(self) -> None:
        self._state.manual_commands.put({"action": "pump_fire", "params": {}})
        logger.info("Orchestrator: manual pump_fire queued")

    def manual_pump_stop(self) -> None:
        self._state.manual_commands.put({"action": "pump_stop", "params": {}})
        logger.info("Orchestrator: manual pump_stop queued")

    def manual_arm_nudge(self, direction: str) -> None:
        valid = {"pan_left", "pan_right", "tilt_up", "tilt_down"}
        if direction not in valid:
            raise ValueError(f"Invalid direction '{direction}'. Must be one of {valid}")
        self._state.manual_commands.put({"action": "arm_nudge", "params": {"direction": direction}})
        logger.info(f"Orchestrator: arm_nudge queued | direction={direction}")

    def toggle_sensor(self, sensor_name: str, enabled: bool) -> None:
        overrides = dict(self._state.sensor_overrides)
        overrides[sensor_name] = enabled
        self._state.sensor_overrides.update(overrides)
        logger.info(f"Orchestrator: sensor override | {sensor_name}={enabled}")

    def set_copilot_decision(self, decision: str) -> None:
        if decision not in ("approved", "rejected"):
            raise ValueError(f"decision must be 'approved' or 'rejected', got {decision!r}")
        self._state.copilot_decision = decision
        logger.info(f"Orchestrator: copilot_decision set | {decision}")

    # ------------------------------------------------------------------
    # Training mode (website → THINK via SystemState queues)
    # ------------------------------------------------------------------

    def get_system_mode(self) -> str:
        """Current system mode as a plain string."""
        return self._state.system_mode.value

    def is_recording(self) -> bool:
        """Is a training-recording session active right now."""
        return bool(self._state.training_recording)

    def get_recording_event_id(self) -> int:
        """event_id of the current/last recording (0 if never started)."""
        return self._state.training_event_id

    def training_capture(self, same_event: bool, target_ts: float = None,
                         timeout_s: float = 5.0) -> dict:
        """
        Request a training capture from the THINK process and wait for the result.

        target_ts (unix epoch float) anchors the align scan to a moment in
        time — typically "latest SEE timestamp at the moment of click", read
        from SystemState by the API route. When omitted, THINK falls back to
        a live-style align (popping the queues) which is used by tests only.

        Returns the response dict {"ok": bool, "result": dict|None, "error": str|None}.
        Raises TimeoutError if THINK does not answer within timeout_s.
        """
        import uuid, time

        request_id = str(uuid.uuid4())
        self._state.training_capture_request.put({
            "request_id": request_id,
            "same_event": same_event,
            "target_ts":  target_ts,
        })
        logger.debug(f"Orchestrator: training capture queued | request_id={request_id}")

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                resp = self._state.training_capture_response.get_nowait()
            except Exception:
                time.sleep(0.02)
                continue

            if resp.get("request_id") != request_id:
                # Not ours — another concurrent capture. Put it back.
                self._state.training_capture_response.put(resp)
                time.sleep(0.02)
                continue

            return resp

        raise TimeoutError(
            f"training capture timed out after {timeout_s}s "
            f"(request_id={request_id})"
        )

    def training_save_label(self, row_id: int, true_danger_level: int,
                            true_action: str = None) -> None:
        """
        Queue a human training label for THINK to write to the DB.

        Fire-and-forget: returns immediately so the website is not blocked on
        the DB write. THINK drains training_label_queue on its own loop.
        """
        self._state.training_label_queue.put({
            "row_id": row_id,
            "true_danger_level": true_danger_level,
            "true_action": true_action,
        })
        logger.info(f"Orchestrator: training label queued | row_id={row_id}")

    def training_recording_start(self, same_event: bool,
                                 timeout_s: float = 5.0) -> dict:
        """
        Begin a recording session.

        Decides event_id by asking THINK (it owns the live DB connection) via
        the existing training_capture queue mechanism — we reuse it as a small
        round-trip without doing a capture. Then flips recording on.
        """
        if self._state.training_recording:
            raise RuntimeError("recording is already active — stop it first")

        # Ask THINK to compute the next event_id. We piggyback on the capture
        # queue with a special flag so THINK does NOT actually do a capture —
        # see ThinkEngine._service_capture_requests.
        import uuid, time
        request_id = str(uuid.uuid4())
        self._state.training_capture_request.put({
            "request_id":   request_id,
            "same_event":   same_event,
            "event_id_only": True,        # do not capture, only return event_id
        })
        deadline = time.time() + timeout_s
        resp = None
        while time.time() < deadline:
            try:
                r = self._state.training_capture_response.get_nowait()
            except Exception:
                time.sleep(0.02); continue
            if r.get("request_id") != request_id:
                self._state.training_capture_response.put(r)
                time.sleep(0.02); continue
            resp = r; break
        if resp is None:
            raise TimeoutError("recording_start: THINK did not answer")
        if not resp.get("ok"):
            raise RuntimeError(f"recording_start: {resp.get('error')}")

        self._state.training_event_id = resp["result"]["event_id"]
        self._state.training_recording = True
        logger.info(
            f"Orchestrator: recording started | event_id={self._state.training_event_id} "
            f"| same_event={same_event}"
        )
        return {
            "ok": True,
            "event_id": self._state.training_event_id,
            "same_event": same_event,
            "recording": True,
        }

    def training_recording_stop(self) -> dict:
        """
        Stop the current recording session. THINK stops consuming the queues
        in training mode (back to capture-only). The label-stream queue is
        drained so a stale tail label doesn't leak into the next recording.
        """
        if not self._state.training_recording:
            return {"ok": True, "recording": False, "note": "was not recording"}

        self._state.training_recording = False
        # Drain any unused labels so the next recording starts clean.
        drained = 0
        while not self._state.training_label_stream.empty():
            try:
                self._state.training_label_stream.get_nowait()
                drained += 1
            except Exception:
                break
        logger.info(
            f"Orchestrator: recording stopped | event_id={self._state.training_event_id} "
            f"| drained_unused_labels={drained}"
        )
        return {
            "ok": True,
            "recording": False,
            "event_id": self._state.training_event_id,
            "drained_unused_labels": drained,
        }

    def training_recording_push_label(self, true_danger_level: int,
                                      true_action: str = None,
                                      valid_until: float = None) -> None:
        """
        Push a label onto the recording's label stream. Applies to every row
        whose timestamp < valid_until. With valid_until=None the label applies
        until a new label arrives (open-ended head).
        """
        self._state.training_label_stream.put({
            "true_danger_level": true_danger_level,
            "true_action":       true_action,
            "valid_until":       valid_until,
        })
        logger.info(
            f"Orchestrator: recording label pushed | danger={true_danger_level} "
            f"| valid_until={valid_until}"
        )

    def train_model(self, timeout_s: float = 120.0) -> dict:
        """
        Request model training from the THINK process and wait for the result.

        Training runs inside THINK (it owns the DB + model). This pushes a
        request onto train_request and blocks on train_response until THINK
        answers. Training can take seconds — timeout is generous (default 120s).

        Returns the response dict {"ok": bool, "result": dict|None, "error": str|None}.
        Raises TimeoutError if THINK does not answer in time.
        """
        import uuid, time

        request_id = str(uuid.uuid4())
        self._state.train_request.put({"request_id": request_id})
        logger.info(f"Orchestrator: train request queued | request_id={request_id}")

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                resp = self._state.train_response.get_nowait()
            except Exception:
                time.sleep(0.05)
                continue

            if resp.get("request_id") != request_id:
                self._state.train_response.put(resp)
                time.sleep(0.05)
                continue

            return resp

        raise TimeoutError(
            f"model training timed out after {timeout_s}s (request_id={request_id})"
        )
    
    
    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_state_summary(self) -> dict:
        return {
            "system_mode": self._state.system_mode.value,
            "system_running": self._state.system_running,
            "sense_running": self._state.sense_running,
            "see_running": self._state.see_running,
            "think_running": self._state.think_running,
            "act_running": self._state.act_running,
            "db_connected": self._state.db_connected,
            "active_sensor_count": self._state.active_sensor_count,
            "faulted_sensors": self._state.faulted_sensors,
            "danger_level": self._state.danger_level,
            "recommended_actions": self._state.recommended_actions,
            "camera_feed_active": self._state.camera_feed_active,
            "prediction_id": self._state.prediction_id,
            "copilot_decision": self._state.copilot_decision,
            "latest_heat_matrix": self._state.latest_heat_matrix,
        }