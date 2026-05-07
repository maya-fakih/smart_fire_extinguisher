import json
import time
import multiprocessing
from multiprocessing import Manager

from core import SystemState
from exceptions import ConfigError, StateInitError, ModeError

from sense import SensorFuser
from see import VisionFuser
from think import ThinkEngine
from act import ActEngine
from notify import NotificationService


class SystemOrchestrator:
    """
    Entry point for the entire system.
    - Loads config.json
    - Creates SystemState (shared across all processes)
    - Spawns each layer as an independent OS process
    - Provides API for dashboard (set_mode, set_camera_feed, get_state_summary, etc.)
    """

    def __init__(self, config_path: str):
        self._config_path = config_path
        self._config = self._load_config(config_path)

        self._manager = None
        self._state = None

        self._sensor_fuser = None
        self._vision_fuser = None
        self._think_engine = None
        self._act_engine = None
        self._notifier = None

        self._sense_process = None
        self._see_process = None
        self._think_process = None
        self._act_process = None

        self._init_manager()
        self._init_state()
        self._init_layers()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ConfigError(f"Config file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Config file is not valid JSON: {e}")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_manager(self) -> None:
        try:
            self._manager = Manager()
        except Exception as e:
            raise StateInitError(f"Failed to start multiprocessing manager: {e}")

    def _init_state(self) -> None:
        try:
            system_mode = self._config.get("system", {}).get("system_mode", "surveillance")
            self._state = SystemState(self._manager, system_mode)
        except Exception as e:
            raise StateInitError(f"Failed to initialize SystemState: {e}")

    def _init_layers(self) -> None:
        self._sensor_fuser = SensorFuser(self._config, self._state)
        self._vision_fuser = VisionFuser(self._config, self._state)
        self._think_engine = ThinkEngine(self._config, self._state)
        self._act_engine = ActEngine(self._config, self._state)
        self._notifier = NotificationService(self._config)

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._state.system_running = True

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
        self._see_process.start()
        self._think_process.start()
        self._act_process.start()

    def stop(self) -> None:
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
                process.terminate()
                process.join(timeout=2)

        if self._manager:
            self._manager.shutdown()

    def restart_all(self) -> None:
        self.stop()
        self._init_layers()
        self.start()

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        try:
            self._state.system_mode = mode
        except ValueError as e:
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

        # Restart so layers pick up the new values
        self.restart_all()

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
            "recommended_action": self._state.recommended_action,
            "camera_feed_active": self._state.camera_feed_active,
        }