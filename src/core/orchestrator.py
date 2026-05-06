import json
from multiprocessing import Manager
from core.system_state import SystemState
from exceptions import ConfigError, StateInitError, ModeError

# layers — uncomment as we build them
# from sense.sensor_fuser import SensorFuser
# from see.vision_fuser import VisionFuser
# from think.think_engine import ThinkEngine
# from act.act_engine import ActEngine
# from notify.notification_service import NotificationService


class SystemOrchestrator:
    def __init__(self, config_path: str):
        self._config       = self._load_config(config_path)
        self._manager      = None
        self._state        = None
        self._sensor_fuser = None
        self._vision_fuser = None
        self._think_engine = None
        self._act_engine   = None
        self._notifier     = None

        self._init_manager()
        self._init_state()

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ConfigError(f"Config file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ConfigError(f"Config file is not valid JSON: {e}")

    def _init_manager(self) -> None:
        try:
            self._manager = Manager()
        except Exception as e:
            raise StateInitError(f"Failed to start multiprocessing manager: {e}")

    def _init_state(self) -> None:
        try:
            system_mode  = self._config.get("system", {}).get("system_mode", "surveillance")
            self._state  = SystemState(self._manager, system_mode)
        except Exception as e:
            raise StateInitError(f"Failed to initialize SystemState: {e}")

    def _init_layers(self) -> None:
        # uncomment as we build each layer
        # self._sensor_fuser = SensorFuser(self._config, self._state)
        # self._vision_fuser = VisionFuser(self._config, self._state)
        # self._think_engine = ThinkEngine(self._config, self._state)
        # self._act_engine   = ActEngine(self._config, self._state)
        # self._notifier     = NotificationService(self._config)
        pass

    def restart_layer(self, layer_name: str) -> None:
        """Stop and restart a single layer process by name.
        Valid names: 'sense', 'see', 'think', 'act'
        """
        # TODO: implement per-layer restart logic
        # we will implement it once we have the layers built out :)
        pass

    def start(self) -> None:
        self._init_layers()
        # self._sensor_fuser.start()
        # self._vision_fuser.start()
        # self._think_engine.start()
        # self._act_engine.start()

    def stop(self) -> None:
        try:
            # self._sensor_fuser.stop()
            # self._vision_fuser.stop()
            # self._think_engine.stop()
            # self._act_engine.stop()
            pass
        finally:
            if self._manager:
                self._manager.shutdown()

    def set_mode(self, mode: str) -> None:
        try:
            self._state.system_mode = mode
        except ValueError as e:
            raise ModeError(f"Invalid mode: {e}")