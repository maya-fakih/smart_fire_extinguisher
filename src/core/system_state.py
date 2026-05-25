from multiprocessing import Manager
from typing import Optional
from core.enums import SystemMode


class SystemState:
    """
    Shared blackboard for all four layers.
    Backed by a multiprocessing.Manager dict so writes are visible across OS processes.
    Each field has exactly one writer (enforced by convention, not by code).
    Queues carry SensorSnapshot and VisionSnapshot to the THINK layer.
    Fast-path values (latest_heat_matrix, latest_fire_x/y) bypass queues so
    ACT can read fresh values for closed-loop tracking.
    """

    _VALID_COPILOT_DECISIONS = {"approved", "rejected"}

    def __init__(self, manager, system_mode: str):
        self._data = manager.dict()

        # Queues — only two exist by design
        self.sense_queue = manager.Queue()
        self.see_queue   = manager.Queue()

        # control flags
        self.sensor_triggered    = False
        self.active_sensor_count = 0
        self.faulted_sensors     = []
        self.system_running      = False
        self.sense_running       = False
        self.see_running         = False
        self.think_running       = False
        self.act_running         = False
        self.system_mode         = system_mode
        self.db_connected        = False
        self.camera_feed_active  = False

        # THINK output
        self.danger_level         = 0
        self.recommended_actions  = []
        self.prediction_id        = 0

        # Fast-path tracking inputs (SEE → ACT, SENSE → ACT)
        self.latest_heat_matrix = None
        self.latest_fire_x      = None
        self.latest_fire_y      = None

        # ACT ↔ website
        self.copilot_decision = None
        # Website → ACT: manual command queue (pump_fire, arm_nudge)
        # Each entry: {"action": str, "params": dict}
        self.manual_commands = manager.Queue()

        # Website → THINK: training labels (training mode only)
        # Each entry: {"row_id": int, "true_danger_level": int,
        #              "true_action": str|None}
        # THINK drains this each loop iteration and writes the label to the
        # matching DB row (the row inserted at capture time).
        self.training_label_queue = manager.Queue()

        # Website ↔ THINK: training capture request/response (training mode only)
        # Capture must run inside the THINK process (it owns _align + the DB).
        # The API route pushes a request and blocks on the response queue.
        #   request entry : {"request_id": str, "same_event": bool}
        #   response entry: {"request_id": str, "ok": bool, "result": dict|None,
        #                    "error": str|None}
        self.training_capture_request  = manager.Queue()
        self.training_capture_response = manager.Queue()

        # Website ↔ THINK: model-training request/response (any mode)
        # Training is offline glue — assemble validated rows, fit, save weights.
        # Runs in the THINK process (owns the DB + model). Serviced every loop
        # iteration regardless of mode, unlike capture which is training-only.
        #   request entry : {"request_id": str}
        #   response entry: {"request_id": str, "ok": bool, "result": dict|None,
        #                    "error": str|None}
        self.train_request  = manager.Queue()
        self.train_response = manager.Queue()

        # ── Training-mode recording (continuous label-while-streaming) ────────
        # When training_recording = True, THINK consumes sense_queue/see_queue
        # in training mode (just like the live pipeline) and writes rows to
        # DB tagged with the current label from training_label_stream.
        #
        # training_label_stream entries:
        #   {"true_danger_level": int, "true_action": str|None,
        #    "valid_until": float (unix epoch) | None}
        # The head label applies to every incoming row whose timestamp < valid_until.
        # When a row's timestamp >= valid_until, the head label is popped and
        # the next one becomes current. A label with valid_until=None applies
        # to all subsequent rows until a new label is pushed.
        self.training_recording    = False
        self.training_label_stream = manager.Queue()
        # training_event_id: explicit event_id for the current recording.
        # Set when the human starts a recording (new event or continue last);
        # used directly so we don't ping the DB on every row.
        self.training_event_id = 0

        # Website → SENSE: per-sensor soft enable/disable
        self.sensor_overrides = manager.dict()

        # Website → ACT: arm manual override expiry timestamp
        # While time.time() < this value, tracking loop pauses.
        self.arm_manual_mode_until = 0.0


    # --- bool ---
    @property
    def db_connected(self) -> bool: return self._data['db_connected']
    @db_connected.setter
    def db_connected(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"db_connected must be bool, got {type(value)}")
        self._data['db_connected'] = value

    @property
    def camera_feed_active(self) -> bool: return self._data['camera_feed_active']
    @camera_feed_active.setter
    def camera_feed_active(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"camera_feed_active must be bool, got {type(value)}")
        self._data['camera_feed_active'] = value

    @property
    def system_running(self) -> bool: return self._data['system_running']
    @system_running.setter
    def system_running(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"system_running must be bool, got {type(value)}")
        self._data['system_running'] = value

    @property
    def sensor_triggered(self) -> bool: return self._data['sensor_triggered']
    @sensor_triggered.setter
    def sensor_triggered(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"sensor_triggered must be bool, got {type(value)}")
        self._data['sensor_triggered'] = value

    @property
    def sense_running(self) -> bool: return self._data['sense_running']
    @sense_running.setter
    def sense_running(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"sense_running must be bool, got {type(value)}")
        self._data['sense_running'] = value

    @property
    def see_running(self) -> bool: return self._data['see_running']
    @see_running.setter
    def see_running(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"see_running must be bool, got {type(value)}")
        self._data['see_running'] = value

    @property
    def think_running(self) -> bool: return self._data['think_running']
    @think_running.setter
    def think_running(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"think_running must be bool, got {type(value)}")
        self._data['think_running'] = value

    @property
    def act_running(self) -> bool: return self._data['act_running']
    @act_running.setter
    def act_running(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"act_running must be bool, got {type(value)}")
        self._data['act_running'] = value

    # --- int ---
    @property
    def active_sensor_count(self) -> int: return self._data['active_sensor_count']
    @active_sensor_count.setter
    def active_sensor_count(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"active_sensor_count must be int, got {type(value)}")
        if value < 0:
            raise ValueError(f"active_sensor_count cannot be negative, got {value}")
        self._data['active_sensor_count'] = value

    @property
    def danger_level(self) -> int: return self._data['danger_level']
    @danger_level.setter
    def danger_level(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"danger_level must be int, got {type(value)}")
        if value not in range(0, 6):
            raise ValueError(f"danger_level must be 0-5, got {value}")
        self._data['danger_level'] = value

    @property
    def prediction_id(self) -> int: return self._data['prediction_id']
    @prediction_id.setter
    def prediction_id(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f"prediction_id must be int, got {type(value)}")
        if value < 0:
            raise ValueError(f"prediction_id cannot be negative, got {value}")
        self._data['prediction_id'] = value

    # --- enum ---
    @property
    def system_mode(self) -> SystemMode: return self._data['system_mode']
    @system_mode.setter
    def system_mode(self, value: str):
        try:
            self._data['system_mode'] = SystemMode(value)
        except ValueError:
            raise ValueError(
                f"Invalid system_mode '{value}'. Must be one of "
                f"{[m.value for m in SystemMode]}"
            )

    # --- list of dicts ---
    @property
    def faulted_sensors(self) -> list: return self._data['faulted_sensors']
    @faulted_sensors.setter
    def faulted_sensors(self, value: list):
        if not isinstance(value, list):
            raise TypeError(f"faulted_sensors must be a list, got {type(value)}")
        for entry in value:
            if not isinstance(entry, dict) or 'name' not in entry or 'faulted_at' not in entry:
                raise ValueError(f"Each faulted sensor must have 'name' and 'faulted_at', got {entry}")
        self._data['faulted_sensors'] = value

    # --- list of strings (recommended_actions) ---
    @property
    def recommended_actions(self) -> list: return self._data['recommended_actions']
    @recommended_actions.setter
    def recommended_actions(self, value: list):
        if not isinstance(value, list):
            raise TypeError(f"recommended_actions must be a list, got {type(value)}")
        for entry in value:
            if not isinstance(entry, str):
                raise TypeError(f"Each action must be str, got {type(entry)}")
        self._data['recommended_actions'] = value

    # --- 2D float matrix (heat) ---
    @property
    def latest_heat_matrix(self) -> Optional[list]: return self._data['latest_heat_matrix']
    @latest_heat_matrix.setter
    def latest_heat_matrix(self, value):
        if value is None:
            self._data['latest_heat_matrix'] = None
            return
        if not isinstance(value, list) or not all(isinstance(row, list) for row in value):
            raise TypeError("latest_heat_matrix must be None or list[list[float]]")
        self._data['latest_heat_matrix'] = value

    # --- normalized fire coords ---
    @property
    def latest_fire_x(self) -> Optional[float]: return self._data['latest_fire_x']
    @latest_fire_x.setter
    def latest_fire_x(self, value):
        if value is None:
            self._data['latest_fire_x'] = None
            return
        if not isinstance(value, (int, float)):
            raise TypeError(f"latest_fire_x must be None or float, got {type(value)}")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"latest_fire_x must be in [0, 1], got {value}")
        self._data['latest_fire_x'] = float(value)

    @property
    def latest_fire_y(self) -> Optional[float]: return self._data['latest_fire_y']
    @latest_fire_y.setter
    def latest_fire_y(self, value):
        if value is None:
            self._data['latest_fire_y'] = None
            return
        if not isinstance(value, (int, float)):
            raise TypeError(f"latest_fire_y must be None or float, got {type(value)}")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"latest_fire_y must be in [0, 1], got {value}")
        self._data['latest_fire_y'] = float(value)

    # --- copilot decision (Optional[str]) ---
    @property
    def copilot_decision(self) -> Optional[str]: return self._data['copilot_decision']
    @copilot_decision.setter
    def copilot_decision(self, value):
        if value is None:
            self._data['copilot_decision'] = None
            return
        if value not in self._VALID_COPILOT_DECISIONS:
            raise ValueError(
                f"copilot_decision must be None or one of "
                f"{self._VALID_COPILOT_DECISIONS}, got {value!r}"
            )
        self._data['copilot_decision'] = value
    
    # --- arm manual override expiry (float timestamp) ---
    @property
    def arm_manual_mode_until(self) -> float:
        return self._data.get('arm_manual_mode_until', 0.0)
    @arm_manual_mode_until.setter
    def arm_manual_mode_until(self, value: float):
        self._data['arm_manual_mode_until'] = float(value)