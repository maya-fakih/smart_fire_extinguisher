# src/think/think_engine.py

import time
import json
from core.system_state import SystemState
from think.database.think_database import ThinkDatabase
from think.ml.base_model import BaseModel
from think.ml.xgboost_model import XGBoostModel
from think.snapshot import ThinkSnapshot
from exceptions import DatabaseError, AlignmentError, ModelError


class ThinkEngine:
    def __init__(self, config: dict, state: SystemState):
        self._config = config
        self._state = state
        self._model: BaseModel = None

        think_cfg = self._config.get("think", {})
        self._max_gap_ms = think_cfg.get("max_gap_ms", 500)
        self._model_path = think_cfg.get("model_weights_path", "model_weights/")
        self._active_model = think_cfg.get("active_model", "xgboost")

        self._sense_enabled = self._config.get("sense", {}).get("enabled", True)
        self._see_enabled = self._config.get("see", {}).get("enabled", True)

        self._running = False
        self._db = ThinkDatabase(self._max_gap_ms)

    # --- lifecycle ---

    def start(self):
        try:
            self._db.connect()
            self._state.db_connected = True
        except DatabaseError:
            self._state.db_connected = False
            raise

        self._load_model()
        self._running = True
        self._state.think_running = True
        self._run_loop()

    def stop(self):
        self._running = False
        self._state.think_running = False
        self._db.close()

    # --- main loop ---

    def _run_loop(self):
        while self._running and self._state.system_running:
            try:
                if not self._state.sensor_triggered:
                    time.sleep(0.1)
                    continue

                snap = self._align()
                if snap is None:
                    continue

                self._process(snap)

            except AlignmentError:
                continue
            except DatabaseError:
                self._state.db_connected = False
                time.sleep(1)
            except ModelError:
                time.sleep(1)
            except Exception:
                time.sleep(1)

    # --- processing pipeline ---

    def _process(self, snap: ThinkSnapshot):
        self._db.log_event(snap)
        features = self._db.build_feature_vector(self._db.last_row_id)
        danger_level = self._model.predict(features)
        action = self._lookup_action(danger_level)
        self._db.update_prediction(danger_level, action)
        self._state.danger_level = danger_level
        self._state.recommended_action = action

    # --- alignment ---

    def _align(self):
        # Drop partials: misaligned snapshots are worse than no snapshot.
        sense_snap = None
        see_snap = None

        if self._sense_enabled:
            if self._state.sense_queue.empty():
                return None
            sense_snap = self._state.sense_queue.get()

        if self._see_enabled:
            if self._state.see_queue.empty():
                return None
            see_snap = self._state.see_queue.get()

        if sense_snap and see_snap:
            gap_ms = abs((sense_snap.timestamp - see_snap.timestamp).total_seconds() * 1000)
            if gap_ms > self._max_gap_ms:
                raise AlignmentError(f"Timestamp gap {gap_ms}ms exceeds max {self._max_gap_ms}ms")

        if not sense_snap and not see_snap:
            return None

        return ThinkSnapshot(
            timestamp=sense_snap.timestamp if sense_snap else see_snap.timestamp,
            sensor_snapshot=sense_snap,
            vision_snapshot=see_snap,
        )

    # --- helpers ---

    def _lookup_action(self, danger_level: int) -> str:
        poa_map = self._config.get("think", {}).get("poa_map", {})
        return poa_map.get(str(danger_level), "monitor")

    def _load_model(self):
        active = self._active_model
        try:
            if active == "xgboost":
                self._model = XGBoostModel(self._config)
            elif active == "rule_engine":
                raise ModelError("RuleEngine is not yet implemented. Switch to 'xgboost' in config.")
            elif active == "neural_net":
                raise ModelError("NeuralModel is not yet implemented. Switch to 'xgboost' in config.")
            else:
                raise ModelError(f"Unknown model type '{active}'. Available: xgboost")
            self._model.load(self._model_path)
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"Failed to load model '{active}': {e}")