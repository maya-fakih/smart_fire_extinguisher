# src/think/think_engine.py

import time
import json
import logging
from core.system_state import SystemState
from think.database.think_database import ThinkDatabase
from think.ml.base_model import BaseModel
from think.ml.xgboost_model import XGBoostModel
from think.snapshot import ThinkSnapshot
from exceptions import DatabaseError, AlignmentError, ModelError

logger = logging.getLogger(__name__)


class ThinkEngine:
    def __init__(self, config: dict, state: SystemState):
        self._config = config
        self._state = state
        self._model: BaseModel = None

        think_cfg = self._config.get("think", {})
        self._max_gap_ms = think_cfg.get("max_gap_ms", 500)
        self._model_path = think_cfg.get("model_weights_path", "model_weights/")
        self._active_model = think_cfg.get("active_model", "xgboost")

        self._running = False
        self._db = ThinkDatabase(self._config)

    # --- lifecycle ---

    def start(self):
        logger.info("ThinkEngine: starting")
        try:
            self._db.connect()
            self._state.db_connected = True
            logger.info("ThinkEngine: database connected")
        except DatabaseError as e:
            logger.error(
                f"ThinkEngine: database connection failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            self._state.db_connected = False
            raise

        self._load_model()
        self._running = True
        self._state.think_running = True
        logger.info(f"ThinkEngine: model loaded (active={self._active_model})")
        self._run_loop()

    def stop(self):
        logger.info("ThinkEngine: stop requested")
        self._running = False
        self._state.think_running = False
        self._db.close()
        logger.info("ThinkEngine: database closed")

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

            except AlignmentError as e:
                logger.warning(f"ThinkEngine: alignment error - {e}")
                continue
            except DatabaseError as e:
                logger.error(
                    f"ThinkEngine: database error - {type(e).__name__}: {e}",
                    exc_info=True
                )
                self._state.db_connected = False
                time.sleep(1)
            except ModelError as e:
                logger.error(
                    f"ThinkEngine: model error - {type(e).__name__}: {e}",
                    exc_info=True
                )
                time.sleep(1)
            except Exception as e:
                logger.error(
                    f"ThinkEngine: unexpected error - {type(e).__name__}: {e}",
                    exc_info=True
                )
                time.sleep(1)

    # --- processing pipeline ---

    def _process(self, snap: ThinkSnapshot):
        logger.debug(f"ThinkEngine: received snapshot | timestamp={snap.timestamp.isoformat()}")
        self._db.log_event(snap)
        features = self._db.build_feature_vector(self._db.last_row_id)
        logger.debug(f"ThinkEngine: feature_vector_built | features={features}")
        danger_level = self._model.predict(features)
        logger.info(f"ThinkEngine: model_prediction | danger_level={danger_level}")
        actions = self._lookup_actions(danger_level)
        logger.debug(f"ThinkEngine: actions_mapped | actions={actions}")
        self._db.update_prediction(danger_level, ",".join(actions))
        self._state.danger_level         = danger_level
        self._state.recommended_actions  = actions
        self._state.prediction_id        = self._state.prediction_id + 1
    
    # --- alignment ---

    def _align(self):
        sense_snap = None
        see_snap = None

        if not self._state.sense_queue.empty():
            sense_snap = self._state.sense_queue.get()
            logger.debug(f"ThinkEngine: dequeued from sense_queue | timestamp={sense_snap.timestamp.isoformat()}")

        if not self._state.see_queue.empty():
            see_snap = self._state.see_queue.get()
            logger.debug(f"ThinkEngine: dequeued from see_queue | timestamp={see_snap.timestamp.isoformat()}")

        if sense_snap is None and see_snap is None:
            return None

        if sense_snap and see_snap:
            gap_ms = abs((sense_snap.timestamp - see_snap.timestamp).total_seconds() * 1000)
            if gap_ms > self._max_gap_ms:
                logger.warning(
                    f"ThinkEngine: alignment_gap_exceeded | "
                    f"gap_ms={gap_ms} | max={self._max_gap_ms}"
                )
                raise AlignmentError(f"Timestamp gap {gap_ms}ms exceeds max {self._max_gap_ms}ms")

        return ThinkSnapshot(
            timestamp=sense_snap.timestamp if sense_snap else see_snap.timestamp,
            sensor_snapshot=sense_snap,
            vision_snapshot=see_snap,
        )

    # --- helpers ---

    def _lookup_actions(self, danger_level: int) -> list[str]:
        """
        poa_map values are lists of action names (stackable). Single-string
        legacy values are auto-wrapped in a list for backwards compatibility.
        """
        poa_map = self._config.get("think", {}).get("poa_map", {})
        actions = poa_map.get(str(danger_level), ["monitor"])
        return actions if isinstance(actions, list) else [actions]
    
    def _load_model(self):
        active = self._active_model
        logger.debug(f"ThinkEngine: loading model | active_model={active}")
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
            logger.info(f"ThinkEngine: model loaded successfully | type={active}")
        except ModelError as e:
            logger.error(
                f"ThinkEngine: model loading failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise
        except Exception as e:
            error_msg = f"Failed to load model '{active}': {e}"
            logger.error(
                f"ThinkEngine: model loading failed - {type(e).__name__}: {error_msg}",
                exc_info=True
            )
            raise ModelError(error_msg)