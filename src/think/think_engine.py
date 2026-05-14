# src/think/think_engine.py

import time
import logging
from core.system_state import SystemState
from think.database.think_database import ThinkDatabase
from think.ml.base_model import BaseModel
from think.ml.xgboost_model import XGBoostModel
from think.snapshot import ThinkSnapshot
from exceptions import DatabaseError, AlignmentError, ModelError

logger = logging.getLogger(__name__)


class ThinkEngine:
    def __init__(self, config: dict, state: SystemState, notifier=None):
        self._config        = config
        self._state         = state
        self._notifier      = notifier
        self._model: BaseModel = None

        think_cfg            = self._config.get("think", {})
        self._max_gap_ms     = think_cfg.get("max_gap_ms", 500)
        self._model_path     = think_cfg.get("model_weights_path", "model_weights/")
        self._active_model   = think_cfg.get("active_model", "xgboost")

        self._running = False
        self._db      = ThinkDatabase(self._config)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        logger.info("ThinkEngine: starting")

        # ── Connect to DB ─────────────────────────────────────────────────────
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
            if self._notifier is not None:
                from notify import EventType
                self._notifier.notify(
                    EventType.DATABASE_DISCONNECTED,
                    payload={"error": str(e)},
                    source_layer="think",
                )
            raise

        # ── Load model (may force training mode if weights missing) ───────────
        # Log first if we're already in training mode from config
        if self._state.system_mode.value == "training":
            logger.info(
                "ThinkEngine: starting in training mode — "
                "predictions disabled, collecting data for human labeling"
            )

        self._load_model()

        self._running = True
        self._state.think_running = True
        logger.info(
            f"ThinkEngine: started | mode={self._state.system_mode.value} | "
            f"model={'loaded' if self._model else 'none (training mode)'}"
        )
        self._run_loop()

    def stop(self):
        logger.info("ThinkEngine: stop requested")
        self._running = False
        self._state.think_running = False
        self._db.close()
        logger.info("ThinkEngine: database closed")

    # ── Main loop ─────────────────────────────────────────────────────────────

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
                if self._notifier is not None:
                    from notify import EventType
                    self._notifier.notify(
                        EventType.ALIGNMENT_DRIFT,
                        payload={"error": str(e)},
                        source_layer="think",
                    )
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

    # ── Processing pipeline ───────────────────────────────────────────────────

    def _process(self, snap: ThinkSnapshot):
        logger.debug(f"ThinkEngine: received snapshot | timestamp={snap.timestamp.isoformat()}")

        # ── Training mode: store snapshot for human labeling, skip prediction ─
        # Human labels the data via the website (sets true_danger_level).
        # Once enough labeled data exists, train the model and switch mode.
        if self._state.system_mode.value == "training":
            logger.debug("ThinkEngine: training mode — storing snapshot, awaiting human label")
            self._db.log_event(snap)
            # Don't set danger_level or prediction_id — nothing to act on
            return

        # ── Prediction mode: must have a model loaded ─────────────────────────
        if self._model is None:
            # Should not happen — _load_model forces training mode if no weights.
            # Safety net in case state was changed externally without a model.
            logger.error(
                "ThinkEngine: no model loaded but not in training mode — "
                "forcing back to training mode"
            )
            self._state.system_mode = "training"
            return

        # ── Normal prediction pipeline ────────────────────────────────────────
        self._db.log_event(snap)
        features    = self._db.build_feature_vector(self._db.last_row_id)
        logger.debug(f"ThinkEngine: feature_vector_built | features={features}")

        danger_level = self._model.predict(features)
        logger.info(f"ThinkEngine: model_prediction | danger_level={danger_level}")

        actions = self._lookup_actions(danger_level)
        logger.debug(f"ThinkEngine: actions_mapped | actions={actions}")

        self._db.update_prediction(danger_level, ",".join(actions))

        self._state.danger_level        = danger_level
        self._state.recommended_actions = actions
        self._state.prediction_id       = self._state.prediction_id + 1

    # ── Alignment ─────────────────────────────────────────────────────────────

    def _align(self):
        sense_snap = None
        see_snap   = None

        if not self._state.sense_queue.empty():
            sense_snap = self._state.sense_queue.get()
            logger.debug(
                f"ThinkEngine: dequeued from sense_queue | "
                f"timestamp={sense_snap.timestamp.isoformat()}"
            )

        if not self._state.see_queue.empty():
            see_snap = self._state.see_queue.get()
            logger.debug(
                f"ThinkEngine: dequeued from see_queue | "
                f"timestamp={see_snap.timestamp.isoformat()}"
            )

        if sense_snap is None and see_snap is None:
            return None

        if sense_snap and see_snap:
            gap_ms = abs(
                (sense_snap.timestamp - see_snap.timestamp).total_seconds() * 1000
            )
            if gap_ms > self._max_gap_ms:
                logger.warning(
                    f"ThinkEngine: alignment_gap_exceeded | "
                    f"gap_ms={gap_ms} | max={self._max_gap_ms}"
                )
                raise AlignmentError(
                    f"Timestamp gap {gap_ms}ms exceeds max {self._max_gap_ms}ms"
                )

        return ThinkSnapshot(
            timestamp       = sense_snap.timestamp if sense_snap else see_snap.timestamp,
            sensor_snapshot = sense_snap,
            vision_snapshot = see_snap,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _lookup_actions(self, danger_level: int) -> list[str]:
        """
        Map danger_level → list of action names via poa_map in config.
        Single-string legacy values are auto-wrapped for backwards compatibility.
        """
        poa_map = self._config.get("think", {}).get("poa_map", {})
        actions = poa_map.get(str(danger_level), ["monitor"])
        return actions if isinstance(actions, list) else [actions]

    def _load_model(self):
        """
        Load model weights from disk.

        If weights are missing or corrupt:
        - Log a warning (not an error — this is expected on first boot)
        - Set self._model = None
        - Force system_mode → training so the system collects labeled data
          instead of attempting predictions with no model

        Training mode is the safe fallback. The system will boot and run
        normally — it just won't make predictions until a model is trained
        and weights are saved to model_weights/.
        """
        active = self._active_model
        logger.debug(f"ThinkEngine: loading model | active_model={active}")

        try:
            if active == "xgboost":
                self._model = XGBoostModel(self._config)
            elif active == "rule_engine":
                raise ModelError("RuleEngine not implemented. Switch to 'xgboost' in config.")
            elif active == "neural_net":
                raise ModelError("NeuralModel not implemented. Switch to 'xgboost' in config.")
            else:
                raise ModelError(f"Unknown model type '{active}'. Available: xgboost")

            self._model.load(self._model_path)
            logger.info(f"ThinkEngine: model loaded successfully | type={active}")

        except ModelError as e:
            # Model file missing or corrupt — expected on first boot before training.
            # Don't crash. Force training mode so the system collects data.
            logger.warning(
                f"ThinkEngine: model weights not found or invalid — "
                f"forcing training mode | reason={e}"
            )
            self._model = None

            # Only force the switch if not already in training mode
            if self._state.system_mode.value != "training":
                logger.warning(
                    f"ThinkEngine: system was in '{self._state.system_mode.value}' mode — "
                    f"switching to training mode because no model weights exist"
                )
                self._state.system_mode = "training"

            if self._notifier is not None:
                from notify import EventType, Severity
                self._notifier.notify(
                    EventType.MODEL_LOAD_FAILED,
                    payload={
                        "model": active,
                        "error": str(e),
                        "action": "forced training mode — collect data and train",
                    },
                    severity=Severity.WARN,
                    source_layer="think",
                )

        except Exception as e:
            # Unexpected error (not just missing file) — this one we do raise
            error_msg = f"Unexpected error loading model '{active}': {e}"
            logger.error(
                f"ThinkEngine: {error_msg}",
                exc_info=True
            )
            if self._notifier is not None:
                from notify import EventType
                self._notifier.notify(
                    EventType.MODEL_LOAD_FAILED,
                    payload={"model": active, "error": error_msg},
                    source_layer="think",
                )
            raise ModelError(error_msg)