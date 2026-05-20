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
        self._notifiaer      = notifier
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
                # ── Always drain the training-label queue first ───────────────
                # Website → THINK: human labels from training mode.
                self._drain_training_labels()

                # ── Service model-training requests (any mode) ────────────────
                # Training can be triggered from any mode — the offline flow
                # says "system keeps running normally" — so this is outside
                # the training-mode branch below.
                self._service_train_requests()

                # ── Training mode: the human drives, not the pipeline ─────────
                # In training mode THINK does two things and nothing else:
                #   1. drain the label queue (done above)
                #   2. service capture requests from the API (below)
                # THINK is the only process that runs _align + owns the DB, so
                # capture must happen here — the API just sends a request and
                # waits on the response queue.
                if self._state.system_mode.value == "training":
                    self._service_capture_requests()
                    if self._state.training_recording:
                        self._process_recording_stream()
                    time.sleep(0.05)
                    continue

                # ── Align SENSE + SEE (prediction modes only) ─────────────────
                # THINK does NOT gate on sensor_triggered — that is SEE's concern.
                # THINK only waits until it has data to align. If a layer is
                # faulted (its running bool is False), THINK does not wait on
                # that layer's queue — it aligns with that side as None and the
                # feature vector fills the missing fields with nan.
                sense_ok = self._state.sense_running
                see_ok   = self._state.see_running

                # Nothing to consume from either live queue → idle briefly.
                if self._state.sense_queue.empty() and self._state.see_queue.empty():
                    time.sleep(0.05)
                    continue

                # Normal case: wait for both. If a layer is faulted, don't wait
                # on it — proceed with whatever the healthy layer produced.
                if sense_ok and see_ok:
                    if self._state.sense_queue.empty() or self._state.see_queue.empty():
                        time.sleep(0.05)
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

    # ── Training-label queue drain ────────────────────────────────────────────

    def _drain_training_labels(self):
        """
        Drain the website → THINK training-label queue.

        Each entry: {"row_id": int, "true_danger_level": int,
                     "true_action": str|None}. The row_id is the primary key
        returned at capture time — the label writes straight back to that row.
        """
        while not self._state.training_label_queue.empty():
            try:
                label = self._state.training_label_queue.get_nowait()
            except Exception:
                break
            try:
                self._db.save_training_label(
                    row_id            = label["row_id"],
                    true_danger_level = label["true_danger_level"],
                    true_action       = label.get("true_action"),
                )
                logger.info(
                    f"ThinkEngine: training label saved | "
                    f"row_id={label['row_id']} | "
                    f"danger={label['true_danger_level']}"
                )
            except Exception as e:
                logger.error(
                    f"ThinkEngine: failed to save training label - "
                    f"{type(e).__name__}: {e}",
                    exc_info=True
                )

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

    # ── Training-capture request servicing ────────────────────────────────────

    def _service_capture_requests(self):
        """
        Service training-capture requests from the API (training mode only).

        The API route pushes {request_id, same_event} onto
        training_capture_request and blocks on training_capture_response.
        This runs capture_training_snapshot inside the THINK process — the
        only process that owns _align and the live DB connection — then pushes
        the result back tagged with the same request_id so the API can match it.
        """
        while not self._state.training_capture_request.empty():
            try:
                req = self._state.training_capture_request.get_nowait()
            except Exception:
                break

            request_id = req.get("request_id")
            try:
                if req.get("event_id_only"):
                    # Recording-start round-trip: do NOT do a capture, only
                    # tell the orchestrator what event_id the next recording
                    # should use.
                    last = self._db.get_latest_event_id()
                    eid  = last if req.get("same_event", True) else last + 1
                    result = {"event_id": eid}
                else:
                    result = self.capture_training_snapshot(
                        same_event=req.get("same_event", True),
                        target_ts=req.get("target_ts"),
                    )
                self._state.training_capture_response.put({
                    "request_id": request_id,
                    "ok": True,
                    "result": result,
                    "error": None,
                })
                logger.info(f"ThinkEngine: capture serviced | request_id={request_id}")
            except Exception as e:
                logger.error(
                    f"ThinkEngine: capture request failed - {type(e).__name__}: {e}",
                    exc_info=True
                )
                self._state.training_capture_response.put({
                    "request_id": request_id,
                    "ok": False,
                    "result": None,
                    "error": f"{type(e).__name__}: {e}",
                })

    # ── Training-mode recording stream (live label-while-streaming) ───────────
    # Different from capture-and-save: while training_recording is True THINK
    # acts like the live pipeline (popping sense_queue/see_queue, aligning),
    # but instead of running XGBoost it reads the *current* label from the
    # label-stream queue and writes the row to DB already labeled. The label
    # is reused for many rows until time crosses its valid_until, then the
    # next label takes over. No batches, no end-of-recording processing pause.

    def _process_recording_stream(self):
        """
        One iteration of recording-mode consumption.

        Same alignment as the live pipeline (destructive pop, anchored on SEE)
        — that's correct here because we WANT to consume the queues. The arm
        tracking does not depend on these queues; it reads
        SystemState.latest_fire_x/y which SEE writes outside the queue.

        Each aligned row gets the *current* label (peek-not-pop on the label
        queue) and is inserted via log_training_capture_labeled.
        """
        # Nothing to do if no SEE frame is waiting.
        if self._state.see_queue.empty():
            return
        try:
            snap = self._align_live()
        except AlignmentError as e:
            logger.warning(f"ThinkEngine: recording alignment skip - {e}")
            return
        if snap is None:
            return

        label = self._current_label(snap.timestamp.timestamp())
        if label is None:
            # No label set yet — drop the row rather than write an unlabeled
            # one. (We agreed: no unlabeled firehose.)
            logger.debug("ThinkEngine: recording row dropped — no current label")
            return

        try:
            self._db.log_training_capture_labeled(
                snap,
                self._state.training_event_id,
                label["true_danger_level"],
                label.get("true_action"),
            )
        except Exception as e:
            logger.error(
                f"ThinkEngine: recording insert failed - {type(e).__name__}: {e}",
                exc_info=True
            )

    def _current_label(self, row_ts: float):
        """
        Peek the head of training_label_stream. If its valid_until is set and
        the row's timestamp has crossed it, pop and check the next. The
        survivor (if any) is the active label for this row.

        Labels are kept on a small per-process cache (self._label_head) so we
        don't drain the queue every iteration — only when we have to.
        """
        if not hasattr(self, "_label_head"):
            self._label_head = None

        # Pull in the newest labels that have arrived.
        while not self._state.training_label_stream.empty():
            try:
                next_lbl = self._state.training_label_stream.get_nowait()
            except Exception:
                break
            # If head exists and has no valid_until, the new one supersedes it
            # only after that head is naturally retired. Simpler model: any
            # new label replaces the head, with the old head's valid_until
            # carried forward as the new head's start. But the user-stated
            # rule is "head applies until row_ts >= valid_until, then pop".
            # So we keep a *list* of pending labels in order; head is index 0.
            if not hasattr(self, "_label_queue_cache"):
                self._label_queue_cache = []
            self._label_queue_cache.append(next_lbl)

        if not hasattr(self, "_label_queue_cache"):
            self._label_queue_cache = []

        # Retire expired head labels until the head applies to row_ts.
        while self._label_queue_cache:
            head = self._label_queue_cache[0]
            valid_until = head.get("valid_until")
            if valid_until is not None and row_ts >= valid_until:
                # This label's window has passed.
                self._label_queue_cache.pop(0)
                continue
            return head

        return None

    # ── Training-mode capture (runs inside the THINK process) ─────────────────

    def capture_training_snapshot(self, same_event: bool = True,
                                  target_ts: float = None,
                                  true_danger_level: int = None,
                                  true_action: str = None) -> dict:
        """
        Training-mode capture. Runs inside the THINK process, invoked by
        _service_capture_requests or _service_stream_request.

        target_ts (unix epoch float) anchors the alignment search — _align
        scans both queues non-destructively for the closest snapshots to
        target_ts within max_gap_ms. When omitted, falls back to live align
        (destructive pop), which is used only by tests / legacy paths.

        If true_danger_level is provided, the inserted row is labeled
        immediately in the same INSERT (used by stream mode, saves a roundtrip).
        Otherwise the row is unlabeled and a separate save_training_label
        call attaches the label later (the single-capture flow).

        Returns: {row_id, event_id, same_event, sensor:{...}, vision:{...}}.
        """
        snap = self._align(target_ts=target_ts)
        if snap is None:
            raise AlignmentError("capture: no SENSE/SEE snapshots available to align")

        last_event_id = self._db.get_latest_event_id()
        event_id = last_event_id if same_event else last_event_id + 1

        if true_danger_level is not None:
            row_id = self._db.log_training_capture_labeled(
                snap, event_id, true_danger_level, true_action
            )
        else:
            row_id = self._db.log_training_capture(snap, event_id)

        s = snap.sensor_snapshot
        v = snap.vision_snapshot
        return {
            "row_id": row_id,
            "event_id": event_id,
            "same_event": same_event,
            "sensor": {
                "triggered_sensors": s.triggered_sensors if s else None,
                "sensor_readings": s.sensor_readings if s else None,
                "sensor_normalized": s.sensor_normalized if s else None,
            } if s else None,
            "vision": {
                "composite_label": v.composite_label if v else None,
                "fire_count": v.fire_count if v else None,
                "smoke_count": v.smoke_count if v else None,
                "cluster_count": v.cluster_count if v else None,
                "fire_union_area": v.fire_union_area if v else None,
                "smoke_union_area": v.smoke_union_area if v else None,
                "scene_label": v.scene_label if v else None,
                "scene_confidence": v.scene_confidence if v else None,
                "glimpsed_fire": v.glimpsed_fire if v else None,
                "image_url": v.image_url if v else None,
            } if v else None,
        }

    # ── Model-training request servicing ──────────────────────────────────────

    def _service_train_requests(self):
        """
        Service model-training requests from the API (any mode).

        The API route pushes {request_id} onto train_request and blocks on
        train_response. Training runs inside the THINK process — it owns the DB
        and the model. The result (or error) is pushed back tagged with the
        same request_id.

        Note: train_model can take seconds. While it runs, this loop iteration
        is busy and the prediction path is paused — acceptable, training is an
        explicit human-triggered action and rare.
        """
        while not self._state.train_request.empty():
            try:
                req = self._state.train_request.get_nowait()
            except Exception:
                break

            request_id = req.get("request_id")
            try:
                result = self.train_model()
                self._state.train_response.put({
                    "request_id": request_id,
                    "ok": True,
                    "result": result,
                    "error": None,
                })
                logger.info(f"ThinkEngine: train request serviced | request_id={request_id}")
            except Exception as e:
                logger.error(
                    f"ThinkEngine: train request failed - {type(e).__name__}: {e}",
                    exc_info=True
                )
                self._state.train_response.put({
                    "request_id": request_id,
                    "ok": False,
                    "result": None,
                    "error": f"{type(e).__name__}: {e}",
                })

    # ── Model training (offline, triggered from website) ──────────────────────

    def train_model(self) -> dict:
        """
        Train the model on all validated rows in the DB.

        Glue only — the ML pieces already exist. Steps:
          1. get_validated_rows() → all rows with validated = TRUE
          2. build_feature_vector(row['id']) per row → feature dict
          3. assemble X (sorted feature order — see note) and y = true_danger_level
          4. train/test split (ratio from config['think']['training'])
          5. model.fit(X_train, y_train), then evaluate on the test split
          6. model.save(model_weights_path)

        Feature-ordering safety: every X row is built as
        [vec[k] for k in sorted(vec.keys())] — the SAME sorted() that
        XGBoostModel.predict() applies. build_feature_vector returns the same
        keys for every row, so the column order is identical across training
        and live prediction. This is the P2 landmine guard.

        Returns a dict with training results (sample counts + metrics) for the
        website to display. Raises ModelError if there is not enough data.
        """
        train_cfg   = self._config.get("think", {}).get("training", {})
        test_split  = train_cfg.get("test_split", 0.2)
        min_rows    = train_cfg.get("min_rows_to_train", 20)
        random_seed = train_cfg.get("random_state", 42)

        # 1. validated rows
        rows = self._db.get_validated_rows()
        if len(rows) < min_rows:
            raise ModelError(
                f"Not enough validated rows to train: have {len(rows)}, "
                f"need at least {min_rows} (config: think.training.min_rows_to_train)"
            )

        # 2-3. assemble X / y with a fixed sorted feature order
        import numpy as np
        X, y, skipped = [], [], 0
        feature_names = None
        for row in rows:
            vec = self._db.build_feature_vector(row["id"])
            if not vec:
                skipped += 1
                continue
            label = row.get("true_danger_level")
            if label is None:
                skipped += 1
                continue
            keys = sorted(vec.keys())
            if feature_names is None:
                feature_names = keys
            elif keys != feature_names:
                # Different feature keys between rows would desync columns.
                # build_feature_vector should never do this — guard anyway.
                logger.error(
                    f"train_model: feature key mismatch on row {row['id']} — skipping"
                )
                skipped += 1
                continue
            X.append([vec[k] for k in keys])
            y.append(int(label))

        if len(X) < min_rows:
            raise ModelError(
                f"Not enough usable rows after feature building: {len(X)} "
                f"({skipped} skipped — empty feature vector or missing label)"
            )

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        # 4. train/test split
        from sklearn.model_selection import train_test_split
        # stratify only if every class has at least 2 samples, else split plain
        unique, counts = np.unique(y, return_counts=True)
        stratify = y if counts.min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_seed, stratify=stratify
        )

        # 5. fit + evaluate — train a fresh model instance
        model = XGBoostModel(self._config)
        model.fit(X_train, y_train)
        # evaluate() compares against shifted labels (fit shifts 1-5 → 0-4)
        metrics = model.evaluate(X_test, [label - 1 for label in y_test])

        # 6. save weights, then hot-swap into the running engine
        model.save(self._model_path)
        self._model = model

        result = {
            "ok": True,
            "rows_total": len(rows),
            "rows_used": len(X),
            "rows_skipped": skipped,
            "feature_count": len(feature_names) if feature_names else 0,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "metrics": metrics,
            "model_path": self._model_path,
        }
        logger.info(
            f"ThinkEngine: training complete | used={len(X)} "
            f"accuracy={metrics.get('accuracy'):.3f} f1_macro={metrics.get('f1_macro'):.3f}"
        )
        return result

    # ── Alignment ─────────────────────────────────────────────────────────────

    def _align(self, target_ts: float = None):
        """
        Align SENSE + SEE snapshots.

        Two modes:

          target_ts=None  → live pipeline. Pop SEE as the anchor (SEE is the
            slower producer), then walk SENSE forward against it: discard any
            SENSE older than SEE-window, stop on the first within max_gap_ms
            OR the first newer than SEE (do NOT pop past — gap only grows
            from there). If the "first newer" path fires it means nothing
            within window existed; log a WARNING with the gap so we can see
            it and raise max_gap_ms if it happens a lot.

          target_ts=float → training capture. Scan, do NOT pop. Snapshot the
            current queue contents into a list, search for the SEE closest to
            target_ts within max_gap_ms, then SENSE closest to that SEE.
            Restore everything in original order. The live loops downstream
            (ACT consuming SystemState.latest_fire_x/y, etc.) are unaffected.
            "Don't-pop-past" applies in spirit (we pick the best, not the
            first older); resource cost is intentional — training is rare.

        Returns ThinkSnapshot or None if neither side yielded anything.
        Raises AlignmentError when both sides exist but the gap exceeds
        max_gap_ms (live path only — training tolerates wider misses).
        """
        if target_ts is not None:
            return self._align_for_capture(target_ts)
        return self._align_live()

    # ── Live align (destructive — pops the queues) ────────────────────────────

    def _align_live(self):
        sense_snap = None
        see_snap   = None

        if not self._state.see_queue.empty():
            see_snap = self._state.see_queue.get()
            logger.debug(
                f"ThinkEngine: dequeued from see_queue | "
                f"timestamp={see_snap.timestamp.isoformat()}"
            )

        if see_snap is None:
            # No SEE → just pull SENSE if present, return what we have.
            if not self._state.sense_queue.empty():
                sense_snap = self._state.sense_queue.get()
            if sense_snap is None:
                return None
            return ThinkSnapshot(
                timestamp       = sense_snap.timestamp,
                sensor_snapshot = sense_snap,
                vision_snapshot = None,
            )

        # SEE is the anchor — walk SENSE forward looking for the best match.
        # prev = last SENSE we discarded as older-and-out-of-window;
        #        used as a fallback if we end up with nothing in-window.
        prev = None
        chosen = None
        while not self._state.sense_queue.empty():
            candidate = self._state.sense_queue.get()
            gap_ms = abs(
                (candidate.timestamp - see_snap.timestamp).total_seconds() * 1000
            )

            if gap_ms <= self._max_gap_ms:
                chosen = candidate
                break

            if candidate.timestamp >= see_snap.timestamp:
                # First SENSE newer than SEE and still out of window — do NOT
                # pop past this point. Either prev (older, discarded) or
                # candidate (newer, fresh) is our best option.
                logger.warning(
                    f"ThinkEngine: alignment_first_newer_fallback | "
                    f"see_ts={see_snap.timestamp.isoformat()} | "
                    f"sense_ts={candidate.timestamp.isoformat()} | "
                    f"gap_ms={gap_ms:.0f} | max={self._max_gap_ms} | "
                    f"consider increasing max_gap_ms"
                )
                # Put it back — fresh SENSE belongs to the next live cycle.
                self._state.sense_queue.put(candidate)
                chosen = prev  # may be None — fall through
                break

            # Older and out of window: discard, remember as prev, continue.
            prev = candidate

        # If we walked the queue empty without finding anything, prev is the
        # closest we ever saw. If prev is also None, SEE goes solo.
        if chosen is None:
            chosen = prev

        if chosen is not None:
            gap_ms = abs(
                (chosen.timestamp - see_snap.timestamp).total_seconds() * 1000
            )
            if gap_ms > self._max_gap_ms:
                # Best we could do was still out of window. Don't pretend it's
                # aligned — raise so the loop logs an AlignmentError and skips
                # this cycle, consistent with prior behavior.
                raise AlignmentError(
                    f"Best SENSE candidate {gap_ms:.0f}ms from SEE — exceeds "
                    f"max {self._max_gap_ms}ms"
                )

        return ThinkSnapshot(
            timestamp       = see_snap.timestamp,
            sensor_snapshot = chosen,         # may be None — SEE-only is allowed
            vision_snapshot = see_snap,
        )

    # ── Training-capture align (non-destructive — scan only) ──────────────────

    def _align_for_capture(self, target_ts: float):
        """
        Non-destructive scan of both queues. target_ts is a unix-epoch float
        (the latest SEE timestamp at the moment of the human's click).
        """
        sense_items = self._drain(self._state.sense_queue)
        see_items   = self._drain(self._state.see_queue)
        try:
            see_snap   = self._pick_closest(see_items,   target_ts, label="SEE")
            sense_snap = self._pick_closest(sense_items, target_ts, label="SENSE") \
                         if see_snap is None else \
                         self._pick_closest(sense_items,
                                            see_snap.timestamp.timestamp(),
                                            label="SENSE")
        finally:
            # Restore everything in original order — live loops keep working.
            for s in sense_items: self._state.sense_queue.put(s)
            for s in see_items:   self._state.see_queue.put(s)

        if sense_snap is None and see_snap is None:
            return None

        return ThinkSnapshot(
            timestamp       = (see_snap.timestamp if see_snap else sense_snap.timestamp),
            sensor_snapshot = sense_snap,
            vision_snapshot = see_snap,
        )

    @staticmethod
    def _drain(q) -> list:
        """Drain a Queue into a list without blocking. Order preserved."""
        items = []
        while True:
            try:
                items.append(q.get_nowait())
            except Exception:
                break
        return items

    def _pick_closest(self, items: list, target_ts: float, label: str):
        """
        Find the item whose .timestamp is closest to target_ts within
        max_gap_ms. If nothing within window: warn and return the closest
        anyway (don't pop past — same idea, just scanning instead of popping).
        Returns None only if items is empty.
        """
        if not items:
            return None
        best = min(items,
                   key=lambda s: abs(s.timestamp.timestamp() - target_ts))
        gap_ms = abs(best.timestamp.timestamp() - target_ts) * 1000
        if gap_ms > self._max_gap_ms:
            logger.warning(
                f"ThinkEngine: capture_align_first_newer_fallback | "
                f"side={label} | target_ts={target_ts:.3f} | "
                f"best_ts={best.timestamp.isoformat()} | gap_ms={gap_ms:.0f} | "
                f"max={self._max_gap_ms} | consider increasing max_gap_ms"
            )
        return best

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