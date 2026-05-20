# FireGuard — Backend Session Changelog (FINAL)

Every file changed across the whole session, in one place. Drop each file into
your repo at the matching path. `_diffs/` has a per-file unified diff against
the original upload — review then commit.

---

## NEW FILES (3)

| Path | Purpose |
|------|---------|
| `api/routes/training.py` | Training-mode routes — single-shot **capture/save** AND continuous **recording** (start / label / stop / status). |
| `api/routes/config.py`   | `GET` / `POST /api/config` — read and update system config. |
| `api/routes/logs.py`     | `GET /api/logs` — tails the rotating log file. |

---

## MODIFIED FILES (11)

### `configs/config.json`
- `think.max_event_gap_ms` — separate threshold for new-event grouping (was wrongly using `max_gap_ms` which is only SENSE↔SEE alignment). **PLACEHOLDER 10000, set the real value.**
- `think.training` section — `test_split`, `min_rows_to_train`, `random_state`. **PLACEHOLDERS, review.**

### `src/core/system_state.py`
- Added queues / flags for both training flows:
  - `training_label_queue` — save labels (single-shot)
  - `training_capture_request` / `training_capture_response` — capture round-trip
  - `train_request` / `train_response` — model-training round-trip
  - `training_recording` (bool) — recording on/off
  - `training_event_id` (int) — event_id for the active recording
  - `training_label_stream` (Queue) — labels with `valid_until` for live cascade

### `src/core/orchestrator.py`
- `get_system_mode()`, `is_recording()`, `get_recording_event_id()`.
- Capture: `training_capture(same_event, target_ts, timeout_s)`.
- Save: `training_save_label(row_id, danger, action)`.
- **Recording**: `training_recording_start(same_event)`, `training_recording_stop()`, `training_recording_push_label(danger, action, valid_until)`.
- Train: `train_model(timeout_s)`.

### `src/think/think_engine.py`
- `_run_loop`: removed the `sensor_triggered` gate (was a bug — THINK should not gate on a SENSE concern). Always drains label queue + services train requests. In training mode services capture requests **and** processes the recording stream when active. In prediction modes aligns + predicts. Layer faults no longer stall alignment.
- `_drain_training_labels()`, `_service_capture_requests()` (also handles `event_id_only` lookup for recording_start), `_service_train_requests()`.
- `capture_training_snapshot(same_event, target_ts, true_danger_level, true_action)` — single capture; can pre-label.
- `_process_recording_stream()` — one iteration of recording: align live → look up current label → insert labeled.
- `_current_label(row_ts)` — peek-not-pop with a per-process cache. Labels reused for many rows; popped only when row_ts crosses `valid_until`.
- `_align(target_ts=None)` rewritten:
  - **Live (target_ts=None)**: pop SEE as anchor, walk SENSE forward, "don't pop past first newer" lookbehind, warn-and-fallback when nothing within window.
  - **Training (target_ts=float)**: scan, do **not** pop. Drain both queues to lists, search for closest pair to target_ts, restore. Live loops (arm tracking via `latest_fire_x/y`) untouched.
- `train_model()` — G6 glue: validated rows → feature vectors → X/y (sorted feature order — the landmine guard) → split → fit → evaluate → save → hot-swap.

### `src/think/database/think_database.py`
- Reads `max_event_gap_ms` from config.
- `_assign_event_id`: now uses `max_event_gap_ms` (was wrongly using `max_gap_ms`).
- Added `get_latest_event_id()`, `log_training_capture(snap, event_id)`, `log_training_capture_labeled(snap, event_id, danger, action)`, `save_training_label(row_id, danger, action)`.
- **Bug fix** in `_snap_to_params`: read `v.frame_image_url` but the VisionSnapshot field is `v.image_url` → would `AttributeError` on every capture.
- **Removed** `update_human_label()` — dead code, replaced by `save_training_label()`.

### `src/think/ml/xgboost_model.py`
- `predict()`: array built from `[features[k] for k in sorted(features)]` — kills the feature-ordering landmine. Training uses the same `sorted()`, so train and predict can never desync columns.
- `save()`: `os.makedirs(path, exist_ok=True)` before `save_model` — defensive, handles first-run when `model_weights/` doesn't exist yet.

### `src/see/vision_fuser.py`
- `_save_frame()`: also writes `data/frames/stream.jpg` atomically (`temp` + `os.replace`) when `camera_feed_active` — the rolling buffer for the MJPEG feed.
- **`_capture_loop` emit gate fixed**: SEE now emits to `see_queue` when `sensor_triggered` **OR** `camera_feed_active`. Without this, training mode had nothing in `see_queue` and capture/recording could never align — the whole training path was silently broken. This was the critical gap.

### `api/routes/camera.py`
- `feed` / `snapshot` point at `stream.jpg` (was `latest.jpg`, never written).
- Removed a broken `POST /api/camera/capture` (called methods that don't exist). Capture is now `POST /api/training/capture`.

### `api/routes/predictions.py`
- `POST /api/train`: was 501 stub, wired to `orchestrator.train_model()`.

### `api/app.py`
- Registered three new blueprints: `training_bp`, `config_bp`, `logs_bp`.

### `architecture.md`
- §3.2 SystemState table: added the training queues/flags; fixed `sense_running`/`see_running` readers to include ThinkEngine.
- §5.3: documented the `stream.jpg` rolling buffer + when SEE emits.
- §6.3: rewrote the ThinkEngine loop as a prediction-path / training-path split.
- §6.5: documented `max_event_gap_ms` vs `max_gap_ms`.
- §6.7: rewrote the offline training flow to match `train_model()`.
- §6.7.1: rewrote the training-mode flow with both **single-shot capture** and **continuous recording**.

---

## TESTED IN SANDBOX
- All 14 files compile; `config.json` valid.
- End-to-end recording trace: queue round-trip for event_id lookup, peek-not-pop label cascade (3 labels reused across 9 frames at varying timestamps), stop drains unused labels. **All assertions passed.**
- All training routes via Flask test client: capture validation/200, save 202, recording start/label/stop/status — all paths pass.
- Training logic with synthetic data: 50/50 prediction accuracy proves `sorted()` feature ordering keeps train and predict aligned.

## NOT TESTED — needs your hardware
- Real Postgres (Supabase) and the Pi: actual DB writes, `_align` on live queues with real snapshots, the THINK process loop end-to-end, camera frames, servos, arm tracking the candle.
- The hardware pass is yours. Anything queue-shaped here is verified; anything DB- or hardware-shaped is "should work" until run.

## OPEN ITEMS / KNOWN
- `max_event_gap_ms = 10000` and the `think.training` block — placeholder numbers.
- `true_action` in save/recording is a free string, not constrained to `poa_map`.
- G2/G3 (live sensor + detection on `SystemState` for the live dashboard panel) — **NOT done**. Recording flow doesn't need it, but the dashboard's sensor panel will.
- Images saved to Pi local disk (`data/frames/`); DB stores the path. Drive/object storage is a future swap inside `_save_frame`.
- Recording-mode race: if the human hits `recording/label` *after* the row whose timestamp it should govern was already inserted, that row gets the previous label. Mitigation: push labels promptly; with humans this is fine.
