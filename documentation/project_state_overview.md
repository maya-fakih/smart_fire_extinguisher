# Fire Robot — Project State Overview
Generated: Thu May 14, 2026 — 19:57 Beirut time

---

## What Was Fixed This Session (Integration Issues)

### Issue 1 — SEE: Coordinates not normalized + missing SystemState write
**Problem:** `FireDetector._build_clusters()` was computing `origin_x/y` as raw pixel
values. ACT's arm controller expected normalized [0,1] with 0.5 = center. Additionally,
`vision_fuser.py` never wrote the dominant cluster center to `state.latest_fire_x/y` at
all — ACT's visual servoing feedback was completely dead.

**Fix:** `_build_clusters()` now accepts `frame_width` and `frame_height` instead of
`frame_area`, and normalizes `origin_x/y` before storing on `FireCluster`. `VisionFuser._capture_loop()`
now writes `clusters[0].origin_x/y` to state every frame, and clears to `None` when no
fire detected.

**Why it matters:** Hardware-agnostic coordinates mean the same trained XGBoost model
works with any camera resolution. ACT's arm tracking now has a live feedback signal.

---

### Issue 2 — SENSE: `latest_heat_matrix` never written to SystemState
**Problem:** ACT's arm controller reads `state.latest_heat_matrix` as its primary
feedback signal (preferred over camera). The field existed in SystemState but nothing
ever wrote to it. Heat-based arm pointing was silently dead.

**Fix:** One conditional write added to `SensorFuser._sensor_loop()` after each poll:
```python
if sensor.name == "heat_grid":
    self._state.latest_heat_matrix = physical
```
Sensor identified by name (not device type) so hardware can be swapped in config
without touching code.

---

### Issue 3 — `exceptions/__init__.py` missing `ActuatorFaultError`
**Problem:** ACT layer (act_engine, arm_controller, pump_actuator) all imported
`ActuatorFaultError` from the `exceptions` package, but `__init__.py` only exported
6 of the defined exceptions — `ActuatorFaultError` was not among them. ImportError
on startup.

**Fix:** `exceptions/__init__.py` updated to re-export all defined exception classes.

---

### Issue 4 — `see/__init__.py` imported from wrong module after class split
**Problem:** After `Detection` and `FireCluster` were split into their own files
(`see/models/detection.py`, `see/models/fire_cluster.py`), `see/__init__.py` still
tried to import them from `see.snapshot`. ImportError on startup.

**Fix:** `see/__init__.py` updated to import from the correct modules.

---

### Issue 5 — `notification_service.py` was an empty file
**Problem:** Orchestrator imported and constructed `NotificationService` at startup.
The file was 0 bytes — ImportError on startup.

**Fix:** Full `NotificationService` implemented with:
- 22 event types across 6 categories (fire events, sensor faults, camera faults,
  actuator faults, system faults, lifecycle)
- 3 severity levels (info / warn / critical)
- Log delivery (always) + PostgreSQL delivery (when DB available)
- Lazy DB connection — DB down never crashes the notifier
- `DEFAULT_SEVERITY` map so callers don't have to think about severity

---

### Issue 6 — Notification wiring missing from all 4 layers
**Problem:** Each layer had fault paths that only logged. No layer called
`NotificationService`. ACT's `_notify()` stub logged a string and did nothing else.

**Fix:** All 4 layers accept `notifier=None` in constructor (backward compatible).
Orchestrator constructs one shared `NotificationService` and passes it to every layer.
Each layer fires the appropriate `EventType` at the source of each fault.

---

### Issue 7 — `config.json` had no `vision` section
**Problem:** `VisionFuser.__init__()` reads `config["vision"]` on construction.
The key did not exist in `config.json`. KeyError crash on startup.

**Fix:** `vision` section added to `config.json` with camera fps, YOLO model path,
conf_threshold, labels path, and frame storage paths. `configs/labels.json` populated
with `{"0": "fire", "1": "other", "2": "smoke"}`.

---

### Issue 8 — `main.py` was empty (0 bytes)
**Problem:** No entry point. System could not be started.

**Fix:** `main.py` written with argument parsing (`--config`), logging setup from
config, `SystemOrchestrator` construction, SIGINT/SIGTERM signal handling, clean
shutdown on exit.

---

### Issue 9 — ThinkEngine crashed on missing model weights
**Problem:** `_load_model()` raised `ModelError` and propagated it up through `start()`,
killing the ThinkProcess if `model_weights/xgboost_model.json` didn't exist. Expected
on every first boot before training.

**Fix:** `_load_model()` catches `ModelError`, sets `self._model = None`, and forces
`system_mode = "training"` instead of crashing. `_process()` checks mode first —
training mode stores snapshots to DB for human labeling and returns without predicting.
System degrades gracefully and waits for training data.

---

### Issue 10 — `SensorFaultError` defined twice
**Problem:** `SensorFaultError` was defined in `exceptions/exceptions.py` AND
redefined locally in `sense/sensors/sensor_base.py`. Two different classes with the
same name. `except SensorFaultError` in any layer importing from `exceptions` would
not catch the exception raised by SENSE (which used its own local class).

**Fix:** Local definition deleted from `sensor_base.py`. Import added:
`from exceptions import SensorFaultError`.

---

### Bonus — SEE models split into one class per file
`fire_detector.py` originally contained 3 classes (`Detection`, `FireCluster`,
`FireDetector`). Split into:
- `see/models/detection.py`
- `see/models/fire_cluster.py`
- `see/models/fire_detector.py`
`see/models/__init__.py` updated to export all three.

---

## Current System State

| Layer | Status |
|---|---|
| SENSE | Complete. Polls sensors, writes to state, emits SensorSnapshot to queue on threshold. Heat matrix write to state confirmed. |
| SEE | Complete (Jana). Normalizes coordinates, writes fire center to state every frame, emits VisionSnapshot to queue when sensor triggered. |
| THINK | Complete. Training mode fallback on missing weights. Predicts in normal modes, collects for labeling in training mode. |
| ACT | Complete. Pan/tilt arm with visual servoing, pump relay, 3 modes (autopilot/copilot/surveillance). |
| NOTIFY | Complete. Full event taxonomy, DB persistence, log fallback. |
| DASHBOARD | Stub only — `app.py` is empty. |
| main.py | Written. Entry point exists. |

---

## Deferred Issues (Not Blocking, Come Back To)

### D1 — DB schema: `recommended_action TEXT` → `recommended_actions JSONB`
Current `think_schema` stores recommended actions as a comma-joined TEXT string.
Should be JSONB for proper multi-action support. Deferred to avoid a migration
during active development. Do this before the demo.

### D2 — `SensorFaultError` defined in orchestrator layer_process_crashed notification
Orchestrator does not yet detect when a child process dies and fire
`LAYER_PROCESS_CRASHED`. Worth adding a health-check loop in `orchestrator.start()`
that watches `process.is_alive()` and notifies + logs when a process dies.

### D3 — Pump safety cutoff notification not wired
`PumpActuator._safety_cutoff()` logs when the hardware timer fires but does not call
`NotificationService`. Requires passing the notifier into actuators (currently only
passed to engines). Small refactor, do before demo.

### D4 — Architecture doc sections 7.4 and 7.6 still describe IK/DHSolver
These sections were written before the pan/tilt + visual servoing decision. Need
rewrite to describe the actual arm design. Do before report submission.

### D5 — Per-sensor polling intervals
All sensors currently share `polling_interval_active_ms`. Heat sensor may want a
faster poll rate than gas sensor. Config structure supports it, code does not yet
read per-sensor intervals.

### D6 — Frame storage strategy
Frames are currently saved to `data/frames/` locally. Pi SD card will fill fast at
30fps. Agreed approach: discard frames during normal operation, save only on
`danger_level >= 3` events, async upload to Google Drive. Not implemented yet.
Needed for training data collection and website live stream.

### D7 — YOLO `.pt` → `.rpk` conversion
The `.rpk` file for the IMX500 is not a simple script conversion. Requires Sony's
proprietary toolchain. Agreed approach: training repo publishes pre-built `.rpk` as
a GitHub release artifact. Makefile rule downloads it:
```makefile
download-model:
    wget https://github.com/your-org/fire-yolo/releases/latest/download/fire_smoke.rpk \
         -O model_weights/rpk/fire_smoke.rpk
```

### D8 — `vision` config: resolution
Removed from config since `frame.shape` is the source of truth at runtime. If
`IMX500Camera` still needs resolution for picamera2 hardware config, give it a
sensible default in the constructor rather than reading from config.

### D9 — Minor: SENSE main loop doesn't exit if all sensor threads die
`SensorFuser._main_loop()` keeps sleeping if every sensor thread has faulted.
Should detect this and log loudly / update state.

### D10 — Minor: alignment gap behavior
Currently a single large timestamp gap raises `AlignmentError` and skips the
snapshot. Consider whether repeated gaps should trigger a notification or mode
change rather than silent skipping.

---

## What To Work On Next

### 1. Backend API (Flask / FastAPI)
The website needs REST endpoints to talk to the running system via `SystemOrchestrator`.
Minimum endpoints needed:

| Endpoint | Method | What |
|---|---|---|
| `/api/state` | GET | Returns `orchestrator.get_state_summary()` |
| `/api/mode` | POST | `{"mode": "copilot"}` → calls `orchestrator.set_mode()` |
| `/api/notifications` | GET | Queries `notifications` table, supports `?severity=critical` filter |
| `/api/notifications/<id>/acknowledge` | POST | Marks notification as read |
| `/api/predictions` | GET | Queries `think_schema` table, paginated |
| `/api/predictions/<id>/label` | POST | `{"true_danger_level": 2}` → human label for training |
| `/api/copilot/decision` | POST | `{"decision": "approved"}` → sets `state.copilot_decision` |
| `/api/camera/feed` | GET | Toggle camera feed active |
| `/api/train` | POST | Triggers XGBoost training on labeled rows in DB |

### 2. Frontend
- Live dashboard: system mode, danger level, active sensors, recent notifications
- Copilot panel: shows pending prediction, approve/reject buttons
- Training panel: shows unlabeled snapshots with sensor readings + frame image,
  human assigns `true_danger_level`, submits label
- Notifications feed: filterable by severity, acknowledgeable
- Mode switcher: surveillance / copilot / autopilot / training

### 3. Simulation mode
Add `"simulation": true` flag to each layer's config section. When true, each layer
generates realistic fake data instead of talking to hardware. Goal: full end-to-end
pipeline test on a laptop without any Pi hardware.
- SENSE simulation: random sensor readings with occasional threshold spikes
- SEE simulation: random fire cluster positions and confidence values
- ACT simulation: log servo commands instead of writing to GPIO
- THINK: runs normally (no simulation needed — it's pure logic)

### 4. Training pipeline
- Collect labeled rows in `think_schema` (human sets `true_danger_level` via website)
- `/api/train` endpoint triggers `XGBoostModel.fit()` on validated rows
- Save model to `model_weights/xgboost_model.json`
- System auto-detects new weights and switches out of training mode

### 5. Hardware test (once hardware team delivers)
- Update servo pins, pump pin in `config.json act.actuators` to match real wiring
- Discover servo invert flags on bench (pixel error sign vs rotation direction)
- Tune `heat_use_threshold_c` and `tolerance_normalized` empirically
- Run `python src/main.py` on Pi and watch logs

---

## Demo Plan (FYP)
- System runs in **copilot mode** with trained model
- Model predicts low danger (candle/lighter = level 1-2) → no auto-action
- Human reviewer on website sees prediction + camera feed
- Human clicks **approve** on copilot panel
- ACT activates pump + arm tracks fire
- Shows full pipeline working end-to-end with human in the loop
