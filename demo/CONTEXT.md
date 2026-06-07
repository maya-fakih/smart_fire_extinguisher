# Smart Fire Extinguisher — POC Context & Continuation Guide

> One-stop doc so you never have to re-explain the project. Last updated: meeting prep sprint.

---

## What we built (`demo/`)

A single-loop, no-threading, no-database POC that runs directly on the Pi.

| File | Purpose |
|---|---|
| `demo/train_model.py` | Generates `fake_data.csv` + trains `model.json` |
| `demo/main.py` | **The whole demo** — reads sensors, runs XGBoost, drives servos, renders UI |
| `demo/model.json` | Trained XGBoost (auto-generated, not committed) |
| `demo/fake_data.csv` | Synthetic training data (show at meeting) |

### How to run
```bash
cd smart_fire_extinguisher
python demo/train_model.py      # once — generates model.json
python demo/main.py             # the demo loop
# Keys: Q = quit,  C = toggle camera always-on
```

---

## Hardware wiring (from `configs/config.json`)

| Component | Interface | Details |
|---|---|---|
| AMG8833 thermal grid (8×8) | I2C | addr `0x69`, board SCL/SDA |
| ADS1115 ADC → smoke sensor | I2C | addr `0x48`, ch 0, gain 1 |
| IMX500 camera | CSI | RPK model loaded at boot |
| Pan servo | GPIO PWM | pin 12 (BCM), range −90°↔+90° |
| Tilt servo | GPIO PWM | pin 13 (BCM), range −45°↔+45° |
| Pump relay | GPIO | pin 17 (BCM) — not triggered in POC |

---

## System architecture (full repo)

```
SENSE  →  SensorFuser (AMG8833 + ADS1115)  →  SensorSnapshot  →  sense_queue
SEE    →  IMX500Camera + FireDetector       →  VisionSnapshot  →  see_queue
THINK  →  ThinkEngine (XGBoost)             →  SystemState.danger_level
ACT    →  ArmController (pan/tilt) + PumpActuator
```

All four layers run as **separate OS processes** in the full system (`src/main.py`).  
The POC collapses all four into one sequential loop.

---

## Sensor logic

### AMG8833 (heat_grid)
- Reads 8×8 grid of °C values via `adafruit_amg88xx`
- Threshold: **27°C** → triggers `sensor_triggered`
- Arm tracking threshold: **50°C** (`heat_use_threshold_c`) — **⚠ OVERRIDDEN in demo to 26.5°C** because candle only reaches 28-32°C. The config value is for production large-fire scenarios.
- Hotspot = argmax of flat array; `err_x/err_y` normalized to [−1, +1]
- `heat_flip_y = True` (ceiling-mount correction — row index inverted)
- Configured biases: `x_bias = −0.12`, `y_bias = −0.28`

### ADS1115 (smoke)
- Raw ADC → ppm via equation from config:  
  `A * (((3.3 - (raw/32767)*3.3) / ((raw/32767)*3.3)) * RL/Ro) ** B`
- Params: RL=8200, Ro=9500, A=574.25, B=−2.222
- Threshold: **300 ppm**

---

## XGBoost model

### Danger levels (output 1–5)
| Level | Label | Actions |
|---|---|---|
| 1 | SAFE | monitor |
| 2 | LOW | monitor |
| 3 | MEDIUM | alert |
| 4 | HIGH | alert + suppress |
| 5 | CRITICAL | alert + suppress + evacuate |

### Feature vector (22 features, sorted alphabetically at predict time)
Sensor chain features (last 5 readings):
- `smoke_latest/avg/variance/velocity/acceleration`
- `heat_grid_latest/avg/variance/velocity/acceleration`

Vision features (from IMX500 YOLO detections):
- `fire_count`, `smoke_count`, `cluster_count`
- `fire_union_area`, `smoke_union_area` (fraction of frame)
- `scene_confidence`, `composite_label_encoded` (0-3), `scene_label_encoded` (0-3)
- `fire_union_area_velocity`, `smoke_union_area_velocity`
- `glimpsed_fire`, `human_near_fire`

### Training data
`fake_data.csv` — 600 rows (120 per level), synthetic but calibrated to **real observed readings**:

| Danger | Heat range (AMG8833) | Smoke (ppm) | Notes |
|---|---|---|---|
| 1 | 20-24°C | 0-80 | ambient room |
| 2 | 25-28.5°C | 50-200 | candle detected marginally |
| 3 | 28-33°C | 150-500 | candle/lighter close (screenshot: 28.3°C = this) |
| 4 | 32-42°C | 400-1500 | larger flame |
| 5 | 40-55°C | 1000-5000 | serious fire |

**Key calibration fact from logs + screenshot:** AMG8833 reads candle/lighter at 27-32°C hotspot with room at 20-23°C. The config's default `raw_max=80°C` is the sensor ceiling, not what fire reads in practice.

---

## Camera / IMX500

- RPK model: `model_weights/rpk/fire_smoke_detection.rpk` — YOLO fine-tuned for fire/smoke
- Runs **on-chip** (no CPU inference). `imx500.get_outputs(metadata)` returns tensors.
- Output format: `[boxes (N,4 xyxy normalised), scores (N,), classes (N,), num_dets]`
- Confidence threshold: **0.2** (config: `vision.models.fire.conf_threshold`)
- Camera always-on option: toggle with `C` key in demo. Otherwise activates when sensor threshold hit.

---

## Arm tracking

Dead-band step controller (no PID, no IK):
1. Compute `err_x, err_y` from heat grid hotspot (normalized [−1, +1])
2. If `|err| > tolerance/2 (0.025)`, step by `step_deg (3°)` in error direction
3. Servo value mapped: `2*(deg - servo_min)/(servo_max - servo_min) - 1`
4. Arm only moves when `hotspot_temp > 50°C`

Working standalone test: `test_arm_base_sweep.py` — sweeps pan 0°→90°→−90°→0°.

---

## What's left / known gaps

### For the POC demo
- [ ] **Real data collection**: The XGBoost is trained on synthetic data. After the meeting, collect real labeled samples via the full pipeline's training mode and retrain.
- [ ] **YOLO output parsing validation**: `_parse_detections()` in `demo/main.py` uses the standard IMX500 output format. If detections don't appear, print `outputs` shapes and adjust the reshape logic — the exact tensor layout can vary by rpk build.
- [ ] **Smoke sensor calibration**: Ro=9500Ω is a default. Real calibration in clean air would improve ppm accuracy.

### For the full system (post-meeting)
- [ ] **GPIO sensor / UART sensor**: `SensorParser` stubs exist (`gpio_sensor.py`, `uart_sensor.py`) but are not implemented. Only I2C is live.
- [ ] **Pump actuation**: Relay on GPIO 17 is wired but the POC never triggers it. `PumpActuator` in full system handles this.
- [ ] **Human detection**: `human_near_fire` feature is always 0.0 in POC. The full pipeline has a scene classifier for this.
- [ ] **Notifications**: `src/notify/` module exists (Telegram, etc.) — not wired in POC.
- [ ] **Training mode UI**: Full system has a web dashboard for labeling. POC skips this.
- [ ] **Multi-process orchestration**: Full `src/main.py` uses `multiprocessing` + `SystemState` shared dict. POC is single-loop intentionally.
- [ ] **Database**: Full system writes every prediction to Postgres (`think_schema`). POC is stateless.

---

## Key files in full repo (for reference)

| Path | What it does |
|---|---|
| `src/main.py` | Full orchestrator — spawns 4 processes |
| `src/sense/sensors/i2c_sensor.py` | AMG8833 + ADS1115 read logic |
| `src/see/camera.py` | IMX500 + Picamera2 management |
| `src/see/models/fire_detector.py` | YOLO metadata parser + clustering |
| `src/think/ml/xgboost_model.py` | XGBoost fit/predict/save/load |
| `src/think/database/think_database.py` | Feature vector builder (real DB chain) |
| `src/act/actuators/arm_controller.py` | Pan/tilt tracking loop (threaded) |
| `test_arm_base_sweep.py` | ✅ Working arm test — use to verify servos |
| `configs/config.json` | All pins, addresses, thresholds, model params |

---

## Quick sanity checks before demo

```bash
# Test I2C devices are visible
i2cdetect -y 1
# Should show 0x48 (ADS1115) and 0x69 (AMG8833)

# Test servos independently
python test_arm_base_sweep.py

# Test model loads
python -c "import xgboost as xgb; m=xgb.XGBClassifier(); m.load_model('demo/model.json'); print('OK')"

# Full demo
python demo/train_model.py && python demo/main.py
```
