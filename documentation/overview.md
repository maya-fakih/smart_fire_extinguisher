# FYP — Smart Autonomous Fire Extinguisher System
> Last updated: session 1
> Use this file to resume context with Claude next session.

---

## Project Overview
A smart autonomous fire extinguisher robot. Not a basic demo — a full pipeline from sensor fusion to actuation. Four main parts: CV/YOLO training, ML prediction, robotic arm control, and a web frontend.

---

## Team & Resources
- University supercomputer available for training
- Hardware: Pi 5 (brain), ESP32 (comms), 2-DOF arm with DC motors + encoders, heat matrix, optical smoke sensor, infrared sensor, AI Pi cam, mini LiDAR (later)
- GitHub repo for training: `yolo26-fire-training` — https://github.com/maya-fakih/yolo26-fire-training.git

---

## Part 1 — YOLO Training

### Why YOLOv26
- Only ~4 months old at time of writing, minimal open source fire work exists on it
- 42% efficiency gain from updated anchor-free task-aligned assigner (TAL) — cleaner gradient signal, fewer false positives on irregular shapes like fire and smoke
- Critical advantage: two models near-simultaneously on Pi 5
  - Thread 1 (always on): infrared watchdog at low frequency
  - Thread 2 (triggered): YOLOv26 full inference

### Dataset Strategy
- Merge 5+ datasets: wildfire, domestic, electric, campfire, smoke-only
- Challenge: inconsistent label schemas (COCO vs YOLO vs VOC), class name conflicts
- Normalise all to unified schema: `fire`, `smoke`, `ember`
- Split: 80/10/10 train/val/test, stratified by fire type

### Repo Structure
```
yolo26-fire-training/
├── data/
│   ├── raw/              # per-source datasets untouched
│   ├── merged/           # normalised unified schema
│   └── splits/           # train/val/test manifests
├── scripts/
│   ├── merge_datasets.py
│   ├── normalise_labels.py
│   ├── augment.py        # brightness, rotation, smoke overlay
│   └── validate_split.py
├── config/
│   └── yolov26-fire.yaml
├── train.py
├── export.py             # → yolov26-fire.pt
└── README.md
```

### Training Config
```yaml
model: yolov26n
imgsz: 640
epochs: 100
batch: 16
classes: [fire, smoke, ember]
device: cuda
```

### Output
- `yolov26-fire.pt` — single deployable weights file
- Inference gives: bbox `[x, y, w, h]` normalised 0–1, class label, confidence score

---

## Part 2 — FireEvent Class & ML Prediction

### FireEvent Class (see event.py)
Plain Python class — no polymorphic interface needed at this stage.
Designed to be extensible later for rain, flood, gas leak events.

#### Key Attributes
| Attribute | Type | Description |
|---|---|---|
| `heat_matrix` | `float[][]` | raw heat matrix readings from IoT |
| `smoke_level` | `float` | optical smoke sensor ppm |
| `lidar_z` | `float` | depth in metres |
| `bbox_x, bbox_y` | `float` | YOLO bounding box centre [0–1] |
| `fire_label` | `str` | e.g. "wildfire", "domestic" |
| `frame_area_ratio` | `float` | `bbox_w × bbox_h` — already 0–1 |
| `timestamp` | `datetime` | event time |
| `danger_level` | `float` | computed [0.0 – 1.0] |
| `growth_rate` | `float` | Δdanger / Δt |
| `predicted_action` | `str` | model output |

#### Key Equations
```
danger_level = α·frame_area_ratio + β·smoke_level_norm + γ·max(heat_matrix)
growth_rate  = (danger_level[t] − danger_level[t−1]) / Δt
```

#### Key Methods
- `compute_danger_level()` → float
- `compute_growth_rate(prev_event)` → float
- `save_to_csv(path)` — append row to fire_log.csv
- `load_history(path)` → List[FireEvent]
- `predict_action(model)` → str
- `cluster_danger(events)` → label
- `to_dict()` — serialise for API / email

#### CSV Schema
```
timestamp, bbox_x, bbox_y, frame_area_ratio, smoke_level,
max_heat, lidar_z, fire_label, danger_level, growth_rate
```

### Adaptive Sampling Strategy
| State | Condition | Sample rate | Camera |
|---|---|---|---|
| IDLE | below threshold | 1 per minute | off |
| LOCKED | fire confirmed | 1 per second | on |

- State transition is one-way until `frame_area_ratio` drops below reset threshold
- CSV file acts as live database — direct pandas read for model input, no SQL overhead

### ML Algorithm — Decision Pending
Two candidates, team to decide:

**XGBoost**
- Input: fixed window of last N CSV rows as flat feature vector
- Gradient boosted trees — non-linear, tabular, fast
- Stateless per call — Pi-friendly, interpretable feature importance
- Weakness: does not model sequence natively, window size is manual hyperparameter

**LSTM**
- Input: N rows as sequence shape `(N, features)`
- Hidden state: `h_t = f(x_t, h_{t-1})` — recursive relationship is architecturally native
- More expressive, heavier, needs more data to converge
- Inference only on Pi 5 (train offline)

**Plan:** Train both, compare MAE on `danger_level` and F1 on action class. Comparison is a results section deliverable.

---

## Part 3 — Robotic Arm

- 2-DOF arm, DC motors with encoders, controlled from Pi 5
- While sampling: arm roams using heat matrix as gradient ascent heuristic — maximise peak heat coordinate → lock onto fire source
- Action logic:
```python
if fire_label == "electric":
    trigger_circuit_breaker()
elif danger_level > THRESHOLD:
    activate_pump()
else:
    continue_tracking()
```
- ESP32 handles hardware comms — sub-100ms latency target

---

## Part 4 — Backend + Frontend

### Backend (Python)
- FastAPI — receives events from Pi 5 over HTTP POST
- Supabase — stores stats, alert history, email logs (not raw data points)
- Ollama / DeepSeek local — generates alert email text, free, no external API
- Sends actuator control commands back to Pi 5

### Frontend (Web)
- Live danger level gauge
- Fire event history with daily histograms
- LLM-generated alert email preview per event
- FI control panel — human accepts recommended action, robot executes
- Human is in the **action loop only** — not detection, not classification

---

## Presentation
- Tool: Gamma.ai
- Theme: dark charcoal + orange/amber accents, monospace font, terminal-style code blocks
- 13 slides, all technical, no motivation/intro fluff
- Diagrams generated: FireEvent class, ML pipeline flowchart, 4-layer system architecture
- Color system used in diagrams:
  - Background: `#111111`
  - Box fill: `#2A1400`
  - Orange border: `#F5A623`
  - White title text: `#FFFFFF`
  - Beige subtitle text: `#F5C070`
  - Arrow color: `#E8821A`

---

## Evaluation Targets
| What | Method | Target |
|---|---|---|
| Detection accuracy | mAP@0.5 on fire test set | > 0.85 |
| Danger prediction | MAE on danger_level | < 0.05 |
| Action classification | F1 score | > 0.90 |
| End-to-end latency | trigger → pump activation | < 2 s |

---

## Open Contributions
- `yolo26-fire-training` repo — reproducible pipeline, community can retrain on new fire data
- `FireEvent` schema — extensible base for other natural event prediction (rain, flood, gas)
- First published mAP benchmarks for YOLOv26 on multi-class fire dataset

---

## Notes & Decisions Log
- CNNs (TCN) ruled out — already have a CNN for detection, want to keep ML prediction lightweight
- No polymorphic interface for FireEvent for now — single class is enough, can extend later
- Supabase for stats/emails only — raw CSV data points not pushed to DB
- DeepSeek / Ollama local to avoid external API costs