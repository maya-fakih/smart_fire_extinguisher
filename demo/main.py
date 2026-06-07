"""
demo/main.py  —  Smart Fire Extinguisher POC
=============================================
Single-file, single-loop demo.  No threads, no DB, no web server.

Hardware assumed (from config.json):
  - AMG8833 thermal 8×8 grid   I2C addr 0x69  → heat_grid sensor
  - ADS1115 ADC (smoke sensor)  I2C addr 0x48  → smoke sensor (ch 0)
  - IMX500 camera               → runs fire_smoke_detection.rpk on-chip
  - Pan servo  GPIO 12 (BCM)
  - Tilt servo GPIO 13 (BCM)
  - Pump relay GPIO 17 (BCM)    (not triggered in POC, wired but safe)

Loop (runs at ~polling_interval_active_ms = 200 ms):
  1. Read AMG8833 → 8×8 grid, hotspot, err_x, err_y
  2. Read ADS1115 → smoke ppm
  3. Build feature vector → XGBoost → danger_level (1-5)
  4. If cam_always_on OR sensor triggered → capture frame, detect fire
  5. Move pan/tilt toward hotspot (dead-band step controller)
  6. Render OpenCV UI:
       - Top banner:  danger level + label + colours
       - Left panel:  camera feed + YOLO bounding boxes
       - Right panel: heat grid coloured cells + numbers

Run:
    cd smart_fire_extinguisher
    python demo/train_model.py   # once
    python demo/main.py
    
Keys: Q = quit,  C = toggle camera always-on
"""

import sys, json, math, time
from pathlib import Path

import cv2
import numpy as np
import xgboost as xgb
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory

# ── Platform I2C (adafruit) ───────────────────────────────────────────────────
import board
import busio
import adafruit_amg88xx
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# ── IMX500 camera ─────────────────────────────────────────────────────────────
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
DEMO_DIR    = REPO_ROOT / "demo"
CONFIG_PATH = REPO_ROOT / "configs" / "config.json"
MODEL_PATH  = DEMO_DIR  / "model.json"
RPK_PATH    = str(REPO_ROOT / "model_weights" / "rpk" / "fire_smoke_detection.rpk")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

SMOKE_CFG   = CFG["sensors"]["smoke"]
HEAT_CFG    = CFG["sensors"]["heat_grid"]
CAM_CFG     = CFG["vision"]["camera"]
ARM_CFG     = CFG["act"]["actuators"]["arm"]
THINK_CFG   = CFG["think"]
POLL_MS     = CFG["system"]["polling_interval_active_ms"]

# Sensor thresholds
SMOKE_THRESH = SMOKE_CFG["threshold_physical"]   # 300 ppm
HEAT_THRESH  = HEAT_CFG["threshold_physical"]    # 27 °C

# Camera
CAM_W, CAM_H = CAM_CFG["resolution"]            # 320, 240
CONF_THRESH   = CFG["vision"]["models"]["fire"]["conf_threshold"]   # 0.2

# Arm limits / steps
PAN_CFG  = ARM_CFG["joints"]["pan"]
TILT_CFG = ARM_CFG["joints"]["tilt"]
ARM_TOL  = ARM_CFG["feedback"]["tolerance_normalized"]   # 0.05
HEAT_FLIP_Y = ARM_CFG["feedback"]["heat_flip_y"]         # True
HEAT_FLIP_X = ARM_CFG["feedback"]["heat_flip_x"]         # False

# ⚠ DEMO OVERRIDE: config says 50°C (designed for large fire) but candle/lighter
# only reaches 28-32°C on AMG8833. Override so arm actually tracks in demo.
# Swap back to ARM_CFG["feedback"]["heat_use_threshold_c"] for production.
ARM_HEAT_TRACK_THRESHOLD = 29   # °C — just above room baseline (~20-23°C)

# XGBoost labels (sorted keys must match train_model.py FEATURE_COLS order)
FEATURE_KEYS = sorted([
    "smoke_latest", "smoke_avg", "smoke_variance", "smoke_velocity", "smoke_acceleration",
    "heat_grid_latest", "heat_grid_avg", "heat_grid_variance",
    "heat_grid_velocity", "heat_grid_acceleration",
    "fire_count", "smoke_count", "cluster_count",
    "fire_union_area", "smoke_union_area", "scene_confidence",
    "composite_label_encoded", "scene_label_encoded",
    "fire_union_area_velocity", "smoke_union_area_velocity",
    "glimpsed_fire", "human_near_fire",
])

# POA map from config
POA_MAP = {int(k): v for k, v in THINK_CFG["poa_map"].items()}

# ─────────────────────────────────────────────────────────────────────────────
# UI constants
# ─────────────────────────────────────────────────────────────────────────────
UI_W, UI_H     = 960, 540
BANNER_H       = 70
PANEL_W        = UI_W // 2              # 480 px each side
CONTENT_H      = UI_H - BANNER_H       # 470 px

# Heat grid display
GRID_CELL      = 44                    # pixels per cell
GRID_ROWS      = GRID_COLS = 8
GRID_W         = GRID_COLS * GRID_CELL  # 352
GRID_H         = GRID_ROWS * GRID_CELL  # 352
GRID_X_OFFSET  = (PANEL_W - GRID_W) // 2
GRID_Y_OFFSET  = (CONTENT_H - GRID_H) // 2

DANGER_COLORS = {   # BGR, for each level 1-5
    1: (60, 180, 60),    # green
    2: (30, 200, 220),   # yellow-ish
    3: (0, 165, 255),    # orange
    4: (0, 80, 255),     # red-orange
    5: (0, 0, 220),      # deep red
}
DANGER_LABELS = {
    1: "SAFE — MONITORING",
    2: "LOW — ANOMALY",
    3: "MEDIUM — ALERT",
    4: "HIGH — SUPPRESSING",
    5: "CRITICAL — EVACUATE",
}

# Heat-cell colour map: blue (cold) → orange → red (hot)
def _heat_color(val: float, vmin=20.0, vmax=35.0) -> tuple:
    """Color scale tuned for candle/lighter range: 20°C (blue) → 35°C (red)."""
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    if t < 0.4:
        r = int(50 + t / 0.4 * 100)
        g = int(50 + t / 0.4 * 50)
        b = int(200 - t / 0.4 * 150)
    elif t < 0.7:
        s = (t - 0.4) / 0.3
        r = int(150 + s * 100)
        g = int(100 - s * 70)
        b = int(50)
    else:
        s = (t - 0.7) / 0.3
        r = int(250)
        g = int(30 - s * 20)
        b = int(50 - s * 40)
    return (b, g, r)   # BGR


# ─────────────────────────────────────────────────────────────────────────────
# Smoke equation helper (mirrors i2c_sensor._read_ads1115)
# ─────────────────────────────────────────────────────────────────────────────
def _raw_to_ppm(raw: float) -> float:
    p = SMOKE_CFG["eq_params"]
    RL, Ro, A, B = p["RL"], p["Ro"], p["A"], p["B"]
    raw = max(1.0, raw)
    ratio = (3.3 - (raw / 32767.0) * 3.3) / ((raw / 32767.0) * 3.3)
    return float(A * (ratio * RL / Ro) ** B)


# ─────────────────────────────────────────────────────────────────────────────
# Hardware init
# ─────────────────────────────────────────────────────────────────────────────
print("[init] Connecting I2C sensors …")
i2c = busio.I2C(board.SCL, board.SDA)
time.sleep(0.5)  # AMG8833 needs ~500ms after power-on before it responds

# AMG8833 heat grid
amg = adafruit_amg88xx.AMG88XX(i2c, addr=int(HEAT_CFG["address"], 16))

# ADS1115 smoke sensor
ads = ADS.ADS1115(i2c, address=int(SMOKE_CFG["address"], 16))
ads.gain = SMOKE_CFG["gain"]
smoke_ch = AnalogIn(ads, SMOKE_CFG["channel"])

# Pan/tilt servos
_factory = LGPIOFactory()
pan_servo  = Servo(pin=int(PAN_CFG["pin"]),  pin_factory=_factory)
tilt_servo = Servo(pin=int(TILT_CFG["pin"]), pin_factory=_factory)
pan_servo.value  = 0.0   # center
tilt_servo.value = 0.0

pan_angle  = 0.0
tilt_angle = 0.0

# IMX500 camera
print("[init] Loading IMX500 camera + RPK model (may take a minute) …")
imx500 = IMX500(RPK_PATH)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (CAM_W, CAM_H), "format": "RGB888"},
    controls={"FrameRate": CAM_CFG["fps"]},
))
picam2.start()
print("[init] Camera ready.")

# XGBoost model
print("[init] Loading XGBoost model …")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(MODEL_PATH))
print("[init] Ready. Starting main loop. Press Q to quit, C to toggle camera.")


# ─────────────────────────────────────────────────────────────────────────────
# Servo helpers
# ─────────────────────────────────────────────────────────────────────────────
def _deg_to_val(deg, smin, smax):
    span = smax - smin
    return max(-1.0, min(1.0, 2.0 * (deg - smin) / span - 1.0))

def _clip(v, lo, hi): return max(lo, min(hi, v))

def _step_arm(err_x: float, err_y: float):
    global pan_angle, tilt_angle
    if abs(err_x) > ARM_TOL / 2:
        sign = 1 if err_x > 0 else -1
        if PAN_CFG.get("invert", False): sign *= -1
        pan_angle = _clip(pan_angle + sign * PAN_CFG["step_deg"],
                          PAN_CFG["limit_min_deg"], PAN_CFG["limit_max_deg"])
        pan_servo.value = _deg_to_val(pan_angle, PAN_CFG["servo_min_deg"], PAN_CFG["servo_max_deg"])
    if abs(err_y) > ARM_TOL / 2:
        sign = 1 if err_y > 0 else -1
        if TILT_CFG.get("invert", False): sign *= -1
        tilt_angle = _clip(tilt_angle + sign * TILT_CFG["step_deg"],
                           TILT_CFG["limit_min_deg"], TILT_CFG["limit_max_deg"])
        tilt_servo.value = _deg_to_val(tilt_angle, TILT_CFG["servo_min_deg"], TILT_CFG["servo_max_deg"])


# ─────────────────────────────────────────────────────────────────────────────
# YOLO detection parser (IMX500 on-chip inference → bounding boxes)
# ─────────────────────────────────────────────────────────────────────────────
# Label map from labels.json
with open(REPO_ROOT / "configs" / "labels.json") as f:
    _labels_raw = json.load(f)
# labels.json is {id: name} or a list — normalise to dict
if isinstance(_labels_raw, list):
    LABELS = {i: v for i, v in enumerate(_labels_raw)}
else:
    LABELS = {int(k): v for k, v in _labels_raw.items()}


def _parse_detections(metadata) -> list:
    """
    Parse IMX500 on-chip YOLO output into list of dicts:
      {label, conf, x1, y1, x2, y2}  (pixel coords in CAM_W × CAM_H)
    Returns [] on failure or no detections above CONF_THRESH.
    """
    try:
        outputs = imx500.get_outputs(metadata)
        if outputs is None:
            return []
        # Standard YOLOv8 IMX500 output: [boxes, scores, classes, num_dets]
        # shapes vary — try to unpack robustly
        if len(outputs) < 3:
            return []
        boxes   = np.array(outputs[0]).reshape(-1, 4)   # [N, 4] normalised xyxy
        scores  = np.array(outputs[1]).reshape(-1)
        classes = np.array(outputs[2]).reshape(-1).astype(int)
        dets = []
        for i, (sc, cls) in enumerate(zip(scores, classes)):
            if sc < CONF_THRESH:
                continue
            x1n, y1n, x2n, y2n = boxes[i]
            x1 = int(x1n * CAM_W); y1 = int(y1n * CAM_H)
            x2 = int(x2n * CAM_W); y2 = int(y2n * CAM_H)
            label = LABELS.get(cls, str(cls))
            dets.append({"label": label, "conf": float(sc),
                          "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return dets
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector builder (rolling window — simple single-step version for demo)
# ─────────────────────────────────────────────────────────────────────────────
_smoke_hist       = []
_heat_hist        = []
_HIST_LEN         = 5     # chain length from config

def _update_hist(lst, val):
    lst.append(val)
    if len(lst) > _HIST_LEN: lst.pop(0)

def _series_stats(lst):
    arr = np.array(lst, dtype=float)
    latest = arr[-1]
    avg    = float(np.mean(arr))
    var    = float(np.var(arr)) if len(arr) >= 2 else 0.0
    vel    = float(arr[-1] - arr[0]) / max(len(arr) - 1, 1)
    acc    = 0.0
    if len(arr) >= 3:
        diffs = np.diff(arr)
        acc = float(diffs[-1] - diffs[0]) / max(len(diffs) - 1, 1)
    return latest, avg, var, vel, acc

def _build_features(smoke_ppm: float, heat_max: float, dets: list) -> dict:
    _update_hist(_smoke_hist, smoke_ppm)
    _update_hist(_heat_hist, heat_max)

    sl, sa, sv, svl, sac = _series_stats(_smoke_hist)
    hl, ha, hv, hvl, hac = _series_stats(_heat_hist)

    fire_dets  = [d for d in dets if "fire" in d["label"].lower()]
    smoke_dets = [d for d in dets if "smoke" in d["label"].lower()]
    fire_count  = float(len(fire_dets))
    smoke_count = float(len(smoke_dets))
    cluster_count = max(fire_count, smoke_count)

    def _union_area(ds):
        if not ds: return 0.0
        total = sum((d["x2"] - d["x1"]) * (d["y2"] - d["y1"]) for d in ds)
        return float(total) / (CAM_W * CAM_H)

    fua = _union_area(fire_dets)
    sua = _union_area(smoke_dets)
    scene_conf = max([d["conf"] for d in dets], default=0.0)

    # Composite label: 0=none,1=smoke,2=fire,3=fire_smoke
    comp = 0
    if fire_count > 0 and smoke_count > 0: comp = 3
    elif fire_count > 0:                   comp = 2
    elif smoke_count > 0:                  comp = 1
    # Scene label: 0=clear,1=hazy,2=smoky,3=fire
    scene_enc = 3 if fire_count > 0 else (2 if smoke_count > 0 else (1 if smoke_ppm > 150 else 0))

    return {
        "smoke_latest": sl, "smoke_avg": sa, "smoke_variance": sv,
        "smoke_velocity": svl, "smoke_acceleration": sac,
        "heat_grid_latest": hl, "heat_grid_avg": ha, "heat_grid_variance": hv,
        "heat_grid_velocity": hvl, "heat_grid_acceleration": hac,
        "fire_count": fire_count, "smoke_count": smoke_count,
        "cluster_count": cluster_count,
        "fire_union_area": fua, "smoke_union_area": sua,
        "scene_confidence": scene_conf,
        "composite_label_encoded": float(comp),
        "scene_label_encoded": float(scene_enc),
        "fire_union_area_velocity": 0.0,
        "smoke_union_area_velocity": 0.0,
        "glimpsed_fire": 1.0 if fire_count > 0 else 0.0,
        "human_near_fire": 0.0,
    }


def _predict(features: dict) -> int:
    vals = [features[k] for k in FEATURE_KEYS]
    pred = xgb_model.predict(np.array([vals]))[0]
    return int(pred) + 1   # shift back 0-4 → 1-5


# ─────────────────────────────────────────────────────────────────────────────
# UI renderer
# ─────────────────────────────────────────────────────────────────────────────
def _render(frame, heat_grid, dets, danger, smoke_ppm, err_x, err_y,
            hotspot_r, hotspot_c, hotspot_t, cam_on):
    canvas = np.zeros((UI_H, UI_W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # ── Banner ────────────────────────────────────────────────────────────────
    bg = DANGER_COLORS[danger]
    cv2.rectangle(canvas, (0, 0), (UI_W, BANNER_H), bg, -1)

    # Danger label
    label_txt = f"DANGER {danger}  |  {DANGER_LABELS[danger]}"
    cv2.putText(canvas, label_txt, (16, 44),
                cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)

    # Sensor summary top-right
    info = f"smoke={smoke_ppm:.0f}ppm  heat_max={hotspot_t:.1f}C  cam={'ON' if cam_on else 'OFF'}"
    cv2.putText(canvas, info, (UI_W - 480, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Left panel: camera feed ───────────────────────────────────────────────
    left_y0 = BANNER_H
    if frame is not None:
        disp = cv2.resize(frame, (PANEL_W, CONTENT_H))
        # Draw detections
        for d in dets:
            # scale box to display size
            sx = PANEL_W / CAM_W; sy = CONTENT_H / CAM_H
            x1 = int(d["x1"] * sx); y1 = int(d["y1"] * sy)
            x2 = int(d["x2"] * sx); y2 = int(d["y2"] * sy)
            is_fire = "fire" in d["label"].lower()
            color = (0, 60, 255) if is_fire else (0, 180, 255)
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            conf_txt = f"{d['label']} {d['conf']:.2f}"
            cv2.putText(disp, conf_txt, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if is_fire:
                cv2.putText(disp, f"danger:{danger}", (x1, max(y1 - 22, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        canvas[left_y0:left_y0 + CONTENT_H, 0:PANEL_W] = disp
    else:
        # No frame — show placeholder
        cv2.putText(canvas, "CAMERA OFF", (PANEL_W // 2 - 80, left_y0 + CONTENT_H // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (100, 100, 100), 2)

    # ── Right panel: heat grid ────────────────────────────────────────────────
    rx0 = PANEL_W
    # Divider
    cv2.line(canvas, (rx0, BANNER_H), (rx0, UI_H), (80, 80, 80), 2)

    # Title
    cv2.putText(canvas, "HEAT GRID  8x8  LIVE",
                (rx0 + GRID_X_OFFSET, BANNER_H + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Grid cells
    gy0 = BANNER_H + 40
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val = heat_grid[r][c]
            cx0 = rx0 + GRID_X_OFFSET + c * GRID_CELL
            cy0 = gy0 + r * GRID_CELL
            color = _heat_color(val)
            # Highlight hotspot
            is_hot = (r == hotspot_r and c == hotspot_c)
            cv2.rectangle(canvas,
                          (cx0 + 1, cy0 + 1),
                          (cx0 + GRID_CELL - 1, cy0 + GRID_CELL - 1),
                          color, -1)
            if is_hot:
                cv2.rectangle(canvas, (cx0, cy0),
                              (cx0 + GRID_CELL, cy0 + GRID_CELL),
                              (0, 0, 255), 2)
            cv2.putText(canvas, f"{val:.0f}",
                        (cx0 + 5, cy0 + GRID_CELL - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

    # err_x / err_y readout below grid
    info_y = gy0 + GRID_H + 20
    col_centered = "centered" if abs(err_x) < ARM_TOL else ("right" if err_x > 0 else "left")
    row_dir      = "centered" if abs(err_y) < ARM_TOL else ("fire below" if err_y > 0 else "fire above")
    ex_color = (60, 200, 60) if abs(err_x) < ARM_TOL else (0, 165, 255)
    ey_color = (60, 200, 60) if abs(err_y) < ARM_TOL else (0, 165, 255)
    cv2.putText(canvas, f"err_x (pan)  {err_x:+.3f}  {col_centered}",
                (rx0 + 12, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ex_color, 1)
    cv2.putText(canvas, f"err_y (tilt) {err_y:+.3f}  {row_dir}",
                (rx0 + 12, info_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ey_color, 1)
    cv2.putText(canvas,
                f"hotspot: row {hotspot_r}, col {hotspot_c}  {hotspot_t:.1f}C  "
                f"range {min(heat_grid[r][c] for r in range(8) for c in range(8)):.1f}-"
                f"{hotspot_t:.1f}C",
                (rx0 + 12, info_y + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Actions strip
    actions = POA_MAP.get(danger, ["monitor"])
    cv2.putText(canvas, "Actions: " + "  •  ".join(a.upper() for a in actions),
                (12, UI_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    cv2.putText(canvas, "Q=Quit  C=Toggle Camera",
                (rx0 + 12, UI_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
cam_always_on = False    # toggle with C key
dets          = []
frame         = None
danger        = 1
err_x = err_y = 0.0
hotspot_r = hotspot_c = 0
hotspot_t = 20.0
smoke_ppm = 0.0

print("[main] Loop running …")
try:
    while True:
        t0 = time.time()

        # ── 1. Read heat grid ─────────────────────────────────────────────────
        try:
            heat_grid = [list(row) for row in amg.pixels]   # 8×8 °C
        except Exception as e:
            print(f"[warn] AMG8833 read error: {e}")
            heat_grid = [[20.0] * 8 for _ in range(8)]

        arr = np.array(heat_grid, dtype=float)
        flat_idx  = int(arr.argmax())
        hotspot_r, hotspot_c = divmod(flat_idx, 8)
        hotspot_t = float(arr.max())

        cx = 3.5; cy = 3.5   # grid center (0-indexed)
        raw_err_x = (hotspot_c - cx) / cx
        raw_err_y = (hotspot_r - cy) / cy
        if HEAT_FLIP_X: raw_err_x = -raw_err_x
        if HEAT_FLIP_Y: raw_err_y = -raw_err_y
        # Add configured biases
        heat_bias = ARM_CFG["feedback"]["sensor_offsets"]["heat"]
        err_x = raw_err_x + heat_bias["x_bias"]
        err_y = raw_err_y + heat_bias["y_bias"]

        # ── 2. Read smoke sensor ──────────────────────────────────────────────
        try:
            raw_adc = float(smoke_ch.value)
            smoke_ppm = _raw_to_ppm(raw_adc)
        except Exception as e:
            print(f"[warn] ADS1115 read error: {e}")
            smoke_ppm = 0.0

        # ── 3. Sensor trigger check ───────────────────────────────────────────
        sensor_triggered = (smoke_ppm > SMOKE_THRESH) or (hotspot_t > HEAT_THRESH)

        # ── 4. Camera capture + detection ─────────────────────────────────────
        if cam_always_on or sensor_triggered:
            try:
                req      = picam2.capture_request()
                frame    = req.make_array("main")
                metadata = req.get_metadata()
                req.release()
                dets = _parse_detections(metadata)
            except Exception as e:
                print(f"[warn] Camera error: {e}")
                frame = None
                dets  = []
        else:
            frame = None
            dets  = []

        # ── 5. XGBoost prediction ─────────────────────────────────────────────
        features = _build_features(smoke_ppm, hotspot_t, dets)
        try:
            danger = _predict(features)
        except Exception as e:
            print(f"[warn] Predict error: {e}")

        # ── 6. Move arm (only if heat anomaly detected above demo threshold) ───
        if hotspot_t > ARM_HEAT_TRACK_THRESHOLD:
            _step_arm(err_x, err_y)

        # ── 7. Render UI ──────────────────────────────────────────────────────
        canvas = _render(frame, heat_grid, dets, danger, smoke_ppm,
                         err_x, err_y, hotspot_r, hotspot_c, hotspot_t,
                         cam_always_on or sensor_triggered)
        cv2.imshow("Smart Fire Extinguisher — POC", canvas)

        # ── Key handler ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("[main] Q pressed — quitting.")
            break
        if key == ord('c') or key == ord('C'):
            cam_always_on = not cam_always_on
            print(f"[main] Camera always-on: {cam_always_on}")

        # ── Timing ────────────────────────────────────────────────────────────
        elapsed_ms = (time.time() - t0) * 1000
        sleep_ms   = max(0, POLL_MS - elapsed_ms)
        time.sleep(sleep_ms / 1000.0)

except KeyboardInterrupt:
    print("\n[main] Interrupted.")
finally:
    pan_servo.value  = 0.0
    tilt_servo.value = 0.0
    pan_servo.close()
    tilt_servo.close()
    picam2.stop()
    cv2.destroyAllWindows()
    print("[main] Cleaned up. Bye!")