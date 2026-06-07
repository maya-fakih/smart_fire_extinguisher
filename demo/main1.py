"""
demo/main.py  —  Smart Fire Extinguisher POC
=============================================
Architecture:
  - Picamera2 native preview  →  camera window with YOLO boxes drawn via pre_callback
  - Separate OpenCV window    →  heat grid + XGBoost danger banner

Keys: Q = quit,  C = toggle camera always-on
"""

import json, math, time, threading
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import xgboost as xgb

import board, busio
import adafruit_amg88xx
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Config
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
DEMO_DIR    = REPO_ROOT / "demo"
CONFIG_PATH = REPO_ROOT / "configs" / "config.json"
MODEL_PATH  = DEMO_DIR  / "model.json"
RPK_PATH    = str(REPO_ROOT / "model_weights" / "rpk" / "fire_smoke_detection.rpk")
LABELS_PATH = REPO_ROOT / "configs" / "labels.json"

with open(CONFIG_PATH) as f:
    CFG = json.load(f)

SMOKE_CFG = CFG["sensors"]["smoke"]
HEAT_CFG  = CFG["sensors"]["heat_grid"]
CAM_CFG   = CFG["vision"]["camera"]
ARM_CFG   = CFG["act"]["actuators"]["arm"]
THINK_CFG = CFG["think"]
POLL_MS   = CFG["system"]["polling_interval_active_ms"]

SMOKE_THRESH = SMOKE_CFG["threshold_physical"]
HEAT_THRESH  = HEAT_CFG["threshold_physical"]
CAM_W, CAM_H = CAM_CFG["resolution"]
CONF_THRESH  = CFG["vision"]["models"]["fire"]["conf_threshold"]

PAN_CFG  = ARM_CFG["joints"]["pan"]
TILT_CFG = ARM_CFG["joints"]["tilt"]
ARM_TOL  = ARM_CFG["feedback"]["tolerance_normalized"]
HEAT_FLIP_Y = ARM_CFG["feedback"]["heat_flip_y"]
HEAT_FLIP_X = ARM_CFG["feedback"]["heat_flip_x"]
POA_MAP = {int(k): v for k, v in THINK_CFG["poa_map"].items()}

# DEMO: candle tops out at 28-32°C, room baseline 24-27°C
ARM_HEAT_TRACK_THRESHOLD = 28.5   # °C

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

# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────
with open(LABELS_PATH) as f:
    _labels_raw = json.load(f)
if isinstance(_labels_raw, list):
    LABELS = {i: v for i, v in enumerate(_labels_raw)}
else:
    LABELS = {int(k): v for k, v in _labels_raw.items()}

# ─────────────────────────────────────────────────────────────────────────────
# UI constants (heat-grid window only)
# ─────────────────────────────────────────────────────────────────────────────
GRID_CELL  = 52
GRID_ROWS  = GRID_COLS = 8
GRID_W     = GRID_COLS * GRID_CELL   # 416
GRID_H     = GRID_ROWS * GRID_CELL   # 416
BANNER_H   = 80
WIN_W      = GRID_W + 80             # 496
WIN_H      = BANNER_H + GRID_H + 70  # 566

DANGER_COLORS = {
    1: (60, 180, 60),
    2: (30, 200, 220),
    3: (0, 165, 255),
    4: (0, 80, 255),
    5: (0, 0, 220),
}
DANGER_LABELS = {
    1: "SAFE",
    2: "LOW",
    3: "MEDIUM",
    4: "HIGH",
    5: "CRITICAL",
}

def _heat_color(val, vmin=20.0, vmax=35.0):
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    if t < 0.4:
        r = int(50 + t / 0.4 * 100)
        g = int(50 + t / 0.4 * 50)
        b = int(200 - t / 0.4 * 150)
    elif t < 0.7:
        s = (t - 0.4) / 0.3
        r = int(150 + s * 100); g = int(100 - s * 70); b = 50
    else:
        s = (t - 0.7) / 0.3
        r = 250; g = int(30 - s * 20); b = int(50 - s * 40)
    return (b, g, r)

# ─────────────────────────────────────────────────────────────────────────────
# Shared state (written by main loop, read by camera pre_callback)
# ─────────────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_state = {
    "danger": 1,
    "smoke_ppm": 0.0,
    "hotspot_t": 20.0,
    "dets": [],
    "cam_on": True,   # always on by default
}

def _get_state():
    with _state_lock:
        return dict(_state)

def _set_state(**kw):
    with _state_lock:
        _state.update(kw)

# ─────────────────────────────────────────────────────────────────────────────
# Smoke helper
# ─────────────────────────────────────────────────────────────────────────────
def _raw_to_ppm(raw):
    p = SMOKE_CFG["eq_params"]
    RL, Ro, A, B = p["RL"], p["Ro"], p["A"], p["B"]
    raw = max(1.0, raw)
    ratio = (3.3 - (raw / 32767.0) * 3.3) / max(1e-9, (raw / 32767.0) * 3.3)
    return float(A * (ratio * RL / Ro) ** B)

# ─────────────────────────────────────────────────────────────────────────────
# Hardware init
# ─────────────────────────────────────────────────────────────────────────────
print("[init] I2C sensors …")
i2c = busio.I2C(board.SCL, board.SDA)
time.sleep(0.5)
amg = adafruit_amg88xx.AMG88XX(i2c, addr=int(HEAT_CFG["address"], 16))
time.sleep(0.5)

ads = ADS.ADS1115(i2c, address=int(SMOKE_CFG["address"], 16))
ads.gain = SMOKE_CFG["gain"]
smoke_ch = AnalogIn(ads, SMOKE_CFG["channel"])

_factory = LGPIOFactory()
pan_servo  = Servo(pin=int(PAN_CFG["pin"]),  pin_factory=_factory)
tilt_servo = Servo(pin=int(TILT_CFG["pin"]), pin_factory=_factory)
pan_servo.value  = 0.0
tilt_servo.value = 0.0
pan_angle  = 0.0
tilt_angle = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# IMX500 camera (native preview pattern from imx500_object_detection_demo.py)
# ─────────────────────────────────────────────────────────────────────────────
print("[init] IMX500 camera + RPK …")
imx500 = IMX500(RPK_PATH)
intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
intrinsics.task = "object detection"
intrinsics.update_with_defaults()

picam2 = Picamera2(imx500.camera_num)
cam_config = picam2.create_preview_configuration(
    controls={"FrameRate": intrinsics.inference_rate},
    buffer_count=12,
)
imx500.show_network_fw_progress_bar()

_last_dets_cam = []   # detections drawn by pre_callback

def _parse_dets(metadata):
    """Parse IMX500 output → list of Detection-like dicts with pixel coords."""
    global _last_dets_cam
    try:
        np_outputs = imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return _last_dets_cam
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if intrinsics.bbox_normalization:
            iw, ih = imx500.get_input_size()
            boxes = boxes / ih
        if intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        dets = []
        for box, sc, cls in zip(boxes, scores, classes):
            if sc < CONF_THRESH:
                continue
            # convert_inference_coords gives pixel (x,y,w,h) in ISP frame
            coords = imx500.convert_inference_coords(box, metadata, picam2)
            label = LABELS.get(int(cls), str(int(cls)))
            dets.append({"label": label, "conf": float(sc), "box": coords})
        _last_dets_cam = dets
        return dets
    except Exception:
        return _last_dets_cam


def _draw_on_camera(request):
    """pre_callback: draw boxes + danger banner onto the live preview frame."""
    s = _get_state()
    dets    = s["dets"]
    danger  = s["danger"]
    color   = DANGER_COLORS[danger]
    d_label = DANGER_LABELS[danger]

    with MappedArray(request, "main") as m:
        arr = m.array
        h, w = arr.shape[:2]

        # ── Banner across top of camera frame ────────────────────────────────
        overlay = arr.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), color, -1)
        cv2.addWeighted(overlay, 0.75, arr, 0.25, 0, arr)
        txt = f"DANGER {danger}  {d_label}  |  smoke={s['smoke_ppm']:.0f}ppm  heat={s['hotspot_t']:.1f}C"
        cv2.putText(arr, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Bounding boxes ────────────────────────────────────────────────────
        for d in dets:
            x, y, bw, bh = d["box"]
            is_fire = "fire" in d["label"].lower()
            clr = (0, 60, 255) if is_fire else (0, 200, 100)
            cv2.rectangle(arr, (x, y), (x + bw, y + bh), clr, 2)
            conf_txt = f"{d['label']} {d['conf']:.2f}"
            cv2.putText(arr, conf_txt, (x + 4, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, clr, 1, cv2.LINE_AA)
            if is_fire:
                cv2.putText(arr, f"danger:{danger}", (x + 4, y + bh - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # ── Vertical flip correction (camera is mounted upside-down) ─────────
        # Note: MappedArray is in-place, flip rows
        arr[:] = arr[::-1, :, :]


picam2.pre_callback = _draw_on_camera
picam2.start(cam_config, show_preview=True)
print("[init] Camera running.")

# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────
print("[init] XGBoost …")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(MODEL_PATH))
print("[init] Ready.")

# ─────────────────────────────────────────────────────────────────────────────
# Servo helpers
# ─────────────────────────────────────────────────────────────────────────────
def _deg_to_val(deg, smin, smax):
    span = smax - smin
    return max(-1.0, min(1.0, 2.0 * (deg - smin) / span - 1.0))

def _clip(v, lo, hi): return max(lo, min(hi, v))

def _step_arm(err_x, err_y):
    global pan_angle, tilt_angle
    if abs(err_x) > ARM_TOL / 2:
        sign = (1 if err_x > 0 else -1) * (-1 if PAN_CFG.get("invert", False) else 1)
        pan_angle = _clip(pan_angle + sign * PAN_CFG["step_deg"],
                          PAN_CFG["limit_min_deg"], PAN_CFG["limit_max_deg"])
        pan_servo.value = _deg_to_val(pan_angle, PAN_CFG["servo_min_deg"], PAN_CFG["servo_max_deg"])
    if abs(err_y) > ARM_TOL / 2:
        sign = (1 if err_y > 0 else -1) * (-1 if TILT_CFG.get("invert", False) else 1)
        tilt_angle = _clip(tilt_angle + sign * TILT_CFG["step_deg"],
                           TILT_CFG["limit_min_deg"], TILT_CFG["limit_max_deg"])
        tilt_servo.value = _deg_to_val(tilt_angle, TILT_CFG["servo_min_deg"], TILT_CFG["servo_max_deg"])

# ─────────────────────────────────────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────────────────────────────────────
_smoke_hist, _heat_hist = [], []
_HIST_LEN = 5

def _upd(lst, v):
    lst.append(v)
    if len(lst) > _HIST_LEN: lst.pop(0)

def _stats(lst):
    arr = np.array(lst, dtype=float)
    vel = float(arr[-1] - arr[0]) / max(len(arr) - 1, 1)
    acc = 0.0
    if len(arr) >= 3:
        d = np.diff(arr); acc = float(d[-1] - d[0]) / max(len(d) - 1, 1)
    return float(arr[-1]), float(arr.mean()), float(arr.var() if len(arr) >= 2 else 0), vel, acc

def _build_features(smoke_ppm, heat_max, dets):
    _upd(_smoke_hist, smoke_ppm); _upd(_heat_hist, heat_max)
    sl, sa, sv, svl, sac = _stats(_smoke_hist)
    hl, ha, hv, hvl, hac = _stats(_heat_hist)
    fd = [d for d in dets if "fire"  in d["label"].lower()]
    sd = [d for d in dets if "smoke" in d["label"].lower()]
    fc, sc2 = float(len(fd)), float(len(sd))
    def ua(ds): return sum((d["box"][2]*d["box"][3]) for d in ds) / (CAM_W*CAM_H) if ds else 0.0
    fua, sua = ua(fd), ua(sd)
    sconf = max([d["conf"] for d in dets], default=0.0)
    comp = 3 if fc>0 and sc2>0 else (2 if fc>0 else (1 if sc2>0 else 0))
    senc = 3 if fc>0 else (2 if sc2>0 else (1 if smoke_ppm>150 else 0))
    return {
        "smoke_latest":sl,"smoke_avg":sa,"smoke_variance":sv,"smoke_velocity":svl,"smoke_acceleration":sac,
        "heat_grid_latest":hl,"heat_grid_avg":ha,"heat_grid_variance":hv,"heat_grid_velocity":hvl,"heat_grid_acceleration":hac,
        "fire_count":fc,"smoke_count":sc2,"cluster_count":max(fc,sc2),
        "fire_union_area":fua,"smoke_union_area":sua,"scene_confidence":sconf,
        "composite_label_encoded":float(comp),"scene_label_encoded":float(senc),
        "fire_union_area_velocity":0.0,"smoke_union_area_velocity":0.0,
        "glimpsed_fire":1.0 if fc>0 else 0.0,"human_near_fire":0.0,
    }

def _predict(feats):
    vals = [feats[k] for k in FEATURE_KEYS]
    return int(xgb_model.predict(np.array([vals]))[0]) + 1

# ─────────────────────────────────────────────────────────────────────────────
# Heat-grid window renderer
# ─────────────────────────────────────────────────────────────────────────────
def _render_heat(heat_grid, danger, smoke_ppm, err_x, err_y,
                 hotspot_r, hotspot_c, hotspot_t):
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)

    # Banner
    bg = DANGER_COLORS[danger]
    cv2.rectangle(canvas, (0, 0), (WIN_W, BANNER_H), bg, -1)
    cv2.putText(canvas, f"DANGER {danger}  —  {DANGER_LABELS[danger]}",
                (10, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"smoke={smoke_ppm:.0f}ppm   heat_max={hotspot_t:.1f}C",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # Grid title
    cv2.putText(canvas, "AMG8833  8x8  HEAT GRID",
                (10, BANNER_H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

    GX0 = (WIN_W - GRID_W) // 2
    GY0 = BANNER_H + 28

    # NOTE: AMG is mounted inverted vertically → flip rows for display
    # (heat_flip_y=True in config means physical rows are inverted)
    display_grid = heat_grid[::-1] if HEAT_FLIP_Y else heat_grid

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val = display_grid[r][c]
            cx0 = GX0 + c * GRID_CELL
            cy0 = GY0 + r * GRID_CELL
            clr = _heat_color(val)
            is_hot = (r == (7 - hotspot_r if HEAT_FLIP_Y else hotspot_r) and c == hotspot_c)
            cv2.rectangle(canvas, (cx0+1, cy0+1), (cx0+GRID_CELL-1, cy0+GRID_CELL-1), clr, -1)
            if is_hot:
                cv2.rectangle(canvas, (cx0, cy0), (cx0+GRID_CELL, cy0+GRID_CELL), (0,0,255), 3)
            cv2.putText(canvas, f"{val:.0f}",
                        (cx0+5, cy0+GRID_CELL-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255,255,255), 1)

    # err readout
    iy = GY0 + GRID_H + 10
    ex_clr = (60,200,60) if abs(err_x)<ARM_TOL else (0,165,255)
    ey_clr = (60,200,60) if abs(err_y)<ARM_TOL else (0,165,255)
    xdir = "centered" if abs(err_x)<ARM_TOL else ("→ right" if err_x>0 else "← left")
    ydir = "centered" if abs(err_y)<ARM_TOL else ("↓ below" if err_y>0 else "↑ above")
    cv2.putText(canvas, f"pan  err_x={err_x:+.3f}  {xdir}",
                (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ex_clr, 1)
    cv2.putText(canvas, f"tilt err_y={err_y:+.3f}  {ydir}",
                (10, iy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, ey_clr, 1)
    acts = POA_MAP.get(danger, ["monitor"])
    cv2.putText(canvas, "Actions: " + " • ".join(a.upper() for a in acts),
                (10, iy+42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    cv2.putText(canvas, "Q=Quit", (WIN_W-80, iy+42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)

    return canvas

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
danger = 1; err_x = err_y = 0.0
hotspot_r = hotspot_c = 0; hotspot_t = 20.0; smoke_ppm = 0.0

print("[main] Running. Press Q in heat-grid window to quit.")
try:
    while True:
        t0 = time.time()

        # 1. Heat grid
        try:
            heat_grid = [list(row) for row in amg.pixels]
        except Exception as e:
            print(f"[warn] AMG: {e}")
            heat_grid = [[20.0]*8 for _ in range(8)]

        arr = np.array(heat_grid, dtype=float)
        flat_idx = int(arr.argmax())
        hotspot_r, hotspot_c = divmod(flat_idx, 8)
        hotspot_t = float(arr.max())

        cx, cy = 3.5, 3.5
        raw_ex = (hotspot_c - cx) / cx
        raw_ey = (hotspot_r - cy) / cy
        if HEAT_FLIP_X: raw_ex = -raw_ex
        if HEAT_FLIP_Y: raw_ey = -raw_ey
        bias = ARM_CFG["feedback"]["sensor_offsets"]["heat"]
        err_x = raw_ex + bias["x_bias"]
        err_y = raw_ey + bias["y_bias"]

        # 2. Smoke
        try:
            smoke_ppm = _raw_to_ppm(float(smoke_ch.value))
        except Exception as e:
            print(f"[warn] ADS: {e}")
            smoke_ppm = 0.0

        # 3. Camera detections (grab latest from pre_callback list)
        dets = list(_last_dets_cam)

        # 4. XGBoost
        feats = _build_features(smoke_ppm, hotspot_t, dets)
        try:
            danger = _predict(feats)
        except Exception as e:
            print(f"[warn] predict: {e}")

        # 5. Push state for camera pre_callback overlay
        _set_state(danger=danger, smoke_ppm=smoke_ppm, hotspot_t=hotspot_t, dets=dets)

        # 6. Arm tracking
        if hotspot_t > ARM_HEAT_TRACK_THRESHOLD:
            _step_arm(err_x, err_y)

        # 7. Capture metadata so detections stay fresh (pre_callback already drew)
        try:
            meta = picam2.capture_metadata()
            _parse_dets(meta)
        except Exception:
            pass

        # 8. Heat grid window
        canvas = _render_heat(heat_grid, danger, smoke_ppm, err_x, err_y,
                              hotspot_r, hotspot_c, hotspot_t)
        cv2.imshow("Heat Grid + XGBoost", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            print("[main] Quit.")
            break

        elapsed = (time.time() - t0) * 1000
        time.sleep(max(0, POLL_MS - elapsed) / 1000.0)

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
