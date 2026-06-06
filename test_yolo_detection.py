"""
test_yolo_detection.py
======================
Isolated test for the IMX500 + RPK YOLO pipeline.

What this does
--------------
Runs the camera + model exactly as the real VisionFuser does it, but strips
everything else out (no SystemState, no queues, no arm, no database).

For every frame it prints to the terminal:
  - What get_outputs() returned (raw tensors, shapes, values)
  - How many boxes survived the confidence filter and why others were dropped
  - The label, confidence, and pixel coords of every surviving detection

It also opens a live OpenCV window showing the annotated frame so you can
see bounding boxes in real time. Press Q to quit.

This test answers:
  1. Is the RPK loading and producing outputs at all?
  2. Are the box coords in pixel space or normalized [0,1]?
  3. Is the conf_threshold eating all detections?
  4. Are class IDs matching labels.json?

Run on the Pi:
  cd ~/Desktop/fire_robot
  python3 tests/test_yolo_detection.py

Optional flags:
  --conf 0.3          lower threshold (default: 0.5 from config)
  --conf 0.1          try very low to see if ANY detections come through
  --no-window         headless mode — only terminal output, no cv2 window
"""

import json
import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Logging — every frame prints at INFO, raw tensor dumps at DEBUG
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_yolo")

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
def load_config():
    with open(PROJECT_ROOT / "configs" / "config.json") as f:
        cfg = json.load(f)
    with open(PROJECT_ROOT / "configs" / "labels.json") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return cfg, labels


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def run(conf_threshold: float, show_window: bool):
    cfg, labels = load_config()

    vision_cfg = cfg["vision"]
    camera_cfg = vision_cfg["camera"]
    rpk_path   = vision_cfg["models"]["fire"]["rpk"]

    resolution = tuple(camera_cfg["resolution"])   # (320, 240)
    fps        = camera_cfg["fps"]

    logger.info("=" * 60)
    logger.info("YOLO / RPK ISOLATION TEST")
    logger.info(f"  RPK path       : {rpk_path}")
    logger.info(f"  Resolution     : {resolution}")
    logger.info(f"  FPS            : {fps}")
    logger.info(f"  Conf threshold : {conf_threshold}")
    logger.info(f"  Labels         : {labels}")
    logger.info(f"  Show window    : {show_window}")
    logger.info("=" * 60)

    # ── Import hardware libs here so the file is importable on non-Pi too ──
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500

    # ── Step 1: Load RPK onto IMX500 ────────────────────────────────────────
    logger.info("Loading RPK onto IMX500 chip (may take a minute on first run)...")
    imx500 = IMX500(rpk_path)
    logger.info("RPK loaded OK")

    # ── Step 2: Start camera (AFTER imx500 init — order matters) ────────────
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": resolution, "format": "RGB888"},
        controls={"FrameRate": fps},
    )
    picam2.configure(config)
    picam2.start()
    logger.info("Camera started")

    frame_count  = 0
    none_count   = 0
    detect_count = 0

    try:
        while True:
            # ── Capture ─────────────────────────────────────────────────────
            request  = picam2.capture_request()
            frame    = request.make_array("main")
            metadata = request.get_metadata()
            request.release()

            frame_h, frame_w = frame.shape[:2]
            frame_area = frame_w * frame_h
            frame_count += 1

            # ── Get raw outputs from IMX500 ──────────────────────────────────
            outputs = imx500.get_outputs(metadata)

            if outputs is None:
                none_count += 1
                if none_count == 1 or none_count % 30 == 0:
                    logger.warning(
                        f"[frame {frame_count}] get_outputs() = None "
                        f"(none_count={none_count}) — model not producing output yet"
                    )
                if show_window:
                    annotated = frame.copy()
                    cv2.putText(annotated, "No outputs from IMX500", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("YOLO Test", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # ── Unpack tensors ───────────────────────────────────────────────
            # Standard IMX500 YOLO output: [boxes, scores, classes]
            # Each is a nested array — [0] gets the first (and only) batch
            try:
                boxes_raw   = outputs[0][0]
                scores_raw  = outputs[1][0]
                classes_raw = outputs[2][0]
            except (IndexError, TypeError) as e:
                logger.error(f"[frame {frame_count}] Failed to unpack outputs: {e}")
                logger.error(f"  outputs type={type(outputs)}, len={len(outputs) if outputs else 'N/A'}")
                for i, o in enumerate(outputs or []):
                    logger.error(f"  outputs[{i}] shape={np.asarray(o).shape} dtype={np.asarray(o).dtype}")
                continue

            boxes   = np.atleast_1d(boxes_raw)
            scores  = np.atleast_1d(scores_raw)
            classes = np.atleast_1d(classes_raw)

            # ── Print raw tensor snapshot every 30 frames ────────────────────
            if frame_count % 30 == 1:
                logger.info(f"[frame {frame_count}] RAW TENSORS:")
                logger.info(f"  boxes  shape={boxes.shape}   dtype={boxes.dtype}")
                logger.info(f"  scores shape={scores.shape}  dtype={scores.dtype}")
                logger.info(f"  classes shape={classes.shape} dtype={classes.dtype}")
                if len(boxes) > 0:
                    b0 = np.atleast_1d(boxes[0]).ravel()[:4]
                    s0 = float(np.atleast_1d(scores[0]).ravel()[0])
                    c0 = int(np.atleast_1d(classes[0]).ravel()[0])
                    logger.info(f"  FIRST BOX  : {[float(v) for v in b0]}")
                    logger.info(f"  FIRST SCORE: {s0:.4f}  (0-1 scale? or 0-100?)")
                    logger.info(f"  FIRST CLASS: {c0}  → label={labels.get(c0, 'UNKNOWN')}")
                    # KEY DIAGNOSTIC: are coords normalized or pixel?
                    if all(v <= 1.0 for v in b0):
                        logger.warning(
                            "  ⚠ Box values all ≤ 1.0 — coords look NORMALIZED not pixels!"
                            " Boxes will draw as 1px dots. Need imx500.convert_inference_coords()?"
                        )
                    else:
                        logger.info("  ✓ Box values > 1.0 — coords look like pixels, good")

            # ── Filter detections ────────────────────────────────────────────
            survived      = []
            dropped_conf  = 0
            dropped_label = 0

            for box, score, class_id in zip(boxes, scores, classes):
                score    = float(np.atleast_1d(score).ravel()[0])
                class_id = int(np.atleast_1d(class_id).ravel()[0])
                box      = np.atleast_1d(box).ravel()[:4]

                if score < conf_threshold:
                    dropped_conf += 1
                    continue

                if class_id not in labels or labels[class_id] == "other":
                    dropped_label += 1
                    continue

                x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                survived.append({
                    "label": labels[class_id],
                    "conf":  score,
                    "x": x, "y": y, "w": w, "h": h,
                    "class_id": class_id,
                })

            # ── Log every frame that has detections ──────────────────────────
            if survived:
                detect_count += 1
                logger.info(
                    f"[frame {frame_count}] DETECTIONS={len(survived)} "
                    f"(dropped_conf={dropped_conf} dropped_label={dropped_label})"
                )
                for d in survived:
                    logger.info(
                        f"  {d['label']:6s} conf={d['conf']:.3f} "
                        f"box=({d['x']:.1f}, {d['y']:.1f}, {d['w']:.1f}, {d['h']:.1f})"
                    )
            elif frame_count % 30 == 1:
                logger.info(
                    f"[frame {frame_count}] no detections "
                    f"(raw={len(boxes)} dropped_conf={dropped_conf} dropped_label={dropped_label})"
                )

            # ── Annotate frame ───────────────────────────────────────────────
            if show_window:
                annotated = frame.copy()

                for d in survived:
                    x, y, w, h = d["x"], d["y"], d["w"], d["h"]
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    color  = (0, 0, 255) if d["label"] == "fire" else (255, 128, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    txt = f"{d['label']} {d['conf']:.2f}"
                    cv2.putText(annotated, txt, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Status overlay
                status = (f"frame={frame_count} det={detect_count} "
                          f"conf_thr={conf_threshold} none={none_count}")
                cv2.putText(annotated, status, (5, frame_h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                cv2.imshow("YOLO Test — press Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Q pressed — quitting")
                    break

    finally:
        picam2.stop()
        if show_window:
            cv2.destroyAllWindows()
        logger.info(
            f"Done. frames={frame_count} with_detections={detect_count} "
            f"none_outputs={none_count}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated YOLO / RPK detection test")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold override (default: from config.json)")
    parser.add_argument("--no-window", action="store_true",
                        help="Headless mode — terminal output only, no OpenCV window")
    args = parser.parse_args()

    # Default conf from config if not overridden
    if args.conf is None:
        with open(PROJECT_ROOT / "configs" / "config.json") as f:
            args.conf = json.load(f)["vision"]["models"]["fire"]["conf_threshold"]

    run(conf_threshold=args.conf, show_window=not args.no_window)
