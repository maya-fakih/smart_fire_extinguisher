"""
test_pipeline.py
================
End-to-end pipeline test — runs the full SEE layer (camera → YOLO →
FireDetector → VisionSnapshot) in isolation, with no orchestrator,
no sensor fuser, no database, no arm.

What this proves:
  ✓ IMX500 + RPK loads and produces outputs
  ✓ FireDetector parses and filters detections correctly
  ✓ Bounding boxes are in the right coordinate space
  ✓ VisionSnapshot is assembled correctly
  ✓ Logs appear (child logging fix works)
  ✓ The stream.jpg is written so you can see it on the dashboard

Run on the Pi:
  cd ~/Desktop/fire_robot
  python3 test_pipeline.py

Optional:
  python3 test_pipeline.py --conf 0.1    # lower threshold to catch weak detections
  python3 test_pipeline.py --frames 60   # run for 60 frames then exit
"""

import sys, os, json, time, logging, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(name)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/test_pipeline.log"),
    ]
)
logger = logging.getLogger("test_pipeline")

# ── Config ────────────────────────────────────────────────────────────────────
with open(PROJECT_ROOT / "configs" / "config.json") as f:
    cfg = json.load(f)
with open(PROJECT_ROOT / "configs" / "labels.json") as f:
    labels = {int(k): v for k, v in json.load(f).items()}

vision_cfg = cfg["vision"]
camera_cfg = vision_cfg["camera"]
fire_cfg   = vision_cfg["models"]["fire"]

STREAM_DIR  = "/dev/shm/fire_robot"
STREAM_PATH = os.path.join(STREAM_DIR, "stream.png")  # PNG always works on all OpenCV builds
os.makedirs(STREAM_DIR, exist_ok=True)

import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conf",   type=float, default=fire_cfg["conf_threshold"],
                   help="Confidence threshold (default from config)")
    p.add_argument("--frames", type=int,   default=0,
                   help="Stop after N frames (0 = run forever)")
    return p.parse_args()


def draw_detections(frame, detections, clusters):
    """Draw YOLO boxes and cluster crosshairs onto a copy of the frame."""
    annotated = frame.copy()
    fh, fw = annotated.shape[:2]

    for det in detections:
        xc, yc, w, h = det.bbox
        x1, y1 = int(xc - w/2), int(yc - h/2)
        x2, y2 = int(xc + w/2), int(yc + h/2)
        color  = (0, 0, 255) if det.label == "fire" else (255, 128, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{det.label} {det.confidence:.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for cl in clusters:
        cx = int(cl.origin_x * fw)
        cy = int(cl.origin_y * fh)
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
        cv2.putText(annotated, f"cluster:{cl.composite_label} area={cl.total_area_ratio:.3f}",
                    (cx + 14, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    return annotated


def run():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("FULL PIPELINE TEST (SEE layer in isolation)")
    logger.info(f"  RPK          : {fire_cfg['rpk']}")
    logger.info(f"  Resolution   : {camera_cfg['resolution']}")
    logger.info(f"  FPS          : {camera_cfg['fps']}")
    logger.info(f"  Conf thresh  : {args.conf}")
    logger.info(f"  Max frames   : {'∞' if args.frames == 0 else args.frames}")
    logger.info(f"  Stream path  : {STREAM_PATH}")
    logger.info("=" * 60)

    # ── Check 1: RPK file exists ──────────────────────────────────────────────
    rpk = fire_cfg["rpk"]
    if not os.path.exists(rpk):
        logger.critical(f"FAIL: RPK not found at {rpk}")
        sys.exit(1)
    logger.info(f"✓ RPK exists: {rpk}")

    # ── Check 2: labels.json sane ─────────────────────────────────────────────
    logger.info(f"✓ Labels loaded: {labels}")
    fire_ids  = [k for k,v in labels.items() if v == "fire"]
    smoke_ids = [k for k,v in labels.items() if v == "smoke"]
    if not fire_ids:
        logger.warning("⚠  No 'fire' class in labels.json — detections will all be dropped!")
    else:
        logger.info(f"✓ Fire class IDs: {fire_ids}  Smoke class IDs: {smoke_ids}")

    # ── Import hardware-dependent modules ─────────────────────────────────────
    logger.info("Importing picamera2 + IMX500...")
    from see.camera      import IMX500Camera
    from see.models.fire_detector import FireDetector
    from see.snapshot    import VisionSnapshot
    from datetime        import datetime

    # ── Check 3: Camera starts ────────────────────────────────────────────────
    logger.info("Starting IMX500Camera (RPK loads onto chip — may take ~60s first time)...")
    camera = IMX500Camera(
        model_path = rpk,
        resolution = tuple(camera_cfg["resolution"]),
        fps        = camera_cfg["fps"],
    )
    camera.start()
    logger.info("✓ Camera started")

    # ── Check 4: FireDetector builds ─────────────────────────────────────────
    detector = FireDetector(
        imx500         = camera.imx500,
        conf_threshold = args.conf,
        labels         = labels,
        picam2         = camera.picam2,
    )
    detector.load()
    logger.info("✓ FireDetector built")

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_n        = 0
    detect_frames  = 0
    none_frames    = 0
    total_fire     = 0
    total_smoke    = 0

    logger.info("Starting capture loop. Watch stream.jpg on dashboard feed. Ctrl+C to stop.")

    try:
        while True:
            result = camera.capture()
            if result is None:
                logger.warning("camera.capture() returned None — camera not active?")
                time.sleep(0.1)
                continue

            frame, metadata = result
            fh, fw = frame.shape[:2]
            frame_n += 1

            # ── Run YOLO via FireDetector ──────────────────────────────────────
            clusters, raw_detections = detector.detect(
                metadata     = metadata,
                frame_width  = fw,
                frame_height = fh,
            )

            # ── Log results ───────────────────────────────────────────────────
            if raw_detections:
                detect_frames += 1
                total_fire  += sum(1 for d in raw_detections if d.label == "fire")
                total_smoke += sum(1 for d in raw_detections if d.label == "smoke")
                logger.info(
                    f"[frame {frame_n}] ✓ DETECTIONS: {len(raw_detections)} "
                    f"(fire={sum(1 for d in raw_detections if d.label=='fire')} "
                    f"smoke={sum(1 for d in raw_detections if d.label=='smoke')}) "
                    f"clusters={len(clusters)}"
                )
                for cl in clusters:
                    logger.info(
                        f"  cluster → {cl.composite_label} "
                        f"area={cl.total_area_ratio:.3f} "
                        f"origin=({cl.origin_x:.3f}, {cl.origin_y:.3f}) "
                        f"conf={cl.primary_confidence:.3f}"
                    )
            elif frame_n % 30 == 1:
                logger.info(f"[frame {frame_n}] no detections  (total_frames={frame_n} detect_frames={detect_frames})")

            # ── Write annotated stream.jpg ────────────────────────────────────
            annotated = draw_detections(frame, raw_detections, clusters)
            # Overlay stats
            cv2.putText(
                annotated,
                f"frame={frame_n} det={detect_frames} fire={total_fire} smoke={total_smoke} conf={args.conf}",
                (5, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )
            tmp = STREAM_PATH + ".tmp.png"
            cv2.imwrite(tmp, annotated)  # PNG — no encoder issues
            os.replace(tmp, STREAM_PATH)

            if args.frames > 0 and frame_n >= args.frames:
                logger.info(f"Reached --frames {args.frames} — stopping.")
                break

    except KeyboardInterrupt:
        logger.info("Ctrl+C — stopping.")
    finally:
        camera.stop()
        logger.info("=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info(f"  Total frames     : {frame_n}")
        logger.info(f"  Frames with dets : {detect_frames}  ({100*detect_frames/max(frame_n,1):.1f}%)")
        logger.info(f"  Total fire boxes : {total_fire}")
        logger.info(f"  Total smoke boxes: {total_smoke}")
        if detect_frames == 0:
            logger.warning("⚠  ZERO detections in entire run. Check:")
            logger.warning("   1. Is there visible fire/smoke in camera view?")
            logger.warning("   2. Try --conf 0.05 to see if anything comes through")
            logger.warning("   3. Run test_yolo_detection.py to check raw tensor values")
            logger.warning("   4. Verify labels.json class IDs match the RPK model")
        else:
            logger.info("✓ Pipeline working end-to-end!")
        logger.info("=" * 60)


if __name__ == "__main__":
    run()