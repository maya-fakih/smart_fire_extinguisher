"""
test_yolo_raw_stream.py
=======================
Writes ALL raw IMX500 YOLO outputs directly to /dev/shm/fire_robot/stream.jpg
with zero filtering — every slot the chip returns gets drawn, regardless of
confidence or class. View it on the dashboard camera feed.

Run:
    cd ~/Desktop/fire_robot
    python3 tests/test_yolo_raw_stream.py
"""
import json, sys, time, os, logging
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("raw_stream")

STREAM_DIR  = "/dev/shm/fire_robot"
STREAM_PATH = os.path.join(STREAM_DIR, "stream.jpg")
os.makedirs(STREAM_DIR, exist_ok=True)

with open(PROJECT_ROOT / "configs" / "config.json") as f:
    cfg = json.load(f)
with open(PROJECT_ROOT / "configs" / "labels.json") as f:
    labels = {int(k): v for k, v in json.load(f).items()}

rpk        = cfg["vision"]["models"]["fire"]["rpk"]
resolution = tuple(cfg["vision"]["camera"]["resolution"])
fps        = cfg["vision"]["camera"]["fps"]

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

logger.info(f"Loading RPK: {rpk}")
imx500 = IMX500(rpk)
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": resolution, "format": "RGB888"},
    controls={"FrameRate": fps}
))
picam2.start()
logger.info("Camera started — writing raw YOLO output to stream.jpg")
logger.info("Open the dashboard camera feed to view. Ctrl+C to stop.")

frame_n = 0
try:
    while True:
        req    = picam2.capture_request()
        frame  = req.make_array("main")
        meta   = req.get_metadata()
        req.release()
        frame_n += 1

        outputs = imx500.get_outputs(meta)
        annotated = frame.copy()

        if outputs is not None:
            n_valid  = int(np.asarray(outputs[3]).ravel()[0])
            boxes    = np.asarray(outputs[0])   # ALL 300 slots, no slice
            scores   = np.asarray(outputs[1])
            classes  = np.asarray(outputs[2])
            fh, fw   = frame.shape[:2]

            # Draw every slot with score > 0 — no conf threshold at all
            drawn = 0
            for i in range(len(boxes)):
                score = float(scores[i])
                if score <= 0:
                    continue
                cid   = int(classes[i])
                label = labels.get(cid, f"cls{cid}")
                x1, y1, x2, y2 = [float(v) for v in boxes[i][:4]]
                color = (0,0,255) if label=="fire" else (255,128,0) if label=="smoke" else (200,200,200)
                cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                cv2.putText(annotated, f"{label} {score:.2f}",
                            (int(x1), max(int(y1)-6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                drawn += 1

            if frame_n % 30 == 1 or drawn > 0:
                logger.info(f"frame={frame_n} n_valid={n_valid} drawn(score>0)={drawn}")
        else:
            if frame_n % 30 == 1:
                logger.info(f"frame={frame_n} outputs=None")

        # Write to stream.jpg atomically
        tmp = STREAM_PATH + ".tmp"
        cv2.imwrite(tmp, annotated, [cv2.IMWRITE_JPEG_QUALITY, 55])
        os.replace(tmp, STREAM_PATH)

finally:
    picam2.stop()
    logger.info("Done.")
