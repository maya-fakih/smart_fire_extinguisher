# api/routes/camera.py
"""
GET /api/camera/feed    → MJPEG stream from saved latest frame
GET /api/camera/snapshot → single JPEG of latest frame

NOTE: Assumes VisionFuser writes the latest frame to data/frames/latest.jpg.
If your VisionFuser saves frames with different naming, adjust LATEST_FRAME_PATH
or have it always overwrite a single 'latest.jpg' alongside its other saves.
"""

import os
import time
import logging
from flask import Blueprint, Response, send_file, current_app

logger = logging.getLogger(__name__)
camera_bp = Blueprint("camera", __name__)

LATEST_FRAME_PATH = "data/frames/stream.jpg"  # was latest.jpg
FPS = 10  # stream rate, frontend-friendly


@camera_bp.route("/api/camera/snapshot", methods=["GET"])
def snapshot():
    if not os.path.exists(LATEST_FRAME_PATH):
        return {"error": "no frame available"}, 404
    return send_file(LATEST_FRAME_PATH, mimetype="image/jpeg")


@camera_bp.route("/api/camera/feed", methods=["GET"])
def feed():
    """MJPEG stream — repeatedly serves the latest saved frame."""
    def generate():
        while True:
            if os.path.exists(LATEST_FRAME_PATH):
                try:
                    with open(LATEST_FRAME_PATH, "rb") as f:
                        frame_bytes = f.read()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_bytes + b"\r\n"
                    )
                except Exception as e:
                    logger.warning(f"camera feed read failed: {e}")
            time.sleep(1.0 / FPS)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )