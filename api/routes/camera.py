# api/routes/camera.py
import os
import time
import logging
from flask import Blueprint, Response, send_file, current_app

logger = logging.getLogger(__name__)
camera_bp = Blueprint("camera", __name__)

# ── stream.jpg lives in RAM (tmpfs) — no SD card wear ────────────────────
STREAM_DIR  = "/dev/shm/fire_robot"
STREAM_PATH = os.path.join(STREAM_DIR, "stream.jpg")
FPS = 10

os.makedirs(STREAM_DIR, exist_ok=True)


@camera_bp.route("/api/camera/snapshot", methods=["GET"])
def snapshot():
    """Single JPEG frame — fallback for proxies that break MJPEG."""
    orch = current_app.config["ORCHESTRATOR"]
    if not orch.get_state_summary().get("camera_feed_active", False):
        return {"error": "camera is off"}, 403
    if not os.path.exists(STREAM_PATH):
        return {"error": "frame not ready — camera warming up"}, 503
    response = send_file(STREAM_PATH, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return response


@camera_bp.route("/api/camera/feed", methods=["GET"])
def feed():
    """
    MJPEG stream — the standard way to stream a Pi camera.

    Browser opens a single long-lived connection. Each frame is pushed as a
    multipart JPEG chunk. No polling, no caching issues, no intervals.

    camera_feed_active is set True on connect and False when the browser
    disconnects (finally block), so VisionFuser knows when to write frames.
    """
    orch = current_app.config["ORCHESTRATOR"]
    orch.set_camera_feed(True)

    def generate():
        try:
            while True:
                if os.path.exists(STREAM_PATH):
                    try:
                        with open(STREAM_PATH, "rb") as f:
                            frame_bytes = f.read()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + frame_bytes + b"\r\n"
                        )
                    except Exception as e:
                        logger.warning(f"camera feed read failed: {e}")
                time.sleep(1.0 / FPS)
        finally:
            orch.set_camera_feed(False)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )