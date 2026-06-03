# api/routes/camera.py
import os
import time
import logging
from flask import Blueprint, Response, send_file, current_app

logger = logging.getLogger(__name__)
camera_bp = Blueprint("camera", __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LATEST_FRAME_PATH = os.path.join(BASE_DIR, "data/frames/stream.jpg")
FRAMES_DIR = os.path.join(BASE_DIR, "data/frames")
FPS = 10

# Ensure the frames directory exists at import time so the VisionFuser
# never has to race to create it before the first write.
os.makedirs(FRAMES_DIR, exist_ok=True)


@camera_bp.route("/api/camera/snapshot", methods=["GET"])
def snapshot():
    orch = current_app.config["ORCHESTRATOR"]
    # Do NOT call set_camera_feed(True) here — that's the toggle's job.
    # Calling it on every poll meant camera_feed_active was never cleared after
    # toggle-off: the frontend stopped polling so no more GETs arrived, but the
    # flag was already True from the last poll and nothing ever reset it.
    if not orch.get_state_summary().get("camera_feed_active", False):
        return {"error": "camera is off"}, 403
    if not os.path.exists(LATEST_FRAME_PATH):
        # 503 = camera is on but stream.jpg not written yet (warming up).
        return {"error": "frame not ready — camera warming up"}, 503
    response = send_file(LATEST_FRAME_PATH, mimetype="image/jpeg")
    # Defeat every layer of browser caching — without these, the browser serves
    # the same cached JPEG on every poll and the feed appears frozen.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return response


@camera_bp.route("/api/camera/feed", methods=["GET"])
def feed():
    orch = current_app.config["ORCHESTRATOR"]
    orch.set_camera_feed(True)

    def generate():
        try:
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
        finally:
            orch.set_camera_feed(False)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )