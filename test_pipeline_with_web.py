#!/usr/bin/env python3
"""
test_pipeline_with_web.py
=========================
Same as test_pipeline.py but WITH a real-time web interface that shows
the annotated frames with bounding boxes over fires.

Run: python3 test_pipeline_with_web.py
Open browser: http://localhost:5001
"""

import sys
import os
import json
import time
import logging
import argparse
import threading
from pathlib import Path
from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_pipeline_web")

# ── Config ────────────────────────────────────────────────────────────────────
with open(PROJECT_ROOT / "configs" / "config.json") as f:
    cfg = json.load(f)
with open(PROJECT_ROOT / "configs" / "labels.json") as f:
    labels = {int(k): v for k, v in json.load(f).items()}

vision_cfg = cfg["vision"]
camera_cfg = vision_cfg["camera"]
fire_cfg = vision_cfg["models"]["fire"]

# ── Flask setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
current_frame_b64 = None
current_detections = []
frame_lock = threading.Lock()

# HTML Template with Canvas for drawing boxes
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔥 Fire Robot - YOLO Real-time Detection</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .header h1 {
            color: #ff6b6b;
            font-size: 2em;
            text-shadow: 0 0 10px rgba(255,107,107,0.5);
        }
        .header p {
            color: #888;
        }
        .main-panel {
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 20px;
        }
        .video-container {
            position: relative;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            border: 2px solid #ff6b6b;
        }
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        .overlay-text {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            padding: 8px 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 14px;
            color: #0f0;
            pointer-events: none;
        }
        .sidebar {
            background: rgba(30,30,50,0.9);
            border-radius: 12px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .stats-box {
            background: rgba(0,0,0,0.5);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .stats-box h3 {
            color: #ffd93d;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .stat-label {
            color: #aaa;
        }
        .stat-value {
            color: #ff6b6b;
            font-weight: bold;
        }
        .detection-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .detection-item {
            background: rgba(255,255,255,0.1);
            margin: 10px 0;
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #ff4444;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .detection-fire {
            border-left-color: #ff4444;
        }
        .detection-smoke {
            border-left-color: #888;
        }
        .confidence-bar {
            height: 4px;
            background: rgba(255,255,255,0.2);
            border-radius: 2px;
            margin-top: 5px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff4444, #ffaa44);
            border-radius: 2px;
            transition: width 0.2s ease;
        }
        .alert-banner {
            background: linear-gradient(90deg, #ff0000, #ff4444);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            margin-top: 20px;
            font-weight: bold;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            margin-left: 8px;
        }
        .badge-fire { background: #ff4444; }
        .badge-smoke { background: #888; }
        .fps-counter {
            color: #0f0;
            font-family: monospace;
            font-size: 14px;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🔥 FIRE ROBOT - YOLO REAL-TIME DETECTION</h1>
        <p>Model: fire_smoke_detection.rpk | Sony AI Pi Cam</p>
    </div>

    <div class="main-panel">
        <div class="video-container">
            <img id="videoFeed" src="/video_feed" alt="Camera Feed">
            <div class="overlay-text">
                <span class="fps-counter" id="fps">FPS: --</span> |
                <span id="detCount">Detections: 0</span>
            </div>
        </div>

        <div class="sidebar">
            <div class="stats-box">
                <h3>📊 Detection Statistics</h3>
                <div class="stat-row">
                    <span class="stat-label">Total Frames:</span>
                    <span class="stat-value" id="totalFrames">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Fire Detections:</span>
                    <span class="stat-value" id="fireCount">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Smoke Detections:</span>
                    <span class="stat-value" id="smokeCount">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Confidence Threshold:</span>
                    <span class="stat-value" id="confThresh">0</span>
                </div>
            </div>

            <div class="stats-box">
                <h3>🔥 Live Detections</h3>
                <div id="detectionList" class="detection-list">
                    <div style="text-align: center; color: #888;">Waiting for detections...</div>
                </div>
            </div>

            <div id="alertBanner" style="display: none;" class="alert-banner">
                🚨 FIRE DETECTED! 🚨
            </div>
        </div>
    </div>
</div>

<script>
    const socket = io();
    let fireTotal = 0;
    let smokeTotal = 0;

    socket.on('detection_update', (data) => {
        // Update counters
        document.getElementById('fps').innerHTML = `FPS: ${data.fps}`;
        document.getElementById('detCount').innerHTML = `Detections: ${data.count}`;
        document.getElementById('totalFrames').innerHTML = data.total_frames;
        document.getElementById('fireCount').innerHTML = data.fire_total;
        document.getElementById('smokeCount').innerHTML = data.smoke_total;

        // Update detection list
        const listDiv = document.getElementById('detectionList');
        if (data.detections.length > 0) {
            listDiv.innerHTML = '';
            data.detections.forEach((det, idx) => {
                const item = document.createElement('div');
                item.className = `detection-item detection-${det.label}`;
                item.innerHTML = `
                    <div style="display: flex; justify-content: space-between;">
                        <strong>${det.label.toUpperCase()}</strong>
                        <span>${(det.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${det.confidence * 100}%"></div>
                    </div>
                    <div style="font-size: 11px; color: #aaa; margin-top: 5px;">
                        Box: [${Math.round(det.bbox[0])}, ${Math.round(det.bbox[1])}, ${Math.round(det.bbox[2])}, ${Math.round(det.bbox[3])}]
                    </div>
                `;
                listDiv.appendChild(item);
            });

            // Show alert if fire detected
            const hasFire = data.detections.some(d => d.label === 'fire');
            const alertBanner = document.getElementById('alertBanner');
            if (hasFire) {
                alertBanner.style.display = 'block';
                setTimeout(() => {
                    if (!data.detections.some(d => d.label === 'fire')) {
                        alertBanner.style.display = 'none';
                    }
                }, 2000);
            }
        } else {
            listDiv.innerHTML = '<div style="text-align: center; color: #888;">No active detections</div>';
        }
    });

    console.log('WebSocket connected — waiting for detections');
</script>
</body>
</html>
'''


def draw_detections(frame, detections, clusters, frame_n, fire_total, smoke_total, conf_thresh):
    """Draw bounding boxes and labels on frame"""
    annotated = frame.copy()
    fh, fw = annotated.shape[:2]

    for det in detections:
        xc, yc, w, h = det.bbox
        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        x2 = int(xc + w / 2)
        y2 = int(yc + h / 2)

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)

        if det.label == "fire":
            color = (0, 0, 255)  # Red (BGR)
        else:
            color = (255, 128, 0)  # Blue-ish for smoke

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Draw label background
        label_text = f"{det.label.upper()} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - text_h - 8), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(annotated, label_text, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw cluster crosshairs
    for cl in clusters:
        cx = int(cl.origin_x * fw)
        cy = int(cl.origin_y * fh)
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # Draw stats overlay
    cv2.putText(annotated, f"Frame: {frame_n} | Fire: {fire_total} | Smoke: {smoke_total} | Conf: {conf_thresh}",
                (5, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return annotated


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint"""
    def generate():
        global current_frame_b64
        while True:
            if current_frame_b64:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       base64.b64decode(current_frame_b64) + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_pipeline(conf_thresh, max_frames):
    """Main detection pipeline - runs in background thread"""
    global current_frame_b64, current_detections

    from see.camera import IMX500Camera
    from see.models.fire_detector import FireDetector

    # Initialize camera
    camera = IMX500Camera(
        model_path=fire_cfg["rpk"],
        resolution=tuple(camera_cfg["resolution"]),
        fps=camera_cfg["fps"]
    )
    camera.start()
    logger.info("✓ Camera started")

    # Initialize detector
    detector = FireDetector(
        imx500=camera.imx500,
        conf_threshold=conf_thresh,
        labels=labels,
        picam2=camera.picam2
    )
    detector.load()
    logger.info("✓ FireDetector ready")

    frame_n = 0
    detect_frames = 0
    fire_total = 0
    smoke_total = 0
    fps = 0
    last_time = time.time()
    frame_count_fps = 0

    try:
        while True:
            result = camera.capture()
            if result is None:
                time.sleep(0.01)
                continue

            frame, metadata = result
            fh, fw = frame.shape[:2]
            frame_n += 1
            frame_count_fps += 1

            # Calculate FPS every second
            if time.time() - last_time >= 1.0:
                fps = frame_count_fps
                frame_count_fps = 0
                last_time = time.time()

            # Run detection
            clusters, raw_detections = detector.detect(
                metadata=metadata,
                frame_width=fw,
                frame_height=fh
            )

            # Update totals
            if raw_detections:
                detect_frames += 1
                fire_total += sum(1 for d in raw_detections if d.label == "fire")
                smoke_total += sum(1 for d in raw_detections if d.label == "smoke")
                logger.info(f"[frame {frame_n}] {len(raw_detections)} detections (fire={fire_total}, smoke={smoke_total})")

            # Draw annotations
            annotated = draw_detections(frame, raw_detections, clusters, frame_n, fire_total, smoke_total, conf_thresh)

            # Encode to JPEG and store globally
            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            current_frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

            # Send detection data via WebSocket
            detection_data = [
                {
                    'label': d.label,
                    'confidence': d.confidence,
                    'bbox': list(d.bbox)
                }
                for d in raw_detections
            ]
            socketio.emit('detection_update', {
                'detections': detection_data,
                'count': len(raw_detections),
                'total_frames': frame_n,
                'fire_total': fire_total,
                'smoke_total': smoke_total,
                'fps': fps
            })

            if max_frames > 0 and frame_n >= max_frames:
                break

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        camera.stop()
        logger.info(f"Final stats: {frame_n} frames, {detect_frames} with detections, {fire_total} fires, {smoke_total} smoke")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conf", type=float, default=fire_cfg["conf_threshold"],
                   help="Confidence threshold")
    p.add_argument("--frames", type=int, default=0,
                   help="Max frames (0 = infinite)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("=" * 60)
    logger.info("FIRE DETECTION PIPELINE WITH WEB UI")
    logger.info(f"  Confidence threshold: {args.conf}")
    logger.info(f"  Open browser at: http://localhost:5001")
    logger.info("=" * 60)

    # Start pipeline in background thread
    pipeline_thread = threading.Thread(
        target=run_pipeline,
        args=(args.conf, args.frames),
        daemon=True
    )
    pipeline_thread.start()

    # Start Flask server
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, use_reloader=False)