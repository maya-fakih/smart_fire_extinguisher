#!/usr/bin/env python3
"""
Simple Web Viewer for YOLO Fire Detection
No Flask-SocketIO - just pure MJPEG streaming
"""

import sys
import os
import json
import time
import threading
import base64
import cv2
import numpy as np
from flask import Flask, Response, render_template_string

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(PROJECT_ROOT + "/src"))

# HTML template with no WebSocket - just refreshing image
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="2">
    <title>🔥 Fire Robot - YOLO Detection</title>
    <style>
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            font-family: 'Courier New', monospace;
            color: #fff;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            text-align: center;
        }
        h1 {
            color: #ff4444;
            text-shadow: 0 0 10px rgba(255,68,68,0.5);
        }
        .video-box {
            background: #000;
            border: 3px solid #ff4444;
            border-radius: 10px;
            padding: 5px;
            margin: 20px 0;
        }
        img {
            width: 100%;
            border-radius: 5px;
        }
        .stats {
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            text-align: left;
            font-family: monospace;
            font-size: 14px;
        }
        .fire-alert {
            background: #ff0000;
            color: white;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            font-weight: bold;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-fire { background: #ff0000; }
        button {
            background: #ff4444;
            border: none;
            padding: 10px 20px;
            color: white;
            font-weight: bold;
            margin: 5px;
            cursor: pointer;
            border-radius: 5px;
        }
        button:hover { background: #cc0000; }
    </style>
</head>
<body>
<div class="container">
    <h1>🔥 FIRE ROBOT - YOLO REAL-TIME DETECTION</h1>
    <p>Sony AI Pi Cam | YOLO .rpk | Confidence Threshold: {{ conf_thresh }}</p>
    
    <div class="video-box">
        <img id="videoFeed" src="/video_feed" alt="Camera Feed">
    </div>
    
    <div id="fireAlert" style="display: none;" class="fire-alert">
        🚨🚨🚨 FIRE DETECTED! 🚨🚨🚨
    </div>
    
    <div class="stats">
        <strong>📊 Detection Statistics (from server):</strong><br>
        <span id="stats"></span>
    </div>
    
    <div>
        <button onclick="location.reload()">🔄 Refresh</button>
        <button onclick="window.location.href='/snapshot'">📸 Take Snapshot</button>
    </div>
</div>

<script>
    // Poll for stats every 2 seconds (no WebSocket needed)
    function updateStats() {
        fetch('/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('stats').innerHTML = 
                    `Total Frames: ${data.total_frames}<br>
                     Fire Detections: ${data.fire_total}<br>
                     Frames with Fire: ${data.detect_frames}<br>
                     Detection Rate: ${data.detection_rate}%<br>
                     Current FPS: ${data.fps}`;
                
                const alertDiv = document.getElementById('fireAlert');
                if (data.has_fire) {
                    alertDiv.style.display = 'block';
                } else {
                    alertDiv.style.display = 'none';
                }
            })
            .catch(err => console.error(err));
    }
    
    updateStats();
    setInterval(updateStats, 2000);
</script>
</body>
</html>
'''

# Global variables
current_frame_b64 = None
detection_stats = {
    'total_frames': 0,
    'detect_frames': 0,
    'fire_total': 0,
    'fps': 0,
    'has_fire': False
}

app = Flask(__name__)

def draw_detections(frame, detections, clusters, frame_n, fire_total, conf_thresh):
    """Draw bounding boxes on frame"""
    annotated = frame.copy()
    fh, fw = annotated.shape[:2]
    
    for det in detections:
        xc, yc, w, h = det.bbox
        x1 = int(xc - w/2)
        y1 = int(yc - h/2)
        x2 = int(xc + w/2)
        y2 = int(yc + h/2)
        
        # Clamp to frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        
        if det.label == "fire":
            color = (0, 0, 255)  # Red
        else:
            color = (255, 128, 0)  # Blue-ish
        
        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label = f"{det.label.upper()} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw cluster crosshairs
    for cl in clusters:
        cx = int(cl.origin_x * fw)
        cy = int(cl.origin_y * fh)
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 0), 
                      markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    
    # Draw stats
    cv2.putText(annotated, f"Frame: {frame_n} | Fire: {fire_total} | Conf: {conf_thresh} | FPS: {detection_stats['fps']}", 
               (5, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    return annotated

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, conf_thresh=args.conf if 'args' in dir() else 0.2)

@app.route('/video_feed')
def video_feed():
    def generate():
        global current_frame_b64
        while True:
            if current_frame_b64:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       base64.b64decode(current_frame_b64) + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return detection_stats

@app.route('/snapshot')
def snapshot():
    if current_frame_b64:
        import base64
        img_data = base64.b64decode(current_frame_b64)
        return Response(img_data, mimetype='image/jpeg')
    return {"error": "No frame"}, 404

def run_pipeline(conf_thresh):
    """Main detection pipeline"""
    global current_frame_b64, detection_stats
    
    from see.camera import IMX500Camera
    from see.models.fire_detector import FireDetector
    
    # Load config
    with open(PROJECT_ROOT + "/configs/config.json", 'r') as f:
        cfg = json.load(f)
    with open(PROJECT_ROOT + "/configs/labels.json", 'r') as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    
    vision_cfg = cfg["vision"]
    camera_cfg = vision_cfg["camera"]
    fire_cfg = vision_cfg["models"]["fire"]
    
    # Start camera
    print("Starting camera...")
    camera = IMX500Camera(
        model_path=fire_cfg["rpk"],
        resolution=tuple(camera_cfg["resolution"]),
        fps=camera_cfg["fps"]
    )
    camera.start()
    print("✓ Camera started")
    
    # Init detector
    detector = FireDetector(
        imx500=camera.imx500,
        conf_threshold=conf_thresh,
        labels=labels,
        picam2=camera.picam2
    )
    detector.load()
    print("✓ FireDetector ready")
    
    frame_n = 0
    fire_total = 0
    detect_frames = 0
    fps_counter = 0
    last_time = time.time()
    
    print(f"Starting detection loop. Open browser to http://localhost:5000")
    
    try:
        while True:
            result = camera.capture()
            if result is None:
                time.sleep(0.01)
                continue
            
            frame, metadata = result
            fh, fw = frame.shape[:2]
            frame_n += 1
            fps_counter += 1
            
            # Update FPS
            if time.time() - last_time >= 1.0:
                detection_stats['fps'] = fps_counter
                fps_counter = 0
                last_time = time.time()
            
            # Run detection
            clusters, raw_detections = detector.detect(
                metadata=metadata,
                frame_width=fw,
                frame_height=fh
            )
            
            # Update stats
            if raw_detections:
                detect_frames += 1
                fire_total += sum(1 for d in raw_detections if d.label == "fire")
                detection_stats['has_fire'] = any(d.label == "fire" for d in raw_detections)
                print(f"[frame {frame_n}] {len(raw_detections)} detections (fire total={fire_total})")
            else:
                detection_stats['has_fire'] = False
            
            detection_stats['total_frames'] = frame_n
            detection_stats['detect_frames'] = detect_frames
            detection_stats['fire_total'] = fire_total
            detection_stats['detection_rate'] = round(100 * detect_frames / frame_n, 1) if frame_n > 0 else 0
            
            # Draw and encode
            annotated = draw_detections(frame, raw_detections, clusters, frame_n, fire_total, conf_thresh)
            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            current_frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.stop()
        print(f"\nFinal: {frame_n} frames, {detect_frames} with fire, {fire_total} fire detections")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    args = parser.parse_args()
    
    # Store conf for template
    import __main__
    __main__.args = args
    
    print("="*60)
    print("🔥 FIRE DETECTION WEB VIEWER")
    print(f"   Confidence threshold: {args.conf}")
    print(f"   Open browser: http://localhost:5000")
    print("="*60)
    
    # Start pipeline in background
    pipeline_thread = threading.Thread(target=run_pipeline, args=(args.conf,), daemon=True)
    pipeline_thread.start()
    
    # Start Flask
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)