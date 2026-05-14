# vision_fuser.py — owns everything in the SEE layer
# runs the capture loop, assembles VisionSnapshot, sends to THINK via see_queue
# activated when sensor_triggered = True OR camera_feed_active = True

from datetime import datetime
from camera import IMX500Camera
from models.fire_detector import FireDetector, Detection, FireCluster
from snapshot import VisionSnapshot
import threading
import os
import cv2


# ── VisionFuser ───────────────────────────────────────────────────────────────
# the boss of the SEE layer
# owns the camera and fire detector
# loops: capture → analyze → assemble snapshot → emit to queue
class VisionFuser:

    # ── Init ──────────────────────────────────────────────────────────────────
    # receives config dict and SystemState from Orchestrator
    # reads vision section of config and builds camera + fire detector
    def __init__(self, config: dict, state):

        # ── Read vision config ────────────────────────────────────────────────
        vision_cfg  = config["vision"]
        camera_cfg  = vision_cfg["camera"]
        fire_cfg    = vision_cfg["models"]["fire"]
        storage_cfg = vision_cfg["storage"]

        # ── Read labels from labels.json ──────────────────────────────────────
        # labels map class ids to names: {0: "fire", 1: "other", 2: "smoke"}
        import json
        with open(vision_cfg["labels"], "r") as f:
            self._labels = json.load(f)         # loaded once at startup

        # ── Build camera ──────────────────────────────────────────────────────
        # camera owns IMX500 hardware and loads .rpk model onto chip
        self._camera = IMX500Camera(
            model_path = fire_cfg["rpk"],
            resolution = tuple(camera_cfg["resolution"]),   # [640,480] → (640,480)
            fps        = camera_cfg["fps"]
        )

        # ── Build fire detector ───────────────────────────────────────────────
        # fire detector receives imx500 object from camera AFTER camera starts
        # so we store conf_threshold now and pass imx500 later in start()
        self._conf_threshold = fire_cfg["conf_threshold"]
        self._fire_detector  = None             # built in start() after camera starts

        # ── Storage settings ──────────────────────────────────────────────────
        self._frame_path       = storage_cfg["frame_image_path"]
        self._frame_url_prefix = storage_cfg["frame_image_url_prefix"]

        # ── SystemState ───────────────────────────────────────────────────────
        self._state   = state                   # shared blackboard with all layers
        self._queue   = state.see_queue         # where we put VisionSnapshot for THINK

        # ── Control flags ─────────────────────────────────────────────────────
        self._running = False                   # is the capture loop running?
        self._thread  = None                    # capture loop runs in its own thread

    # ── Start ─────────────────────────────────────────────────────────────────
    # starts the camera, builds fire detector, launches capture loop
    def start(self) -> None:

        # start camera first — loads .rpk onto IMX500 chip
        self._camera.start()

        # NOW build fire detector — needs imx500 object from camera
        self._fire_detector = FireDetector(
            imx500          = self._camera.imx500,
            conf_threshold  = self._conf_threshold,
            labels          = self._labels
        )
        self._fire_detector.load()              # does nothing but keeps contract ✅

        # launch capture loop in its own thread so it doesn't block other layers
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # tell SystemState we are running
        self._state.see_running = True

    # ── Stop ──────────────────────────────────────────────────────────────────
    # stops the capture loop and powers off camera
    def stop(self) -> None:
        self._running = False
        self._camera.stop()
        self._state.see_running = False

    # ── Capture Loop ──────────────────────────────────────────────────────────
    # runs continuously in its own thread
    # capture → analyze → assemble snapshot → emit if sensor triggered
    def _capture_loop(self) -> None:

        while self._running and self._state.system_running:

            # ── Capture frame + metadata from IMX500 camera ──────────────────
            result = self._camera.capture()
            if result is None:
                continue                        # camera not ready, try again

            frame, metadata = result
            frame_height, frame_width = frame.shape[:2]  # numpy array → (h, w, channels)

            # ── Analyze with FireDetector ─────────────────────────────────────
            clusters, raw_detections = self._fire_detector.detect(
                metadata     = metadata,
                frame_width  = frame_width,
                frame_height = frame_height
            )

            # ── Save frame to disk ────────────────────────────────────────────
            frame_url = self._save_frame(frame)

            # ── Assemble VisionSnapshot ───────────────────────────────────────
            snap = self.snapshot(clusters, raw_detections, frame_url, frame_width, frame_height)

            # ── Emit to see_queue only if sensor is triggered ─────────────────
            # if only camera_feed_active → stream to website but don't send to THINK
            if self._state.sensor_triggered:
                self.emit_trigger(snap)

    # ── Snapshot ──────────────────────────────────────────────────────────────
    # assembles VisionSnapshot from FireDetector results
    # this is the contract between SEE and THINK — fills every field
    def snapshot(self, clusters: list, raw_detections: list, frame_url: str, frame_width: int, frame_height: int) -> VisionSnapshot:

        frame_area = frame_width * frame_height

        # ── Fire and smoke counts ─────────────────────────────────────────────
        fire_count  = sum(1 for d in raw_detections if d.label == "fire")
        smoke_count = sum(1 for d in raw_detections if d.label == "smoke")

        # ── Union areas across ALL detections ─────────────────────────────────
        fire_union_area  = sum(
            d.bbox[2] * d.bbox[3] for d in raw_detections if d.label == "fire"
        ) / frame_area

        smoke_union_area = sum(
            d.bbox[2] * d.bbox[3] for d in raw_detections if d.label == "smoke"
        ) / frame_area

        # ── Composite label ───────────────────────────────────────────────────
        # what did we find overall in this frame?
        if fire_count > 0 and smoke_count > 0:
            composite_label = "fire_smoke"
        elif fire_count > 0:
            composite_label = "fire"
        elif smoke_count > 0:
            composite_label = "smoke"
        else:
            composite_label = "none"

        # ── Glimpsed fire ─────────────────────────────────────────────────────
        # True if ANY fire box was detected, even low confidence
        # raw_detections already filtered by conf_threshold in FireDetector
        # so glimpsed_fire = any fire box made it through
        glimpsed_fire = any(d.label == "fire" for d in raw_detections)

        # ── Human near fire ───────────────────────────────────────────────────
        # TODO: not implemented yet — needs human detection model
        human_near_fire = False

        return VisionSnapshot(
            timestamp        = datetime.now(),
            scene_label      = "unknown",       # SceneClassifier removed for FYP scope
            scene_confidence = 0.0,
            composite_label  = composite_label,
            glimpsed_fire    = glimpsed_fire,
            human_near_fire  = human_near_fire,
            fire_count       = fire_count,
            smoke_count      = smoke_count,
            fire_union_area  = fire_union_area,
            smoke_union_area = smoke_union_area,
            cluster_count    = len(clusters),
            fire_clusters    = clusters,
            image_url        = frame_url,
            raw_detections   = raw_detections,
        )

    # ── Emit Trigger ──────────────────────────────────────────────────────────
    # puts VisionSnapshot into see_queue for THINK to consume
    def emit_trigger(self, snapshot: VisionSnapshot) -> None:
        self._queue.put(snapshot)

    # ── Save Frame ────────────────────────────────────────────────────────────
    # saves captured frame to disk and returns its URL
    # URL is stored in VisionSnapshot so THINK can reference the image
    def _save_frame(self, frame) -> str:
        

        # create storage folder if it doesn't exist
        os.makedirs(self._frame_path, exist_ok=True)

        # filename = timestamp so every frame has a unique name
        filename  = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath  = os.path.join(self._frame_path, filename)

        # save frame as jpg using OpenCV
        cv2.imwrite(filepath, frame)

        # return the URL that the website uses to serve this image
        return self._frame_url_prefix + filename