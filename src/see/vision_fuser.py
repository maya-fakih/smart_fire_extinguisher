"""
Vision Fuser Module - SEE Layer Orchestrator.

Coordinates the entire SEE layer:
- Manages IMX500 camera hardware
- Runs the continuous capture/analyze loop
- Assembles VisionSnapshot data contracts
- Emits processed vision data to THINK layer via queue
- Writes dominant cluster center to SystemState for ACT arm tracking

The VisionFuser is the main entry point and orchestrator for all vision tasks.

Classes:
    VisionFuser: Main orchestrator class
"""

# vision_fuser.py — owns everything in the SEE layer
# runs the capture loop, assembles VisionSnapshot, sends to THINK via see_queue
# activated when sensor_triggered = True OR camera_feed_active = True

from datetime import datetime
from see.camera import IMX500Camera
from see.models.fire_detector import FireDetector
from see.models.detection import Detection
from see.models.fire_cluster import FireCluster
from see.snapshot import VisionSnapshot
import threading
import time
import os
import cv2
import logging

logger = logging.getLogger(__name__)


# ── VisionFuser ───────────────────────────────────────────────────────────────
# the boss of the SEE layer
# owns the camera and fire detector
# loops: capture → analyze → assemble snapshot → emit to queue
class VisionFuser:
    """
    Orchestrator of the SEE (vision/perception) layer.

    Manages the complete vision pipeline:
    1. Initializes and controls camera hardware
    2. Builds fire detector for inference analysis
    3. Runs continuous capture loop in background thread
    4. Processes detections and assembles VisionSnapshot outputs
    5. Emits snapshots to THINK layer via see_queue
    6. Writes dominant cluster center to SystemState for ACT arm tracking

    Attributes:
        _camera: IMX500Camera instance
        _fire_detector: FireDetector instance
        _state: Shared SystemState blackboard
        _queue: Output queue to THINK layer
        _running: Whether capture loop is active
        _thread: Background thread running capture loop
    """

    # ── Init ──────────────────────────────────────────────────────────────────
    # receives config dict and SystemState from Orchestrator
    # reads vision section of config and builds camera + fire detector
    def __init__(self, config: dict, state, notifier=None):
        """
        Initialize VisionFuser with configuration.

        Reads vision configuration from config dict and builds child components
        (camera and fire detector). Does not start hardware yet - call start().

        Args:
            config: Configuration dict with 'vision' key containing:
                - camera: resolution, fps settings
                - models.fire: rpk model path, conf_threshold
                - labels: path to labels.json
                - storage: frame save paths
            state: SystemState object (shared blackboard)
        """

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
        self._frame_path       = storage_cfg["frame_image_path"]       # default if permanent_path not set
        self._frame_url_prefix = storage_cfg["frame_url_prefix"]
        # Permanent frames (sensor-triggered) go to USB to avoid SD card wear.
        # Falls back to frame_image_path if USB is not mounted.
        self._frame_permanent_path = storage_cfg.get("frame_permanent_path", self._frame_path)
        # stream.jpg lives in RAM (tmpfs) — written ~30x/sec, zero disk I/O.
        self._stream_dir = "/dev/shm/fire_robot"
        # Lower quality = smaller JPEG = faster through tunnel. 55 is fine for a dashboard.
        self._stream_jpeg_quality = camera_cfg.get("stream_jpeg_quality", 55)

        # ── Activation gate settings ──────────────────────────────────────────
        # SEE only does work when sensor_triggered or camera_feed_active.
        # When neither, the loop sleeps this many ms then re-checks.
        activation_cfg     = vision_cfg.get("activation", {})
        self._idle_sleep_s = activation_cfg.get("idle_sleep_ms", 50) / 1000.0

        # ── SystemState ───────────────────────────────────────────────────────
        self._state   = state                   # shared blackboard with all layers
        self._notifier = notifier               # notification service
        self._queue   = state.see_queue         # where we put VisionSnapshot for THINK

        # ── Control flags ─────────────────────────────────────────────────────
        self._running = False                   # is the capture loop running?
        self._thread  = None                    # capture loop runs in its own thread

    # ── Start ─────────────────────────────────────────────────────────────────
    # starts the camera, builds fire detector, launches capture loop
    def start(self) -> None:
        # Child process: ignore SIGINT so Ctrl+C is handled only by parent.
        import signal as _signal
        _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
        """
        Start camera and launch capture loop.

        This method:
        1. Starts camera hardware (loads YOLO onto IMX500)
        2. Builds FireDetector instance
        3. Launches _capture_loop in background thread
        4. Updates SystemState to indicate layer is running

        Returns:
            None
        """

        # Catch-all: if ANYTHING in start() crashes, log it loudly before the
        # process dies — otherwise the camera just silently disappears.
        try:
            self._start_inner()
        except Exception as e:
            logger.critical(
                f"VisionFuser: start() CRASHED — SeeProcess will exit | "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    def _start_inner(self) -> None:

        # start camera first — loads .rpk onto IMX500 chip
        try:
            self._camera.start()
        except Exception as e:
            if self._notifier is not None:
                from notify import EventType
                self._notifier.notify(
                    EventType.CAMERA_FAILED_TO_START,
                    payload={"error": f"{type(e).__name__}: {e}"},
                    source_layer="see",
                )
            raise

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

        # tell SystemState we are running — retry if manager proxy isn't
        # connected yet (same race as _main_loop, see comment there).
        for _attempt in range(20):
            try:
                self._state.see_running = True
                break
            except Exception as e:
                logger.warning(
                    f"VisionFuser: start() state write failed ({e}) — "
                    f"retry {_attempt + 1}/20"
                )
                time.sleep(0.25)
        logger.info(
            f"VisionFuser: started | conf_threshold={self._conf_threshold} | "
            f"camera_active={self._camera.is_active}"
        )

        # Keep the SeeProcess alive — same pattern as SensorFuser._main_loop().
        # Without this, start() returns immediately, the multiprocessing.Process
        # target function completes, the OS process exits, and the daemon thread
        # (and camera hardware) die with it — which is exactly why stream.jpg
        # was never written and every /api/camera/snapshot returned 404.
        self._main_loop()

    # ── Stop ──────────────────────────────────────────────────────────────────
    # signals the capture loop and _main_loop to exit
    def stop(self) -> None:
        """
        Signal the capture loop and _main_loop to exit.

        Sets _running = False so _main_loop unblocks and handles cleanup
        (thread join + camera stop). Safe to call multiple times.

        Returns:
            None
        """
        self._running = False

    # ── Main Loop ─────────────────────────────────────────────────────────────
    # Keeps the SeeProcess alive while the capture thread runs.
    # Mirrors SensorFuser._main_loop() exactly — without this the process
    # exits as soon as start() returns and the daemon thread dies with it.
    def _main_loop(self) -> None:
        """
        Block the SeeProcess main function until system stops.

        start() calls this after launching _capture_loop in a daemon thread.
        Without this, the multiprocessing.Process target returns immediately,
        the process exits, and the daemon capture thread (plus camera) die.
        """
        while self._running:
            try:
                sr = self._state.system_running
                if not sr:
                    logger.info(
                        f"VisionFuser: _main_loop exiting — system_running={sr}"
                    )
                    break
            except Exception as e:
                logger.warning(
                    f"VisionFuser: _main_loop state read failed ({e}) — retrying"
                )
                time.sleep(0.1)
                continue
            time.sleep(0.5)

        # System is stopping — clean up
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._camera.stop()
        self._state.see_running = False
        logger.info("VisionFuser: stopped")

    # ── Capture Loop ──────────────────────────────────────────────────────────
    # runs continuously in its own thread
    # capture → analyze → write state → assemble snapshot → emit if sensor triggered
    def _capture_loop(self) -> None:
        """
        Main capture loop (runs in background thread).

        Continuous loop that:
        1. Captures frame from camera
        2. Sends to FireDetector for analysis
        3. Writes dominant cluster center to SystemState (for ACT arm tracking)
        4. Saves frame to disk
        5. Assembles VisionSnapshot
        6. Emits to see_queue only if sensor_triggered is True

        Exits when _running becomes False or system_running becomes False.

        Returns:
            None
        """

        _none_count = 0
        _gate_logged = False

        while self._running:
            try:
                if not self._state.system_running:
                    break
            except Exception:
                time.sleep(0.1)
                continue

            # ── Activation gate ──────────────────────────────────────────────
            # SEE only does work when sensor_triggered or camera_feed_active.
            # Otherwise idle — saves power on the IMX500 + arm.
            # Camera signal is cleared on entry to idle so ACT's fusion never
            # averages stale camera coords with fresh heat data.
            if not (self._state.sensor_triggered or self._state.camera_feed_active):
                if self._state.latest_fire_x is not None or self._state.latest_fire_y is not None:
                    self._state.latest_fire_x = None
                    self._state.latest_fire_y = None
                _gate_logged = False
                time.sleep(self._idle_sleep_s)
                continue

            if not _gate_logged:
                logger.info(
                    f"VisionFuser: capture loop activated | "
                    f"sensor_triggered={self._state.sensor_triggered} | "
                    f"camera_feed_active={self._state.camera_feed_active}"
                )
                _gate_logged = True

            # ── BUG-7: wrap the full work cycle so an exception in any step
            # doesn't silently kill the thread and leave see_running=True.
            try:
                # ── Capture frame + metadata from IMX500 camera ──────────────────
                result = self._camera.capture()
                if result is None:
                    _none_count += 1
                    if _none_count == 1 or _none_count % 50 == 0:
                        logger.warning(
                            f"VisionFuser: camera.capture() returned None "
                            f"(count={_none_count}) — camera_active={self._camera.is_active}"
                        )
                    continue
                _none_count = 0

                frame, metadata = result
                frame_height, frame_width = frame.shape[:2]  # numpy array → (h, w, channels)
                if _none_count == 0 and not hasattr(self, '_first_frame_logged'):
                    self._first_frame_logged = True
                    logger.info(
                        f"VisionFuser: first frame captured | "
                        f"resolution={frame_width}x{frame_height}"
                    )

                # ── Analyze with FireDetector ─────────────────────────────────────
                clusters, raw_detections = self._fire_detector.detect(
                    metadata     = metadata,
                    frame_width  = frame_width,
                    frame_height = frame_height
                )

                # ── Write dominant cluster center to SystemState for ACT ──────────
                # ACT's arm controller reads latest_fire_x/y as visual servoing feedback
                # Values are normalized [0, 1] with (0.5, 0.5) = image center
                # clusters[0] = first cluster (treated as dominant for now)
                # TODO: verify clusters are sorted by danger_score before this is final
                if clusters:
                    self._state.latest_fire_x = clusters[0].origin_x
                    self._state.latest_fire_y = clusters[0].origin_y
                else:
                    # no fire detected → clear state so ACT doesn't act on stale data
                    self._state.latest_fire_x = None
                    self._state.latest_fire_y = None

                # ── Save frame to disk ────────────────────────────────────────────
                frame_url = self._save_frame(frame, raw_detections, clusters)

                # ── Assemble VisionSnapshot ───────────────────────────────────────
                snap = self.snapshot(clusters, raw_detections, frame_url, frame_width, frame_height)

                # ── Emit to see_queue ─────────────────────────────────────────────
                # Prediction modes: only push when the sensor has fired (FIX-3a)
                # — the live camera feed alone must not flood THINK with frames.
                #
                # Training-recording is the deliberate exception: the recorder
                # NEEDS a steady frame stream on see_queue to align + label, even
                # with no fire present (e.g. recording an empty office as level 1).
                # Gating that on sensor_triggered made no-fire recordings save
                # nothing. So while recording, emit on camera_feed_active too.
                if self._state.sensor_triggered or (
                    self._state.training_recording and self._state.camera_feed_active
                ):
                    self.emit_trigger(snap)

            except Exception as e:
                logger.error(
                    f"VisionFuser: capture loop iteration failed - {type(e).__name__}: {e}",
                    exc_info=True
                )
                # Note: we deliberately don't notify here — none of the existing
                # EventType values cleanly describe "arbitrary exception in the
                # capture loop". Adding a new SEE_LAYER_ERROR event type is a
                # follow-up. The logger.error above already records the failure
                # with full traceback.
                # Brief back-off so a persistent failure doesn't spin at 100% CPU.
                time.sleep(0.5)

    # ── Snapshot ──────────────────────────────────────────────────────────────
    # assembles VisionSnapshot from FireDetector results
    # this is the contract between SEE and THINK — fills every field
    def snapshot(self, clusters: list, raw_detections: list, frame_url: str, frame_width: int, frame_height: int) -> VisionSnapshot:
        """
        Assemble VisionSnapshot from fire detection results.

        Constructs the output data contract from raw detection data:
        - Counts fire and smoke detections
        - Computes area coverage statistics
        - Determines composite label (fire/smoke/both/none)
        - Packages all into VisionSnapshot for THINK layer

        Args:
            clusters: List of FireCluster objects from FireDetector
            raw_detections: All Detection objects (unmerged, unfiltered)
            frame_url: URL where frame was saved
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            VisionSnapshot: Complete vision output snapshot
        """

        frame_area = frame_width * frame_height

        # ── Fire and smoke counts ─────────────────────────────────────────────
        fire_count  = sum(1 for d in raw_detections if d.label == "fire")
        smoke_count = sum(1 for d in raw_detections if d.label == "smoke")

        # ── Union areas across ALL detections (overlap-corrected) ─────────────
        # Delegates to FireDetector's union helper (Shapely-backed plane sweep).
        # The previous implementation summed raw areas which double-counted
        # any overlap — fixed as part of BUG-8.
        fire_boxes       = [d for d in raw_detections if d.label == "fire"]
        smoke_boxes      = [d for d in raw_detections if d.label == "smoke"]
        fire_union_area  = self._fire_detector.compute_union_area_pixels(fire_boxes)  / frame_area
        smoke_union_area = self._fire_detector.compute_union_area_pixels(smoke_boxes) / frame_area

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
        """
        Emit VisionSnapshot to THINK layer.

        Puts snapshot into see_queue where THINK layer will consume it.
        Called only when sensor_triggered is True (not for camera feed only).

        Args:
            snapshot: VisionSnapshot to emit

        Returns:
            None
        """
        self._queue.put(snapshot)

    # ── Save Frame ────────────────────────────────────────────────────────────
    # saves captured frame to disk and returns its URL
    # URL is stored in VisionSnapshot so THINK can reference the image
    def _save_frame(self, frame, raw_detections=None, clusters=None) -> str:
        """
        Save captured frame to disk and return its URL.

        Creates frame storage directory if needed, saves frame with
        timestamp filename, and returns the web-accessible URL.

        Also overwrites stream.jpg atomically (temp + os.replace) when
        camera_feed_active is True — this is the rolling buffer that
        powers the MJPEG feed. One file, always overwritten, no accumulation.

        Args:
            frame: Numpy array frame from camera

        Returns:
            str: Web URL to access the saved frame
        """

        os.makedirs(self._stream_dir, exist_ok=True)

        # ── Timestamped frame (permanent, only on sensor-triggered events) ────
        # Goes to USB (frame_permanent_path) to avoid SD card wear.
        # Falls back to frame_image_path if USB is not available.
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"

        if self._state.sensor_triggered:
            # Save to USB only — never fall back to SD card.
            save_dir = self._frame_permanent_path
            try:
                os.makedirs(save_dir, exist_ok=True)
                filepath = os.path.join(save_dir, filename)
                ok = cv2.imwrite(filepath, frame)
                if not ok:
                    raise IOError(f"cv2.imwrite returned False for {filepath}")
            except Exception as e:
                # USB not mounted or write failed — skip this frame silently.
                # Log once per 50 failures to avoid flooding.
                if not hasattr(self, '_frame_fail_count'):
                    self._frame_fail_count = 0
                self._frame_fail_count += 1
                if self._frame_fail_count == 1 or self._frame_fail_count % 50 == 0:
                    logger.warning(
                        f"VisionFuser: permanent frame save skipped "
                        f"(count={self._frame_fail_count}) — {type(e).__name__}: {e}"
                    )

        # ── Stream buffer (in RAM — /dev/shm/) ───────────────────────────────
        # Written every frame when camera feed is active. Lives in tmpfs so
        # there is zero SD card I/O — just memory writes.
        #
        # Annotation: YOLO boxes are drawn on a COPY of the frame before write,
        # so the on-disk permanent save (USB) stays clean for training data.
        # The annotated copy goes only to stream.jpg (the live MJPEG feed).
        if self._state.camera_feed_active:
            stream_path = os.path.join(self._stream_dir, "stream.jpg")
            tmp_path    = os.path.join(self._stream_dir, "stream.tmp.jpg")
            try:
                annotated = self._annotate_frame(frame, raw_detections, clusters)
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._stream_jpeg_quality]
                ok = cv2.imwrite(tmp_path, annotated, encode_params)
                if ok:
                    os.replace(tmp_path, stream_path)
            except Exception as e:
                logger.warning(f"stream.jpg write failed: {e}")

        return self._frame_url_prefix + filename

    # ─────────────────────────────────────────────────────────────────────────
    # Annotation helper — draws YOLO boxes onto a COPY of the frame.
    # Used only for the live MJPEG stream. The on-disk permanent save uses the
    # raw unannotated frame so training data stays clean.
    # ─────────────────────────────────────────────────────────────────────────
    def _annotate_frame(self, frame, raw_detections, clusters):
        """
        Draw bounding boxes and labels on a copy of the frame.

        Colors:
          fire  → red    (BGR: 0, 0, 255)
          smoke → blue   (BGR: 255, 128, 0)
          cluster center → small green crosshair on the dominant cluster

        Returns the annotated copy. Falls back to the original frame on any
        error — we never want annotation issues to break the live stream.
        """
        try:
            if not raw_detections and not clusters:
                return frame
            annotated = frame.copy()

            # ── Per-detection boxes ──────────────────────────────────────────
            for det in (raw_detections or []):
                xc, yc, w, h = det.bbox
                x1 = int(xc - w / 2); y1 = int(yc - h / 2)
                x2 = int(xc + w / 2); y2 = int(yc + h / 2)
                if det.label == "fire":
                    color = (0, 0, 255)        # red in BGR
                elif det.label == "smoke":
                    color = (255, 128, 0)      # blue-ish in BGR
                else:
                    color = (200, 200, 200)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{det.label} {det.confidence:.2f}"
                cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # ── Dominant cluster crosshair (where ACT would aim) ─────────────
            if clusters:
                c = clusters[0]
                fh, fw = annotated.shape[:2]
                cx = int(c.origin_x * fw)
                cy = int(c.origin_y * fh)
                cv2.drawMarker(annotated, (cx, cy), (0, 255, 0),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=20, thickness=2)

            return annotated
        except Exception as e:
            logger.debug(f"_annotate_frame: fell back to raw frame — {e}")
            return frame