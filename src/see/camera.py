"""
Camera Module - IMX500 Hardware Management.

This module owns and manages the Raspberry Pi IMX500 camera hardware.
It loads the YOLO detection model onto the IMX500 chip (runs on-device inference)
and handles frame capture with associated metadata from the hardware accelerator.

Classes:
    IMX500Camera: Manages camera initialization, YOLO model loading, and frame capture.
"""

# camera.py — owns the IMX500 hardware and Picamera2
# responsible for loading the YOLO .rpk model onto the IMX500 chip
# and capturing frames + metadata for FireDetector to analyze
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
import numpy as np


# ── IMX500Camera ──────────────────────────────────────────────────────────────
# owns the camera hardware — IMX500 chip + Picamera2
# loads the YOLO .rpk file onto the chip at startup
# captures frames + metadata and returns them to VisionFuser
class IMX500Camera:
    """
    Manages the IMX500 camera hardware and YOLO model.
    
    The IMX500 is a specialized camera with an embedded AI accelerator chip.
    This class handles:
    - Loading the YOLO model (.rpk file) onto the chip for on-device inference
    - Configuring camera resolution and frame rate
    - Capturing frames and YOLO inference metadata
    
    Attributes:
        _model_path (str): Path to the YOLO .rpk model file
        _resolution (tuple): Target camera resolution (width, height)
        _fps (int): Target frames per second
        _imx500 (IMX500): IMX500 device object for model loading
        _picam2 (Picamera2): Camera interface for frame capture
        _active (bool): Whether camera is currently running
    """

    # ── Init ─────────────────────────────────────────────────────────────────
    # receives camera settings and model path from VisionFuser (who reads config)
    # does NOT open config.json itself
    def __init__(self, model_path: str, resolution: tuple[int, int], fps: int):
        """
        Initialize IMX500 camera configuration (does not start hardware yet).
        
        Args:
            model_path (str): Absolute path to YOLO .rpk model file
            resolution (tuple): Target resolution as (width, height), e.g. (640, 480)
            fps (int): Target frames per second, e.g. 30
        """
        self._model_path = model_path           # path to .rpk YOLO model file
        self._resolution = resolution           # (width, height) e.g. (640, 480)
        self._fps        = fps                  # frames per second e.g. 30
        self._imx500     = None                 # IMX500 object, created in start()
        self._picam2     = None                 # Picamera2 object, created in start()
        self._active     = False                # is the camera currently running?

    # ── Start ─────────────────────────────────────────────────────────────────
    # powers on the camera and loads the YOLO .rpk model onto the IMX500 chip
    # MUST be called before capture()
    # IMX500 must be created BEFORE Picamera2 — order matters!
    def start(self) -> None:

        """
        Power on camera and load YOLO model onto IMX500 chip.
        
        This method:
        1. Initializes IMX500 device and loads YOLO model (may take minutes on first run)
        2. Creates Picamera2 interface (must happen AFTER IMX500 init)
        3. Configures resolution and frame rate from config
        4. Starts the capture stream
        
        Must be called before capture(). Call only once per session.
        
        Returns:
            None
        """

        # step 1: load .rpk model onto IMX500 chip
        # this uploads the YOLO firmware to the camera hardware
        # may take a few minutes on first run (progress bar shown in console)
        self._imx500 = IMX500(self._model_path)

        # step 2: create Picamera2 AFTER IMX500 is ready
        self._picam2 = Picamera2()

        # step 3: configure camera settings from config
        # main stream → the image we capture
        # controls → fps setting
        config = self._picam2.create_preview_configuration(
            main={"size": self._resolution, "format": "RGB888"},    # resolution from config
            controls={"FrameRate": self._fps}   # fps from config
        )
        self._picam2.configure(config)

        # step 4: start capturing
        self._picam2.start()
        self._active = True

    # ── Stop ──────────────────────────────────────────────────────────────────
    # powers off the camera cleanly
    # called by VisionFuser when sensor_triggered goes False
    def stop(self) -> None:
        """
        Power off camera cleanly and release hardware resources.
        
        Stops the capture stream and marks camera as inactive.
        Safe to call multiple times.
        
        Returns:
            None
        """
        if self._picam2 and self._active:
            self._picam2.stop()
            self._active = False

    # ── Capture ───────────────────────────────────────────────────────────────
    # grabs one frame + IMX500 metadata from the camera
    # returns (frame, metadata) → VisionFuser passes both to FireDetector
    # frame    → numpy array image, saved to disk and streamed to website
    # metadata → contains YOLO inference results from IMX500 chip
    def capture(self) -> tuple[np.ndarray, dict] | None:

        """
        Capture one frame and YOLO inference metadata from camera.
        
        Returns:
            tuple[np.ndarray, dict] | None: 
                - frame: Numpy array (H, W, 3) in BGR format
                - metadata: Dict containing IMX500 YOLO inference results
                - None if camera is not active
        """

        if not self._active:
            return None                         # camera not running, return nothing

        # capture one request from Picamera2
        # request contains BOTH the image AND the IMX500 inference metadata
        request = self._picam2.capture_request()

        # extract the image as a numpy array (rows x cols x RGB channels)
        frame = request.make_array("main")

        # extract metadata — this contains the YOLO results from IMX500 chip
        metadata = request.get_metadata()

        # release the request so Picamera2 can reuse the buffer
        request.release()

        return frame, metadata

    # ── Properties ───────────────────────────────────────────────────────────
    @property
    def imx500(self) -> IMX500:
        """Get IMX500 device object for inference."""
        return self._imx500                     # VisionFuser passes this to FireDetector

    @property
    def picam2(self):
        """Get the Picamera2 object — needed by convert_inference_coords()."""
        return self._picam2

    @property
    def is_active(self) -> bool:
        """Check if camera is currently running."""
        return self._active                     # is camera currently running?

    @property
    def resolution(self) -> tuple[int, int]:
        """Get configured camera resolution as (width, height)."""
        return self._resolution                 # (width, height)