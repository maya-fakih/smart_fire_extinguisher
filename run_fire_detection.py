import time
from picamera2 import Picamera2
from picamera2.devices import IMX500
# Essential: Ensure imx500_object_detection_demo.py is in the same directory
from imx500_object_detection_demo import parse_and_draw_object_detection_results

# Initialize and configure the camera with the .rpk model
MODEL_PATH = "./model_weights/rpk/fire_smoke_detection.rpk"
imx500 = IMX500(MODEL_PATH)
picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration()

# Start camera and enable drawing
imx500.show_network_fw_progress_bar()
picam2.pre_callback = parse_and_draw_object_detection_results
picam2.start(config, show_preview=True)

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    picam2.stop()
