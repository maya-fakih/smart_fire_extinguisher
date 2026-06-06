"""
test_arm_base_sweep.py
======================
Isolated hardware test for the arm pan (base) joint only.

Goal
----
Move the pan servo through:
  Step 1 — go to 0°   (servo center, physical "home")
  Step 2 — go to 90°  (full right limit per config)
  Step 3 — go to -90° (full left limit = physical "180°" sweep from step 2)
  Step 4 — return to 0° (physical "90°" midpoint of the full range)

Why the angle mapping?
  The config defines pan limits as [-90, +90] with servo_min/max the same,
  so gpiozero maps:
    -90° → servo value -1.0  (one physical extreme)
      0° → servo value  0.0  (center)
    +90° → servo value +1.0  (other physical extreme)
  The full 0→180→90 sweep you asked for is therefore:
    physical 0   → commanded   0°
    physical 180 → commanded ±90° (we go +90 first, then -90 for the return)
    physical 90  → commanded   0°  (back to center)

Run directly on the Pi:
  cd /path/to/smart_fire_extinguisher
  python -m tests.test_arm_base_sweep        # from project root
  # or
  python tests/test_arm_base_sweep.py        # with PYTHONPATH set

Dependencies (already in requirements-pi.txt): gpiozero, lgpio, numpy
"""

import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make sure src/ is importable when run directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
SRC_DIR      = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_arm_base_sweep")

# ---------------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.json"


def load_pan_config() -> dict:
    """Pull only the arm->pan section from config.json."""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    arm_cfg = cfg["act"]["actuators"]["arm"]
    pan_cfg = arm_cfg["joints"]["pan"]
    return pan_cfg


def deg_to_servo_value(deg: float, servo_min: float, servo_max: float) -> float:
    """Map degrees → gpiozero Servo.value in [-1, +1]."""
    span = servo_max - servo_min
    if span <= 0:
        return 0.0
    v = 2.0 * (deg - servo_min) / span - 1.0
    return max(-1.0, min(1.0, v))


def move_to(servo, deg: float, pan_cfg: dict, label: str) -> None:
    """Command the servo to a given angle and log the result."""
    val = deg_to_servo_value(
        deg,
        pan_cfg["servo_min_deg"],
        pan_cfg["servo_max_deg"],
    )
    servo.value = val
    logger.info(
        f"  [{label}]  commanded {deg:+.1f}°  →  servo_value={val:+.4f}"
    )


# ---------------------------------------------------------------------------
# Main sweep routine
# ---------------------------------------------------------------------------
def run_sweep(dwell_s: float = 1.5) -> None:
    """
    Execute the pan sweep and then release the GPIO pin cleanly.

    Parameters
    ----------
    dwell_s : float
        How long to hold each position before moving (seconds).
    """
    from gpiozero import Servo
    from gpiozero.pins.lgpio import LGPIOFactory

    pan_cfg = load_pan_config()
    pin     = int(pan_cfg["pin"])

    logger.info("=" * 60)
    logger.info("ARM BASE SWEEP TEST")
    logger.info(f"  Pan pin           : GPIO {pin}")
    logger.info(f"  Servo range       : {pan_cfg['servo_min_deg']}° → {pan_cfg['servo_max_deg']}°")
    logger.info(f"  Limit range       : {pan_cfg['limit_min_deg']}° → {pan_cfg['limit_max_deg']}°")
    logger.info(f"  Dwell per step    : {dwell_s} s")
    logger.info("=" * 60)

    factory = LGPIOFactory()
    servo   = Servo(pin=pin, pin_factory=factory)

    try:
        # ------------------------------------------------------------------
        # Step 1: Move to 0° — physical "home" / center
        # ------------------------------------------------------------------
        logger.info("Step 1/4 — moving to 0° (home / center)")
        move_to(servo, 0.0, pan_cfg, "0°")
        time.sleep(dwell_s)

        # ------------------------------------------------------------------
        # Step 2: Move to +90° — one physical extreme (physical "180°" from -90°)
        # ------------------------------------------------------------------
        logger.info("Step 2/4 — moving to +90° (physical 180° position)")
        move_to(servo, 90.0, pan_cfg, "+90°")
        time.sleep(dwell_s)

        # ------------------------------------------------------------------
        # Step 3: Move to -90° — opposite physical extreme (full sweep done)
        # ------------------------------------------------------------------
        logger.info("Step 3/4 — moving to -90° (opposite extreme, completes 180° sweep)")
        move_to(servo, -90.0, pan_cfg, "-90°")
        time.sleep(dwell_s)

        # ------------------------------------------------------------------
        # Step 4: Return to 0° — physical midpoint ("back to 90°")
        # ------------------------------------------------------------------
        logger.info("Step 4/4 — returning to 0° (physical 90° midpoint)")
        move_to(servo, 0.0, pan_cfg, "0°")
        time.sleep(dwell_s)

        logger.info("Sweep complete ✓")

    finally:
        # Always release the pin, even if the test throws
        logger.info("Releasing GPIO pin …")
        servo.close()
        logger.info("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arm pan (base) sweep test")
    parser.add_argument(
        "--dwell",
        type=float,
        default=1.5,
        metavar="SECONDS",
        help="Time to hold each position in seconds (default: 1.5)",
    )
    args = parser.parse_args()

    run_sweep(dwell_s=args.dwell)
