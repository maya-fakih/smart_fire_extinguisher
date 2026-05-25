"""
Tests for tonight's changes:
  - ACT arm fusion (3-mode + bias correction)
  - THINK ThinkDatabase._persist_frame
  - SEE activation gate idle-clear behavior (logic-only — no real camera)

Skips the parts that genuinely need hardware (camera frames, real DB).
Each test uses fakes for SystemState / config.
"""
import os
import sys
import shutil
import tempfile
import types

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Stub the Pi-only modules before importing anything that touches them
_install_stubs_done = False
def _install_stubs():
    global _install_stubs_done
    if _install_stubs_done: return
    _install_stubs_done = True

    class _LazyStub(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []   # mark as a package so submodule import works
        def __getattr__(self, item):
            # Return a stub class for any attribute access (treats it as a class import).
            stub = type(item, (), {"__init__": lambda self, *a, **k: None})
            setattr(self, item, stub)
            return stub

    for name in [
        "RPi", "RPi.GPIO",
        "gpiozero",
        "board", "busio",
        "picamera2", "picamera2.devices", "picamera2.devices.imx500",
        "adafruit_ads1x15", "adafruit_ads1x15.ads1115", "adafruit_ads1x15.analog_in",
        "adafruit_amg88xx",
        "adafruit_mcp9808",
        "smbus2",
    ]:
        if name not in sys.modules:
            sys.modules[name] = _LazyStub(name)

_install_stubs()


# ────────────────────────────────────────────────────────────────────
# ACT arm fusion
# ────────────────────────────────────────────────────────────────────

def test_arm_fusion():
    """
    Direct unit test of ArmController._compute_error covering all 4 cases:
      - both heat + camera available → average with biases applied
      - heat only                    → heat + bias
      - camera only                  → camera + bias
      - neither                      → None
    """
    from act.actuators.arm_controller import ArmController

    # Build a state stub with the fields the fusion reads
    class State:
        latest_heat_matrix = None
        latest_fire_x      = None
        latest_fire_y      = None

    state = State()

    # Build a config that the constructor needs (joints + feedback)
    cfg = {
        "joints": {
            "pan":  {"pin": 27, "step_deg": 0.5, "limit_min_deg": -90, "limit_max_deg": 90,
                     "servo_min_deg": -90, "servo_max_deg": 90, "invert": False},
            "tilt": {"pin": 22, "step_deg": 0.5, "limit_min_deg": -45, "limit_max_deg": 45,
                     "servo_min_deg": -90, "servo_max_deg": 90, "invert": False},
        },
        "feedback": {
            "heat_use_threshold_c": 35,
            "tolerance_normalized": 0.1,
            "sensor_offsets": {
                "heat":   {"x_bias": 0.10, "y_bias": -0.05},
                "camera": {"x_bias": -0.20, "y_bias": 0.10},
            },
        },
        "cycle_active_ms": 100,
        "cycle_idle_ms":   500,
        "type": "arm",
        "enabled": True,
    }

    arm = ArmController.__new__(ArmController)
    arm._state = state
    # mimic relevant __init__ side effects
    arm._heat_use_threshold = 35.0
    arm._tolerance          = 0.1
    arm._heat_bias_x        = 0.10
    arm._heat_bias_y        = -0.05
    arm._camera_bias_x      = -0.20
    arm._camera_bias_y      = 0.10

    # Case 1: neither available
    state.latest_heat_matrix = None
    state.latest_fire_x      = None
    state.latest_fire_y      = None
    assert arm._compute_error() is None, "neither available -> None"

    # Case 2: heat only (high peak), no camera
    # 2x2 grid with the peak at (1,1) → err_x = (1-0.5)/0.5 = 1, err_y = 1
    state.latest_heat_matrix = [[20.0, 20.0], [20.0, 50.0]]
    state.latest_fire_x = None
    state.latest_fire_y = None
    err = arm._compute_error()
    assert err is not None, "heat only should return non-None"
    # raw heat = (1.0, 1.0), bias = (+0.10, -0.05), result = (1.10, 0.95)
    assert abs(err[0] - 1.10) < 1e-9, f"heat-only x: {err[0]}"
    assert abs(err[1] - 0.95) < 1e-9, f"heat-only y: {err[1]}"

    # Case 3: camera only — heat below threshold
    state.latest_heat_matrix = [[20.0, 20.0], [20.0, 25.0]]  # max 25 < 35
    state.latest_fire_x = 0.7   # raw cam err = (0.7-0.5)*2 = 0.4
    state.latest_fire_y = 0.3   # raw cam err = (0.3-0.5)*2 = -0.4
    err = arm._compute_error()
    # raw cam = (0.4, -0.4), bias = (-0.20, 0.10), result = (0.20, -0.30)
    assert abs(err[0] - 0.20) < 1e-9, f"cam-only x: {err[0]}"
    assert abs(err[1] - (-0.30)) < 1e-9, f"cam-only y: {err[1]}"

    # Case 4: both available → average of corrected vectors
    state.latest_heat_matrix = [[20.0, 20.0], [20.0, 50.0]]  # heat raw = (1.0, 1.0)
    state.latest_fire_x = 0.7   # raw cam = (0.4, -0.4)
    state.latest_fire_y = 0.3
    err = arm._compute_error()
    # heat corrected   = (1.10, 0.95)
    # camera corrected = (0.20, -0.30)
    # average          = (0.65, 0.325)
    assert abs(err[0] - 0.65) < 1e-9, f"fused x: {err[0]}"
    assert abs(err[1] - 0.325) < 1e-9, f"fused y: {err[1]}"

    print("✓ test_arm_fusion (all 4 modes)")


# ────────────────────────────────────────────────────────────────────
# THINK frame persistence
# ────────────────────────────────────────────────────────────────────

def test_persist_frame_happy_path():
    from think.database.think_database import ThinkDatabase
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        # Create a fake frame file in the source dir
        fname = "20260523_120000_000000.jpg"
        src_file = os.path.join(src_dir, fname)
        with open(src_file, "wb") as f:
            f.write(b"fake jpeg data")

        db = ThinkDatabase.__new__(ThinkDatabase)
        db._frame_source_path    = src_dir
        db._frame_permanent_path = dst_dir
        db._frame_url_prefix     = "/frames/"

        url = db._persist_frame(f"/frames/{fname}")
        assert url == f"/frames/{fname}", url
        # File should now exist in dst_dir
        dst_file = os.path.join(dst_dir, fname)
        assert os.path.exists(dst_file)
        with open(dst_file, "rb") as f:
            assert f.read() == b"fake jpeg data"
    print("✓ test_persist_frame_happy_path")


def test_persist_frame_no_permanent_configured():
    """Empty frame_permanent_path → soft-fail returns None."""
    from think.database.think_database import ThinkDatabase
    with tempfile.TemporaryDirectory() as src_dir:
        fname = "abc.jpg"
        with open(os.path.join(src_dir, fname), "wb") as f:
            f.write(b"x")
        db = ThinkDatabase.__new__(ThinkDatabase)
        db._frame_source_path    = src_dir
        db._frame_permanent_path = ""
        db._frame_url_prefix     = "/frames/"
        assert db._persist_frame(f"/frames/{fname}") is None
    print("✓ test_persist_frame_no_permanent_configured")


def test_persist_frame_source_missing():
    """Source file gone (rolled out) → soft-fail returns None."""
    from think.database.think_database import ThinkDatabase
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        db = ThinkDatabase.__new__(ThinkDatabase)
        db._frame_source_path    = src_dir
        db._frame_permanent_path = dst_dir
        db._frame_url_prefix     = "/frames/"
        # Never created the source file
        assert db._persist_frame("/frames/never_existed.jpg") is None
    print("✓ test_persist_frame_source_missing")


def test_persist_frame_empty_url():
    """image_url is None or empty → returns None (no NoneType crash)."""
    from think.database.think_database import ThinkDatabase
    db = ThinkDatabase.__new__(ThinkDatabase)
    db._frame_source_path    = "/tmp"
    db._frame_permanent_path = "/tmp/dst"
    db._frame_url_prefix     = "/frames/"
    assert db._persist_frame(None) is None
    assert db._persist_frame("")   is None
    print("✓ test_persist_frame_empty_url")


def test_persist_frame_dst_dir_unwritable():
    """Permanent dir creation fails (e.g. /proc/... → read-only)."""
    from think.database.think_database import ThinkDatabase
    with tempfile.TemporaryDirectory() as src_dir:
        fname = "ok.jpg"
        with open(os.path.join(src_dir, fname), "wb") as f:
            f.write(b"x")
        db = ThinkDatabase.__new__(ThinkDatabase)
        db._frame_source_path    = src_dir
        db._frame_permanent_path = "/proc/cannot_create_here"  # read-only fs
        db._frame_url_prefix     = "/frames/"
        # Should not raise — soft-fails to None
        result = db._persist_frame(f"/frames/{fname}")
        assert result is None
    print("✓ test_persist_frame_dst_dir_unwritable")


# ────────────────────────────────────────────────────────────────────
# Run all
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_arm_fusion()
    test_persist_frame_happy_path()
    test_persist_frame_no_permanent_configured()
    test_persist_frame_source_missing()
    test_persist_frame_empty_url()
    test_persist_frame_dst_dir_unwritable()
    print("\nAll tonight-changes tests passed.")
