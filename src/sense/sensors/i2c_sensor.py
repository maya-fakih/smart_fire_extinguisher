"""
i2c_sensor.py — Concrete sensor for generic I2C devices.

Covers both scalar I2C sensors (e.g. BME280 temperature/humidity) and
matrix I2C sensors (e.g. MLX90640 32×24 thermal array, AMG8831 8×8 grid).

Matrix detection: if self.matrix_shape is set in config, read() returns the
scalar mean of the full pixel array, and read_matrix() returns the raw flat
list.  SensorFuser calls read_matrix() separately when building raw_matrices
for the SensorSnapshot.

Hardware dependencies (install whichever your sensor needs):
  pip install smbus2
  pip install adafruit-circuitpython-mlx90640   # for MLX90640
  pip install adafruit-circuitpython-amg88xx    # for AMG8831
  pip install adafruit-circuitpython-bme280     # for BME280

The imports are guarded so the module loads cleanly on Windows/macOS dev
machines; a RuntimeError is raised only when read() is actually called on
hardware that is absent.
"""

from __future__ import annotations

import logging
import struct

from sense.sensor import Sensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional hardware imports
# ---------------------------------------------------------------------------
try:
    import smbus2

    _SMBUS_AVAILABLE = True
except ImportError:
    _SMBUS_AVAILABLE = False
    logger.warning("smbus2 not installed. I2CSensor scalar reads unavailable.")

try:
    import board
    import busio
    import adafruit_mlx90640

    _MLX90640_AVAILABLE = True
except ImportError:
    _MLX90640_AVAILABLE = False

try:
    import board
    import busio
    import adafruit_amg88xx

    _AMG88XX_AVAILABLE = True
except ImportError:
    _AMG88XX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Known matrix sensor drivers keyed by I2C address (lower-case hex string)
# ---------------------------------------------------------------------------
_MATRIX_DRIVERS = {
    "0x33": "mlx90640",   # Melexis MLX90640 32×24 thermal camera
    "0x69": "mlx90640",   # alternate MLX90640 address
    "0x68": "amg88xx",    # Panasonic AMG8831 8×8 thermal grid
}


class I2CSensor(Sensor):
    """
    Generic I2C sensor supporting scalar and matrix readings.

    Args:
        address:  I2C device address as a hex string, e.g. "0x69".
        i2c_bus:  Linux I2C bus index (default 1 for Raspberry Pi).
        register: Optional register address for scalar reads (default 0x00).
        **kwargs: Passed to Sensor.__init__.
    """

    def __init__(
        self,
        address: str,
        i2c_bus: int = 1,
        register: int = 0x00,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.address = address.lower()
        self.i2c_bus = i2c_bus
        self.register = register

        # Resolved integer address
        self._address_int: int = int(self.address, 16)

        # Detected driver type ("mlx90640", "amg88xx", "generic", or None)
        self._driver_type: str | None = _MATRIX_DRIVERS.get(self.address)

        # Lazy hardware handles
        self._bus = None           # smbus2.SMBus — for generic scalar reads
        self._mlx = None           # adafruit_mlx90640 instance
        self._amg = None           # adafruit_amg88xx instance
        self._i2c = None           # busio.I2C — shared for Adafruit drivers

        # Cached flat pixel list for matrix sensors
        self._last_matrix: list[float] = []

    # ------------------------------------------------------------------
    # Hardware initialisation
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        """Lazy-initialise the correct driver for this sensor's address."""
        if self._driver_type == "mlx90640":
            self._init_mlx90640()
        elif self._driver_type == "amg88xx":
            self._init_amg88xx()
        else:
            self._init_generic_smbus()

    def _init_generic_smbus(self) -> None:
        """Open an smbus2 handle for generic scalar I2C sensors."""
        if self._bus is not None:
            return
        if not _SMBUS_AVAILABLE:
            raise RuntimeError(
                "smbus2 is not installed. Cannot read from I2CSensor."
            )
        try:
            self._bus = smbus2.SMBus(self.i2c_bus)
            logger.info(
                "I2CSensor '%s': smbus2 opened on bus %d, address %s.",
                self.name,
                self.i2c_bus,
                self.address,
            )
        except Exception as exc:
            logger.error(
                "I2CSensor '%s': smbus2 init failed — %s", self.name, exc
            )
            raise

    def _init_mlx90640(self) -> None:
        """Initialise the MLX90640 32×24 thermal camera via Adafruit driver."""
        if self._mlx is not None:
            return
        if not _MLX90640_AVAILABLE:
            raise RuntimeError(
                "adafruit-circuitpython-mlx90640 not installed."
            )
        try:
            self._i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
            self._mlx = adafruit_mlx90640.MLX90640(self._i2c)
            self._mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
            logger.info("I2CSensor '%s': MLX90640 initialised.", self.name)
        except Exception as exc:
            logger.error(
                "I2CSensor '%s': MLX90640 init failed — %s", self.name, exc
            )
            raise

    def _init_amg88xx(self) -> None:
        """Initialise the AMG8831 8×8 thermal grid via Adafruit driver."""
        if self._amg is not None:
            return
        if not _AMG88XX_AVAILABLE:
            raise RuntimeError(
                "adafruit-circuitpython-amg88xx not installed."
            )
        try:
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._amg = adafruit_amg88xx.AMG88XX(self._i2c)
            logger.info("I2CSensor '%s': AMG8831 initialised.", self.name)
        except Exception as exc:
            logger.error(
                "I2CSensor '%s': AMG8831 init failed — %s", self.name, exc
            )
            raise

    # ------------------------------------------------------------------
    # Sensor ABC implementation
    # ------------------------------------------------------------------

    def read(self) -> float:
        """
        Read a scalar value from the I2C device.

        For matrix sensors (MLX90640, AMG8831):
            Fetches the full pixel array, caches it in self._last_matrix,
            and returns the mean temperature as the scalar representative value.

        For generic I2C sensors:
            Reads two bytes from self.register and interprets them as a
            big-endian unsigned 16-bit integer.

        Returns:
            float — the scalar reading (raw ADC count or mean temperature).
        """
        self._init_hardware()

        if self._driver_type == "mlx90640":
            return self._read_mlx90640()
        if self._driver_type == "amg88xx":
            return self._read_amg88xx()
        return self._read_generic()

    # ------------------------------------------------------------------
    # Driver-specific read helpers
    # ------------------------------------------------------------------

    def _read_mlx90640(self) -> float:
        frame = [0.0] * 768  # 32 × 24 pixels
        try:
            self._mlx.getFrame(frame)
        except Exception as exc:
            raise IOError(f"MLX90640 getFrame failed: {exc}") from exc

        self._last_matrix = frame
        mean = sum(frame) / len(frame)
        return mean

    def _read_amg88xx(self) -> float:
        pixels = self._amg.pixels  # 8×8 list of lists
        flat = [temp for row in pixels for temp in row]
        self._last_matrix = flat
        mean = sum(flat) / len(flat)
        return mean

    def _read_generic(self) -> float:
        """Read 2 bytes from self.register, return as unsigned 16-bit int."""
        try:
            data = self._bus.read_i2c_block_data(
                self._address_int, self.register, 2
            )
            value = struct.unpack(">H", bytes(data))[0]
            return float(value)
        except Exception as exc:
            raise IOError(
                f"I2CSensor '{self.name}' read error at register "
                f"0x{self.register:02X}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Matrix support
    # ------------------------------------------------------------------

    def read_matrix(self) -> list[float]:
        """
        Return the most recently captured full pixel array as a flat list.

        Returns an empty list for non-matrix sensors.
        SensorFuser calls this after poll() to populate raw_matrices in
        the SensorSnapshot.
        """
        return list(self._last_matrix)