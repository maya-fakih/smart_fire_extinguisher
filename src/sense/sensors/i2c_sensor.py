# src/sense/sensors/i2c_sensor.py

from __future__ import annotations
import logging
import struct
from sense.sensor_base import Sensor

logger = logging.getLogger(__name__)

try:
    import smbus2
    _SMBUS_AVAILABLE = True
except ImportError:
    _SMBUS_AVAILABLE = False
    logger.warning("smbus2 not installed. I2CSensor unavailable.")

try:
    import board, busio, adafruit_mlx90640
    _MLX90640_AVAILABLE = True
except ImportError:
    _MLX90640_AVAILABLE = False

try:
    import board, busio, adafruit_amg88xx
    _AMG88XX_AVAILABLE = True
except ImportError:
    _AMG88XX_AVAILABLE = False

_MATRIX_DRIVERS = {
    "0x33": "mlx90640",
    "0x69": "mlx90640",
    "0x68": "amg88xx",
}


class I2CSensor(Sensor):
    """
    Generic I2C sensor. Supports scalar reads and thermal matrix sensors
    (MLX90640 at 0x33/0x69, AMG8831 at 0x68). For all other addresses
    reads 2 bytes from the given register and returns them as uint16.
    Config keys: address (hex string), bus (default 1), register (default 0x00).
    """

    def __init__(self, address: str, i2c_bus: int = 1, register: int = 0x00, **kwargs):
        super().__init__(**kwargs)
        self.address = str(address).lower()
        self.i2c_bus = i2c_bus
        self.register = register
        self._address_int = int(self.address, 16)
        self._driver_type = _MATRIX_DRIVERS.get(self.address)
        self._bus = None
        self._mlx = None
        self._amg = None
        self._i2c = None
        self._last_matrix: list[float] = []

    def _ping(self) -> None:
        if not _SMBUS_AVAILABLE:
            raise RuntimeError("smbus2 not installed")
        with smbus2.SMBus(self.i2c_bus) as bus:
            bus.write_quick(self._address_int)
        self._init_hardware()

    def _init_hardware(self) -> None:
        if self._driver_type == "mlx90640":
            if not _MLX90640_AVAILABLE:
                raise RuntimeError("adafruit-circuitpython-mlx90640 not installed")
            self._i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
            self._mlx = adafruit_mlx90640.MLX90640(self._i2c)
            self._mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
            logger.info("I2CSensor '%s': MLX90640 initialised", self.name)
        elif self._driver_type == "amg88xx":
            if not _AMG88XX_AVAILABLE:
                raise RuntimeError("adafruit-circuitpython-amg88xx not installed")
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._amg = adafruit_amg88xx.AMG88XX(self._i2c)
            logger.info("I2CSensor '%s': AMG8831 initialised", self.name)
        else:
            if not _SMBUS_AVAILABLE:
                raise RuntimeError("smbus2 not installed")
            self._bus = smbus2.SMBus(self.i2c_bus)
            logger.info("I2CSensor '%s': smbus2 opened bus %d addr %s", self.name, self.i2c_bus, self.address)

    def read(self) -> float:
        if self._driver_type == "mlx90640":
            return self._read_mlx90640()
        if self._driver_type == "amg88xx":
            return self._read_amg88xx()
        return self._read_generic()

    def _read_mlx90640(self) -> float:
        if self._mlx is None:
            raise IOError(f"I2CSensor '{self.name}': MLX90640 not initialised")
        frame = [0.0] * 768
        self._mlx.getFrame(frame)
        self._last_matrix = frame
        return sum(frame) / len(frame)

    def _read_amg88xx(self) -> float:
        if self._amg is None:
            raise IOError(f"I2CSensor '{self.name}': AMG8831 not initialised")
        flat = [t for row in self._amg.pixels for t in row]
        if len(flat) != 64:
            raise IOError(f"I2CSensor '{self.name}': AMG8831 returned {len(flat)} pixels, expected 64")
        self._last_matrix = flat
        return sum(flat) / len(flat)

    def _read_generic(self) -> float:
        if self._bus is None:
            raise IOError(f"I2CSensor '{self.name}': I2C bus not initialised")
        data = self._bus.read_i2c_block_data(self._address_int, self.register, 2)
        if len(data) != 2:
            raise IOError(f"I2CSensor '{self.name}': expected 2 bytes, got {len(data)}")
        return float(struct.unpack(">H", bytes(data))[0])

    def read_matrix(self) -> list[float]:
        return list(self._last_matrix)