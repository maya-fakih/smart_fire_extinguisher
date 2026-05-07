# src/sense/sensors/uart_sensor.py

from __future__ import annotations
import logging
from sense.sensor_base import Sensor

logger = logging.getLogger(__name__)

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    logger.warning("pyserial not installed. UARTSensor unavailable.")


class UARTSensor(Sensor):
    """
    Reads a numeric value from a UART serial device.
    Expects each readline() to return a UTF-8 line containing a float,
    optionally prefixed with a label: "CO2:423.7" or just "423.7".
    Config keys: port, baud_rate (default 9600), timeout_s (default 2.0).
    """

    def __init__(self, path: str, baudrate: int = 9600, timeout: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None

    def _ping(self) -> None:
        if not _SERIAL_AVAILABLE:
            raise RuntimeError("pyserial not installed")
        ser = serial.Serial(port=self.path, baudrate=self.baudrate, timeout=0.1)
        if not ser.is_open:
            ser.close()
            raise IOError(f"UARTSensor '{self.name}': could not open {self.path}")
        ser.close()
        self._serial = serial.Serial(port=self.path, baudrate=self.baudrate, timeout=self.timeout)
        logger.info("UARTSensor '%s': opened %s at %d baud", self.name, self.path, self.baudrate)

    def read(self) -> float:
        if not _SERIAL_AVAILABLE:
            raise RuntimeError("pyserial not installed")
        if self._serial is None or not self._serial.is_open:
            raise IOError(f"UARTSensor '{self.name}': port not open")
        raw = self._serial.readline()
        if not raw:
            raise IOError(f"UARTSensor '{self.name}': read timeout on {self.path}")
        line = raw.decode("utf-8", errors="ignore").strip()
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        if not line:
            raise IOError(f"UARTSensor '{self.name}': no numeric value in line")
        return float(line)

    def read_matrix(self) -> list[float]:
        return []

    def stop(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("UARTSensor '%s': port closed", self.name)