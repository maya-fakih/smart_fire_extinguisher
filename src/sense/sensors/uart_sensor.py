"""
uart_sensor.py — Concrete sensor for UART/serial devices.

Reads a single float value from a serial port. The expected wire protocol
is a UTF-8 line ending in '\n' that contains either:
  • A bare number:            "423.7\n"
  • A labelled number:        "CO2:423.7\n"  (everything after ':' is parsed)

This covers most hobbyist UART sensors (MH-Z19 CO2, GPS NMEA subset,
custom Arduino/STM32 serial reporters, etc.).

Hardware dependency: pyserial
  pip install pyserial

Gracefully absent on non-Pi machines — RuntimeError raised only on read().
"""

from __future__ import annotations

import logging

from sense.sensor import Sensor

logger = logging.getLogger(__name__)

try:
    import serial

    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    logger.warning(
        "pyserial not installed. UARTSensor will raise RuntimeError on read()."
    )


class UARTSensor(Sensor):
    """
    Reads a numeric value from a UART serial device.

    Args:
        path:     Serial device path, e.g. "/dev/ttyAMA0" or "/dev/ttyUSB0".
        baudrate: Serial baud rate (default 9600).
        timeout:  Read timeout in seconds (default 2.0).
        **kwargs: Passed to Sensor.__init__.
    """

    def __init__(
        self,
        path: str,
        baudrate: int = 9600,
        timeout: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path = path
        self.baudrate = baudrate
        self.timeout = timeout

        # Lazy hardware handle
        self._serial: "serial.Serial | None" = None

    # ------------------------------------------------------------------
    # Hardware initialisation
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        """Open the serial port on first use."""
        if self._serial is not None and self._serial.is_open:
            return

        if not _SERIAL_AVAILABLE:
            raise RuntimeError(
                "pyserial is not installed. Cannot read from UARTSensor."
            )

        try:
            self._serial = serial.Serial(
                port=self.path,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            logger.info(
                "UARTSensor '%s': serial port %s opened at %d baud.",
                self.name,
                self.path,
                self.baudrate,
            )
        except Exception as exc:
            logger.error(
                "UARTSensor '%s': failed to open %s — %s",
                self.name,
                self.path,
                exc,
            )
            raise

    # ------------------------------------------------------------------
    # Sensor ABC implementation
    # ------------------------------------------------------------------

    def read(self) -> float:
        """
        Read one line from the serial port and parse a float from it.

        Protocol:
          • Reads up to self.timeout seconds for a '\n'-terminated line.
          • Strips whitespace.
          • If the line contains ':', only the part after ':' is parsed.
          • Remaining string is cast to float.

        Returns:
            float — the parsed sensor value.

        Raises:
            RuntimeError: if pyserial is not installed.
            IOError:      on timeout, decode failure, or parse failure.
        """
        self._init_hardware()

        try:
            raw_bytes = self._serial.readline()
        except Exception as exc:
            raise IOError(
                f"UARTSensor '{self.name}': serial read error — {exc}"
            ) from exc

        if not raw_bytes:
            raise IOError(
                f"UARTSensor '{self.name}': read timeout on {self.path}."
            )

        try:
            line = raw_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise IOError(
                f"UARTSensor '{self.name}': decode error — {exc}"
            ) from exc

        # Handle "LABEL:value" format
        if ":" in line:
            line = line.split(":", 1)[1].strip()

        try:
            value = float(line)
        except ValueError as exc:
            raise IOError(
                f"UARTSensor '{self.name}': cannot parse float from {line!r} — {exc}"
            ) from exc

        return value

    # ------------------------------------------------------------------
    # Matrix support (not applicable for UART scalar sensors)
    # ------------------------------------------------------------------

    def read_matrix(self) -> list[float]:
        """UART sensors are scalar — always returns an empty list."""
        return []

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Close the serial port before stopping the thread."""
        super().stop()
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
                logger.info(
                    "UARTSensor '%s': serial port %s closed.",
                    self.name,
                    self.path,
                )
            except Exception as exc:
                logger.warning(
                    "UARTSensor '%s': error closing port — %s", self.name, exc
                )