# src/sense/sensors/i2c_sensor.py

import math
from sense.sensors.sensor_base import Sensor
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_amg88xx


class I2CSensor(Sensor):

    _DEVICE_HANDLERS = {
        "ads1115": {
            "setup": "_setup_ads1115",
            "read":  "_read_ads1115",
        },
        "amg8833": {
            "setup": "_setup_amg8833",
            "read":  "_read_amg8833",
        },
    }

    def __init__(self, sensor_cfg: dict):
        """
        Args:
            sensor_cfg: Full sensor config dict. Must include:
                - 'name'         injected by SensorParser
                - '_bus_object'  resolved busio.I2C instance, injected by SensorParser
        """
        super().__init__(sensor_cfg)

        self.bus = sensor_cfg['_bus_object']
        if self.bus is None:
            raise ValueError(
                f"{self.name}: '_bus_object' is None. "
                f"Check that bus '{sensor_cfg.get('bus')}' exists in system.i2c_buses."
            )

        self.device_type = sensor_cfg['device_type']
        self.address = int(sensor_cfg['address'], 16)

        if self.device_type not in self._DEVICE_HANDLERS:
            raise ValueError(f"{self.name}: Unsupported I2C device type '{self.device_type}'")

        setup_method = getattr(self, self._DEVICE_HANDLERS[self.device_type]["setup"])
        setup_method(sensor_cfg)

        self._equation  = sensor_cfg.get('equation', None)
        self._eq_params = sensor_cfg.get('eq_params', {})

    # ------------------------------------------------------------------
    # Device setup
    # ------------------------------------------------------------------

    def _setup_ads1115(self, config: dict):
        self.channel = config['channel']
        self.gain = config.get('gain', 1)

        self._ads = ADS.ADS1115(self.bus, address=self.address)
        self._ads.gain = self.gain

        
        if self.channel not in range(4):
            raise ValueError(f"{self.name}: Invalid ADS1115 channel '{self.channel}'")
        self._channel = AnalogIn(self._ads, self.channel)

    def _setup_amg8833(self, config: dict):
        self._amg = adafruit_amg88xx.AMG88XX(self.bus, addr=self.address)

    # ------------------------------------------------------------------
    # Hardware ping
    # ------------------------------------------------------------------

    def _ping(self) -> None:
        while not self.bus.try_lock():
            pass
        try:
            devices = self.bus.scan()
            if self.address not in devices:
                raise IOError(f"{self.name}: Device not found at {hex(self.address)}")
        finally:
            self.bus.unlock()

    # ------------------------------------------------------------------
    # Read — returns physical value
    # ------------------------------------------------------------------

    def read(self):
        """
        Read from hardware and return physical value.

        ADS1115: reads raw ADC integer, applies equation from config → float
        AMG8833: driver already returns °C → list[list[float]] returned as-is
        """
        read_method = getattr(self, self._DEVICE_HANDLERS[self.device_type]["read"])
        return read_method()

    def _read_ads1115(self) -> float:
        raw = float(self._channel.value)
        if raw <= 0:
            raw = 1.0          # bad/disconnected reading — clamp to avoid complex number
        if self._equation is None:
            raise ValueError(
                f"{self.name}: 'equation' missing from config. "
                f"Cannot convert raw ADC value to physical units."
            )
        context = {"raw": raw, "__builtins__": {}, **vars(math), **self._eq_params}
        return float(eval(self._equation, context))

    def _read_amg8833(self) -> list:
        """
        AMG8833 driver returns pixels already in °C.
        Full 8x8 grid returned — ThinkEngine handles aggregation.
        """
        return [list(row) for row in self._amg.pixels]