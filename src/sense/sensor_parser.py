import json
from typing import List
from sensors.adc_sensor import ADCSensor
from sensors.i2c_sensor import I2CSensor
from sensors.uart_sensor import UARTSensor
from sensors.gpio_sensor import GPIOSensor
from sensor_base import Sensor


class SensorParser:

    def __init__(self, config_path: str):
        # path to config.json file
        self.config_path = config_path

    def _load_config(self) -> dict:
        # read and return the full config.json
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def load(self) -> List[Sensor]:
        # open and read config.json
        config = self._load_config()

        sensors = []
        for name, entry in config["sensors"].items():
            # skip disabled sensors
            if not entry.get("enabled", False):
                print(f"[SKIP] {name} is disabled")
                continue

            # validate the sensor entry first
            if not self._validate(name, entry):
                print(f"[SKIP] {name} failed validation")
                continue

            # build the correct sensor type
            sensor = self._build_sensor(name, entry)
            if sensor:
                sensors.append(sensor)

        return sensors

    def _validate(self, name: str, entry: dict) -> bool:
        # list of fields every sensor must have
        required_fields = [
            "enabled", "active", "interface",
            "raw_min", "raw_max",
            "physical_min", "physical_max",
            "threshold_physical", "unit",
            "valid_min", "valid_max", "max_retries"
        ]

        for field in required_fields:
            if field not in entry:
                # log the missing field
                print(f"[ERROR] {name} is missing field: {field}")
                return False

        return True

    def _build_sensor(self, name: str, entry: dict) -> Sensor:
        # read the interface type
        interface = entry.get("interface")

        # build the correct sensor based on interface
        if interface == "adc":
            return ADCSensor(name=name, **entry)

        elif interface == "i2c":
            return I2CSensor(name=name, **entry)

        elif interface == "uart":
            return UARTSensor(name=name, **entry)

        elif interface == "gpio":
            return GPIOSensor(name=name, **entry)

        else:
            print(f"[ERROR] {name} has unknown interface: {interface}")
            return None

    def save(self, sensors: List[Sensor]) -> None:
        # ⚠️ config writes must go through SystemOrchestrator.update_config()
        # SensorParser should not write directly to config.json at runtime
        # this method is intentionally left without direct file writes
        raise NotImplementedError(
            "Config writes must go through SystemOrchestrator.update_config(). "
            "Do not write directly to config.json from SensorParser."
        )

    def disable(self, name: str) -> None:
        # ⚠️ config writes must go through SystemOrchestrator.update_config()
        # SensorParser should not write directly to config.json at runtime
        # this method is intentionally left without direct file writes
        raise NotImplementedError(
            "Config writes must go through SystemOrchestrator.update_config(). "
            "Do not write directly to config.json from SensorParser."
        )