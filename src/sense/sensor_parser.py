from sense.sensors.sensor_base import Sensor
from sense.sensors import ADCSensor, GPIOSensor, I2CSensor, UARTSensor


class SensorParser:
    """
    Reads the 'sensors' section of config.json and builds the correct
    Sensor subclass for each entry.

    This is the ONLY place in the codebase where sensor types are branched on.
    Once the list is returned, SensorFuser treats every sensor as a generic Sensor.
    """

    _INTERFACE_MAP = {
        "adc": ADCSensor,
        "gpio": GPIOSensor,
        "i2c": I2CSensor,
        "uart": UARTSensor,
    }

    @classmethod
    def build_sensors(cls, config: dict) -> list[Sensor]:
        """
        Parse the sensors section of config.json and return a list of
        Sensor objects ready for SensorFuser to use.

        Args:
            config: The full config dict from orchestrator.

        Returns:
            A list of Sensor objects (may be empty if no sensors enabled).
        """
        sensors = []
        sensor_configs = config.get("sensors", {})

        for sensor_name, sensor_cfg in sensor_configs.items():
            if not sensor_cfg.get("enabled", True):
                continue

            interface = sensor_cfg.get("interface", "").lower()
            if interface not in cls._INTERFACE_MAP:
                # TODO: log to file or trigger notification when notify layer is ready
                print(
                    f"[SensorParser] WARNING: Unknown interface '{interface}' "
                    f"for sensor '{sensor_name}'. Skipping."
                )
                continue

            sensor_class = cls._INTERFACE_MAP[interface]
            sensor = sensor_class(sensor_name, sensor_cfg)
            sensors.append(sensor)

        return sensors