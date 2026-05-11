# src/act/actuator_parser.py

import logging

import RPi.GPIO as GPIO

from core.system_state import SystemState
from act.actuators.actuator_base import Actuator
from act.actuators.pump_actuator import PumpActuator
from act.actuators.arm_controller import ArmController

logger = logging.getLogger(__name__)


class ActuatorParser:
    """
    Reads `act.actuators` from config and builds the correct Actuator
    subclass for each entry. Dispatches on the 'type' field (not 'interface'),
    because multiple actuators may share an interface but differ in role
    (pump and arm are both GPIO-driven, but very different beasts).

    ArmController is special — it takes (config, state) instead of (config)
    only, because its tracking thread reads from SystemState continuously.
    """

    _TYPE_MAP = {
        "pump": PumpActuator,
        "arm":  ArmController,
        # "alarm": AlarmActuator,  — not implemented yet (no hardware)
    }

    @classmethod
    def build_actuators(cls, config: dict, state: SystemState) -> list[Actuator]:
        # Centralised GPIO mode setup — done once here, not in each actuator.
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        actuators = []
        act_cfg = config.get("act", {}).get("actuators", {})

        for actuator_name, actuator_cfg in act_cfg.items():
            if not actuator_cfg.get("enabled", True):
                continue

            actuator_type = actuator_cfg.get("type", "").lower()
            if actuator_type not in cls._TYPE_MAP:
                logger.warning(
                    f"Unknown actuator type '{actuator_type}' for '{actuator_name}'. Skipping."
                )
                continue

            actuator_cfg = {**actuator_cfg, "name": actuator_name}
            actuator_class = cls._TYPE_MAP[actuator_type]

            try:
                if actuator_type == "arm":
                    actuators.append(actuator_class(actuator_cfg, state))
                else:
                    actuators.append(actuator_class(actuator_cfg))
                logger.info(f"ActuatorParser: built {actuator_type} '{actuator_name}'")
            except Exception as e:
                logger.error(
                    f"ActuatorParser: failed to build '{actuator_name}' - "
                    f"{type(e).__name__}: {e}",
                    exc_info=True,
                )

        return actuators