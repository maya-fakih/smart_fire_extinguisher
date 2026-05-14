# src/sense/sensors/sensor_base.py

from abc import ABC, abstractmethod
from typing import Any
import time
import logging
from exceptions import SensorFaultError

logger = logging.getLogger(__name__)


class Sensor(ABC):
    """
    Abstract base class for all sensor types.

    Every sensor (I2C, UART, GPIO) inherits from this class.
    Provides common functionality: conversion, validation, threshold checking.
    Each subclass only needs to implement read() and _ping().

    read() must return a physical value — either:
      - float: for scalar sensors (ppm, °C, etc.)
      - list[list[float]]: for matrix sensors (e.g. AMG8833 heat grid, already in °C)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Sensor configuration dict. Must include 'name', injected by SensorParser.
        """
        self.name      = config['name']
        self.enabled   = config.get('enabled', True)
        self.interface = config['interface']

        self.raw_min  = float(config['raw_min'])
        self.raw_max  = float(config['raw_max'])

        self.physical_min       = float(config['physical_min'])
        self.physical_max       = float(config['physical_max'])
        self.threshold_physical = float(config['threshold_physical'])
        self.unit               = config['unit']

        self._max_retries = config.get('max_retries', 3)
        self._faulted     = False
        self._fault_count = 0

    @property
    def faulted(self) -> bool:
        return self._faulted

    # ============================================
    # ABSTRACT METHODS (child must implement)
    # ============================================

    @abstractmethod
    def read(self) -> Any:
        """
        Read from hardware and return physical value.

        Child class must:
        1. Read raw value from hardware
        2. Convert to physical units (apply equation or let driver handle it)
        3. Return physical value

        Returns:
            float: for scalar sensors (ppm, °C, etc.)
            list[list[float]]: for matrix sensors (already in physical units)
        """
        pass

    @abstractmethod
    def _ping(self) -> None:
        """
        Test that the hardware is reachable and correctly configured.

        Raises:
            Exception: On any hardware failure
        """
        pass

    # ============================================
    # HARDWARE VALIDATION
    # ============================================

    def ping(self) -> bool:
        try:
            self._ping()
            logger.debug(f"Sensor {self.name}: ping successful")
            return True
        except Exception as e:
            logger.error(
                f"Sensor {self.name}: ping failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            self._faulted = True
            return False

    # ============================================
    # CONVERSION
    # ============================================

    def to_normalized(self, physical_value: float) -> float:
        physical_range = self.physical_max - self.physical_min
        if physical_range == 0:
            return 0.0
        normalized = (physical_value - self.physical_min) / physical_range
        return max(0.0, min(1.0, normalized))

    # ============================================
    # VALIDATION & THRESHOLD
    # ============================================

    def is_valid(self, physical_value: float) -> bool:
        return self.physical_min <= physical_value <= self.physical_max

    def threshold_hit(self, physical_value: float) -> bool:
        return physical_value >= self.threshold_physical

    # ============================================
    # POLLING
    # ============================================

    def poll(self) -> tuple[Any, float, bool]:
        """
        Complete sensor reading cycle with retry logic.
        Called by SensorFuser — handles retries, validation, threshold.

        For matrix sensors, validation and threshold run on max() of the grid.
        The raw grid is returned as-is in the first tuple element.

        Returns:
            tuple: (physical_value, normalized_value, threshold_hit)

        Raises:
            SensorFaultError: If sensor fails after all retries
        """
        for attempt in range(self._max_retries):
            try:
                physical_value = self.read()

                scalar = (
                    max(temp for row in physical_value for temp in row)
                    if isinstance(physical_value, list)
                    else physical_value
                )

                if not self.is_valid(scalar):
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"Sensor {self.name}: Invalid reading {scalar}{self.unit} "
                            f"(valid: {self.physical_min}-{self.physical_max}). "
                            f"Retrying (attempt {attempt + 1}/{self._max_retries})"
                        )
                        time.sleep(0.1)
                        continue
                    else:
                        raise ValueError(
                            f"{self.name}: Invalid reading {scalar}{self.unit} "
                            f"(valid range: {self.physical_min}-{self.physical_max})"
                        )

                normalized_value  = self.to_normalized(scalar)
                threshold_crossed = self.threshold_hit(scalar)

                logger.debug(
                    f"Sensor {self.name}: poll_result | "
                    f"physical={physical_value} {self.unit} | "
                    f"normalized={normalized_value:.3f} | "
                    f"threshold_hit={threshold_crossed}"
                )

                self._fault_count = 0
                return (physical_value, normalized_value, threshold_crossed)

            except Exception as e:
                self._fault_count += 1
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"Sensor {self.name}: Read failed (attempt {attempt + 1}/{self._max_retries}): "
                        f"{type(e).__name__}: {e}"
                    )
                    time.sleep(0.1)
                    continue
                else:
                    error_msg = (
                        f"Sensor {self.name}: failed after {self._max_retries} retries: {str(e)}"
                    )
                    logger.error(
                        f"Sensor {self.name}: {error_msg}",
                        exc_info=True
                    )
                    raise SensorFaultError(error_msg)

        raise SensorFaultError(f"{self.name} polling failed")