# src/sense/sensor_base.py

from abc import ABC, abstractmethod
from exceptions import SensorFaultError


class Sensor(ABC):

    def __init__(
        self,
        name: str,
        raw_min: float = 0,
        raw_max: float = 4095,
        physical_min: float = 0,
        physical_max: float = 100,
        threshold_physical: float = 50,
        valid_min: float = None,
        valid_max: float = None,
        max_retries: int = 3,
        **kwargs,
    ):
        self._name = name
        self._raw_min = float(raw_min)
        self._raw_max = float(raw_max)
        self._physical_min = float(physical_min)
        self._physical_max = float(physical_max)
        self._threshold_physical = float(threshold_physical)
        self._valid_min = float(valid_min) if valid_min is not None else self._physical_min
        self._valid_max = float(valid_max) if valid_max is not None else self._physical_max
        self._max_retries = int(max_retries)
        self._faulted = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def faulted(self) -> bool:
        return self._faulted

    @abstractmethod
    def read(self) -> float:
        """Return the raw sensor value directly from hardware."""
        pass

    @abstractmethod
    def _ping(self) -> None:
        """
        Test that the hardware is reachable and correctly configured.
        Must raise on any failure — exception is caught by ping().
        """
        pass

    def ping(self) -> bool:
        """
        Validate sensor hardware at startup.
        Calls _ping(); sets _faulted=True and returns False on any exception.
        """
        try:
            self._ping()
            return True
        except Exception:
            self._faulted = True
            return False

    def poll(self) -> tuple[float, float, bool]:
        """
        Read the sensor up to max_retries times.
        Returns (physical_value, normalized_value, threshold_hit).
        A reading outside [valid_min, valid_max] counts as a failed attempt.
        Sets _faulted=True and raises SensorFaultError after all retries fail.
        """
        last_exc: Exception = RuntimeError("no attempts made")
        for _ in range(self._max_retries):
            try:
                raw = self.read()
                physical = self.to_physical(raw)
                if not (self._valid_min <= physical <= self._valid_max):
                    last_exc = SensorFaultError(
                        f"{self._name}: reading {physical:.3f} outside valid range "
                        f"[{self._valid_min}, {self._valid_max}]"
                    )
                    continue
                return physical, self.to_normalized(physical), self.threshold_hit(physical)
            except SensorFaultError:
                raise
            except Exception as exc:
                last_exc = exc

        self._faulted = True
        raise SensorFaultError(
            f"{self._name}: failed after {self._max_retries} retries — {last_exc}"
        )

    def to_physical(self, raw: float) -> float:
        span_raw = self._raw_max - self._raw_min
        if span_raw == 0:
            return self._physical_min
        return self._physical_min + (raw - self._raw_min) * (
            (self._physical_max - self._physical_min) / span_raw
        )

    def to_normalized(self, physical: float) -> float:
        span = self._physical_max - self._physical_min
        if span == 0:
            return 0.0
        return max(0.0, min(1.0, (physical - self._physical_min) / span))

    def threshold_hit(self, physical: float) -> bool:
        return physical >= self._threshold_physical