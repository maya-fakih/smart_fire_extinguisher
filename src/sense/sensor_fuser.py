# src/sense/sensor_fuser.py

import time
import threading
import logging
from datetime import datetime
from typing import Any

import board
import busio

from core.system_state import SystemState
from sense.sensor_parser import SensorParser
from sense.snapshot import SensorSnapshot

logger = logging.getLogger(__name__)


class SensorFuser:
    """
    Owns all sensors, runs polling threads, builds SensorSnapshots,
    and writes them to SystemState.sense_queue when thresholds are crossed.
    """

    def __init__(self, config: dict, state: SystemState, notifier=None):
        system_cfg = config.get("system", {})
        self._polling_idle_ms   = system_cfg.get("polling_interval_idle_ms",   10000) / 1000.0
        self._polling_active_ms = system_cfg.get("polling_interval_active_ms", 1000)  / 1000.0

        # Init I2C buses from config before building sensors — stored for cleanup on restart
        self._i2c_buses = self._init_i2c_buses(system_cfg.get("i2c_buses", {}))

        # Build sensors — config no longer needed after this
        self._sensors = SensorParser.build_sensors(config, self._i2c_buses)

        self._state   = state
        self._notifier = notifier
        self._running = False
        self._threads = []
        self._lock    = threading.Lock()

        self._latest_readings:   dict[str, Any]   = {}  # sensor_name -> physical value (float or grid)
        self._latest_normalized: dict[str, float] = {}  # sensor_name -> normalized float
        self._threshold_flags:   dict[str, bool]  = {}  # sensor_name -> threshold crossed
        self._faulted_sensors:   list[dict]        = []  # [{"name": ..., "faulted_at": ...}]

    # ------------------------------------------------------------------
    # Bus initialisation
    # ------------------------------------------------------------------

    def _init_i2c_buses(self, buses_cfg: dict) -> dict:
        """
        Build busio.I2C objects from the system.i2c_buses config section.

        Each entry: { "scl": "SCL", "sda": "SDA" }
        Pin names map to board attributes (board.SCL, board.D3, etc.)

        Returns:
            dict of bus_name -> busio.I2C
        """
        buses = {}
        for bus_name, bus_cfg in buses_cfg.items():
            scl_pin = getattr(board, bus_cfg["scl"])
            sda_pin = getattr(board, bus_cfg["sda"])
            buses[bus_name] = busio.I2C(scl_pin, sda_pin)
        return buses

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        logger.info("SensorFuser: starting")
        self._running = True
        self._state.sense_running = True
        self._update_state_sensor_counts()
        self._spawn_sensor_threads()
        logger.info(f"SensorFuser: spawned {len(self._threads)} sensor threads")
        self._main_loop()
        logger.info("SensorFuser: stopped")

    def stop(self):
        logger.info("SensorFuser: stop requested")
        self._running = False

    def cleanup(self):
        """Deinit all I2C buses — call before reinitializing layers on restart."""
        for bus in self._i2c_buses.values():
            try:
                bus.deinit()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def _spawn_sensor_threads(self):
        for sensor in self._sensors:
            if sensor.enabled:
                thread = threading.Thread(
                    target=self._sensor_loop,
                    args=(sensor,),
                    name=f"Sensor-{sensor.name}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)

    def _main_loop(self):
        while self._running and self._state.system_running:
            time.sleep(0.5)

        self._running = False
        for thread in self._threads:
            thread.join(timeout=1)
        self._state.sense_running = False

    # ------------------------------------------------------------------
    # Per-sensor thread loop
    # ------------------------------------------------------------------

    def _sensor_loop(self, sensor):
        while self._running and self._state.system_running:
            try:
                # Respect runtime enable/disable from website (soft override).
                override = self._state.sensor_overrides.get(sensor.name)
                effective_enabled = override if override is not None else sensor.enabled
                if not effective_enabled:
                    time.sleep(1.0)
                    continue

                physical, normalized, threshold_hit = sensor.poll()

                if sensor.name == "heat_grid":
                    self._state.latest_heat_matrix = physical

                with self._lock:
                    self._latest_readings[sensor.name]   = physical
                    self._latest_normalized[sensor.name] = normalized
                    self._threshold_flags[sensor.name]   = threshold_hit
                    any_triggered = any(self._threshold_flags.values())

                if threshold_hit:
                    logger.info(
                        f"Sensor {sensor.name}: threshold_crossed | "
                        f"value={physical} | threshold={sensor.threshold_physical}"
                    )
                    self._on_threshold_triggered()
                elif not any_triggered:
                    self._state.sensor_triggered = False

                interval = (
                    self._polling_active_ms
                    if self._state.sensor_triggered
                    else self._polling_idle_ms
                )
                time.sleep(interval)

            except Exception as e:
                logger.error(
                    f"Sensor {sensor.name}: exception in _sensor_loop - "
                    f"{type(e).__name__}: {e}",
                    exc_info=True
                )
                if not sensor.faulted:
                    self._mark_sensor_faulted(sensor)
                break

    # ------------------------------------------------------------------
    # Threshold handling
    # ------------------------------------------------------------------

    def _on_threshold_triggered(self):
        self._state.sensor_triggered = True
        self._push_snapshot()

    def _push_snapshot(self):
        with self._lock:
            # Use stored threshold flags — set by poll(), based on per-sensor threshold_physical
            triggered_names = [
                name for name, hit in self._threshold_flags.items() if hit
            ]

            snapshot = SensorSnapshot(
                timestamp=datetime.now(),
                sensor_readings=dict(self._latest_readings),
                sensor_normalized=dict(self._latest_normalized),
                enabled_sensors=[s.name for s in self._sensors if s.enabled and not s.faulted],
                triggered_sensors=triggered_names,
                faulty_sensors=[f["name"] for f in self._faulted_sensors],
            )

        logger.info(
            f"SensorSnapshot: pushed to sense_queue | "
            f"timestamp={snapshot.timestamp.isoformat()} | "
            f"triggered_sensors={snapshot.triggered_sensors}"
        )
        self._state.sense_queue.put(snapshot)

    # ------------------------------------------------------------------
    # Fault handling
    # ------------------------------------------------------------------

    def _mark_sensor_faulted(self, sensor):
        sensor._faulted = True
        with self._lock:
            self._faulted_sensors.append({
                "name": sensor.name,
                "faulted_at": datetime.now().isoformat(),
            })
        logger.info(f"Sensor {sensor.name}: marked as faulted")
        self._state.faulted_sensors = list(self._faulted_sensors)
        self._update_state_sensor_counts()
        # Fire notification — website/email picks this up
        if self._notifier is not None:
            from notify import EventType
            self._notifier.notify(
                EventType.SENSOR_FAULTED,
                payload={"sensor": sensor.name},
                source_layer="sense",
            )
            # Check if every sensor has now faulted — escalates to critical
            if self._state.active_sensor_count == 0:
                self._notifier.notify(
                    EventType.ALL_SENSORS_FAULTED,
                    payload={"faulted": [f["name"] for f in self._faulted_sensors]},
                    source_layer="sense",
                )

    def _update_state_sensor_counts(self):
        active_count = sum(
            1 for s in self._sensors
            if s.enabled and not s.faulted
        )
        self._state.active_sensor_count = active_count