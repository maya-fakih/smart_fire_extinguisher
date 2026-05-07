import time
import threading
from datetime import datetime
from core.system_state import SystemState
from sense.sensor_parser import SensorParser
from sense.snapshot import SensorSnapshot


class SensorFuser:
    """
    Owns all sensors, runs polling threads, builds SensorSnapshots,
    and writes them to SystemState.sense_queue when thresholds are crossed.

    Runs continuously from boot as a separate OS process.
    """

    def __init__(self, config: dict, state: SystemState):
        # Pull config values we need, then discard the dict
        system_cfg = config.get("system", {})
        self._polling_idle_ms = system_cfg.get("polling_interval_idle_ms", 10000) / 1000.0
        self._polling_active_ms = system_cfg.get("polling_interval_active_ms", 1000) / 1000.0

        # Build sensors via parser — after this, config is no longer needed
        self._sensors = SensorParser.build_sensors(config)

        self._state = state
        self._running = False
        self._threads = []
        self._lock = threading.Lock()
        self._latest_readings = {}       # sensor_name -> physical value
        self._latest_normalized = {}     # sensor_name -> normalized value
        self._faulted_sensors = []       # list of {"name": ..., "faulted_at": ...}

    # ------------------------------------------------------------------
    # Lifecycle (called by orchestrator)
    # ------------------------------------------------------------------

    def start(self):
        """Main entry point. Spawns threads and blocks until shutdown."""
        self._running = True
        self._state.sense_running = True
        self._update_state_sensor_counts()
        self._spawn_sensor_threads()
        self._main_loop()

    def stop(self):
        """Signal the main loop to exit."""
        self._running = False

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def _spawn_sensor_threads(self):
        """Create and start one daemon thread per enabled sensor."""
        for sensor in self._sensors:
            if sensor.enabled:
                thread = threading.Thread(
                    target=self._sensor_loop,
                    args=(sensor,),
                    name=f"Sensor-{sensor.name}",
                    daemon=True
                )
                thread.start()
                self._threads.append(thread)

    def _main_loop(self):
        """Keep the process alive and watch for shutdown signal."""
        while self._running and self._state.system_running:
            time.sleep(0.5)

        # Clean shutdown
        self._running = False
        for thread in self._threads:
            thread.join(timeout=1)
        self._state.sense_running = False

    # ------------------------------------------------------------------
    # Per-sensor thread loop
    # ------------------------------------------------------------------

    def _sensor_loop(self, sensor):
        """Runs in its own thread. Polls one sensor continuously."""
        while self._running and self._state.system_running:
            try:
                physical, normalized, threshold_hit = sensor.poll()

                # Store latest readings (thread-safe)
                with self._lock:
                    self._latest_readings[sensor.name] = physical
                    self._latest_normalized[sensor.name] = normalized

                # On threshold crossing, build snapshot and push to queue
                if threshold_hit:
                    self._on_threshold_triggered()

                # Sleep at appropriate rate
                if self._state.sensor_triggered:
                    time.sleep(self._polling_active_ms)
                else:
                    time.sleep(self._polling_idle_ms)

            except Exception:
                # Sensor returned bad data — increment fault counter
                sensor._fault_count += 1
                if sensor._fault_count >= sensor._max_retries:
                    self._mark_sensor_faulted(sensor)
                time.sleep(self._polling_active_ms)

    # ------------------------------------------------------------------
    # Threshold handling
    # ------------------------------------------------------------------

    def _on_threshold_triggered(self):
        """Called by any sensor thread when its threshold is crossed."""
        self._state.sensor_triggered = True
        self._push_snapshot()

    def _push_snapshot(self):
        """Build a SensorSnapshot from latest readings and push to sense_queue."""
        with self._lock:
            triggered_names = [
                name for name, val in self._latest_normalized.items()
                if val > 0.5  # threshold_hit means normalized exceeds config threshold
            ]

            snapshot = SensorSnapshot(
                timestamp=datetime.now(),
                sensor_readings=dict(self._latest_readings),
                sensor_normalized=dict(self._latest_normalized),
                enabled_sensors=[s.name for s in self._sensors if s.enabled and not s.faulted],
                triggered_sensors=triggered_names,
                faulty_sensors=[f["name"] for f in self._faulted_sensors],
            )

        self._state.sense_queue.put(snapshot)

    # ------------------------------------------------------------------
    # Fault handling
    # ------------------------------------------------------------------

    def _mark_sensor_faulted(self, sensor):
        """Mark a sensor as faulted and update SystemState."""
        sensor._faulted = True
        with self._lock:
            self._faulted_sensors.append({
                "name": sensor.name,
                "faulted_at": datetime.now().isoformat(),
            })
        self._state.faulted_sensors = list(self._faulted_sensors)
        self._update_state_sensor_counts()

    def _update_state_sensor_counts(self):
        """Update SystemState with current active sensor count."""
        active_count = sum(
            1 for s in self._sensors
            if s.enabled and not s.faulted
        )
        self._state.active_sensor_count = active_count