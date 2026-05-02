import time
import threading
from typing import List, Tuple
from datetime import datetime
from sensor_parser import SensorParser
from sensor_base import Sensor
from snapshot import SensorSnapshot


class SensorFuser:

    def __init__(self, config_path: str):
        # load all sensors from config.json using the parser
        parser = SensorParser(config_path)
        self.sensors: List[Sensor] = parser.load()

        # polling intervals from config
        self.interval_idle_ms = 10000
        self.interval_active_ms = 1000

        # threads — one per sensor
        self._threads: List[threading.Thread] = []

        # monitor thread — watches all sensors
        self._monitor_thread = None

        # is the fuser running?
        self._running = False

    def start(self) -> None:
        self._running = True

        # start each sensor in its own thread
        for sensor in self.sensors:
            t = threading.Thread(
                target=sensor.run_thread,
                args=(self.interval_idle_ms,),
                daemon=True
            )
            self._threads.append(t)
            t.start()
            print(f"[INFO] Started thread for sensor: {sensor.name}")

        # start the monitor thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        print("[INFO] SensorFuser started!")

    def stop(self) -> None:
        self._running = False

        # stop all sensors
        for sensor in self.sensors:
            sensor.stop()

        print("[INFO] SensorFuser stopped!")

    def evaluate(self) -> Tuple[bool, List[Sensor]]:
        # check which sensors are above their threshold
        triggered = []
        for sensor in self.sensors:
            if sensor.threshold_hit():
                triggered.append(sensor)

        # return True if any sensor triggered
        return len(triggered) > 0, triggered

    def snapshot(self) -> SensorSnapshot:
        # collect readings from all active sensors
        readings = {}
        normalized = {}
        triggered_sensors = []
        disabled_sensors = []
        raw_matrices = {}

        for sensor in self.sensors:
            # collect disabled sensors
            if sensor.fault:
                disabled_sensors.append(sensor.name)
                continue

            # collect readings
            readings[sensor.name] = sensor.latest_physical
            normalized[sensor.name] = sensor.to_normalized()

            # collect triggered sensors
            if sensor.triggered:
                triggered_sensors.append(sensor.name)

            # collect matrix data if exists
            if hasattr(sensor, 'matrix_shape'):
                raw_matrices[sensor.name] = sensor.latest_raw

        # build and return the snapshot
        return SensorSnapshot(
            timestamp=datetime.now(),
            sensor_readings=readings,
            normalized=normalized,
            triggered_sensors=triggered_sensors,
            disabled_sensors=disabled_sensors,
            raw_matrices=raw_matrices
        )

    def _monitor_loop(self) -> None:
        # keep checking sensors while running
        while self._running:
            triggered, triggered_list = self.evaluate()

            if triggered:
                print(f"[ALERT] Sensors triggered: {[s.name for s in triggered_list]}")

                # take a snapshot
                snap = self.snapshot()

                # send snapshot to next layer
                self.emit_trigger(snap)

                # check more frequently when fire detected
                time.sleep(self.interval_active_ms / 1000)
            else:
                # check less frequently when calm
                time.sleep(self.interval_idle_ms / 1000)

    def emit_trigger(self, snapshot: SensorSnapshot) -> None:
        # this sends the snapshot to SystemState sense_queue
        # SystemOrchestrator connects this at boot time
        print(f"[TRIGGER] Snapshot emitted at {snapshot.timestamp}")