import time
import threading
from typing import List, Tuple
from sensor_parser import SensorParser
from sensor_base import Sensor
from snapshot import SensorSnapshot


class SensorFuser:

    def __init__(self, config_path: str, state, notifier=None):
        # load all sensors from config.json using the parser
        parser = SensorParser(config_path)
        self.sensors: List[Sensor] = parser.load()

        # reference to SystemState — needed to wake up downstream layers
        self._state = state

        # reference to NotificationService — needed to notify on sensor fault
        self._notifier = notifier

        # read polling intervals from config instead of hardcoding
        config = parser._load_config()
        self.interval_idle_ms = config["system"]["polling_interval_idle_ms"]
        self.interval_active_ms = config["system"]["polling_interval_active_ms"]

        # threads — one per sensor
        self._threads: List[threading.Thread] = []

        # monitor thread — watches all sensors
        self._monitor_thread = None

        # is the fuser running?
        self._running = False

        # event to signal sensor threads to switch polling rate
        self._active_event = threading.Event()

    def start(self) -> None:
        self._running = True

        # update SystemState — fuser is now running
        self._state["sense_running"] = True
        self._state["active_sensor_count"] = len(self.sensors)
        self._state["faulted_sensors"] = []

        # start each sensor in its own thread
        for sensor in self.sensors:
            t = threading.Thread(
                target=self._sensor_thread,
                args=(sensor,),
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
        self._state["sense_running"] = False

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
            # collect disabled/faulted sensors
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

        # build and return the snapshot using time.time() for timestamp
        return SensorSnapshot(
            timestamp=time.time(),
            sensor_readings=readings,
            sensor_normalized=normalized,
            triggered_sensors=triggered_sensors,
            disabled_sensors=disabled_sensors,
            raw_matrices=raw_matrices
        )

    def _sensor_thread(self, sensor: Sensor) -> None:
        # each sensor runs in its own thread
        # switches between idle and active polling rate
        while self._running and not sensor.fault:
            sensor.poll()

            # use active interval when fire detected, idle when calm
            if self._active_event.is_set():
                time.sleep(self.interval_active_ms / 1000)
            else:
                time.sleep(self.interval_idle_ms / 1000)

    def _handle_fault(self, sensor: Sensor) -> None:
        # update SystemState when a sensor faults
        faulted = list(self._state["faulted_sensors"])
        if sensor.name not in faulted:
            faulted.append(sensor.name)
            self._state["faulted_sensors"] = faulted
            self._state["active_sensor_count"] -= 1

        # notify admin about the fault
        if self._notifier:
            self._notifier.notify_fault(sensor.name, "hardware_fault")

        print(f"[FAULT] Sensor faulted: {sensor.name}")

    def _monitor_loop(self) -> None:
        # keep checking sensors while running
        while self._running:

            # check for newly faulted sensors
            for sensor in self.sensors:
                if sensor.fault:
                    self._handle_fault(sensor)

            triggered, triggered_list = self.evaluate()

            if triggered:
                print(f"[ALERT] Sensors triggered: {[s.name for s in triggered_list]}")

                # write to SystemState to wake up downstream layers
                self._state["sensor_triggered"] = True

                # switch sensor threads to fast polling
                self._active_event.set()

                # take a snapshot
                snap = self.snapshot()

                # send snapshot to sense_queue
                self.emit_trigger(snap)

                # check more frequently when fire detected
                time.sleep(self.interval_active_ms / 1000)
            else:
                # no trigger — reset SystemState
                self._state["sensor_triggered"] = False

                # switch sensor threads back to slow polling
                self._active_event.clear()

                # check less frequently when calm
                time.sleep(self.interval_idle_ms / 1000)

    def emit_trigger(self, snapshot: SensorSnapshot) -> None:
        # put snapshot into sense_queue for ThinkEngine to consume
        self._state["sense_queue"].put(snapshot)
        print(f"[TRIGGER] Snapshot sent to sense_queue at {snapshot.timestamp}")