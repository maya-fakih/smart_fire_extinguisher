# Notification Wiring — Layer Diffs

Apply these edits to the 4 layer files. Each is small and contained.

---

## File: `src/sense/sensor_fuser.py`

### Change 1 — accept notifier in __init__

Find:
```python
def __init__(self, config: dict, state: SystemState):
```

Replace with:
```python
def __init__(self, config: dict, state: SystemState, notifier=None):
```

Right after the `self._state = state` line (around line 35), add:
```python
        self._notifier = notifier
```

### Change 2 — notify on sensor fault

Find the `_mark_sensor_faulted` method:
```python
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
```

Add at the end (right before `self._update_state_sensor_counts()` or after it, either works):
```python
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
```

---

## File: `src/see/vision_fuser.py`

### Change 1 — accept notifier in __init__

Find:
```python
    def __init__(self, config: dict, state):
```

Replace with:
```python
    def __init__(self, config: dict, state, notifier=None):
```

Right after `self._state = state` and `self._queue = state.see_queue`, add:
```python
        self._notifier = notifier
```

### Change 2 — notify on camera start failure

Find the `start()` method, locate:
```python
        # start camera first — loads .rpk onto IMX500 chip
        self._camera.start()
```

Wrap that in try/except:
```python
        # start camera first — loads .rpk onto IMX500 chip
        try:
            self._camera.start()
        except Exception as e:
            if self._notifier is not None:
                from notify import EventType
                self._notifier.notify(
                    EventType.CAMERA_FAILED_TO_START,
                    payload={"error": f"{type(e).__name__}: {e}"},
                    source_layer="see",
                )
            raise
```

### Change 3 — notify on frame save failure

Find the `_save_frame` method's `cv2.imwrite` line:
```python
        # save frame as jpg using OpenCV
        cv2.imwrite(filepath, frame)
```

Replace with:
```python
        # save frame as jpg using OpenCV
        try:
            ok = cv2.imwrite(filepath, frame)
            if not ok:
                raise IOError(f"cv2.imwrite returned False for {filepath}")
        except Exception as e:
            if self._notifier is not None:
                from notify import EventType
                self._notifier.notify(
                    EventType.FRAME_STORAGE_FAILED,
                    payload={"path": filepath, "error": f"{type(e).__name__}: {e}"},
                    source_layer="see",
                )
            # don't re-raise; degraded but vision keeps running
```

---

## File: `src/think/think_engine.py`

### Change 1 — accept notifier in __init__

Find:
```python
    def __init__(self, config: dict, state: SystemState):
        self._config = config
        self._state = state
```

Replace with:
```python
    def __init__(self, config: dict, state: SystemState, notifier=None):
        self._config = config
        self._state = state
        self._notifier = notifier
```

### Change 2 — notify on DB disconnect

In `start()`, find:
```python
        except DatabaseError as e:
            logger.error(
                f"ThinkEngine: database connection failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            self._state.db_connected = False
            raise
```

Add a notify before raise:
```python
        except DatabaseError as e:
            logger.error(
                f"ThinkEngine: database connection failed - {type(e).__name__}: {e}",
                exc_info=True
            )
            self._state.db_connected = False
            if self._notifier is not None:
                from notify import EventType
                self._notifier.notify(
                    EventType.DATABASE_DISCONNECTED,
                    payload={"error": str(e)},
                    source_layer="think",
                )
            raise
```

### Change 3 — notify on model load failure

In `_load_model()`, find both `raise` paths (one for `ModelError` block, one for generic `Exception` block). After each `logger.error(...)`, before `raise`, add:
```python
            if self._notifier is not None:
                from notify import EventType
                self._notifier.notify(
                    EventType.MODEL_LOAD_FAILED,
                    payload={"model": self._active_model, "error": str(e)},
                    source_layer="think",
                )
```

### Change 4 — notify on persistent alignment drift

In `_run_loop()`, find:
```python
            except AlignmentError as e:
                logger.warning(f"ThinkEngine: alignment error - {e}")
                continue
```

This already logs but doesn't notify. Replace with:
```python
            except AlignmentError as e:
                logger.warning(f"ThinkEngine: alignment error - {e}")
                if self._notifier is not None:
                    from notify import EventType
                    self._notifier.notify(
                        EventType.ALIGNMENT_DRIFT,
                        payload={"error": str(e)},
                        source_layer="think",
                    )
                continue
```

---

## File: `src/act/act_engine.py`

### Change 1 — accept notifier in __init__

Find:
```python
    def __init__(self, config: dict, state: SystemState):
        self._config = config
        self._state = state
```

Replace with:
```python
    def __init__(self, config: dict, state: SystemState, notifier=None):
        self._config = config
        self._state = state
        self._notifier = notifier
```

### Change 2 — rewire `_notify` to use the real service

Find:
```python
    def _notify(self, danger: int, actions: list, context: str) -> None:
        # Once NotificationService is implemented, swap this for a real call.
        logger.info(
            f"ActEngine: NOTIFY | danger={danger} | "
            f"actions={actions} | context={context}"
        )
```

Replace with:
```python
    def _notify(self, danger: int, actions: list, context: str) -> None:
        """
        Fire a notification for a fire-related event. The exact EventType
        depends on the context string set by _handle_new_prediction.
        """
        from notify import EventType

        # Map context → event type
        if "auto-executing" in context:
            event_type = EventType.PREDICTION_AUTO_EXECUTING
        elif "copilot approval" in context:
            event_type = EventType.COPILOT_APPROVAL_REQUESTED
        elif "manual action required" in context:
            event_type = EventType.SURVEILLANCE_MANUAL_NEEDED
        else:
            event_type = EventType.PREDICTION_AUTO_EXECUTING  # safe default

        logger.info(
            f"ActEngine: NOTIFY | danger={danger} | "
            f"actions={actions} | context={context}"
        )

        if self._notifier is not None:
            self._notifier.notify(
                event_type,
                payload={
                    "danger_level": danger,
                    "actions": actions,
                    "context": context,
                    "mode": self._state.system_mode.value,
                },
                source_layer="act",
            )
```

### Change 3 — notify on copilot timeout

In `_check_copilot_resolution`, find the timeout branch:
```python
        elif elapsed > self._copilot_timeout_s:
            logger.warning(
                f"ActEngine: copilot TIMEOUT after {self._copilot_timeout_s}s | "
                f"pred_id={self._copilot_pending_pred_id}"
            )
            self._copilot_approved_pump = False
            self._reset_copilot_wait()
```

Add a notify call right before `self._reset_copilot_wait()`:
```python
        elif elapsed > self._copilot_timeout_s:
            logger.warning(
                f"ActEngine: copilot TIMEOUT after {self._copilot_timeout_s}s | "
                f"pred_id={self._copilot_pending_pred_id}"
            )
            self._copilot_approved_pump = False
            if self._notifier is not None:
                from notify import EventType
                self._notifier.notify(
                    EventType.COPILOT_TIMEOUT,
                    payload={
                        "pred_id": self._copilot_pending_pred_id,
                        "elapsed_s": elapsed,
                    },
                    source_layer="act",
                )
            self._reset_copilot_wait()
```

### Change 4 — notify on pump safety cutoff

This one lives in `pump_actuator.py`, not act_engine.py. PumpActuator doesn't have a notifier reference today. **Simpler fix:** since ActEngine reconciles the pump every tick, it can detect a safety cutoff (pump went from active→inactive without ActEngine asking) and notify from there. For now, just leave the existing PumpActuator log line as-is — we already get the log entry. We can wire the notifier into PumpActuator later if you want, but it'd require passing the notifier into actuators too. Out of scope for tonight.