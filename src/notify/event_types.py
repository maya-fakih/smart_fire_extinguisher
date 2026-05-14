# src/notify/event_types.py

"""
Event type taxonomy for the NotificationService.

Every notification fired anywhere in the system uses one of these event types.
This is the contract between code that raises notifications and code that
consumes them (website filters, email rules, audit queries).

Grouped by category:
  A — Fire/danger events (ACT layer)
  B — Sensor/hardware faults (SENSE layer)
  C — Camera/vision faults (SEE layer)
  D — Actuator faults (ACT layer)
  E — System/process faults (Orchestrator + any layer)
  F — System lifecycle (Orchestrator)

To add a new event type:
  1. Add the string here
  2. Add the default severity in DEFAULT_SEVERITY below
  3. Call NotificationService.notify(EventType.X, ...) at the source
"""

from enum import Enum


class EventType(str, Enum):
    # ── A: Fire/danger events ────────────────────────────────────────────────
    PREDICTION_AUTO_EXECUTING   = "prediction_auto_executing"      # autopilot mode firing actuators
    COPILOT_APPROVAL_REQUESTED  = "copilot_approval_requested"     # waiting on user
    COPILOT_TIMEOUT             = "copilot_timeout"                # no decision in time → defaulted reject
    SURVEILLANCE_MANUAL_NEEDED  = "surveillance_manual_needed"     # surveillance mode, user must act

    # ── B: Sensor/hardware faults ────────────────────────────────────────────
    SENSOR_FAULTED              = "sensor_faulted"                 # one sensor failed
    ALL_SENSORS_FAULTED         = "all_sensors_faulted"            # SENSE running blind

    # ── C: Camera/vision faults ──────────────────────────────────────────────
    CAMERA_FAILED_TO_START      = "camera_failed_to_start"         # IMX500 didn't come up
    CAMERA_FRAME_DROPPED        = "camera_frame_dropped"           # capture() returned None repeatedly
    YOLO_LOAD_FAILED            = "yolo_load_failed"               # .rpk wouldn't load on chip
    FRAME_STORAGE_FAILED        = "frame_storage_failed"           # disk write failed

    # ── D: Actuator faults ───────────────────────────────────────────────────
    PUMP_SAFETY_CUTOFF          = "pump_safety_cutoff"             # max_duration_s hit — fire may not be out
    PUMP_ACTIVATION_FAILED      = "pump_activation_failed"         # GPIO write failed
    ARM_SERVO_FAILURE           = "arm_servo_failure"              # servo didn't respond
    ACTUATOR_PIN_RECONFIGURED   = "actuator_pin_reconfigured"      # GPIO pin no longer OUT

    # ── E: System/process faults ─────────────────────────────────────────────
    LAYER_PROCESS_CRASHED       = "layer_process_crashed"          # SenseProcess/SeeProcess/etc died
    DATABASE_DISCONNECTED       = "database_disconnected"          # THINK lost DB
    MODEL_LOAD_FAILED           = "model_load_failed"              # XGBoost/etc couldn't load
    ALIGNMENT_DRIFT             = "alignment_drift"                # repeated alignment errors

    # ── F: System lifecycle ──────────────────────────────────────────────────
    SYSTEM_STARTED              = "system_started"
    SYSTEM_STOPPED              = "system_stopped"
    MODE_CHANGED                = "mode_changed"
    CONFIG_UPDATED              = "config_updated"


class Severity(str, Enum):
    INFO     = "info"      # lifecycle, non-actionable
    WARN     = "warn"      # something degraded but system still functional
    CRITICAL = "critical"  # immediate user attention required


# ── Default severity per event type ──────────────────────────────────────────
# This is the recommended severity for each event type. Callers can override
# at the call site (e.g. multiple sensors faulted at once might escalate
# SENSOR_FAULTED from warn to critical), but most callers should just trust
# this default.
DEFAULT_SEVERITY: dict[EventType, Severity] = {
    # A
    EventType.PREDICTION_AUTO_EXECUTING:  Severity.CRITICAL,
    EventType.COPILOT_APPROVAL_REQUESTED: Severity.CRITICAL,
    EventType.COPILOT_TIMEOUT:            Severity.WARN,
    EventType.SURVEILLANCE_MANUAL_NEEDED: Severity.CRITICAL,
    # B
    EventType.SENSOR_FAULTED:             Severity.WARN,
    EventType.ALL_SENSORS_FAULTED:        Severity.CRITICAL,
    # C
    EventType.CAMERA_FAILED_TO_START:     Severity.CRITICAL,
    EventType.CAMERA_FRAME_DROPPED:       Severity.WARN,
    EventType.YOLO_LOAD_FAILED:           Severity.CRITICAL,
    EventType.FRAME_STORAGE_FAILED:       Severity.WARN,
    # D
    EventType.PUMP_SAFETY_CUTOFF:         Severity.CRITICAL,
    EventType.PUMP_ACTIVATION_FAILED:     Severity.CRITICAL,
    EventType.ARM_SERVO_FAILURE:          Severity.WARN,
    EventType.ACTUATOR_PIN_RECONFIGURED:  Severity.CRITICAL,
    # E
    EventType.LAYER_PROCESS_CRASHED:      Severity.CRITICAL,
    EventType.DATABASE_DISCONNECTED:      Severity.WARN,
    EventType.MODEL_LOAD_FAILED:          Severity.CRITICAL,
    EventType.ALIGNMENT_DRIFT:            Severity.WARN,
    # F
    EventType.SYSTEM_STARTED:             Severity.INFO,
    EventType.SYSTEM_STOPPED:             Severity.INFO,
    EventType.MODE_CHANGED:               Severity.INFO,
    EventType.CONFIG_UPDATED:             Severity.INFO,
}