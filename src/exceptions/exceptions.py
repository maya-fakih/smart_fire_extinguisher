class ConfigError(Exception):
    """Raised when config.json is missing, unreadable, or has invalid structure."""
    pass

class StateInitError(Exception):
    """Raised when SystemState fails to initialize — manager or queue creation failed."""
    pass

class LayerFaultError(Exception):
    """Raised when a layer process crashes or fails to start."""
    pass

class SensorFaultError(Exception):
    """Raised when a sensor fails to read after max retries."""
    pass

class ActuatorFaultError(Exception):
    """Raised when an actuator fails to activate or deactivate."""
    pass

class ModelError(Exception):
    """Raised when the ML model fails to load, predict, or retrain."""
    pass

class DatabaseError(Exception):
    """Raised when a database operation fails — connection, read, or write."""
    pass

class ModeError(Exception):
    """Raised when an invalid or unsupported system mode is requested."""
    pass

class AlignmentError(Exception):
    """Raised when THINK cannot find aligned SensorSnapshot and VisionSnapshot within max_gap_ms."""
    pass

class IKError(Exception):
    """Raised when the IK solver cannot find a valid solution within joint limits."""
    pass