from enum import Enum

class SystemMode(str, Enum):
    AUTOPILOT   = "autopilot"
    COPILOT     = "copilot"
    SURVEILLANCE = "surveillance"
    TRAINING    = "training"