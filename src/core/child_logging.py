"""
child_logging.py
----------------
Re-initializes Python logging inside a multiprocessing child process.

Why: logging.config.dictConfig() runs only in main.py (the parent).
Child processes spawned via multiprocessing.Process start with a blank
logging state — no handlers, no config — so every logger.info/debug/error
call silently vanishes. This makes debugging impossible.

Call setup_child_logging() as the FIRST line of every child process
start() method, before any other work.
"""

import logging
import logging.config
import logging.handlers
import os


def setup_child_logging(level: str = "DEBUG") -> None:
    """Re-initialize logging for a child process."""
    os.makedirs("logs", exist_ok=True)
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "[%(asctime)s] %(name)s [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/fire_robot.log",
                "maxBytes": 10485760,
                "backupCount": 5,
                "formatter": "detailed",
                "level": level,
            },
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "detailed",
                "level": level,
            },
        },
        "root": {"handlers": ["file", "console"], "level": level},
    })
