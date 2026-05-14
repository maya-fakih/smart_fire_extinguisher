# src/notify/notification_service.py

"""
NotificationService — single sink for all system notifications.

Architecture decisions:
  - Singleton-style: one instance constructed by Orchestrator, passed to
    every layer (SensorFuser, VisionFuser, ThinkEngine, ActEngine) at
    construction. Each layer fires notify() directly at the source of
    a fault — no polling, no state-flag indirection.
  - Persisted to a `notifications` table in PostgreSQL so the website
    can poll/audit and notifications survive layer crashes.
  - Always logs as well, so notifications appear in the rotating log
    file even if the DB is down.
  - DB failures during notify() are caught and logged — a broken DB
    must never crash the layer that's trying to report a problem.

Usage from any layer:
    self._notifier.notify(
        EventType.SENSOR_FAULTED,
        payload={"sensor": "heat_grid", "reason": "I2C timeout"},
    )

Severity is auto-resolved from DEFAULT_SEVERITY unless explicitly overridden.
"""

import os
import json
import logging
import time
from typing import Optional
from datetime import datetime

import psycopg2
from psycopg2.extras import Json

from notify.event_types import EventType, Severity, DEFAULT_SEVERITY

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Central notification sink for all layers.

    Each notify() call:
      1. Logs at the right level (INFO/WARN/ERROR) so it appears in log files.
      2. Inserts a row into the `notifications` table for the website to read.

    Construction does NOT open a DB connection — start() opens lazily on
    the first notify() call. This keeps the orchestrator's construction
    path side-effect-free.
    """

    def __init__(self, config: dict):
        self._config = config
        self._connection = None
        self._connected = False
        # Track which event types have already logged a DB failure so we
        # don't spam logs with the same error on every notify().
        self._db_failure_logged = False

    # ------------------------------------------------------------------
    # Connection management — lazy connect, never crash the caller
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> bool:
        """
        Open a DB connection if we don't have one. Returns True on success.
        Never raises — returns False and logs once if DB is unreachable.
        """
        if self._connected and self._connection is not None:
            return True
        try:
            self._connection = psycopg2.connect(
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS"),
            )
            self._connection.autocommit = True
            self._connected = True
            self._db_failure_logged = False
            logger.info("NotificationService: DB connection established")
            return True
        except Exception as e:
            if not self._db_failure_logged:
                logger.error(
                    f"NotificationService: DB unreachable, falling back to "
                    f"log-only mode - {type(e).__name__}: {e}"
                )
                self._db_failure_logged = True
            self._connected = False
            return False

    def close(self) -> None:
        """Close the DB connection. Called on system shutdown."""
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
            self._connected = False

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def notify(
        self,
        event_type: EventType,
        payload: Optional[dict] = None,
        severity: Optional[Severity] = None,
        source_layer: Optional[str] = None,
    ) -> None:
        """
        Record a notification.

        Args:
            event_type: One of EventType — the catalog of known events.
            payload: Free-form dict with event-specific context (sensor name,
                     danger level, error message, etc). Stored as JSONB in DB.
            severity: Optional override of the default severity for this
                      event type. Most callers should omit this.
            source_layer: Optional name of the layer firing this (e.g. "sense",
                          "see", "think", "act", "orchestrator"). Helps debugging.

        Never raises — failures are caught and logged.
        """
        try:
            sev = severity or DEFAULT_SEVERITY.get(event_type, Severity.INFO)
            payload = payload or {}
            timestamp = datetime.now()

            # 1. Log every notification — chosen level matches severity.
            self._log_notification(event_type, sev, payload, source_layer)

            # 2. Persist to DB if reachable. Never block the caller.
            if self._ensure_connected():
                self._insert_notification(event_type, sev, payload, source_layer, timestamp)

        except Exception as e:
            # Absolute last-resort guard — a broken notifier must NEVER
            # crash the calling layer. Log and move on.
            logger.error(
                f"NotificationService: notify() crashed - {type(e).__name__}: {e}",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_notification(
        self,
        event_type: EventType,
        severity: Severity,
        payload: dict,
        source_layer: Optional[str],
    ) -> None:
        source_tag = f"[{source_layer}] " if source_layer else ""
        message = (
            f"{source_tag}NOTIFY | event={event_type.value} | "
            f"severity={severity.value} | payload={payload}"
        )
        if severity == Severity.CRITICAL:
            logger.error(message)
        elif severity == Severity.WARN:
            logger.warning(message)
        else:
            logger.info(message)

    def _insert_notification(
        self,
        event_type: EventType,
        severity: Severity,
        payload: dict,
        source_layer: Optional[str],
        timestamp: datetime,
    ) -> None:
        try:
            with self._connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO notifications
                        (timestamp, event_type, severity, source_layer, payload, acknowledged)
                    VALUES
                        (%s, %s, %s, %s, %s, FALSE)
                    """,
                    (
                        timestamp,
                        event_type.value,
                        severity.value,
                        source_layer,
                        Json(payload),
                    ),
                )
        except Exception as e:
            # DB write failed — mark connection bad so next notify retries.
            logger.error(
                f"NotificationService: DB insert failed - "
                f"{type(e).__name__}: {e}"
            )
            self._connected = False
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None