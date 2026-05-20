# src/think/database/think_database.py

import os
import time
import logging
import psycopg2
import numpy as np
from psycopg2.extras import RealDictCursor
from exceptions import DatabaseError

logger = logging.getLogger(__name__)


class ThinkDatabase:
    def __init__(self, config: dict):
        self._config = config
        think_cfg = config.get("think", {})
        self._max_gap_ms = think_cfg.get("max_gap_ms", 500)
        # Gap that separates one fire EVENT from a brand-new one. Much larger
        # than max_gap_ms (which only covers SENSE↔SEE hardware-delay alignment).
        self._max_event_gap_ms = think_cfg.get("max_event_gap_ms", 10000)
        self._chain_length = think_cfg.get("chain_length", 5)
        self._sensor_list = list(config.get("sensors", {}).keys())
        self._label_encoding = think_cfg.get("label_encoding", {})

        self._connection = None
        self._connected = False
        self._last_row_id = None

    # --- connection ---

    def connect(self):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"Database: connect attempt {attempt}/{max_retries}")
                self._connection = psycopg2.connect(
                    host=os.getenv("DB_HOST"),
                    port=os.getenv("DB_PORT"),
                    dbname=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASS"),
                    cursor_factory=RealDictCursor
                )
                self._connected = True
                logger.info("Database: connected successfully")
                return
            except Exception as e:
                self._connected = False
                if attempt == max_retries:
                    error_msg = f"Failed to connect after {max_retries} attempts: {e}"
                    logger.error(
                        f"Database: connection failed - {type(e).__name__}: {error_msg}",
                        exc_info=True
                    )
                    raise DatabaseError(error_msg)
                logger.warning(
                    f"Database: connection attempt {attempt} failed - {type(e).__name__}: {e}. Retrying..."
                )
                time.sleep(1)

    @property
    def is_connected(self):
        return self._connected

    @property
    def last_row_id(self):
        return self._last_row_id

    def close(self):
        if self._connection:
            self._connection.close()
            self._connected = False

    # --- write operations ---

    def log_event(self, snap) -> None:
        try:
            logger.debug(f"Database: inserting event | timestamp={snap.timestamp.isoformat()}")
            with self._connection.cursor() as cur:
                cur.execute("""
                    INSERT INTO think_schema (
                        event_id, timestamp,
                        triggered_sensors, sensor_readings, sensor_normalized,
                        composite_label, glimpsed_fire, human_near_fire,
                        fire_count, smoke_count, fire_union_area, smoke_union_area,
                        cluster_count, scene_label, scene_confidence,
                        fire_clusters, raw_detections, frame_image_url
                    ) VALUES (
                        %(event_id)s, %(timestamp)s,
                        %(triggered_sensors)s, %(sensor_readings)s, %(sensor_normalized)s,
                        %(composite_label)s, %(glimpsed_fire)s, %(human_near_fire)s,
                        %(fire_count)s, %(smoke_count)s, %(fire_union_area)s, %(smoke_union_area)s,
                        %(cluster_count)s, %(scene_label)s, %(scene_confidence)s,
                        %(fire_clusters)s, %(raw_detections)s, %(frame_image_url)s
                    ) RETURNING id
                """, self._snap_to_params(snap))
                self._last_row_id = cur.fetchone()["id"]
            self._connection.commit()
            self._assign_event_id()
            logger.info(f"Database: event_inserted | row_id={self._last_row_id}")
        except Exception as e:
            self._connection.rollback()
            logger.error(
                f"Database: failed to log event - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Failed to log event: {e}")

    def _assign_event_id(self) -> None:
        try:
            logger.debug(f"Database: assigning event_id | row_id={self._last_row_id}")
            with self._connection.cursor() as cur:
                cur.execute("""
                    SELECT id, event_id, timestamp
                    FROM think_schema
                    WHERE id < %s
                    ORDER BY id DESC
                    LIMIT 1
                """, (self._last_row_id,))
                prev = cur.fetchone()

                if prev and prev["event_id"] is not None:
                    cur.execute("SELECT timestamp FROM think_schema WHERE id = %s", (self._last_row_id,))
                    current = cur.fetchone()
                    gap_ms = abs(current["timestamp"] - prev["timestamp"]) * 1000
                    # New EVENT when the quiet gap exceeds max_event_gap_ms.
                    # (Not max_gap_ms — that is only SENSE↔SEE alignment.)
                    event_id = prev["event_id"] if gap_ms <= self._max_event_gap_ms else self._last_row_id
                    logger.debug(f"Database: event_chain_gap | gap_ms={gap_ms} | max_event={self._max_event_gap_ms}")
                else:
                    event_id = self._last_row_id

                cur.execute("""
                    UPDATE think_schema SET event_id = %s WHERE id = %s
                """, (event_id, self._last_row_id))
            self._connection.commit()
        except Exception as e:
            self._connection.rollback()
            logger.error(
                f"Database: failed to assign event_id - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Failed to assign event_id: {e}")

    # --- training mode (human-driven, not the live pipeline) -----------------

    def get_latest_event_id(self) -> int:
        """
        Return the most recent event_id in the table, or 0 if the table is empty.
        Used by the training path to decide the default event_id (same as last)
        or to generate a new one (last + 1).
        """
        try:
            with self._connection.cursor() as cur:
                cur.execute(
                    "SELECT event_id FROM think_schema "
                    "WHERE event_id IS NOT NULL ORDER BY id DESC LIMIT 1"
                )
                row = cur.fetchone()
                return row["event_id"] if row and row["event_id"] is not None else 0
        except Exception as e:
            logger.error(f"Database: get_latest_event_id failed - {type(e).__name__}: {e}")
            raise DatabaseError(f"Failed to read latest event_id: {e}")

    def log_training_capture(self, snap, event_id: int) -> int:
        """
        Insert a raw snapshot row for training mode with an EXPLICIT event_id.

        Unlike log_event(), this does NOT call _assign_event_id() — in training
        mode the human controls event grouping (a 'new event' button on the
        website), so the time-gap auto-grouping must not run. The caller passes
        the event_id directly: same as the previous capture, or previous + 1.

        Returns the new row id (needed so the website can later attach a label).
        """
        try:
            params = self._snap_to_params(snap)
            params["event_id"] = event_id
            logger.debug(f"Database: inserting training capture | event_id={event_id}")
            with self._connection.cursor() as cur:
                cur.execute("""
                    INSERT INTO think_schema (
                        event_id, timestamp,
                        triggered_sensors, sensor_readings, sensor_normalized,
                        composite_label, glimpsed_fire, human_near_fire,
                        fire_count, smoke_count, fire_union_area, smoke_union_area,
                        cluster_count, scene_label, scene_confidence,
                        fire_clusters, raw_detections, frame_image_url
                    ) VALUES (
                        %(event_id)s, %(timestamp)s,
                        %(triggered_sensors)s, %(sensor_readings)s, %(sensor_normalized)s,
                        %(composite_label)s, %(glimpsed_fire)s, %(human_near_fire)s,
                        %(fire_count)s, %(smoke_count)s, %(fire_union_area)s, %(smoke_union_area)s,
                        %(cluster_count)s, %(scene_label)s, %(scene_confidence)s,
                        %(fire_clusters)s, %(raw_detections)s, %(frame_image_url)s
                    ) RETURNING id
                """, params)
                row_id = cur.fetchone()["id"]
            self._connection.commit()
            logger.info(f"Database: training_capture_inserted | row_id={row_id} | event_id={event_id}")
            return row_id
        except Exception as e:
            self._connection.rollback()
            logger.error(
                f"Database: failed to log training capture - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Failed to log training capture: {e}")

    def log_training_capture_labeled(self, snap, event_id: int,
                                     true_danger_level: int,
                                     true_action: str = None) -> int:
        """
        Insert a training-capture row with the human label already attached
        and validated=TRUE — used by stream mode where the label is known at
        capture time, saving a separate UPDATE roundtrip per row.

        Same INSERT as log_training_capture plus the label columns.
        """
        danger_labels = {1: "MINIMAL", 2: "LOW", 3: "MODERATE",
                         4: "HIGH",     5: "CRITICAL"}
        danger_label  = danger_labels.get(true_danger_level, "UNKNOWN")
        try:
            params = self._snap_to_params(snap)
            params["event_id"]          = event_id
            params["true_danger_level"] = true_danger_level
            params["true_action"]       = true_action
            params["danger_label"]      = danger_label
            logger.debug(
                f"Database: inserting labeled training capture | "
                f"event_id={event_id} | danger={true_danger_level}"
            )
            with self._connection.cursor() as cur:
                cur.execute("""
                    INSERT INTO think_schema (
                        event_id, timestamp,
                        triggered_sensors, sensor_readings, sensor_normalized,
                        composite_label, glimpsed_fire, human_near_fire,
                        fire_count, smoke_count, fire_union_area, smoke_union_area,
                        cluster_count, scene_label, scene_confidence,
                        fire_clusters, raw_detections, frame_image_url,
                        true_danger_level, true_action, danger_label, validated
                    ) VALUES (
                        %(event_id)s, %(timestamp)s,
                        %(triggered_sensors)s, %(sensor_readings)s, %(sensor_normalized)s,
                        %(composite_label)s, %(glimpsed_fire)s, %(human_near_fire)s,
                        %(fire_count)s, %(smoke_count)s, %(fire_union_area)s, %(smoke_union_area)s,
                        %(cluster_count)s, %(scene_label)s, %(scene_confidence)s,
                        %(fire_clusters)s, %(raw_detections)s, %(frame_image_url)s,
                        %(true_danger_level)s, %(true_action)s, %(danger_label)s, TRUE
                    ) RETURNING id
                """, params)
                row_id = cur.fetchone()["id"]
            self._connection.commit()
            logger.info(
                f"Database: labeled_capture_inserted | row_id={row_id} | "
                f"event_id={event_id}"
            )
            return row_id
        except Exception as e:
            self._connection.rollback()
            logger.error(
                f"Database: failed to log labeled capture - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Failed to log labeled training capture: {e}")

    def save_training_label(self, row_id: int, true_danger_level: int,
                            true_action: str = None) -> None:
        """
        Attach a human label to a training-capture row, matched by the row's
        unique primary-key id. Capture inserted the row and returned its id;
        the website echoes that id back on save. Exact match, no ambiguity.

        Sets: true_danger_level, true_action, danger_label, validated = TRUE.
        """
        danger_labels = {1: "MINIMAL", 2: "LOW", 3: "MODERATE", 4: "HIGH", 5: "CRITICAL"}
        danger_label = danger_labels.get(true_danger_level, "UNKNOWN")
        try:
            logger.debug(
                f"Database: saving training label | row_id={row_id} | "
                f"true_danger_level={true_danger_level} ({danger_label})"
            )
            with self._connection.cursor() as cur:
                cur.execute("""
                    UPDATE think_schema
                    SET true_danger_level = %s,
                        true_action       = %s,
                        danger_label      = %s,
                        validated         = TRUE
                    WHERE id = %s
                """, (true_danger_level, true_action, danger_label, row_id))
                if cur.rowcount == 0:
                    raise DatabaseError(f"No think_schema row with id={row_id}")
            self._connection.commit()
            logger.info(f"Database: training_label_saved | row_id={row_id}")
        except Exception as e:
            self._connection.rollback()
            logger.error(
                f"Database: failed to save training label - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Failed to save training label: {e}")

    def update_prediction(self, danger_level: int, action: str) -> None:
        danger_labels = {1: "MINIMAL", 2: "LOW", 3: "MODERATE", 4: "HIGH", 5: "CRITICAL"}
        danger_label = danger_labels.get(danger_level, "UNKNOWN")
        try:
            logger.debug(
                f"Database: updating prediction | row_id={self._last_row_id} | "
                f"danger_level={danger_level} ({danger_label}) | action={action}"
            )
            with self._connection.cursor() as cur:
                cur.execute("""
                    UPDATE think_schema
                    SET danger_level = %s,
                        danger_label = %s,
                        recommended_action = %s
                    WHERE id = %s
                """, (danger_level, danger_label, action, self._last_row_id))
            self._connection.commit()
        except Exception as e:
            self._connection.rollback()
            logger.error(
                f"Database: failed to update prediction - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Failed to update prediction: {e}")

    # --- read operations ---

    def get_event_chain(self, event_id: int) -> list:
        try:
            with self._connection.cursor() as cur:
                cur.execute("""
                    SELECT * FROM think_schema
                    WHERE event_id = %s
                    ORDER BY timestamp ASC
                """, (event_id,))
                return cur.fetchall()
        except Exception as e:
            raise DatabaseError(f"Failed to get event chain: {e}")

    def get_last_chain(self) -> list:
        if self._last_row_id is None:
            return []
        try:
            with self._connection.cursor() as cur:
                cur.execute("SELECT event_id FROM think_schema WHERE id = %s", (self._last_row_id,))
                row = cur.fetchone()
                if row and row["event_id"] is not None:
                    return self.get_event_chain(row["event_id"])
            return []
        except Exception as e:
            raise DatabaseError(f"Failed to get last chain: {e}")

    def get_validated_rows(self) -> list:
        try:
            with self._connection.cursor() as cur:
                cur.execute("""
                    SELECT * FROM think_schema
                    WHERE validated = TRUE
                    ORDER BY timestamp ASC
                """)
                return cur.fetchall()
        except Exception as e:
            raise DatabaseError(f"Failed to fetch validated rows: {e}")

    # --- feature building ---

    def build_feature_vector(self, row_id: int) -> dict:
        """
        Build feature vector for the chain ending at row_id.
        Walks back from the given row_id (NOT _last_row_id) so this works for
        both live prediction and offline training (CSV export).
        Returns a flat dict[str, float] for XGBoost. Missing data → np.nan.
        """
        logger.debug(f"Database: building feature vector | row_id={row_id}")
        try:
            # 1. Find this row's event_id
            with self._connection.cursor() as cur:
                cur.execute("SELECT event_id FROM think_schema WHERE id = %s", (row_id,))
                row = cur.fetchone()
                if not row or row["event_id"] is None:
                    logger.warning(f"Database: no event_id found for row_id={row_id}")
                    return {}
                event_id = row["event_id"]

                # 2. Get last N rows of this event up to row_id
                cur.execute("""
                    SELECT * FROM (
                        SELECT * FROM think_schema
                        WHERE event_id = %s AND id <= %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ) sub
                    ORDER BY timestamp ASC
                """, (event_id, row_id, self._chain_length))
                chain = cur.fetchall()
        except Exception as e:
            logger.error(
                f"Database: failed to fetch chain - {type(e).__name__}: {e}",
                exc_info=True
            )
            raise DatabaseError(f"Failed to fetch chain for feature vector: {e}")

        if not chain:
            logger.debug(f"Database: empty chain for row_id={row_id}")
            return {}

        features = {}
        latest = chain[-1]

        # 3. Sensor features (per sensor in canonical list)
        for sensor in self._sensor_list:
            vals, ts = self._extract_sensor_series(chain, sensor)
            valid = [v for v in vals if not np.isnan(v)]

            features[f"{sensor}_latest"] = vals[-1] if not np.isnan(vals[-1]) else np.nan
            features[f"{sensor}_avg"] = float(np.nanmean(vals)) if valid else np.nan
            features[f"{sensor}_variance"] = float(np.nanvar(vals)) if len(valid) >= 2 else np.nan
            features[f"{sensor}_velocity"] = self._safe_velocity(vals, ts)
            features[f"{sensor}_acceleration"] = self._safe_acceleration(vals, ts)

        # 4. Vision features — from latest row + velocities across chain
        vision_present = latest.get("composite_label") is not None

        if vision_present:
            features["fire_count"] = self._nan_if_none(latest.get("fire_count"))
            features["smoke_count"] = self._nan_if_none(latest.get("smoke_count"))
            features["cluster_count"] = self._nan_if_none(latest.get("cluster_count"))
            features["fire_union_area"] = self._nan_if_none(latest.get("fire_union_area"))
            features["smoke_union_area"] = self._nan_if_none(latest.get("smoke_union_area"))
            features["scene_confidence"] = self._nan_if_none(latest.get("scene_confidence"))

            gf = latest.get("glimpsed_fire")
            hnf = latest.get("human_near_fire")
            features["glimpsed_fire"] = float(gf) if gf is not None else np.nan
            features["human_near_fire"] = float(hnf) if hnf is not None else np.nan

            fua_vals, fua_ts = self._extract_vision_series(chain, "fire_union_area")
            sua_vals, sua_ts = self._extract_vision_series(chain, "smoke_union_area")
            features["fire_union_area_velocity"] = self._safe_velocity(fua_vals, fua_ts)
            features["smoke_union_area_velocity"] = self._safe_velocity(sua_vals, sua_ts)

            comp = latest.get("composite_label")
            scene = latest.get("scene_label")
            features["composite_label_encoded"] = self._label_encoding.get("composite_label", {}).get(comp, np.nan)
            features["scene_label_encoded"] = self._label_encoding.get("scene_label", {}).get(scene, np.nan)
        else:
            for k in ["fire_count", "smoke_count", "cluster_count",
                      "fire_union_area", "smoke_union_area", "scene_confidence",
                      "glimpsed_fire", "human_near_fire",
                      "fire_union_area_velocity", "smoke_union_area_velocity",
                      "composite_label_encoded", "scene_label_encoded"]:
                features[k] = np.nan

        return features

    # --- feature helpers ---

    def _extract_sensor_series(self, chain: list, sensor_name: str) -> tuple:
        """
        Pull one sensor's normalized values across the chain.
        Missing readings → np.nan.
        Defensive: if a sensor stored an array (e.g. heat matrix not yet aggregated
        at the sensor layer), fall back to the max value.
        """
        values = []
        timestamps = []
        for row in chain:
            normalized = row.get("sensor_normalized")
            if normalized is None:
                values.append(np.nan)
            else:
                v = normalized.get(sensor_name, np.nan)
                if isinstance(v, list):
                    v = max(v) if v else np.nan
                values.append(v if v is not None else np.nan)
            timestamps.append(row["timestamp"])
        return values, timestamps

    def _extract_vision_series(self, chain: list, field: str) -> tuple:
        """Pull one vision field across the chain. NaN if missing."""
        values = []
        timestamps = []
        for row in chain:
            v = row.get(field)
            values.append(np.nan if v is None else v)
            timestamps.append(row["timestamp"])
        return values, timestamps

    def _safe_velocity(self, values: list, timestamps: list) -> float:
        """Δvalue/Δtime over last 2 non-NaN points. NaN if not computable."""
        pairs = [(v, t) for v, t in zip(values, timestamps) if not np.isnan(v)]
        if len(pairs) < 2:
            return np.nan
        (v1, t1), (v2, t2) = pairs[-2], pairs[-1]
        dt = t2 - t1
        if dt == 0:
            return np.nan
        return (v2 - v1) / dt

    def _safe_acceleration(self, values: list, timestamps: list) -> float:
        """Δvelocity/Δtime over last 3 non-NaN points. NaN if not computable."""
        pairs = [(v, t) for v, t in zip(values, timestamps) if not np.isnan(v)]
        if len(pairs) < 3:
            return np.nan
        (v1, t1), (v2, t2), (v3, t3) = pairs[-3], pairs[-2], pairs[-1]
        dt1 = t2 - t1
        dt2 = t3 - t2
        if dt1 == 0 or dt2 == 0:
            return np.nan
        vel1 = (v2 - v1) / dt1
        vel2 = (v3 - v2) / dt2
        # midpoint times for the two velocity intervals
        dt_between = ((t3 + t2) / 2) - ((t2 + t1) / 2)
        if dt_between == 0:
            return np.nan
        return (vel2 - vel1) / dt_between

    def _nan_if_none(self, v):
        """Convert None or 'falsy-but-not-zero' to np.nan; preserve real numbers including 0."""
        return np.nan if v is None else v

    # --- utilities ---

    def export_csv(self, path: str) -> None:
        import csv
        rows = self.get_validated_rows()
        if not rows:
            raise DatabaseError("No validated rows to export")
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            raise DatabaseError(f"Failed to export CSV: {e}")

    def clear_logs(self) -> None:
        try:
            with self._connection.cursor() as cur:
                cur.execute("DELETE FROM think_schema")
            self._connection.commit()
        except Exception as e:
            self._connection.rollback()
            raise DatabaseError(f"Failed to clear logs: {e}")

    # --- helpers ---

    def _snap_to_params(self, snap) -> dict:
        s = snap.sensor_snapshot
        v = snap.vision_snapshot

        return {
            "event_id": None,
            "timestamp": snap.timestamp.timestamp(),
            "triggered_sensors": s.triggered_sensors if s else None,
            "sensor_readings": s.sensor_readings if s else None,
            "sensor_normalized": s.sensor_normalized if s else None,
            "composite_label": v.composite_label if v else None,
            "glimpsed_fire": v.glimpsed_fire if v else None,
            "human_near_fire": v.human_near_fire if v else None,
            "fire_count": v.fire_count if v else None,
            "smoke_count": v.smoke_count if v else None,
            "fire_union_area": v.fire_union_area if v else None,
            "smoke_union_area": v.smoke_union_area if v else None,
            "cluster_count": v.cluster_count if v else None,
            "scene_label": v.scene_label if v else None,
            "scene_confidence": v.scene_confidence if v else None,
            "fire_clusters": v.fire_clusters if v else None,
            "raw_detections": v.raw_detections if v else None,
            "frame_image_url": v.image_url if v else None,
        }