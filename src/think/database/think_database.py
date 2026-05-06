# src/think/database/think_database.py

import os
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from exceptions import DatabaseError


class ThinkDatabase:
    def __init__(self, max_gap_ms: int):
        self._connection = None
        self._connected = False
        self._last_row_id = None
        self._max_gap_ms = max_gap_ms

    # --- connection ---

    def connect(self):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self._connection = psycopg2.connect(
                    host=os.getenv("DB_HOST"),
                    port=os.getenv("DB_PORT"),
                    dbname=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASS"),
                    cursor_factory=RealDictCursor
                )
                self._connected = True
                return
            except Exception as e:
                self._connected = False
                if attempt == max_retries:
                    raise DatabaseError(f"Failed to connect after {max_retries} attempts: {e}")
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
        except Exception as e:
            self._connection.rollback()
            raise DatabaseError(f"Failed to log event: {e}")
        
    def _assign_event_id(self) -> None:
        try:
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
                    event_id = prev["event_id"] if gap_ms <= self._max_gap_ms else self._last_row_id
                else:
                    event_id = self._last_row_id

                cur.execute("""
                    UPDATE think_schema SET event_id = %s WHERE id = %s
                """, (event_id, self._last_row_id))
            self._connection.commit()
        except Exception as e:
            self._connection.rollback()
            raise DatabaseError(f"Failed to assign event_id: {e}")

    def update_prediction(self, danger_level: int, action: str) -> None:
        danger_labels = {1: "MINIMAL", 2: "LOW", 3: "MODERATE", 4: "HIGH", 5: "CRITICAL"}
        danger_label = danger_labels.get(danger_level, "UNKNOWN")
        try:
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
            raise DatabaseError(f"Failed to update prediction: {e}")

    def update_human_label(self, true_danger: int, true_action: str = None) -> None:
        try:
            with self._connection.cursor() as cur:
                cur.execute("""
                    UPDATE think_schema
                    SET validated = TRUE,
                        true_danger_level = %s,
                        true_action = %s
                    WHERE id = %s
                """, (true_danger, true_action, self._last_row_id))
            self._connection.commit()
        except Exception as e:
            self._connection.rollback()
            raise DatabaseError(f"Failed to update human label: {e}")

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
        chain = self.get_last_chain()
        if not chain:
            return {}
        # TODO: compute velocities, accelerations, encode categoricals
        return {}

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
            "frame_image_url": v.frame_image_url if v else None,
        }