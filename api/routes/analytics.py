# api/routes/analytics.py
"""
GET /api/analytics/danger    → danger level over time
GET /api/analytics/sensors   → sensor readings history
GET /api/analytics/training  → dataset stats (labeled vs unlabeled, class dist)
"""

import logging
from flask import Blueprint, jsonify, request
from db import get_db

logger = logging.getLogger(__name__)
analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/api/analytics/danger", methods=["GET"])
def danger_over_time():
    """Query param: hours (default 1)"""
    try:
        hours = int(request.args.get("hours", 1))
        db = get_db()
        with db.cursor() as cur:
            cur.execute(
                """
                SELECT timestamp, danger_level
                FROM think_schema
                WHERE to_timestamp(timestamp) > NOW() - (%s || ' hours')::interval
                ORDER BY timestamp ASC
                """,
                (str(hours),),
            )
            rows = cur.fetchall()
        return jsonify({"points": rows, "count": len(rows)}), 200
    except Exception as e:
        logger.error(f"analytics/danger: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@analytics_bp.route("/api/analytics/sensors", methods=["GET"])
def sensor_history():
    """Returns JSONB sensor_readings for charting. Query: hours, sensor."""
    try:
        hours = int(request.args.get("hours", 1))
        sensor = request.args.get("sensor")  # optional filter
        db = get_db()
        with db.cursor() as cur:
            cur.execute(
                """
                SELECT timestamp, sensor_readings
                FROM think_schema
                WHERE to_timestamp(timestamp) > NOW() - (%s || ' hours')::interval
                ORDER BY timestamp ASC
                """,
                (str(hours),),
            )
            rows = cur.fetchall()

        # Optionally project a single sensor's value out of the JSONB
        if sensor:
            projected = []
            for r in rows:
                readings = r.get("sensor_readings") or {}
                if sensor in readings:
                    projected.append({"timestamp": r["timestamp"], "value": readings[sensor]})
            return jsonify({"sensor": sensor, "points": projected, "count": len(projected)}), 200
        return jsonify({"points": rows, "count": len(rows)}), 200
    except Exception as e:
        logger.error(f"analytics/sensors: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@analytics_bp.route("/api/analytics/training", methods=["GET"])
def training_stats():
    """Counts of labeled/unlabeled and class distribution."""
    try:
        db = get_db()
        with db.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM think_schema")
            total = cur.fetchone()["total"]
            cur.execute("SELECT COUNT(*) AS labeled FROM think_schema WHERE validated = TRUE")
            labeled = cur.fetchone()["labeled"]
            cur.execute(
                """
                SELECT true_danger_level, COUNT(*) AS count
                FROM think_schema
                WHERE validated = TRUE
                GROUP BY true_danger_level
                ORDER BY true_danger_level
                """
            )
            dist = cur.fetchall()
        return jsonify({
            "total": total,
            "labeled": labeled,
            "unlabeled": total - labeled,
            "class_distribution": dist,
        }), 200
    except Exception as e:
        logger.error(f"analytics/training: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500