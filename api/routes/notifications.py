# api/routes/notifications.py
"""GET /api/notifications, POST /api/notifications/<id>/acknowledge"""

import logging
from flask import Blueprint, jsonify, request
from db import get_db

logger = logging.getLogger(__name__)
notifications_bp = Blueprint("notifications", __name__)


@notifications_bp.route("/api/notifications", methods=["GET"])
def list_notifications():
    """
    Query params:
      severity = info|warn|critical    (optional filter)
      limit    = int (default 50)
      offset   = int (default 0)
      unack_only = true|false (default false)
    """
    try:
        severity   = request.args.get("severity")
        limit      = int(request.args.get("limit", 50))
        offset     = int(request.args.get("offset", 0))
        unack_only = request.args.get("unack_only", "false").lower() == "true"

        sql = "SELECT * FROM notifications WHERE 1=1"
        params = []
        if severity:
            sql += " AND severity = %s"
            params.append(severity)
        if unack_only:
            sql += " AND acknowledged = FALSE"
        sql += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        db = get_db()
        with db.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        # RealDictCursor returns dicts already
        return jsonify({"notifications": rows, "count": len(rows)}), 200
    except Exception as e:
        logger.error(f"GET /api/notifications: {e}", exc_info=True)
        msg = "Database unavailable" if "psycopg2" in type(e).__module__ else str(e)
        return jsonify({"error": msg}), 500


@notifications_bp.route("/api/notifications/<int:notif_id>/acknowledge", methods=["POST"])
def acknowledge(notif_id: int):
    try:
        db = get_db()
        with db.cursor() as cur:
            cur.execute(
                "UPDATE notifications SET acknowledged = TRUE WHERE id = %s RETURNING id",
                (notif_id,),
            )
            row = cur.fetchone()
            db.commit()
        if not row:
            return jsonify({"error": "not found"}), 404
        return jsonify({"ok": True, "id": notif_id}), 200
    except Exception as e:
        logger.error(f"acknowledge: {e}", exc_info=True)
        msg = "Database unavailable" if "psycopg2" in type(e).__module__ else str(e)
        return jsonify({"error": msg}), 500