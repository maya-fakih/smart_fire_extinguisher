# api/routes/predictions.py
"""
GET  /api/predictions               → list predictions (paginated)
POST /api/predictions/<id>/label    → human assigns true_danger_level for training
POST /api/train                     → trigger XGBoost training
"""

import logging
from flask import Blueprint, jsonify, request, current_app
from db import get_db

logger = logging.getLogger(__name__)
predictions_bp = Blueprint("predictions", __name__)


@predictions_bp.route("/api/predictions", methods=["GET"])
def list_predictions():
    """
    Query params:
      unlabeled_only = true|false  (default false) → for training panel
      limit, offset
    """
    try:
        unlabeled = request.args.get("unlabeled_only", "false").lower() == "true"
        limit  = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))

        sql = "SELECT * FROM think_schema"
        params = []
        if unlabeled:
            sql += " WHERE validated = FALSE OR validated IS NULL"
        sql += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        db = get_db()
        with db.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return jsonify({"predictions": rows, "count": len(rows)}), 200
    except Exception as e:
        logger.error(f"GET /api/predictions: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@predictions_bp.route("/api/predictions/<int:pred_id>/label", methods=["POST"])
def label_prediction(pred_id: int):
    """
    Body: {"true_danger_level": 2, "true_action": "monitor"}  (true_action optional)
    """
    try:
        body = request.get_json(force=True)
        if "true_danger_level" not in body:
            return jsonify({"error": "missing 'true_danger_level'"}), 400
        tdl = int(body["true_danger_level"])
        if tdl not in range(0, 6):
            return jsonify({"error": "true_danger_level must be 0-5"}), 400
        ta = body.get("true_action")

        db = get_db()
        with db.cursor() as cur:
            cur.execute(
                """
                UPDATE think_schema
                SET true_danger_level = %s, true_action = %s, validated = TRUE
                WHERE id = %s RETURNING id
                """,
                (tdl, ta, pred_id),
            )
            row = cur.fetchone()
            db.commit()
        if not row:
            return jsonify({"error": "not found"}), 404
        return jsonify({"ok": True, "id": pred_id}), 200
    except Exception as e:
        logger.error(f"label: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@predictions_bp.route("/api/train", methods=["POST"])
def train_model():
    """
    Trigger XGBoost training on all validated rows.
    TODO (deferred): wire into XGBoostModel.fit() — requires loading
    labeled rows, building feature matrix, calling fit, saving weights.
    Stub for now so frontend can be built against it.
    """
    return jsonify({
        "ok": False,
        "error": "training pipeline not yet wired — see project_state_overview.md"
    }), 501