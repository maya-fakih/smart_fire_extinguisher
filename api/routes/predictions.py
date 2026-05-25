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
        if tdl not in range(1, 6):
            return jsonify({"error": "true_danger_level must be 1-5"}), 400
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
    Asynchronously trigger model training.

    Returns 202 Accepted with {"job_id": ..., "status": "running"} so the
    frontend can poll /api/train/status/<job_id> instead of holding the HTTP
    connection open for up to ~120s.

    Returns 409 Conflict if another training job is already running. Only
    one job at a time is allowed — the underlying THINK queue serializes
    them anyway, returning a fresh job_id for one that's secretly queued
    would just confuse the UI.
    """
    try:
        registry = current_app.config["TRAIN_JOBS"]
        job_id = registry.submit()
        return jsonify({"job_id": job_id, "status": "running"}), 202
    except Exception as e:
        # TrainingAlreadyRunning lives in train_jobs.py — checked by class name
        # to avoid an import that pulls the registry module into route-import time.
        if type(e).__name__ == "TrainingAlreadyRunning":
            running_id = getattr(e, "running_job_id", None)
            return jsonify({
                "error": "training already running",
                "running_job_id": running_id,
            }), 409
        logger.error(f"POST /api/train: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@predictions_bp.route("/api/train/status/<job_id>", methods=["GET"])
def train_status(job_id):
    """
    Poll the state of a training job.

    Response shapes:
      running -> 200 {"job_id", "status": "running", "started_at"}
      done    -> 200 {"job_id", "status": "done",    "started_at", "ended_at", "result": {...metrics, rows_used, ...}}
      failed  -> 200 {"job_id", "status": "failed",  "started_at", "ended_at", "error": "..."}
      not found / evicted -> 404
    """
    try:
        registry = current_app.config["TRAIN_JOBS"]
        job = registry.get(job_id)
        if job is None:
            return jsonify({"error": "job not found", "job_id": job_id}), 404
        return jsonify(job), 200
    except Exception as e:
        logger.error(f"GET /api/train/status/{job_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500