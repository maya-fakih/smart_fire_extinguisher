# api/routes/training.py
"""
POST /api/training/record/start   → start a recording session
POST /api/training/record/stop    → stop the current recording session
POST /api/training/record/label   → push a label onto the recording stream
"""

import logging
from flask import Blueprint, jsonify, request, current_app

logger = logging.getLogger(__name__)
training_bp = Blueprint("training", __name__)


def _orch():
    return current_app.config["ORCHESTRATOR"]


@training_bp.route("/api/training/record/start", methods=["POST"])
def recording_start():
    """
    Body: {"same_event": true}  (optional, default true)
    Starts a recording session. THINK begins consuming queues and writing
    labeled rows to DB. Returns the event_id for this session.
    """
    try:
        body = request.get_json(force=True) or {}
        same_event = body.get("same_event", True)
        result = _orch().training_recording_start(same_event=same_event)
        return jsonify(result), 200
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    except TimeoutError as e:
        return jsonify({"error": str(e)}), 504
    except Exception as e:
        logger.error(f"recording/start: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@training_bp.route("/api/training/record/stop", methods=["POST"])
def recording_stop():
    """Stop the current recording session."""
    try:
        result = _orch().training_recording_stop()
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"recording/stop: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@training_bp.route("/api/training/record/label", methods=["POST"])
def recording_label():
    """
    Body: {"true_danger_level": 3, "true_action": "suppress"}
    true_action is optional. Pushes a label onto the recording stream —
    applies to all subsequent rows until the next label is pushed.
    true_danger_level must be 1-5.
    """
    try:
        body = request.get_json(force=True)
        if "true_danger_level" not in body:
            return jsonify({"error": "missing 'true_danger_level'"}), 400
        tdl = int(body["true_danger_level"])
        if tdl not in range(1, 6):
            return jsonify({"error": "true_danger_level must be 1-5"}), 400
        _orch().training_recording_push_label(
            true_danger_level=tdl,
            true_action=body.get("true_action"),
        )
        return jsonify({"ok": True, "true_danger_level": tdl}), 200
    except Exception as e:
        logger.error(f"recording/label: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@training_bp.route("/api/training/dataset/columns", methods=["GET"])
def dataset_columns():
    """
    Returns the ordered list of column names from think_schema (validated rows).
    Used by the frontend to validate CSV uploads before merging.
    """
    try:
        from db import get_db
        db = get_db()
        with db.cursor() as cur:
            cur.execute("SELECT * FROM think_schema WHERE validated = TRUE LIMIT 1")
            row = cur.fetchone()
        if not row:
            # No rows yet — fall back to information_schema
            with db.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'think_schema'
                    ORDER BY ordinal_position
                """)
                cols = [r["column_name"] for r in cur.fetchall()]
        else:
            cols = list(row.keys())
        return jsonify({"columns": cols}), 200
    except Exception as e:
        logger.error(f"dataset/columns: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@training_bp.route("/api/training/dataset/merge", methods=["POST"])
def dataset_merge():
    """
    Accepts a JSON body: {"rows": [ {col: val, ...}, ... ]}
    All rows must have true_danger_level (1-5) and validated = TRUE is forced.
    Columns are validated against the live schema — order and names must match exactly.
    Returns {ok, inserted, skipped, errors[]}.
    """
    try:
        from db import get_db
        import time as _time
        body = request.get_json(force=True)
        rows = body.get("rows")
        if not rows or not isinstance(rows, list):
            return jsonify({"error": "body must have 'rows' array"}), 400

        db = get_db()

        # Get live column order from schema (skip auto cols)
        AUTO_COLS = {"id", "validated", "timestamp", "event_id"}
        with db.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'think_schema'
                ORDER BY ordinal_position
            """)
            all_cols = [r["column_name"] for r in cur.fetchall()]
        insertable_cols = [c for c in all_cols if c not in AUTO_COLS]

        # Validate incoming column set matches exactly
        incoming_cols = [k for k in rows[0].keys() if k not in AUTO_COLS]
        if sorted(incoming_cols) != sorted(insertable_cols):
            missing  = set(insertable_cols) - set(incoming_cols)
            extra    = set(incoming_cols) - set(insertable_cols)
            msg_parts = []
            if missing: msg_parts.append(f"missing: {sorted(missing)}")
            if extra:   msg_parts.append(f"unexpected: {sorted(extra)}")
            return jsonify({"error": f"Column mismatch — {'; '.join(msg_parts)}"}), 422

        inserted, skipped, errors = 0, 0, []
        next_event_id_cur = db.cursor()
        next_event_id_cur.execute("SELECT COALESCE(MAX(event_id), 0) + 1 AS eid FROM think_schema")
        merge_event_id = next_event_id_cur.fetchone()["eid"]
        next_event_id_cur.close()

        for i, row in enumerate(rows):
            tdl = row.get("true_danger_level")
            if tdl is None or int(tdl) not in range(1, 6):
                errors.append(f"row {i}: true_danger_level missing or out of range (1-5)")
                skipped += 1
                continue
            try:
                cols_to_insert = insertable_cols + ["validated", "timestamp", "event_id"]
                vals = [row.get(c) for c in insertable_cols] + [True, _time.time(), merge_event_id]
                placeholders = ", ".join(["%s"] * len(cols_to_insert))
                col_list     = ", ".join(cols_to_insert)
                with db.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO think_schema ({col_list}) VALUES ({placeholders})",
                        vals,
                    )
                inserted += 1
            except Exception as e:
                errors.append(f"row {i}: {e}")
                skipped += 1

        db.commit()
        return jsonify({"ok": True, "inserted": inserted, "skipped": skipped, "errors": errors}), 200

    except Exception as e:
        logger.error(f"dataset/merge: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500