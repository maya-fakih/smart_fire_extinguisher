# api/routes/state.py
"""GET /api/state, POST /api/mode, POST /api/camera/toggle"""

import logging
from flask import Blueprint, jsonify, request, current_app

logger = logging.getLogger(__name__)
state_bp = Blueprint("state", __name__)


def _orch():
    return current_app.config["ORCHESTRATOR"]


@state_bp.route("/api/state", methods=["GET"])
def get_state():
    try:
        summary = _orch().get_state_summary()
        # SystemMode enum is already .value'd in get_state_summary
        return jsonify(summary), 200
    except Exception as e:
        logger.error(f"GET /api/state: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@state_bp.route("/api/mode", methods=["POST"])
def set_mode():
    """Body: {"mode": "copilot"}"""
    try:
        body = request.get_json(force=True)
        mode = body.get("mode")
        if not mode:
            return jsonify({"error": "missing 'mode'"}), 400
        _orch().set_mode(mode)
        return jsonify({"ok": True, "mode": mode}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"POST /api/mode: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@state_bp.route("/api/camera/toggle", methods=["POST"])
def toggle_camera():
    """Body: {"active": true}"""
    try:
        body = request.get_json(force=True)
        if "active" not in body:
            return jsonify({"error": "missing 'active'"}), 400
        _orch().set_camera_feed(bool(body["active"]))
        return jsonify({"ok": True, "camera_feed_active": bool(body["active"])}), 200
    except Exception as e:
        logger.error(f"POST /api/camera/toggle: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500