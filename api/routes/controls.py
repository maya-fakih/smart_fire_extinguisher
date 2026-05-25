# api/routes/controls.py
"""Pump, arm, sensors, copilot decision."""

import logging
from flask import Blueprint, jsonify, request, current_app

logger = logging.getLogger(__name__)
controls_bp = Blueprint("controls", __name__)


def _orch():
    return current_app.config["ORCHESTRATOR"]


@controls_bp.route("/api/pump/fire", methods=["POST"])
def pump_fire():
    try:
        _orch().manual_pump_fire()
        return jsonify({"ok": True}), 200
    except Exception as e:
        logger.error(f"pump/fire: {e}")
        return jsonify({"error": str(e)}), 500


@controls_bp.route("/api/pump/stop", methods=["POST"])
def pump_stop():
    try:
        _orch().manual_pump_stop()
        return jsonify({"ok": True}), 200
    except Exception as e:
        logger.error(f"pump/stop: {e}")
        return jsonify({"error": str(e)}), 500


@controls_bp.route("/api/arm/nudge", methods=["POST"])
def arm_nudge():
    """Body: {"direction": "pan_left|pan_right|tilt_up|tilt_down"}"""
    try:
        body = request.get_json(force=True)
        direction = body.get("direction")
        if not direction:
            return jsonify({"error": "missing 'direction'"}), 400
        _orch().manual_arm_nudge(direction)
        return jsonify({"ok": True, "direction": direction}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"arm/nudge: {e}")
        return jsonify({"error": str(e)}), 500


@controls_bp.route("/api/sensors/<name>/toggle", methods=["POST"])
def toggle_sensor(name: str):
    """Body: {"enabled": true|false}"""
    try:
        body = request.get_json(force=True)
        if "enabled" not in body:
            return jsonify({"error": "missing 'enabled'"}), 400
        _orch().toggle_sensor(name, bool(body["enabled"]))
        return jsonify({"ok": True, "sensor": name, "enabled": bool(body["enabled"])}), 200
    except Exception as e:
        logger.error(f"sensors/toggle: {e}")
        return jsonify({"error": str(e)}), 500


@controls_bp.route("/api/copilot/decision", methods=["POST"])
def copilot_decision():
    """Body: {"decision": "approved"|"rejected"}"""
    try:
        body = request.get_json(force=True)
        decision = body.get("decision")
        _orch().set_copilot_decision(decision)
        return jsonify({"ok": True, "decision": decision}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"copilot/decision: {e}")
        return jsonify({"error": str(e)}), 500