# api/routes/config.py
"""
GET  /api/config   → return the full live config dict.
POST /api/config   → apply dot-path changes via orchestrator.update_config().

Body for POST:
    {"changes": {"system.system_mode": "autopilot",
                 "sensors.smoke.threshold_physical": 500}}

update_config() validates that every key path already exists, writes the file,
fires CONFIG_UPDATED, and restarts all layers so the new values take effect.
Restarting kills any in-flight recording/training session — the UI warns about
this before saving.
"""

import logging
from flask import Blueprint, jsonify, request, current_app

logger = logging.getLogger(__name__)
config_bp = Blueprint("config", __name__)


def _orch():
    return current_app.config["ORCHESTRATOR"]


@config_bp.route("/api/config", methods=["GET"])
def get_config():
    """Return the full config currently held in memory by the orchestrator."""
    try:
        return jsonify(_orch().get_config()), 200
    except Exception as e:
        logger.error(f"config/get: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@config_bp.route("/api/config", methods=["POST"])
def update_config():
    """
    Apply a flat dict of dot-path changes. Every path must already exist in
    the config (update_config rejects unknown keys). On success the system
    restarts to pick up the new values.
    """
    try:
        body = request.get_json(force=True) or {}
        changes = body.get("changes")
        if not changes or not isinstance(changes, dict):
            return jsonify({"error": "body must have a 'changes' object"}), 400

        _orch().update_config(changes)
        return jsonify({
            "ok": True,
            "applied": list(changes.keys()),
            "restarted": True,
        }), 200
    except Exception as e:
        # ConfigError (invalid path/key) and anything else both land here.
        logger.error(f"config/update: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400