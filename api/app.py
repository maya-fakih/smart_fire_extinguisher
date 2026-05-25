# api/app.py
"""Flask app factory. Orchestrator is stored on app.config for routes to access."""

from flask import Flask
from flask_cors import CORS

from routes.state import state_bp
from routes.controls import controls_bp
from routes.notifications import notifications_bp
from routes.predictions import predictions_bp
from routes.analytics import analytics_bp
from routes.camera import camera_bp
from db import init_app as init_db
from train_jobs import TrainJobRegistry


def create_app(orchestrator) -> Flask:
    app = Flask(__name__)
    app.config["ORCHESTRATOR"] = orchestrator

    # Async-training jobs registry. Read sizing from the orchestrator's
    # already-loaded config so we don't reparse config.json here.
    api_cfg = orchestrator.get_config_section("api")
    app.config["TRAIN_JOBS"] = TrainJobRegistry(
        orchestrator,
        history_size      = api_cfg.get("train_job_history_size", 50),
        default_timeout_s = api_cfg.get("train_job_timeout_s", 120.0),
    )

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    init_db(app)

    app.register_blueprint(state_bp)
    app.register_blueprint(controls_bp)
    app.register_blueprint(notifications_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(camera_bp)

    @app.route("/api/health")
    def health():
        return {"status": "ok"}

    return app