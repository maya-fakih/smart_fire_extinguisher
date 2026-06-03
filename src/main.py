# src/main.py
"""
Fire Robot — Entry Point.
Starts hardware orchestrator + Flask API server (in background thread).
"""

import argparse
import logging
import logging.config
import json
import signal
import sys
import time
import threading
import os

from core.orchestrator import SystemOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="Fire Robot System")
    parser.add_argument("--config", type=str, default="configs/config.json")
    parser.add_argument("--api-port", type=int, default=5000)
    return parser.parse_args()


def setup_logging(config_path: str) -> None:
    # BUG-11 fix: ensure the logs directory exists before logging.dictConfig
    # tries to open the RotatingFileHandler. Without this, the file handler
    # silently fails on fresh deployments and we lose all file logging.
    os.makedirs("logs", exist_ok=True)
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        if "logging" in config:
            logging.config.dictConfig(config["logging"])
        else:
            logging.basicConfig(level=logging.DEBUG)
    except Exception:
        logging.basicConfig(level=logging.DEBUG)


def start_api_server(orchestrator, port: int) -> None:
    """Flask runs in the same process so it can call orchestrator methods directly."""
    try:
        api_path = os.path.join(os.path.dirname(__file__), "..", "api")
        sys.path.insert(0, os.path.abspath(api_path))
        from app import create_app
        flask_app = create_app(orchestrator)
        # FIX-3c: threaded=True lets Flask handle each request in its own thread.
        # Without it, the MJPEG /api/camera/feed route (a blocking generator)
        # monopolises the single worker and makes every other API endpoint
        # (state, controls, notifications, etc.) hang until the stream ends.
        flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        logging.getLogger(__name__).error(f"Flask failed: {e}", exc_info=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.config)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Fire Robot starting up")
    logger.info("=" * 60)

    try:
        orchestrator = SystemOrchestrator(args.config)
    except Exception as e:
        logger.critical(f"Init failed: {e}", exc_info=True)
        sys.exit(1)

    shutdown_requested = {"value": False}

    def handle_signal(signum, frame):
        logger.info(f"Signal {signal.Signals(signum).name} — shutting down")
        shutdown_requested["value"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        orchestrator.start()
    except Exception as e:
        logger.critical(f"Start failed: {e}", exc_info=True)
        orchestrator.shutdown()
        sys.exit(1)

    # Flask API in a daemon thread so signal handling stays in main thread
    api_thread = threading.Thread(
        target=start_api_server,
        args=(orchestrator, args.api_port),
        name="FlaskAPI",
        daemon=True,
    )
    api_thread.start()
    logger.info(f"Flask API listening on port {args.api_port}")

    try:
        while not shutdown_requested["value"]:
            time.sleep(0.5)
    finally:
        logger.info("Shutting down...")
        orchestrator.shutdown()
        logger.info("Stopped cleanly.")


if __name__ == "__main__":
    main()