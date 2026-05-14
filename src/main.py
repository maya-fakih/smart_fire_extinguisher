# src/main.py
"""
Fire Robot — Entry Point.

Usage:
    python src/main.py
    python src/main.py --config configs/config.json

What this does:
    1. Loads config
    2. Sets up logging
    3. Constructs SystemOrchestrator (builds state, notifier, all layers)
    4. Starts all 4 processes (SENSE / SEE / THINK / ACT)
    5. Blocks until SIGINT (Ctrl+C) or SIGTERM
    6. Shuts down cleanly on exit
"""

import argparse
import logging
import logging.config
import json
import signal
import sys
import time

from core.orchestrator import SystemOrchestrator


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fire Robot System")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="Path to config.json (default: configs/config.json)",
    )
    return parser.parse_args()


# ── Logging setup ─────────────────────────────────────────────────────────────
def setup_logging(config_path: str) -> None:
    """
    Load logging config from config.json.
    Falls back to basic logging if config is missing or broken.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        if "logging" in config:
            logging.config.dictConfig(config["logging"])
        else:
            logging.basicConfig(level=logging.DEBUG)
    except Exception:
        logging.basicConfig(level=logging.DEBUG)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    setup_logging(args.config)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Fire Robot starting up")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 60)

    # ── Construct orchestrator ────────────────────────────────────────────────
    # This builds SystemState, NotificationService, and all 4 layer objects.
    # Nothing talks to hardware yet — that happens in start().
    try:
        orchestrator = SystemOrchestrator(args.config)
    except Exception as e:
        logger.critical(
            f"Failed to initialize system: {type(e).__name__}: {e}",
            exc_info=True,
        )
        sys.exit(1)

    # ── Signal handling ───────────────────────────────────────────────────────
    # SIGINT  = Ctrl+C (interactive)
    # SIGTERM = kill from systemd or process manager
    # Both trigger a clean shutdown.
    shutdown_requested = {"value": False}   # dict so closure can mutate it

    def handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Signal received: {sig_name} — shutting down cleanly")
        shutdown_requested["value"] = True

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Start all processes ───────────────────────────────────────────────────
    try:
        orchestrator.start()
        logger.info("All processes started — system is running. Press Ctrl+C to stop.")
    except Exception as e:
        logger.critical(
            f"Failed to start system: {type(e).__name__}: {e}",
            exc_info=True,
        )
        orchestrator.shutdown()
        sys.exit(1)

    # ── Block until shutdown signal ───────────────────────────────────────────
    try:
        while not shutdown_requested["value"]:
            time.sleep(0.5)
    finally:
        logger.info("Shutting down...")
        orchestrator.shutdown()
        logger.info("Fire Robot stopped cleanly.")


if __name__ == "__main__":
    main()