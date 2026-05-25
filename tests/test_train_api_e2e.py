"""
End-to-end test of /api/train and /api/train/status/<job_id> using a real
Flask test client + a fake orchestrator. Exercises:

  1. POST /api/train -> 202 with job_id
  2. GET status -> running -> done with full metrics
  3. Second POST while running -> 409 Conflict
  4. GET status for bogus id -> 404
  5. Failure path -> 200 done? no, 200 status=failed with error

Run from project root:
    python3 tests/test_train_api_e2e.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flask
from flask import Flask, Blueprint
from train_jobs import TrainJobRegistry


class FakeOrchestrator:
    def __init__(self, behavior="ok", delay=0.3):
        self.behavior = behavior
        self.delay    = delay
    def train_model(self, timeout_s=120.0):
        time.sleep(self.delay)
        if self.behavior == "raise":
            raise RuntimeError("simulated failure")
        return {
            "ok": True,
            "rows_total": 100, "rows_used": 95, "rows_skipped": 5,
            "feature_count": 30, "train_samples": 76, "test_samples": 19,
            "metrics": {"accuracy": 0.91, "f1_macro": 0.88, "loss": 0.31},
            "model_path": "model_weights/",
        }
    def get_config_section(self, section):
        if section == "api":
            return {"train_job_history_size": 50, "train_job_timeout_s": 120.0}
        return {}


def build_app(behavior="ok", delay=0.3):
    """Build a minimal Flask app with only the predictions blueprint —
    bypass create_app() so we don't pull in db/psycopg2."""
    # Have to import predictions before registering, but predictions imports db.
    # Trick: pre-stub the db module by clearing it and inserting a fake.
    import sys as _sys
    fake_db = type(_sys)("db")
    fake_db.get_db = lambda: None
    fake_db.init_app = lambda app: None
    _sys.modules["db"] = fake_db

    from routes.predictions import predictions_bp

    app = Flask(__name__)
    orch = FakeOrchestrator(behavior=behavior, delay=delay)
    app.config["ORCHESTRATOR"] = orch
    app.config["TRAIN_JOBS"] = TrainJobRegistry(orch, history_size=50, default_timeout_s=120.0)
    app.register_blueprint(predictions_bp)
    return app


def poll_until_terminal(client, job_id, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/api/train/status/{job_id}")
        body = resp.get_json()
        if body and body.get("status") in ("done", "failed"):
            return resp.status_code, body
        time.sleep(0.05)
    raise AssertionError("never reached terminal")


def test_happy_path():
    app = build_app(behavior="ok", delay=0.2)
    client = app.test_client()
    resp = client.post("/api/train")
    assert resp.status_code == 202, resp.get_json()
    body = resp.get_json()
    job_id = body["job_id"]
    assert body["status"] == "running"

    # Status immediately = running
    s = client.get(f"/api/train/status/{job_id}").get_json()
    assert s["status"] == "running", s

    # Poll until done
    code, final = poll_until_terminal(client, job_id)
    assert code == 200
    assert final["status"] == "done"
    assert final["result"]["metrics"]["accuracy"] == 0.91
    assert final["result"]["rows_used"] == 95
    print("✓ test_happy_path")


def test_409_when_running():
    app = build_app(behavior="ok", delay=0.5)
    client = app.test_client()
    r1 = client.post("/api/train")
    assert r1.status_code == 202
    jid1 = r1.get_json()["job_id"]
    # Second POST while r1 is still running
    r2 = client.post("/api/train")
    assert r2.status_code == 409, r2.get_json()
    err = r2.get_json()
    assert "already running" in err["error"]
    assert err["running_job_id"] == jid1
    poll_until_terminal(client, jid1, timeout_s=5.0)
    print("✓ test_409_when_running")


def test_404_unknown_job():
    app = build_app()
    client = app.test_client()
    r = client.get("/api/train/status/not-a-real-uuid")
    assert r.status_code == 404
    assert r.get_json()["error"] == "job not found"
    print("✓ test_404_unknown_job")


def test_failure_surface():
    app = build_app(behavior="raise", delay=0.1)
    client = app.test_client()
    r = client.post("/api/train")
    jid = r.get_json()["job_id"]
    code, final = poll_until_terminal(client, jid)
    assert code == 200
    assert final["status"] == "failed"
    assert "simulated failure" in final["error"]
    assert final["result"] is None
    print("✓ test_failure_surface")


if __name__ == "__main__":
    test_happy_path()
    test_409_when_running()
    test_404_unknown_job()
    test_failure_surface()
    print("\nAll API e2e tests passed.")
