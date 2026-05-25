"""
Unit test for TrainJobRegistry. Runs in-process against a fake orchestrator
that lets us control train_model() behavior precisely (sleep, return,
raise, ok=False).

Run from project root:
    PYTHONPATH=api python3 tests/test_train_jobs.py
"""

import sys, os, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train_jobs import (
    TrainJobRegistry,
    TrainingAlreadyRunning,
    STATUS_RUNNING, STATUS_DONE, STATUS_FAILED,
)


class FakeOrchestrator:
    """Lets each test dictate the behavior of train_model()."""
    def __init__(self, behavior="ok", delay=0.05, payload=None):
        self.behavior = behavior   # "ok", "raise", "ok_false", "slow_ok"
        self.delay    = delay
        self.payload  = payload or {"ok": True, "metrics": {"accuracy": 0.9, "f1_macro": 0.85}, "rows_used": 42}
        self.calls    = 0

    def train_model(self, timeout_s=120.0):
        self.calls += 1
        time.sleep(self.delay)
        if self.behavior == "ok":
            return self.payload
        if self.behavior == "slow_ok":
            time.sleep(0.3)
            return self.payload
        if self.behavior == "raise":
            raise RuntimeError("trainer exploded")
        if self.behavior == "ok_false":
            return {"ok": False, "error": "not enough rows"}
        raise ValueError(f"unknown behavior {self.behavior}")


def _wait_until_terminal(registry, job_id, timeout_s=3.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = registry.get(job_id)
        if job and job["status"] in (STATUS_DONE, STATUS_FAILED):
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach terminal in {timeout_s}s")


def test_happy_path():
    orch = FakeOrchestrator(behavior="ok", delay=0.05)
    reg  = TrainJobRegistry(orch, history_size=10)
    jid  = reg.submit()
    # Immediately after submit, status should be running
    snap = reg.get(jid)
    assert snap is not None
    assert snap["status"] == STATUS_RUNNING, snap
    # Wait for completion
    final = _wait_until_terminal(reg, jid)
    assert final["status"] == STATUS_DONE, final
    assert final["result"]["metrics"]["accuracy"] == 0.9
    assert final["error"] is None
    assert final["ended_at"] >= final["started_at"]
    print("✓ test_happy_path")


def test_failure_path():
    orch = FakeOrchestrator(behavior="raise", delay=0.05)
    reg  = TrainJobRegistry(orch, history_size=10)
    jid  = reg.submit()
    final = _wait_until_terminal(reg, jid)
    assert final["status"] == STATUS_FAILED, final
    assert "trainer exploded" in final["error"]
    assert final["result"] is None
    print("✓ test_failure_path")


def test_ok_false_is_treated_as_failure():
    """Orchestrator returns {ok: False, error: ...} without raising.
    Registry should convert this into a 'failed' record."""
    orch = FakeOrchestrator(behavior="ok_false", delay=0.05)
    reg  = TrainJobRegistry(orch, history_size=10)
    jid  = reg.submit()
    final = _wait_until_terminal(reg, jid)
    assert final["status"] == STATUS_FAILED, final
    assert "not enough rows" in final["error"]
    print("✓ test_ok_false_is_treated_as_failure")


def test_409_when_running():
    """While one job runs, submitting another raises TrainingAlreadyRunning."""
    orch = FakeOrchestrator(behavior="slow_ok", delay=0.05)
    reg  = TrainJobRegistry(orch, history_size=10)
    jid1 = reg.submit()
    # Job 1 should still be running for ~300ms — try to submit job 2 immediately
    try:
        reg.submit()
        raise AssertionError("expected TrainingAlreadyRunning")
    except TrainingAlreadyRunning as e:
        assert e.running_job_id == jid1
    # Wait it out then we should be able to submit again
    _wait_until_terminal(reg, jid1, timeout_s=5.0)
    jid2 = reg.submit()
    assert jid2 != jid1
    _wait_until_terminal(reg, jid2)
    print("✓ test_409_when_running")


def test_eviction_oldest_terminal():
    """history_size=2 -> third *terminal* job pushes out the oldest."""
    orch = FakeOrchestrator(behavior="ok", delay=0.02)
    reg  = TrainJobRegistry(orch, history_size=2)
    jids = []
    for _ in range(3):
        jid = reg.submit()
        _wait_until_terminal(reg, jid)
        jids.append(jid)
    # Oldest (jids[0]) should have been evicted
    assert reg.get(jids[0]) is None, "oldest job should have been evicted"
    assert reg.get(jids[1]) is not None
    assert reg.get(jids[2]) is not None
    print("✓ test_eviction_oldest_terminal")


def test_concurrent_submit_race():
    """Hammer submit() from N threads. Exactly one should win, others should
    raise TrainingAlreadyRunning. This is the lock's job."""
    orch = FakeOrchestrator(behavior="slow_ok", delay=0.05)
    reg  = TrainJobRegistry(orch, history_size=20)

    barrier   = threading.Barrier(10)
    successes = []
    rejects   = []
    lock_for_results = threading.Lock()

    def attacker():
        barrier.wait()
        try:
            jid = reg.submit()
            with lock_for_results:
                successes.append(jid)
        except TrainingAlreadyRunning:
            with lock_for_results:
                rejects.append(True)

    threads = [threading.Thread(target=attacker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(successes) == 1, f"expected exactly 1 winner, got {len(successes)}: {successes}"
    assert len(rejects)   == 9, f"expected 9 rejects, got {len(rejects)}"
    _wait_until_terminal(reg, successes[0], timeout_s=5.0)
    print("✓ test_concurrent_submit_race")


def test_get_returns_copy_not_reference():
    """Caller can mutate the returned dict without corrupting the registry."""
    orch = FakeOrchestrator(behavior="ok", delay=0.02)
    reg  = TrainJobRegistry(orch, history_size=10)
    jid  = reg.submit()
    _wait_until_terminal(reg, jid)
    snap = reg.get(jid)
    snap["status"] = "tampered"
    snap["error"]  = "I changed this"
    fresh = reg.get(jid)
    assert fresh["status"] == STATUS_DONE
    assert fresh["error"] is None
    print("✓ test_get_returns_copy_not_reference")


def test_not_found():
    orch = FakeOrchestrator()
    reg  = TrainJobRegistry(orch, history_size=10)
    assert reg.get("nope-not-a-real-id") is None
    print("✓ test_not_found")


if __name__ == "__main__":
    test_happy_path()
    test_failure_path()
    test_ok_false_is_treated_as_failure()
    test_409_when_running()
    test_eviction_oldest_terminal()
    test_concurrent_submit_race()
    test_get_returns_copy_not_reference()
    test_not_found()
    print("\nAll TrainJobRegistry tests passed.")
