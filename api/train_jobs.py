# api/train_jobs.py
"""
Thread-safe in-memory registry for asynchronous training jobs.

Owns the per-job state that the HTTP layer needs to round-trip:
    POST /api/train                  -> create job, return job_id
    GET  /api/train/status/<job_id>  -> read job state, return to caller

The registry is *not* a persistent history — it is a short-lived scratchpad
sized by config (api.train_job_history_size). When the ring buffer is full,
the oldest *terminal* (done|failed) job is evicted. Running jobs are never
evicted. Real audit trail lives in the log files (we log at INFO on every
state transition).

Thread-safety:
    All reads and mutations go through a single RLock. The "is anything
    running?" check and the insert happen under the same lock to prevent
    a race where two near-simultaneous POSTs both pass the check.

Why a custom class instead of a dict + Lock inline in predictions.py:
    1. The races (running check vs insert, evict vs read) are easy to get
       wrong in route code. Encapsulating them here means the route is
       readable and the locking is auditable in one place.
    2. The orchestrator-side blocking call lives here too, so the route
       handler doesn't have to know anything about queue request_ids.
"""

import logging
import threading
import time
import uuid
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


# Terminal states never transition out of these.
STATUS_RUNNING = "running"
STATUS_DONE    = "done"
STATUS_FAILED  = "failed"

_TERMINAL = (STATUS_DONE, STATUS_FAILED)


class TrainingAlreadyRunning(Exception):
    """Raised when a new job is requested while another is still running."""
    def __init__(self, running_job_id: str):
        super().__init__(
            f"a training job is already running (job_id={running_job_id})"
        )
        self.running_job_id = running_job_id


class TrainJobRegistry:
    """
    Bounded, thread-safe registry of training jobs.

    A job is a small dict:
        {
            "job_id":     str,
            "status":     "running" | "done" | "failed",
            "started_at": float (unix epoch seconds),
            "ended_at":   float | None,
            "result":     dict | None,   # set on done
            "error":      str  | None,   # set on failed
        }
    """

    def __init__(self, orchestrator, history_size: int = 50,
                 default_timeout_s: float = 120.0):
        if history_size < 1:
            raise ValueError("history_size must be >= 1")
        self._orchestrator     = orchestrator
        self._history_size     = int(history_size)
        self._default_timeout  = float(default_timeout_s)
        # OrderedDict preserves insertion order — eviction walks oldest-first
        # and skips anything still running.
        self._jobs: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ API

    def submit(self) -> str:
        """
        Start a new training job.

        Atomically: rejects if any job is currently running, otherwise inserts
        a 'running' record and spawns a worker thread that calls
        orchestrator.train_model() and updates the record when finished.

        Returns the new job_id.
        Raises TrainingAlreadyRunning if one is already in flight.
        """
        with self._lock:
            running = self._find_running_locked()
            if running is not None:
                raise TrainingAlreadyRunning(running["job_id"])

            job_id = str(uuid.uuid4())
            job = {
                "job_id":     job_id,
                "status":     STATUS_RUNNING,
                "started_at": time.time(),
                "ended_at":   None,
                "result":     None,
                "error":      None,
            }
            self._jobs[job_id] = job
            self._evict_if_needed_locked()

        logger.info(f"TrainJobs: submitted | job_id={job_id}")

        # Spawn the worker OUTSIDE the lock — Thread.start is fast but we
        # don't want any hypothetical blocking to hold the registry.
        worker = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            name=f"train-job-{job_id[:8]}",
            daemon=True,
        )
        worker.start()
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        """Return a *copy* of the job dict, or None if not found."""
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def list_all(self) -> list:
        """Return shallow copies of every job, oldest first. For debugging."""
        with self._lock:
            return [dict(j) for j in self._jobs.values()]

    # ----------------------------------------------------------- internals

    def _find_running_locked(self) -> Optional[dict]:
        """Must be called with self._lock held."""
        for job in self._jobs.values():
            if job["status"] == STATUS_RUNNING:
                return job
        return None

    def _evict_if_needed_locked(self) -> None:
        """
        Drop the oldest *terminal* jobs until size <= history_size.
        Running jobs are never evicted — if every slot is running, the
        registry temporarily exceeds history_size. That can't actually
        happen under the current single-job-at-a-time policy, but the
        defensive behavior is the right one.
        """
        while len(self._jobs) > self._history_size:
            evicted = None
            for jid, job in self._jobs.items():
                if job["status"] in _TERMINAL:
                    evicted = jid
                    break
            if evicted is None:
                # Nothing terminal to evict — give up rather than drop a live job.
                return
            self._jobs.pop(evicted, None)
            logger.debug(f"TrainJobs: evicted | job_id={evicted}")

    def _run_job(self, job_id: str) -> None:
        """
        Worker thread body. Calls the blocking orchestrator.train_model()
        and writes the result back to the registry. Never raises — all
        exceptions are converted to a 'failed' record so the registry
        stays consistent.
        """
        logger.info(f"TrainJobs: starting orchestrator.train_model() | job_id={job_id}")
        result = None
        error  = None
        try:
            result = self._orchestrator.train_model(timeout_s=self._default_timeout)
            # The orchestrator returns a dict {ok: bool, result, error}.
            # If ok=False, treat it as a failure even though no exception fired.
            if isinstance(result, dict) and result.get("ok") is False:
                error = result.get("error") or "training reported ok=False"
                result = None
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.exception(f"TrainJobs: train_model raised | job_id={job_id}")

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                # Evicted while running — shouldn't happen, defensive only.
                logger.warning(
                    f"TrainJobs: completed but record vanished | job_id={job_id}"
                )
                return
            if error is not None:
                job["status"] = STATUS_FAILED
                job["error"]  = error
                logger.error(f"TrainJobs: failed | job_id={job_id} | error={error}")
            else:
                job["status"] = STATUS_DONE
                job["result"] = result
                # Surface useful metrics into the log so logs ARE the history.
                metrics = (result or {}).get("metrics") if isinstance(result, dict) else None
                if isinstance(metrics, dict):
                    logger.info(
                        f"TrainJobs: done | job_id={job_id} | "
                        f"accuracy={metrics.get('accuracy')} | "
                        f"f1_macro={metrics.get('f1_macro')} | "
                        f"rows_used={(result or {}).get('rows_used')}"
                    )
                else:
                    logger.info(f"TrainJobs: done | job_id={job_id}")
            job["ended_at"] = time.time()