"""Job queue for the web server: many clients, a fixed number of model workers.

Each check is a Job. Jobs wait in one FIFO queue per mode and are taken by that mode's worker threads
(Deep: 1 GPU worker, Quick: 2 CPU workers by default), so the models are loaded once and never run more jobs at
the same time than the hardware allows. The web layer only adds jobs and reads their state, so it never blocks.

Limits: a full queue refuses new jobs (QueueFull, the server answers 503); a client may have `max_active` jobs
waiting or running and `max_per_window` jobs per `window` seconds (RateLimited, 429). A job that runs longer than
`timeout` seconds or is cancelled stops at its next progress step. Finished jobs and their files are removed after
`ttl` seconds, and at most `max_kept` finished jobs are kept in memory (oldest dropped first).

No model code here: the work is a function run(job, progress) passed in, so the queue is tested without a GPU.
"""
from __future__ import annotations

import collections
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional


class QueueFull(Exception):
    pass


class RateLimited(Exception):
    pass


class JobStopped(Exception):
    """Raised inside a running job when it is cancelled or over its time limit."""


class UserError(Exception):
    """A problem with the user's file (no face, unreadable video): shown to them as a friendly message."""

    def __init__(self, title: str, message: str) -> None:
        super().__init__(message)
        self.title, self.message = title, message


@dataclass
class Job:
    mode: str
    path: Path
    name: str
    client: str
    is_video: bool
    sha256: str = ""
    id: str = field(default_factory=lambda: secrets.token_urlsafe(16))  # 128-bit, unguessable
    state: str = "queued"          # queued -> running -> done | error | cancelled
    stage: str = "waiting"
    done: int = 0
    total: int = 0
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    result: Optional[dict] = None
    analysis: Any = None           # kept for the overlay / switch-off / report follow-ups
    error: Optional[dict] = None   # {"title", "message"}
    cancel: bool = False

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")


class JobManager:
    def __init__(self, run: Callable[[Job, Callable], None], workers: Dict[str, int], max_waiting: int = 20,
                 max_active: int = 1, max_per_window: int = 10, window: float = 600, timeout: float = 180,
                 ttl: float = 1800, max_kept: int = 30, on_drop: Optional[Callable[[Job], None]] = None,
                 warmup: Optional[Callable[[str], None]] = None) -> None:
        self.run, self.workers = run, dict(workers)
        self.max_waiting, self.max_active, self.max_per_window, self.window = max_waiting, max_active, max_per_window, window
        self.timeout, self.ttl, self.max_kept = timeout, ttl, max_kept
        self.on_drop = on_drop or (lambda job: None)
        self.warmup = warmup  # warmup(mode), run once in each worker thread before its first job
        self.jobs: Dict[str, Job] = {}
        self.waiting: Dict[str, Deque[Job]] = {m: collections.deque() for m in workers}
        self.recent: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)  # client -> submit times
        self.durations: Dict[str, Deque[float]] = {m: collections.deque(maxlen=20) for m in workers}
        self.cv = threading.Condition()
        self.threads: List[threading.Thread] = []
        self.stopping = False
        self.warm = 0  # workers that finished warming up

    # ---------- lifecycle ----------
    def start(self) -> "JobManager":
        for mode, n in self.workers.items():
            for i in range(n):
                t = threading.Thread(target=self._worker, args=(mode,), name=f"worker-{mode}-{i}", daemon=True)
                t.start()
                self.threads.append(t)
        t = threading.Thread(target=self._janitor, name="janitor", daemon=True)
        t.start()
        self.threads.append(t)
        return self

    def stop(self) -> None:
        with self.cv:
            self.stopping = True
            self.cv.notify_all()

    # ---------- web-layer calls ----------
    def submit(self, job: Job) -> Job:
        if job.mode not in self.waiting:
            raise ValueError(f"unknown mode {job.mode}")
        now = time.time()
        with self.cv:
            if sum(len(q) for q in self.waiting.values()) >= self.max_waiting:
                raise QueueFull()
            if self.max_active and sum(j.active and j.client == job.client for j in self.jobs.values()) >= self.max_active:
                raise RateLimited("You already have a check running. Wait for it to finish, or cancel it.")
            times = self.recent[job.client]
            while times and now - times[0] > self.window:
                times.popleft()
            if self.max_per_window and len(times) >= self.max_per_window:
                wait = int(self.window - (now - times[0])) + 1
                raise RateLimited(f"You've reached {self.max_per_window} checks in {int(self.window // 60)} minutes. "
                                  f"Please try again in {max(wait // 60, 1)} minute{'s' if wait >= 120 else ''}.")
            times.append(now)
            self.jobs[job.id] = job
            self.waiting[job.mode].append(job)
            self.cv.notify_all()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self.cv:
            return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self.cv:
            job = self.jobs.get(job_id)
            if job is None or not job.active:
                return False
            job.cancel = True
            if job.state == "queued":
                self.waiting[job.mode].remove(job)
                self._finish(job, "cancelled")
            return True

    def position(self, job: Job) -> int:
        """1 = next in line; 0 = running or finished."""
        with self.cv:
            q = self.waiting.get(job.mode, ())
            return list(q).index(job) + 1 if job in q else 0

    def eta(self, job: Job) -> Optional[float]:
        """Rough seconds until this job is finished, from the last 20 job durations of its mode."""
        if not job.active:
            return None
        d = self.durations[job.mode]
        avg = sum(d) / len(d) if d else (20.0 if job.mode == "Deep" else 12.0)
        pos = self.position(job)
        if job.state == "running":
            return max(avg - (time.time() - (job.started or time.time())), 1.0)
        return avg * (1 + (pos - 1) // self.workers[job.mode]) + avg

    def stats(self) -> dict:
        with self.cv:
            return {"waiting": {m: len(q) for m, q in self.waiting.items()},
                    "running": sum(j.state == "running" for j in self.jobs.values()),
                    "kept": len(self.jobs)}

    # ---------- workers ----------
    def _worker(self, mode: str) -> None:
        if self.warmup:
            try:
                self.warmup(mode)
            except Exception:  # noqa: BLE001  (the first job then pays the start-up cost, or reports the error)
                pass
        with self.cv:
            self.warm += 1
        while True:
            with self.cv:
                while not self.waiting[mode] and not self.stopping:
                    self.cv.wait(timeout=1.0)
                if self.stopping:
                    return
                job = self.waiting[mode].popleft()
                job.state, job.started, job.stage = "running", time.time(), "starting"

            def progress(stage: str, done: int = 0, total: int = 0, job=job) -> None:
                if job.cancel:
                    raise JobStopped("cancelled")
                if time.time() - job.started > self.timeout:
                    raise JobStopped("timeout")
                job.stage, job.done, job.total = stage, done, total

            try:
                self.run(job, progress)
                state = "done"
            except JobStopped as e:
                state = "cancelled" if str(e) == "cancelled" else "error"
                if state == "error":
                    job.error = {"title": "This check took too long",
                                 "message": "Please try a shorter clip or a smaller file."}
            except UserError as e:
                state, job.error = "error", {"title": e.title, "message": e.message}
            except Exception as e:  # noqa: BLE001  (a bug must not kill the worker)
                state = "error"
                job.error = {"title": "Something went wrong on our side",
                             "message": f"Please try again. If it keeps happening, tell us ({type(e).__name__})."}
            with self.cv:
                if state == "done":
                    self.durations[mode].append(time.time() - job.started)
                self._finish(job, state)

    def _finish(self, job: Job, state: str) -> None:  # caller holds self.cv
        job.state, job.finished = state, time.time()
        if state != "done":
            job.analysis = None
        try:
            Path(job.path).unlink(missing_ok=True)  # uploads are deleted as soon as the check ends
        except OSError:
            pass
        self.cv.notify_all()

    def _janitor(self, every: float = 30.0) -> None:
        while not self.stopping:
            self.sweep()
            with self.cv:
                self.cv.wait(timeout=every)

    def sweep(self) -> None:
        """Drop finished jobs older than ttl, and the oldest finished ones beyond max_kept."""
        now = time.time()
        with self.cv:
            done = sorted((j for j in self.jobs.values() if not j.active), key=lambda j: j.finished or 0)
            drop = [j for j in done if now - (j.finished or now) > self.ttl]
            keep = [j for j in done if j not in drop]
            drop += keep[: max(len(keep) - self.max_kept, 0)]
            for j in drop:
                del self.jobs[j.id]
                self.on_drop(j)
