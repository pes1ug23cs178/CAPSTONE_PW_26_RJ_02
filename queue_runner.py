"""Sequential GPU job queue that survives crashes, restarts and closed terminals.

python queue_runner.py [--queue queue.yaml] [--exit-when-empty]

- Re-reads queue.yaml between jobs, so jobs can be appended while it runs.
- Each job: name, cmd, optional `after` (name of a job that must be `done` first).
- State in runs/_queue/state.json (pending / running / done / failed / blocked); logs in runs/_queue/<name>.log.
- Restart-safe: `done` jobs are skipped; a job left `running` by a dead runner is retried.
- A failed job does not stop the queue; jobs that depend on it are marked `blocked`.
- While a job runs, Windows is asked not to sleep (SetThreadExecutionState).
- Writes runs/_queue/heartbeat.txt every 30 s so a watcher can tell the runner is alive.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
QDIR = REPO / "runs" / "_queue"
STATE = QDIR / "state.json"
LOCK = QDIR / "runner.pid"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE)


def keep_awake(on: bool) -> None:
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def heartbeat(stop: threading.Event, info: dict) -> None:
    while not stop.is_set():
        (QDIR / "heartbeat.txt").write_text(f"{now()} pid={os.getpid()} job={info.get('job')}\n")
        stop.wait(30)


def next_job(jobs: list, state: dict):
    changed = False
    for job in jobs:
        name = job["name"]
        st = state.setdefault(name, {"status": "pending"})
        if st["status"] in ("done", "failed", "blocked"):
            continue
        dep = job.get("after")
        if dep:
            dep_status = state.get(dep, {}).get("status")
            if dep_status in ("failed", "blocked"):
                st.update(status="blocked", reason=f"dependency {dep} {dep_status}", at=now())
                changed = True
                continue
            if dep_status != "done":
                continue
        return job, changed
    return None, changed


def run_job(job: dict, state: dict, info: dict) -> None:
    name = job["name"]
    log_path = QDIR / f"{name}.log"
    st = state[name]
    st.update(status="running", start=now(), cmd=job["cmd"], log=str(log_path.relative_to(REPO)))
    st.pop("end", None)
    save_state(state)
    info["job"] = name
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    keep_awake(True)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {now()} START {name}: {job['cmd']}\n")
        log.flush()
        proc = subprocess.Popen(job["cmd"], shell=True, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env)
        st["pid"] = proc.pid
        save_state(state)
        rc = proc.wait()
        log.write(f"===== {now()} END {name}: exit {rc}\n")
    keep_awake(False)
    info["job"] = None
    st.update(status="done" if rc == 0 else "failed", end=now(), exit_code=rc, hours=round((time.time() - t0) / 3600, 2))
    save_state(state)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=str(REPO / "queue.yaml"))
    ap.add_argument("--exit-when-empty", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    QDIR.mkdir(parents=True, exist_ok=True)

    if LOCK.exists():
        old = int(LOCK.read_text().strip() or 0)
        if old and old != os.getpid() and pid_alive(old):
            sys.exit(f"Another queue_runner is running (pid {old}).")
    LOCK.write_text(str(os.getpid()))

    state = load_state()
    for name, st in state.items():  # recover from a dead runner
        if st.get("status") == "running":
            st.update(status="pending", note=f"interrupted, retried {now()}")
    save_state(state)

    info: dict = {"job": None}
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop, info), daemon=True).start()
    print(f"{now()} queue_runner pid {os.getpid()} watching {args.queue}", flush=True)
    try:
        while True:
            jobs = (yaml.safe_load(Path(args.queue).read_text()) or {}).get("jobs", [])
            state = load_state()
            job, changed = next_job(jobs, state)
            if changed:
                save_state(state)
            if job is None:
                if args.exit_when_empty:
                    break
                time.sleep(args.poll)
                continue
            print(f"{now()} start {job['name']}", flush=True)
            run_job(job, state, info)
            print(f"{now()} {job['name']} -> {state[job['name']]['status']}", flush=True)
    finally:
        stop.set()
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()


if __name__ == "__main__":
    main()
