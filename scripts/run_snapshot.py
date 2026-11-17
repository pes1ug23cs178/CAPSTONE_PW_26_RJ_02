"""Run a queue job from a frozen copy of the code.

Windows DataLoader workers re-import modules from disk every time they start, so editing the repository
while a job trains can crash it (this killed B1 at epoch 3 on 17 Nov). This wrapper copies the code into
runs/_code/<stamp>_<pid>/, runs the command there, and points GRACE_REPO_ROOT at the real repository so
splits, configs, data, runs, cache and experiments.csv still resolve to the real folders.

Usage (in queue.yaml): python scripts/run_snapshot.py -- python train.py --config configs/b1_swin.yaml --seed 42
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE_DIRS = ["data", "models", "utils", "explain", "scripts", "app"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        sys.exit("usage: run_snapshot.py -- <command ...>")
    snap = REPO / "runs" / "_code" / f"{datetime.now():%Y%m%d-%H%M%S}_{os.getpid()}"
    snap.mkdir(parents=True)
    for d in CODE_DIRS:
        if (REPO / d).is_dir():
            shutil.copytree(REPO / d, snap / d, ignore=IGNORE)
    for f in REPO.glob("*.py"):
        shutil.copy2(f, snap / f.name)
    env = dict(os.environ, GRACE_REPO_ROOT=str(REPO), PYTHONUNBUFFERED="1")
    print(f"[run_snapshot] code frozen in {snap}", flush=True)
    rc = subprocess.call(" ".join(argv), shell=True, cwd=snap, env=env)
    sys.exit(rc)


if __name__ == "__main__":
    main()
