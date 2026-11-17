"""Shared helpers: seeding, git hash, run folders, logging, experiments.csv."""
from __future__ import annotations

import csv
import logging
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from utils.config import REPO_ROOT

EXPERIMENT_FIELDS = [
    "run_id", "config", "parent", "seed", "commit", "start", "end", "gpu_hours",
    "best_epoch", "val_video_auc", "ffpp_test_video_auc", "dfd_test_video_auc",
    "celebdf_test_video_auc", "notes",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_run_dir(runs_root: str, name: str, seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = Path(runs_root) / f"{name}_s{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("grace")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def append_experiment(row: Dict[str, object], path: Optional[Path] = None) -> None:
    path = path or (REPO_ROOT / "experiments.csv")
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in EXPERIMENT_FIELDS})


def default_num_workers() -> int:
    return int(os.environ.get("GRACE_NUM_WORKERS", 6))
