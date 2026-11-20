"""Switch Celeb-DF from the INTERIM test set to the official List_of_testing_videos.txt (received 20 Nov 2026).

python scripts/celebdf_official.py prepare
    - keeps the interim split as splits/celebdf_test_interim.csv
    - re-runs data/make_splits.py (now in "official" mode) and checks every other split file is byte-identical
    - moves each run's interim Celeb-DF outputs (metrics_/predictions_/roc_/video_cues_celebdf_test*) into
      runs/<run>/celebdf_interim/ so scripts/eval_all.py recomputes them on the official list (nothing is deleted)
python scripts/celebdf_official.py sync
    - copies the official clean Celeb-DF video AUC from runs/<run>/metrics_celebdf_test.json into
      experiments.csv (celebdf_test_video_auc); the interim value is kept in celebdf_interim_video_auc
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import REPO_ROOT  # noqa: E402

SPLITS = REPO_ROOT / "splits"
PREFIXES = ("metrics_celebdf_test", "predictions_celebdf_test", "roc_celebdf_test", "video_cues_celebdf_test")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def prepare() -> None:
    if not (SPLITS / "official" / "List_of_testing_videos.txt").exists():
        sys.exit("official list missing")
    interim = SPLITS / "celebdf_test_interim.csv"
    if not interim.exists():
        shutil.copy2(SPLITS / "celebdf_test.csv", interim)
    before = {p.name: md5(p) for p in SPLITS.glob("*.csv") if p.name not in ("celebdf_test.csv", "celebdf_test_interim.csv")}
    subprocess.run([sys.executable, str(REPO_ROOT / "data" / "make_splits.py")], check=True, cwd=REPO_ROOT)
    after = {n: md5(SPLITS / n) for n in before}
    changed = [n for n in before if before[n] != after[n]]
    if changed:
        sys.exit(f"other splits changed: {changed}")
    new = pd.read_csv(SPLITS / "celebdf_test.csv", dtype=str)
    print(f"official Celeb-DF test: {new.video_id.nunique()} videos, {len(new)} crops, "
          f"labels {new.drop_duplicates('video_id').label.value_counts().to_dict()}")
    moved = 0
    for run in sorted((REPO_ROOT / "runs").iterdir()):
        if not run.is_dir() or run.name.startswith("_"):
            continue
        files = [f for f in run.iterdir() if f.is_file() and f.name.startswith(PREFIXES)]
        if not files:
            continue
        dst = run / "celebdf_interim"
        dst.mkdir(exist_ok=True)
        for f in files:
            shutil.move(str(f), str(dst / f.name))
            moved += 1
    print(f"archived {moved} interim Celeb-DF files into runs/*/celebdf_interim/")


def sync() -> None:
    p = REPO_ROOT / "experiments.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if "celebdf_interim_video_auc" not in df.columns:
        df.insert(df.columns.get_loc("celebdf_test_video_auc") + 1, "celebdf_interim_video_auc", df["celebdf_test_video_auc"])
    n = 0
    for i, rid in df["run_id"].items():
        m = REPO_ROOT / "runs" / rid / "metrics_celebdf_test.json"
        if m.exists():
            df.at[i, "celebdf_test_video_auc"] = f"{json.loads(m.read_text())['video_auc']:.4f}"
            n += 1
        elif (REPO_ROOT / "runs" / rid / "celebdf_interim").exists():
            df.at[i, "celebdf_test_video_auc"] = ""  # not re-evaluated on the official list: no official number
    df.to_csv(p, index=False)
    print(f"experiments.csv: official Celeb-DF AUC for {n} runs (interim kept in celebdf_interim_video_auc)")


if __name__ == "__main__":
    {"prepare": prepare, "sync": sync}[sys.argv[1]]()
