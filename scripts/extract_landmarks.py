"""T2.1: 68-point 2-D landmarks for every REAL crop in ffpp_train.csv and ffpp_val.csv.

Crops are already aligned faces, so the whole 224x224 image is passed as the face box and the
detector is skipped. Saves cache/landmarks/<video_id>/<frame_stem>.npy (68 x 2, float32, pixels),
logs failures to cache/landmarks/failures.csv and draws cache/landmarks_check.png (16 crops).

python scripts/extract_landmarks.py [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

# face-alignment wraps its nets in torch.compile, which needs Triton (absent on Windows) and then
# fails on every call. Disable dynamo before torch is imported.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_rgb  # noqa: E402
from utils.config import REPO_ROOT, load_paths  # noqa: E402


def landmark_path(cache_root: Path, video_id: str, frame_path: str) -> Path:
    return cache_root / "landmarks" / video_id / f"{Path(frame_path).stem}.npy"


def draw_check(rows, cache_root: Path, data_root: Path, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, (vid, fp) in zip(axes.flat, rows):
        ax.imshow(load_rgb(data_root / fp))
        lm = np.load(landmark_path(cache_root, vid, fp))
        ax.scatter(lm[:, 0], lm[:, 1], s=4, c="lime")
        ax.set_title(vid, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    import face_alignment

    paths = load_paths()
    data_root, cache_root = paths["data_root"], paths["cache_root"]
    df = pd.concat([pd.read_csv(REPO_ROOT / "splits" / f"{s}.csv", dtype=str) for s in ("ffpp_train", "ffpp_val")])
    df = df[df["method"] == "real"].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    todo = [(r.video_id, r.frame_path) for r in df.itertuples() if not landmark_path(cache_root, r.video_id, r.frame_path).exists()]
    print(f"{len(df)} real crops, {len(todo)} to process", flush=True)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cuda", face_detector="sfd")
    box = [[0, 0, 224, 224]]
    fail_path = cache_root / "landmarks" / "failures.csv"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for k, (vid, fp) in enumerate(todo, 1):
        try:
            img = load_rgb(data_root / fp)
            with torch.no_grad():
                preds = fa.get_landmarks_from_image(img, detected_faces=box)
            if not preds:
                raise RuntimeError("no landmarks")
            out = landmark_path(cache_root, vid, fp)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, preds[0][:, :2].astype(np.float32))
        except Exception as e:  # noqa: BLE001
            failures.append({"video_id": vid, "frame_path": fp, "error": str(e)[:200]})
        if k % 2000 == 0:
            print(f"{k}/{len(todo)} done, {len(failures)} failures", flush=True)

    with open(fail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "frame_path", "error"])
        w.writeheader()
        w.writerows(failures)
    total = len(df)
    rate = len(failures) / max(total, 1)
    print(f"failures: {len(failures)} / {total} = {rate:.2%}", flush=True)
    ok = [(r.video_id, r.frame_path) for r in df.itertuples() if landmark_path(cache_root, r.video_id, r.frame_path).exists()]
    draw_check(random.Random(0).sample(ok, min(16, len(ok))), cache_root, data_root, cache_root / "landmarks_check.png")
    if rate >= 0.02:
        sys.exit(f"Failure rate {rate:.2%} >= 2%")


if __name__ == "__main__":
    main()
