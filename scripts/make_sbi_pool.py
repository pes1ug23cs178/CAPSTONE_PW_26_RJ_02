"""T5.1a: freeze a pool of 40,000 SBI images from ffpp_train real crops (fixed seed), so the teacher's
outputs can be cached for distillation. Writes cache/sbi_pool/{images,boundary}/, cache/sbi_pool/sbi_pool.csv
and cache/sbi_pool/student_train.csv (= ffpp_train rows + pool rows, split-CSV schema + c1..c10, boundary_path).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import REPO_ROOT, load_paths  # noqa: E402


def make_chunk(args):
    start, rows, seed, out_dir, data_root, cache = args
    sys.path.insert(0, str(REPO_ROOT))
    from data.dataset import load_rgb
    from data.sbi import make_sbi
    rng = np.random.default_rng(seed + start)
    recs = []
    for i, (vid, fp, src) in enumerate(rows, start):
        lm_p = Path(cache) / "landmarks" / vid / f"{Path(fp).stem}.npy"
        img = load_rgb(Path(data_root) / fp)
        out, concepts, boundary, recipe, _ = make_sbi(img, np.load(lm_p), rng)
        stem = f"{i:06d}"
        ip = Path(out_dir) / "images" / f"{stem}.png"
        bp = Path(out_dir) / "boundary" / f"{stem}.npy"
        cv2.imwrite(str(ip), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        np.save(bp, boundary.astype(np.float32))
        rec = {"video_id": f"sbipool_{vid}", "frame_path": ip.relative_to(Path(data_root)).as_posix(), "label": 1,
               "dataset": "sbi_pool", "method": "sbi", "source_id": src, "landmarks_path": "",
               "boundary_path": bp.relative_to(Path(data_root)).as_posix(), "region": recipe["region"]}
        rec.update({f"c{k + 1}": float(concepts[k]) for k in range(10)})
        recs.append(rec)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--split", default="ffpp_train", help="training split under splits/ (fullff_train for full FF++)")
    args = ap.parse_args()
    paths = load_paths()
    data_root, cache = paths["data_root"], paths["cache_root"]
    free_gb = shutil.disk_usage(cache.parent).free / 1e9
    if free_gb < 50:
        sys.exit(f"Only {free_gb:.0f} GB free; runbook requires 50 GB before caching")
    train = pd.read_csv(REPO_ROOT / "splits" / f"{args.split}.csv", dtype=str, keep_default_na=False)
    real = train[train["method"] == "real"]
    real = real[[(cache / "landmarks" / r.video_id / f"{Path(r.frame_path).stem}.npy").exists() for r in real.itertuples()]]
    pick = real.sample(n=args.n, replace=len(real) < args.n, random_state=4242).reset_index(drop=True)
    out_dir = cache / "sbi_pool"
    for sub in ("images", "boundary"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    rows = list(zip(pick["video_id"], pick["frame_path"], pick["source_id"]))
    chunk = 1000
    jobs = [(s, rows[s:s + chunk], 4242, str(out_dir), str(data_root), str(cache)) for s in range(0, len(rows), chunk)]
    recs = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(make_chunk, jobs):
            recs += r
    pool_df = pd.DataFrame(recs)
    pool_df.to_csv(out_dir / "sbi_pool.csv", index=False)
    student = pd.concat([train, pool_df], ignore_index=True)
    student.to_csv(out_dir / "student_train.csv", index=False)
    print(f"SBI pool: {len(pool_df)} images, regions {pool_df['region'].value_counts().to_dict()}; "
          f"student_train.csv: {len(student)} rows")


if __name__ == "__main__":
    main()
