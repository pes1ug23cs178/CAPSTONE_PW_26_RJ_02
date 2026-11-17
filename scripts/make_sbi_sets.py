"""T3.1: fixed SBI evaluation sets (fixed seeds).

cache/sbi_val/  : 2,000 SBI images from ffpp_val real crops   (concept accuracy for tuning)
cache/sbi_test/ : 2,000 SBI images from ffpp_test real crops  (concept accuracy for reporting; pointing game)

Each set has images/*.png, masks/*.npy (224x224 float16 blend mask), boundary/*.npy (16x16) and
sbi_<set>.csv in the split-CSV schema (so FaceFrameDataset can read it) plus columns c1..c10, region,
mask_path, boundary_path and recipe flags.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_rgb  # noqa: E402
from data.sbi import make_sbi  # noqa: E402
from utils.config import REPO_ROOT, load_paths  # noqa: E402

SETS = {"sbi_val": ("ffpp_val", 1001), "sbi_test": ("ffpp_test", 2002)}
N = 2000


def main() -> None:
    paths = load_paths()
    data_root, cache = paths["data_root"], paths["cache_root"]
    for set_name, (split, seed) in SETS.items():
        df = pd.read_csv(REPO_ROOT / "splits" / f"{split}.csv", dtype=str)
        df = df[df["method"] == "real"].reset_index(drop=True)
        df = df[[ (cache / "landmarks" / r.video_id / f"{Path(r.frame_path).stem}.npy").exists() for r in df.itertuples()]]
        if len(df) == 0:
            sys.exit(f"No landmarks for {split} real crops; run extract_landmarks.py --splits {split}")
        rows = df.sample(n=N, replace=len(df) < N, random_state=seed).reset_index(drop=True)
        out_dir = cache / set_name
        for sub in ("images", "masks", "boundary"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        records = []
        for i, r in enumerate(rows.itertuples()):
            img = load_rgb(data_root / r.frame_path)
            lm = np.load(cache / "landmarks" / r.video_id / f"{Path(r.frame_path).stem}.npy")
            out, concepts, boundary, recipe, mask = make_sbi(img, lm, rng)
            stem = f"{i:05d}"
            img_p, mask_p, b_p = out_dir / "images" / f"{stem}.png", out_dir / "masks" / f"{stem}.npy", out_dir / "boundary" / f"{stem}.npy"
            cv2.imwrite(str(img_p), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            np.save(mask_p, mask.astype(np.float16))
            np.save(b_p, boundary.astype(np.float32))
            rec = {"video_id": f"{set_name}_{stem}", "frame_path": img_p.relative_to(data_root).as_posix(), "label": 1,
                   "dataset": set_name, "method": "sbi", "source_id": r.source_id, "landmarks_path": "",
                   "source_frame": r.frame_path, "region": recipe["region"],
                   "mask_path": mask_p.relative_to(data_root).as_posix(), "boundary_path": b_p.relative_to(data_root).as_posix(),
                   "recipe": json.dumps(recipe)}
            rec.update({f"c{k + 1}": float(concepts[k]) for k in range(10)})
            records.append(rec)
        pd.DataFrame(records).to_csv(out_dir / f"{set_name}.csv", index=False)
        regions = Counter(r["region"] for r in records)
        flags = Counter(k for r in records for k, v in json.loads(r["recipe"]).items() if v is True)
        print(f"{set_name}: {len(records)} images from {rows['video_id'].nunique()} videos | regions {dict(regions)} | transforms {dict(flags)}")


if __name__ == "__main__":
    main()
