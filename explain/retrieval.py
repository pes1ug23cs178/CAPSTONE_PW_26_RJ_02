"""T6.1: nearest training faces in the teacher's z space (FAISS inner product on L2-normalized z).

TrainingIndex(cache/teacher) -> .nearest(z, k=3) returns the k most similar training crops from k
different videos, with path, video_id, method and similarity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


class TrainingIndex:
    def __init__(self, cache_dir: Path, student_csv: Path) -> None:
        import faiss
        idx = pd.read_csv(cache_dir / "index.csv", dtype={"frame_path": str})
        meta = pd.read_csv(student_csv, dtype=str, keep_default_na=False).set_index("frame_path")
        zs = []
        for path in sorted(cache_dir.glob("shard_*.npz")):
            zs.append(np.load(path)["z"].astype(np.float32))
        z = np.concatenate(zs)
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        self.index = faiss.IndexFlatIP(z.shape[1])
        self.index.add(z)
        order = idx.sort_values(["shard", "row"])["frame_path"].tolist()
        self.meta = meta.loc[order, ["video_id", "method", "label"]].reset_index()

    def nearest(self, z: np.ndarray, k: int = 3, search: int = 200) -> List[Dict]:
        q = (z / (np.linalg.norm(z) + 1e-8)).astype(np.float32)[None]
        sims, ids = self.index.search(q, search)
        out, seen = [], set()
        for s, i in zip(sims[0], ids[0]):
            row = self.meta.iloc[int(i)]
            if row["video_id"] in seen:
                continue
            seen.add(row["video_id"])
            out.append({"frame_path": row["frame_path"], "video_id": row["video_id"], "method": row["method"],
                        "label": int(row["label"]), "similarity": float(s)})
            if len(out) == k:
                break
        return out
