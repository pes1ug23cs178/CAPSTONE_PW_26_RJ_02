"""In-memory lookup of cached teacher outputs (cache/teacher), used in the MAIN process so DataLoader
workers do not each hold a copy."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


class TeacherCache:
    def __init__(self, cache_dir: Path) -> None:
        idx = pd.read_csv(cache_dir / "index.csv", dtype={"frame_path": str})
        shards = sorted(cache_dir.glob("shard_*.npz"))
        arrays: Dict[str, List[np.ndarray]] = {"logit": [], "z": [], "concepts": [], "maps": []}
        offsets = {}
        total = 0
        for s, path in enumerate(shards):
            d = np.load(path)
            offsets[s] = total
            total += len(d["logit"])
            for k in arrays:
                arrays[k].append(d[k])
        self.t = {k: torch.from_numpy(np.concatenate(v)) for k, v in arrays.items()}
        self.row = {fp: offsets[s] + r for fp, s, r in zip(idx["frame_path"], idx["shard"], idx["row"])}

    def lookup(self, paths: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        rows = torch.tensor([self.row[p] for p in paths])
        return {f"teacher_{k}": v[rows].to(device, non_blocking=True).float() for k, v in self.t.items()}
