"""Cell-balanced batch sampler (generalizes DomainClassBalancedBatchSampler from the old script).

Cells are named groups of dataset indices, e.g. real / Deepfakes / Face2Face / FaceSwap / sbi.
Each batch draws a fixed number from every cell (equal by default, or by `weights`), with a fixed
number of steps per epoch. Pools are shuffled and cycled, so small cells are re-used.
"""
from __future__ import annotations

import random
from typing import Dict, Iterator, List, Optional

from torch.utils.data import Sampler


def cell_quota(cells: List[str], batch_size: int, weights: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    w = {c: float((weights or {}).get(c, 1.0)) for c in cells}
    total = sum(w.values())
    raw = {c: batch_size * w[c] / total for c in cells}
    quota = {c: int(raw[c]) for c in cells}
    leftover = batch_size - sum(quota.values())
    for c in sorted(cells, key=lambda c: raw[c] - quota[c], reverse=True)[:leftover]:
        quota[c] += 1
    return quota


class CellBalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, cell_indices: Dict[str, List[int]], batch_size: int, steps_per_epoch: int,
                 weights: Optional[Dict[str, float]] = None, seed: int = 0) -> None:
        self.cells = list(cell_indices.keys())
        self.cell_indices = {c: list(v) for c, v in cell_indices.items()}
        self.batch_size = batch_size
        self.steps = steps_per_epoch
        self.quota = cell_quota(self.cells, batch_size, weights)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed * 1000 + self.epoch)
        pools = {c: rng.sample(v, len(v)) for c, v in self.cell_indices.items()}
        cursor = {c: 0 for c in self.cells}
        for _ in range(self.steps):
            batch: List[int] = []
            for c in self.cells:
                need = self.quota[c]
                while need > 0:
                    pool = pools[c]
                    if cursor[c] >= len(pool):
                        rng.shuffle(pool)
                        cursor[c] = 0
                    take = min(need, len(pool) - cursor[c])
                    batch.extend(pool[cursor[c]:cursor[c] + take])
                    cursor[c] += take
                    need -= take
            rng.shuffle(batch)
            yield batch
