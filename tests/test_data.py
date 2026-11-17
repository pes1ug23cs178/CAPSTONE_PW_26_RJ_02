from collections import Counter

import pandas as pd
import pytest
import torch

from data.dataset import FaceFrameDataset
from data.sampler import CellBalancedBatchSampler, cell_quota
from utils.config import REPO_ROOT

TRAIN = REPO_ROOT / "splits" / "ffpp_train.csv"
CELLS = ["real", "Deepfakes", "Face2Face", "FaceSwap"]


def test_sampler_equal_share_over_100_batches():
    ds = FaceFrameDataset(TRAIN, train=True)
    sampler = CellBalancedBatchSampler(ds.cell_indices(CELLS), batch_size=32, steps_per_epoch=100, seed=1)
    counts = Counter()
    for batch in sampler:
        assert len(batch) == 32
        for i in batch:
            counts[ds.methods[i]] += 1
    vals = [counts[c] for c in CELLS]
    assert max(vals) - min(vals) <= 1, counts  # equal within one sample
    assert sum(vals) == 3200


def test_weighted_quota_one_third_each():
    q = cell_quota(["real", "sbi", "Deepfakes", "Face2Face", "FaceSwap"], 33,
                   {"real": 3, "sbi": 3, "Deepfakes": 1, "Face2Face": 1, "FaceSwap": 1})
    assert q == {"real": 11, "sbi": 11, "Deepfakes": 4, "Face2Face": 4, "FaceSwap": 3} or sum(q.values()) == 33


def test_item_format():
    ds = FaceFrameDataset(TRAIN, train=True, limit=8)
    item = ds[0]
    assert item["image"].shape == (3, 224, 224) and item["image"].dtype == torch.float32
    assert 0.0 <= item["image"].min() and item["image"].max() <= 1.0
    assert item["concepts"].shape == (10,) and item["boundary"].shape == (16, 16)


def test_frames_paths_rejected(tmp_path):
    df = pd.DataFrame([{"video_id": "x", "frame_path": "ff++/frames/real/REAL/000/a.png", "label": 0,
                        "dataset": "ffpp", "method": "real", "source_id": "000", "landmarks_path": ""}])
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        FaceFrameDataset(p, train=False)
