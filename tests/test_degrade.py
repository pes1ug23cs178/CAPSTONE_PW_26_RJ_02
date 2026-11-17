import numpy as np
import pandas as pd

from data.degrade import build_h264_cache, downscale, image_degrade_fn, student_degrade
from data.dataset import FaceFrameDataset
from utils.config import REPO_ROOT, load_paths


def img():
    return (np.random.default_rng(0).random((224, 224, 3)) * 255).astype(np.uint8)


def test_jpeg_and_down_change_image_keep_shape():
    x = img()
    for spec in ("jpeg:30", "down:0.25"):
        y = image_degrade_fn(spec)(x)
        assert y.shape == x.shape and y.dtype == np.uint8 and np.abs(y.astype(int) - x).mean() > 1
    assert image_degrade_fn("clean") is None
    assert downscale(x, 1.0).shape == x.shape


def test_student_degrade_shapes():
    for _ in range(20):
        assert student_degrade(img()).shape == (224, 224, 3)


def test_h264_cache_roundtrip(tmp_path):
    df = pd.read_csv(REPO_ROOT / "splits/ffpp_test.csv", dtype=str)
    one = df[df["video_id"] == df["video_id"].iloc[0]]
    csv = tmp_path / "one.csv"
    one.to_csv(csv, index=False)
    paths = load_paths()
    mapping = build_h264_cache(str(csv), 40, paths["data_root"], paths["cache_root"], workers=2)
    assert len(mapping) == len(one)
    ds = FaceFrameDataset(csv, train=False, path_map=mapping)
    a = ds[0]["image"]
    clean = FaceFrameDataset(csv, train=False)[0]["image"]
    assert a.shape == clean.shape and (a - clean).abs().mean() > 0.002
