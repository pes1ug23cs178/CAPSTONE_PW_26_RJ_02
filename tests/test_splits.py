"""Leakage guard for splits/*.csv. Run: pytest tests/test_splits.py -s"""
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pytest

from utils.config import REPO_ROOT, load_paths

SPLITS = ["ffpp_train", "ffpp_val", "ffpp_test", "dfd_test", "celebdf_test"]


@pytest.fixture(scope="module")
def frames():
    out = {}
    for s in SPLITS:
        p = REPO_ROOT / "splits" / f"{s}.csv"
        assert p.exists(), f"missing {p}; run python data/make_splits.py"
        out[s] = pd.read_csv(p, dtype=str, keep_default_na=False)
    return out


def test_no_video_in_two_splits(frames):
    owner = {}
    for s, df in frames.items():
        for v in df["video_id"].unique():
            assert v not in owner, f"video {v} in {owner.get(v)} and {s}"
            owner[v] = s


def test_no_ffpp_source_in_two_splits(frames):
    owner = defaultdict(set)
    for s in ("ffpp_train", "ffpp_val", "ffpp_test"):
        for src in frames[s]["source_id"].unique():
            owner[src].add(s)
    bad = {k: v for k, v in owner.items() if len(v) > 1}
    assert not bad, f"FF++ source IDs in several splits: {list(bad.items())[:5]}"


def test_dfd_and_celebdf_never_in_training(frames):
    train_sets = set(frames["ffpp_train"]["dataset"]) | set(frames["ffpp_val"]["dataset"])
    assert train_sets == {"ffpp"}
    dfd_sources = set(frames["dfd_test"]["source_id"])
    assert not dfd_sources & set(frames["ffpp_train"]["source_id"])


def test_no_frames_paths(frames):
    for s, df in frames.items():
        assert not df["frame_path"].str.contains("/frames/").any(), s


def test_every_file_exists(frames):
    root = load_paths()["data_root"]
    for s, df in frames.items():
        missing = [p for p in df["frame_path"] if not (root / p).is_file()]
        assert not missing, f"{s}: {len(missing)} missing, e.g. {missing[:3]}"


def test_print_counts(frames):
    for s, df in frames.items():
        g = df.groupby(["dataset", "method", "label"]).agg(videos=("video_id", "nunique"), crops=("frame_path", "size"))
        print(f"\n== {s} ==\n{g}")
