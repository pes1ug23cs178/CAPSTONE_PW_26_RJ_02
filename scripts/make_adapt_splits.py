"""Extra splits for the optional experiments (run after the main queue if time allows).

cache/adapt/d2_train.csv   D2 (runbook filler): ffpp_train (labelled) + UNLABELLED Celeb-DF crops for unsupervised
                           domain adaptation. The pool excludes every celebdf_test video. Identities overlap with the
                           test set, as in Celeb-DF's official train/test split (59 celebrities in total; excluding test
                           identities would leave only 208 videos). Test videos themselves are never seen. Column unlabeled=1 marks
                           target rows (their labels are never used).
cache/adapt/x1_train.csv   X1 (extra): ffpp_train + all DFD videos as labelled training data (DFD is then no longer a
                           test set for X1; X1 is tested on ffpp_test and celebdf_test only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.make_splits import celebdf_split, crops_of, rows_for  # noqa: E402
from utils.config import REPO_ROOT, load_paths  # noqa: E402


def main() -> None:
    root = load_paths()["data_root"]
    out = REPO_ROOT / "cache" / "adapt"
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(REPO_ROOT / "splits" / "ffpp_train.csv", dtype=str, keep_default_na=False)
    test = pd.read_csv(REPO_ROOT / "splits" / "celebdf_test.csv", dtype=str, keep_default_na=False)
    test_videos = set(test["video_id"])
    test_ids = set(test["source_id"])
    faces = root / "celebdf_v2" / "celebdf_v2" / "faces"
    rows = []
    for sub, label, method in ((faces / "real" / "REAL", 0, None), (faces / "fake" / "Celeb-synthesis", 1, "Celeb-synthesis")):
        for vdir in sorted(sub.iterdir()):
            if not vdir.is_dir() or not crops_of(vdir):
                continue
            name = vdir.name
            vid = f"cdf_{name}"
            ids = {p for p in name.split("_") if p.startswith("id")} or {f"yt_{name}"}
            if vid in test_videos:
                continue
            m = method or ("Celeb-real" if name.startswith("id") else "YouTube-real")
            rows += rows_for(vdir, root, vid, label, "celebdf", m, sorted(ids)[0])
    target = pd.DataFrame(rows)
    target["unlabeled"] = "1"
    d2 = pd.concat([train.assign(unlabeled="0"), target], ignore_index=True)
    d2.to_csv(out / "d2_train.csv", index=False)
    dfd = pd.read_csv(REPO_ROOT / "splits" / "dfd_test.csv", dtype=str, keep_default_na=False)
    pd.concat([train, dfd], ignore_index=True).to_csv(out / "x1_train.csv", index=False)
    print(f"D2 target pool: {target['video_id'].nunique()} Celeb-DF videos, {len(target)} crops "
          f"(excluded the {len(test_videos)} test videos); real/fake videos: "
          f"{target.groupby('label')['video_id'].nunique().to_dict()} (labels not used in training)")
    print(f"X1 train: {len(train)} FF++ + {len(dfd)} DFD crops")


if __name__ == "__main__":
    main()
