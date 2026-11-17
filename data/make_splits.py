"""Write leak-free split CSVs (one row per aligned face crop) into splits/.

FF++    : official split by source-video ID (splits/official/ffpp_{train,val,test}.json).
          Real = YouTube originals (3-digit names). Fakes Deepfakes/Face2Face/FaceSwap named AAA_BBB
          go to the split holding AAA.
DFD     : DFD actor real videos (non-3-digit names in FF++ REAL) + DeepFakeDetection fakes -> test only.
Celeb-DF: official List_of_testing_videos.txt (518 videos) if present at
          splits/official/List_of_testing_videos.txt; otherwise an INTERIM test set with the same
          composition (108 Celeb-real, 70 YouTube-real, 340 Celeb-synthesis) sampled with a fixed seed.
          Celeb-DF is never used for training, so either choice is leak-free.

Never reads any frames/ folder.

Usage: python data/make_splits.py
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import REPO_ROOT, load_paths  # noqa: E402

FIELDS = ["video_id", "frame_path", "label", "dataset", "method", "source_id", "landmarks_path"]
FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceSwap"]
YT_RE = re.compile(r"^\d{3}$")


def crops_of(video_dir: Path) -> List[Path]:
    aligned = video_dir / "aligned"
    if not aligned.is_dir():
        return []
    return sorted(p for p in aligned.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def rel(p: Path, data_root: Path) -> str:
    s = p.resolve().relative_to(data_root).as_posix()
    if "/frames/" in f"/{s}":
        raise RuntimeError(f"Refusing frames path: {s}")
    return s


def rows_for(video_dir: Path, data_root: Path, video_id: str, label: int, dataset: str,
             method: str, source_id: str) -> List[Dict]:
    return [
        {"video_id": video_id, "frame_path": rel(p, data_root), "label": label, "dataset": dataset,
         "method": method, "source_id": source_id, "landmarks_path": ""}
        for p in crops_of(video_dir)
    ]


def ffpp_splits(data_root: Path, official: Path) -> Dict[str, List[Dict]]:
    id_to_split: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        for pair in json.loads((official / f"ffpp_{split}.json").read_text()):
            for vid in pair:
                id_to_split[vid] = split
    out: Dict[str, List[Dict]] = {"ffpp_train": [], "ffpp_val": [], "ffpp_test": [], "dfd_test": []}
    faces = data_root / "ff++" / "faces"
    missing = Counter()

    for vdir in sorted((faces / "real" / "REAL").iterdir()):
        if not vdir.is_dir():
            continue
        name = vdir.name
        if YT_RE.match(name):
            split = id_to_split.get(name)
            if split is None:
                missing["real_not_in_official"] += 1
                continue
            out[f"ffpp_{split}"] += rows_for(vdir, data_root, f"ffpp_{name}", 0, "ffpp", "real", name)
        else:  # DFD actor video, e.g. 01__walking_down_street
            actor = name.split("__")[0]
            out["dfd_test"] += rows_for(vdir, data_root, f"dfd_real_{name}", 0, "dfd", "real", f"dfd_{actor}")

    for method in FFPP_METHODS:
        mdir = faces / "fake" / method
        if not mdir.is_dir():
            missing[f"no_dir_{method}"] += 1
            continue
        for vdir in sorted(mdir.iterdir()):
            if not vdir.is_dir():
                continue
            target = vdir.name.split("_")[0]
            split = id_to_split.get(target)
            if split is None:
                missing[f"{method}_not_in_official"] += 1
                continue
            out[f"ffpp_{split}"] += rows_for(vdir, data_root, f"ffpp_{method}_{vdir.name}", 1, "ffpp", method, target)

    dfd_dir = faces / "fake" / "DeepFakeDetection"
    for vdir in sorted(dfd_dir.iterdir()) if dfd_dir.is_dir() else []:
        if vdir.is_dir():
            actor = vdir.name.split("_")[0]
            out["dfd_test"] += rows_for(vdir, data_root, f"dfd_fake_{vdir.name}", 1, "dfd", "DeepFakeDetection", f"dfd_{actor}")
    if missing:
        print("FF++ videos skipped:", dict(missing))
    return out


def celebdf_split(data_root: Path, official: Path, seed: int = 0) -> (List[Dict], str):
    faces = data_root / "celebdf_v2" / "celebdf_v2" / "faces"
    real_dir, fake_dir = faces / "real" / "REAL", faces / "fake" / "Celeb-synthesis"
    on_disk = {}
    for d, kind in ((real_dir, "real"), (fake_dir, "fake")):
        for vdir in sorted(d.iterdir()):
            if vdir.is_dir() and crops_of(vdir):  # only videos that have aligned crops
                on_disk[vdir.name] = (vdir, kind)

    list_file = official / "List_of_testing_videos.txt"
    chosen: List[str] = []
    if list_file.exists():
        mode = "official"
        for line in list_file.read_text().splitlines():
            if not line.strip():
                continue
            _, path = line.split(maxsplit=1)
            name = Path(path.strip()).stem
            if name in on_disk:
                chosen.append(name)
            else:
                print(f"Celeb-DF official test video missing on disk: {path}")
    else:
        mode = "interim"
        rng = random.Random(seed)
        celeb_real = sorted(n for n, (_, k) in on_disk.items() if k == "real" and n.startswith("id"))
        yt_real = sorted(n for n, (_, k) in on_disk.items() if k == "real" and not n.startswith("id"))
        synth = sorted(n for n, (_, k) in on_disk.items() if k == "fake")
        chosen = rng.sample(celeb_real, 108) + rng.sample(yt_real, 70) + rng.sample(synth, 340)

    rows: List[Dict] = []
    for name in chosen:
        vdir, kind = on_disk[name]
        if kind == "real":
            method = "Celeb-real" if name.startswith("id") else "YouTube-real"
            source = name.split("_")[0] if name.startswith("id") else f"yt_{name}"
            rows += rows_for(vdir, data_root, f"cdf_{name}", 0, "celebdf", method, source)
        else:
            rows += rows_for(vdir, data_root, f"cdf_{name}", 1, "celebdf", "Celeb-synthesis", name.split("_")[0])
    return rows, mode


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def count_table(splits: Dict[str, List[Dict]]) -> str:
    lines = ["| split | method | label | videos | crops |", "| --- | --- | --- | --- | --- |"]
    for name, rows in splits.items():
        vids = defaultdict(set)
        crops = Counter()
        for r in rows:
            key = (r["method"], r["label"])
            vids[key].add(r["video_id"])
            crops[key] += 1
        for key in sorted(vids):
            lines.append(f"| {name} | {key[0]} | {key[1]} | {len(vids[key])} | {crops[key]} |")
        lines.append(f"| {name} | **all** | | **{len({r['video_id'] for r in rows})}** | **{len(rows)}** |")
    return "\n".join(lines)


def main() -> None:
    paths = load_paths()
    data_root = paths["data_root"]
    official = REPO_ROOT / "splits" / "official"
    splits = ffpp_splits(data_root, official)
    cdf_rows, mode = celebdf_split(data_root, official)
    splits["celebdf_test"] = cdf_rows
    for name, rows in splits.items():
        write_csv(REPO_ROOT / "splits" / f"{name}.csv", rows)
    table = count_table(splits)
    (REPO_ROOT / "splits" / "counts.md").write_text(
        f"Celeb-DF test mode: **{mode}**\n\n{table}\n", encoding="utf-8")
    print(f"Celeb-DF test mode: {mode}")
    print(table)


if __name__ == "__main__":
    main()
