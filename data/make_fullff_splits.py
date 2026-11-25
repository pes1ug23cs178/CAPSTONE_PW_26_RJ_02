"""Split CSVs for the full FF++ c23 download (ff++_full/faces, all five methods), official split by source ID.

Same rules as data/make_splits.py: real = YouTube originals (3-digit IDs); a fake named AAA_BBB goes to the
split that holds AAA (official pairs are never split). DFD and Celeb-DF test CSVs are not touched.
Writes splits/fullff_{train,val,test}.csv and splits/fullff_counts.md.

python data/make_fullff_splits.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.make_splits import count_table, rows_for, write_csv  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]


def main() -> None:
    official = REPO_ROOT / "splits" / "official"
    id_to_split = {}
    for split in ("train", "val", "test"):
        for pair in json.loads((official / f"ffpp_{split}.json").read_text()):
            for vid in pair:
                id_to_split[vid] = split
    faces = REPO_ROOT / "ff++_full" / "faces"
    out = {f"fullff_{s}": [] for s in ("train", "val", "test")}
    for vdir in sorted((faces / "original").iterdir()):
        s = id_to_split[vdir.name]
        out[f"fullff_{s}"] += rows_for(vdir, REPO_ROOT, f"ffpp_{vdir.name}", 0, "ffpp", "real", vdir.name)
    for m in METHODS:
        for vdir in sorted((faces / m).iterdir()):
            a, b = vdir.name.split("_")
            assert id_to_split[a] == id_to_split[b], vdir.name  # official pairs stay together
            s = id_to_split[a]
            out[f"fullff_{s}"] += rows_for(vdir, REPO_ROOT, f"ffpp_{m}_{vdir.name}", 1, "ffpp", m, a)
    for name, rows in out.items():
        write_csv(REPO_ROOT / "splits" / f"{name}.csv", rows)
    table = count_table(out)
    (REPO_ROOT / "splits" / "fullff_counts.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
