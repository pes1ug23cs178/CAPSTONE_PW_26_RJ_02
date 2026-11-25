"""DeeperForensics-1.0 test split (third unseen test set, evaluation only) -> splits/dfo_test.csv.

DF-1.0 fakes put new faces on FF++ videos; the file name starts with the FF++ target ID (e.g. 012_W006 = FF++ 012
with actor W006). To keep training data out of the test, only fakes whose target is in the FF++ official TEST split
are used (140), with those same 140 FF++ test originals as the real class (the usual DF-1.0 protocol: reals from
FF++). Crops of both classes come from scripts/extract_faces.py (same pipeline).

python data/make_dfo_split.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.make_splits import count_table, rows_for, write_csv  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


def main() -> None:
    test_ids = {v for pair in json.loads((REPO_ROOT / "splits" / "official" / "ffpp_test.json").read_text()) for v in pair}
    train_val = {v for s in ("train", "val")
                 for pair in json.loads((REPO_ROOT / "splits" / "official" / f"ffpp_{s}.json").read_text()) for v in pair}
    rows = []
    fakes = sorted((REPO_ROOT / "deeperforensics" / "faces" / "end_to_end").iterdir())
    for vdir in fakes:
        target, actor = vdir.name.split("_")
        assert target in test_ids and target not in train_val, vdir.name
        rows += rows_for(vdir, REPO_ROOT, f"dfo_fake_{vdir.name}", 1, "dfo", "DeeperForensics", f"dfo_{actor}")
    for vid in sorted(test_ids):
        rows += rows_for(REPO_ROOT / "ff++_full" / "faces" / "original" / vid, REPO_ROOT, f"dfo_real_{vid}", 0, "dfo",
                         "real", vid)
    write_csv(REPO_ROOT / "splits" / "dfo_test.csv", rows)
    print(count_table({"dfo_test": rows}))


if __name__ == "__main__":
    main()
