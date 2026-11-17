"""Runbook rule after C2: if C2 is not better than C1 cross-dataset, retry with SBI share 1/4 (C2q) and keep
whichever wins on FF++ VAL (best epoch's (video AUC, frame AUC); ties keep the plan default 1/3).
The winning SBI cell weight is written into the GRACE configs that inherit C2's batch mix (G1, G2-DCT, G2-FFT;
G3/A3 are generated from G2 later). Test results are NOT used for this choice."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import REPO_ROOT  # noqa: E402

TARGETS = ["g1_concepts.yaml", "g2_dct.yaml", "g2_fft.yaml"]


def best_val(prefix: str):
    runs = sorted(p for p in glob.glob(str(REPO_ROOT / "runs" / f"{prefix}_s42_*")) if Path(p, "history.json").exists())
    if not runs:
        raise SystemExit(f"no finished run for {prefix}")
    hist = json.loads(Path(runs[-1], "history.json").read_text())
    return max((h["val_video_auc"], h["val_frame_auc"]) for h in hist), Path(runs[-1]).name


def main() -> None:
    (c2, c2_run), (c2q, c2q_run) = best_val("c2_clip_sbi"), best_val("c2q_clip_sbi_quarter")
    weight = 2 if c2q > c2 else 3
    for name in TARGETS:
        p = REPO_ROOT / "configs" / name
        cfg = yaml.safe_load(p.read_text())
        cfg["data"]["cell_weights"]["sbi"] = weight
        p.write_text(yaml.safe_dump(cfg, sort_keys=False, width=120))
    out = {"c2": {"run": c2_run, "val": c2}, "c2q": {"run": c2q_run, "val": c2q},
           "chosen_sbi_weight": weight, "share": "1/4" if weight == 2 else "1/3", "updated": TARGETS}
    (REPO_ROOT / "results").mkdir(exist_ok=True)
    (REPO_ROOT / "results" / "sbi_share_decision.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
