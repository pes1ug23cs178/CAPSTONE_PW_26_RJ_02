"""Runbook rule after G1: concept share below 0.8 -> one retry with residual penalty 1e-2 (G1r); keep the better and
report the trade-off. Decided on FF++ VAL only (best epoch of each run):
G1r is kept if its val video AUC is within 0.002 of G1's and its val concept share is higher; otherwise G1 is kept.
The chosen residual penalty goes into configs/g2_fft.yaml (not yet trained) and results/residual_decision.json, which
make_g3_configs.py reads for G3. G2-DCT already trained with 1e-3 (documented in STATUS.md)."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import REPO_ROOT  # noqa: E402


def best(prefix: str):
    runs = sorted(p for p in glob.glob(str(REPO_ROOT / "runs" / f"{prefix}_s42_*")) if Path(p, "history.json").exists())
    if not runs:
        raise SystemExit(f"no finished run for {prefix}")
    hist = json.loads(Path(runs[-1], "history.json").read_text())
    h = max(hist, key=lambda r: (r["val_video_auc"], r["val_frame_auc"]))
    return {"run": Path(runs[-1]).name, "epoch": h["epoch"], "val_video_auc": h["val_video_auc"],
            "val_frame_auc": h["val_frame_auc"], "val_concept_share": h.get("val_concept_share")}


def main() -> None:
    g1, g1r = best("g1_concepts"), best("g1r_concepts_res1e2")
    keep_r = g1r["val_video_auc"] >= g1["val_video_auc"] - 0.002 and (g1r["val_concept_share"] or 0) > (g1["val_concept_share"] or 0)
    residual = 1e-2 if keep_r else 1e-3
    p = REPO_ROOT / "configs" / "g2_fft.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["loss"]["residual"] = residual
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, width=120))
    out = {"g1": g1, "g1r": g1r, "kept": "g1r" if keep_r else "g1", "residual": residual}
    (REPO_ROOT / "results").mkdir(exist_ok=True)
    (REPO_ROOT / "results" / "residual_decision.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
