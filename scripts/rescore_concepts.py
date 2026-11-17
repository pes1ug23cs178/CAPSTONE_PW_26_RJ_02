"""Recompute concept AUC on cache/sbi_val and cache/sbi_test for finished runs (used after the c1 fix in
evaluate.concept_auc: c1 is scored against the real source crops). Writes <run>/concept_auc.json.
Usage: python scripts/rescore_concepts.py <run prefix> [...]   e.g. g1_concepts_s42"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import concept_auc, load_model_from_ckpt  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for prefix in sys.argv[1:]:
        runs = sorted(p for p in glob.glob(str(REPO_ROOT / "runs" / f"{prefix}_*")) if Path(p, "best.pth").exists())
        if not runs:
            raise SystemExit(f"no run with best.pth for {prefix}")
        run = Path(runs[-1])
        model, _ = load_model_from_ckpt(str(run / "best.pth"), device)
        model.eval()
        res = {s: concept_auc(model, REPO_ROOT / "cache" / s / f"{s}.csv", device) for s in ("sbi_val", "sbi_test")}
        (run / "concept_auc.json").write_text(json.dumps(res, indent=2))
        print(run.name, json.dumps(res))


if __name__ == "__main__":
    main()
