"""Temperature scaling fitted on FF++ validation (frame-level logits) -> runs/<run>/temperature.json.

python scripts/calibrate.py --models g3_full s3_kd_full [b1_swin ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import load_model_from_ckpt, predict  # noqa: E402
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    lg = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def nll(logits, labels, t):
    return float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(logits) / t, torch.tensor(labels, dtype=torch.float64)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["g3_full", "s3_kd_full"])
    args = ap.parse_args()
    for run in resolve_runs(args.models, [42]):
        pfile = run / "predictions_ffpp_val.csv"
        if pfile.exists() and "logit" in pd.read_csv(pfile, nrows=1).columns:
            pred = pd.read_csv(pfile)
        else:
            model, _ = load_model_from_ckpt(str(run / "best.pth"), torch.device("cuda"))
            pred = predict(model, REPO_ROOT / "splits/ffpp_val.csv", torch.device("cuda"))
        logits, labels = pred["logit"].to_numpy(np.float64), pred["label"].to_numpy(np.float64)
        t = fit_temperature(logits, labels)
        out = {"temperature": t, "val_nll_before": nll(logits, labels, 1.0), "val_nll_after": nll(logits, labels, t)}
        (run / "temperature.json").write_text(json.dumps(out, indent=2))
        print(run.name, out, flush=True)


if __name__ == "__main__":
    main()
