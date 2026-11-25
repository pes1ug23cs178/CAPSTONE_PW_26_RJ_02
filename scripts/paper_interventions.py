"""Interventions on the concept-bottleneck decision, computed exactly from saved predictions (CPU only).

The teacher's frame logit is  l = sum_k w_k s_k + b + r(z),  with s_k = sigmoid(c_k) stored as c1..c10 in
predictions_*.csv, concept_part = sum_k w_k s_k, and w, b read from the checkpoint (EMA weights). Because the
decision is linear in s, setting a concept to zero, do(s_k = 0), changes the logit by exactly -w_k s_k; no
forward pass is needed. Variants per frame:
  full                 l
  do(s_k=0)            l - w_k s_k                         (one concept switched off)
  concepts only        sum_k w_k s_k + b                   (residual r(z) removed)
  residual only        b + r(z)                            (all ten concepts switched off)
Video score = mean frame probability. Reported per run, then mean and std over seeds:
  video AUC of each variant, its change from the full model, the mean logit contribution w_k s_k on fake and on
  real frames, and the share of fake videos flagged by the full model (FF++-val threshold) whose verdict flips
  to real under the variant.
Writes results/paper_interventions.csv (one row per model, seed, split, condition, variant).

python scripts/paper_interventions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402

MODELS = ["f1_grace_dct", "f1h_grace_dct_h264"]
SPLITS = ["ffpp_test", "dfd_test", "celebdf_test"]
CONDITIONS = ["", "_h26440"]
C = [f"c{k}" for k in range(1, 11)]


def sig(x):
    return 1 / (1 + np.exp(-x))


def video_scores(d, logit):
    v = pd.DataFrame({"video_id": d.video_id, "label": d.label, "p": sig(logit)}).groupby("video_id")
    v = v.agg(label=("label", "first"), p=("p", "mean"))
    return v.label.to_numpy(), v.p.to_numpy()


def main():
    rows = []
    for model in MODELS:
        for run in resolve_runs([model], [42, 1, 2]):
            ck = torch.load(run / "best.pth", map_location="cpu", weights_only=False)
            sd = ck.get("ema") or ck["model"]
            w = sd["decision.w"].float().numpy()
            b = float(sd["decision.b"].float().numpy()[0])
            thr = json.loads((run / "val_threshold.json").read_text())["threshold"]
            seed = int(ck["config"].get("seed", -1)) if isinstance(ck.get("config"), dict) else -1
            for split in SPLITS:
                for cond in CONDITIONS:
                    f = run / f"predictions_{split}{cond}.csv"
                    if not f.exists():
                        continue
                    d = pd.read_csv(f)
                    s = d[C].to_numpy()
                    contrib = s * w
                    err = np.abs(contrib.sum(1) - d.concept_part.to_numpy()).max()
                    assert err < 1e-3, f"concept weights do not reproduce concept_part ({err})"
                    logit = d.logit.to_numpy()
                    resid = logit - d.concept_part.to_numpy() - b
                    fake = d.label.to_numpy() == 1
                    y, p_full = video_scores(d, logit)
                    auc_full = roc_auc_score(y, p_full)
                    flagged = (y == 1) & (p_full >= thr)
                    variants = {"full": logit, "concepts only": contrib.sum(1) + b, "residual only": b + resid}
                    for k in range(10):
                        variants[f"off {C[k]}"] = logit - contrib[:, k]
                    for name, lv in variants.items():
                        yv, pv = video_scores(d, lv)
                        k = int(name.split("c")[-1]) - 1 if name.startswith("off") else None
                        rows.append(dict(
                            model=model, seed=seed, run=run.name, split=split.replace("_test", ""),
                            condition=cond.strip("_") or "clean", variant=name, video_auc=roc_auc_score(yv, pv),
                            delta_auc=roc_auc_score(yv, pv) - auc_full,
                            flip_rate=float(((pv < thr) & flagged).sum() / max(flagged.sum(), 1)),
                            contrib_fake=float(contrib[fake, k].mean()) if k is not None else np.nan,
                            contrib_real=float(contrib[~fake, k].mean()) if k is not None else np.nan,
                            weight=float(w[k]) if k is not None else np.nan,
                            concept_share=float(d.concept_part.var() / d.logit.var()),
                            n_fake_flagged=int(flagged.sum())))
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "paper_interventions.csv"
    df.to_csv(out, index=False)
    g = df.groupby(["model", "split", "condition", "variant"])[["video_auc", "delta_auc", "flip_rate", "contrib_fake",
                                                                "contrib_real", "weight"]].agg(["mean", "std"])
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(g.round(3))
    print(f"wrote {out} ({len(df)} rows, seeds per model: {df.groupby('model').run.nunique().to_dict()})")


if __name__ == "__main__":
    main()
