"""Paper statistics: bootstrap 95% confidence intervals and paired model comparisons on video AUC.

python scripts/paper_stats.py [--boot 1000]

For each model the reported number is the mean video AUC over its seeds. A bootstrap sample resamples the *videos*
of a test set (with replacement, stratified by label); in each sample every seed's AUC is recomputed and averaged,
so the interval covers the seed-mean number shown in the tables. Paired comparisons use the same resampled videos
for both models, giving a CI for the difference and a two-sided bootstrap p-value. Cross-dataset = mean of the DFD
and Celeb-DF AUCs, with the two sets resampled independently. Uses only saved predictions_*.csv files (no GPU).
Writes results/paper_stats.md and results/paper_stats.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402

MODELS = {"b1_swin": "B1", "c1_clip_lora": "C1", "c2_clip_sbi": "C2", "g1_concepts": "G1", "g2_dct": "G2",
          "f1_grace_dct": "F1", "s0_student": "S0", "s3_kd_full": "S3", "r1_univfd": "R1", "r2_sbi_effb4": "R2",
          "f1h_grace_dct_h264": "F1h", "s3h_kd_full_h264": "S3h", "g3_full": "G3", "a3_linear_head": "A3"}
PAIRS = [("b1_swin", "c1_clip_lora"), ("c1_clip_lora", "c2_clip_sbi"), ("c2_clip_sbi", "g1_concepts"),
         ("g1_concepts", "g2_dct"), ("g2_dct", "f1_grace_dct"), ("c1_clip_lora", "f1_grace_dct"),
         ("b1_swin", "f1_grace_dct"), ("s0_student", "s3_kd_full"), ("b1_swin", "s3_kd_full"),
         ("r1_univfd", "f1_grace_dct"), ("r2_sbi_effb4", "f1_grace_dct"), ("r1_univfd", "c1_clip_lora"),
         ("f1_grace_dct", "f1h_grace_dct_h264"), ("s3_kd_full", "s3h_kd_full_h264"), ("a3_linear_head", "g3_full")]
SPLITS = ["dfd_test", "celebdf_test"]
DEGRADES = ["", "_jpeg30", "_h26440"]


def video_table(run: Path, tag: str) -> pd.Series | None:
    f = run / f"predictions_{tag}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, usecols=["video_id", "label", "prob"])
    g = d.groupby("video_id").agg(label=("label", "first"), prob=("prob", "mean"))
    return g


def load(model: str, tag: str, seeds) -> tuple[np.ndarray, np.ndarray] | None:
    """-> labels (n,), probs (n_seeds, n) on the videos common to all seeds."""
    tabs = [t for t in (video_table(r, tag) for r in resolve_runs([model], seeds)) if t is not None]
    if not tabs:
        return None
    ids = sorted(set.intersection(*(set(t.index) for t in tabs)))
    return tabs[0].loc[ids, "label"].to_numpy(), np.stack([t.loc[ids, "prob"].to_numpy() for t in tabs]), ids


def boot_idx(y: np.ndarray, n: int, rng) -> np.ndarray:
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.stack([np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n)])


def seed_mean_auc(y, P, idx=None):
    if idx is None:
        return float(np.mean([roc_auc_score(y, p) for p in P]))
    return float(np.mean([roc_auc_score(y[idx], p[idx]) for p in P]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows, pair_rows = [], []
    for deg in DEGRADES:
        data = {}
        for m in MODELS:
            per = {}
            for s in SPLITS:
                r = load(m, s + deg, args.seeds)
                if r is not None:
                    per[s] = r
            if len(per) == len(SPLITS):
                data[m] = per
        # one set of resampled indices per split, shared by every model (paired), on each model's common videos
        boots = {}
        for s in SPLITS:
            ids = sorted(set.intersection(*(set(data[m][s][2]) for m in data)))
            for m in data:
                y, P, mids = data[m][s]
                keep = np.isin(np.array(mids), ids)
                data[m][s] = (y[keep], P[:, keep], ids)
            y0 = data[next(iter(data))][s][0]
            boots[s] = boot_idx(y0, args.boot, rng)
        dist = {}
        for m in data:
            d = {}
            for s in SPLITS:
                y, P, _ = data[m][s]
                d[s] = np.array([seed_mean_auc(y, P, i) for i in boots[s]])
                d[s + "_point"] = seed_mean_auc(y, P)
            d["cross"] = (d["dfd_test"] + d["celebdf_test"]) / 2
            d["cross_point"] = (d["dfd_test_point"] + d["celebdf_test_point"]) / 2
            dist[m] = d
            for s in SPLITS + ["cross"]:
                lo, hi = np.percentile(d[s], [2.5, 97.5])
                rows.append(dict(condition=deg.strip("_") or "clean", model=MODELS[m], split=s.replace("_test", ""),
                                 n_seeds=data[m][SPLITS[0]][1].shape[0], auc=d[s + "_point"], ci_low=lo, ci_high=hi))
        for a, b in PAIRS:
            if a not in dist or b not in dist:
                continue
            for s in SPLITS + ["cross"]:
                diff = dist[b][s] - dist[a][s]
                point = dist[b][s + "_point"] - dist[a][s + "_point"]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
                pair_rows.append(dict(condition=deg.strip("_") or "clean", comparison=f"{MODELS[b]} - {MODELS[a]}",
                                      split=s.replace("_test", ""), diff=point, ci_low=lo, ci_high=hi, p_boot=p))
    out = REPO_ROOT / "results"
    df, pf = pd.DataFrame(rows), pd.DataFrame(pair_rows)
    df.to_csv(out / "paper_stats.csv", index=False)
    pf.to_csv(out / "paper_stats_pairs.csv", index=False)
    lines = ["# Paper statistics (generated by scripts/paper_stats.py)", "",
             f"Video AUC, seed mean; 95% bootstrap CI over videos ({args.boot} resamples, stratified by label). "
             "Paired differences use the same resampled videos for both models.", "", "## Per model", "",
             "| Condition | Model | Seeds | Split | AUC | 95% CI |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['model']} | {r['n_seeds']} | {r['split']} | {r['auc']:.3f} | "
                     f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] |")
    lines += ["", "## Paired differences (later model minus earlier)", "",
              "| Condition | Comparison | Split | Δ AUC | 95% CI | p (bootstrap) |", "| --- | --- | --- | --- | --- | --- |"]
    for r in pair_rows:
        lines.append(f"| {r['condition']} | {r['comparison']} | {r['split']} | {r['diff']:+.3f} | "
                     f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['p_boot']:.3f} |")
    (out / "paper_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
