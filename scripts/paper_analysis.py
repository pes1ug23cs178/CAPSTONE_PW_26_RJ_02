"""Extra analyses for the paper (CPU only; reads saved predictions) -> results/paper_analysis.md (+ csv files).

1. Calibration: expected calibration error (ECE, 15 equal-width bins) of the fake probability, before and after the
   temperature fitted on FF++ val, frame level and video level, per model and test set (seed 42 runs).
2. Concepts on real forgeries: mean concept score (sigmoid) of the final teacher per manipulation method on the three
   test sets. Concepts were trained only on self-blended images, so this checks whether they carry over sensibly.
3. Failure analysis on Celeb-DF (final teacher, mean over its seeds): AUC per real subgroup (Celeb-real vs fakes,
   YouTube-real vs fakes), AUC by image-sharpness and brightness tertile (computed from the face crops), per-identity
   error rates, and the 15 most confidently wrong videos.

python scripts/paper_analysis.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.common import final_teacher_name  # noqa: E402
from utils.config import REPO_ROOT, load_paths  # noqa: E402

SPLITS = ["ffpp_test", "dfd_test", "celebdf_test"]
CONCEPTS = ["blend strength", "colour mismatch", "blur", "sharpen", "resampling", "mouth", "eyes", "full face",
            "geometry", "irregular boundary"]


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(total)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calibration(models):
    rows = []
    for name in models:
        found = resolve_runs([name], [42])
        if not found:
            continue
        run = found[0]
        t_file = run / "temperature.json"
        T = json.loads(t_file.read_text())["temperature"] if t_file.exists() else 1.0
        for s in SPLITS:
            f = run / f"predictions_{s}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, usecols=["video_id", "label", "logit"])
            y = d["label"].to_numpy()
            for level in ("frame", "video"):
                if level == "frame":
                    yy, raw, cal = y, sigmoid(d["logit"].to_numpy()), sigmoid(d["logit"].to_numpy() / T)
                else:
                    g = d.assign(raw=sigmoid(d["logit"]), cal=sigmoid(d["logit"] / T)).groupby("video_id").agg(
                        label=("label", "first"), raw=("raw", "mean"), cal=("cal", "mean"))
                    yy, raw, cal = g["label"].to_numpy(), g["raw"].to_numpy(), g["cal"].to_numpy()
                rows.append(dict(model=name, split=s.replace("_test", ""), level=level, T=round(T, 3),
                                 ece_raw=ece(yy, raw), ece_temp=ece(yy, cal)))
    return pd.DataFrame(rows)


def concepts_by_method(run):
    parts = []
    for s in SPLITS:
        d = pd.read_csv(run / f"predictions_{s}.csv")
        d["set"] = s.replace("_test", "")
        parts.append(d)
    d = pd.concat(parts)
    d["group"] = d["set"] + ": " + d["method"].astype(str)
    tab = d.groupby("group")[[f"c{k}" for k in range(1, 11)]].mean()
    tab.columns = CONCEPTS
    tab.insert(0, "frames", d.groupby("group").size())
    return tab


def crop_stats(paths, root):
    sharp, bright = [], []
    for p in paths:
        img = cv2.imread(str(root / p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            sharp.append(np.nan); bright.append(np.nan); continue
        sharp.append(float(cv2.Laplacian(img, cv2.CV_64F).var()))
        bright.append(float(img.mean()))
    return sharp, bright


def failure(final, root):
    runs = resolve_runs([final], [42, 1, 2])
    vids = []
    for r in runs:
        d = pd.read_csv(r / "predictions_celebdf_test.csv", usecols=["video_id", "frame_path", "label", "method", "prob"])
        vids.append(d.groupby("video_id").agg(label=("label", "first"), method=("method", "first"), prob=("prob", "mean")))
    v = vids[0][["label", "method"]].copy()
    v["prob"] = np.mean([x["prob"].reindex(v.index).to_numpy() for x in vids], axis=0)
    frames = pd.read_csv(runs[0] / "predictions_celebdf_test.csv", usecols=["video_id", "frame_path"])
    sample = frames.groupby("video_id").head(8)
    s, b = crop_stats(sample["frame_path"].tolist(), root)
    sample = sample.assign(sharpness=s, brightness=b).groupby("video_id")[["sharpness", "brightness"]].mean()
    v = v.join(sample)
    thr = json.loads((runs[0] / "val_threshold.json").read_text())["threshold"]
    out = {"n_seeds": len(runs), "threshold": thr}
    fakes = v[v.label == 1]
    sub = []
    for grp in ("Celeb-real", "YouTube-real"):
        reals = v[(v.label == 0) & (v.method == grp)]
        both = pd.concat([reals, fakes])
        sub.append(dict(subgroup=f"{grp} vs Celeb-synthesis", n_real=len(reals), n_fake=len(fakes),
                        auc=roc_auc_score(both.label, both.prob)))
    for col in ("sharpness", "brightness"):
        v[f"{col}_tertile"] = pd.qcut(v[col], 3, labels=["low", "mid", "high"])
        for t, g in v.groupby(f"{col}_tertile", observed=True):
            if g.label.nunique() == 2:
                sub.append(dict(subgroup=f"{col} {t}", n_real=int((g.label == 0).sum()), n_fake=int((g.label == 1).sum()),
                                auc=roc_auc_score(g.label, g.prob)))
    v["wrong"] = (v.prob >= thr).astype(int) != v.label
    v["identity"] = [m.group(0) if (m := re.search(r"id\d+", str(i))) else "youtube" for i in v.index]
    ident = v[v.method != "YouTube-real"].groupby("identity").agg(videos=("wrong", "size"), errors=("wrong", "sum"))
    ident["error_rate"] = ident.errors / ident.videos
    v["confidence_wrong"] = np.where(v.label == 1, 1 - v.prob, v.prob)
    worst = v[v.wrong].sort_values("confidence_wrong", ascending=False).head(15)
    return out, pd.DataFrame(sub), ident.sort_values("error_rate", ascending=False), worst, v


def main():
    root = load_paths()["data_root"]
    final = final_teacher_name()
    res = REPO_ROOT / "results"
    cal = calibration(["b1_swin", "c1_clip_lora", "c2_clip_sbi", "g1_concepts", "g2_dct", final, "s3_kd_full", "f1h_grace_dct_h264",
                       "r1_univfd", "r2_sbi_effb4"])
    cal.to_csv(res / "paper_calibration.csv", index=False)
    ctab = concepts_by_method(resolve_runs([final], [42])[0])
    ctab.to_csv(res / "paper_concepts_by_method.csv")
    info, sub, ident, worst, v = failure(final, root)
    v.to_csv(res / "paper_celebdf_videos.csv")
    L = ["# Paper analyses (generated by scripts/paper_analysis.py)", "",
         "## 1. Calibration (ECE, 15 bins; lower is better)", "",
         "| Model | Set | Level | T | ECE raw | ECE after temperature |", "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {r.model} | {r.split} | {r.level} | {r.T} | {r.ece_raw:.3f} | {r.ece_temp:.3f} |" for r in cal.itertuples()]
    L += ["", f"## 2. Mean concept score of {final} per manipulation method (sigmoid, frame level)", "",
          "| Group | Frames | " + " | ".join(CONCEPTS) + " |", "| --- " * (len(CONCEPTS) + 2) + "|"]
    for g, r in ctab.iterrows():
        L.append(f"| {g} | {int(r.frames)} | " + " | ".join(f"{r[c]:.2f}" for c in CONCEPTS) + " |")
    L += ["", f"## 3. Celeb-DF failure analysis ({final}, mean over {info['n_seeds']} seeds; threshold {info['threshold']:.3f} from FF++ val)", "",
          "| Subgroup | Real videos | Fake videos | Video AUC |", "| --- | --- | --- | --- |"]
    L += [f"| {r.subgroup} | {r.n_real} | {r.n_fake} | {r.auc:.3f} |" for r in sub.itertuples()]
    L += ["", "Identities with the highest error rate (Celeb-real + Celeb-synthesis videos of that person):", "",
          "| Identity | Videos | Errors | Error rate |", "| --- | --- | --- | --- |"]
    L += [f"| {i} | {r.videos} | {int(r.errors)} | {r.error_rate:.2f} |" for i, r in ident.head(10).iterrows()]
    L += ["", "Most confidently wrong videos:", "", "| Video | Truth | Group | Fake probability | Sharpness | Brightness |",
          "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {i} | {'fake' if r.label else 'real'} | {r.method} | {r.prob:.3f} | {r.sharpness:.0f} | {r.brightness:.0f} |"
          for i, r in worst.iterrows()]
    (res / "paper_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
