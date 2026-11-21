"""Restoration-deletion test for explanation maps -> results/explain_restore.csv.

The blur-based deletion test in explain_eval.py can add evidence instead of removing it: a blurred patch looks like
"blur in a region", one of the forgery concepts. Here we use the held-out self-blended test set (cache/sbi_test),
where the untouched real frame is known (`source_frame`). Patches are *restored* to the real pixels in order of
map importance (14x14 patches, 20 steps), so each step truly removes forgery evidence. A faithful map makes the fake
probability fall fast: lower AUC of the probability curve is better. Floors and ceilings: `random` order, and
`oracle` (patches ordered by the true blend mask).

Evaluated on the SBI test images the final teacher flags as fake (its val threshold), at most 500.
python scripts/explain_restore.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_rgb  # noqa: E402
from evaluate import load_model_from_ckpt  # noqa: E402
from scripts.eval_all import resolve_runs  # noqa: E402
from scripts.explain_eval import METHOD_MODEL, PATCH, REGION_CONCEPT, STEPS, fake_prob, method_map  # noqa: E402
from utils.config import REPO_ROOT, load_paths  # noqa: E402


def restore_curve(model, img, real, m, device):
    g = 224 // PATCH
    imp = m[: g * PATCH, : g * PATCH].reshape(g, PATCH, g, PATCH).mean(axis=(1, 3)).ravel()
    order = np.argsort(-imp, kind="stable")
    per_step = int(np.ceil(len(order) / STEPS))
    cur, seq = img.copy(), [img.copy()]
    for s in range(STEPS):
        for p in order[s * per_step:(s + 1) * per_step]:
            r, c = divmod(int(p), g)
            sl = (slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH))
            cur[sl] = real[sl]
        seq.append(cur.copy())
    p = fake_prob(model, seq, device)
    return float(np.trapezoid(p, dx=1 / STEPS)), float(p[0]), float(p[STEPS // 4])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="results/explain_restore.csv")
    args = ap.parse_args()
    device = torch.device("cuda")
    root = load_paths()["data_root"]
    names = {"g3": "final", "b1": "b1_swin", "c1": "c1_clip_lora", "s3": "s3_kd_full"}
    runs = {k: resolve_runs([v], [42])[0] for k, v in names.items()}
    models = {k: load_model_from_ckpt(str(r / "best.pth"), device)[0] for k, r in runs.items()}
    thr = json.loads((runs["g3"] / "val_threshold.json").read_text())["threshold"]
    sbi = pd.read_csv(REPO_ROOT / "cache" / "sbi_test" / "sbi_test.csv")
    keep = []
    for r in sbi.itertuples():
        if fake_prob(models["g3"], [load_rgb(root / r.frame_path)], device)[0] >= thr:
            keep.append(r)
        if len(keep) >= args.limit:
            break
    print(f"{len(keep)} flagged SBI test images", flush=True)
    rows = []
    for meth in ["G3 c1", "G3 region", "B1 Grad-CAM", "C1 rollout", "S3 c1", "random", "oracle"]:
        model = models[METHOD_MODEL.get(meth, "g3")]
        aucs, p0s, p25s = [], [], []
        for r in keep:
            if meth == "G3 region" and r.region not in REGION_CONCEPT:
                continue
            img, real = load_rgb(root / r.frame_path), load_rgb(root / r.source_frame)
            if meth == "oracle":
                m = np.load(root / r.mask_path).astype(np.float32)
            else:
                m = method_map(meth, models, img, device, r.region)
            a, p0, p25 = restore_curve(model, img, real, m, device)
            aucs.append(a); p0s.append(p0); p25s.append(p25)
        rows.append({"method": meth, "model": METHOD_MODEL.get(meth, "g3"), "restore_auc": float(np.mean(aucs)),
                     "prob_start": float(np.mean(p0s)), "prob_after_25pct": float(np.mean(p25s)), "n": len(aucs)})
        print(rows[-1], flush=True)
    from utils.common import final_teacher_name
    tlabel = final_teacher_name().split("_")[0].upper()
    df = pd.DataFrame(rows)
    df["method"] = df["method"].str.replace(r"^G3", tlabel, regex=True)
    out = REPO_ROOT / args.out
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
