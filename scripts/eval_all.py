"""T4.3 / T7.1: evaluate many models on many splits and degradations.

python scripts/eval_all.py --models b1_swin c1_clip_lora ... [--seeds 42] \
    --splits ffpp_test dfd_test celebdf_test \
    --degrades clean jpeg:90 jpeg:70 jpeg:50 jpeg:30 down:0.5 down:0.25 h264:28 h264:34 h264:40 [--final]

--models takes config names (latest run folder runs/<name>_s<seed>_* is used) or run folder paths.
Existing metrics files are reused unless --final. Writes results/robustness.csv and
results/fig_robustness_<split>.png (video AUC vs degradation level, one line per model).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate_split, load_model_from_ckpt  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402

DEFAULT_DEGRADES = ["clean", "jpeg:90", "jpeg:70", "jpeg:50", "jpeg:30", "down:0.5", "down:0.25", "h264:28", "h264:34", "h264:40"]


def resolve_runs(names: List[str], seeds: List[int]) -> List[Path]:
    runs = []
    for name in names:
        p = Path(name)
        if p.is_dir():
            runs.append(p)
            continue
        for seed in seeds:
            cands = sorted((REPO_ROOT / "runs").glob(f"{name}_s{seed}_*"))
            cands = [c for c in cands if (c / "best.pth").exists() and (c / "metrics_ffpp_test.json").exists()]
            if cands:
                runs.append(cands[-1])
            else:
                print(f"WARNING: no finished run for {name} seed {seed}", flush=True)
    return runs


def tag_for(split: str, degrade: str) -> str:
    return split + ("" if degrade == "clean" else f"_{degrade.replace(':', '')}")


def plot(df: pd.DataFrame, split: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df["split"] == split]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    families = [("jpeg", "JPEG quality", [100, 90, 70, 50, 30]), ("down", "Downscale factor", [1.0, 0.5, 0.25]),
                ("h264", "H.264 CRF", [0, 28, 34, 40])]
    for ax, (fam, xlabel, levels) in zip(axes, families):
        for model, g in d.groupby("model"):
            clean = g[g["degrade"] == "clean"]["video_auc"]
            xs, ys = [], []
            for lv in levels:
                if lv in (100, 1.0, 0):
                    if len(clean):
                        xs.append(str(lv) if fam != "h264" else "clean"); ys.append(float(clean.iloc[0]))
                    continue
                spec = f"{fam}:{lv:g}"
                row = g[g["degrade"] == spec]
                if len(row):
                    xs.append(str(lv)); ys.append(float(row["video_auc"].iloc[0]))
            if ys:
                ax.plot(xs, ys, marker="o", label=model)
        ax.set(title=xlabel, xlabel=xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"Video AUC ({split})")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--splits", nargs="+", default=["ffpp_test", "dfd_test", "celebdf_test"])
    ap.add_argument("--degrades", nargs="+", default=DEFAULT_DEGRADES)
    ap.add_argument("--final", action="store_true", help="recompute every metrics file")
    ap.add_argument("--out", default="results/robustness.csv")
    args = ap.parse_args()
    device = torch.device("cuda")
    rows = []
    for run in resolve_runs(args.models, args.seeds):
        model, ckpt = None, None
        thr_file = run / "val_threshold.json"
        threshold = json.loads(thr_file.read_text())["threshold"] if thr_file.exists() else None
        name = ckpt_name = run.name.rsplit("_s", 1)[0]
        seed = int(run.name.rsplit("_s", 1)[1].split("_")[0])
        for split in args.splits:
            for degrade in args.degrades:
                mfile = run / f"metrics_{tag_for(split, degrade)}.json"
                if mfile.exists() and not args.final:
                    m = json.loads(mfile.read_text())
                else:
                    if model is None:
                        model, ckpt = load_model_from_ckpt(str(run / "best.pth"), device)
                    m = evaluate_split(model, REPO_ROOT / "splits" / f"{split}.csv", run, run.name, device, threshold,
                                       degrade=None if degrade == "clean" else degrade)
                    print(f"{run.name} {split} {degrade}: video AUC {m['video_auc']:.4f}", flush=True)
                rows.append({"model": name, "seed": seed, "run": run.name, "split": split, "degrade": degrade,
                             "video_auc": m["video_auc"], "frame_auc": m["frame_auc"], "eer": m["eer"],
                             "acc": m.get("acc"), "f1": m.get("f1"), "concept_share": m.get("concept_share"),
                             "gate_mean": m.get("gate_mean")})
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    agg = df.groupby(["model", "split", "degrade"])["video_auc"].mean().reset_index()
    for split in args.splits:
        plot(agg, split, out.parent / f"fig_robustness_{split}.png")
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
