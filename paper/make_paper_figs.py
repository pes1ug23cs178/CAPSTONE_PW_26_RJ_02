"""Figures for the paper -> paper/figs/. Reads results/robustness.csv only (no GPU, no dataset access).

python paper/make_paper_figs.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figs"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})

# Okabe-Ito colour-blind-safe palette
MODELS = [("b1_swin", "B1 Swin-B", "#999999", "o"), ("r1_univfd", "R1 frozen CLIP", "#E69F00", "v"),
          ("c1_clip_lora", "C1 CLIP+LoRA", "#56B4E9", "s"), ("f1_grace_dct", "F1 GRACE", "#0072B2", "D"),
          ("f1h_grace_dct_h264", "F1h GRACE+H.264", "#D55E00", "^"), ("s3_kd_full", "S3 student", "#009E73", "P")]


def robustness():
    d = pd.read_csv(ROOT / "results" / "robustness.csv")
    d = d[d.split.isin(["dfd_test", "celebdf_test"])]
    per_seed = d.groupby(["model", "seed", "degrade"]).video_auc.mean().reset_index()  # cross = mean of the two sets
    g = per_seed.groupby(["model", "degrade"]).video_auc.agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05), sharey=True)
    for ax, (kind, xs, lab, rev) in zip(axes, [("jpeg", [90, 70, 50, 30], "JPEG quality", True),
                                                ("h264", [28, 34, 40], "H.264 CRF", False)]):
        for m, name, col, mk in MODELS:
            pts = [("clean", 0)] + [(f"{kind}:{x}", i + 1) for i, x in enumerate(xs)]
            rows = [(x, g[(g.model == m) & (g.degrade == k)]) for k, x in pts]
            rows = [(x, r.iloc[0]) for x, r in rows if len(r)]
            if not rows:
                continue
            x = [a for a, _ in rows]
            mu = [r["mean"] for _, r in rows]
            sd = [0 if pd.isna(r["std"]) else r["std"] for _, r in rows]
            ax.plot(x, mu, color=col, marker=mk, ms=2.8, lw=1.0, label=name)
            ax.fill_between(x, [a - b for a, b in zip(mu, sd)], [a + b for a, b in zip(mu, sd)], color=col, alpha=0.15, lw=0)
        ax.set_xlabel(lab)
        ax.set_xticks(range(len(xs) + 1), ["clean"] + [str(x) for x in xs])
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("Cross-dataset video AUC")
    axes[0].set_ylim(0.5, 0.95)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.80))
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, 1.0),
               columnspacing=0.8, handlelength=1.6)
    fig.savefig(OUT / "fig_robustness.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_robustness.png", dpi=300, bbox_inches="tight")


def box(ax, x, y, w, h, title, sub, fc="#FFFFFF", ec="#555555"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=6.3, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=4.7, color="#333333")


def arrow(ax, a, b, text=None, ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#444444", ls=ls,
                                                     shrinkA=0, shrinkB=0, mutation_scale=6))
    if text:
        ax.text((a[0] + b[0]) / 2 + 0.01, (a[1] + b[1]) / 2, text, fontsize=5.5, color="#444444", va="center")


def architecture():
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.25, 0.89, 0.50, 0.09, "Face crop 224×224", "real / SBI (recipe → labels) / FF++ fake")
    box(ax, 0.02, 0.68, 0.52, 0.12, "CLIP ViT-L/14 + LoRA", "frozen, LoRA r=16; CLS + 16×16 patches")
    box(ax, 0.62, 0.68, 0.36, 0.12, "Block-DCT branch", "luma 8×8 DCT, CNN, 256-d")
    box(ax, 0.02, 0.44, 0.52, 0.13, "Concept head", "1×1 conv: 10 maps, top-10% mean", fc="#E8F1FA", ec="#0072B2")
    box(ax, 0.62, 0.44, 0.36, 0.13, "Gated fusion", r"$z=\mathrm{LN}(g\,p_c+(1-g)\,p_f)$")
    box(ax, 0.02, 0.20, 0.96, 0.13, "Concept-bottleneck decision",
        r"logit $= \Sigma_k\, w_k\,\sigma(c_k) + b + r(z)$, small residual $r$ (L2 penalty)", fc="#E8F1FA", ec="#0072B2")
    box(ax, 0.02, 0.01, 0.44, 0.11, "Explanation", "concepts, maps, contributions")
    box(ax, 0.54, 0.01, 0.44, 0.11, "EfficientNet-B0 student", "KD of logit, z, concepts, maps")
    arrow(ax, (0.40, 0.89), (0.28, 0.80)); arrow(ax, (0.60, 0.89), (0.80, 0.80))
    arrow(ax, (0.20, 0.68), (0.20, 0.57), "patches")
    arrow(ax, (0.50, 0.68), (0.66, 0.57), "CLS"); arrow(ax, (0.80, 0.68), (0.80, 0.57))
    arrow(ax, (0.20, 0.44), (0.20, 0.33), "$c$"); arrow(ax, (0.80, 0.44), (0.80, 0.33), "$z$")
    arrow(ax, (0.24, 0.20), (0.24, 0.12)); arrow(ax, (0.76, 0.20), (0.76, 0.12), "KD", ls="--")
    fig.savefig(OUT / "fig_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_architecture.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    robustness()
    architecture()
    shutil.copy(ROOT / "ESA Report" / "build" / "figs" / "fig_explanations_top.png", OUT / "fig_explanations.png")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
