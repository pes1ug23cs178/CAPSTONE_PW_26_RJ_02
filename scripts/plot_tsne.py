"""T4.4: t-SNE of z (backbone feature for Swin) on 3,000 test crops: 1,000 each from FF++, DFD and
Celeb-DF, balanced real/fake. Coloured by dataset, marker by label. One panel per model.

python scripts/plot_tsne.py --models g2_dct g3_full [--extra b1_swin]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import FaceFrameDataset  # noqa: E402
from evaluate import load_model_from_ckpt  # noqa: E402
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


def sample_csv(out: Path, per_set: int = 1000, seed: int = 0) -> Path:
    parts = []
    for split in ("ffpp_test", "dfd_test", "celebdf_test"):
        df = pd.read_csv(REPO_ROOT / "splits" / f"{split}.csv", dtype=str)
        for lab in ("0", "1"):
            d = df[df["label"] == lab]
            parts.append(d.sample(n=min(per_set // 2, len(d)), random_state=seed))
    s = pd.concat(parts)
    s.to_csv(out, index=False)
    return out


@torch.no_grad()
def features(run: Path, csv: Path, device) -> np.ndarray:
    model, _ = load_model_from_ckpt(str(run / "best.pth"), device)
    dl = DataLoader(FaceFrameDataset(csv, train=False), batch_size=128, num_workers=6)
    zs = []
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zs.append(model(b["image"].to(device))["z"].float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(zs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["g2_dct", "g3_full"])
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    csv = sample_csv(REPO_ROOT / "cache" / "tsne_sample.csv")
    meta = pd.read_csv(csv, dtype=str)
    runs = resolve_runs(args.models, [42])
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5.5), squeeze=False)
    colors = {"ffpp": "tab:blue", "dfd": "tab:orange", "celebdf": "tab:green"}
    device = torch.device("cuda")
    for ax, run in zip(axes[0], runs):
        z = features(run, csv, device)
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(z)
        for ds, c in colors.items():
            for lab, marker in (("0", "o"), ("1", "x")):
                m = (meta["dataset"] == ds).to_numpy() & (meta["label"] == lab).to_numpy()
                ax.scatter(emb[m, 0], emb[m, 1], s=6, c=c, marker=marker, alpha=0.6,
                           label=f"{ds} {'real' if lab == '0' else 'fake'}")
        ax.set_title(run.name.rsplit("_s", 1)[0])
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][-1].legend(fontsize=7, markerscale=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_tsne.png", dpi=200)
    print("wrote results/fig_tsne.png")


if __name__ == "__main__":
    main()
