"""T6.2: qualitative explanation grid -> results/fig_explanations.png (300 dpi).

16 test crops by fixed seed from G3's test predictions: 8 correctly flagged fakes, 4 errors (misses or
false alarms), 4 correct reals, mixed over FF++, DFD and Celeb-DF.
Columns: input | G3 top-concept map | G3 concept bars | B1 Grad-CAM | C1 attention rollout | S3 top-concept map (jpeg:30)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import FaceFrameDataset, load_rgb  # noqa: E402
from data.degrade import ImageDegrade  # noqa: E402
from data.sbi import CONCEPT_NAMES  # noqa: E402
from evaluate import load_model_from_ckpt  # noqa: E402
from explain.maps import attention_rollout, concept_maps, gradcam  # noqa: E402
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT, load_paths  # noqa: E402


def pick_rows(g3: Path, seed: int = 7) -> pd.DataFrame:
    thr = json.loads((g3 / "val_threshold.json").read_text())["threshold"]
    parts = []
    for split in ("ffpp_test", "dfd_test", "celebdf_test"):
        p = pd.read_csv(g3 / f"predictions_{split}.csv")
        p["split"] = split
        parts.append(p)
    p = pd.concat(parts)
    p["pred"] = (p["prob"] >= thr).astype(int)
    tp = p[(p.label == 1) & (p.pred == 1)]
    err = p[p.label != p.pred]
    tn = p[(p.label == 0) & (p.pred == 0)]
    take = lambda d, n: pd.concat([g.sample(n=min(len(g), max(1, n // 3)), random_state=seed) for _, g in d.groupby("split")]).head(n)  # noqa: E731
    return pd.concat([take(tp, 8), take(err, 4), take(tn, 4)]).reset_index(drop=True)


def overlay(img, m, cmap="jet"):
    import matplotlib.cm as cm
    heat = (cm.get_cmap(cmap)(m)[..., :3] * 255).astype(np.uint8)
    return (0.55 * img + 0.45 * heat).astype(np.uint8)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    device = torch.device("cuda")
    runs = {n: resolve_runs([n], [42])[0] for n in ("g3_full", "b1_swin", "c1_clip_lora", "s3_kd_full")}
    rows = pick_rows(runs["g3_full"])
    models = {n: load_model_from_ckpt(str(r / "best.pth"), device)[0] for n, r in runs.items()}
    data_root = load_paths()["data_root"]
    jpeg30 = ImageDegrade("jpeg", 30)
    fig, axes = plt.subplots(len(rows), 6, figsize=(18, 3 * len(rows)))
    titles = ["input", "G3 top concept", "G3 concepts", "B1 Grad-CAM", "C1 rollout", "S3 top concept (jpeg:30)"]
    for i, r in rows.iterrows():
        img = load_rgb(data_root / r.frame_path)
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)[None].to(device)
        cm_g3 = concept_maps(models["g3_full"], x)
        with torch.no_grad():
            o = models["g3_full"](x)
        scores = torch.sigmoid(o["concepts"][0].float()).cpu().numpy()
        top = int(np.argmax(scores))
        xs = torch.from_numpy(jpeg30(img)).permute(2, 0, 1).float().div(255)[None].to(device)
        cm_s3 = concept_maps(models["s3_kd_full"], xs)
        with torch.no_grad():
            top_s = int(torch.sigmoid(models["s3_kd_full"](xs)["concepts"][0].float()).argmax())
        cells = [img, overlay(img, cm_g3[top]), None, overlay(img, gradcam(models["b1_swin"], x)),
                 overlay(img, attention_rollout(models["c1_clip_lora"], x)), overlay(jpeg30(img), cm_s3[top_s])]
        for j, ax in enumerate(axes[i]):
            if j == 2:
                ax.barh(range(10), scores, color=["tab:red" if k == top else "tab:gray" for k in range(10)])
                ax.set_yticks(range(10)); ax.set_yticklabels(CONCEPT_NAMES, fontsize=6); ax.set_xlim(0, 1)
                ax.invert_yaxis()
            else:
                ax.imshow(cells[j]); ax.axis("off")
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
        axes[i][0].set_title(f"{r.split} | {'fake' if r.label else 'real'} | p={r.prob:.2f}", fontsize=7)
        axes[i][1].set_title(f"{CONCEPT_NAMES[top]}" + ("\nG3 top concept" if i == 0 else ""), fontsize=7)
    fig.tight_layout()
    out = REPO_ROOT / "results" / "fig_explanations.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
