"""T2.4 check: draw 64 SBI examples (image + mask outline + recipe) to cache/sbi_check.png,
and write cache/landmarks/mean_landmarks.npy (fallback for crops whose landmarks failed)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_rgb  # noqa: E402
from data.sbi import make_sbi  # noqa: E402
from utils.config import REPO_ROOT, load_paths  # noqa: E402


def main() -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = load_paths()
    cache, data_root = paths["cache_root"], paths["data_root"]
    files = sorted((cache / "landmarks").glob("*/*.npy"))
    files = [f for f in files if f.name != "mean_landmarks.npy"]
    if not files:
        sys.exit("No landmarks cached yet")
    sample = random.Random(0).sample(files, min(2000, len(files)))
    np.save(cache / "landmarks" / "mean_landmarks.npy", np.mean([np.load(f) for f in sample], axis=0).astype(np.float32))

    split = sys.argv[1] if len(sys.argv) > 1 else "ffpp_train"  # fullff_train for the full-FF++ crops
    df = pd.read_csv(REPO_ROOT / "splits" / f"{split}.csv", dtype=str)
    df = df[df["method"] == "real"]
    lookup = {(r.video_id, Path(r.frame_path).stem): r.frame_path for r in df.itertuples()}
    pick = random.Random(1).sample(files, min(64, len(files)))
    pick = (pick * (64 // len(pick) + 1))[:64]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(8, 8, figsize=(20, 22))
    for ax, f in zip(axes.flat, pick):
        fp = lookup.get((f.parent.name, f.stem))
        if fp is None:
            ax.axis("off")
            continue
        img = load_rgb(data_root / fp)
        out, concepts, boundary, recipe, mask = make_sbi(img, np.load(f), rng)
        shown = out.copy()
        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(shown, contours, -1, (0, 255, 0), 1)
        ax.imshow(shown)
        tags = [k for k in ("colour", "blurred", "sharpened", "resampled", "geometry", "elastic") if recipe.get(k)]
        ax.set_title(f"{recipe['region']} a={recipe['blend_ratio']} k={recipe['blur_kernel']}\n{','.join(tags)}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    out_path = cache / "sbi_check.png"
    fig.savefig(out_path, dpi=80)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
