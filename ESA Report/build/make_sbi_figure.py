"""Report figure: real face -> self-blended fake -> blend mask, with the logged recipe (uses the project's data engine)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from data.dataset import load_rgb  # noqa: E402
from data.sbi import SBIMaker, make_sbi  # noqa: E402
from utils.config import load_paths  # noqa: E402

root = load_paths()["data_root"]
df = pd.read_csv(REPO / "splits" / "ffpp_train.csv", dtype=str)
real = df[df.method == "real"].drop_duplicates("video_id")
maker = SBIMaker({})
picks, rng = [], np.random.default_rng(3)
wanted = ["full_face", "mouth", "eyes_brows", "nose_cheeks", "full_face"]
for _, r in real.sample(frac=1, random_state=11).iterrows():
    lm_path = maker.cache_root / "landmarks" / r.video_id / f"{Path(r.frame_path).stem}.npy"
    if not lm_path.exists():
        continue
    img, lm = load_rgb(root / r.frame_path), np.load(lm_path)
    for _ in range(40):  # draw until the recipe uses the region we want for this column
        out, c, b, rec, mask = make_sbi(img, lm, rng)
        if rec["region"] == wanted[len(picks)]:
            picks.append((img, out, mask, rec))
            break
    if len(picks) == len(wanted):
        break

fig, axes = plt.subplots(3, len(picks), figsize=(2.3 * len(picks), 7.4))
for j, (img, out, mask, rec) in enumerate(picks):
    tf = [n for n, k in (("colour", "colour"), ("blur", "blurred"), ("sharpen", "sharpened"), ("resample", "resampled"),
                         ("shift", "geometry")) if rec.get(k)]
    for i, (im, cmap) in enumerate(((img, None), (out, None), (mask, "magma"))):
        ax = axes[i][j]
        ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][j].set_title(f"region: {rec['region'].replace('_', ' ')}", fontsize=9)
    axes[1][j].set_title(f"a = {rec['blend_ratio']}; {', '.join(tf) or 'none'}" + ("; warped" if rec.get("elastic") else ""),
                         fontsize=8)
for i, lab in enumerate(("real face", "self-blended fake", "blend mask m")):
    axes[i][0].set_ylabel(lab, fontsize=10)
fig.tight_layout()
out_path = Path(__file__).resolve().parent / "figs" / "sbi_examples.png"
fig.savefig(out_path, dpi=200, facecolor="white")
print("wrote", out_path, len(picks))
