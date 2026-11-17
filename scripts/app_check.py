"""T6.4 acceptance check (pipeline level): run the app's analysis on cache/app_test_videos in Quick and Deep
modes, time each, check the no-face photo gives a clear message, and write results/app_check.md.
Runbook targets: a 10 s video in under 30 s (Quick, CPU) and under 60 s (Deep).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app as A  # noqa: E402
from app.pipeline import analyse  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


def main() -> None:
    vids = sorted((REPO_ROOT / "cache" / "app_test_videos").glob("*.mp4"))
    rows = ["| File | Mode | Verdict | Probability | Faces | Flicker | Drift | Seconds |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    crop = A.cropper()
    for mode in ("Quick", "Deep"):
        eng = A.engine(mode)
        analyse(str(vids[0]), eng, crop, is_video=True, n_frames=4)  # warm-up (model load, CUDA init)
        for v in vids:
            t = time.time()
            a = analyse(str(v), eng, crop, is_video=True)
            dt = time.time() - t
            rows.append(f"| {v.name} | {mode} | {a.verdict or a.message} | {a.prob:.3f} | {len(a.faces)}/{a.n_frames} | "
                        f"{a.flicker if a.flicker is None else round(a.flicker, 3)} | {a.drift if a.drift is None else round(a.drift, 3)} | {dt:.1f} |")
            print(rows[-1], flush=True)
    nf = analyse(str(REPO_ROOT / "cache" / "app_test_videos" / "no_face.jpg"), A.engine("Quick"), crop, is_video=False)
    rows.append(f"| no_face.jpg | Quick | {nf.message} | | 0 | | | |")
    out = REPO_ROOT / "results" / "app_check.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("# App check (T6.4)\n\nTest videos are built from test-split face crops (see "
                   "`scripts/make_app_test_videos.py`); file names give the true label.\n\n" + "\n".join(rows) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
