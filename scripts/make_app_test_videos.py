"""Build 5 short test videos for the app check (T6.4): 2 real, 3 fake, from TEST-split videos.

The original source videos are not on this machine, so each video is the test video's aligned face crops in
frame order, enlarged and placed with a slow drift on a 640x480 background (10 s at 25 fps). This exercises
detection, alignment, the timeline and the verdict; it is not a substitute for real camera footage.
Writes cache/app_test_videos/<label>_<video_id>.mp4 and a no-face photo for the "no face" check.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_rgb  # noqa: E402
from data.degrade import ffmpeg_exe  # noqa: E402
from utils.config import REPO_ROOT, load_paths  # noqa: E402

PICKS = [("celebdf_test", 0, 1), ("ffpp_test", 0, 1), ("celebdf_test", 1, 1), ("ffpp_test", 1, 1), ("dfd_test", 1, 1)]


def main() -> None:
    root = load_paths()["data_root"]
    out_dir = REPO_ROOT / "cache" / "app_test_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, label, n in PICKS:
        df = pd.read_csv(REPO_ROOT / "splits" / f"{split}.csv", dtype=str)
        vids = df[df["label"] == str(label)]["video_id"].drop_duplicates().sample(n=n, random_state=3)
        for vid in vids:
            frames = sorted(df[df["video_id"] == vid]["frame_path"])
            with tempfile.TemporaryDirectory() as tmp:
                k = 0
                for t in range(250):
                    crop = load_rgb(root / frames[int(t / 250 * len(frames))])
                    face = cv2.resize(crop, (280, 280), interpolation=cv2.INTER_CUBIC)
                    canvas = np.full((480, 640, 3), (70, 80, 95), np.uint8)
                    x, y = 180 + int(40 * np.sin(t / 40)), 100 + int(20 * np.cos(t / 50))
                    canvas[y:y + 280, x:x + 280] = face
                    cv2.imwrite(str(Path(tmp) / f"{k:05d}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                    k += 1
                out = out_dir / f"{'fake' if label else 'real'}_{vid}.mp4"
                subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "25", "-i", str(Path(tmp) / "%05d.png"),
                                "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)
                print(out.name)
    no_face = np.full((480, 640, 3), 140, np.uint8)
    cv2.putText(no_face, "no face here", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    cv2.imwrite(str(out_dir / "no_face.jpg"), no_face)


if __name__ == "__main__":
    main()
