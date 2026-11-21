"""Degradations.

Evaluation (T4.3), selected by code:
  jpeg:q    re-encode each crop as JPEG (OpenCV), q in {90, 70, 50, 30}
  down:s    downscale by s (INTER_AREA), upscale back (bilinear), s in {0.5, 0.25}
  h264:crf  encode each video's crops in frame order as MP4 (libx264, veryfast, 25 fps, 224x224),
            decode back; cached under cache/h264/crf<crf>/<video_id>/<frame_stem>.png

Student training (T5.2), applied per sample: clean 0.2, JPEG q20-60 0.35, downscale 0.25-0.5 0.25,
JPEG q30-50 after downscale 0.5 0.2.
With h264=True (S3h): a quarter of the degraded samples use single-frame H.264 (CRF 24-38) instead.

Training augmentation h264_frame(img, crf) (F1h, S3h): one image encoded as a single H.264 frame with libx264 via
PyAV and decoded back. It is intra-only (no motion prediction), so it approximates, but is not the same as, the
video-level h264:crf evaluation; CRF 40 is never used in training, so the strongest test level stays unseen.
"""
from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from data.dataset import jpeg


def downscale(img: np.ndarray, s: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def parse(spec: Optional[str]) -> Tuple[str, Optional[float]]:
    if spec in (None, "", "clean"):
        return "clean", None
    kind, val = spec.split(":")
    return kind, float(val)


class ImageDegrade:
    """Picklable per-crop degradation (Windows DataLoader workers pickle the dataset)."""

    def __init__(self, kind: str, val: float) -> None:
        self.kind, self.val = kind, val

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.kind == "jpeg":
            return jpeg(img, int(self.val))
        if self.kind == "down":
            return downscale(img, self.val)
        raise ValueError(self.kind)


class StudentDegrade:
    def __init__(self, h264: bool = False) -> None:
        self.h264 = h264

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return student_degrade(img, self.h264)


def image_degrade_fn(spec: Optional[str]) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    kind, val = parse(spec)
    if kind in ("clean", "h264"):  # h264 is handled by remapping paths to the decoded cache
        return None
    if kind in ("jpeg", "down"):
        return ImageDegrade(kind, val)
    raise ValueError(f"Unknown degradation {spec!r}")


def h264_frame(img: np.ndarray, crf: int) -> np.ndarray:
    """Encode one RGB uint8 image (even height/width) as a single libx264 frame and decode it back."""
    import av
    h, w = img.shape[:2]
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p"
    enc.options = {"crf": str(int(crf)), "preset": "veryfast", "tune": "zerolatency"}
    packets = list(enc.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24"))) + list(enc.encode(None))
    dec = av.CodecContext.create("h264", "r")
    frames = []
    for pkt in packets:
        frames += dec.decode(pkt)
    frames += dec.decode(None)
    return frames[0].to_ndarray(format="rgb24") if frames else img


def student_degrade(img: np.ndarray, h264: bool = False) -> np.ndarray:
    r = random.random()
    if h264 and r >= 0.2 and random.random() < 0.25:
        return h264_frame(img, random.randint(24, 38))
    if r < 0.2:
        return img
    if r < 0.55:
        return jpeg(img, random.randint(20, 60))
    if r < 0.80:
        return downscale(img, random.uniform(0.25, 0.5))
    return jpeg(downscale(img, 0.5), random.randint(30, 50))


# ----------------------------------------------------------------------------- H.264 cache

def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _encode_video(args) -> Optional[str]:
    exe, frames, out_dir, crf = args
    out_dir = Path(out_dir)
    if out_dir.is_dir() and len(list(out_dir.glob("*.png"))) == len(frames):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, f in enumerate(frames):
            shutil.copyfile(f, tmp / f"{i:05d}.png")
        mp4 = tmp / "v.mp4"
        enc = [exe, "-y", "-loglevel", "error", "-framerate", "25", "-i", str(tmp / "%05d.png"), "-s", "224x224",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)), "-pix_fmt", "yuv420p", str(mp4)]
        dec = [exe, "-y", "-loglevel", "error", "-i", str(mp4), str(tmp / "d_%05d.png")]
        for cmd in (enc, dec):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                return f"{out_dir.name}: {r.stderr[:200]}"
        for i, f in enumerate(frames):
            shutil.move(str(tmp / f"d_{i + 1:05d}.png"), out_dir / f"{Path(f).stem}.png")
    return None


def build_h264_cache(csv_path: str, crf: int, data_root: Path, cache_root: Path, workers: int = 12) -> Dict[str, str]:
    """Returns {frame_path: cached decoded frame path relative to data_root}."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    exe = ffmpeg_exe()
    base = cache_root / "h264" / f"crf{int(crf)}"
    jobs, mapping = [], {}
    for vid, g in df.groupby("video_id", sort=False):
        frames = sorted(g["frame_path"].tolist())
        out_dir = base / vid
        jobs.append((exe, [str(data_root / f) for f in frames], str(out_dir), crf))
        for f in frames:
            mapping[f] = (out_dir / f"{Path(f).stem}.png").relative_to(data_root).as_posix()
    with ThreadPoolExecutor(workers) as pool:
        errors = [e for e in pool.map(_encode_video, jobs) if e]
    if errors:
        raise RuntimeError(f"H.264 cache failed for {len(errors)} videos, e.g. {errors[:3]}")
    return mapping
