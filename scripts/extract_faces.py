"""Face crops from videos with the app's cropper (app/pipeline.py FaceCropper: SFD detector, eye-levelled,
224 x 224), largest face per frame. Frames: linspace(5, n - 5, 32, endpoint=False), the rule that
reproduces the frame numbers of the Feb 2026 crops; frames are read in order (seeking in H.264 is inexact). Crop side = MARGIN x the SFD box:
0.93 by default, measured against the Feb 2026 RetinaFace crops (scripts/calibrate_crops.py, 85 crops,
median side ratio 0.931, centre offset ~1%, same eye levelling), so new crops match the existing test crops.

Extract (resumable: a video with meta.json is skipped):
  python scripts/extract_faces.py extract <video_glob> <out_root> [--n 32] [--limit N]
  -> <out_root>/<video_stem>/aligned/frame_XXXXXX_face_00.png  (XXXXXX = index of the sampled frame, 0..31)
     <out_root>/<video_stem>/meta.json  (video frame numbers, failures, yaw/pitch)

Check against the old crops (same video frames, old crop vs new crop):
  python scripts/extract_faces.py compare <old_video_dir> <video_path>
  <old_video_dir> holds alignment_metadata.json and aligned/; prints mean |difference| and the best shift.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import pipeline  # noqa: E402
from pipeline import FaceCropper  # noqa: E402


def sample_index(total: int, n: int = 32) -> np.ndarray:
    if total <= 10 + n:
        return np.linspace(0, max(total - 1, 0), num=min(n, max(total, 1))).astype(int)
    return np.linspace(5, total - 5, n, endpoint=False).astype(int)


def read_frames(path, wanted):
    """{frame number: RGB frame} for the wanted frame numbers, decoding the video in order."""
    want, out, i = set(int(w) for w in wanted), {}, 0
    cap = cv2.VideoCapture(str(path))
    last = max(want) if want else -1
    while i <= last and cap.grab():
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out, i


def frame_count(path) -> int:
    cap, n = cv2.VideoCapture(str(path)), 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def extract(a):
    pipeline.MARGIN = a.margin
    vids = sorted(glob.glob(a.videos))[: a.limit or None]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fc = FaceCropper()
    t0, done = time.time(), 0
    for k, v in enumerate(vids):
        d = out / Path(v).stem
        if (d / "meta.json").exists():
            continue
        (d / "aligned").mkdir(parents=True, exist_ok=True)
        total = frame_count(v)
        idx = sample_index(total, a.n)
        frames, _ = read_frames(v, idx)
        rec = []
        for j, i in enumerate(idx):
            f = frames.get(int(i))
            r = fc.crop_with_pose(f) if f is not None else None
            if r is None:
                rec.append({"sample": j, "frame": int(i), "ok": False})
                continue
            crop, yaw, pitch = r
            cv2.imwrite(str(d / "aligned" / f"frame_{j:06d}_face_00.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            rec.append({"sample": j, "frame": int(i), "ok": True, "yaw": round(float(yaw), 1),
                        "pitch": round(float(pitch), 1)})
        meta = {"video": Path(v).as_posix(), "total_frames": total, "crops": sum(r["ok"] for r in rec), "frames": rec}
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f"{k + 1}/{len(vids)} videos, {el / done:.1f} s per video", flush=True)
    print(f"finished: {done} new videos in {(time.time() - t0) / 60:.1f} min")


def compare(a):
    pipeline.MARGIN = a.margin
    old = Path(a.old_dir)
    meta = json.loads((old / "alignment_metadata.json").read_text())
    fc = FaceCropper()
    nums = [int(re.search(r"_frame_(\d+)", x["frame_path"]).group(1)) for x in meta["aligned_faces"]]
    frames, total = read_frames(a.video, nums)
    same = list(sample_index(frame_count(a.video))) == nums
    diffs, shifts = [], []
    for x, n in zip(meta["aligned_faces"], nums):
        oldc = cv2.cvtColor(cv2.imread(str(old / "aligned" / Path(x["aligned_face_path"]).name)), cv2.COLOR_BGR2RGB)
        f = frames.get(n)
        r = fc.crop(f) if f is not None else None
        if r is None:
            continue
        g1 = cv2.cvtColor(oldc, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY).astype(np.float32)
        diffs.append(float(np.abs(g1 - g2).mean()))
        (dx, dy), _ = cv2.phaseCorrelate(g1, g2)
        shifts.append((dx, dy))
    s = np.array(shifts)
    print(f"frame rule reproduces old frame numbers: {same}; " + f"{len(diffs)} frames: mean |diff| {np.mean(diffs):.1f} / 255, shift dx {s[:, 0].mean():+.1f} dy "
          f"{s[:, 1].mean():+.1f} px (median |shift| {np.median(np.hypot(s[:, 0], s[:, 1])):.1f})")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("videos")
    e.add_argument("out")
    e.add_argument("--n", type=int, default=32)
    e.add_argument("--limit", type=int, default=0)
    c = sp.add_parser("compare")
    c.add_argument("old_dir")
    c.add_argument("video")
    for q in (e, c):
        q.add_argument("--margin", type=float, default=0.93)
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else compare(a)


if __name__ == "__main__":
    main()
