"""Measure the geometry of the old (Feb 2026, RetinaFace) crops relative to the SFD face box used by the new
extraction, so new crops can match the old ones.

For each old crop of a video that is now downloaded: SIFT-match the crop against video frames near its
recorded frame number, estimate the similarity transform (crop -> frame), keep the best frame (most inliers),
then run SFD on that frame and express the old crop as
    side_ratio = old crop side / max(SFD box w, h)
    dx, dy     = (old crop centre - SFD box centre) / max(SFD box w, h)
    dangle     = old crop rotation - eye-line angle from SFD landmarks

python scripts/calibrate_crops.py <n_videos> [--per 8]     (prints medians; writes results/crop_calibration.csv)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "ff++" / "faces" / "real" / "REAL"
NEW = ROOT / "ff++_full" / "original_sequences" / "youtube" / "c23" / "videos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_videos", type=int)
    ap.add_argument("--per", type=int, default=8)
    a = ap.parse_args()
    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cuda", face_detector="sfd")
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    vids = [v for v in sorted(p.stem for p in NEW.glob("*.mp4")) if (OLD / v / "alignment_metadata.json").exists()]
    rows = []
    for v in vids[: a.n_videos]:
        meta = json.loads((OLD / v / "alignment_metadata.json").read_text())
        cap = cv2.VideoCapture(str(NEW / f"{v}.mp4"))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        items = meta["aligned_faces"]
        for x in items[:: max(1, len(items) // a.per)][: a.per]:
            n = int(re.search(r"_frame_(\d+)", x["frame_path"]).group(1))
            old = cv2.imread(str(OLD / v / "aligned" / Path(x["aligned_face_path"]).name))
            k1, d1 = sift.detectAndCompute(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), None)
            if d1 is None or len(k1) < 20:
                continue
            best = None
            for j in range(max(0, n - 15), min(len(frames), n + 16)):
                g = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
                k2, d2 = sift.detectAndCompute(g, None)
                if d2 is None:
                    continue
                good = [m for m, m2 in bf.knnMatch(d1, d2, k=2) if m.distance < 0.7 * m2.distance]
                if len(good) < 12:
                    continue
                src = np.float32([k1[m.queryIdx].pt for m in good])
                dst = np.float32([k2[m.trainIdx].pt for m in good])
                M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M is None:
                    continue
                ni = int(inl.sum())
                if best is None or ni > best[0]:
                    best = (ni, j, M)
            if best is None or best[0] < 25:
                continue
            ni, j, M = best
            scale = math.hypot(M[0, 0], M[1, 0])  # frame px per crop px
            ang = math.degrees(math.atan2(M[1, 0], M[0, 0]))
            cx, cy = M @ np.array([112.0, 112.0, 1.0])
            res = fa.get_landmarks_from_image(cv2.cvtColor(frames[j], cv2.COLOR_BGR2RGB), return_bboxes=True)
            if res is None or res[0] is None or len(res[0]) == 0:
                continue
            lms, _, boxes = res
            i = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]))
            b, lm = boxes[i], lms[i]
            s = max(b[2] - b[0], b[3] - b[1])
            le, re_ = lm[36:42].mean(0), lm[42:48].mean(0)
            eye = math.degrees(math.atan2(re_[1] - le[1], re_[0] - le[0]))
            rows.append({"video": v, "old_frame": n, "best_frame": j, "inliers": ni,
                         "side_ratio": 224 * scale / s, "dx": (cx - (b[0] + b[2]) / 2) / s,
                         "dy": (cy - (b[1] + b[3]) / 2) / s, "crop_angle": -ang, "eye_angle": eye,
                         "dangle": -ang - eye, "fps": cap.get(cv2.CAP_PROP_FPS)})
        cap.release()
        print(v, len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "crop_calibration.csv", index=False)
    print(df[["side_ratio", "dx", "dy", "dangle", "inliers"]].describe().round(3))
    print("frame offset (best - recorded):", (df.best_frame - df.old_frame).value_counts().head(8).to_dict())


if __name__ == "__main__":
    main()
