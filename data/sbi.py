"""M1: self-blended image (SBI) engine with recipe logging.

make_sbi(img_uint8, landmarks, rng) -> (image, concepts[10], boundary_map[16,16], recipe)

1. Region: full face (0.4), mouth (0.25), eyes+brows (0.2), nose+cheeks (0.15); mask = filled convex hull.
2. Boundary: elastic warp of the mask with p=0.5; Gaussian blur (odd kernel 3..31); scale to [0, 1].
3. Source transforms on a copy (at least one): colour (0.5), texture blur/sharpen (0.5),
   resampling (0.4), geometry (0.4).
4. Blend with ratio a in {0.25, 0.5, 0.75, 1.0}: out = img * (1 - a*mask) + src * a*mask.
5. Concepts c1..c10: c1=a, c2=a*colour strength, c3=a if blurred, c4=a if sharpened, c5=a if resampled,
   c6=1 mouth, c7=1 eyes, c8=1 full face, c9=a if geometry, c10=1 if elastic.
6. Boundary map: 4*mask*(1-mask), average-pooled to 16x16, scaled to max 1.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

CONCEPT_NAMES = [
    "blend_strength", "colour_mismatch", "blur", "sharpen", "resampling",
    "mouth_region", "eye_region", "full_face", "geometric_misalignment", "irregular_boundary",
]
REGIONS = ["full_face", "mouth", "eyes_brows", "nose_cheeks"]
REGION_P = [0.4, 0.25, 0.2, 0.15]
REGION_POINTS = {
    "full_face": list(range(68)),
    "mouth": list(range(48, 68)),
    "eyes_brows": list(range(17, 27)) + list(range(36, 48)),
    "nose_cheeks": list(range(27, 36)) + [2, 3, 13, 14],
}
REGION_DILATE = {"mouth": 8, "eyes_brows": 6}


def region_mask(landmarks: np.ndarray, region: str, size: int) -> np.ndarray:
    pts = np.round(landmarks[REGION_POINTS[region]]).astype(np.int32)
    pts = np.clip(pts, 0, size - 1)
    mask = np.zeros((size, size), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    d = REGION_DILATE.get(region, 0)
    if d:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    return mask.astype(np.float32)


def elastic(mask: np.ndarray, rng: np.random.Generator, alpha: float = 12.0, sigma: float = 8.0) -> np.ndarray:
    h, w = mask.shape
    dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask, gx + dx * 3, gy + dy * 3, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def colour_shift(img: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    shift = rng.uniform(-20, 20, size=3)
    hue = rng.uniform(-10, 10)  # degrees
    bright = rng.uniform(-0.10, 0.10)
    out = img.astype(np.float32) + shift
    out = np.clip(out * (1 + bright), 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(round(hue / 2))) % 180  # OpenCV hue is degrees / 2
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    strength = float(np.clip((np.abs(shift).mean() / 20 + abs(hue) / 10 + abs(bright) / 0.10) / 3, 0, 1))
    return out, strength


def make_sbi(img: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator):
    size = img.shape[0]
    recipe: Dict = {}
    region = REGIONS[int(rng.choice(len(REGIONS), p=REGION_P))]
    mask = region_mask(landmarks, region, size)
    recipe["region"] = region
    recipe["elastic"] = bool(rng.random() < 0.5)
    if recipe["elastic"]:
        mask = elastic(mask, rng)
    k = int(rng.choice(np.arange(3, 32, 2)))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    if mask.max() <= 0:  # degenerate landmarks: fall back to a centred ellipse
        mask = np.zeros((size, size), np.float32)
        cv2.ellipse(mask, (size // 2, size // 2), (size // 3, size * 2 // 5), 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        recipe["fallback_mask"] = True
    mask = mask / mask.max()
    recipe["blur_kernel"] = k

    use = {"colour": rng.random() < 0.5, "texture": rng.random() < 0.5,
           "resample": rng.random() < 0.4, "geometry": rng.random() < 0.4}
    if not any(use.values()):
        use[["colour", "texture", "resample", "geometry"][int(rng.integers(4))]] = True
    src = img.copy()
    colour_strength = 0.0
    blurred = sharpened = False
    if use["colour"]:
        src, colour_strength = colour_shift(src, rng)
    if use["texture"]:
        if rng.random() < 0.5:
            src = cv2.GaussianBlur(src, (0, 0), float(rng.uniform(1.0, 2.0)))
            blurred = True
        else:
            blur = cv2.GaussianBlur(src, (0, 0), 1.0)
            src = cv2.addWeighted(src, 1.6, blur, -0.6, 0)
            sharpened = True
    if use["resample"]:
        s = float(rng.uniform(0.5, 0.9))
        small = cv2.resize(src, (max(8, int(size * s)), max(8, int(size * s))), interpolation=cv2.INTER_AREA)
        src = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)
        recipe["resample_scale"] = round(s, 3)
    if use["geometry"]:
        tx, ty = rng.uniform(-0.03, 0.03, size=2) * size
        sc = float(rng.uniform(0.95, 1.05))
        m = cv2.getRotationMatrix2D((size / 2, size / 2), 0, sc)
        m[:, 2] += (tx, ty)
        src = cv2.warpAffine(src, m, (size, size), borderMode=cv2.BORDER_REFLECT)
    recipe.update(colour=bool(use["colour"]), colour_strength=round(colour_strength, 3), blurred=blurred,
                  sharpened=sharpened, resampled=bool(use["resample"]), geometry=bool(use["geometry"]))

    a = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
    recipe["blend_ratio"] = a
    am = (a * mask)[..., None]
    out = np.clip(img.astype(np.float32) * (1 - am) + src.astype(np.float32) * am, 0, 255).astype(np.uint8)

    concepts = np.array([
        a, a * colour_strength, a if blurred else 0.0, a if sharpened else 0.0, a if use["resample"] else 0.0,
        1.0 if region == "mouth" else 0.0, 1.0 if region == "eyes_brows" else 0.0, 1.0 if region == "full_face" else 0.0,
        a if use["geometry"] else 0.0, 1.0 if recipe["elastic"] else 0.0,
    ], dtype=np.float32)
    edge = 4.0 * mask * (1.0 - mask)
    boundary = cv2.resize(edge, (16, 16), interpolation=cv2.INTER_AREA)
    if boundary.max() > 0:
        boundary = boundary / boundary.max()
    return out, concepts, boundary.astype(np.float32), recipe, mask


class SBIMaker:
    """Callable used by FaceFrameDataset for the `sbi` cell: loads cached landmarks for a crop."""

    def __init__(self, cfg: Dict) -> None:
        from utils.config import load_paths
        self.cache_root = Path(load_paths()["cache_root"])
        self.mean_lm: Optional[np.ndarray] = None
        mean_path = self.cache_root / "landmarks" / "mean_landmarks.npy"
        if mean_path.exists():
            self.mean_lm = np.load(mean_path)

    def landmarks_for(self, video_id: str, frame_path: str) -> np.ndarray:
        p = self.cache_root / "landmarks" / video_id / f"{Path(frame_path).stem}.npy"
        if p.exists():
            return np.load(p)
        if self.mean_lm is None:
            raise FileNotFoundError(f"No landmarks for {frame_path} and no mean_landmarks.npy")
        return self.mean_lm

    def __call__(self, img: np.ndarray, video_id: str, frame_path: str, flipped: bool = False):
        lm = self.landmarks_for(video_id, frame_path).copy()
        if flipped:
            lm[:, 0] = img.shape[1] - 1 - lm[:, 0]
        rng = np.random.default_rng(random.getrandbits(63))
        out, concepts, boundary, _recipe, _mask = make_sbi(img, lm, rng)
        return out, concepts, boundary
