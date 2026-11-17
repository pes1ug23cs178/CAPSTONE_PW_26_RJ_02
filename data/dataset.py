"""FaceFrameDataset: one aligned face crop per row of a split CSV.

Returns a dict (see Interfaces > Batch dictionary). Images are float tensors in [0, 1], unnormalized;
each model normalizes inside forward().

Index convention for the sampler: 0..N-1 are rows of the CSV. N + i means "make a self-blended
image (SBI) from real row i" (needs `sbi_fn`, added on Day 2).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.config import load_paths

DOMAIN_IDS = {"real": 0, "Deepfakes": 1, "Face2Face": 2, "FaceSwap": 3, "sbi": 4}
NUM_CONCEPTS = 10
IMG_SIZE = 224


def load_rgb(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def light_augment(img: np.ndarray, cfg: Dict, allow_flip: bool = True, compress: bool = True) -> np.ndarray:
    """Flip p=0.5, small colour jitter, JPEG q60-100 p=0.3, Gaussian blur p=0.1 (T1.4 defaults)."""
    if allow_flip and random.random() < cfg.get("flip", 0.5):
        img = img[:, ::-1]
    cj = cfg.get("color_jitter", 0.1)
    if cj > 0:
        b = 1.0 + random.uniform(-cj, cj)
        c = 1.0 + random.uniform(-cj, cj)
        s = 1.0 + random.uniform(-cj, cj)
        f = img.astype(np.float32)
        mean = f.mean()
        f = (f - mean) * c + mean
        f = f * b
        gray = f.mean(axis=2, keepdims=True)
        f = gray + (f - gray) * s
        img = np.clip(f, 0, 255).astype(np.uint8)
    if compress:
        img = compress_augment(img, cfg)
    return np.ascontiguousarray(img)


def compress_augment(img: np.ndarray, cfg: Dict) -> np.ndarray:
    if random.random() < cfg.get("jpeg_p", 0.3):
        lo, hi = cfg.get("jpeg_q", [60, 100])
        img = jpeg(img, random.randint(lo, hi))
    if random.random() < cfg.get("blur_p", 0.1):
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return np.ascontiguousarray(img)


class FaceFrameDataset(Dataset):
    def __init__(self, csv_path: str, train: bool, augment_cfg: Optional[Dict] = None,
                 sbi_fn: Optional[Callable] = None, limit: Optional[int] = None,
                 degrade_fn: Optional[Callable] = None, path_map: Optional[Dict[str, str]] = None) -> None:
        """degrade_fn: applied to every loaded crop after augmentation (evaluation degradations and the
        student's input degradation). path_map: frame_path -> substitute path (H.264 decoded cache)."""
        self.degrade_fn = degrade_fn
        self.path_map = path_map
        self.csv_path = str(csv_path)
        df = pd.read_csv(csv_path, dtype={"video_id": str, "source_id": str}, keep_default_na=False)
        bad = df["frame_path"].str.contains(r"(?:^|/)frames/", regex=True)
        if bad.any():
            raise ValueError(f"{csv_path}: {int(bad.sum())} paths under frames/ (forbidden), e.g. {df.loc[bad, 'frame_path'].iloc[0]}")
        if limit:
            df = df.sample(n=min(limit, len(df)), random_state=0).reset_index(drop=True)
        self.df = df
        self.root = load_paths()["data_root"]
        self.train = train
        self.augment_cfg = augment_cfg or {}
        self.sbi_fn = sbi_fn
        self.paths: List[str] = df["frame_path"].tolist()
        self.labels = df["label"].astype(int).to_numpy()
        self.methods: List[str] = df["method"].tolist()
        self.video_ids: List[str] = df["video_id"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def cell_indices(self, cells: List[str]) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        real_idx = [i for i, m in enumerate(self.methods) if m == "real"]
        for cell in cells:
            if cell == "real":
                out[cell] = real_idx
            elif cell == "sbi":
                out[cell] = [len(self) + i for i in real_idx]
            else:
                out[cell] = [i for i, m in enumerate(self.methods) if m == cell]
            if not out[cell]:
                raise ValueError(f"Cell {cell!r} has no samples in {self.csv_path}")
        return out

    def _base_item(self, i: int, image: np.ndarray, method: str, label: int) -> Dict:
        is_ffpp_fake = label == 1 and method != "sbi"
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0),
            "label": torch.tensor(float(label)),
            "concepts": torch.zeros(NUM_CONCEPTS),
            "concept_mask": torch.zeros(NUM_CONCEPTS) if is_ffpp_fake else torch.ones(NUM_CONCEPTS),
            "boundary": torch.zeros(16, 16),
            "boundary_mask": torch.tensor(0.0 if is_ffpp_fake else 1.0),
            "domain_id": torch.tensor(DOMAIN_IDS.get(method, -1), dtype=torch.long),
            "video_id": self.video_ids[i],
            "method": method,
            "path": self.paths[i],
        }

    def __getitem__(self, index: int) -> Dict:
        if index >= len(self):
            return self._sbi_item(index - len(self))
        path = self.paths[index]
        if self.path_map is not None:
            path = self.path_map.get(path, path)
        img = load_rgb(self.root / path)
        if self.train:
            img = light_augment(img, self.augment_cfg)
        if self.degrade_fn is not None:
            img = np.ascontiguousarray(self.degrade_fn(img))
        return self._base_item(index, img, self.methods[index], int(self.labels[index]))

    def _sbi_item(self, i: int) -> Dict:
        if self.sbi_fn is None:
            raise RuntimeError("SBI index requested but no sbi_fn was given")
        img = load_rgb(self.root / self.paths[i])
        flipped = False
        if self.train:
            # flip + colour jitter BEFORE blending (landmarks are mirrored to match), compression AFTER,
            # so the boundary map stays aligned with the image and artefacts look like a re-encoded fake.
            flipped = random.random() < self.augment_cfg.get("flip", 0.5)
            if flipped:
                img = np.ascontiguousarray(img[:, ::-1])
            img = light_augment(img, self.augment_cfg, allow_flip=False, compress=False)
        sbi_img, concepts, boundary = self.sbi_fn(img, self.video_ids[i], self.paths[i], flipped)
        if self.train:
            sbi_img = np.ascontiguousarray(compress_augment(sbi_img, self.augment_cfg))
        if self.degrade_fn is not None:
            sbi_img = np.ascontiguousarray(self.degrade_fn(sbi_img))
        item = self._base_item(i, sbi_img, "sbi", 1)
        item["concepts"] = torch.as_tensor(concepts, dtype=torch.float32)
        item["boundary"] = torch.as_tensor(boundary, dtype=torch.float32)
        item["video_id"] = f"sbi_{self.video_ids[i]}"
        return item
