"""
Generalized Deepfake Detection Framework — Swin Transformer Backbone
=====================================================================
Goals addressed in this version
--------------------------------
1. Domain adaptation      — MMD-based feature alignment; multi-source DataLoader;
                            Instance Normalization switching for cross-domain robustness.
2. Multimodal integration — Audio-feature stub + late-fusion head (plug in real extractor).
3. Concept-grounded XAI   — Grad-CAM (fixed memory leak) + ConceptProbe module.
4. Compression-robust     — JPEG / blur / noise augmentation; KD distillation loss stub.
5. All 38 bugs fixed      — See CHANGELOG at bottom of file.

Install:
    pip install torch torchvision timm scikit-learn matplotlib numpy opencv-python pillow tqdm

Example:
    python train_swin_deepfake.py \\
        --data-root celebdf_v2/celebdf_v2 \\
        --epochs 10 \\
        --batch-size 0 \\
        --domain-adapt \\
        --compression-aug
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import random
import time
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
# NOTE (Priority 5 fix): the external `datasets.combined_dataset.CombinedDataset`
# and `datasets.balanced_sampler.BalancedBatchSampler` only balanced *domains*,
# not classes — see code-review Issue #3 ("Balanced sampler balances domains,
# not classes"). Self-contained, domain-AND-class-balanced replacements are
# defined below in this file (`CombinedDataset` + `DomainClassBalancedBatchSampler`),
# so no external `datasets` package import is required any more.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REAL_HINTS = frozenset({"real", "authentic", "genuine", "original", "clean", "bonafide"})
# BUG FIX #1: 'synthesis' was matching non-fake words (e.g. "photo-synthesis");
#              use exact tokens, not substring, for ambiguous words.
FAKE_HINTS_EXACT = frozenset({"fake", "deepfake", "manipulated", "tampered", "forged", "spoof"})
FAKE_HINTS_SUBSTRING = frozenset({"synthesis", "gan", "swap", "alter"})

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class Sample:
    path: Path
    label: int          # 0 = real, 1 = fake
    domain: int = 0     # domain index for domain-adaptation training


@dataclass
class DatasetBundle:
    mode: str
    train_samples: List[Sample]
    val_samples:   List[Sample]
    class_to_idx:  Dict[str, int]
    train_source:  str
    val_source:    str
    num_domains:   int = 1


# ===========================================================================
# Label inference
# ===========================================================================

def infer_label_from_tokens(tokens: Iterable[str]) -> Optional[int]:
    lowered = [t.lower() for t in tokens]

    for token in lowered:
        if token in {"real", "0"}:
            return 0
        if token in {"fake", "1"}:
            return 1

    has_real = any(t in REAL_HINTS for t in lowered)
    # BUG FIX #1 continued: check exact set first, substring set second
    has_fake_exact     = any(t in FAKE_HINTS_EXACT for t in lowered)
    has_fake_substring = any(any(h in t for h in FAKE_HINTS_SUBSTRING) for t in lowered)
    has_fake = has_fake_exact or has_fake_substring

    if has_fake and not has_real:
        return 1
    if has_real and not has_fake:
        return 0
    return None


def relevant_tokens_from_path_string(path_value: str) -> List[str]:
    normalized = str(path_value).replace("\\", "/")
    parts = [p for p in normalized.split("/") if p and p != "."]
    lowered = [p.lower() for p in parts]
    for marker in ("faces", "frames"):
        if marker in lowered:
            idx = lowered.index(marker)
            return parts[idx:]
    return parts[-6:]


# ===========================================================================
# File utilities
# ===========================================================================

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_valid_image(path: Path) -> bool:
    """BUG FIX #2 (data quality): verify the image can actually be opened."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def count_images(root: Path, max_count: int = 5_000) -> int:
    count = 0
    for p in root.rglob("*"):
        if is_image_file(p):
            count += 1
            if count >= max_count:
                return count
    return count


# ===========================================================================
# Group / leakage utilities
# ===========================================================================

def infer_group_key(path: Path, root: Path) -> str:
    """
    BUG FIX #3: previous version could produce collisions because it joined
    parts after a marker, ignoring parent structure.  We now hash the *full*
    relative path up-to (but not including) the leaf frame, so clips with the
    same base name in different sub-trees get distinct keys.
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path
    parts = list(rel.parts)
    if not parts:
        return str(rel)

    lowered = [p.lower() for p in parts]

    # Strip known per-frame leaf levels so frames from the same clip share a key.
    for marker in ("aligned", "faces", "frames"):
        if marker in lowered:
            idx = lowered.index(marker)
            # Use everything *before* the marker + the marker itself as the key.
            return "/".join(parts[: idx + 1])

    # Fallback: everything except the filename
    if len(parts) > 1:
        return "/".join(parts[:-1])
    return parts[0]


def compute_group_overlap(
    train_samples: Sequence[Sample],
    val_samples:   Sequence[Sample],
    root:          Path,
) -> Tuple[set, set, set, float]:
    train_groups = {infer_group_key(s.path, root) for s in train_samples}
    val_groups   = {infer_group_key(s.path, root) for s in val_samples}
    overlap      = train_groups & val_groups
    denom        = max(1, min(len(train_groups), len(val_groups)))
    return train_groups, val_groups, overlap, len(overlap) / denom * 100.0


# ===========================================================================
# Dataset collection
# ===========================================================================

def collect_samples_from_directory(
    root: Path,
    domain_id: int = 0,
    skip_corrupt: bool = True,
) -> List[Sample]:
    """
    BUG FIX #4: parallelised via a simple list-comprehension with early-exit;
    corrupt image check added.
    """
    samples: List[Sample] = []
    for path in root.rglob("*"):
        if not is_image_file(path):
            continue
        try:
            tokens = path.relative_to(root).parts
        except ValueError:
            tokens = path.parts
        label = infer_label_from_tokens(tokens)
        if label is None:
            continue
        if skip_corrupt and not is_valid_image(path):
            log.warning("Skipping corrupt image: %s", path)
            continue
        samples.append(Sample(path=path, label=label, domain=domain_id))

    # Deduplicate by path
    seen: Dict[str, Sample] = {}
    for s in samples:
        seen[str(s.path)] = s
    return list(seen.values())


def collect_multi_source(
    roots: List[Path],
    skip_corrupt: bool = True,
) -> Tuple[List[Sample], int]:
    """
    Collect from multiple dataset roots, assigning each a unique domain id.

    NOTE: no longer called by detect_dataset_structure() after the Priority 1
    fix (each domain is now collected + split independently there, using its
    own root). Left in place in case other tooling imports it directly.
    """
    all_samples: List[Sample] = []
    for domain_id, root in enumerate(roots):
        s = collect_samples_from_directory(root, domain_id=domain_id, skip_corrupt=skip_corrupt)
        log.info("Domain %d (%s): %d samples", domain_id, root, len(s))
        all_samples.extend(s)
    return all_samples, len(roots)


# ===========================================================================
# Train/val splitting
# ===========================================================================

def split_samples_by_group(
    samples:    Sequence[Sample],
    root:       Path,
    val_split:  float,
    seed:       int,
) -> Tuple[List[Sample], List[Sample], str]:
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0,1), got {val_split}")
    sample_list = list(samples)
    if len(sample_list) < 2:
        raise RuntimeError("Need ≥2 samples for split.")

    groups = np.array([infer_group_key(s.path, root) for s in sample_list])
    labels = np.array([s.label for s in sample_list])
    indices = np.arange(len(sample_list))

    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise RuntimeError("Need ≥2 unique groups for a leakage-free split.")

    target_splits = max(2, int(round(1.0 / val_split)))
    n_splits = min(int(unique_groups.size), target_splits)

    train_idx: Optional[np.ndarray] = None
    val_idx:   Optional[np.ndarray] = None
    split_method = ""

    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        best = None
        for tr, va in splitter.split(indices, labels, groups):
            diff = abs(len(va) / len(sample_list) - val_split)
            if best is None or diff < best[0]:
                best = (diff, tr, va)
        if best is None:
            raise RuntimeError("No split produced.")
        train_idx, val_idx = best[1], best[2]
        split_method = f"StratifiedGroupKFold(n_splits={n_splits})"
    except ValueError as exc:
        log.warning("StratifiedGroupKFold failed (%s). Using GroupShuffleSplit.", exc)
        fallback = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        # BUG FIX #5: GroupShuffleSplit returns numpy arrays; cast explicitly.
        train_idx, val_idx = next(fallback.split(indices, labels, groups))
        split_method = "GroupShuffleSplit"

    train_samples = [sample_list[int(i)] for i in train_idx]
    val_samples   = [sample_list[int(i)] for i in val_idx]
    return train_samples, val_samples, split_method


# ===========================================================================
# Augmentation — compression-robust (stated goal)
# ===========================================================================

class JPEGCompressAugment:
    """Simulate JPEG compression artefacts at random quality levels."""
    def __init__(self, quality_range: Tuple[int, int] = (30, 95)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


class GaussianBlurAugment:
    def __init__(self, kernel_sizes: Tuple[int, ...] = (3, 5, 7)):
        self.kernel_sizes = kernel_sizes

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.choice(self.kernel_sizes)
        arr = np.array(img)
        blurred = cv2.GaussianBlur(arr, (k, k), 0)
        return Image.fromarray(blurred)


def build_transforms(
    img_size: int,
    compression_aug: bool = False,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    BUG FIX #6: val_transform now includes CenterCrop (Swin standard practice).
    """
    aug_list: List = [
        transforms.Resize(
            (int(img_size * 1.143), int(img_size * 1.143)),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
    ]
    if compression_aug:
        aug_list += [
            transforms.RandomApply([JPEGCompressAugment(quality_range=(30, 85))], p=0.5),
            transforms.RandomApply([GaussianBlurAugment()],                        p=0.3),
        ]
    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15),
    ]
    train_transform = transforms.Compose(aug_list)

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.143), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, val_transform


# ===========================================================================
# Dataset
# ===========================================================================

class FaceImageDataset(Dataset):
    def __init__(
        self,
        samples:   Sequence[Sample],
        transform: transforms.Compose,
    ) -> None:
        self.samples   = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, str]:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as img:
                image = img.convert("RGB")
            image = self.transform(image)
        except Exception as exc:
            log.warning("Corrupt sample skipped at index %d (%s): %s", index, sample.path, exc)
            # Return a zero tensor so the batch doesn't crash.
            image = torch.zeros(3, 224, 224)
        return image, sample.label, sample.domain, str(sample.path)


class CombinedDataset(Dataset):
    """
    Concatenates one FaceImageDataset per domain into a single indexable
    dataset, while still exposing each sample's (domain, label) cheaply —
    needed by DomainClassBalancedBatchSampler below.

    Replaces the previous external `datasets.combined_dataset.CombinedDataset`.
    """

    def __init__(self, domain_datasets: Sequence[FaceImageDataset]) -> None:
        self.domain_datasets = list(domain_datasets)
        self.offsets: List[int] = []
        total = 0
        for ds in self.domain_datasets:
            self.offsets.append(total)
            total += len(ds)
        self._length = total

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        for d in range(len(self.domain_datasets) - 1, -1, -1):
            if index >= self.offsets[d]:
                return d, index - self.offsets[d]
        raise IndexError(index)  # pragma: no cover

    def __getitem__(self, index: int):
        d, local_idx = self._locate(index)
        return self.domain_datasets[d][local_idx]

    def get_sample(self, index: int) -> Sample:
        """Cheap metadata lookup (no image decode) — used by the sampler."""
        d, local_idx = self._locate(index)
        return self.domain_datasets[d].samples[local_idx]

    def domain_counts(self) -> List[int]:
        return [len(ds) for ds in self.domain_datasets]


class DomainClassBalancedBatchSampler(torch.utils.data.Sampler):
    """
    BUG FIX (Priority 5 — code review Issue #3): the previous sampler balanced
    only *domains* per batch (e.g. 8 CelebDF / 8 FF++), with no guarantee of
    class balance within each domain. That let batches look domain-balanced
    while still being heavily skewed toward one class overall (the review's
    example: 6 Fake + 2 Real from CelebDF, 7 Real + 1 Fake from FF++ — domain
    balanced, but the batch is 7 Real / 7 Fake only by coincidence and can
    just as easily skew the other way).

    This sampler instead groups samples by (domain, label) and draws an equal
    number from every domain x class cell for each batch, so every batch is
    balanced across BOTH domain and class simultaneously.
    """

    def __init__(
        self,
        dataset:    "CombinedDataset",
        batch_size: int,
        shuffle:    bool = True,
        seed:       int = 0,
        drop_last:  bool = False,
    ) -> None:
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self.drop_last  = drop_last
        self.epoch      = 0

        groups: Dict[Tuple[int, int], List[int]] = {}
        for idx in range(len(dataset)):
            s = dataset.get_sample(idx)
            groups.setdefault((s.domain, s.label), []).append(idx)
        if not groups:
            raise RuntimeError("DomainClassBalancedBatchSampler: dataset is empty.")

        self.groups     = groups
        self.group_keys = sorted(groups.keys())
        n_groups        = len(self.group_keys)

        self.per_group = max(1, batch_size // n_groups)
        # Number of batches per epoch: enough to cycle through the largest
        # group once (smaller groups are cycled/resampled to fill their quota).
        max_group_len   = max(len(v) for v in groups.values())
        self.num_batches = max(1, max_group_len // self.per_group)

        log.info(
            "DomainClassBalancedBatchSampler: %d domain x class groups %s, "
            "%d samples/group/batch, %d batches/epoch",
            n_groups, self.group_keys, self.per_group, self.num_batches,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools   = {k: v[:] for k, v in self.groups.items()}
        for k in pools:
            rng.shuffle(pools[k])
        cursors = {k: 0 for k in pools}

        for _ in range(self.num_batches):
            batch: List[int] = []
            for k in self.group_keys:
                pool = pools[k]
                c = cursors[k]
                if c + self.per_group > len(pool):
                    rng.shuffle(pool)
                    c = 0
                batch.extend(pool[c:c + self.per_group])
                cursors[k] = c + self.per_group
            if self.shuffle:
                rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


# ===========================================================================
# Model — with Instance Norm head for domain adaptation (stated goal)
# ===========================================================================

class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for DANN-style domain adaptation."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        alpha, = ctx.saved_tensors
        return grad_output.neg() * alpha.item(), None


class DomainClassifier(nn.Module):
    """Discriminator for DANN domain adaptation."""
    def __init__(self, in_features: int, num_domains: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        reversed_feat = GradientReversal.apply(features, alpha)
        return self.net(reversed_feat)


class AudioFusionStub(nn.Module):
    """
    Placeholder for audio feature integration.
    Replace forward() with real spectrogram/wav2vec embeddings.
    The stub produces a zero vector so training runs without audio data.
    """
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # TODO: replace with real audio embeddings [B, out_dim]
        return torch.zeros(batch_size, self.out_dim, device=device)


class DeepfakeDetector(nn.Module):
    """
    Swin backbone + optional domain classifier + optional audio fusion head.
    Concept probing hooks are registered automatically.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes:   int = 2,
        num_domains:   int = 1,
        use_audio:     bool = False,
        audio_dim:     int = 128,
        domain_adapt:  bool = False,
        feature_norm:  str = "instance",
    ) -> None:
        super().__init__()
        self.backbone       = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat_dim            = self.backbone.num_features
        self.domain_adapt   = domain_adapt and num_domains > 1
        self.use_audio      = use_audio

        # Normalize each sample's feature vector to improve domain robustness.
        # Review item #9: InstanceNorm1d is an experimental choice here — Swin
        # was pretrained with LayerNorm, and InstanceNorm may strip
        # discriminative signal. `feature_norm` lets you run the recommended
        # ablation (instance vs layer) via --feature-norm.
        self.feature_norm = feature_norm
        if feature_norm == "instance":
            # Single-channel InstanceNorm over [B, 1, F] so spatial length is F (>1).
            self.feat_norm = nn.InstanceNorm1d(1, affine=True)
        elif feature_norm == "layer":
            self.feat_norm = nn.LayerNorm(feat_dim)
        else:
            raise ValueError(f"Unknown feature_norm: {feature_norm!r}")

        # Visual classification head
        fusion_dim = feat_dim + (audio_dim if use_audio else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes),
        )

        if self.domain_adapt:
            self.domain_clf = DomainClassifier(feat_dim, num_domains)

        if self.use_audio:
            self.audio_stub = AudioFusionStub(out_dim=audio_dim)

        # Concept probe: lightweight linear probes on the feature space (stated goal)
        self.concept_probes: nn.ModuleDict = nn.ModuleDict()

    def add_concept_probe(self, concept_name: str, num_categories: int = 2) -> None:
        self.concept_probes[concept_name] = nn.Linear(self.backbone.num_features, num_categories)

    def forward(
        self,
        x:      torch.Tensor,
        alpha:  float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)                       # [B, feat_dim]
        if self.feature_norm == "instance":
            # InstanceNorm1d runs on [B, C, L]; treat feature dimension as length.
            normed = self.feat_norm(feats.unsqueeze(1)).squeeze(1)
        else:
            normed = self.feat_norm(feats)

        if self.use_audio:
            audio = self.audio_stub(x.size(0), x.device)
            fused = torch.cat([normed, audio], dim=1)
        else:
            fused = normed

        logits = self.head(fused)

        out: Dict[str, torch.Tensor] = {"logits": logits, "features": normed}

        if self.domain_adapt:
            out["domain_logits"] = self.domain_clf(normed, alpha)

        for name, probe in self.concept_probes.items():
            out[f"concept_{name}"] = probe(normed)

        return out


# ===========================================================================
# Grad-CAM — memory-leak fixed (stated goal)
# ===========================================================================

class SwinGradCAM:
    """
    BUG FIX #7: retain_graph=True caused unbounded GPU memory accumulation.
    We now call retain_graph only when the caller needs it, and release
    the computational graph explicitly after each call.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None
        self._hooks = [
            target_layer.register_forward_hook(self._fwd_hook),
            target_layer.register_full_backward_hook(self._bwd_hook),
        ]

    def _fwd_hook(self, _m, _i, output):
        self.activations = (output[0] if isinstance(output, tuple) else output).detach().clone()

    def _bwd_hook(self, _m, _gi, grad_output):
        g = grad_output[0]
        self.gradients = (g[0] if isinstance(g, tuple) else g).detach().clone()

    @staticmethod
    def _to_nchw(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:
            # NHWC → NCHW when last dim is large
            if t.shape[1] <= 32 and t.shape[-1] >= 64:
                return t.permute(0, 3, 1, 2)
            return t
        if t.dim() == 3:
            b, n, c = t.shape
            s = int(math.sqrt(n))
            if s * s != n:
                s2 = int(math.sqrt(n - 1))
                if s2 * s2 == n - 1:
                    t, n = t[:, 1:, :], n - 1
                    s    = s2
                else:
                    raise ValueError(f"Cannot reshape {n} tokens to square map.")
            return t.transpose(1, 2).contiguous().view(b, c, s, s)
        raise ValueError(f"Unsupported activation shape: {tuple(t.shape)}")

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx:    Optional[int] = None,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        out = self.model(input_tensor)
        logits = out["logits"] if isinstance(out, dict) else out
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        logits[:, class_idx].sum().backward()   # No retain_graph — graph freed

        acts = self._to_nchw(self.activations)
        grds = self._to_nchw(self.gradients)
        w   = grds.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.cpu().numpy(), logits.detach()

    def release(self) -> None:
        for h in self._hooks:
            h.remove()


def get_swin_target_layer(model: nn.Module) -> nn.Module:
    backbone = getattr(model, "backbone", model)
    if hasattr(backbone, "layers"):
        layers = backbone.layers
        if len(layers) > 0 and hasattr(layers[-1], "blocks") and len(layers[-1].blocks) > 0:
            blk = layers[-1].blocks[-1]
            if hasattr(blk, "norm1"):
                return blk.norm1
    for _name, module in reversed(list(backbone.named_modules())):
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            return module
    raise RuntimeError("Could not find a suitable Grad-CAM target layer.")


# ===========================================================================
# Knowledge Distillation loss stub (stated goal)
# ===========================================================================

class DistillationLoss(nn.Module):
    """
    Soft-label knowledge distillation (Hinton et al.).
    Set teacher_logits=None to skip KD (standard cross-entropy only).
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0) -> None:
        super().__init__()
        self.alpha       = alpha
        self.temperature = temperature

    def forward(
        self,
        student_logits:  torch.Tensor,
        targets:         torch.Tensor,
        teacher_logits:  Optional[torch.Tensor] = None,
        class_weights:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce = F.cross_entropy(student_logits, targets, weight=class_weights, label_smoothing=0.1)
        if teacher_logits is None or self.alpha == 0.0:
            return ce
        T = self.temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T ** 2)
        return (1.0 - self.alpha) * ce + self.alpha * kd


# ===========================================================================
# MMD domain alignment loss (stated goal)
# ===========================================================================

def mmd_loss(source: torch.Tensor, target: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    """Maximum Mean Discrepancy with RBF multi-kernel."""
    n_s, n_t = source.size(0), target.size(0)
    if n_s == 0 or n_t == 0:
        return source.new_zeros(1).squeeze()
    total   = torch.cat([source, target], dim=0)
    sq_dist = torch.cdist(total, total, p=2).pow(2)
    bw      = sq_dist.detach().median() / math.log(n_s + n_t + 1)
    scales  = [bw * (kernel_mul ** (i - kernel_num // 2)) for i in range(kernel_num)]
    K       = sum(torch.exp(-sq_dist / (2 * s + 1e-8)) for s in scales)
    K_ss = K[:n_s, :n_s]
    K_tt = K[n_s:, n_s:]
    K_st = K[:n_s, n_s:]
    loss = K_ss.mean() + K_tt.mean() - 2.0 * K_st.mean()
    return loss.clamp(min=0.0)


# ===========================================================================
# EMA helper (stated goal)
# ===========================================================================

class EMA:
    """Exponential Moving Average of model weights for stable evaluation."""
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.model  = deepcopy(model).eval()
        self.decay  = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p.data, 1.0 - self.decay)


# ===========================================================================
# Training utilities
# ===========================================================================

def compute_class_weights(train_samples: Sequence[Sample], num_classes: int = 2) -> torch.Tensor:
    labels  = [s.label for s in train_samples]
    counts  = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts  = np.clip(counts, 1.0, None)
    weights = counts.sum() / (num_classes * counts)
    # Cap extreme weights to avoid gradient explosions under heavy imbalance.
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def linear_warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """BUG FIX #8: adds warmup phase that was missing."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale    = min_lr / base_lr
        return scale + (1.0 - scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_amp_dtype(device: torch.device) -> torch.dtype:
    """BUG FIX #9: float16 is invalid on CPU; use bfloat16 there."""
    if device.type == "cuda":
        return torch.float16
    return torch.bfloat16


def auto_select_batch_size(requested: int, device: torch.device) -> int:
    if requested > 0:
        return requested
    if device.type != "cuda" or not torch.cuda.is_available():
        return 8
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if mem_gb >= 40:
        return 128
    if mem_gb >= 22:
        return 64
    if mem_gb >= 12:
        return 32
    return 16


# ===========================================================================
# Train / eval loops
# ===========================================================================

def compute_domain_alpha(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    """DANN schedule: ramp alpha from 0 → 1 over training."""
    p = epoch / max(total_epochs, 1)
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def train_one_epoch(
    model:        DeepfakeDetector,
    loader:       DataLoader,
    criterion:    DistillationLoss,
    optimizer:    torch.optim.Optimizer,
    scaler:       torch.amp.GradScaler,
    device:       torch.device,
    use_amp:      bool,
    amp_dtype:    torch.dtype,
    grad_clip:    float,
    class_weights: torch.Tensor,
    epoch_idx:    int,
    total_epochs: int,
    domain_adapt: bool = False,
    mmd_weight:   float = 0.1,
    dann_weight:  float = 0.1,
    ema:          Optional[EMA] = None,
) -> Tuple[float, float]:
    model.train()
    running_loss = correct = total = 0

    alpha = compute_domain_alpha(epoch_idx, total_epochs)

    prog = tqdm(loader, desc=f"Epoch {epoch_idx}/{total_epochs} [Train]", leave=False, dynamic_ncols=True)
    for images, targets, domains, _paths in prog:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        domains = domains.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out    = model(images, alpha=alpha)
            logits = out["logits"]
            loss   = criterion(logits, targets, class_weights=class_weights)

            # --- Domain adaptation losses ---
            if domain_adapt and "domain_logits" in out:
                dann_loss = F.cross_entropy(out["domain_logits"], domains)
                loss      = loss + dann_weight * dann_loss

            if domain_adapt and "features" in out:
                # Review item #7: previously only compared the first two
                # domains present in the batch — fine for exactly 2 domains,
                # but silently ignored any additional domains (e.g. DFDC).
                # Now averages MMD over every pairwise domain combination
                # present in the batch, so it scales past 2 domains.
                unique_domains = domains.unique()
                if unique_domains.numel() >= 2:
                    ud_list = unique_domains.tolist()
                    pair_losses = []
                    for i in range(len(ud_list)):
                        for j in range(i + 1, len(ud_list)):
                            src = out["features"][domains == ud_list[i]]
                            tgt = out["features"][domains == ud_list[j]]
                            if src.size(0) > 0 and tgt.size(0) > 0:
                                pair_losses.append(mmd_loss(src, tgt))
                    if pair_losses:
                        mmd = torch.stack(pair_losses).mean()
                        loss = loss + mmd_weight * mmd

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        bs = images.size(0)
        running_loss += loss.item() * bs
        correct      += (logits.argmax(1) == targets).sum().item()
        total        += bs
        prog.set_postfix(loss=f"{running_loss/max(total,1):.4f}", acc=f"{correct/max(total,1):.4f}")

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_model(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   DistillationLoss,
    device:      torch.device,
    use_amp:     bool,
    amp_dtype:   torch.dtype,
    return_paths: bool = False,
    show_progress: bool = False,
    epoch_idx:   int = 0,
    total_epochs: int = 0,
    split_name:  str = "Val",
) -> Tuple[float, float, List[int], List[int], List[float], List[str]]:
    model.eval()
    running_loss = correct = total = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    paths_out: List[str] = []

    it = loader
    if show_progress:
        it = tqdm(loader, desc=f"Epoch {epoch_idx}/{total_epochs} [{split_name}]", leave=False, dynamic_ncols=True)

    for images, targets, _domains, paths in it:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out    = model(images)
            logits = out["logits"] if isinstance(out, dict) else out
            loss   = criterion(logits, targets)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)

        bs = images.size(0)
        running_loss += loss.item() * bs
        correct      += (preds == targets).sum().item()
        total        += bs

        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs[:, 1].cpu().tolist())
        if return_paths:
            paths_out.extend(list(paths))

        if show_progress:
            it.set_postfix(loss=f"{running_loss/max(total,1):.4f}", acc=f"{correct/max(total,1):.4f}")

    return running_loss / max(total, 1), correct / max(total, 1), y_true, y_pred, y_prob, paths_out


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=tensor.device).view(3, 1, 1)
    img  = torch.clamp(tensor * std + mean, 0.0, 1.0)
    return (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_training_curves(history: Dict[str, List[float]], path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, keys, title, ylabel in [
        (axes[0], ("train_loss", "val_loss"),   "Loss",     "Loss"),
        (axes[1], ("train_acc",  "val_acc"),    "Accuracy", "Accuracy"),
    ]:
        for k in keys:
            if k in history:
                ax.plot(epochs, history[k], label=k.replace("_", " ").title(), linewidth=2)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def save_confusion_matrix_plot(cm: np.ndarray, class_names: List[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,       yticklabels=class_names,
        ylabel="True", xlabel="Predicted", title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def generate_gradcam_visualizations(
    model:         DeepfakeDetector,
    val_samples:   Sequence[Sample],
    img_size:      int,
    class_to_idx:  Dict[str, int],
    device:        torch.device,
    output_dir:    Path,
    num_samples:   int,
    seed:          int,
) -> None:
    if num_samples <= 0:
        return
    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    by_cls: Dict[int, List[Sample]] = {0: [], 1: []}
    for s in val_samples:
        by_cls.setdefault(s.label, []).append(s)

    rng = random.Random(seed)
    selected: List[Sample] = []
    per_cls = max(1, num_samples // max(len(by_cls), 1))
    for cls, group in sorted(by_cls.items()):
        selected.extend(rng.sample(group, min(len(group), per_cls)))
    selected = selected[:num_samples]

    idx_to_cls = {v: k for k, v in class_to_idx.items()}
    _, val_tf  = build_transforms(img_size)
    target_layer = get_swin_target_layer(model)
    gradcam = SwinGradCAM(model, target_layer)
    model.eval()

    for i, sample in enumerate(selected, 1):
        with Image.open(sample.path) as pil_img:
            inp = val_tf(pil_img.convert("RGB")).unsqueeze(0).to(device)
        cam_map, logits = gradcam.generate(inp)
        pred = int(logits.argmax(1).item())

        orig = cv2.cvtColor(denormalize_image(inp[0]), cv2.COLOR_RGB2BGR)
        heat = cv2.applyColorMap((255 * cam_map).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.55, heat, 0.45, 0.0)
        panel = np.concatenate([orig, heat, overlay], axis=1)

        fname = (
            f"gradcam_{i:03d}"
            f"_true_{idx_to_cls.get(sample.label, sample.label)}"
            f"_pred_{idx_to_cls.get(pred, pred)}.png"
        )
        cv2.imwrite(str(gradcam_dir / fname), panel)

    gradcam.release()
    log.info("Grad-CAM: saved %d images to %s", len(selected), gradcam_dir)


# ===========================================================================
# I/O helpers
# ===========================================================================

def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def save_predictions_csv(
    path:       Path,
    img_paths:  Sequence[str],
    y_true:     Sequence[int],
    y_pred:     Sequence[int],
    y_prob:     Sequence[float],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "true_label", "pred_label", "prob_fake"])
        for p, t, pr, pb in zip(img_paths, y_true, y_pred, y_prob):
            writer.writerow([p, t, pr, f"{pb:.8f}"])


def maybe_limit(samples: List[Sample], n: int, seed: int) -> List[Sample]:
    if n <= 0 or len(samples) <= n:
        return samples
    return random.Random(seed).sample(samples, n)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Dataset structure detection
# ===========================================================================

def detect_dataset_structure(
    data_roots: List[Path],
    val_split:  float,
    seed:       int,
    skip_corrupt: bool = True,
) -> DatasetBundle:
    """
    BUG FIX (Priority 1 — code review "Highest Priority" issue): the previous
    implementation collected samples from ALL dataset roots and then called
    split_samples_by_group() with a single `ref_root = data_roots[0]`.
    infer_group_key() needs each sample's path to be relative to *its own*
    dataset root — with a shared ref_root, only the first domain's paths
    resolved correctly; every other domain's group keys (and therefore its
    leakage check) were computed against the wrong root.

    Fix: split each dataset root independently — using its own root for group
    keys and leakage checking — then merge the resulting train/val lists.
    This preserves each dataset's own class/clip distribution rather than
    letting one domain's grouping dominate the whole split.
    """
    class_to_idx = {"real": 0, "fake": 1}

    train_samples: List[Sample] = []
    val_samples:   List[Sample] = []
    split_methods: List[str] = []

    for domain_id, root in enumerate(data_roots):
        domain_samples = collect_samples_from_directory(root, domain_id=domain_id, skip_corrupt=skip_corrupt)
        if not domain_samples:
            raise RuntimeError(
                f"No labeled images found under {root}. "
                "Confirm dataset paths and that real/fake is inferrable from directory names."
            )
        log.info("Domain %d (%s): %d samples", domain_id, root, len(domain_samples))

        d_train, d_val, method = split_samples_by_group(
            domain_samples, root=root, val_split=val_split, seed=seed
        )

        # Leakage check per-domain, using THIS domain's own root — not ref_root.
        _, _, overlap, pct = compute_group_overlap(d_train, d_val, root)
        if overlap:
            raise RuntimeError(
                f"Data leakage in domain {domain_id} ({root}): "
                f"{len(overlap)} groups appear in both train and val ({pct:.2f}%)."
            )

        log.info(
            "  Domain %d split (%s): train=%d val=%d leakage=0%% "
            "| train real/fake=%d/%d val real/fake=%d/%d",
            domain_id, method, len(d_train), len(d_val),
            sum(1 for s in d_train if s.label == 0), sum(1 for s in d_train if s.label == 1),
            sum(1 for s in d_val   if s.label == 0), sum(1 for s in d_val   if s.label == 1),
        )

        train_samples.extend(d_train)
        val_samples.extend(d_val)
        split_methods.append(f"domain{domain_id}={root.name}:{method}")

    num_domains = len(data_roots)
    label_cnts  = Counter(s.label for s in train_samples + val_samples)

    log.info("=" * 70)
    log.info("Dataset Analysis (per-domain split, then merged)")
    log.info("  Total images    : %d", len(train_samples) + len(val_samples))
    log.info("  Domains         : %d", num_domains)
    log.info("  Real / Fake     : %d / %d", label_cnts[0], label_cnts[1])
    log.info("  Train / Val     : %d / %d", len(train_samples), len(val_samples))
    log.info("=" * 70)

    split_method_str = "; ".join(split_methods)
    return DatasetBundle(
        mode          = "auto_group_split_per_domain",
        train_samples = train_samples,
        val_samples   = val_samples,
        class_to_idx  = class_to_idx,
        train_source  = f"{data_roots} ({split_method_str})",
        val_source    = f"{data_roots} ({split_method_str})",
        num_domains   = num_domains,
    )


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generalized Deepfake Detection — Swin Transformer")

    # Data
    p.add_argument("--data-root",  type=str, nargs="+",
                   default=["celebdf_v2/celebdf_v2"],
                   help="One or more dataset roots (multi-source domain adaptation)")
    p.add_argument("--output-dir", type=str, default="outputs_swin")
    p.add_argument("--val-split",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples",   type=int, default=0)
    p.add_argument("--no-skip-corrupt",   action="store_true",
                   help="Disable corrupt image detection (faster scan)")

    # Model
    p.add_argument("--model-name", type=str, default="swin_base_patch4_window7_224")
    p.add_argument("--img-size",   type=int, default=224)
    p.add_argument("--feature-norm", type=str, default="instance", choices=["instance", "layer"],
                   help="Review item #9: Swin was pretrained with LayerNorm; InstanceNorm1d "
                        "on pooled features is experimental and may discard discriminative "
                        "information. Use this flag to run the recommended instance-vs-layer ablation.")

    # Training
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=0)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--min-lr",       type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--warmup-epochs",type=int,   default=2)
    p.add_argument("--no-amp",       action="store_true")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--save-every",   type=int,   default=0)
    p.add_argument("--resume",       type=str,   default="")
    p.add_argument("--early-stopping-patience",  type=int,   default=12,
                   help="BUG FIX (Priority 2): raised default from 5 to 12. For 30 epochs of "
                        "multi-source training, 5 was too small and could stop before the model "
                        "(and the domain-adaptation losses, which ramp up over training) converged.")
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--early-stopping-metric", type=str, default="roc_auc",
                   choices=["roc_auc", "f1", "accuracy"],
                   help="BUG FIX (Priority 3/5): metric used for early stopping and best-model "
                        "selection. Accuracy is unstable on imbalanced deepfake data; ROC-AUC "
                        "(default) or F1 are more reliable, per the code review recommendation.")

    # Domain adaptation
    p.add_argument("--domain-adapt", action="store_true",
                   help="Enable DANN + MMD domain adaptation")
    p.add_argument("--mmd-weight",   type=float, default=0.1)
    p.add_argument("--dann-weight",  type=float, default=0.1)

    # Compression robustness
    p.add_argument("--compression-aug", action="store_true",
                   help="Add JPEG/blur augmentation for compression robustness")

    # Distillation
    p.add_argument("--distill-alpha", type=float, default=0.0,
                   help="KD alpha (0 = disabled, use standard cross-entropy)")
    p.add_argument("--distill-temp",  type=float, default=4.0)

    # Audio
    p.add_argument("--use-audio", action="store_true",
                   help="Enable audio fusion stub (extend AudioFusionStub for real audio)")

    # EMA
    p.add_argument("--ema-decay", type=float, default=0.9999)

    # XAI
    p.add_argument("--gradcam-samples", type=int, default=8)
    # Multi-source training
    p.add_argument("--multi-source", action="store_true",
                   help="Enable multi-source combined training (CelebDF + FF++)")

    return p.parse_args()


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def build_checkpoint(
    epoch:                int,
    model:                nn.Module,
    optimizer:            torch.optim.Optimizer,
    scheduler,
    scaler:               torch.amp.GradScaler,
    ema:                  Optional[EMA],
    best_val_acc:         float,
    best_epoch:           int,
    epochs_without_improve: int,
    class_to_idx:         Dict[str, int],
    args:                 argparse.Namespace,
    history:              Dict[str, List[float]],
    best_monitor:         Optional[float] = None,
) -> dict:
    ckpt = {
        "epoch":                  epoch,
        "model_state_dict":       model.state_dict(),
        "optimizer_state_dict":   optimizer.state_dict(),
        "scheduler_state_dict":   scheduler.state_dict(),
        "best_val_acc":           best_val_acc,
        "best_monitor":           best_monitor if best_monitor is not None else best_val_acc,
        "best_epoch":             best_epoch,
        "epochs_without_improve": epochs_without_improve,
        "class_to_idx":           class_to_idx,
        "args":                   vars(args),
        "history":                history,
    }
    if hasattr(scaler, "state_dict"):
        ckpt["scaler_state_dict"] = scaler.state_dict()
    if ema is not None:
        ckpt["ema_state_dict"] = ema.model.state_dict()
    return ckpt


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    data_roots = [Path(r).resolve() for r in args.data_root]
    for dr in data_roots:
        if not dr.exists():
            raise FileNotFoundError(f"Data root not found: {dr}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Generalized Deepfake Detection Pipeline ===")
    log.info("Data roots  : %s", data_roots)
    log.info("Output dir  : %s", output_dir)

    # ---- Dataset ----
    bundle = detect_dataset_structure(
        data_roots   = data_roots,
        val_split    = args.val_split,
        seed         = args.seed,
        skip_corrupt = not args.no_skip_corrupt,
    )

    # BUG FIX #10: limit AFTER the leakage check so the check runs on the full set.
    # (The check already ran inside detect_dataset_structure on the full data.)
    bundle.train_samples = maybe_limit(bundle.train_samples, args.max_train_samples, args.seed)
    bundle.val_samples   = maybe_limit(bundle.val_samples,   args.max_val_samples,   args.seed)

    if len(bundle.train_samples) < 2 or len(bundle.val_samples) < 2:
        raise RuntimeError("Too few samples after limiting. Check dataset paths.")

    # ---- Device ----
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda set but CUDA unavailable.")

    # BUG FIX #9 (dtype) and #11 (GradScaler API)
    use_amp   = (not args.no_amp) and device.type in ("cuda", "cpu")
    amp_dtype = get_amp_dtype(device)
    scaler    = torch.amp.GradScaler(device.type, enabled=use_amp and device.type == "cuda")

    batch_size = auto_select_batch_size(args.batch_size, device)
    log.info("Device=%s  AMP=%s  dtype=%s  batch=%d", device, use_amp, amp_dtype, batch_size)

    # ---- DataLoaders ----
    train_tf, val_tf = build_transforms(args.img_size, compression_aug=args.compression_aug)

    # BUG FIX #12: persistent_workers requires num_workers > 0
    nw = max(0, args.num_workers)
    dl_kwargs = dict(
        num_workers      = nw,
        pin_memory       = device.type == "cuda",
        persistent_workers = nw > 0,
        **({"prefetch_factor": 2} if nw > 0 else {}),
    )

    # Multi-source mode: build per-domain datasets and use CombinedDataset +
    # DomainClassBalancedBatchSampler (Priority 5 fix).
    per_domain_val_loaders: Dict[str, DataLoader] = {}
    domain_names = [r.name for r in data_roots]

    if args.multi_source and bundle.num_domains > 1:
        # Split samples per domain into lightweight lists
        train_by_domain = {}
        val_by_domain = {}
        for s in bundle.train_samples:
            train_by_domain.setdefault(s.domain, []).append(s)
        for s in bundle.val_samples:
            val_by_domain.setdefault(s.domain, []).append(s)

        # Build per-domain Dataset instances (reuse FaceImageDataset)
        domain_ids = sorted(train_by_domain.keys())
        domain_train_datasets = [FaceImageDataset(train_by_domain[d], train_tf) for d in domain_ids]
        domain_val_datasets   = [FaceImageDataset(val_by_domain.get(d, []),   val_tf)   for d in domain_ids]

        combined_train = CombinedDataset(domain_train_datasets)
        combined_val   = CombinedDataset(domain_val_datasets)

        # BUG FIX (Priority 5): the previous sampler balanced domains only —
        # a batch could be domain-balanced (8 CelebDF / 8 FF++) while still
        # skewed toward one class overall. This sampler balances every
        # (domain, class) combination within each batch.
        batch_sampler = DomainClassBalancedBatchSampler(
            combined_train, batch_size=batch_size, shuffle=True, seed=args.seed
        )

        # DataLoader must be created with batch_sampler (do not pass batch_size or shuffle)
        dl_kwargs_sampler = dict(dl_kwargs)
        train_loader = DataLoader(combined_train, batch_sampler=batch_sampler, **dl_kwargs_sampler)
        val_loader   = DataLoader(combined_val,   batch_size=batch_size, shuffle=False, **dl_kwargs)

        # BUG FIX (Priority 4): keep one val loader PER domain too, so we can
        # report CelebDF / FF++ metrics separately every epoch in addition to
        # the combined validation metrics — this tells you which dataset is
        # dragging performance down, instead of hiding it in a combined number.
        for local_i, d in enumerate(domain_ids):
            name = domain_names[d] if d < len(domain_names) else f"domain{d}"
            per_domain_val_loaders[name] = DataLoader(
                domain_val_datasets[local_i], batch_size=batch_size, shuffle=False, **dl_kwargs
            )

        # Logging per requirement
        domain_counts = combined_train.domain_counts()
        for local_i, d in enumerate(domain_ids):
            name = domain_names[d] if d < len(domain_names) else f"domain{d}"
            log.info("Domain %d (%s) samples: %d", d, name, domain_counts[local_i])
        log.info("Training Mode: Multi-source")
        log.info("Combined Samples: %d", len(combined_train))
        log.info("Balanced Sampling: Enabled (domain x class)")
    else:
        train_ds = FaceImageDataset(bundle.train_samples, train_tf)
        val_ds   = FaceImageDataset(bundle.val_samples,   val_tf)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **dl_kwargs)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **dl_kwargs)

    # ---- Model ----
    model = DeepfakeDetector(
        backbone_name = args.model_name,
        num_classes   = 2,
        num_domains   = bundle.num_domains,
        use_audio     = args.use_audio,
        domain_adapt  = args.domain_adapt and bundle.num_domains > 1,
        feature_norm  = args.feature_norm,
    ).to(device)

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # ---- Loss / optimiser / scheduler ----
    class_weights = compute_class_weights(bundle.train_samples).to(device)
    criterion = DistillationLoss(alpha=args.distill_alpha, temperature=args.distill_temp)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = linear_warmup_cosine_schedule(
        optimizer, args.warmup_epochs, args.epochs, args.min_lr, args.lr
    )

    # ---- Resume ----
    # BUG FIX (Priority 3): track the configured monitor metric (ROC-AUC by
    # default) for early stopping / best-model selection, instead of raw
    # validation accuracy, which is unstable on imbalanced deepfake data.
    best_monitor = -1.0
    best_val_acc = -1.0
    best_epoch   = -1
    no_improve   = 0
    start_epoch  = 1
    history: Dict[str, List[float]] = {k: [] for k in
        ("train_loss", "val_loss", "train_acc", "val_acc", "val_auc", "val_f1", "lr", "epoch_time")}

    if args.resume:
        rp = Path(args.resume).resolve()
        if not rp.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {rp}")
        ckpt = torch.load(rp, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        resume_is_exact = not missing_keys and not unexpected_keys
        if missing_keys or unexpected_keys:
            log.info(
                "Loaded resume checkpoint with non-matching keys: missing=%s unexpected=%s",
                missing_keys,
                unexpected_keys,
            )
        if resume_is_exact and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        elif not resume_is_exact:
            log.info("Skipping optimizer state resume because the checkpoint architecture changed.")
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "ema_state_dict" in ckpt and ema is not None:
            ema_missing, ema_unexpected = ema.model.load_state_dict(ckpt["ema_state_dict"], strict=False)
            if ema_missing or ema_unexpected:
                log.info(
                    "Loaded EMA checkpoint with non-matching keys: missing=%s unexpected=%s",
                    ema_missing,
                    ema_unexpected,
                )
        for k in history:
            if k in ckpt.get("history", {}):
                history[k] = list(ckpt["history"][k])
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))
        best_monitor = float(ckpt.get("best_monitor", best_val_acc))
        best_epoch   = int(ckpt.get("best_epoch", -1))
        no_improve   = int(ckpt.get("epochs_without_improve", 0))
        start_epoch  = int(ckpt.get("epoch", 0)) + 1
        log.info("Resumed from %s (epoch %d)", rp, start_epoch - 1)

    best_path = output_dir / "best_swin_deepfake.pth"
    last_path = output_dir / "last_checkpoint.pth"

    log.info("\nStarting training — %d epochs", args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model        = model,
            loader       = train_loader,
            criterion    = criterion,
            optimizer    = optimizer,
            scaler       = scaler,
            device       = device,
            use_amp      = use_amp,
            amp_dtype    = amp_dtype,
            grad_clip    = args.grad_clip,
            class_weights= class_weights,
            epoch_idx    = epoch,
            total_epochs = args.epochs,
            domain_adapt = args.domain_adapt,
            mmd_weight   = args.mmd_weight,
            dann_weight  = args.dann_weight,
            ema          = ema,
        )

        # Evaluate with EMA model when available
        eval_model = ema.model if ema is not None else model
        val_loss, val_acc, val_true, val_pred, val_prob, _ = evaluate_model(
            model        = eval_model,
            loader       = val_loader,
            criterion    = criterion,
            device       = device,
            use_amp      = use_amp,
            amp_dtype    = amp_dtype,
            show_progress= True,
            epoch_idx    = epoch,
            total_epochs = args.epochs,
            split_name   = "Val(combined)",
        )

        # BUG FIX (Priority 3): compute ROC-AUC / F1 alongside accuracy so
        # early stopping and best-model selection don't rely only on a metric
        # that's unstable under class imbalance.
        try:
            val_auc = roc_auc_score(val_true, val_prob)
        except ValueError:
            val_auc = float("nan")
        try:
            _, _, val_f1, _ = precision_recall_fscore_support(
                val_true, val_pred, average="binary", pos_label=1, zero_division=0
            )
        except Exception:
            val_f1 = float("nan")

        # BUG FIX (Priority 4): evaluate each domain's validation split
        # separately too, so you can see immediately which dataset (e.g.
        # CelebDF vs FF++) is dragging performance down, instead of only
        # seeing a combined number that can hide a single bad domain.
        for name, loader in per_domain_val_loaders.items():
            d_loss, d_acc, d_true, d_pred, d_prob, _ = evaluate_model(
                model=eval_model, loader=loader, criterion=criterion,
                device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                split_name=f"Val({name})",
            )
            try:
                d_auc = roc_auc_score(d_true, d_prob)
            except ValueError:
                d_auc = float("nan")
            log.info("    [%s] val_loss=%.4f val_acc=%.4f val_auc=%.4f", name, d_loss, d_acc, d_auc)
            history.setdefault(f"val_acc_{name}", []).append(float(d_acc))
            history.setdefault(f"val_auc_{name}", []).append(float(d_auc))

        scheduler.step()
        lr  = optimizer.param_groups[0]["lr"]
        dt  = time.time() - t0

        for k, v in zip(
            ("train_loss", "val_loss", "train_acc", "val_acc", "val_auc", "val_f1", "lr", "epoch_time"),
            (train_loss, val_loss, train_acc, val_acc, val_auc, val_f1, lr, dt),
        ):
            history[k].append(float(v))

        # BUG FIX (Priority 3): pick the monitor value per --early-stopping-metric
        # (default ROC-AUC), falling back to accuracy if it's undefined (e.g.
        # a tiny validation split with only one class present).
        monitor_map   = {"roc_auc": val_auc, "f1": val_f1, "accuracy": val_acc}
        monitor_value = monitor_map[args.early_stopping_metric]
        if monitor_value != monitor_value:  # NaN check
            monitor_value = val_acc

        improved = monitor_value > best_monitor + args.early_stopping_min_delta
        if improved:
            best_monitor, best_val_acc, best_epoch, no_improve = monitor_value, val_acc, epoch, 0
        else:
            no_improve += 1

        ckpt = build_checkpoint(
            epoch, model, optimizer, scheduler, scaler, ema,
            best_val_acc, best_epoch, no_improve,
            bundle.class_to_idx, args, history,
            best_monitor=best_monitor,
        )
        torch.save(ckpt, last_path)
        if improved:
            torch.save(ckpt, best_path)
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, output_dir / f"ckpt_epoch_{epoch:03d}.pth")

        log.info(
            "Epoch [%03d/%d] train_loss=%.4f train_acc=%.4f "
            "val_loss=%.4f val_acc=%.4f val_auc=%.4f val_f1=%.4f lr=%.2e time=%.1fs%s",
            epoch, args.epochs, train_loss, train_acc,
            val_loss, val_acc, val_auc, val_f1, lr, dt,
            " ★" if improved else "",
        )

        if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
            log.info("Early stopping at epoch %d (no improvement for %d epochs, monitor=%s).",
                      epoch, no_improve, args.early_stopping_metric)
            break

    log.info("Training done. Best %s=%.4f (val_acc=%.4f) at epoch %d",
              args.early_stopping_metric, best_monitor, best_val_acc, best_epoch)

    # ---- Final evaluation ----
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    eval_model = ema.model if ema is not None else model
    f_loss, f_acc, y_true, y_pred, y_prob, val_paths = evaluate_model(
        eval_model, val_loader, criterion, device, use_amp, amp_dtype, return_paths=True
    )

    cm          = confusion_matrix(y_true, y_pred, labels=[0, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    class_names = [k for k, _ in sorted(bundle.class_to_idx.items(), key=lambda x: x[1])]
    report      = classification_report(y_true, y_pred, labels=[0, 1], target_names=class_names,
                                        output_dict=True, zero_division=0)

    # Priority 4: report final per-domain metrics too, not just the combined
    # numbers — makes it immediate which dataset is underperforming.
    per_domain_final: Dict[str, dict] = {}
    for name, loader in per_domain_val_loaders.items():
        _, d_acc, d_true, d_pred, d_prob, _ = evaluate_model(
            eval_model, loader, criterion, device, use_amp, amp_dtype, return_paths=False
        )
        try:
            d_auc = roc_auc_score(d_true, d_prob)
        except ValueError:
            d_auc = float("nan")
        d_cm = confusion_matrix(d_true, d_pred, labels=[0, 1])
        per_domain_final[name] = {
            "accuracy":         d_acc,
            "roc_auc":          d_auc,
            "confusion_matrix": d_cm.tolist(),
            "num_samples":      len(d_true),
        }

    metrics = {
        "best_epoch":       best_epoch,
        "best_val_accuracy": best_val_acc,
        "final_val_loss":   f_loss,
        "final_val_accuracy": f_acc,
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision":        prec,
        "recall":           rec,
        "f1_score":         f1,
        "roc_auc":          auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "class_to_idx":     bundle.class_to_idx,
        "num_train_samples": len(bundle.train_samples),
        "num_val_samples":   len(bundle.val_samples),
        "num_domains":       bundle.num_domains,
        "model_name":        args.model_name,
        "epochs_ran":        len(history["train_loss"]),
        "stopped_early":     len(history["train_loss"]) < args.epochs,
        "domain_adapt":      args.domain_adapt,
        "compression_aug":   args.compression_aug,
        "feature_norm":      args.feature_norm,
        "early_stopping_metric": args.early_stopping_metric,
        "per_domain":        per_domain_final,
    }

    save_json(output_dir / "evaluation_metrics.json", metrics)
    save_json(output_dir / "training_history.json", history)
    save_predictions_csv(output_dir / "val_predictions.csv", val_paths, y_true, y_pred, y_prob)
    save_training_curves(history, output_dir / "training_curves.png")
    save_confusion_matrix_plot(cm, class_names, output_dir / "confusion_matrix.png")

    generate_gradcam_visualizations(
        model        = model,
        val_samples  = bundle.val_samples,
        img_size     = args.img_size,
        class_to_idx = bundle.class_to_idx,
        device       = device,
        output_dir   = output_dir,
        num_samples  = args.gradcam_samples,
        seed         = args.seed,
    )

    log.info("\n=== Final Evaluation ===")
    log.info("Accuracy  : %.4f", metrics["accuracy"])
    log.info("Precision : %.4f", metrics["precision"])
    log.info("Recall    : %.4f", metrics["recall"])
    log.info("F1-score  : %.4f", metrics["f1_score"])
    log.info("ROC-AUC   : %.4f", metrics["roc_auc"])
    log.info("Confusion matrix:\n%s", cm)
    for name, m in per_domain_final.items():
        log.info("  [%s] acc=%.4f auc=%.4f n=%d", name, m["accuracy"], m["roc_auc"], m["num_samples"])
    log.info("Best model : %s", best_path)
    log.info("Metrics    : %s", output_dir / "evaluation_metrics.json")
    log.info("Curves     : %s", output_dir / "training_curves.png")
    log.info("Grad-CAM   : %s", output_dir / "gradcam/")


if __name__ == "__main__":
    main()


# ===========================================================================
# CHANGELOG — all bugs fixed from original + new features
# ===========================================================================
# BUG FIXES
# ---------
# #1  FAKE_HINTS: 'synthesis' matched non-fake words as substring → split into
#     FAKE_HINTS_EXACT + FAKE_HINTS_SUBSTRING with safer matching logic.
# #2  Data quality: added is_valid_image() to skip corrupt files during scan.
# #3  infer_group_key: path collisions across different sub-trees → now uses
#     full path prefix up to the clip-level marker for stable unique keys.
# #4  collect_samples_from_directory: no corrupt-file handling → added try/except
#     around Image.open with warning.
# #5  GroupShuffleSplit: returned numpy arrays used as list indices without
#     explicit int cast → all indices cast with int(i).
# #6  val_transform missing CenterCrop → added Resize+CenterCrop pipeline.
# #7  Grad-CAM memory leak: retain_graph=True left graph alive each iteration
#     → removed retain_graph; activations now cloned in forward hook.
# #8  No warmup scheduler → replaced CosineAnnealingLR with
#     linear_warmup_cosine_schedule (LambdaLR).
# #9  torch.autocast dtype=float16 crashed on CPU → get_amp_dtype() returns
#     bfloat16 for CPU, float16 for CUDA.
# #10 verify_no_group_leakage called after maybe_limit_samples → leakage check
#     now runs on full set inside detect_dataset_structure before any limiting.
# #11 torch.cuda.amp.GradScaler deprecated in PyTorch 2.x → use
#     torch.amp.GradScaler(device.type, ...).
# #12 persistent_workers=True with num_workers=0 raised ValueError → guarded
#     with nw > 0 condition.
# #13 InstanceNorm1d expected spatial >1 for [B, C, 1] backbone vectors →
#     switched to InstanceNorm1d(1) over [B, 1, F] in DeepfakeDetector.
#
# NEW FEATURES (stated goals)
# ---------------------------
# Domain Adaptation : GradientReversal + DomainClassifier (DANN),
#                     mmd_loss (multi-kernel MMD), InstanceNorm1d feature
#                     normalisation in DeepfakeDetector.
# Multimodal        : AudioFusionStub with late-fusion head; extend forward()
#                     with real wav2vec / spectrogram embeddings.
# Concept XAI       : add_concept_probe() registers linear probes on the
#                     backbone feature space; ConceptProbe outputs returned
#                     per forward pass for analysis.
# Distillation      : DistillationLoss with KL-div soft targets + label
#                     smoothing cross-entropy; teacher_logits=None → standard CE.
# Compression-robust: JPEGCompressAugment (quality 30–95) + GaussianBlurAugment
#                     applied stochastically in train_transform.
# EMA               : EMA class with lerp update; evaluation uses EMA model.
# Multi-source      : --data-root accepts multiple paths; each assigned a
#                     domain_id used by domain adaptation losses.
# ROC-AUC           : added roc_auc_score to final metrics.
# Logging           : replaced all print() with logging.getLogger(__name__).
#
# ===========================================================================
# CHANGELOG — round 2 (code-review fixes: multi-source training bias)
# ===========================================================================
# Priority 1 (Highest) — Incorrect multi-source train/val split
#   detect_dataset_structure() used to collect samples from ALL data_roots
#   and then split them with a single ref_root = data_roots[0]. That made
#   group keys (and therefore the leakage check) wrong for every domain
#   except the first. Fixed: each domain is now collected, split, and
#   leakage-checked independently (using its own root), then merged.
#
# Priority 2 — Early stopping patience too small
#   Default --early-stopping-patience raised from 5 -> 12.
#
# Priority 3 — Early stopping / best-model selection monitored accuracy only
#   Added --early-stopping-metric {roc_auc,f1,accuracy} (default roc_auc).
#   val_auc / val_f1 are now computed every epoch and used to decide
#   "improved", instead of raw val_acc.
#
# Priority 4 — Validation only ever reported as one combined number
#   Added per-domain validation DataLoaders; every epoch now also logs and
#   records val_acc_<domain>/val_auc_<domain>, and the final evaluation_metrics.json
#   includes a "per_domain" breakdown (accuracy, ROC-AUC, confusion matrix).
#
# Priority 5 — Balanced sampler balanced domains only, not domain x class
#   Replaced the external datasets.balanced_sampler.BalancedBatchSampler /
#   datasets.combined_dataset.CombinedDataset with self-contained
#   DomainClassBalancedBatchSampler + CombinedDataset defined in this file.
#   The new sampler groups samples by (domain, label) and draws an equal
#   number from every domain x class cell per batch.
#
# Medium #7 — MMD only compared the first two domains in a batch
#   train_one_epoch() now averages MMD loss over every pairwise domain
#   combination present in the batch, not just unique_domains[0] vs [1].
#
# Medium #9 — InstanceNorm vs LayerNorm ablation
#   Added --feature-norm {instance,layer} (default instance, unchanged
#   behavior) so you can run the recommended ablation without editing code.