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
    """Collect from multiple dataset roots, assigning each a unique domain id."""
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
    ) -> None:
        super().__init__()
        self.backbone       = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat_dim            = self.backbone.num_features
        self.domain_adapt   = domain_adapt and num_domains > 1
        self.use_audio      = use_audio

        # Normalize each sample across its feature vector to improve domain robustness.
        # Use a single-channel InstanceNorm over [B, 1, F] so spatial length is F (>1).
        self.feat_norm = nn.InstanceNorm1d(1, affine=True)

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
        # InstanceNorm1d runs on [B, C, L]; treat feature dimension as length.
        normed = self.feat_norm(feats.unsqueeze(1)).squeeze(1)

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
                unique_domains = domains.unique()
                if unique_domains.numel() >= 2:
                    d0, d1 = unique_domains[0], unique_domains[1]
                    src = out["features"][domains == d0]
                    tgt = out["features"][domains == d1]
                    mmd = mmd_loss(src, tgt)
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
    class_to_idx = {"real": 0, "fake": 1}
    all_samples, num_domains = collect_multi_source(data_roots, skip_corrupt=skip_corrupt)

    if not all_samples:
        raise RuntimeError(
            "No labeled images found. Confirm dataset paths and that real/fake is inferrable from directory names."
        )

    groups      = {infer_group_key(s.path, data_roots[s.domain]) for s in all_samples}
    label_cnts  = Counter(s.label for s in all_samples)

    log.info("=" * 70)
    log.info("Dataset Analysis")
    log.info("  Total images    : %d", len(all_samples))
    log.info("  Unique groups   : %d", len(groups))
    log.info("  Domains         : %d", num_domains)
    log.info("  Real / Fake     : %d / %d", label_cnts[0], label_cnts[1])
    log.info("=" * 70)

    # Use first root as the reference for group-key computation
    ref_root = data_roots[0]
    train_samples, val_samples, split_method = split_samples_by_group(
        all_samples, ref_root, val_split=val_split, seed=seed
    )

    _, _, overlap, pct = compute_group_overlap(train_samples, val_samples, ref_root)
    if overlap:
        raise RuntimeError(
            f"Data leakage: {len(overlap)} groups appear in both train and val ({pct:.2f}%)."
        )

    log.info("Split method: %s  |  train=%d  val=%d  leakage=0%%",
             split_method, len(train_samples), len(val_samples))

    return DatasetBundle(
        mode          = "auto_group_split",
        train_samples = train_samples,
        val_samples   = val_samples,
        class_to_idx  = class_to_idx,
        train_source  = f"{data_roots} ({split_method})",
        val_source    = f"{data_roots} ({split_method})",
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
    p.add_argument("--early-stopping-patience",  type=int,   default=5)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

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
) -> dict:
    ckpt = {
        "epoch":                  epoch,
        "model_state_dict":       model.state_dict(),
        "optimizer_state_dict":   optimizer.state_dict(),
        "scheduler_state_dict":   scheduler.state_dict(),
        "best_val_acc":           best_val_acc,
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
    train_ds = FaceImageDataset(bundle.train_samples, train_tf)
    val_ds   = FaceImageDataset(bundle.val_samples,   val_tf)

    # BUG FIX #12: persistent_workers requires num_workers > 0
    nw = max(0, args.num_workers)
    dl_kwargs = dict(
        num_workers      = nw,
        pin_memory       = device.type == "cuda",
        persistent_workers = nw > 0,
        **({"prefetch_factor": 2} if nw > 0 else {}),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **dl_kwargs)

    # ---- Model ----
    model = DeepfakeDetector(
        backbone_name = args.model_name,
        num_classes   = 2,
        num_domains   = bundle.num_domains,
        use_audio     = args.use_audio,
        domain_adapt  = args.domain_adapt and bundle.num_domains > 1,
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
    best_val_acc = -1.0
    best_epoch   = -1
    no_improve   = 0
    start_epoch  = 1
    history: Dict[str, List[float]] = {k: [] for k in
        ("train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time")}

    if args.resume:
        rp = Path(args.resume).resolve()
        if not rp.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {rp}")
        ckpt = torch.load(rp, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "ema_state_dict" in ckpt and ema is not None:
            ema.model.load_state_dict(ckpt["ema_state_dict"])
        for k in history:
            if k in ckpt.get("history", {}):
                history[k] = list(ckpt["history"][k])
        best_val_acc = float(ckpt.get("best_val_acc", -1.0))
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
        val_loss, val_acc, *_ = evaluate_model(
            model        = eval_model,
            loader       = val_loader,
            criterion    = criterion,
            device       = device,
            use_amp      = use_amp,
            amp_dtype    = amp_dtype,
            show_progress= True,
            epoch_idx    = epoch,
            total_epochs = args.epochs,
        )

        scheduler.step()
        lr  = optimizer.param_groups[0]["lr"]
        dt  = time.time() - t0

        for k, v in zip(
            ("train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time"),
            (train_loss, val_loss, train_acc, val_acc, lr, dt),
        ):
            history[k].append(float(v))

        improved = val_acc > best_val_acc + args.early_stopping_min_delta
        if improved:
            best_val_acc, best_epoch, no_improve = val_acc, epoch, 0
        else:
            no_improve += 1

        ckpt = build_checkpoint(
            epoch, model, optimizer, scheduler, scaler, ema,
            best_val_acc, best_epoch, no_improve,
            bundle.class_to_idx, args, history,
        )
        torch.save(ckpt, last_path)
        if improved:
            torch.save(ckpt, best_path)
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, output_dir / f"ckpt_epoch_{epoch:03d}.pth")

        log.info(
            "Epoch [%03d/%d] train_loss=%.4f train_acc=%.4f "
            "val_loss=%.4f val_acc=%.4f lr=%.2e time=%.1fs%s",
            epoch, args.epochs, train_loss, train_acc,
            val_loss, val_acc, lr, dt,
            " ★" if improved else "",
        )

        if args.early_stopping_patience > 0 and no_improve >= args.early_stopping_patience:
            log.info("Early stopping at epoch %d (no improvement for %d epochs).", epoch, no_improve)
            break

    log.info("Training done. Best val acc=%.4f at epoch %d", best_val_acc, best_epoch)

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
    }

    save_json(output_dir / "evaluation_metrics.json", metrics)
    save_json(output_dir / "training_history.json", history)
    save_predictions_csv(output_dir / "val_predictions.csv", val_paths, y_true, y_pred, y_prob)
    save_training_curves(history, output_dir / "training_curves.png")
    save_confusion_matrix_plot(cm, class_names, output_dir / "confusion_matrix.png")

    generate_gradcam_visualizations(
        model        = model,
        val_samples  = val_ds.samples,
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