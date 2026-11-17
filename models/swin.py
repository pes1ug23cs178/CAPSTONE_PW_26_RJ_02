"""Swin-B detector (B1, B2), ported from train_swin_deepfake.py.

Backbone swin_base_patch4_window7_224 (timm) -> 1024-d pooled feature -> feature norm
-> LayerNorm + Dropout + Linear(1024, 2). Optional DANN domain head on the normed feature.
The model normalizes its [0, 1] input with ImageNet mean/std inside forward().
"""
from __future__ import annotations

from typing import Dict, List

import timm
import torch
import torch.nn as nn

from models.domain import DomainHead

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SwinDetector(nn.Module):
    def __init__(self, backbone_name: str = "swin_base_patch4_window7_224", feature_norm: str = "instance",
                 domain_cfg: Dict | None = None, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.feat_dim = feat_dim
        self.feature_norm = feature_norm
        if feature_norm == "instance":
            self.feat_norm = nn.InstanceNorm1d(1, affine=True)
        elif feature_norm == "layer":
            self.feat_norm = nn.LayerNorm(feat_dim)
        else:
            raise ValueError(f"Unknown feature_norm {feature_norm!r}")
        self.head = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(0.3), nn.Linear(feat_dim, 2))
        domain_cfg = domain_cfg or {}
        self.domain_enabled = bool(domain_cfg.get("enabled", False))
        if self.domain_enabled:
            self.domain_head = DomainHead(feat_dim, int(domain_cfg["num_domains"]))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor, alpha: float = 0.0) -> Dict[str, torch.Tensor]:
        x = (x - self.mean) / self.std
        feats = self.backbone(x)
        if self.feature_norm == "instance":
            z = self.feat_norm(feats.unsqueeze(1)).squeeze(1)
        else:
            z = self.feat_norm(feats)
        h = self.head[0](z)  # LayerNorm: the representation the classifier actually uses
        logits2 = self.head[2](self.head[1](h))
        out = {"logits2": logits2, "logit": logits2[:, 1] - logits2[:, 0], "z": z}
        if self.domain_enabled:
            # domain head reads the same normalised features as the classifier (see models/domain.py)
            out["domain_logits"] = self.domain_head(h, alpha)
        return out

    def param_groups(self, lr: Dict[str, float]) -> List[Dict]:
        head_params = list(self.head.parameters()) + list(self.feat_norm.parameters())
        if self.domain_enabled:
            head_params += list(self.domain_head.parameters())
        return [
            {"params": list(self.backbone.parameters()), "lr": lr["backbone"], "name": "backbone"},
            {"params": head_params, "lr": lr["heads"], "name": "heads"},
        ]

    def gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layers[-1].blocks[-1].norm1
