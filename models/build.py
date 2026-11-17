"""Build any model from config['model']. New backbones register here as they are added."""
from __future__ import annotations

from typing import Dict

import torch.nn as nn


def build_model(model_cfg: Dict, pretrained: bool = True) -> nn.Module:
    backbone = model_cfg["backbone"]
    if backbone == "swin_b":
        from models.swin import SwinDetector
        return SwinDetector(
            backbone_name=model_cfg.get("timm_name", "swin_base_patch4_window7_224"),
            feature_norm=model_cfg.get("feature_norm", "instance"),
            domain_cfg=model_cfg.get("domain"),
            pretrained=pretrained,
        )
    if backbone in ("clip_vitl14_lora", "clip_vitl14_frozen"):
        head = model_cfg.get("head", "linear")
        if head == "linear" and model_cfg.get("freq_branch", "none") == "none" and not (model_cfg.get("domain") or {}).get("enabled"):
            from models.clip_lora import ClipLinearDetector
            return ClipLinearDetector(model_cfg, pretrained=pretrained)
        from models.grace import GraceDetector  # concept / frequency / fusion / domain variants (Day 3-4)
        return GraceDetector(model_cfg, pretrained=pretrained)
    raise ValueError(f"Unknown backbone {backbone!r}")
