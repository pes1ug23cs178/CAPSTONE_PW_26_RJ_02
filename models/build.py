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
    raise ValueError(f"Unknown backbone {backbone!r}")
