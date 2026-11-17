"""M9: EfficientNet-B0 student with the same concept bottleneck as the teacher.

Backbone timm efficientnet_b0 (ImageNet), final stage 1280 x 7 x 7. Concept head 1x1 conv -> 10 maps (7x7),
score = top-10% mean (5 of 49 cells). z_s = LayerNorm(Linear(1280 -> 512)(GAP)). Decision layer = M7.
ImageNet normalization inside forward().
"""
from __future__ import annotations

from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn

from models.concepts import ConceptDecision, ConceptHead
from models.swin import IMAGENET_MEAN, IMAGENET_STD


class StudentDetector(nn.Module):
    def __init__(self, model_cfg: Dict, pretrained: bool = True) -> None:
        super().__init__()
        # full model without classifier/pooling: forward_features -> [B, 1280, 7, 7] incl. the pretrained conv_head
        self.backbone = timm.create_model(model_cfg.get("timm_name", "efficientnet_b0"), pretrained=pretrained,
                                          num_classes=0, global_pool="")
        k = int(model_cfg.get("num_concepts", 10))
        z_dim = int(model_cfg.get("z_dim", 512))
        self.concept_head = ConceptHead(1280, k)
        self.proj = nn.Sequential(nn.Linear(1280, z_dim), nn.LayerNorm(z_dim))
        self.decision = ConceptDecision(k, z_dim)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor, alpha: float = 0.0, off: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        feat = self.backbone.forward_features((x - self.mean) / self.std)  # [B, 1280, 7, 7]
        z = self.proj(feat.mean(dim=(2, 3)))
        ch = self.concept_head(feat)
        dec = self.decision(ch["concepts"], z, off)
        return {"z": z, "maps": ch["maps"], "concepts": ch["concepts"], "w": self.decision.w, **dec}

    def param_groups(self, lr: Dict[str, float]) -> List[Dict]:
        return [
            {"params": list(self.backbone.parameters()), "lr": lr.get("backbone", lr["heads"]), "name": "backbone"},
            {"params": list(self.concept_head.parameters()) + list(self.proj.parameters())
             + list(self.decision.parameters()), "lr": lr["heads"], "name": "heads"},
        ]
