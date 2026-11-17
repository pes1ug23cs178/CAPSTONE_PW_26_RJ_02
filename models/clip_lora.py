"""M2: CLIP ViT-L/14 (OpenAI weights via timm) with LoRA on every attention qkv projection.

All original weights are frozen; LoRA r=16, alpha=32, dropout 0.1 on `qkv` (~1.6 M trainable).
forward(x in [0, 1]) normalizes with CLIP mean/std and returns
  cls     [B, 1024]          L2-normalized token 0
  patches [B, 1024, 16, 16]  tokens 1..256
Heads are added by the detector classes below (C1 = linear head on cls).
"""
from __future__ import annotations

from typing import Dict, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipLoraBackbone(nn.Module):
    def __init__(self, lora_cfg: Dict | None = None, pretrained: bool = True, grad_checkpointing: bool = False,
                 timm_name: str = "vit_large_patch14_clip_224.openai", frozen: bool = False) -> None:
        super().__init__()
        vit = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        for p in vit.parameters():
            p.requires_grad_(False)
        if grad_checkpointing:
            vit.set_grad_checkpointing(True)
        self.use_lora = not frozen
        if self.use_lora:
            from peft import LoraConfig, inject_adapter_in_model
            lc = lora_cfg or {}
            cfg = LoraConfig(r=int(lc.get("r", 16)), lora_alpha=int(lc.get("alpha", 32)),
                             lora_dropout=float(lc.get("dropout", 0.1)), target_modules=["qkv"], bias="none")
            vit = inject_adapter_in_model(cfg, vit)
            for n, p in vit.named_parameters():
                p.requires_grad_("lora_" in n)
        self.vit = vit
        self.dim = vit.num_features  # 1024
        self.grid = vit.patch_embed.grid_size  # (16, 16)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = (x - self.mean) / self.std
        tokens = self.vit.forward_features(x)  # [B, 1 + 256, 1024], final LayerNorm applied
        cls = F.normalize(tokens[:, 0], dim=-1)
        b, n, c = tokens.shape
        h, w = self.grid
        patches = tokens[:, 1:].transpose(1, 2).reshape(b, c, h, w)
        return {"cls": cls, "patches": patches, "cls_raw": tokens[:, 0]}

    def lora_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.vit.named_parameters() if "lora_" in n]


class ClipLinearDetector(nn.Module):
    """C1: CLIP + LoRA, Linear(1024 -> 2) on the normalized cls token."""

    def __init__(self, model_cfg: Dict, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = ClipLoraBackbone(model_cfg.get("lora"), pretrained, bool(model_cfg.get("grad_checkpointing")),
                                         frozen=model_cfg.get("backbone") == "clip_vitl14_frozen")
        self.head = nn.Linear(self.backbone.dim, 2)

    def forward(self, x: torch.Tensor, alpha: float = 0.0) -> Dict[str, torch.Tensor]:
        f = self.backbone(x)
        logits2 = self.head(f["cls"])
        return {"logits2": logits2, "logit": logits2[:, 1] - logits2[:, 0], "z": f["cls"], "cls": f["cls"]}

    def param_groups(self, lr: Dict[str, float]) -> List[Dict]:
        groups = [{"params": list(self.head.parameters()), "lr": lr["heads"], "name": "heads"}]
        if self.backbone.use_lora:
            groups.insert(0, {"params": self.backbone.lora_parameters(), "lr": lr["lora"], "name": "lora"})
        return groups
