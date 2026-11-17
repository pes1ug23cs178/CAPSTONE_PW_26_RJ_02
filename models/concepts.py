"""M6 + M7: concept head (patch grid -> 10 concept maps) and the concept-bottleneck decision layer."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def topk_mean(maps: torch.Tensor, frac: float = 0.10) -> torch.Tensor:
    """Mean of the top `frac` cells of each map: [B, K, H, W] -> [B, K]."""
    flat = maps.flatten(2)
    k = max(1, int(round(flat.shape[-1] * frac)))
    return flat.topk(k, dim=-1).values.mean(-1)


class ConceptHead(nn.Module):
    """1x1 conv from the patch grid to K concept maps (logits); score c_k = top-10% mean of map k."""

    def __init__(self, in_ch: int = 1024, k: int = 10, top_frac: float = 0.10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, k, kernel_size=1)
        self.top_frac = top_frac

    def forward(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        maps = self.conv(patches)
        return {"maps": maps, "concepts": topk_mean(maps, self.top_frac)}


class ConceptDecision(nn.Module):
    """logit = w . sigmoid(c) + b + r(z); w starts at 1, r's last layer starts at 0 (concept-only start)."""

    def __init__(self, k: int = 10, z_dim: int = 512) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(k))
        self.b = nn.Parameter(torch.zeros(1))
        self.r = nn.Sequential(nn.Linear(z_dim, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.r[-1].weight)
        nn.init.zeros_(self.r[-1].bias)

    def forward(self, concepts: torch.Tensor, z: torch.Tensor, off: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        s = torch.sigmoid(concepts.float())
        if off is not None:  # intervention: switch listed concepts off (sigmoid value -> 0)
            s = s * (1 - off.float())
        contrib = s * self.w
        residual = self.r(z.float()).squeeze(-1)
        concept_part = contrib.sum(-1)
        return {"logit": concept_part + self.b + residual, "contributions": contrib,
                "concept_part": concept_part, "residual": residual}
