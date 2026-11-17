"""M4: fusion of the CLIP cls feature with the frequency feature into z (512-d)."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """p_c = Lin(cls), p_f = Lin(freq), g = sigmoid(Lin([p_c; p_f])), z = LN(g*p_c + (1-g)*p_f)."""

    def __init__(self, cls_dim: int = 1024, freq_dim: int = 256, out: int = 512) -> None:
        super().__init__()
        self.pc = nn.Linear(cls_dim, out)
        self.pf = nn.Linear(freq_dim, out)
        self.gate = nn.Linear(2 * out, out)
        self.norm = nn.LayerNorm(out)

    def forward(self, cls: torch.Tensor, freq: torch.Tensor) -> Dict[str, torch.Tensor]:
        p_c, p_f = self.pc(cls), self.pf(freq)
        g = torch.sigmoid(self.gate(torch.cat([p_c, p_f], dim=-1)))
        return {"z": self.norm(g * p_c + (1 - g) * p_f), "gate_mean": g.mean(), "gate": g.mean(-1)}


class ConcatFusion(nn.Module):
    def __init__(self, cls_dim: int = 1024, freq_dim: int = 256, out: int = 512) -> None:
        super().__init__()
        self.proj = nn.Linear(cls_dim + freq_dim, out)
        self.norm = nn.LayerNorm(out)

    def forward(self, cls: torch.Tensor, freq: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"z": self.norm(self.proj(torch.cat([cls, freq], dim=-1)))}


class ClsProjection(nn.Module):
    """No frequency branch (G1): z = LayerNorm(Linear(1024 -> 512)(cls))."""

    def __init__(self, cls_dim: int = 1024, out: int = 512) -> None:
        super().__init__()
        self.proj = nn.Linear(cls_dim, out)
        self.norm = nn.LayerNorm(out)

    def forward(self, cls: torch.Tensor, freq=None) -> Dict[str, torch.Tensor]:
        return {"z": self.norm(self.proj(cls))}
