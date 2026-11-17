"""Domain alignment pieces, ported from train_swin_deepfake.py (DANN + multi-kernel MMD)."""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Identity forward; multiplies the gradient by -alpha backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


def compute_domain_alpha(progress: float, gamma: float = 10.0) -> float:
    """DANN schedule: 2 / (1 + exp(-gamma * p)) - 1, p = training progress in [0, 1]."""
    p = min(max(progress, 0.0), 1.0)
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


class DomainHead(nn.Module):
    """GRL -> Linear -> ReLU -> Dropout -> Linear(num_domains).

    Callers must feed the SAME representation the real/fake classifier sees (for Swin: after the head's
    LayerNorm). In B2 / B2-fake the domain head read the features before that LayerNorm, so the adversary could
    confuse it by shrinking the single InstanceNorm gamma (5e-4 by epoch 2, then through 0) while the
    scale-invariant classifier gave no resisting gradient, until the real/fake signal flipped and vanished.
    """

    def __init__(self, in_features: int, num_domains: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, features: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.net(grad_reverse(features, alpha))


def mmd_loss(source: torch.Tensor, target: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    """Maximum Mean Discrepancy with a multi-kernel RBF (median-heuristic bandwidth)."""
    n_s, n_t = source.size(0), target.size(0)
    if n_s == 0 or n_t == 0:
        return source.new_zeros(())
    total = torch.cat([source, target], dim=0).float()
    sq_dist = torch.cdist(total, total, p=2).pow(2)
    bw = sq_dist.detach().median() / math.log(n_s + n_t + 1)
    bw = torch.clamp(bw, min=1e-6)
    scales = [bw * (kernel_mul ** (i - kernel_num // 2)) for i in range(kernel_num)]
    k = sum(torch.exp(-sq_dist / (2 * s)) for s in scales)
    loss = k[:n_s, :n_s].mean() + k[n_s:, n_s:].mean() - 2.0 * k[:n_s, n_s:].mean()
    return loss.clamp(min=0.0)


def pairwise_mmd(features: torch.Tensor, domains: torch.Tensor, only: Optional[List[int]] = None) -> torch.Tensor:
    """Mean MMD over every pair of domains present in the batch (optionally restricted to `only`)."""
    present = [int(d) for d in domains.unique().tolist() if only is None or int(d) in only]
    losses = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a = features[domains == present[i]]
            b = features[domains == present[j]]
            if a.size(0) > 1 and b.size(0) > 1:
                losses.append(mmd_loss(a, b))
    if not losses:
        return features.new_zeros(())
    return torch.stack(losses).mean()
