"""Exponential moving average of the trainable weights (frozen weights never change, so they are
copied once and skipped). Buffers (e.g. BatchNorm statistics) are copied every update."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = deepcopy(model).eval()
        self.decay = decay
        src = dict(model.named_parameters())
        self.pairs = []
        for name, ema_p in self.model.named_parameters():
            ema_p.requires_grad_(False)
            if src[name].requires_grad:
                self.pairs.append((ema_p, src[name]))
        self.buffer_pairs = list(zip(self.model.buffers(), model.buffers()))

    @torch.no_grad()
    def update(self, model: nn.Module = None) -> None:
        if self.pairs:
            ema_ps, ps = zip(*self.pairs)
            torch._foreach_lerp_(list(ema_ps), [p.detach() for p in ps], 1.0 - self.decay)
        for ema_b, b in self.buffer_pairs:
            ema_b.copy_(b)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state, strict: bool = True):
        return self.model.load_state_dict(state, strict=strict)
