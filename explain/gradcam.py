"""Grad-CAM for Swin (ported from train_swin_deepfake.py; no retained graph)."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = [
            target_layer.register_forward_hook(self._fwd_hook),
            target_layer.register_full_backward_hook(self._bwd_hook),
        ]

    def _fwd_hook(self, _m, _i, output):
        self.activations = (output[0] if isinstance(output, tuple) else output).detach()

    def _bwd_hook(self, _m, _gi, grad_output):
        self.gradients = grad_output[0].detach()

    @staticmethod
    def _to_nchw(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:  # timm Swin stages are NHWC
            return t.permute(0, 3, 1, 2) if t.shape[-1] > t.shape[1] else t
        if t.dim() == 3:
            b, n, c = t.shape
            s = int(math.sqrt(n))
            return t.transpose(1, 2).reshape(b, c, s, s)
        raise ValueError(f"Unsupported activation shape {tuple(t.shape)}")

    def generate(self, image01: torch.Tensor) -> Tuple[np.ndarray, float]:
        """image01: [1,3,H,W] in [0,1]. Returns (cam HxW in [0,1], fake probability)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            image01 = image01.clone().requires_grad_(True)
            out = self.model(image01)
            logit = out["logit"]
            logit.sum().backward()
        acts = self._to_nchw(self.activations).float()
        grads = self._to_nchw(self.gradients).float()
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=image01.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), float(torch.sigmoid(logit.detach()).item())

    def release(self) -> None:
        for h in self._hooks:
            h.remove()
