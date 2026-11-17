"""T6.1: explanation functions. All maps are 224 x 224 numpy arrays scaled to [0, 1].

concept_maps(model, image)        sigmoid of the 10 concept maps, bilinearly upsampled (teacher or student)
contributions(model, image)       per-concept w_k * sigmoid(c_k), the residual r(z) and the bias
gradcam(model, image)             Grad-CAM for Swin (B1/B2); Grad-CAM++ for the EfficientNet student
attention_rollout(model, image)   CLIP (C1): mean over heads, + identity, product over 24 layers, CLS row
intervene(model, image, off)      logit with the listed concepts' sigmoid values set to 0
`image` is a [1, 3, 224, 224] float tensor in [0, 1] on the model's device.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _norm01(m: torch.Tensor) -> np.ndarray:
    m = m.float()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.detach().cpu().numpy()


def _up(m: torch.Tensor, size: int = 224) -> torch.Tensor:
    return F.interpolate(m[None, None].float(), size=size, mode="bilinear", align_corners=False)[0, 0]


@torch.no_grad()
def concept_maps(model, image: torch.Tensor) -> np.ndarray:
    out = model(image)
    maps = torch.sigmoid(out["maps"][0].float())  # [10, h, w]
    up = F.interpolate(maps[None], size=image.shape[-1], mode="bilinear", align_corners=False)[0]
    return up.clamp(0, 1).cpu().numpy()


@torch.no_grad()
def contributions(model, image: torch.Tensor) -> Dict[str, object]:
    out = model(image)
    contrib = out["contributions"][0].float().cpu().numpy()
    residual = float(out["residual"][0])
    bias = float(model.decision.b)
    logit = float(out["logit"][0])
    total_abs = np.abs(contrib).sum() + abs(residual) + 1e-8
    return {"contributions": contrib, "residual": residual, "bias": bias, "logit": logit,
            "prob": float(torch.sigmoid(out["logit"][0].float())), "concept_share": float(np.abs(contrib).sum() / total_abs),
            "concept_scores": torch.sigmoid(out["concepts"][0].float()).cpu().numpy()}


@torch.no_grad()
def intervene(model, image: torch.Tensor, off: Sequence[int]) -> float:
    mask = torch.zeros(model.decision.w.shape[0], device=image.device)
    if len(off):
        mask[list(off)] = 1.0
    return float(model(image, off=mask)["logit"][0])


def gradcam(model, image: torch.Tensor) -> np.ndarray:
    """Grad-CAM (Swin) or Grad-CAM++ (EfficientNet student) for the fake logit."""
    if hasattr(model, "gradcam_target_layer"):  # Swin
        from explain.gradcam import SwinGradCAM
        cam = SwinGradCAM(model, model.gradcam_target_layer())
        try:
            m, _ = cam.generate(image)
        finally:
            cam.release()
        return m
    layer = model.backbone.bn2  # last conv block output of EfficientNet-B0 (1280 x 7 x 7)
    store = {}
    h1 = layer.register_forward_hook(lambda _m, _i, o: store.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda _m, _gi, go: store.__setitem__("g", go[0]))
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = image.clone().requires_grad_(True)
            model(x)["logit"].sum().backward()
    finally:
        h1.remove(); h2.remove()
    a, g = store["a"][0].float(), store["g"][0].float()
    g2, g3 = g.pow(2), g.pow(3)
    alpha = g2 / (2 * g2 + (a.sum(dim=(1, 2), keepdim=True) * g3) + 1e-8)
    w = (alpha * F.relu(g)).sum(dim=(1, 2))
    cam = F.relu((w[:, None, None] * a).sum(0))
    return _norm01(_up(cam, image.shape[-1]))


def attention_rollout(model, image: torch.Tensor) -> np.ndarray:
    vit = model.backbone.vit
    attns: List[torch.Tensor] = []
    hooks = []
    for blk in vit.blocks:
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda _m, _i, o: attns.append(o.detach().float())))
    try:
        with torch.no_grad():
            model(image)
    finally:
        for h in hooks:
            h.remove()
        for blk in vit.blocks:
            blk.attn.fused_attn = True
    n = attns[0].shape[-1]
    rollout = torch.eye(n, device=image.device)
    for a in attns:
        a = a[0].mean(0) + torch.eye(n, device=image.device)
        a = a / a.sum(-1, keepdim=True)
        rollout = a @ rollout
    cls = rollout[0, 1:]
    s = int(np.sqrt(cls.numel()))
    return _norm01(_up(cls.reshape(s, s), image.shape[-1]))
