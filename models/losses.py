"""Training losses. Each term is returned separately so it can be logged.

Base: BCEWithLogits(logit, label) (equals 2-class cross-entropy for two-logit heads).
Domain (when model.domain is enabled): dann_weight * CE(domain_logits, domain_id)
                                       + mmd_weight * MMD over domains present in the batch.
Concept models add their own terms on Day 3 (concept, map, w_l1, residual).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from models.domain import pairwise_mmd

FAKE_DOMAINS = [1, 2, 3, 4]


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE averaged only where mask == 1; 0 when the batch has no valid entries."""
    mask = mask.float()
    if mask.sum() <= 0:
        return logits.new_zeros(())
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return (loss * mask).sum() / mask.sum()


def compute_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
    label = batch["label"]
    terms: Dict[str, torch.Tensor] = {"bce": F.binary_cross_entropy_with_logits(out["logit"].float(), label.float())}
    weights: Dict[str, float] = {"bce": float(cfg.get("loss", {}).get("bce", 1.0))}

    lw = cfg.get("loss", {})
    if "concepts" in out:  # concept-bottleneck models (G1, G2, G3, students S0/S3)
        terms["concept"] = masked_bce(out["concepts"], batch["concepts"], batch["concept_mask"])
        weights["concept"] = float(lw.get("concept", 0.5))
        bmask = batch["boundary_mask"].float()[:, None, None].expand_as(batch["boundary"])
        map_logits = out["maps"][:, 0].float()
        if map_logits.shape[-1] != batch["boundary"].shape[-1]:
            target = F.adaptive_avg_pool2d(batch["boundary"][:, None], map_logits.shape[-2:])[:, 0]
            target = target / target.flatten(1).amax(1).clamp(min=1e-6)[:, None, None]
            bmask = batch["boundary_mask"].float()[:, None, None].expand_as(target)
        else:
            target = batch["boundary"]
        terms["map"] = masked_bce(map_logits, target, bmask)
        weights["map"] = float(lw.get("map", 0.5))
        terms["w_l1"] = out["w"].abs().sum()
        weights["w_l1"] = float(lw.get("w_l1", 1e-4))
        terms["residual"] = out["residual"].float().pow(2).mean()
        weights["residual"] = float(lw.get("residual", 1e-3))

    dom = cfg["model"].get("domain") or {}
    if dom.get("enabled") and "domain_logits" in out:
        d = batch["domain_id"]
        valid = d >= 0
        if valid.any():
            terms["domain_ce"] = F.cross_entropy(out["domain_logits"][valid].float(), d[valid])
            weights["domain_ce"] = float(dom.get("dann_weight", 0.1))
            only = FAKE_DOMAINS if dom.get("mmd_on", "all") == "fake" else None
            terms["mmd"] = pairwise_mmd(out["z"][valid].float(), d[valid], only=only)
            weights["mmd"] = float(dom.get("mmd_weight", 0.1))

    total = sum(weights[k] * v for k, v in terms.items())
    logs = {k: float(v.detach()) for k, v in terms.items()}
    if "domain_logits" in out and "domain_ce" in terms:
        d = batch["domain_id"]
        valid = d >= 0
        logs["domain_acc"] = float((out["domain_logits"][valid].argmax(1) == d[valid]).float().mean())
    return total, logs
