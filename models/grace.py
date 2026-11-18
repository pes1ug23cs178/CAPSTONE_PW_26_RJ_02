"""GRACE teacher and its ablations, all from one configurable class.

  backbone: CLIP ViT-L/14 + LoRA (M2)             -> cls [B,1024], patches [B,1024,16,16]
  freq_branch: none | dct | fft (M3)              -> freq [B,256]
  fusion: none (cls projection) | gated | concat  -> z [B,512]
  head: concept (M6 + M7) | linear (A3)           -> logit [B]
  domain: DANN head on z (M5)                      -> domain_logits [B,D]

G1 = concept head, no freq. G2 = + dct + gated. G2-fft = + fft + gated. G3 = G2 + domain. A3 = G3 with head linear.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.clip_lora import ClipLoraBackbone
from models.concepts import ConceptDecision, ConceptHead
from models.domain import DomainHead
from models.frequency import build_freq_branch
from models.fusion import ClsProjection, ConcatFusion, GatedFusion


class GraceDetector(nn.Module):
    def __init__(self, model_cfg: Dict, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = model_cfg
        self.backbone = ClipLoraBackbone(model_cfg.get("lora"), pretrained, bool(model_cfg.get("grad_checkpointing")),
                                         frozen=model_cfg.get("backbone") == "clip_vitl14_frozen")
        dim = self.backbone.dim
        z_dim = int(model_cfg.get("z_dim", 512))
        self.freq = build_freq_branch(model_cfg.get("freq_branch", "none"))
        fusion = model_cfg.get("fusion", "none")
        if self.freq is None:
            self.fusion = ClsProjection(dim, z_dim)
        elif fusion == "gated":
            self.fusion = GatedFusion(dim, 256, z_dim)
        elif fusion == "concat":
            self.fusion = ConcatFusion(dim, 256, z_dim)
        else:
            raise ValueError(f"freq_branch set but fusion={fusion!r}")
        self.head_kind = model_cfg.get("head", "concept")
        k = int(model_cfg.get("num_concepts", 10))
        if self.head_kind == "concept":
            self.concept_head = ConceptHead(dim, k)
            self.decision = ConceptDecision(k, z_dim)
        elif self.head_kind == "linear":
            self.linear_head = nn.Linear(z_dim, 1)
        else:
            raise ValueError(f"Unknown head {self.head_kind!r}")
        dom = model_cfg.get("domain") or {}
        self.domain_enabled = bool(dom.get("enabled"))
        # domain.input: "z" (runbook G3: the fused vector) or "patches" (mean-pooled CLIP patch tokens, the features the
        # concept head reads). On z the DANN/MMD terms were satisfied by emptying z (G3 gate -> 0, A3 collapse), so they
        # never touched the features the decision uses; on patches they act on the decision path itself.
        self.domain_input = dom.get("input", "z")
        if self.domain_enabled:
            in_dim = self.backbone.dim if self.domain_input == "patches" else z_dim
            self.domain_head = DomainHead(in_dim, int(dom["num_domains"]), hidden=256, dropout=0.2)

    def forward(self, x: torch.Tensor, alpha: float = 0.0, off: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        f = self.backbone(x)
        freq = self.freq(x) if self.freq is not None else None
        fused = self.fusion(f["cls"], freq)
        z = fused["z"]
        out: Dict[str, torch.Tensor] = {"z": z, "cls": f["cls"]}
        if "gate_mean" in fused:
            out["gate_mean"] = fused["gate_mean"]
            out["gate"] = fused["gate"]
        if self.head_kind == "concept":
            ch = self.concept_head(f["patches"])
            dec = self.decision(ch["concepts"], z, off)
            out.update(maps=ch["maps"], concepts=ch["concepts"], w=self.decision.w, **dec)
        else:
            out["logit"] = self.linear_head(z.float()).squeeze(-1)
        if self.domain_enabled:
            # patch tokens already pass CLIP's final LayerNorm (frozen), so their scale cannot be shrunk by the adversary
            dfeat = f["patches"].mean(dim=(2, 3)) if self.domain_input == "patches" else z
            out["domain_feat"] = dfeat
            out["domain_logits"] = self.domain_head(dfeat, alpha)
        return out

    def param_groups(self, lr: Dict[str, float]) -> List[Dict]:
        heads: List[nn.Parameter] = []
        for name in ("concept_head", "decision", "linear_head", "domain_head"):
            if hasattr(self, name):
                heads += list(getattr(self, name).parameters())
        groups = [
            {"params": list(self.fusion.parameters()), "lr": lr.get("fusion", lr["heads"]), "name": "fusion"},
            {"params": heads, "lr": lr["heads"], "name": "heads"},
        ]
        if self.backbone.use_lora:
            groups.insert(0, {"params": self.backbone.lora_parameters(), "lr": lr["lora"], "name": "lora"})
        if self.freq is not None:
            groups.append({"params": list(self.freq.parameters()), "lr": lr.get("freq", 2e-3), "name": "freq"})
        return groups
