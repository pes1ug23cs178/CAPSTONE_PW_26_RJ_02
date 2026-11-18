"""T5.4: teacher-student explanation agreement -> results/agreement.csv.

On 2,000 crops from celebdf_test and 2,000 from ffpp_test, clean and jpeg:30:
- Spearman correlation between the teacher's and the student's 10 concept scores (per crop, averaged)
- IoU between the two models' top-concept maps (upsampled to 224, each thresholded at its own top 10%)
Students: S0 (recipe labels only) and S3 (distilled). S2's concept head is never trained, so it is skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import FaceFrameDataset  # noqa: E402
from evaluate import degrade_args, load_model_from_ckpt  # noqa: E402
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


@torch.no_grad()
def outputs(model, csv, degrade, device):
    ds = FaceFrameDataset(csv, train=False, **degrade_args(csv, degrade))
    cs, ms, raw = [], [], []
    for b in DataLoader(ds, batch_size=128, num_workers=6):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(b["image"].to(device))
        cs.append(torch.sigmoid(o["concepts"].float()).cpu())
        top = o["concepts"].float().argmax(1)
        m = torch.sigmoid(o["maps"].float())[torch.arange(len(top)), top][:, None]
        ms.append(F.interpolate(m, size=224, mode="bilinear", align_corners=False)[:, 0].cpu())
        raw.append(torch.sigmoid(o["maps"].float()).cpu())  # all K maps at native resolution (16x16 / 7x7)
    return torch.cat(cs).numpy(), torch.cat(ms), torch.cat(raw)


def maps_at(raw: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Map of concept idx[i] for each image, upsampled to 224."""
    m = raw[torch.arange(len(idx)), idx][:, None]
    return F.interpolate(m, size=224, mode="bilinear", align_corners=False)[:, 0]


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a & b).sum(1).float() / (a | b).sum(1).float()).mean().item()


def top10_mask(m: torch.Tensor) -> torch.Tensor:
    k = int(0.1 * m[0].numel())
    thr = m.flatten(1).topk(k, dim=1).values[:, -1:]
    return m.flatten(1) >= thr


def main() -> None:
    device = torch.device("cuda")
    teacher_run = REPO_ROOT / "runs" / json.loads((REPO_ROOT / "cache/teacher/teacher.json").read_text())["run"]
    teacher, _ = load_model_from_ckpt(str(teacher_run / "best.pth"), device)
    students = {r.name.rsplit("_s", 1)[0]: r for r in resolve_runs(["s0_student", "s3_kd_full"], [42])}
    tmp_dir = REPO_ROOT / "cache"
    rows = []
    for split in ("celebdf_test", "ffpp_test"):
        df = pd.read_csv(REPO_ROOT / "splits" / f"{split}.csv", dtype=str).sample(n=2000, random_state=0)
        csv = tmp_dir / f"_agree_{split}.csv"
        df.to_csv(csv, index=False)
        for degrade in (None, "jpeg:30"):
            tc, tm, traw = outputs(teacher, csv, degrade, device)
            tmask = top10_mask(tm)
            ttop = torch.from_numpy(tc.argmax(1))
            tmask_same = top10_mask(maps_at(traw, ttop))
            for name, run in students.items():
                model, _ = load_model_from_ckpt(str(run / "best.pth"), device)
                sc, sm, sraw = outputs(model, csv, degrade, device)
                rho = np.nanmean([spearmanr(a, b).statistic for a, b in zip(tc, sc)])
                smask = top10_mask(sm)
                # top_map_iou: each model's map at its OWN top concept (mixes concept choice with map location);
                # map_iou_same_concept: both maps at the TEACHER's top concept (map location only);
                # top_concept_agree: share of images where both pick the same top concept.
                same = top10_mask(maps_at(sraw, ttop))
                rows.append({"student": name, "split": split, "degrade": degrade or "clean",
                             "spearman_concepts": round(float(rho), 4), "top_map_iou": round(iou(tmask, smask), 4),
                             "map_iou_same_concept": round(iou(tmask_same, same), 4),
                             "top_concept_agree": round(float((sc.argmax(1) == tc.argmax(1)).mean()), 4)})
                print(rows[-1], flush=True)
                del model
                torch.cuda.empty_cache()
        csv.unlink()
    out = REPO_ROOT / "results" / "agreement.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
