"""T1.8 (optional): evaluate an OLD leaky-split checkpoint on the new leak-free test splits.

The old train_swin_deepfake.py DeepfakeDetector used the same attribute names as models.swin
(backbone, feat_norm, head), so its EMA weights load directly. Threshold: chosen on ffpp_val.

python scripts/eval_old_ckpt.py --ckpt outputs_ffpp_v2_20260423_172121/best_swin_deepfake.pth --name old_ffpp_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import choose_threshold, evaluate_split, predict, video_table  # noqa: E402
from models.swin import SwinDetector  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck.get("ema_state_dict") or ck["model_state_dict"]
    old_args = ck.get("args", {})
    model = SwinDetector(old_args.get("model_name", "swin_base_patch4_window7_224"),
                         feature_norm=old_args.get("feature_norm") or "instance", pretrained=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith(("domain_clf", "concept_probes", "audio_stub"))]
    if missing or unexpected:
        sys.exit(f"Key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model = model.to(device).eval()
    out_dir = REPO_ROOT / "runs" / f"_old_{args.name}"
    pred = predict(model, REPO_ROOT / "splits/ffpp_val.csv", device)
    vt = video_table(pred)
    thr = choose_threshold(vt["label"].to_numpy(), vt["prob"].to_numpy())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_threshold.json").write_text(json.dumps(thr, indent=2))
    summary = {"ckpt": args.ckpt, "old_reported": {k: ck.get(k) for k in ("best_val_acc", "best_epoch")}}
    for split in ("ffpp_val", "ffpp_test", "dfd_test", "celebdf_test"):
        m = evaluate_split(model, REPO_ROOT / f"splits/{split}.csv", out_dir, out_dir.name, device, thr["threshold"])
        summary[split] = {k: m[k] for k in ("video_auc", "frame_auc", "eer", "acc")}
        print(split, summary[split], flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
