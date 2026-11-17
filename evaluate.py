"""Evaluate a checkpoint on one split CSV.

python evaluate.py --ckpt runs/<run>/best.pth --split splits/celebdf_test.csv
                   [--degrade jpeg:50] [--threshold-from runs/<run>/val_threshold.json]

Writes into the run folder (or --out-dir):
  metrics_<split>[_<degrade>].json, predictions_<split>[_<degrade>].csv, roc_<split>[_<degrade>].png

Video score = mean fake probability over a video's crops. Thresholded metrics (acc, precision,
recall, F1) are computed at video level with the threshold chosen on FF++ validation
(maximum balanced accuracy over videos).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import FaceFrameDataset  # noqa: E402
from models.build import build_model  # noqa: E402


# ----------------------------------------------------------------------------- metrics

def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def video_table(pred: pd.DataFrame) -> pd.DataFrame:
    return pred.groupby("video_id").agg(label=("label", "first"), prob=("prob", "mean")).reset_index()


def choose_threshold(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Threshold maximizing balanced accuracy (searched over all observed scores)."""
    best_t, best_bacc = 0.5, -1.0
    for t in np.unique(probs):
        pred = probs >= t
        tpr = (pred & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        tnr = (~pred & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        bacc = (tpr + tnr) / 2
        if bacc > best_bacc:
            best_t, best_bacc = float(t), float(bacc)
    return {"threshold": best_t, "val_balanced_acc": best_bacc}


def compute_metrics(pred: pd.DataFrame, threshold: Optional[float]) -> Dict[str, float]:
    vt = video_table(pred)
    out = {
        "n_videos": int(len(vt)), "n_frames": int(len(pred)),
        "video_auc": safe_auc(vt["label"].to_numpy(), vt["prob"].to_numpy()),
        "frame_auc": safe_auc(pred["label"].to_numpy(), pred["prob"].to_numpy()),
        "eer": eer(vt["label"].to_numpy(), vt["prob"].to_numpy()),
        "frame_eer": eer(pred["label"].to_numpy(), pred["prob"].to_numpy()),
        "threshold": threshold,
    }
    if threshold is not None:
        y = vt["label"].to_numpy().astype(int)
        p = (vt["prob"].to_numpy() >= threshold).astype(int)
        tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
        fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.update({
            "acc": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        })
    return out


# ----------------------------------------------------------------------------- inference

def degrade_args(csv_path: str, spec: Optional[str]) -> Dict:
    """Dataset kwargs for an evaluation degradation (T4.3)."""
    from data.degrade import build_h264_cache, image_degrade_fn, parse
    from utils.config import load_paths
    kind, val = parse(spec)
    if kind == "h264":
        paths = load_paths()
        return {"path_map": build_h264_cache(str(csv_path), int(val), paths["data_root"], paths["cache_root"])}
    return {"degrade_fn": image_degrade_fn(spec)}


@torch.no_grad()
def predict(model: torch.nn.Module, csv_path: str, device: torch.device, batch_size: int = 128,
            num_workers: int = 6, limit: Optional[int] = None, degrade: Optional[str] = None,
            random_scores: bool = False) -> pd.DataFrame:
    ds = FaceFrameDataset(csv_path, train=False, limit=limit, **degrade_args(csv_path, degrade))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=device.type == "cuda", persistent_workers=False)
    rows = []
    rng = np.random.default_rng(0)
    if model is not None:
        model.eval()
    for batch in dl:
        extra: Dict[str, np.ndarray] = {}
        if random_scores:
            probs = rng.random(len(batch["label"]))
        else:
            x = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(x)
            logit = out["logit"].float()
            probs = torch.sigmoid(logit).cpu().numpy()
            extra["logit"] = logit.cpu().numpy()
            if "concepts" in out:
                cs = torch.sigmoid(out["concepts"].float()).cpu().numpy()
                for k in range(cs.shape[1]):
                    extra[f"c{k + 1}"] = cs[:, k]
                extra["concept_part"] = out["concept_part"].float().cpu().numpy()
            if "gate" in out:
                extra["gate"] = out["gate"].float().cpu().numpy()
        for j in range(len(probs)):
            row = {"video_id": batch["video_id"][j], "frame_path": batch["path"][j],
                   "label": int(batch["label"][j]), "method": batch["method"][j], "prob": float(probs[j])}
            row.update({k: float(v[j]) for k, v in extra.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def concept_stats(pred: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Concept share = Var(w . sigmoid(c)) / Var(logit) over the split; mean fusion gate."""
    out: Dict[str, Optional[float]] = {}
    if "concept_part" in pred and pred["logit"].var() > 0:
        out["concept_share"] = float(pred["concept_part"].var() / pred["logit"].var())
    if "gate" in pred:
        out["gate_mean"] = float(pred["gate"].mean())
    return out


def concept_auc(model, sbi_csv: Path, device: torch.device, num_workers: int = 6) -> Dict[str, float]:
    """Per-concept AUC on a fixed SBI set: positives = images whose recipe label for that concept is > 0."""
    labels = pd.read_csv(sbi_csv)
    pred = predict(model, str(sbi_csv), device, num_workers=num_workers)
    pred = pred.merge(labels[["frame_path"] + [f"c{k}" for k in range(1, 11)]], on="frame_path", suffixes=("", "_true"))
    out = {}
    for k in range(1, 11):
        y = (pred[f"c{k}_true"] > 0).astype(int).to_numpy()
        out[f"c{k}"] = safe_auc(y, pred[f"c{k}"].to_numpy())
    return out


# ----------------------------------------------------------------------------- checkpoint

def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> (torch.nn.Module, Dict):
    """Loads best.pth written by train.py and returns the model holding the EMA weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    trainable_only = bool(ckpt.get("trainable_only"))
    # trainable-only checkpoints (CLIP + LoRA) hold just the trained weights: rebuild the frozen
    # pretrained backbone first, then overlay them.
    model = build_model(cfg["model"], pretrained=trainable_only)
    state = ckpt.get("ema") or ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected[:5]}")
    if trainable_only:
        trained = {n for n, p in model.named_parameters() if p.requires_grad}
        not_loaded = sorted(trained & set(missing))
        if not_loaded:
            raise RuntimeError(f"Trainable weights missing from checkpoint: {not_loaded[:5]}")
    elif missing:
        raise RuntimeError(f"Checkpoint missing keys: {missing[:5]}")
    return model.to(device).eval(), ckpt


def plot_roc(pred: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vt = video_table(pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, (y, s) in {"video": (vt["label"], vt["prob"]), "frame": (pred["label"], pred["prob"])}.items():
        if y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y, s):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_split(model, csv_path: str, out_dir: Path, run_name: str, device: torch.device,
                   threshold: Optional[float], degrade: Optional[str] = None, limit: Optional[int] = None,
                   batch_size: int = 128, num_workers: int = 6, extra: Optional[Dict] = None,
                   random_scores: bool = False) -> Dict:
    split = Path(csv_path).stem
    tag = split + (f"_{degrade.replace(':', '')}" if degrade and degrade != "clean" else "")
    pred = predict(model, csv_path, device, batch_size, num_workers, limit, degrade, random_scores)
    metrics = {"run": run_name, "split": split, "degrade": degrade or "clean"}
    metrics.update(compute_metrics(pred, threshold))
    metrics.update(concept_stats(pred))
    metrics.setdefault("concept_share", None)
    metrics.setdefault("gate_mean", None)
    if extra:
        metrics.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / f"predictions_{tag}.csv", index=False)
    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    plot_roc(pred, out_dir / f"roc_{tag}.png", f"{run_name} | {tag}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="best.pth from train.py")
    ap.add_argument("--split", required=True)
    ap.add_argument("--degrade", default=None)
    ap.add_argument("--threshold-from", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="evaluate a random subset of N crops")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--random-scores", action="store_true", help="sanity check: random predictions")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, run_name, out_dir = None, "random", Path(args.out_dir or "runs/_eval_random")
    if not args.random_scores:
        model, ckpt = load_model_from_ckpt(args.ckpt, device)
        run_name = Path(args.ckpt).parent.name
        out_dir = Path(args.out_dir or Path(args.ckpt).parent)
    threshold = None
    thr_file = args.threshold_from or (str(Path(args.ckpt).parent / "val_threshold.json") if args.ckpt else None)
    if thr_file and Path(thr_file).exists():
        threshold = json.loads(Path(thr_file).read_text())["threshold"]
    m = evaluate_split(model, args.split, out_dir, run_name, device, threshold, args.degrade, args.limit,
                       args.batch_size, args.num_workers, random_scores=args.random_scores)
    print(json.dumps({k: v for k, v in m.items() if k != "confusion"}, indent=2))


if __name__ == "__main__":
    main()
