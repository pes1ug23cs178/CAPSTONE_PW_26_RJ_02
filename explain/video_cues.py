"""T4.2: video-level cues from per-frame outputs (frames in temporal order).

frame_scores     per-frame fake probability (for the app's timeline)
temporal_flicker mean over concepts of the std of that concept's score across frames
identity_drift   mean cosine distance between consecutive frames' CLIP cls features
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def temporal_flicker(concepts: np.ndarray) -> float:
    """concepts: [T, K] sigmoid scores."""
    if concepts is None or len(concepts) < 2:
        return 0.0
    return float(np.std(concepts, axis=0).mean())


def identity_drift(cls: np.ndarray) -> float:
    """cls: [T, D] features (any norm)."""
    if cls is None or len(cls) < 2:
        return 0.0
    f = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
    return float((1.0 - (f[1:] * f[:-1]).sum(1)).mean())


def video_cues(pred: pd.DataFrame, cls_by_path: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """One row per video: label, prob, flicker, drift, frame_scores (as a ';'-joined string)."""
    concept_cols = [f"c{k}" for k in range(1, 11) if f"c{k}" in pred]
    rows = []
    for vid, g in pred.sort_values("frame_path").groupby("video_id", sort=False):
        cls = np.stack([cls_by_path[p] for p in g["frame_path"]]) if cls_by_path else None
        rows.append({
            "video_id": vid, "label": int(g["label"].iloc[0]), "prob": float(g["prob"].mean()),
            "temporal_flicker": temporal_flicker(g[concept_cols].to_numpy()) if concept_cols else None,
            "identity_drift": identity_drift(cls) if cls is not None else None,
            "frame_scores": ";".join(f"{p:.4f}" for p in g["prob"]),
        })
    return pd.DataFrame(rows)


def cue_summary(cues: pd.DataFrame) -> Dict[str, Optional[float]]:
    out = {}
    for cue in ("temporal_flicker", "identity_drift"):
        if cue in cues and cues[cue].notna().any():
            for lab, name in ((0, "real"), (1, "fake")):
                out[f"{cue}_{name}"] = float(cues.loc[cues["label"] == lab, cue].mean())
    return out
