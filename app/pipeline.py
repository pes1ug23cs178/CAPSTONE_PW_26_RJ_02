"""Inference pipeline behind the demo app (T6.4). Kept free of Gradio so it can be tested.

Face crops: the project's Feb/Apr 2026 preprocessing code is not on this machine, so faces are found
with face-alignment's SFD detector, the crop is rotated so the eye centres are level, and resized to 224 x 224.
Margin: the runbook fallback says 1.3x the box, but on 40 training crops the crop side was 1.02x the SFD box
(IQR 0.95-1.07), so MARGIN = 1.02 matches the training distribution (measured 17 Nov 2026).

Engines:
  QuickEngine: S3 student via ONNX Runtime on the CPU (decision weights w, b read from a JSON beside the
               ONNX file, so contributions and interventions need no PyTorch model).
  DeepEngine:  G3 teacher on the GPU (PyTorch), plus CLIP cls features for identity drift and retrieval.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # face-alignment's torch.compile needs Triton (absent on Windows)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CONCEPT_LABELS = [
    "Blend strength", "Colour mismatch", "Blur in a region", "Over-sharpened region", "Resampling traces",
    "Mouth region edited", "Eye region edited", "Whole face swapped", "Misaligned geometry", "Irregular blend boundary",
]
LIMITATION = ("Trained on FaceForensics++ face swaps and reenactments; it may miss other kinds of fakes. "
              "A result is evidence, not proof.")
BANDS = [(0.35, "Likely real"), (0.65, "Inconclusive"), (1.01, "Likely fake")]


def band(p: float) -> str:
    for hi, name in BANDS:
        if p < hi:
            return name
    return BANDS[-1][1]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MARGIN = 1.02


class FaceCropper:
    def __init__(self, device: Optional[str] = None) -> None:
        import face_alignment
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=dev, face_detector="sfd")

    def crop(self, frame_rgb: np.ndarray, size: int = 224) -> Optional[np.ndarray]:
        h, w = frame_rgb.shape[:2]
        scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
        small = cv2.resize(frame_rgb, (int(w * scale), int(h * scale))) if scale < 1 else frame_rgb
        res = self.fa.get_landmarks_from_image(small, return_bboxes=True)
        if res is None or res[0] is None or len(res[0]) == 0:
            return None
        lms, _, boxes = res
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        i = int(np.argmax(areas))
        lm, box = lms[i] / scale, np.array(boxes[i][:4]) / scale
        le, re = lm[36:42].mean(0), lm[42:48].mean(0)
        angle = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        side = max(box[2] - box[0], box[3] - box[1]) * MARGIN
        m = cv2.getRotationMatrix2D((cx, cy), angle, size / side)
        m[0, 2] += size / 2 - cx
        m[1, 2] += size / 2 - cy
        return cv2.warpAffine(frame_rgb, m, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)


def sample_frames(video_path: str, n: int = 32) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = np.linspace(0, max(total - 1, 0), num=min(n, max(total, 1))).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


@dataclass
class FaceResult:
    crop: np.ndarray
    prob: float
    logit: float
    concepts: np.ndarray          # sigmoid scores [10]
    contributions: np.ndarray     # w_k * sigmoid(c_k) [10]
    residual: float
    maps: np.ndarray              # sigmoid maps [10, 224, 224]
    cls: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None


@dataclass
class Analysis:
    mode: str
    faces: List[FaceResult] = field(default_factory=list)
    n_frames: int = 0
    prob: float = float("nan")
    verdict: str = ""
    flicker: Optional[float] = None
    drift: Optional[float] = None
    message: str = ""


class QuickEngine:
    name = "Quick (S3 student, ONNX on CPU)"

    def __init__(self, onnx_path: Path, meta_path: Path, temperature: float = 1.0) -> None:
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        meta = json.loads(Path(meta_path).read_text())
        self.w, self.b, self.T = np.array(meta["w"], np.float32), float(meta["b"]), float(temperature)
        self.version = meta.get("run", Path(onnx_path).name)

    def run(self, crops: List[np.ndarray]) -> List[FaceResult]:
        x = np.stack(crops).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        logit, concepts, maps = self.sess.run(None, {"image": x})
        out = []
        for i in range(len(crops)):
            s = 1 / (1 + np.exp(-concepts[i]))
            contrib = self.w * s
            m = 1 / (1 + np.exp(-maps[i]))
            up = np.stack([cv2.resize(mm, (224, 224), interpolation=cv2.INTER_LINEAR) for mm in m])
            out.append(FaceResult(crops[i], float(1 / (1 + np.exp(-logit[i] / self.T))), float(logit[i]), s, contrib,
                                  float(logit[i] - contrib.sum() - self.b), up))
        return out

    def rescore(self, face: FaceResult, off: List[int]) -> float:
        new_logit = face.logit - float(face.contributions[off].sum()) if off else face.logit
        return float(1 / (1 + np.exp(-new_logit / self.T)))


class DeepEngine:
    name = "Deep (G3 teacher, GPU)"

    def __init__(self, run_dir: Path, temperature: float = 1.0) -> None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from evaluate import load_model_from_ckpt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_model_from_ckpt(str(Path(run_dir) / "best.pth"), self.device)
        self.T = float(temperature)
        self.version = Path(run_dir).name

    @torch.no_grad()
    def run(self, crops: List[np.ndarray]) -> List[FaceResult]:
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(255).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            o = self.model(x)
        maps = torch.sigmoid(torch.nn.functional.interpolate(o["maps"].float(), size=224, mode="bilinear", align_corners=False))
        res = []
        for i in range(len(crops)):
            lg = float(o["logit"][i])
            res.append(FaceResult(crops[i], float(torch.sigmoid(torch.tensor(lg / self.T))), lg,
                                  torch.sigmoid(o["concepts"][i].float()).cpu().numpy(),
                                  o["contributions"][i].float().cpu().numpy(), float(o["residual"][i]),
                                  maps[i].cpu().numpy(), o["cls"][i].float().cpu().numpy(), o["z"][i].float().cpu().numpy()))
        return res

    def rescore(self, face: FaceResult, off: List[int]) -> float:
        x = torch.from_numpy(face.crop).permute(2, 0, 1).float().div(255)[None].to(self.device)
        mask = torch.zeros(10, device=self.device)
        if off:
            mask[off] = 1
        with torch.no_grad():
            lg = float(self.model(x, off=mask)["logit"][0])
        return float(torch.sigmoid(torch.tensor(lg / self.T)))


def analyse(media_path: str, engine, cropper: FaceCropper, is_video: bool, n_frames: int = 32) -> Analysis:
    from explain.video_cues import identity_drift, temporal_flicker
    a = Analysis(mode=engine.name)
    frames = sample_frames(media_path, n_frames) if is_video else [cv2.cvtColor(cv2.imread(media_path), cv2.COLOR_BGR2RGB)]
    a.n_frames = len(frames)
    crops = [c for c in (cropper.crop(f) for f in frames) if c is not None]
    min_faces = 4 if is_video else 1
    if len(crops) < min_faces:
        a.message = (f"Only {len(crops)} of {len(frames)} sampled frames contain a detectable face; "
                     f"need at least {min_faces}. No verdict given." if is_video else "No face found in this image. No verdict given.")
        return a
    a.faces = engine.run(crops)
    a.prob = float(np.mean([f.prob for f in a.faces]))
    a.verdict = band(a.prob)
    if is_video:
        a.flicker = temporal_flicker(np.stack([f.concepts for f in a.faces]))
        if a.faces[0].cls is not None:
            a.drift = identity_drift(np.stack([f.cls for f in a.faces]))
    return a
