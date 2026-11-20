"""Assemble the Hugging Face Space for the GRACE app into deploy/hf_space/ (inference code only; no data, no training).

    python scripts/build_space.py --model-repo <user>/grace-deepfake-models [--space-title "GRACE Deepfake Check"]

Then push deploy/hf_space/ to the Space repo (see docs/DEPLOY.md). The Space downloads the model files from the
model repo when it starts (GRACE_MODEL_REPO), so the Space itself stays small.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "deploy" / "hf_space"

CODE = ["app", "models", "explain", "utils", "data", "evaluate.py"]
EXTRA = ["results/final_teacher.json", "configs/paths.yaml"]

# exact versions used in the lab (torch as the CPU build: the free Space has no GPU)
REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.11.0
torchvision==0.26.0
timm==1.0.31
peft==0.21.3
onnxruntime==1.32.0
face-alignment==1.5.0
opencv-python-headless==4.13.0.92
numpy==2.4.4
pandas==3.1.1
scikit-learn==1.8.0
matplotlib==3.10.8
PyYAML==6.0.3
Pillow==12.2.0
huggingface_hub==2.4.0
gradio==6.32.0
"""

PACKAGES = "libgl1\nlibglib2.0-0\n"

LAUNCHER = '''"""Hugging Face Space entry point for the GRACE deepfake check (public, CPU)."""
import os

os.environ.setdefault("GRACE_ONLINE", "1")          # public mode: 50 MB uploads, 16 frames, no dataset faces
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # face-alignment must not try to compile itself
os.environ.setdefault("GRACE_DEVICE", "cpu")        # free Space has no GPU; never guess

from app.app import launch  # noqa: E402

launch()
'''

README = """---
title: {title}
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: space_app.py
pinned: false
short_description: Explainable deepfake check for face photos and videos
---

# {title}

Upload a face photo or a short video. GRACE answers **Likely fake**, **Can't tell** or **Looks real** in plain words
and shows the three clues behind the answer on the face.

- **Careful check**: the final teacher model (CLIP ViT-L/14 + LoRA, DCT branch, 10-clue decision layer). On this
  free CPU host a video takes about 30 seconds.
- **Quick check**: a 4.7 M-parameter student model, a few seconds per video.

Capstone project PW26_RJ_02, PES University (Department of CSE), Aug–Dec 2026.
Code, results and report: https://github.com/pes1ug23cs178/CAPSTONE_PW_26_RJ_02

Models are downloaded at start-up from the Hugging Face model repo `{model_repo}`.

**Limits.** Trained on FaceForensics++ face swaps; it can be fooled by new editing tools, very strong compression or
faces seen from the side. Treat the result as one piece of evidence, not proof.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-repo", required=True, help="<user>/grace-deepfake-models")
    ap.add_argument("--space-title", default="GRACE Deepfake Check")
    args = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for item in CODE:
        src = REPO / item
        (shutil.copytree(src, OUT / item, ignore=ignore) if src.is_dir() else shutil.copy2(src, OUT / item))
    for item in EXTRA:
        (OUT / item).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / item, OUT / item)
    (OUT / "space_app.py").write_text(LAUNCHER, encoding="utf-8")
    (OUT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (OUT / "packages.txt").write_text(PACKAGES, encoding="utf-8")
    (OUT / "README.md").write_text(README.format(title=args.space_title, model_repo=args.model_repo), encoding="utf-8")
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Space folder ready: {OUT} ({n} files, {size:.1f} MB)")
    print(f"Set in Space settings: GRACE_MODEL_REPO={args.model_repo} (and HF_TOKEN if the model repo is private)")


if __name__ == "__main__":
    main()
