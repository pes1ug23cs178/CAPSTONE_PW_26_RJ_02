"""T5.3: export the student to ONNX (opset 18 - see note in main(); dynamic batch) and verify against PyTorch.

python scripts/export_onnx.py --ckpt runs/<s3>/best.pth --out release/grace_student.onnx
Input `image` N x 3 x 224 x 224 float32 in [0, 1]; outputs `logit`, `concepts`, `maps`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import FaceFrameDataset  # noqa: E402
from evaluate import load_model_from_ckpt  # noqa: E402
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


class Export(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        o = self.model(image)
        return o["logit"], o["concepts"], o["maps"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="best.pth; default = latest s3_kd_full seed 42 run")
    ap.add_argument("--out", default="release/grace_student.onnx")
    args = ap.parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else resolve_runs(["s3_kd_full"], [42])[0] / "best.pth"
    model, _ = load_model_from_ckpt(str(ckpt), torch.device("cpu"))
    model = Export(model.float().eval())
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(2, 3, 224, 224)
    # The legacy TorchScript exporter mis-exports timm's EfficientNet under torch 2.11 (outputs off by ~2;
    # checked 17 Nov). The torch.export-based exporter matches to 1e-6 but needs opset >= 18.
    n = torch.export.Dim("N", min=1, max=1024)
    torch.onnx.export(model, (dummy,), str(out), opset_version=18, input_names=["image"],
                      output_names=["logit", "concepts", "maps"], dynamo=True, dynamic_shapes={"image": {0: n}},
                      external_data=False, verbose=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    df = pd.read_csv(REPO_ROOT / "splits" / "celebdf_test.csv", dtype=str).sample(n=32, random_state=0)
    tmp = REPO_ROOT / "cache" / "_onnx_check.csv"
    df.to_csv(tmp, index=False)
    x = torch.stack([FaceFrameDataset(tmp, train=False)[i]["image"] for i in range(32)])
    tmp.unlink()
    with torch.no_grad():
        ref = [t.numpy() for t in model(x)]
    got = sess.run(None, {"image": x.numpy()})
    diffs = [float(np.abs(a - b).max()) for a, b in zip(ref, got)]
    print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB) from {ckpt}; max |torch - onnx| logit/concepts/maps = {diffs}")
    if max(diffs) > 1e-3:
        sys.exit("ONNX check FAILED (> 1e-3)")


if __name__ == "__main__":
    main()
