"""T5.3: efficiency table -> results/efficiency.csv (one row per model).

params, GFLOPs (torch FlopCounterMode, one 224x224 image), GPU throughput (frames/s, batch 64, bf16),
and for the student ONNX file: CPU median latency at batch 1 over 200 runs (ONNX Runtime).
python scripts/benchmark.py --models b1_swin g3_full s0_student s3_kd_full --onnx release/grace_student.onnx
"""
from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import load_model_from_ckpt  # noqa: E402
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT  # noqa: E402


def gflops(model) -> float:
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.rand(1, 3, 224, 224, device=next(model.parameters()).device)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return fc.get_total_flops() / 2e9  # multiply-adds -> GMACs reported as GFLOPs convention (FLOPs / 2)


@torch.no_grad()
def gpu_fps(model, batch: int = 64, iters: int = 20) -> float:
    x = torch.rand(batch, 3, 224, 224, device="cuda")
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
    torch.cuda.synchronize()
    return batch * iters / (time.time() - t)


def cpu_latency_onnx(path: Path, threads: int, runs: int = 200) -> float:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    for _ in range(10):
        sess.run(None, {"image": x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {"image": x})
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["b1_swin", "g3_full", "s0_student", "s3_kd_full"])
    ap.add_argument("--onnx", default="release/grace_student.onnx")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    rows = []
    cpu = platform.processor() or platform.machine()
    for run in resolve_runs(args.models, [42]):
        model, _ = load_model_from_ckpt(str(run / "best.pth"), torch.device("cuda"))
        name = run.name.rsplit("_s", 1)[0]
        row = {"model": name, "run": run.name,
               "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
               "trainable_m": round(sum(p.numel() for n, p in model.named_parameters()
                                        if "lora_" in n or not name.startswith(("c", "g", "a"))) / 1e6, 2),
               "gflops": round(gflops(model), 2), "gpu_fps_bs64": round(gpu_fps(model), 1)}
        rows.append(row)
        print(row, flush=True)
        del model
        torch.cuda.empty_cache()
    onnx = REPO_ROOT / args.onnx
    if onnx.exists():
        lat = cpu_latency_onnx(onnx, args.threads)
        for r in rows:
            if r["model"] == "s3_kd_full":
                r.update(cpu_onnx_latency_ms_bs1=round(lat, 2), cpu=cpu, cpu_threads=args.threads)
        print(f"ONNX CPU median latency {lat:.2f} ms ({cpu}, {args.threads} threads)")
    out = REPO_ROOT / "results" / "efficiency.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
