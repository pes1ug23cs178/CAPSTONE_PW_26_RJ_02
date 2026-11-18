"""T5.1b: cache the teacher's outputs (EMA weights, eval mode) on every CLEAN student training crop
(FF++ real + FF++ fakes + frozen SBI pool).

Teacher = G3 (seed 42), unless G3 lost to its parent (G2 / G1) on FF++ validation, then the parent
(runbook Day 5 rule); the choice is appended to results/decisions.md.

Output: cache/teacher/shard_XX.npz with logit [N] f32, z [N,512] f16, concepts [N,10] f32,
maps [N,10,16,16] f16; cache/teacher/index.csv (frame_path, shard, row); cache/teacher/teacher.json.
Check: for 100 random crops the cached logit equals a fresh forward pass within 1e-3.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import FaceFrameDataset  # noqa: E402
from evaluate import load_model_from_ckpt  # noqa: E402
from scripts.eval_all import resolve_runs  # noqa: E402
from utils.config import REPO_ROOT, load_paths  # noqa: E402

SHARD = 50000


def pick_teacher() -> Path:
    """Final teacher chosen on val by scripts/pick_final.py (results/final_teacher.json); falls back to the old
    rule (G3 vs its parent by val AUC) when no final teacher has been picked."""
    from utils.common import final_teacher_name
    if final_teacher_name() != "g3_full":
        choice = resolve_runs(["final"], [42])[0]
        with open(REPO_ROOT / "results" / "decisions.md", "a", encoding="utf-8") as f:
            f.write(f"- {datetime.now():%Y-%m-%d %H:%M} teacher = final teacher {final_teacher_name()}: {choice.name}\n")
        return choice
    g3 = resolve_runs(["g3_full"], [42])[0]
    parent = json.loads(json.dumps(__import__("yaml").safe_load((g3 / "config.yaml").read_text())))["parent"]
    par = resolve_runs([parent], [42])
    auc = lambda r: torch.load(r / "best.pth", map_location="cpu", weights_only=False)["val_video_auc"]  # noqa: E731
    choice, note = g3, "G3"
    if par and auc(par[0]) > auc(g3):
        choice, note = par[0], f"{parent} (G3 val {auc(g3):.4f} < {parent} val {auc(par[0]):.4f})"
    with open(REPO_ROOT / "results" / "decisions.md", "a", encoding="utf-8") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} teacher = {note}: {choice.name}\n")
    return choice


@torch.no_grad()
def run_model(model, ds, device, batch_size=128):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=6)
    outs = {"logit": [], "z": [], "concepts": [], "maps": []}
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(b["image"].to(device))
        outs["logit"].append(o["logit"].float().cpu().numpy())
        outs["z"].append(o["z"].float().cpu().numpy().astype(np.float16))
        outs["concepts"].append(o["concepts"].float().cpu().numpy())
        outs["maps"].append(o["maps"].float().cpu().numpy().astype(np.float16))
    return {k: np.concatenate(v) for k, v in outs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="auto", help="run folder, or auto")
    args = ap.parse_args()
    paths = load_paths()
    cache = paths["cache_root"]
    csv = cache / "sbi_pool" / "student_train.csv"
    run = pick_teacher() if args.teacher == "auto" else Path(args.teacher)
    device = torch.device("cuda")
    model, _ = load_model_from_ckpt(str(run / "best.pth"), device)
    out_dir = cache / "teacher"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    index = []
    for s, start in enumerate(range(0, len(df), SHARD)):
        part = df.iloc[start:start + SHARD]
        tmp = out_dir / f"_part{s}.csv"
        part.to_csv(tmp, index=False)
        arrs = run_model(model, FaceFrameDataset(tmp, train=False), device)
        np.savez(out_dir / f"shard_{s:02d}.npz", **arrs)
        tmp.unlink()
        index += [{"frame_path": fp, "shard": s, "row": i} for i, fp in enumerate(part["frame_path"])]
        print(f"shard {s}: {len(part)} crops", flush=True)
    pd.DataFrame(index).to_csv(out_dir / "index.csv", index=False)
    (out_dir / "teacher.json").write_text(json.dumps({"run": run.name, "csv": str(csv.relative_to(REPO_ROOT))}, indent=2))

    # check: 100 random crops, fresh forward vs cache
    sample = df.sample(n=100, random_state=0)
    tmp = out_dir / "_check.csv"
    sample.to_csv(tmp, index=False)
    fresh = run_model(model, FaceFrameDataset(tmp, train=False), device)["logit"]
    tmp.unlink()
    idx = pd.DataFrame(index).set_index("frame_path")
    shards = {}
    cached = []
    for fp in sample["frame_path"]:
        s, r = idx.loc[fp, "shard"], idx.loc[fp, "row"]
        shards.setdefault(s, np.load(out_dir / f"shard_{s:02d}.npz"))
        cached.append(shards[s]["logit"][r])
    diff = float(np.abs(np.array(cached) - fresh).max())
    print(f"teacher cache check: max |cached - fresh| logit = {diff:.2e}")
    if diff > 1e-3:
        sys.exit("Teacher cache check FAILED")


if __name__ == "__main__":
    main()
