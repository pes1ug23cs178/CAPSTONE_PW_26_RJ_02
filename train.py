"""One training script for every model.

python train.py --config configs/b1_swin.yaml --seed 42 [--override optim.epochs=1 ...]
python train.py --config configs/b1_swin.yaml --overfit-32      # sanity: loss < 0.05 within 200 steps

Flow: 50-step dry run (no NaN / OOM) -> fresh model -> epochs of `steps_per_epoch` balanced batches
-> after each epoch, video AUC on ffpp_val with the EMA weights; best.pth + val_threshold.json on
improvement -> at the end, reload the best EMA weights and evaluate every test split -> experiments.csv.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import FaceFrameDataset  # noqa: E402
from data.sampler import CellBalancedBatchSampler  # noqa: E402
from evaluate import choose_threshold, compute_metrics, concept_auc, concept_stats, evaluate_split, predict, video_table  # noqa: E402
from models.build import build_model  # noqa: E402
from models.domain import compute_domain_alpha  # noqa: E402
from models.ema import EMA  # noqa: E402
from models.losses import compute_loss  # noqa: E402
from utils.common import append_experiment, default_num_workers, git_commit, make_run_dir, set_seed, setup_logging  # noqa: E402
from utils.config import REPO_ROOT, add_config_args, load_config  # noqa: E402

TENSOR_KEYS = ("image", "label", "concepts", "concept_mask", "boundary", "boundary_mask", "domain_id")


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if k in TENSOR_KEYS else v) for k, v in batch.items()}


def build_sbi_fn(cfg: Dict):
    if not cfg["data"].get("sbi"):
        return None
    from data.sbi import SBIMaker  # added on Day 2
    return SBIMaker(cfg)


def make_train_loader(cfg: Dict, seed: int, num_workers: int) -> (DataLoader, CellBalancedBatchSampler):
    data = cfg["data"]
    degrade_fn = None
    if data.get("student_degrade"):
        from data.degrade import StudentDegrade
        degrade_fn = StudentDegrade()
    ds = FaceFrameDataset(REPO_ROOT / data["train"], train=True, augment_cfg=data.get("augment"),
                          sbi_fn=build_sbi_fn(cfg), degrade_fn=degrade_fn)
    sampler = CellBalancedBatchSampler(ds.cell_indices(data["cells"]), cfg["optim"]["batch_size"],
                                       cfg["optim"]["steps_per_epoch"], data.get("cell_weights"), seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    return loader, sampler


def make_optimizer(model, cfg: Dict, total_steps: int, warmup_steps: int):
    groups = model.param_groups(cfg["optim"]["lr"])
    groups = [g for g in groups if any(p.requires_grad for p in g["params"])]
    wd = float(cfg["optim"].get("weight_decay", 0.05))
    split_groups = []
    for g in groups:
        params = [p for p in g["params"] if p.requires_grad]
        # no weight decay on norm scales/shifts and biases (1-D params): decay drove the Swin InstanceNorm
        # gamma to ~0.005 even in B1, close to where the following LayerNorm's eps erases the features
        decay = [p for p in params if p.ndim > 1]
        no_decay = [p for p in params if p.ndim <= 1]
        for ps, w, suffix in ((decay, wd, ""), (no_decay, 0.0, "_nodecay")):
            if ps:
                split_groups.append({**g, "params": ps, "weight_decay": w, "name": g.get("name", "") + suffix})
    opt = torch.optim.AdamW(split_groups, weight_decay=wd)
    min_factor = float(cfg["optim"].get("min_lr_factor", 0.01))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def load_teacher(cfg: Dict):
    dist = cfg.get("distill") or {}
    if not dist.get("enabled"):
        return None
    from data.teacher_cache import TeacherCache
    return TeacherCache(REPO_ROOT / dist.get("cache_dir", "cache/teacher"))


def train_step(model, batch, cfg, opt, sched, ema, alpha: float, use_bf16: bool, grad_clip: float, teacher=None):
    if teacher is not None:
        batch.update(teacher.lookup(batch["path"], batch["image"].device))
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        out = model(batch["image"], alpha=alpha)
    loss, logs = compute_loss(out, batch, cfg)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {logs}")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], grad_clip)
    opt.step()
    sched.step()
    if ema is not None:
        ema.update(model)
    logs["loss"] = float(loss.detach())
    if "gate_mean" in out:
        logs["gate_mean"] = float(out["gate_mean"])
    return logs


def dry_run(cfg: Dict, device: torch.device, log, num_workers: int, steps: int = 50) -> None:
    model = build_model(cfg["model"]).to(device)
    loader, _ = make_train_loader(cfg, cfg["seed"], num_workers)
    teacher = load_teacher(cfg)
    opt, sched = make_optimizer(model, cfg, steps, 5)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    first = last = None
    t0 = time.time()
    for i, batch in enumerate(loader):
        logs = train_step(model, to_device(batch, device), cfg, opt, sched, None, 0.1, True, 1.0, teacher)
        first = logs if first is None else first
        last = logs
        if i + 1 >= steps:
            break
    peak = torch.cuda.max_memory_allocated() / 1e9
    log.info("DRY RUN ok: %d steps in %.0fs, loss %.4f -> %.4f, peak GPU memory %.1f GB",
             steps, time.time() - t0, first["loss"], last["loss"], peak)
    del model, opt, sched, loader, teacher
    gc.collect()
    torch.cuda.empty_cache()


def overfit_32(cfg: Dict, device: torch.device, log) -> None:
    ds = FaceFrameDataset(REPO_ROOT / cfg["data"]["train"], train=False)
    idx = ds.cell_indices(["real"])["real"][:16] + [i for i, l in enumerate(ds.labels) if l == 1][:16]
    batch = torch.utils.data.default_collate([ds[i] for i in idx])
    batch = to_device(batch, device)
    model = build_model(cfg["model"]).to(device).train()
    opt, sched = make_optimizer(model, cfg, 200, 10)
    for step in range(1, 201):
        logs = train_step(model, batch, cfg, opt, sched, None, 0.0, True, 1.0)
        if step % 20 == 0:
            log.info("overfit step %d loss %.4f", step, logs["bce"])
        if logs["bce"] < 0.05:
            log.info("OVERFIT-32 PASS at step %d (bce %.4f)", step, logs["bce"])
            return
    raise SystemExit(f"OVERFIT-32 FAIL: bce {logs['bce']:.4f} after 200 steps")


def save_best(path: Path, model, ema: EMA, cfg: Dict, epoch: int, val_auc: float) -> None:
    def pick(sd):
        if cfg["model"].get("save_trainable_only"):
            names = {n for n, p in model.named_parameters() if p.requires_grad}
            return {k: v for k, v in sd.items() if k in names or "running_" in k or "num_batches" in k}
        return sd
    torch.save({"model": pick(model.state_dict()), "ema": pick(ema.state_dict()), "config": cfg, "epoch": epoch,
                "val_video_auc": val_auc, "trainable_only": bool(cfg["model"].get("save_trainable_only"))}, path)


def main() -> None:
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--overfit-32", action="store_true")
    ap.add_argument("--skip-dry-run", action="store_true")
    ap.add_argument("--no-test-eval", action="store_true")
    ap.add_argument("--num-workers", type=int, default=default_num_workers())
    args = ap.parse_args()

    cfg = load_config(args.config, args.seed, args.override)
    seed = cfg["seed"]
    set_seed(seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    if args.overfit_32:
        overfit_32(cfg, device, setup_logging())
        return

    run_dir = make_run_dir(cfg["paths"]["runs_root"], cfg["name"], seed)
    log = setup_logging(run_dir / "train.log")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(a) for a in sys.argv))
    commit = git_commit()
    (run_dir / "git_commit.txt").write_text(commit)
    log.info("Run dir %s | commit %s | seed %d", run_dir, commit, seed)
    start = datetime.now()

    if not args.skip_dry_run:
        dry_run(cfg, device, log, args.num_workers)
        set_seed(seed)

    model = build_model(cfg["model"]).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %.2f M", n_train / 1e6)
    loader, sampler = make_train_loader(cfg, seed, args.num_workers)
    teacher = load_teacher(cfg)
    o = cfg["optim"]
    epochs, steps = int(o["epochs"]), int(o["steps_per_epoch"])
    total = epochs * steps
    opt, sched = make_optimizer(model, cfg, total, int(o.get("warmup_epochs", 1)) * steps)
    ema = EMA(model, float(o.get("ema", 0.999)))
    grad_clip = float(o.get("grad_clip", 1.0))
    val_csv = REPO_ROOT / cfg["data"]["val"]

    best_auc, best_epoch, history = -1.0, -1, []
    global_step = 0
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        sums: Dict[str, float] = {}
        n = 0
        for batch in loader:
            alpha = compute_domain_alpha(global_step / max(total, 1))
            logs = train_step(model, to_device(batch, device), cfg, opt, sched, ema, alpha, True, grad_clip, teacher)
            global_step += 1
            n += 1
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
            if global_step % 100 == 0:
                avg = {k: round(v / n, 4) for k, v in sums.items()}
                log.info("ep %d step %d/%d %s alpha %.3f lr %.2e %.1f img/s mem %.1fGB", epoch, n, steps, avg, alpha,
                         sched.get_last_lr()[0], n * o["batch_size"] / (time.time() - t0),
                         torch.cuda.max_memory_allocated() / 1e9)
        train_time = time.time() - t0

        # students select on FF++ val with jpeg:50 inputs (data.val_degrade), teachers on clean val
        pred = predict(ema.model, val_csv, device, num_workers=args.num_workers, limit=cfg["data"].get("val_limit"),
                       degrade=cfg["data"].get("val_degrade"))
        vt = video_table(pred)
        val_m = compute_metrics(pred, None)
        rec = {"epoch": epoch, "train_time_s": round(train_time, 1), "val_video_auc": val_m["video_auc"],
               "val_frame_auc": val_m["frame_auc"], **{f"val_{k}": v for k, v in concept_stats(pred).items()},
               **{f"train_{k}": round(v / max(n, 1), 5) for k, v in sums.items()}}
        history.append(rec)
        improved = val_m["video_auc"] > best_auc
        if improved:
            best_auc, best_epoch = val_m["video_auc"], epoch
            save_best(run_dir / "best.pth", model, ema, cfg, epoch, best_auc)
            thr = choose_threshold(vt["label"].to_numpy(), vt["prob"].to_numpy())
            (run_dir / "val_threshold.json").write_text(json.dumps(thr, indent=2))
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        log.info("EPOCH %d done in %.0fs | val video AUC %.4f frame AUC %.4f %s%s", epoch, train_time,
                 val_m["video_auc"], val_m["frame_auc"], {k: round(v, 4) for k, v in concept_stats(pred).items()},
                 " *best*" if improved else "")

    end = datetime.now()
    log.info("Training done. Best val video AUC %.4f at epoch %d", best_auc, best_epoch)
    row = {"run_id": run_dir.name, "config": cfg["config_path"], "parent": cfg.get("parent") or "", "seed": seed,
           "commit": commit, "start": start.isoformat(timespec="minutes"), "end": end.isoformat(timespec="minutes"),
           "gpu_hours": round((end - start).total_seconds() / 3600, 2), "best_epoch": best_epoch,
           "val_video_auc": round(best_auc, 4), "notes": cfg.get("notes", "")}

    if not args.no_test_eval:
        del loader
        from evaluate import load_model_from_ckpt
        best_model, _ = load_model_from_ckpt(str(run_dir / "best.pth"), device)
        threshold = json.loads((run_dir / "val_threshold.json").read_text())["threshold"]
        # sanity: re-evaluating val from the saved checkpoint must reproduce the trainer's AUC
        vm = evaluate_split(best_model, val_csv, run_dir, run_dir.name, device, threshold, num_workers=args.num_workers)
        log.info("Reloaded best.pth val video AUC %.4f (trainer %.4f)", vm["video_auc"], best_auc)
        for test_csv in cfg["data"]["tests"]:
            m = evaluate_split(best_model, REPO_ROOT / test_csv, run_dir, run_dir.name, device, threshold,
                               num_workers=args.num_workers)
            row[f"{m['split']}_video_auc"] = round(m["video_auc"], 4)
            log.info("TEST %s: video AUC %.4f frame AUC %.4f EER %.4f acc %.4f concept_share %s gate_mean %s",
                     m["split"], m["video_auc"], m["frame_auc"], m["eer"], m["acc"], m.get("concept_share"), m.get("gate_mean"))
        if cfg["model"].get("head") == "concept":
            cache_root = Path(cfg["paths"]["cache_root"])
            cauc = {}
            for set_name in ("sbi_val", "sbi_test"):
                csv_path = cache_root / set_name / f"{set_name}.csv"
                if csv_path.exists():
                    cauc[set_name] = concept_auc(best_model, csv_path, device, args.num_workers)
                    log.info("Concept AUC on %s: %s", set_name, {k: round(v, 3) for k, v in cauc[set_name].items()})
            (run_dir / "metrics_concepts.json").write_text(json.dumps(cauc, indent=2))
        row["end"] = datetime.now().isoformat(timespec="minutes")
    append_experiment(row)
    log.info("experiments.csv updated: %s", row)


if __name__ == "__main__":
    main()
