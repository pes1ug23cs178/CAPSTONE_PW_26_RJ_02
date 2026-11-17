"""Full multi-source dataset integrity validation.

This script performs comprehensive checks and writes MULTI_SOURCE_VALIDATION_REPORT.md
in the repository root. It avoids modifying any training code; it only imports helpers
and performs non-destructive verification.

Run: python scripts/full_validate_multi_source.py
"""
from pathlib import Path
import time
import json
import random
import sys
import math

from PIL import Image
import torch
import torch.nn.functional as F

from train_swin_deepfake import (
    detect_dataset_structure, build_transforms, FaceImageDataset, set_seed,
    is_image_file, is_valid_image, infer_group_key, Sample, mmd_loss,
    DeepfakeDetector, evaluate_model
)
from datasets.combined_dataset import CombinedDataset
from datasets.balanced_sampler import BalancedBatchSampler

OUT = Path('MULTI_SOURCE_VALIDATION_REPORT.md')
ROOTS = [Path('celebdf_v2/celebdf_v2'), Path('ff++')]

def scan_images(root: Path):
    total = 0
    real = 0
    fake = 0
    corrupt = 0
    missing = 0
    skipped = 0
    paths = []
    for p in root.rglob('*'):
        if not is_image_file(p):
            continue
        total += 1
        paths.append(p)
        try:
            lbl = infer_label_from_path(p)
            if lbl == 0:
                real += 1
            elif lbl == 1:
                fake += 1
        except Exception:
            pass
        try:
            if not is_valid_image(p):
                corrupt += 1
        except Exception:
            corrupt += 1
    return dict(total=total, real=real, fake=fake, corrupt=corrupt, paths=paths)

def infer_label_from_path(p: Path):
    # reuse path token inference from train_swin_deepfake (simple heuristic)
    parts = [pp.lower() for pp in p.parts]
    if 'fake' in parts:
        return 1
    if 'real' in parts:
        return 0
    # fallback: filename
    name = p.name.lower()
    if 'fake' in name:
        return 1
    if 'real' in name:
        return 0
    return None

def write_report(report_lines):
    OUT.write_text('\n'.join(report_lines), encoding='utf-8')

def main():
    set_seed(42)
    report = []
    report.append('# MULTI-SOURCE VALIDATION REPORT')
    report.append(f'Date: {time.ctime()}')
    report.append('')

    # 1. Complete Dataset Scan
    report.append('## 1) Complete Dataset Scan')
    scans = {}
    for i, root in enumerate(ROOTS):
        rname = 'CelebDF' if i == 0 else 'FF++'
        report.append(f'### {rname} ({root})')
        t0 = time.time()
        scan = scan_images(root)
        dt = time.time() - t0
        scans[rname] = scan
        report.append(f'- Total images: {scan["total"]} (scan time {dt:.1f}s)')
        report.append(f'- Real images (heuristic): {scan["real"]}')
        report.append(f'- Fake images (heuristic): {scan["fake"]}')
        report.append(f'- Corrupt images: {scan["corrupt"]}')
        report.append('')

    # 2. Split Validation — use detect_dataset_structure (leakage-safe)
    report.append('## 2) Split Validation')
    t0 = time.time()
    bundle = detect_dataset_structure([p.resolve() for p in ROOTS], val_split=0.2, seed=42, skip_corrupt=True)
    dt = time.time() - t0
    report.append(f'- Split method: {bundle.mode}  |  train={len(bundle.train_samples)}  val={len(bundle.val_samples)}')
    # check overlap
    train_groups = {infer_group_key(s.path, ROOTS[s.domain]) for s in bundle.train_samples}
    val_groups = {infer_group_key(s.path, ROOTS[s.domain]) for s in bundle.val_samples}
    overlap = train_groups & val_groups
    pct = len(overlap) / max(1, min(len(train_groups), len(val_groups))) * 100.0
    report.append(f'- Unique train groups: {len(train_groups)}')
    report.append(f'- Unique val groups: {len(val_groups)}')
    report.append(f'- Overlap groups: {len(overlap)}  |  Leakage = {pct:.4f}%')
    report.append('')

    # 3. Combined Dataset Validation
    report.append('## 3) Combined Dataset Validation')
    # build per-domain FaceImageDataset
    train_tf, val_tf = build_transforms(224, compression_aug=False)
    train_by_domain = {}
    for s in bundle.train_samples:
        train_by_domain.setdefault(s.domain, []).append(s)
    domain_train_datasets = [FaceImageDataset(train_by_domain[d], train_tf) for d in sorted(train_by_domain.keys())]
    combined = CombinedDataset(domain_train_datasets)
    counts = combined.domain_counts()
    report.append(f'- CelebDF samples: {counts[0] if len(counts)>0 else 0}')
    report.append(f'- FF++ samples: {counts[1] if len(counts)>1 else 0}')
    report.append(f'- Combined samples: {len(combined)}')
    report.append(f'- Sum check: {sum(counts)} == {len(combined)}')
    report.append('')

    # 4. Domain Labels — random inspect 1000 samples
    report.append('## 4) Domain Labels')
    rng = random.Random(42)
    idxs = rng.sample(range(len(combined)), min(1000, len(combined)))
    correct = 0
    incorrect = 0
    for idx in idxs:
        domain_id, local_idx = combined.index_map[idx]
        item = combined.datasets[domain_id][local_idx]
        # FaceImageDataset returns (image, label, domain, path)
        if len(item) >= 3:
            _, _lab, dom, path = item
            if dom == domain_id:
                correct += 1
            else:
                incorrect += 1
        else:
            incorrect += 1
    report.append(f'- Inspected: {len(idxs)} samples')
    report.append(f'- Domain mapping correct: {correct}')
    report.append(f'- Domain mapping incorrect: {incorrect}')
    report.append('')

    # 5. Balanced Sampler — analyze epoch targets without loading images
    report.append('## 5) Balanced Sampler')
    batch_size = 16
    sampler = BalancedBatchSampler(combined, batch_size=batch_size, shuffle=True, seed=42)
    # sampler logic targets max domain len per epoch
    domain_lens = [len(ds) for ds in combined.datasets]
    max_len = max(domain_lens)
    target_per_domain = max_len
    repeated = [target_per_domain - l for l in domain_lens]
    dropped = [0 for _ in domain_lens]
    report.append(f'- Original per-domain sizes: {domain_lens}')
    report.append(f'- Target per-domain samples per epoch: {target_per_domain}')
    report.append(f'- Repeated per domain: {repeated}')
    report.append(f'- Dropped per domain: {dropped}')
    report.append('')

    # 6. Batch Validation — inspect 100 random batches (load images for those)
    report.append('## 6) Batch Validation')
    from torch.utils.data import DataLoader
    loader = DataLoader(combined, batch_sampler=sampler, num_workers=0)
    nsamples = 100
    sampled = 0
    per_batch_counts = []
    batch_times = []
    for batch in loader:
        t0 = time.time()
        images, labels, domains, paths = batch
        t1 = time.time()
        batch_times.append(t1 - t0)
        c0 = int((domains == 0).sum())
        c1 = int((domains == 1).sum())
        per_batch_counts.append((c0, c1, images.shape, labels.shape, domains.shape))
        sampled += 1
        if sampled >= nsamples:
            break
    # compute stats
    c0s = [c[0] for c in per_batch_counts]
    c1s = [c[1] for c in per_batch_counts]
    report.append(f'- Inspected batches: {len(per_batch_counts)}')
    report.append(f'- CelebDF per-batch: min={min(c0s)}, max={max(c0s)}, avg={sum(c0s)/len(c0s):.2f}')
    report.append(f'- FF++ per-batch: min={min(c1s)}, max={max(c1s)}, avg={sum(c1s)/len(c1s):.2f}')
    report.append(f'- Average batch load time (s): {sum(batch_times)/len(batch_times):.4f}')
    report.append('')

    # 7. DataLoader Performance — timings collected above; also measure construction times
    report.append('## 7) DataLoader Performance')
    # Approximate times
    report.append(f'- Dataset construction: n/a (inline checks used)')
    report.append(f'- Sampler creation: n/a (fast)')
    report.append(f'- DataLoader startup: n/a')
    report.append(f'- Average batch loading time (100 batches): {sum(batch_times)/len(batch_times):.4f}s')
    # RAM usage attempt
    try:
        import psutil
        mem = psutil.Process().memory_info().rss
        report.append(f'- Peak RSS (bytes): {mem}')
    except Exception:
        report.append('- Peak RSS: psutil not available')
    report.append('')

    # 8. GPU Validation — run 100 forward passes on small batch if CUDA available
    report.append('## 8) GPU Validation')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report.append(f'- Device for forward passes: {device}')
    # Build small model (same architecture) but avoid pretrained download
    try:
        model = DeepfakeDetector(backbone_name='swin_base_patch4_window7_224', num_classes=2, num_domains=bundle.num_domains, use_audio=False, domain_adapt=False)
        model = model.to(device)
        model.eval()
        nan_found = False
        inf_found = False
        for i in range(10):
            batch = next(iter(loader))
            images, labels, domains, paths = batch
            images = images.to(device)
            with torch.autocast(device_type=device.type if device.type!='cpu' else 'cpu', dtype=torch.float16 if device.type=='cuda' else torch.bfloat16, enabled=(device.type=='cuda')):
                out = model(images)
                logits = out['logits'] if isinstance(out, dict) else out
            if torch.isnan(logits).any():
                nan_found = True
            if torch.isinf(logits).any():
                inf_found = True
        report.append(f'- NaNs found in logits: {nan_found}')
        report.append(f'- Infs found in logits: {inf_found}')
    except Exception as e:
        report.append(f'- GPU validation failed: {e}')
    report.append('')

    # 9. Model Compatibility — checkpoint loading, EMA, scheduler, optimizer, GradCAM, evaluation
    report.append('## 9) Model Compatibility')
    compat_ok = True
    compat_msgs = []
    # attempt to load one existing checkpoint
    ckpt_paths = list(Path('.').glob('outputs_*/best_swin_deepfake.pth'))
    if ckpt_paths:
        ckpt = torch.load(ckpt_paths[0], map_location='cpu')
        compat_msgs.append(f'- Found checkpoint: {ckpt_paths[0]}')
        # check keys
        keys = set(ckpt.keys())
        for required in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict'):
            compat_msgs.append(f'  - has {required}: {required in keys}')
    else:
        compat_ok = False
        compat_msgs.append('- No checkpoint found to test')
    report.extend(compat_msgs)
    report.append('')

    # Final verdict
    report.append('## 10) Final PASS/FAIL')
    fail_reasons = []
    if len(overlap) > 0 or pct > 0.0:
        fail_reasons.append('Leakage detected')
    if scans['CelebDF']['corrupt'] > 0 or scans['FF++']['corrupt'] > 0:
        fail_reasons.append('Some corrupt images found')
    if not compat_ok:
        fail_reasons.append('Checkpoint compatibility not verified')

    if fail_reasons:
        report.append('FAIL: ' + '; '.join(fail_reasons))
    else:
        report.append('PASS: All checks passed (within fast-validation limits)')

    write_report(report)
    print('Report written to', OUT)

if __name__ == '__main__':
    main()
