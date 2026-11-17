"""
Debug script to investigate reported leakage in multi-source validation.
Checks for:
  - Case A: genuine train/val overlap within same dataset
  - Case B: cross-dataset group ID collisions
  - Case C: group ID generation issues
  - Case D: leakage calculation errors
"""
import sys
from pathlib import Path
import torch
from collections import defaultdict

from train_swin_deepfake import (
    detect_dataset_structure, infer_group_key, FaceImageDataset, build_transforms, set_seed, DeepfakeDetector
)

set_seed(42)
ROOTS = [Path('celebdf_v2/celebdf_v2'), Path('ff++')]
DOMAIN_NAMES = ['CelebDF', 'FF++']

print("=" * 80)
print("STEP 1: Locate the Overlapping Group")
print("=" * 80)

# Build split
bundle = detect_dataset_structure([p.resolve() for p in ROOTS], val_split=0.2, seed=42, skip_corrupt=True)

# Extract groups WITH namespacing by domain
print("\nExtracting groups WITH domain namespace:")
train_groups_namespaced = {}
for s in bundle.train_samples:
    group_key = infer_group_key(s.path, ROOTS[s.domain])
    namespaced = f"{DOMAIN_NAMES[s.domain]}_{group_key}"
    train_groups_namespaced.setdefault(namespaced, []).append(s)

val_groups_namespaced = {}
for s in bundle.val_samples:
    group_key = infer_group_key(s.path, ROOTS[s.domain])
    namespaced = f"{DOMAIN_NAMES[s.domain]}_{group_key}"
    val_groups_namespaced.setdefault(namespaced, []).append(s)

print(f"Train groups (namespaced): {len(train_groups_namespaced)}")
print(f"Val groups (namespaced): {len(val_groups_namespaced)}")

# Check overlap with namespacing
overlap_namespaced = set(train_groups_namespaced.keys()) & set(val_groups_namespaced.keys())
print(f"\nOverlap WITH namespacing: {len(overlap_namespaced)} groups")
if overlap_namespaced:
    print("Overlapping groups:")
    for grp in sorted(overlap_namespaced):
        train_samples = train_groups_namespaced[grp]
        val_samples = val_groups_namespaced[grp]
        print(f"\n  Group ID: {grp}")
        print(f"  Train samples: {len(train_samples)}")
        print(f"  Validation samples: {len(val_samples)}")
        print(f"  Train paths (first 3): {[str(s.path) for s in train_samples[:3]]}")
        print(f"  Validation paths (first 3): {[str(s.path) for s in val_samples[:3]]}")

# Also check WITHOUT namespacing (to confirm it's showing false collision)
print("\n" + "=" * 80)
print("STEP 2: Confirm Cross-Dataset Collision (Without Namespacing)")
print("=" * 80)

train_groups_raw = set()
train_groups_raw_map = {}
for s in bundle.train_samples:
    group_key = infer_group_key(s.path, ROOTS[s.domain])
    train_groups_raw.add(group_key)
    train_groups_raw_map.setdefault(group_key, []).append((s.domain, s.path))

val_groups_raw = set()
val_groups_raw_map = {}
for s in bundle.val_samples:
    group_key = infer_group_key(s.path, ROOTS[s.domain])
    val_groups_raw.add(group_key)
    val_groups_raw_map.setdefault(group_key, []).append((s.domain, s.path))

overlap_raw = train_groups_raw & val_groups_raw
print(f"\nOverlap WITHOUT namespacing: {len(overlap_raw)} groups")

if overlap_raw:
    print("\nAnalyzing raw overlaps (cross-dataset collisions):")
    for grp in sorted(overlap_raw):
        train_doms = set(d for d, _ in train_groups_raw_map[grp])
        val_doms = set(d for d, _ in val_groups_raw_map[grp])
        print(f"\n  Group ID: {grp}")
        print(f"  Train domains: {[DOMAIN_NAMES[d] for d in train_doms]}")
        print(f"  Val domains: {[DOMAIN_NAMES[d] for d in val_doms]}")
        
        # Check if it's same-domain or cross-domain collision
        if train_doms == val_doms:
            print(f"  => GENUINE OVERLAP in domain(s): {[DOMAIN_NAMES[d] for d in train_doms]}")
            for domain in train_doms:
                train_in_dom = [p for d, p in train_groups_raw_map[grp] if d == domain]
                val_in_dom = [p for d, p in val_groups_raw_map[grp] if d == domain]
                print(f"     {DOMAIN_NAMES[domain]}: {len(train_in_dom)} train, {len(val_in_dom)} val")
        else:
            print(f"  => CROSS-DATASET COLLISION (NOT genuine leakage)")
            print(f"     Group in CelebDF train: {sum(1 for d, _ in train_groups_raw_map[grp] if d == 0)}")
            print(f"     Group in CelebDF val: {sum(1 for d, _ in val_groups_raw_map[grp] if d == 0)}")
            print(f"     Group in FF++ train: {sum(1 for d, _ in train_groups_raw_map[grp] if d == 1)}")
            print(f"     Group in FF++ val: {sum(1 for d, _ in val_groups_raw_map[grp] if d == 1)}")

print("\n" + "=" * 80)
print("STEP 3-5: Per-Domain Validation")
print("=" * 80)

# Validate CelebDF independently
print("\nCelebDF (independent):")
celebdf_train = [s for s in bundle.train_samples if s.domain == 0]
celebdf_val = [s for s in bundle.val_samples if s.domain == 0]

celebdf_train_groups = {infer_group_key(s.path, ROOTS[0]) for s in celebdf_train}
celebdf_val_groups = {infer_group_key(s.path, ROOTS[0]) for s in celebdf_val}
celebdf_overlap = celebdf_train_groups & celebdf_val_groups

print(f"  Train groups: {len(celebdf_train_groups)}")
print(f"  Val groups: {len(celebdf_val_groups)}")
print(f"  Overlap groups: {len(celebdf_overlap)}")
if celebdf_overlap:
    print(f"  => GENUINE LEAKAGE in CelebDF: {celebdf_overlap}")
else:
    print(f"  => NO leakage in CelebDF")

# Validate FF++ independently
print("\nFF++ (independent):")
ffpp_train = [s for s in bundle.train_samples if s.domain == 1]
ffpp_val = [s for s in bundle.val_samples if s.domain == 1]

ffpp_train_groups = {infer_group_key(s.path, ROOTS[1]) for s in ffpp_train}
ffpp_val_groups = {infer_group_key(s.path, ROOTS[1]) for s in ffpp_val}
ffpp_overlap = ffpp_train_groups & ffpp_val_groups

print(f"  Train groups: {len(ffpp_train_groups)}")
print(f"  Val groups: {len(ffpp_val_groups)}")
print(f"  Overlap groups: {len(ffpp_overlap)}")
if ffpp_overlap:
    print(f"  => GENUINE LEAKAGE in FF++: {ffpp_overlap}")
else:
    print(f"  => NO leakage in FF++")

print("\n" + "=" * 80)
print("STEP 6: Correct Leakage Metrics")
print("=" * 80)

# Sample leakage
leaked_samples = len(celebdf_overlap) * 0  # placeholder; actual count from real_overlap
total_val_samples = len(celebdf_val) + len(ffpp_val)

print(f"\nSample Leakage Calculation:")
print(f"  CelebDF genuine overlaps: {len(celebdf_overlap)} groups")
print(f"  FF++ genuine overlaps: {len(ffpp_overlap)} groups")
print(f"  Total validation samples: {total_val_samples}")
if len(celebdf_overlap) > 0 or len(ffpp_overlap) > 0:
    print(f"  => GENUINE LEAKAGE DETECTED in within-dataset splits")
else:
    print(f"  => NO genuine leakage detected")

print(f"\nGroup Leakage Calculation (as reported):")
print(f"  Reported: 1 / 90 = 1.1111%")
print(f"  Corrected (if namespaced): {len(overlap_namespaced)} / {len(val_groups_namespaced)} = {100.0 * len(overlap_namespaced) / max(1, len(val_groups_namespaced)):.4f}%")

print("\n" + "=" * 80)
print("STEP 7: Dataset Count Verification")
print("=" * 80)

# Count images in each domain manually
print(f"\nCelebDF train samples in bundle: {len(celebdf_train)}")
print(f"CelebDF val samples in bundle: {len(celebdf_val)}")
print(f"FF++ train samples in bundle: {len(ffpp_train)}")
print(f"FF++ val samples in bundle: {len(ffpp_val)}")
print(f"Total combined split: {len(bundle.train_samples)} + {len(bundle.val_samples)} = {len(bundle.train_samples) + len(bundle.val_samples)}")

print("\n" + "=" * 80)
print("STEP 8: GPU Availability Check")
print("=" * 80)

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice selected: {device}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nRoot cause determination:")
if len(celebdf_overlap) == 0 and len(ffpp_overlap) == 0 and len(overlap_raw) > 0:
    print("=> CASE B: Cross-dataset group ID collision (NOT genuine leakage)")
    print("   Fix: Namespace group IDs in validation script")
    print("   Expected result after fix: PASS")
elif len(celebdf_overlap) > 0 or len(ffpp_overlap) > 0:
    print("=> CASE A: Genuine train/validation leakage within same dataset")
    print("   Requires investigation/fix in split logic")
else:
    print("=> NO LEAKAGE DETECTED")
    print("   Expected result: PASS")

print("\n" + "=" * 80)
