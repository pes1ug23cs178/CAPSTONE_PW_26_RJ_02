"""
Inspect actual group keys generated for train/val samples to identify collision.
"""
from pathlib import Path
from collections import defaultdict
from train_swin_deepfake import detect_dataset_structure, infer_group_key

ROOTS = [Path('celebdf_v2/celebdf_v2'), Path('ff++')]
DOMAIN_NAMES = ['CelebDF', 'FF++']

print("Building split...")
bundle = detect_dataset_structure([p.resolve() for p in ROOTS], val_split=0.2, seed=42, skip_corrupt=True)

print(f"Total train samples: {len(bundle.train_samples)}")
print(f"Total val samples: {len(bundle.val_samples)}")

# Extract groups for train/val WITHOUT namespacing
print("\n" + "=" * 80)
print("Train group keys (without namespacing):")
train_groups_raw = defaultdict(list)
for s in bundle.train_samples:
    group_key = infer_group_key(s.path, ROOTS[s.domain])
    train_groups_raw[group_key].append((s.domain, s.path))

print(f"Total unique train groups: {len(train_groups_raw)}")
print("Sample train groups:")
for i, (grp, samples) in enumerate(sorted(train_groups_raw.items())[:10]):
    domains = [DOMAIN_NAMES[d] for d, _ in samples]
    print(f"  {grp}: {len(samples)} samples from {set(domains)}")

# Extract groups for val WITHOUT namespacing
print("\n" + "=" * 80)
print("Val group keys (without namespacing):")
val_groups_raw = defaultdict(list)
for s in bundle.val_samples:
    group_key = infer_group_key(s.path, ROOTS[s.domain])
    val_groups_raw[group_key].append((s.domain, s.path))

print(f"Total unique val groups: {len(val_groups_raw)}")
print("Sample val groups:")
for i, (grp, samples) in enumerate(sorted(val_groups_raw.items())[:10]):
    domains = [DOMAIN_NAMES[d] for d, _ in samples]
    print(f"  {grp}: {len(samples)} samples from {set(domains)}")

# Check for overlaps
print("\n" + "=" * 80)
print("Checking for overlaps...")
overlap_groups = set(train_groups_raw.keys()) & set(val_groups_raw.keys())
print(f"Overlapping group keys: {len(overlap_groups)}")

if overlap_groups:
    print("\nDetailed overlap analysis:")
    for grp in sorted(list(overlap_groups))[:20]:  # show first 20
        train_doms = set(d for d, _ in train_groups_raw[grp])
        val_doms = set(d for d, _ in val_groups_raw[grp])
        print(f"\n  Group: {grp}")
        print(f"    Train domains: {[DOMAIN_NAMES[d] for d in train_doms]}")
        print(f"    Val domains: {[DOMAIN_NAMES[d] for d in val_doms]}")
        
        # Check if genuine overlap
        if train_doms & val_doms:  # same domain in both train and val
            overlapping_dom = list(train_doms & val_doms)[0]
            print(f"    => GENUINE OVERLAP in {DOMAIN_NAMES[overlapping_dom]}")
            train_in_dom = [p for d, p in train_groups_raw[grp] if d == overlapping_dom]
            val_in_dom = [p for d, p in val_groups_raw[grp] if d == overlapping_dom]
            print(f"       Train: {len(train_in_dom)} samples")
            print(f"       Val: {len(val_in_dom)} samples")
            print(f"       Sample train path: {train_in_dom[0]}")
            print(f"       Sample val path: {val_in_dom[0]}")
        else:
            print(f"    => CROSS-DOMAIN COLLISION (fake overlap)")
            for dom in train_doms:
                print(f"       In train ({DOMAIN_NAMES[dom]}): {sum(1 for d, _ in train_groups_raw[grp] if d == dom)} samples")
            for dom in val_doms:
                print(f"       In val ({DOMAIN_NAMES[dom]}): {sum(1 for d, _ in val_groups_raw[grp] if d == dom)} samples")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Count genuine vs cross-domain overlaps
genuine_overlaps = 0
cross_domain_overlaps = 0
for grp in overlap_groups:
    train_doms = set(d for d, _ in train_groups_raw[grp])
    val_doms = set(d for d, _ in val_groups_raw[grp])
    if train_doms & val_doms:
        genuine_overlaps += 1
    else:
        cross_domain_overlaps += 1

print(f"\nGenuine overlaps (same domain in train & val): {genuine_overlaps}")
print(f"Cross-domain collisions (fake overlaps): {cross_domain_overlaps}")

if cross_domain_overlaps > 0 and genuine_overlaps == 0:
    print("\n=> ROOT CAUSE: All reported overlaps are CROSS-DOMAIN COLLISIONS")
    print("=> The reported leakage is a VALIDATION ARTIFACT, not genuine leakage")
    print("=> Fix: Namespace group IDs in validation script")
elif genuine_overlaps > 0:
    print(f"\n=> ROOT CAUSE: {genuine_overlaps} GENUINE LEAKS detected within same dataset")
    print("=> This is true leakage that needs investigation")
else:
    print("\n=> NO LEAKAGE DETECTED")

print("=" * 80)
