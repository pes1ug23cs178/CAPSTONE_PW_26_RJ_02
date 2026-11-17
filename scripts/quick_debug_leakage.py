"""
Quick debug script for leakage investigation (no full re-scan).
Focuses on group collision detection.
"""
import sys
from pathlib import Path
from collections import defaultdict
import torch

# Minimal imports - just what we need
from train_swin_deepfake import infer_group_key

ROOTS = [Path('celebdf_v2/celebdf_v2'), Path('ff++')]
DOMAIN_NAMES = ['CelebDF', 'FF++']

print("Checking for cross-dataset group ID collisions...")
print("=" * 80)

# Scan for group IDs in each domain independently
print("\n1. Scanning CelebDF for group IDs...")
celebdf_groups = defaultdict(int)
for p in ROOTS[0].rglob('*.jpg'):
    group = infer_group_key(p, ROOTS[0])
    if group:
        celebdf_groups[group] += 1

print(f"   Found {len(celebdf_groups)} unique groups in CelebDF")
sample_groups = list(celebdf_groups.items())[:5]
print(f"   Sample groups: {sample_groups}")

print("\n2. Scanning FF++ for group IDs...")
ffpp_groups = defaultdict(int)
for p in ROOTS[1].rglob('*.jpg'):
    group = infer_group_key(p, ROOTS[1])
    if group:
        ffpp_groups[group] += 1

print(f"   Found {len(ffpp_groups)} unique groups in FF++")
sample_groups = list(ffpp_groups.items())[:5]
print(f"   Sample groups: {sample_groups}")

print("\n3. Checking for group ID collisions between domains...")
collision_groups = set(celebdf_groups.keys()) & set(ffpp_groups.keys())
print(f"   Collision groups (same numeric ID): {len(collision_groups)}")

if collision_groups:
    print(f"\n   Colliding group IDs:")
    for grp in sorted(list(collision_groups))[:10]:  # show first 10
        print(f"     - {grp}: CelebDF={celebdf_groups[grp]} images, FF++={ffpp_groups[grp]} images")
    if len(collision_groups) > 10:
        print(f"     ... and {len(collision_groups) - 10} more")
    print(f"\n   => These are CROSS-DATASET collisions, NOT genuine leakage")
    print(f"   => Expected behavior: validation script should namespace groups as 'CelebDF_{group}' and 'FF++_{group}'")
else:
    print(f"   No cross-dataset collisions found")

print("\n4. GPU/CUDA availability:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"   torch.version.cuda: {torch.version.cuda}")
print(f"   torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if collision_groups:
    print(f"""
The reported leakage is likely a VALIDATION ARTIFACT caused by cross-dataset 
group ID collisions. When groups from CelebDF and FF++ happen to have the same 
numeric ID, the validation script treats them as overlapping even though they 
represent different videos/persons in different datasets.

Fix: Namespace group IDs by domain in the validation script before computing overlap.
""")
else:
    print(f"""
No cross-dataset group ID collisions detected. If leakage is still reported,
it may be a genuine within-dataset leakage or a calculation error.
""")
