"""Quick, lightweight check for combined dataset and balanced sampler.

This script avoids expensive image validation and only counts files and
builds a small in-memory CombinedDataset to iterate a few batches.

Run: python scripts/quick_check.py
"""
from pathlib import Path
import random
import sys
from datasets.combined_dataset import CombinedDataset
from datasets.balanced_sampler import BalancedBatchSampler
from train_swin_deepfake import build_transforms, FaceImageDataset, set_seed

set_seed(42)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS)

def collect_sample_paths(root: Path, limit: int = 5000):
    paths = [p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS]
    if len(paths) > limit:
        random.seed(42)
        paths = random.sample(paths, limit)
    return paths

def main():
    base0 = Path('celebdf_v2/celebdf_v2')
    base1 = Path('ff++')
    print('CelebDF root exists:', base0.exists())
    print('FF++ root exists   :', base1.exists())

    # quick counts
    c0 = count_images(base0)
    c1 = count_images(base1)
    print('CelebDF approx images:', c0)
    print('FF++ approx images   :', c1)

    # Build tiny sample lists by reusing the Sample dataclass structure
    from train_swin_deepfake import Sample
    def make_samples(root: Path, domain_id: int):
        paths = collect_sample_paths(root, limit=2000)
        return [Sample(path=p, label=1 if 'fake' in p.parts or 'fake' in p.name.lower() else 0, domain=domain_id) for p in paths]

    s0 = make_samples(base0, 0)
    s1 = make_samples(base1, 1)
    print('Collected sample counts (fast):', len(s0), len(s1))

    # Build FaceImageDataset with transforms but do not load many images
    train_tf, _ = build_transforms(224, compression_aug=False)
    ds0 = FaceImageDataset(s0, train_tf)
    ds1 = FaceImageDataset(s1, train_tf)

    combined = CombinedDataset([ds0, ds1])
    print('Combined length (fast):', len(combined))
    print('Domain counts:', combined.domain_counts())

    sampler = BalancedBatchSampler(combined, batch_size=16, shuffle=True, seed=42)
    from torch.utils.data import DataLoader
    loader = DataLoader(combined, batch_sampler=sampler, num_workers=0)

    print('\nIterating 4 batches (quick):')
    for i, batch in enumerate(loader):
        images, labels, domains, paths = batch
        print(f'Batch {i+1}: images={images.shape}, labels={labels.shape}, domains={domains.shape}')
        print(' CelebDF:', int((domains==0).sum()), ' FF++:', int((domains==1).sum()))
        if i >= 3:
            break

    # Epoch stats
    lens = [len(ds0), len(ds1)]
    mx = max(lens)
    print('\nOriginal per-domain sizes:', lens)
    print('Target per-domain samples per epoch (sampler):', mx)
    print('Repeated per domain:', [mx - l for l in lens])

if __name__ == '__main__':
    main()
