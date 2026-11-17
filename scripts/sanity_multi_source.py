"""Sanity check for CombinedDataset + BalancedBatchSampler

Run this locally in the project root. It performs:
- Builds combined dataset from `celebdf_v2/celebdf_v2` and `ff++`
- Iterates a few batches and prints shapes and per-domain counts
- Performs a 2-step mini-training using a lightweight dummy model to validate
  that domain loss and MMD loss compute without error
- Prints epoch sampling statistics including repeated samples per domain

Usage:
    python scripts/sanity_multi_source.py

Notes:
- This script avoids loading large pretrained backbones and does not run full
  training. It is safe to run on CPU and should finish in a minute or two.
"""

from pathlib import Path
import torch
from train_swin_deepfake import (
    detect_dataset_structure, build_transforms, FaceImageDataset,
    set_seed
)
from datasets.combined_dataset import CombinedDataset
from datasets.balanced_sampler import BalancedBatchSampler
import torch.nn.functional as F

def main():
    set_seed(42)

    root0 = Path('celebdf_v2/celebdf_v2')
    root1 = Path('ff++')
    for r in (root0, root1):
        print('Root exists:', r, r.exists())

    bundle = detect_dataset_structure([root0, root1], val_split=0.2, seed=42, skip_corrupt=True)
    print('Bundle domains:', bundle.num_domains)
    print('Train samples total:', len(bundle.train_samples))
    print('Val samples total:', len(bundle.val_samples))

    train_tf, val_tf = build_transforms(224, compression_aug=False)

    train_by_domain = {}
    for s in bundle.train_samples:
        train_by_domain.setdefault(s.domain, []).append(s)

    domain_train_datasets = [FaceImageDataset(train_by_domain[d], train_tf) for d in sorted(train_by_domain.keys())]
    for i, ds in enumerate(domain_train_datasets):
        print(f'Domain {i} dataset length:', len(ds))

    combined_train = CombinedDataset(domain_train_datasets)
    print('Combined len:', len(combined_train))
    print('Domain counts:', combined_train.domain_counts())

    index_map = combined_train.index_map
    print('Index map size, unique entries:', len(index_map), len(set(index_map)))

    batch_size = 16
    sampler = BalancedBatchSampler(combined_train, batch_size=batch_size, shuffle=True, seed=42)
    from torch.utils.data import DataLoader
    loader = DataLoader(combined_train, batch_sampler=sampler, num_workers=0)

    print('\nIterating 5 batches:')
    for b_i, batch in enumerate(loader):
        images, labels, domains, paths = batch
        print(f'Batch {b_i+1}')
        print(' Images:', images.shape)
        print(' Labels:', labels.shape)
        print(' Domains:', domains.shape)
        c0 = (domains == 0).sum().item()
        c1 = (domains == 1).sum().item()
        print(' CelebDF:', c0)
        print(' FF++   :', c1)
        print(' Paths sample:', paths[:3])
        if b_i >= 4:
            break

    # Mini training using a lightweight dummy model to validate loss computations
    import torch.nn as nn
    class DummyModel(nn.Module):
        def __init__(self, feat_dim=256):
            super().__init__()
            self.feat_dim = feat_dim
            self.proj = nn.Linear(3*224*224, feat_dim)
            self.head = nn.Linear(feat_dim, 2)
            self.domain_clf = nn.Linear(feat_dim, max(1, bundle.num_domains))
        def forward(self, x, alpha=0.0):
            b = x.size(0)
            flat = x.view(b, -1)
            feats = self.proj(flat)
            logits = self.head(feats)
            domain_logits = self.domain_clf(feats)
            return {'logits': logits, 'features': feats, 'domain_logits': domain_logits}

    model = DummyModel().to('cpu')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print('\nRunning 2 training steps with dummy model')
    it = iter(loader)
    for step in range(2):
        try:
            images, labels, domains, paths = next(it)
        except StopIteration:
            it = iter(loader)
            images, labels, domains, paths = next(it)
        images = images.to('cpu')
        labels = labels.to('cpu')
        domains = domains.to('cpu')

        opt.zero_grad()
        out = model(images, alpha=0.0)
        logits = out['logits']
        loss = criterion(logits, labels)
        dann_loss = F.cross_entropy(out['domain_logits'], domains)
        try:
            from train_swin_deepfake import mmd_loss
            unique = domains.unique()
            mmd_val = None
            if unique.numel() >= 2:
                d0, d1 = unique[0].item(), unique[1].item()
                src = out['features'][domains == d0]
                tgt = out['features'][domains == d1]
                if src.size(0) > 0 and tgt.size(0) > 0:
                    mmd_val = float(mmd_loss(src, tgt).item())
                    loss = loss + 0.1 * mmd_loss(src, tgt)
        except Exception:
            mmd_val = 'n/a'

        (loss).backward()
        opt.step()
        print(f'Step {step+1} loss:', float(loss.item()), 'dann_loss:', float(dann_loss.item()), 'mmd:', mmd_val)

    # Save small checkpoint
    ckpt_path = 'temp_sanity_ckpt.pth'
    torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
    print('\nCheckpoint saved to', ckpt_path)

    # Epoch sampling stats
    lengths = [len(domain_train_datasets[d]) for d in range(len(domain_train_datasets))]
    print('\nOriginal per-domain sizes:', lengths)
    max_len = max(lengths)
    print('Target per-domain samples per epoch (sampler):', max_len)
    print('Repeated per domain:', [max_len - l for l in lengths])

    print('\nSanity checks complete')

if __name__ == '__main__':
    main()
