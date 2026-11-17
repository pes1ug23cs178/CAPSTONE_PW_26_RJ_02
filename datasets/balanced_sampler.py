import math
import random
from typing import Dict, Iterable, Iterator, List
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """
    Batch sampler that yields balanced batches across domains for a CombinedDataset.

    Behavior:
    - Accepts a CombinedDataset instance which exposes an `index_map` attribute
      mapping every combined index to (domain_id, idx_in_domain).
    - Produces batches where each domain contributes approximately `batch_size // num_domains`
      samples per batch. Remainder samples are distributed to domains randomly each batch.
    - If a domain runs out of unique indices during an epoch, it is reshuffled and cycled
      (this repeats samples but does not copy files) so the sampler always delivers
      balanced contribution up to `num_domains * max_domain_len` samples per epoch.

    Notes:
    - This is memory-efficient: it only stores lists of integer indices.
    - Use with DataLoader(..., batch_sampler=BalancedBatchSampler(...)).
    """

    def __init__(self, combined_dataset, batch_size: int, shuffle: bool = True, seed: int = 42):
        # combined_dataset must have index_map: List[(domain_id, idx_in_domain)]
        self.combined = combined_dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)

        # Build domain -> combined-index list map
        self.domain_to_indices: Dict[int, List[int]] = {}
        for combined_idx, (domain_id, _local_idx) in enumerate(self.combined.index_map):
            self.domain_to_indices.setdefault(domain_id, []).append(combined_idx)

        self.domains = sorted(self.domain_to_indices.keys())
        self.num_domains = len(self.domains)
        if self.num_domains == 0:
            raise ValueError("CombinedDataset contains no domains")

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)

        # Per-domain pools: shuffle lists
        pools: Dict[int, List[int]] = {}
        lengths = {d: len(self.domain_to_indices[d]) for d in self.domains}
        max_len = max(lengths.values())

        # Target per-domain epoch size ensures equal contribution: max_len samples per domain
        target_per_domain = max_len

        for d in self.domains:
            base = list(self.domain_to_indices[d])
            if self.shuffle:
                rng.shuffle(base)
            # If domain smaller than target, cycle through and extend by reshuffled copies
            if len(base) < target_per_domain:
                extended = []
                while len(extended) < target_per_domain:
                    # reshuffle copy to avoid repeating same order
                    tmp = list(self.domain_to_indices[d])
                    if self.shuffle:
                        rng.shuffle(tmp)
                    extended.extend(tmp)
                base = extended[:target_per_domain]
            else:
                base = base[:target_per_domain]
            pools[d] = base

        # Create iterators per domain
        iterators = {d: iter(pools[d]) for d in self.domains}

        per_domain_batch = self.batch_size // self.num_domains
        remainder = self.batch_size - (per_domain_batch * self.num_domains)

        num_batches = (target_per_domain * self.num_domains) // self.batch_size

        for _ in range(num_batches):
            batch: List[int] = []
            # draw fair share from each domain
            for d in self.domains:
                for _ in range(per_domain_batch):
                    try:
                        batch.append(next(iterators[d]))
                    except StopIteration:
                        # should not happen because we pre-extended, but guard
                        pools_d = list(self.domain_to_indices[d])
                        if self.shuffle:
                            rng.shuffle(pools_d)
                        iterators[d] = iter(pools_d)
                        batch.append(next(iterators[d]))

            # distribute remainder randomly across domains
            if remainder > 0:
                chosen = rng.choices(self.domains, k=remainder)
                for d in chosen:
                    try:
                        batch.append(next(iterators[d]))
                    except StopIteration:
                        pools_d = list(self.domain_to_indices[d])
                        if self.shuffle:
                            rng.shuffle(pools_d)
                        iterators[d] = iter(pools_d)
                        batch.append(next(iterators[d]))

            yield batch

    def __len__(self) -> int:
        # number of batches per epoch
        lengths = [len(self.domain_to_indices[d]) for d in self.domains]
        max_len = max(lengths)
        total = max_len * self.num_domains
        return math.ceil(total / self.batch_size)
