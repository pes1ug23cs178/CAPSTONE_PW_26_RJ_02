from typing import List, Sequence, Tuple

class CombinedDataset:
    """
    Lightweight combined dataset wrapper for multiple existing Dataset instances.

    - Accepts a list of dataset objects (they must implement __len__ and __getitem__).
    - Builds a tiny in-memory index of (domain_id, sample_idx) pairs and redirects
      __getitem__ calls to the underlying dataset. No files are copied or loaded.

    Usage:
        ds = CombinedDataset([ds0, ds1, ds2])
        x, y, domain, path = ds[10]

    Memory overhead: O(number of samples) for a list of small tuples.
    """

    def __init__(self, datasets: Sequence[object]):
        self.datasets: List[object] = list(datasets)
        # Build index mapping: list of (domain_id, idx_in_domain)
        self.index_map: List[Tuple[int, int]] = []
        for domain_id, ds in enumerate(self.datasets):
            n = len(ds)
            # store pairs; these are tiny tuples pointing to original dataset positions
            self.index_map.extend([(domain_id, i) for i in range(n)])

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int):
        domain_id, idx = self.index_map[index]
        # delegate to the original dataset; assume it returns (image, label, domain, path)
        return self.datasets[domain_id][idx]

    def domain_counts(self) -> List[int]:
        return [len(ds) for ds in self.datasets]
