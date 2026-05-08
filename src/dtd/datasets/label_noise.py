from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LabelNoiseSpec:
    enabled: bool = False
    p: float = 0.0
    seed: int = 0
    num_classes: Optional[int] = None
    mode: str = "uniform"  # "uniform" only for now


class LabelNoiseDataset(Dataset):
    def __init__(self, base: Dataset, *, spec: LabelNoiseSpec):
        self.base = base
        self.spec = spec
        if not spec.enabled or spec.p <= 0:
            self.noisy = None
            return
        if spec.num_classes is None:
            raise ValueError("LabelNoiseSpec.num_classes must be set when enabled.")

        rng = np.random.default_rng(spec.seed)
        n = len(base)
        flip = rng.random(n) < float(spec.p)
        # Precompute noisy labels
        ys = []
        for i in range(n):
            _, y = base[i][:2]
            yi = int(y)
            if flip[i]:
                yi = int(rng.integers(0, spec.num_classes))
            ys.append(yi)
        self.noisy = np.asarray(ys, dtype=np.int64)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if self.noisy is None:
            return item
        # assume (x,y,...) structure
        x = item[0]
        y = int(self.noisy[idx])
        if len(item) == 2:
            return x, y
        return (x, y, *item[2:])

