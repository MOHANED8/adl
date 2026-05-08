from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch.utils.data import Dataset


Split = Literal["train", "val", "test", "id_val", "id_test"]


class WILDSImageDataset(Dataset):
    """
    Thin wrapper around `wilds` to provide a torchvision-style dataset.

    Usage via factory:
      name="wilds:camelyon17" or name="wilds:iwildcam" etc.
    """

    def __init__(
        self,
        *,
        dataset_name: str,
        root: str | Path,
        split: str,
        transform=None,
        return_metadata: bool = False,
    ):
        self.dataset_name = dataset_name
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.transform = transform
        self.return_metadata = return_metadata

        try:
            from wilds import get_dataset
        except Exception as e:
            raise RuntimeError("Missing dependency 'wilds'. Install with `pip install wilds`.") from e

        # WILDS handles downloading into root_dir when download=True.
        self._wilds_ds = get_dataset(dataset=dataset_name, root_dir=str(self.root), download=True)

        # Split names vary by dataset; WILDS provides split_dict.
        split_key = split.lower().strip()
        alias = {"valid": "val", "validation": "val", "tr": "train", "te": "test"}
        split_key = alias.get(split_key, split_key)
        if split_key not in self._wilds_ds.split_dict:
            raise ValueError(
                f"Split '{split}' not available for WILDS dataset '{dataset_name}'. "
                f"Available: {sorted(self._wilds_ds.split_dict.keys())}"
            )

        self._subset = self._wilds_ds.get_subset(split_key, transform=self.transform)

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx: int):
        x, y, metadata = self._subset[idx]
        if self.return_metadata:
            return x, y, metadata
        return x, y

