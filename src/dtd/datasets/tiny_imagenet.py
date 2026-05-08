from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class TinyImageNetInfo:
    url: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    # Some mirrors distribute zip; this loader supports both zip extraction by user or direct folder presence.


class TinyImageNet(Dataset):
    def __init__(self, root: str | Path, *, split: Split, download: bool, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if download:
            self._maybe_download()

        if not self.root.exists():
            raise FileNotFoundError(
                f"TinyImageNet root not found at '{self.root}'. "
                f"Expected extracted folder 'tiny-imagenet-200'."
            )

        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self._index()

    def _maybe_download(self):
        # Tiny ImageNet is hosted as a zip; we download and extract automatically.
        parent = self.root.parent
        parent.mkdir(parents=True, exist_ok=True)
        archive = parent / "tiny-imagenet-200.zip"
        if self.root.exists():
            return
        if not archive.exists():
            try:
                urllib.request.urlretrieve(TinyImageNetInfo.url, archive)  # noqa: S310 (controlled URL)
            except Exception as e:
                raise RuntimeError(
                    f"Failed downloading Tiny ImageNet archive to '{archive}'. "
                    f"Download manually from '{TinyImageNetInfo.url}' and extract to '{self.root}'."
                ) from e

        try:
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(parent)
        except Exception as e:
            raise RuntimeError(f"Failed extracting '{archive}' to '{parent}'.") from e

    def _index(self):
        if self.split == "train":
            train_dir = self.root / "train"
            wnids = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
            self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
            for wnid in wnids:
                img_dir = train_dir / wnid / "images"
                for img_path in img_dir.glob("*.JPEG"):
                    self.samples.append((img_path, self.class_to_idx[wnid]))
        elif self.split == "val":
            val_dir = self.root / "val"
            ann = val_dir / "val_annotations.txt"
            if not ann.exists():
                raise FileNotFoundError(f"Missing '{ann}'. Tiny ImageNet val split requires val_annotations.txt.")
            # Build class mapping from wnids.txt for stable ordering
            wnids_file = self.root / "wnids.txt"
            if not wnids_file.exists():
                raise FileNotFoundError(f"Missing '{wnids_file}'.")
            wnids = [l.strip() for l in wnids_file.read_text().splitlines() if l.strip()]
            self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

            # Parse annotations: <img>\t<wnid>\t...
            mapping = {}
            for line in ann.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
            img_dir = val_dir / "images"
            for img_path in img_dir.glob("*.JPEG"):
                wnid = mapping.get(img_path.name)
                if wnid is None:
                    continue
                self.samples.append((img_path, self.class_to_idx[wnid]))
        else:  # test
            test_dir = self.root / "test" / "images"
            # Tiny ImageNet test labels are typically not provided.
            self.class_to_idx = {}
            for img_path in test_dir.glob("*.JPEG"):
                self.samples.append((img_path, -1))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split='{self.split}' under '{self.root}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y

