from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


def build_dataset(
    name: str,
    *,
    root: str | Path,
    split: str,
    transform=None,
    cache_dir: Optional[str | Path] = None,
) -> Dataset:
    name = name.lower().strip()
    root = Path(root)
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    ds: Dataset
    if name in {"cifar10", "cifar-10"}:
        ds = _cifar(root, split=split, which="cifar10", transform=transform)
    elif name in {"cifar100", "cifar-100"}:
        ds = _cifar(root, split=split, which="cifar100", transform=transform)
    elif name in {"synthetic_shapes", "synthetic-shapes"}:
        ds = _synthetic_shapes(root, split=split, transform=transform)
    elif name in {"tiny_imagenet", "tiny-imagenet", "tinyimagenet"}:
        ds = _tiny_imagenet(root, split=split, transform=transform)
    elif name in {"imagenet", "ilsvrc2012"}:
        ds = _imagenet(root, split=split, transform=transform)
    elif name.startswith("wilds:"):
        wilds_name = name.split(":", 1)[1]
        ds = _wilds(root, dataset_name=wilds_name, split=split, transform=transform)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: cifar10, cifar100, synthetic_shapes, tiny_imagenet, imagenet, wilds:<name>."
        )

    if cache_dir is not None:
        from .cache import DiskCacheDataset

        ds = DiskCacheDataset(ds, cache_dir=cache_dir)
    return ds



def _cifar(root: Path, *, split: str, which: str, transform):
    from torchvision.datasets import CIFAR10, CIFAR100

    train = split in {"train", "tr"}
    if which == "cifar10":
        return CIFAR10(root / "cifar10", train=train, download=True, transform=transform)
    return CIFAR100(root / "cifar100", train=train, download=True, transform=transform)


def _synthetic_shapes(root: Path, *, split: str, transform):
    from .synthetic import SyntheticShapesDataset

    norm_split = "val" if split in {"valid", "validation"} else split
    return SyntheticShapesDataset(root / "synthetic_shapes", split=norm_split, transform=transform)


def _tiny_imagenet(root: Path, *, split: str, transform):
    """
    Tiny ImageNet (200 classes).
    This loader expects the common directory layout after download+extract:
      tiny-imagenet-200/
        train/<class>/images/*.JPEG
        val/images/*.JPEG + val/val_annotations.txt
        test/images/*.JPEG (labels often unavailable)
    For convenience, we treat:
      split in {train,val,test}
    """
    from .tiny_imagenet import TinyImageNet

    return TinyImageNet(root / "tiny-imagenet-200", split=split, download=True, transform=transform)


def _imagenet(root: Path, *, split: str, transform):
    """
    ImageNet is optional and typically cannot be auto-downloaded due to licensing.
    We support it when the user has already prepared the folder in torchvision's ImageFolder format:
      imagenet/train/<class>/*.JPEG
      imagenet/val/<class>/*.JPEG
    """
    from torchvision.datasets import ImageFolder

    split = "train" if split in {"train", "tr"} else "val" if split in {"val", "valid"} else split
    base = root / "imagenet" / split
    return ImageFolder(base, transform=transform)


def _wilds(root: Path, *, dataset_name: str, split: str, transform):
    from .wilds_wrapper import WILDSImageDataset

    return WILDSImageDataset(
        dataset_name=dataset_name,
        root=root / "wilds",
        split=split,
        transform=transform,
    )

