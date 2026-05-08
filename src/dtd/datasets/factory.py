from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: str | Path = "data"
    split: Split = "train"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    shuffle: Optional[bool] = None
    drop_last: bool = False
    cache_dir: Optional[str | Path] = None

    # Augmentation / preprocessing
    image_size: int = 224
    normalize: bool = True
    aug: Literal["none", "cifar_basic", "imagenet_basic", "strong"] = "imagenet_basic"

    # Optional label noise (applied after dataset creation; affects supervised training targets)
    label_noise_p: float = 0.0
    label_noise_seed: int = 0


def _default_transforms(spec: DatasetSpec, *, is_train: bool) -> Callable:
    size = spec.image_size

    if spec.aug == "none":
        t = [transforms.Resize((size, size)), transforms.ToTensor()]
    elif spec.aug == "cifar_basic":
        if is_train:
            t = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            t = [transforms.ToTensor()]
    elif spec.aug == "imagenet_basic":
        if is_train:
            t = [
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            t = [
                transforms.Resize(int(size * 1.14)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
    else:  # strong
        if is_train:
            t = [
                transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
            ]
        else:
            t = [
                transforms.Resize(int(size * 1.14)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]

    if spec.normalize:
        # Reasonable defaults; dataset-specific overrides can be added later.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        t.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(t)


def _make_loader(ds, spec: DatasetSpec, *, is_train: bool) -> DataLoader:
    shuffle = spec.shuffle if spec.shuffle is not None else is_train
    return DataLoader(
        ds,
        batch_size=spec.batch_size,
        shuffle=shuffle,
        num_workers=spec.num_workers,
        pin_memory=spec.pin_memory and torch.cuda.is_available(),
        persistent_workers=spec.persistent_workers and spec.num_workers > 0,
        prefetch_factor=spec.prefetch_factor if spec.num_workers > 0 else None,
        drop_last=spec.drop_last and is_train,
    )


def build_dataloaders(
    train: DatasetSpec,
    val: Optional[DatasetSpec] = None,
    test: Optional[DatasetSpec] = None,
):
    """
    Returns dict with keys: train/val/test -> DataLoader (if requested).

    Supported datasets (spec.name):
      - cifar10, cifar100
      - tiny_imagenet
      - imagenet (optional; expects ImageNet layout if already present)
      - wilds:<dataset_name> (e.g. wilds:camelyon17, wilds:iwildcam)
    """
    from .sources import build_dataset  # local import to keep deps modular
    from .label_noise import LabelNoiseDataset, LabelNoiseSpec

    loaders = {}
    for split_name, spec in [("train", train), ("val", val), ("test", test)]:
        if spec is None:
            continue
        is_train = split_name == "train"
        tfm = _default_transforms(spec, is_train=is_train)
        ds = build_dataset(spec.name, root=spec.root, split=spec.split, transform=tfm, cache_dir=spec.cache_dir)
        # Apply label noise only for train split by default (can be overridden by setting on val/test spec)
        if getattr(spec, "label_noise_p", 0.0) and float(spec.label_noise_p) > 0:
            # Try to infer classes from torchvision datasets; otherwise user should set explicitly.
            num_classes = getattr(ds, "classes", None)
            nc = len(num_classes) if isinstance(num_classes, list) else None
            ln = LabelNoiseSpec(
                enabled=True,
                p=float(spec.label_noise_p),
                seed=int(spec.label_noise_seed),
                num_classes=nc,
            )
            ds = LabelNoiseDataset(ds, spec=ln)
        loaders[split_name] = _make_loader(ds, spec, is_train=is_train)
    return loaders

