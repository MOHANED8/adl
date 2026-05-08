from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from dtd.datasets import sources
from dtd.datasets.sources import build_dataset
from dtd.datasets.tiny_imagenet import TinyImageNet
from dtd.datasets.wilds_wrapper import WILDSImageDataset


class DummyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.classes = ["a", "b"]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor([idx]), 0


@pytest.mark.dataset
def test_build_dataset_dispatches_and_wraps_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(sources, "_cifar", lambda root, split, which, transform: DummyDataset(root, split=split, which=which))

    ds = build_dataset("cifar10", root=tmp_path, split="train", cache_dir=tmp_path / "cache")

    assert len(ds) == 1
    assert ds.cache_dir == tmp_path / "cache"
    assert ds.base.kwargs["which"] == "cifar10"


@pytest.mark.dataset
def test_build_dataset_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataset("unknown", root=tmp_path, split="train")


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path)


@pytest.mark.dataset
def test_tiny_imagenet_indexes_train_val_and_test_splits(tmp_path):
    root = tmp_path / "tiny-imagenet-200"
    root.mkdir(parents=True)
    (root / "wnids.txt").write_text("n00000001\nn00000002\n", encoding="utf-8")
    _write_image(root / "train" / "n00000001" / "images" / "a.JPEG")
    _write_image(root / "val" / "images" / "b.JPEG")
    (root / "val" / "val_annotations.txt").write_text("b.JPEG\tn00000002\t0\t0\t1\t1\n", encoding="utf-8")
    _write_image(root / "test" / "images" / "c.JPEG")

    train = TinyImageNet(root, split="train", download=False)
    val = TinyImageNet(root, split="val", download=False)
    test = TinyImageNet(root, split="test", download=False)

    assert len(train) == len(val) == len(test) == 1
    assert train[0][1] == 0
    assert val[0][1] == 1
    assert test[0][1] == -1


@pytest.mark.dataset
def test_tiny_imagenet_reports_missing_validation_metadata(tmp_path):
    root = tmp_path / "tiny-imagenet-200"
    (root / "val" / "images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="val_annotations"):
        TinyImageNet(root, split="val", download=False)


@pytest.mark.dataset
def test_wilds_wrapper_validates_splits_and_metadata(monkeypatch, tmp_path):
    class FakeSubset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return "x", 1, {"site": 2}

    class FakeWildsDataset:
        split_dict = {"train": 0, "val": 1}

        def get_subset(self, split_key, transform=None):
            self.split_key = split_key
            return FakeSubset()

    fake_wilds = ModuleType("wilds")
    fake_wilds.get_dataset = lambda dataset, root_dir, download: FakeWildsDataset()
    monkeypatch.setitem(sys.modules, "wilds", fake_wilds)

    ds = WILDSImageDataset(dataset_name="demo", root=tmp_path, split="valid", return_metadata=True)
    assert len(ds) == 1
    assert ds[0] == ("x", 1, {"site": 2})

    with pytest.raises(ValueError, match="not available"):
        WILDSImageDataset(dataset_name="demo", root=tmp_path, split="ood")


@pytest.mark.dataset
def test_wilds_wrapper_reports_missing_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "wilds", None)
    with pytest.raises(RuntimeError, match="Missing dependency"):
        WILDSImageDataset(dataset_name="demo", root=tmp_path, split="train")
