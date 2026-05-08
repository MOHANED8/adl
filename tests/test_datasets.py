from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from dtd.datasets.cache import DiskCacheDataset
from dtd.datasets.factory import DatasetSpec, _make_loader
from dtd.datasets.label_noise import LabelNoiseDataset, LabelNoiseSpec


class CountingDataset(Dataset):
    classes = ["zero", "one"]

    def __init__(self):
        self.calls = 0

    def __len__(self):
        return 4

    def __getitem__(self, idx: int):
        self.calls += 1
        return torch.tensor([idx], dtype=torch.float32), idx % 2, {"idx": idx}


@pytest.mark.dataset
def test_disk_cache_dataset_reuses_cached_items(tmp_path):
    base = CountingDataset()
    cached = DiskCacheDataset(base, cache_dir=tmp_path, cache_key="fixed")

    first = cached[2]
    second = cached[2]

    assert torch.equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert base.calls == 1
    assert (tmp_path / "fixed" / "00000002.pt").exists()


@pytest.mark.dataset
@pytest.mark.reproducibility
def test_label_noise_is_seeded_and_preserves_extra_fields():
    base = CountingDataset()
    spec = LabelNoiseSpec(enabled=True, p=1.0, seed=7, num_classes=2)

    a = LabelNoiseDataset(base, spec=spec)
    b = LabelNoiseDataset(base, spec=spec)

    assert [a[i][1] for i in range(len(a))] == [b[i][1] for i in range(len(b))]
    assert a[0][2] == {"idx": 0}


@pytest.mark.dataset
def test_label_noise_requires_num_classes_when_enabled():
    with pytest.raises(ValueError, match="num_classes"):
        LabelNoiseDataset(CountingDataset(), spec=LabelNoiseSpec(enabled=True, p=0.5))


@pytest.mark.dataset
def test_make_loader_is_deterministic_when_shuffle_disabled(toy_dataset):
    spec = DatasetSpec(name="toy", batch_size=3, num_workers=0, shuffle=False, persistent_workers=False)
    loader = _make_loader(toy_dataset, spec, is_train=True)

    batches = list(loader)
    assert torch.equal(batches[0][1], torch.tensor([0, 1, 0]))
    assert len(batches) == 3
