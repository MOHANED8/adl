from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def deterministic_seed():
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


class ToyClassificationDataset(Dataset):
    def __init__(self, n: int = 8, *, in_features: int = 4, num_classes: int = 2):
        self.x = torch.arange(n * in_features, dtype=torch.float32).reshape(n, in_features) / 10.0
        self.y = torch.arange(n, dtype=torch.long) % num_classes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


@pytest.fixture
def toy_dataset():
    return ToyClassificationDataset()


@pytest.fixture
def sample_events_file(tmp_path: Path) -> Path:
    path = tmp_path / "events.jsonl"
    rows = [
        {"kind": "train_step", "epoch": 0, "global_step": 0, "loss": 1.2},
        {"kind": "gradient_stats", "epoch": 0, "global_step": 0, "grad_norm": 0.5, "grad_var": 0.1},
        {
            "kind": "activation_stats",
            "epoch": 0,
            "global_step": 0,
            "modules": {"features.0": {"mean": 0.1, "var": 0.2, "instability": 0.0}},
        },
        {"kind": "val_epoch", "epoch": 0, "global_step": 1, "val_acc": 0.9, "val_loss": 0.3},
        {"kind": "val_epoch", "epoch": 1, "global_step": 2, "val_acc": 0.8, "val_loss": 0.4},
        "not-json",
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row if isinstance(row, str) else json.dumps(row))
            f.write("\n")
    return path
