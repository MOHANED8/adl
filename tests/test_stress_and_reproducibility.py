from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dtd.hooks import HookManager, HookSpec


@pytest.mark.stress
def test_hook_manager_bounded_storage_under_repeated_forwards():
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    hooks = HookManager(model, HookSpec(module_globs=("*",), max_per_module=2)).attach()

    for _ in range(25):
        _ = model(torch.randn(4, 8))

    assert hooks.activations
    assert all(len(values) <= 2 for values in hooks.activations.values())


@pytest.mark.reproducibility
def test_torch_seed_produces_reproducible_model_initialization():
    torch.manual_seed(2026)
    a = nn.Linear(3, 2)
    torch.manual_seed(2026)
    b = nn.Linear(3, 2)

    assert torch.equal(a.weight, b.weight)
    assert torch.equal(a.bias, b.bias)
