from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dtd.hooks import HookManager, HookSpec


class AttentionLike(nn.Module):
    def forward(self, x):
        attn = torch.ones(x.shape[0], 1, 2, 2, device=x.device)
        return x, attn


@pytest.mark.hooks
def test_hook_manager_captures_outputs_inputs_gradients_and_attention():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), AttentionLike())
    hooks = HookManager(
        model,
        HookSpec(module_globs=("*",), capture_inputs=True, capture_outputs=True, capture_output_grads=True),
    ).attach()

    x = torch.randn(2, 4, requires_grad=True)
    out, _ = model(x)
    out.sum().backward()

    assert "0" in hooks.activations
    assert "0:in" in hooks.embeddings
    assert "0" in hooks.gradients
    assert "2" in hooks.attention
    assert hooks.activations["0"][0].device.type == "cpu"

    hooks.remove()
    assert hooks.handles == []


@pytest.mark.hooks
def test_hook_manager_respects_module_globs_and_max_per_module():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    hooks = HookManager(model, HookSpec(module_globs=("1",), max_per_module=1)).attach()

    _ = model(torch.ones(1, 2))
    _ = model(torch.ones(1, 2) * 2)

    assert "0" not in hooks.activations
    assert list(hooks.activations) == ["1"]
    assert len(hooks.activations["1"]) == 1
