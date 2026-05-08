from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from dtd.hooks import HookManager, HookSpec


class InstrumentedModel(nn.Module):
    """
    Wraps a model and exposes a unified hook interface:
      - self.hooks.activations
      - self.hooks.gradients
      - self.hooks.attention
      - self.hooks.embeddings
    """

    def __init__(self, model: nn.Module, hook_spec: Optional[HookSpec] = None):
        super().__init__()
        self.model = model
        self.hooks = HookManager(self.model, hook_spec).attach()

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

