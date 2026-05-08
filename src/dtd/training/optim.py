from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class OptimConfig:
    name: Literal["adamw", "sgd", "adam"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # for SGD


def build_optimizer(model, cfg: OptimConfig):
    name = str(cfg.name).lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.weight_decay),
        )
    raise ValueError(f"Unknown optimizer {cfg.name}")

