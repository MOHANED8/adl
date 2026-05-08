from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from dtd.training import TrainConfig
from dtd.training.distributed import (
    DistributedContext,
    RuntimeConfig,
    cleanup_runtime,
    prepare_dataloader,
    set_epoch,
    setup_runtime,
    unwrap_model,
    wrap_model,
)
from dtd.training.logging import configure_logging, get_logger
from dtd.training.optim import OptimConfig


@pytest.mark.unit
def test_train_config_device_alias_updates_runtime():
    cfg = TrainConfig(
        out_dir="runs/test",
        epochs=1,
        optim=OptimConfig(name="sgd", lr=0.1),
        device="cpu",
    )
    assert cfg.runtime.device == "cpu"


@pytest.mark.unit
def test_runtime_setup_and_cleanup_local_mode():
    ctx = setup_runtime(RuntimeConfig(device="cpu", distributed=False, multi_gpu=False))
    cleanup_runtime(ctx)

    assert ctx.enabled is False
    assert ctx.world_size == 1
    assert ctx.device == "cpu"


@pytest.mark.unit
def test_prepare_dataloader_builds_distributed_sampler():
    ds = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    ctx = DistributedContext(enabled=True, world_size=2, rank=0, local_rank=0, device="cpu")

    dist_loader = prepare_dataloader(loader, ctx, shuffle=True)
    set_epoch(dist_loader, 1)

    assert isinstance(dist_loader.sampler, DistributedSampler)
    assert dist_loader.batch_size == 2


@pytest.mark.unit
def test_wrap_and_unwrap_model_leave_local_model_usable():
    model = nn.Linear(4, 2)
    ctx = DistributedContext(enabled=False, world_size=1, rank=0, local_rank=0, device="cpu")

    wrapped = wrap_model(model, ctx, multi_gpu=False)

    assert unwrap_model(wrapped) is model


@pytest.mark.unit
def test_logging_helpers_configure_and_return_logger():
    configure_logging("INFO")
    logger = get_logger("dtd.tests.runtime")

    assert isinstance(logger, logging.Logger)
