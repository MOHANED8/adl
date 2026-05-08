from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime controls for local, multi-GPU, AMP, and distributed execution."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    multi_gpu: bool = False
    distributed: bool = False
    backend: str = "nccl"
    amp: bool = False
    amp_dtype: str = "float16"
    grad_clip_norm: Optional[float] = None
    seed: int = 42


@dataclass(frozen=True)
class DistributedContext:
    """Process-local distributed execution state."""

    enabled: bool
    world_size: int
    rank: int
    local_rank: int
    device: str

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_runtime(cfg: RuntimeConfig) -> DistributedContext:
    """Initialize distributed state from `torchrun` environment variables when requested."""

    requested_distributed = bool(cfg.distributed)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed_enabled = requested_distributed or env_world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = max(env_world_size, 1)

    device = cfg.device
    if distributed_enabled:
        if not dist.is_initialized():
            dist.init_process_group(backend=cfg.backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
    elif cfg.multi_gpu and torch.cuda.is_available():
        device = "cuda"

    return DistributedContext(
        enabled=distributed_enabled,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
    )


def cleanup_runtime(ctx: DistributedContext) -> None:
    """Tear down the process group when distributed execution is active."""

    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: torch.nn.Module, ctx: DistributedContext, *, multi_gpu: bool) -> torch.nn.Module:
    """Wrap the model for DDP or DataParallel when requested."""

    if ctx.enabled:
        if ctx.device.startswith("cuda"):
            device_id = int(ctx.device.split(":")[1])
            return DistributedDataParallel(model, device_ids=[device_id], output_device=device_id)
        return DistributedDataParallel(model)
    if multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return DataParallel(model)
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module for DataParallel/DDP-wrapped models."""

    return model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model


def prepare_dataloader(loader: DataLoader, ctx: DistributedContext, *, shuffle: bool) -> DataLoader:
    """Clone an existing DataLoader with a DistributedSampler when needed."""

    if not ctx.enabled:
        return loader

    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=shuffle,
        drop_last=loader.drop_last,
    )
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        pin_memory_device=loader.pin_memory_device,
    )


def set_epoch(loader: DataLoader, epoch: int) -> None:
    """Advance the DistributedSampler epoch when one is attached."""

    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)
