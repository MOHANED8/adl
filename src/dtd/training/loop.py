from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dtd.training.distributed import (
    DistributedContext,
    RuntimeConfig,
    prepare_dataloader,
    set_epoch,
    unwrap_model,
    wrap_model,
)
from dtd.training.logging import make_logger
from dtd.training.optim import OptimConfig, build_optimizer


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for the main training loop and runtime behavior."""

    out_dir: str | Path = "runs/exp"
    epochs: int = 10
    optim: OptimConfig = OptimConfig()
    log_every_steps: int = 50
    monitor_every_steps: int = 50
    runtime: RuntimeConfig = RuntimeConfig()
    device: Optional[str] = None
    checkpoint_enabled: bool = True
    checkpoint_every_epochs: int = 1
    save_best: bool = True

    def __post_init__(self) -> None:
        if self.device is not None:
            object.__setattr__(
                self,
                "runtime",
                RuntimeConfig(
                    device=self.device,
                    multi_gpu=self.runtime.multi_gpu,
                    distributed=self.runtime.distributed,
                    backend=self.runtime.backend,
                    amp=self.runtime.amp,
                    amp_dtype=self.runtime.amp_dtype,
                    grad_clip_norm=self.runtime.grad_clip_norm,
                    seed=self.runtime.seed,
                ),
            )


class Trainer:
    """Single-process trainer with optional AMP, DataParallel, and DDP support."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg: TrainConfig,
        dist_ctx: Optional[DistributedContext] = None,
        monitors: Optional[list[Any]] = None,
        tracker: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.dist_ctx = dist_ctx or DistributedContext(
            enabled=False,
            world_size=1,
            rank=0,
            local_rank=0,
            device=cfg.runtime.device,
        )
        self.device = torch.device(self.dist_ctx.device)
        self.device_type = self.device.type
        self.model = model.to(self.device)
        self.model = wrap_model(self.model, self.dist_ctx, multi_gpu=cfg.runtime.multi_gpu)
        self.train_loader = prepare_dataloader(train_loader, self.dist_ctx, shuffle=True)
        self.val_loader = prepare_dataloader(val_loader, self.dist_ctx, shuffle=False) if val_loader is not None else None
        self.cfg = cfg
        self.monitors = monitors or []
        self.tracker = tracker
        self.state: dict[str, float] = {}
        self.state_history: dict[str, list[float]] = {}
        self.best_val_acc = float("-inf")

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.out_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl = make_logger(self.out_dir, "events.jsonl") if self.is_main_process else None
        self.tb = SummaryWriter(log_dir=str(self.out_dir / "tb")) if self.is_main_process else None

        self.opt = build_optimizer(self.model, cfg.optim)
        scaler_enabled = self.cfg.runtime.amp and self.device_type == "cuda" and self.cfg.runtime.amp_dtype == "float16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    @property
    def is_main_process(self) -> bool:
        return self.dist_ctx.is_main_process

    @property
    def model_ref(self) -> nn.Module:
        return unwrap_model(self.model)

    def _autocast_context(self):
        if not self.cfg.runtime.amp:
            return contextlib.nullcontext()
        if self.device_type not in {"cuda", "cpu"}:
            return contextlib.nullcontext()
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return torch.autocast(
            device_type=self.device_type,
            dtype=dtype_map.get(self.cfg.runtime.amp_dtype, torch.float16),
            enabled=True,
        )

    def _log_record(self, record: dict[str, Any]) -> None:
        if self.jsonl is not None:
            self.jsonl.log(record)

    def _add_scalar(self, name: str, value: float, step: int) -> None:
        if self.tb is not None:
            self.tb.add_scalar(name, value, step)

    def update_state(self, key: str, value: float) -> None:
        """Store the latest scalar state for online monitoring and thresholds."""

        self.state[key] = float(value)
        self.state_history.setdefault(key, []).append(float(value))
        if len(self.state_history[key]) > 200:
            self.state_history[key] = self.state_history[key][-200:]

    def get_state(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Return the latest state value for a scalar metric."""

        return self.state.get(key, default)

    def save_checkpoint(self, *, epoch: int, global_step: int, val_metrics: Optional[dict[str, float]]) -> None:
        """Persist a training checkpoint for resuming or artifact inspection."""

        if not self.is_main_process or not self.cfg.checkpoint_enabled:
            return
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": self.model_ref.state_dict(),
            "optimizer_state": self.opt.state_dict(),
            "train_config": asdict(self.cfg),
            "val_metrics": val_metrics or {},
        }
        if (epoch + 1) % max(self.cfg.checkpoint_every_epochs, 1) == 0:
            torch.save(payload, self.checkpoint_dir / f"epoch_{epoch:03d}.pt")
        torch.save(payload, self.checkpoint_dir / "last.pt")
        if self.cfg.save_best and val_metrics is not None:
            val_acc = float(val_metrics.get("val_acc", float("-inf")))
            if val_acc >= self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(payload, self.checkpoint_dir / "best.pt")

    def fit(self) -> None:
        global_step = 0
        for epoch in range(self.cfg.epochs):
            set_epoch(self.train_loader, epoch)
            self.model.train()
            pbar = tqdm(
                self.train_loader,
                desc=f"train epoch {epoch}",
                leave=False,
                disable=not self.is_main_process,
            )
            for step, batch in enumerate(pbar):
                x, y = batch[:2]
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                self.opt.zero_grad(set_to_none=True)
                if hasattr(self.model_ref, "hooks"):
                    self.model_ref.hooks.clear()

                with self._autocast_context():
                    logits = self.model(x)
                    loss = torch.nn.functional.cross_entropy(logits, y)

                self.scaler.scale(loss).backward()
                if self.cfg.runtime.grad_clip_norm is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.runtime.grad_clip_norm)
                self.scaler.step(self.opt)
                self.scaler.update()

                if self.is_main_process and global_step % self.cfg.log_every_steps == 0:
                    rec = {
                        "kind": "train_step",
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "loss": float(loss.detach().cpu()),
                    }
                    self._log_record(rec)
                    self._add_scalar("train/loss", rec["loss"], global_step)
                    if self.tracker is not None:
                        self.tracker.log({"train/loss": rec["loss"]}, step=global_step)

                if self.is_main_process and global_step % self.cfg.monitor_every_steps == 0:
                    for m in self.monitors:
                        m.on_step(
                            trainer=self,
                            epoch=epoch,
                            step=step,
                            global_step=global_step,
                            x=x,
                            y=y,
                            logits=logits,
                            loss=loss,
                        )

                global_step += 1

            val = self.evaluate(epoch=epoch, global_step=global_step) if self.val_loader is not None else None
            if self.is_main_process:
                for m in self.monitors:
                    m.on_epoch_end(trainer=self, epoch=epoch, global_step=global_step, val_metrics=val)
                self.save_checkpoint(epoch=epoch, global_step=global_step, val_metrics=val)

        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self.tracker is not None and self.is_main_process:
            self.tracker.finish()

    @torch.no_grad()
    def evaluate(self, *, epoch: int, global_step: int) -> dict[str, float]:
        """Run validation on the current process and log aggregate metrics."""

        self.model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        for batch in tqdm(self.val_loader, desc=f"val epoch {epoch}", leave=False, disable=not self.is_main_process):
            x, y = batch[:2]
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with self._autocast_context():
                logits = self.model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            loss_sum += float(loss.detach().cpu()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)
        acc = correct / max(total, 1)
        avg_loss = loss_sum / max(total, 1)
        rec = {"kind": "val_epoch", "epoch": epoch, "global_step": global_step, "val_acc": acc, "val_loss": avg_loss}
        self._log_record(rec)
        self._add_scalar("val/acc", acc, global_step)
        self._add_scalar("val/loss", avg_loss, global_step)
        if self.tracker is not None:
            self.tracker.log({"val/acc": acc, "val/loss": avg_loss}, step=global_step)
        return {"val_acc": acc, "val_loss": avg_loss}

