from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .stats import (
    approx_tensor_entropy,
    covariance_and_spectrum,
    dead_neuron_ratio,
    moments,
    sparsity,
    to_feature_matrix,
)


@dataclass(frozen=True)
class ActivationMonitorConfig:
    feature_mode: str = "gap"  # "gap" or "flatten"
    max_modules: int = 200
    max_dim: int = 1024
    entropy_bins: int = 64
    drift_ema: float = 0.9


class ActivationMonitor:
    """
    Collects layer-wise activation statistics from `InstrumentedModel.hooks.activations`.
    Stores per-module trajectories and drift/instability scores.
    """

    def __init__(self, cfg: ActivationMonitorConfig = ActivationMonitorConfig()):
        self.cfg = cfg
        self._prev_mean: dict[str, float] = {}
        self._drift_ema: dict[str, float] = {}

    def on_step(self, *, trainer, epoch: int, step: int, global_step: int, **_):
        model = getattr(trainer, "model_ref", trainer.model)
        if not hasattr(model, "hooks"):
            return

        acts = getattr(model.hooks, "activations", {})
        if not acts:
            return

        keys = list(acts.keys())[: self.cfg.max_modules]
        out = {}
        for k in keys:
            vlist = acts.get(k, [])
            if not vlist:
                continue
            v = vlist[-1]
            if not isinstance(v, torch.Tensor):
                continue

            m = moments(v)
            ent = approx_tensor_entropy(v, bins=self.cfg.entropy_bins)
            sp = sparsity(v)
            dead = dead_neuron_ratio(v, channel_dim=1 if v.ndim >= 3 else (v.ndim - 1))

            X = to_feature_matrix(v, mode=self.cfg.feature_mode)
            cov = covariance_and_spectrum(X, max_dim=self.cfg.max_dim)

            # Drift vs previous mean
            prev = self._prev_mean.get(k)
            drift = abs(m["mean"] - prev) if prev is not None else 0.0
            self._prev_mean[k] = m["mean"]
            # EMA drift for instability scoring
            ema = self._drift_ema.get(k, drift)
            ema = self.cfg.drift_ema * ema + (1 - self.cfg.drift_ema) * drift
            self._drift_ema[k] = ema

            out[k] = {
                **m,
                "entropy": float(ent),
                "sparsity": float(sp),
                "dead_ratio": float(dead),
                **cov,
                "drift": float(drift),
                "instability": float(ema),
            }

        if not out:
            return

        log_record = trainer._log_record if hasattr(trainer, "_log_record") else trainer.jsonl.log
        add_scalar = trainer._add_scalar if hasattr(trainer, "_add_scalar") else trainer.tb.add_scalar
        log_record(
            {
                "kind": "activation_stats",
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "modules": out,
            }
        )

        # Light TensorBoard summary: log average instability/drift across modules
        drift_vals = [v["drift"] for v in out.values() if "drift" in v]
        inst_vals = [v["instability"] for v in out.values() if "instability" in v]
        if drift_vals:
            drift_mean = float(sum(drift_vals) / len(drift_vals))
            add_scalar("activations/drift_mean", drift_mean, global_step)
            if hasattr(trainer, "update_state"):
                trainer.update_state("activations/drift_mean", drift_mean)
        if inst_vals:
            instability_mean = float(sum(inst_vals) / len(inst_vals))
            add_scalar("activations/instability_mean", instability_mean, global_step)
            if hasattr(trainer, "update_state"):
                trainer.update_state("activations/instability_mean", instability_mean)

    def on_epoch_end(self, *, trainer, epoch: int, global_step: int, val_metrics: Optional[dict[str, float]], **_):
        # Placeholder: heatmaps/trajectories will be built by an offline report generator consuming JSONL.
        return

