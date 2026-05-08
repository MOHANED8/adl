from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .representation import cosine_dist
from .stats import to_feature_matrix


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    """Estimate ECE using equal-width confidence bins."""

    conf, pred = probs.max(dim=1)
    acc = pred.eq(labels)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if mask.any():
            ece += mask.float().mean() * (acc[mask].float().mean() - conf[mask].mean()).abs()
    return float(ece.item())


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int = 100,
    lr: float = 0.05,
) -> float:
    """Fit a scalar temperature for post-hoc confidence calibration."""

    temperature = torch.nn.Parameter(torch.ones(1, device=logits.device))
    optim = torch.optim.Adam([temperature], lr=lr)
    for _ in range(max(steps, 1)):
        optim.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits / temperature.clamp_min(1e-3), labels)
        loss.backward()
        optim.step()
    return float(temperature.detach().clamp_min(1e-3).item())


def bayesian_calibration_summary(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    bins: int = 10,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> dict[str, float]:
    """Summarize calibration under a Beta-Binomial posterior over bin accuracy."""

    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    posterior_ece = 0.0
    width_sum = 0.0
    weight_sum = 0.0
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf < edges[i + 1] if i < bins - 1 else conf <= edges[i + 1])
        if not mask.any():
            continue
        n = float(mask.sum().item())
        successes = float(correct[mask].sum().item())
        alpha = prior_alpha + successes
        beta = prior_beta + (n - successes)
        posterior_mean = alpha / (alpha + beta)
        posterior_var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
        posterior_std = posterior_var**0.5
        posterior_ece += (n / probs.size(0)) * abs(posterior_mean - float(conf[mask].mean().item()))
        width_sum += (n / probs.size(0)) * 2.0 * posterior_std
        weight_sum += n
    return {
        "bayes_ece": float(posterior_ece),
        "bayes_ci_width": float(width_sum if weight_sum > 0 else 0.0),
    }


def effective_rank(X: torch.Tensor) -> float:
    """Compute the normalized effective rank from singular values."""

    s = torch.linalg.svdvals(X.detach().float().to("cpu"))
    if s.numel() == 0:
        return 0.0
    p = s / s.sum().clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy).item() / max(len(s), 1))


@dataclass(frozen=True)
class UncertaintyMonitorConfig:
    module_name: str = "classifier"
    sample_batches: int = 2
    bins: int = 10
    fit_temperature_scaling: bool = True
    temperature_steps: int = 50
    temperature_lr: float = 0.05


class UncertaintyMonitor:
    """Estimate predictive uncertainty and post-hoc calibration from validation batches."""

    def __init__(self, cfg: UncertaintyMonitorConfig = UncertaintyMonitorConfig()):
        self.cfg = cfg

    @torch.no_grad()
    def on_epoch_end(self, *, trainer, epoch: int, global_step: int, **_) -> None:
        model = trainer.model
        loader = trainer.val_loader or trainer.train_loader
        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= self.cfg.sample_batches:
                break
            x, y = batch[:2]
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            logits = model(x)
            logits_list.append(logits.detach())
            labels_list.append(y.detach())
        if not logits_list:
            return
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        probs = logits.softmax(dim=1)
        entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum(dim=1)).mean().item())
        confidence = float(probs.max(dim=1).values.mean().item())
        margin = float((probs.topk(k=min(2, probs.shape[1]), dim=1).values).diff(dim=1).abs().mean().item())
        ece = expected_calibration_error(probs, labels, bins=self.cfg.bins)
        rec = {
            "kind": "uncertainty_stats",
            "epoch": epoch,
            "global_step": global_step,
            "predictive_entropy": entropy,
            "mean_confidence": confidence,
            "mean_margin": margin,
            "ece": ece,
        }
        rec.update(bayesian_calibration_summary(probs, labels, bins=self.cfg.bins))
        if self.cfg.fit_temperature_scaling:
            with torch.enable_grad():
                temp = fit_temperature(
                    logits,
                    labels,
                    steps=self.cfg.temperature_steps,
                    lr=self.cfg.temperature_lr,
                )
            scaled_probs = (logits / temp).softmax(dim=1)
            rec["temperature"] = temp
            rec["ece_temperature_scaled"] = expected_calibration_error(scaled_probs, labels, bins=self.cfg.bins)
        log_record = trainer._log_record if hasattr(trainer, "_log_record") else trainer.jsonl.log
        add_scalar = trainer._add_scalar if hasattr(trainer, "_add_scalar") else trainer.tb.add_scalar
        log_record(rec)
        add_scalar("uncertainty/predictive_entropy", entropy, global_step)
        add_scalar("uncertainty/ece", ece, global_step)
        if hasattr(trainer, "update_state"):
            trainer.update_state("uncertainty/predictive_entropy", entropy)
            trainer.update_state("uncertainty/ece", ece)

    def on_step(self, **_) -> None:
        return


@dataclass(frozen=True)
class SpecializationMonitorConfig:
    module_name: str = "classifier"
    feature_mode: str = "flatten"
    topk_fraction: float = 0.1


class NeuronSpecializationMonitor:
    """Track concentration of activation energy across neurons within a selected layer."""

    def __init__(self, cfg: SpecializationMonitorConfig = SpecializationMonitorConfig()):
        self.cfg = cfg

    def on_step(self, *, trainer, epoch: int, step: int, global_step: int, **_) -> None:
        model = getattr(trainer, "model_ref", trainer.model)
        acts = getattr(getattr(model, "hooks", None), "activations", {})
        vlist = acts.get(self.cfg.module_name, [])
        if not vlist:
            return
        v = vlist[-1]
        if not isinstance(v, torch.Tensor):
            return
        X = to_feature_matrix(v, mode=self.cfg.feature_mode)
        energy = X.abs().mean(dim=0)
        if energy.numel() == 0:
            return
        sorted_energy, _ = torch.sort(energy, descending=True)
        topk = max(1, int(sorted_energy.numel() * self.cfg.topk_fraction))
        concentration = float(sorted_energy[:topk].sum().item() / sorted_energy.sum().clamp_min(1e-12).item())
        diversity = float(energy.std(unbiased=False).item())
        rec = {
            "kind": "specialization_stats",
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "activation_energy_topk_share": concentration,
            "activation_energy_std": diversity,
        }
        log_record = trainer._log_record if hasattr(trainer, "_log_record") else trainer.jsonl.log
        add_scalar = trainer._add_scalar if hasattr(trainer, "_add_scalar") else trainer.tb.add_scalar
        log_record(rec)
        add_scalar(
            "specialization/topk_share",
            concentration,
            global_step,
        )
        if hasattr(trainer, "update_state"):
            trainer.update_state("specialization/topk_share", concentration)

    def on_epoch_end(self, **_) -> None:
        return


@dataclass(frozen=True)
class CollapseMonitorConfig:
    module_name: str = "classifier"
    feature_mode: str = "flatten"
    collapse_rank_threshold: float = 0.2


class LayerCollapseMonitor:
    """Detect spectral collapse and low-rank degeneration in monitored activations."""

    def __init__(self, cfg: CollapseMonitorConfig = CollapseMonitorConfig()):
        self.cfg = cfg

    def on_step(self, *, trainer, epoch: int, step: int, global_step: int, **_) -> None:
        model = getattr(trainer, "model_ref", trainer.model)
        acts = getattr(getattr(model, "hooks", None), "activations", {})
        vlist = acts.get(self.cfg.module_name, [])
        if not vlist:
            return
        v = vlist[-1]
        if not isinstance(v, torch.Tensor):
            return
        X = to_feature_matrix(v, mode=self.cfg.feature_mode)
        rank = effective_rank(X)
        collapsed = float(rank < self.cfg.collapse_rank_threshold)
        rec = {
            "kind": "collapse_stats",
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "effective_rank": rank,
            "collapse_score": 1.0 - rank,
            "layer_collapsed": bool(collapsed),
        }
        log_record = trainer._log_record if hasattr(trainer, "_log_record") else trainer.jsonl.log
        add_scalar = trainer._add_scalar if hasattr(trainer, "_add_scalar") else trainer.tb.add_scalar
        log_record(rec)
        add_scalar("collapse/effective_rank", rank, global_step)
        if hasattr(trainer, "update_state"):
            trainer.update_state("collapse/effective_rank", rank)
            trainer.update_state("collapse/score", 1.0 - rank)

    def on_epoch_end(self, **_) -> None:
        return


@dataclass(frozen=True)
class SelfSupervisedMonitorConfig:
    module_name: str = "classifier"
    noise_std: float = 0.05


class SelfSupervisedConsistencyMonitor:
    """Compare representations under lightweight view augmentation as a self-supervised probe."""

    def __init__(self, cfg: SelfSupervisedMonitorConfig = SelfSupervisedMonitorConfig()):
        self.cfg = cfg

    @torch.no_grad()
    def on_step(self, *, trainer, epoch: int, step: int, global_step: int, x, **_) -> None:
        model = getattr(trainer, "model_ref", trainer.model)
        if not hasattr(model, "hooks"):
            return
        model.hooks.clear()
        _ = model(x)
        vlist = model.hooks.activations.get(self.cfg.module_name, [])
        if not vlist or not isinstance(vlist[-1], torch.Tensor):
            return
        base = to_feature_matrix(vlist[-1], mode="flatten").detach().float()
        x_view = (x + torch.randn_like(x) * self.cfg.noise_std).clamp(x.min().item(), x.max().item())
        model.hooks.clear()
        _ = model(x_view)
        aug_list = model.hooks.activations.get(self.cfg.module_name, [])
        if not aug_list or not isinstance(aug_list[-1], torch.Tensor):
            return
        aug = to_feature_matrix(aug_list[-1], mode="flatten").detach().float()
        cos = F.cosine_similarity(base, aug, dim=1).mean().item()
        drift = cosine_dist(base.mean(dim=0).cpu().numpy(), aug.mean(dim=0).cpu().numpy())
        rec = {
            "kind": "ssl_consistency",
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "view_cosine_similarity": float(cos),
            "view_mean_drift": float(drift),
        }
        log_record = trainer._log_record if hasattr(trainer, "_log_record") else trainer.jsonl.log
        add_scalar = trainer._add_scalar if hasattr(trainer, "_add_scalar") else trainer.tb.add_scalar
        log_record(rec)
        add_scalar("ssl/view_cosine_similarity", float(cos), global_step)
        if hasattr(trainer, "update_state"):
            trainer.update_state("ssl/view_cosine_similarity", float(cos))

    def on_epoch_end(self, **_) -> None:
        return
