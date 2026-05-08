from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class GradientMonitorConfig:
    ema: float = 0.95
    hutchinson_samples: int = 1
    sharpness_eps: float = 1e-3
    max_params: int = 5000  # for cosine similarity vectorization


def _flat_grad(model, *, max_params: int):
    vecs = []
    n = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().flatten()
        if g.numel() == 0:
            continue
        vecs.append(g.to("cpu"))
        n += g.numel()
        if n >= max_params:
            break
    if not vecs:
        return None
    v = torch.cat(vecs)
    return v


class GradientMonitor:
    """
    Tracks gradient norms/variance/cosine similarity, plus rough sharpness and Hessian-trace proxy.
    """

    def __init__(self, cfg: GradientMonitorConfig = GradientMonitorConfig()):
        self.cfg = cfg
        self._prev_g: Optional[torch.Tensor] = None
        self._ema_norm: Optional[float] = None

    def on_step(self, *, trainer, epoch: int, step: int, global_step: int, x, y, loss, logits, **_):
        model = trainer.model

        g = _flat_grad(model, max_params=self.cfg.max_params)
        if g is None:
            return
        g2 = float((g**2).mean().item())
        gnorm = float(g.norm().item())
        gvar = float(g.var(unbiased=False).item())

        cos = None
        if self._prev_g is not None and self._prev_g.numel() == g.numel():
            denom = (self._prev_g.norm() * g.norm()).clamp_min(1e-12)
            cos = float((torch.dot(self._prev_g, g) / denom).item())
        self._prev_g = g

        ema = self._ema_norm if self._ema_norm is not None else gnorm
        ema = self.cfg.ema * ema + (1 - self.cfg.ema) * gnorm
        self._ema_norm = ema

        exploding = bool(gnorm > 10.0 * max(ema, 1e-12))

        # Sharpness proxy: loss(theta + eps * g/||g||) - loss(theta)
        sharp = None
        try:
            eps = self.cfg.sharpness_eps
            with torch.no_grad():
                scale = eps / (gnorm + 1e-12)
                deltas = []
                params = []
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    dp = p.grad.detach()
                    params.append(p)
                    deltas.append(scale * dp)
                for p, d in zip(params, deltas):
                    p.add_(d)
            logits2 = model(x)
            loss2 = torch.nn.functional.cross_entropy(logits2, y)
            sharp = float((loss2.detach() - loss.detach()).cpu().item())
        except Exception:
            sharp = None
        finally:
            # revert
            try:
                with torch.no_grad():
                    for p, d in zip(params, deltas):  # type: ignore[name-defined]
                        p.sub_(d)
            except Exception:
                pass

        # Hutchinson trace estimator for Hessian of loss wrt params: E[v^T H v]
        htrace = None
        try:
            # This requires a graph; we approximate using grads already computed by re-creating loss.
            model.zero_grad(set_to_none=True)
            logits_h = model(x)
            loss_h = torch.nn.functional.cross_entropy(logits_h, y)
            grads = torch.autograd.grad(loss_h, [p for p in model.parameters() if p.requires_grad], create_graph=True)
            trace_est = 0.0
            for _ in range(self.cfg.hutchinson_samples):
                v = [torch.randn_like(g) for g in grads]
                gv = sum((g * vi).sum() for g, vi in zip(grads, v))
                hv = torch.autograd.grad(gv, [p for p in model.parameters() if p.requires_grad], retain_graph=True)
                trace_est += float(sum((hi * vi).sum().detach().cpu().item() for hi, vi in zip(hv, v)))
            htrace = trace_est / max(self.cfg.hutchinson_samples, 1)
        except Exception:
            htrace = None
        finally:
            model.zero_grad(set_to_none=True)

        rec = {
            "kind": "gradient_stats",
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "grad_norm": gnorm,
            "grad_mean_sq": g2,
            "grad_var": gvar,
            "grad_cos_prev": cos,
            "exploding": exploding,
            "sharpness_proxy": sharp,
            "hessian_trace_proxy": htrace,
        }
        log_record = trainer._log_record if hasattr(trainer, "_log_record") else trainer.jsonl.log
        add_scalar = trainer._add_scalar if hasattr(trainer, "_add_scalar") else trainer.tb.add_scalar
        log_record(rec)

        add_scalar("grads/norm", gnorm, global_step)
        if hasattr(trainer, "update_state"):
            trainer.update_state("grads/norm", gnorm)
        if cos is not None:
            add_scalar("grads/cos_prev", cos, global_step)
        if sharp is not None:
            add_scalar("opt/sharpness_proxy", sharp, global_step)
            if hasattr(trainer, "update_state"):
                trainer.update_state("opt/sharpness_proxy", sharp)
        if htrace is not None:
            add_scalar("opt/hessian_trace_proxy", htrace, global_step)
            if hasattr(trainer, "update_state"):
                trainer.update_state("opt/hessian_trace_proxy", htrace)
        if hasattr(trainer, "update_state"):
            trainer.update_state("grads/exploding", float(exploding))

    def on_epoch_end(self, **_):
        return

