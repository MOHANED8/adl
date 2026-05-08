from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Optional


@dataclass(frozen=True)
class EarlyWarningConfig:
    drift_threshold: float = 0.05
    instability_threshold: float = 0.05
    grad_exploding: bool = True
    sharpness_threshold: float = 0.0
    min_risk_to_alert: float = 0.6
    warmup_steps: int = 5
    threshold_std_scale: float = 2.0


class EarlyWarningSystem:
    """
    Real-time monitor that emits alerts to:
      - console (print)
      - JSONL (via trainer.jsonl already)
      - dashboard (optional streamlit app consumes events.jsonl; added separately)
    """

    def __init__(self, cfg: EarlyWarningConfig = EarlyWarningConfig(), *, mode: str = "console"):
        self.cfg = cfg
        self.mode = mode
        self._history: dict[str, list[float]] = {}

    def _adaptive_threshold(self, key: str, value: float, fallback: float) -> float:
        history = self._history.setdefault(key, [])
        history.append(float(value))
        if len(history) > 200:
            del history[:-200]
        if len(history) < self.cfg.warmup_steps:
            return fallback
        return mean(history) + self.cfg.threshold_std_scale * pstdev(history)

    def on_step(self, *, trainer, epoch: int, step: int, global_step: int, **_):
        metrics = getattr(trainer, "state", {})
        if not metrics:
            return
        breaches = []
        drift = float(metrics.get("activations/drift_mean", 0.0))
        instability = float(metrics.get("activations/instability_mean", 0.0))
        grad_norm = float(metrics.get("grads/norm", 0.0))
        predictive_entropy = float(metrics.get("uncertainty/predictive_entropy", 0.0))
        collapse_score = float(metrics.get("collapse/score", 0.0))
        ssl_similarity = float(metrics.get("ssl/view_cosine_similarity", 1.0))

        if drift > self._adaptive_threshold("drift", drift, self.cfg.drift_threshold):
            breaches.append("drift")
        if instability > self._adaptive_threshold("instability", instability, self.cfg.instability_threshold):
            breaches.append("instability")
        if grad_norm > self._adaptive_threshold("grad_norm", grad_norm, 10.0):
            breaches.append("grad_norm")
        if predictive_entropy > self._adaptive_threshold("entropy", predictive_entropy, 1.0):
            breaches.append("uncertainty")
        if collapse_score > self._adaptive_threshold("collapse", collapse_score, 0.75):
            breaches.append("collapse")
        if ssl_similarity < 1.0 - self._adaptive_threshold("ssl_similarity", 1.0 - ssl_similarity, 0.25):
            breaches.append("ssl_divergence")

        risk = min(1.0, len(breaches) / 4.0)
        rec = {
            "kind": "online_warning_step",
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "adaptive_risk": float(risk),
            "breaches": breaches,
            "recommendation": "intervene" if risk >= self.cfg.min_risk_to_alert else "continue",
        }
        log_record = trainer._log_record if hasattr(trainer, "_log_record") else trainer.jsonl.log
        add_scalar = trainer._add_scalar if hasattr(trainer, "_add_scalar") else trainer.tb.add_scalar
        log_record(rec)
        add_scalar("warnings/adaptive_risk", float(risk), global_step)
        if hasattr(trainer, "update_state"):
            trainer.update_state("warnings/adaptive_risk", float(risk))
        if self.mode == "console" and breaches:
            print(f"[warning] step={global_step} risk={risk:.2f} breaches={','.join(breaches)}")

    def on_epoch_end(self, *, trainer, epoch: int, global_step: int, val_metrics: Optional[dict[str, float]], **_):
        risk = float(getattr(trainer, "get_state", lambda *_: 0.0)("warnings/adaptive_risk", 0.0) or 0.0)
        rec = {
            "kind": "early_warning_epoch",
            "epoch": epoch,
            "global_step": global_step,
            "adaptive_risk": risk,
            "recommendation": "intervene" if risk >= self.cfg.min_risk_to_alert else "continue",
        }
        if val_metrics and "val_loss" in val_metrics:
            rec["val_loss"] = float(val_metrics["val_loss"])
        log_record = trainer._log_record if hasattr(trainer, "_log_record") else trainer.jsonl.log
        log_record(rec)
        if self.mode == "console":
            print(f"[warning] epoch={epoch} recommendation={rec['recommendation']}")

