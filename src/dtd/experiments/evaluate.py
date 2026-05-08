from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dtd.causal import granger_matrix
from dtd.meta.dataset import MetaSequenceDataset, MetaSequenceSpec


@dataclass(frozen=True)
class EvalSpec:
    window: int = 10
    horizon: int = 5
    collapse_drop: float = 0.05
    granger_maxlag: int = 3


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Simple ROC-AUC implementation without extra deps
    y_true = y_true.astype(int)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    P = max(int(y_true.sum()), 1)
    N = max(int((1 - y_true).sum()), 1)
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    tpr = tps / P
    fpr = fps / N
    # trapezoid
    return float(np.trapezoid(tpr, fpr))


def _lead_time_epochs(ds: MetaSequenceDataset, y_pred: np.ndarray, thresh: float = 0.5) -> dict[str, float]:
    """
    Lead time proxy: among positives, how many epochs before the first collapse epoch
    does the predictor cross threshold.
    """
    pos = y_pred[:, 0] >= thresh
    yt = []
    for i in range(len(ds)):
        _, y_overfit, y_when = ds[i]
        yt.append([float(y_overfit.item()), float(y_when.item())])
    yt = np.asarray(yt)
    mask = yt[:, 0] > 0.5
    if not mask.any():
        return {"lead_time_mean": float("nan")}
    # Here y_when is 0..h-1 relative; crossing happens at end of window by construction.
    # So lead-time in epochs is (horizon - y_when).
    lead = (ds.spec.horizon - yt[mask, 1]).astype(float)
    return {"lead_time_mean": float(np.mean(lead))}


def evaluate_run(run_dir: str | Path, spec: EvalSpec = EvalSpec()) -> dict[str, Any]:
    run_dir = Path(run_dir)
    events = run_dir / "events.jsonl"
    ds = MetaSequenceDataset(
        events,
        spec=MetaSequenceSpec(window=spec.window, horizon=spec.horizon, collapse_drop=spec.collapse_drop),
    )
    if len(ds) == 0:
        return {
            "run_dir": str(run_dir),
            "error": "not_enough_epochs_for_window_horizon",
            "needed_epochs": int(spec.window + spec.horizon + 1),
        }

    # Baseline signal model: risk score = normalized combination of a few internal metrics
    # (This is a placeholder until the learned meta-model is trained across multiple runs.)
    X = ds.X
    # pick some common keys if present
    keys = ds.feat_keys
    def col(name: str) -> Optional[np.ndarray]:
        if name not in keys:
            return None
        return X[:, keys.index(name)]

    risk = np.zeros(X.shape[0], dtype=float)
    for k in ["gradient_stats_grad_norm", "gradient_stats_hessian_trace_proxy", "repr_epoch_cka", "act_instability_mean"]:
        c = col(k)
        if c is None:
            continue
        c = np.nan_to_num(c, nan=0.0)
        if np.std(c) > 1e-12:
            risk += (c - np.mean(c)) / (np.std(c) + 1e-12)
    # squashed
    risk = 1.0 / (1.0 + np.exp(-risk))

    # Build labels at window endpoints
    ys = np.asarray([[float(ds[i][1].item()), float(ds[i][2].item())] for i in range(len(ds))], dtype=float)
    y_true = ys[:, 0]
    auc = _auc_roc(y_true, risk[: len(y_true)])

    # False positives at threshold
    thr = 0.5
    yhat = (risk[: len(y_true)] >= thr).astype(int)
    fp = int(((yhat == 1) & (y_true < 0.5)).sum())
    tn = int(((yhat == 0) & (y_true < 0.5)).sum())
    fpr = fp / max(fp + tn, 1)

    # Correlation with generalization: corr(risk, val_acc) using available val_acc series
    val_acc = ds.val_acc
    corr = float("nan")
    if np.isfinite(val_acc).sum() >= 3 and np.std(val_acc[np.isfinite(val_acc)]) > 1e-12:
        a = np.nan_to_num(val_acc, nan=np.nanmean(val_acc))
        corr = float(np.corrcoef(risk, -a)[0, 1])  # higher risk should correlate with lower acc

    # Granger causality: feature -> (1 - val_acc)
    y_series = 1.0 - np.nan_to_num(val_acc, nan=np.nanmean(val_acc))
    pvals = granger_matrix(ds.X, y_series, maxlag=spec.granger_maxlag)
    top = np.argsort(pvals)[: min(10, len(pvals))]
    top_feats = [{"feature": ds.feat_keys[i], "p": float(pvals[i])} for i in top]

    out = {
        "run_dir": str(run_dir),
        "auc_internal_signals": float(auc),
        "false_positive_rate@0.5": float(fpr),
        "lead_time": _lead_time_epochs(ds, y_pred=risk.reshape(-1, 1)),
        "risk_valacc_corr": corr,
        "granger_top": top_feats,
    }
    return out

