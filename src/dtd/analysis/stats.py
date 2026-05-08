from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def safe_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def approx_tensor_entropy(x: torch.Tensor, bins: int = 64) -> float:
    x = x.flatten().detach()
    if x.numel() == 0:
        return float("nan")
    # Histogram-based entropy on CPU for stability
    x = x.to("cpu")
    mn = float(x.min())
    mx = float(x.max())
    if mn == mx:
        return 0.0
    hist = torch.histc(x, bins=bins, min=mn, max=mx)
    p = hist / hist.sum().clamp_min(1.0)
    return float(safe_entropy(p).item())


def moments(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return {"mean": float("nan"), "var": float("nan"), "skew": float("nan"), "kurtosis": float("nan")}
    mean = x.mean()
    var = x.var(unbiased=False)
    std = var.sqrt().clamp_min(1e-12)
    z = (x - mean) / std
    skew = (z**3).mean()
    kurt = (z**4).mean() - 3.0
    return {"mean": float(mean.item()), "var": float(var.item()), "skew": float(skew.item()), "kurtosis": float(kurt.item())}


def sparsity(x: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.detach()
    if x.numel() == 0:
        return float("nan")
    return float((x.abs() <= eps).float().mean().item())


def dead_neuron_ratio(x: torch.Tensor, *, channel_dim: int = 1, eps: float = 1e-12) -> float:
    """
    For activations shaped (B,C,...) or (B,T,C) etc.
    A "dead neuron" is a channel whose absolute activation is ~0 for all items in batch.
    """
    x = x.detach()
    if x.numel() == 0 or x.ndim < 2:
        return float("nan")
    dims = [d for d in range(x.ndim) if d != channel_dim]
    # channel is dead if max abs across batch+spatial is <= eps
    m = x.abs().amax(dim=dims)
    return float((m <= eps).float().mean().item())


def to_feature_matrix(x: torch.Tensor, *, mode: str = "gap") -> torch.Tensor:
    """
    Convert activation tensor to (N, D) for covariance/spectrum.
    - gap: global average pool across non-batch/channel dims -> (B, C)
    - flatten: flatten everything except batch -> (B, D)
    """
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if mode == "gap" and x.ndim >= 3:
        # assume channels dim=1 for CNN-style tensors
        dims = list(range(2, x.ndim))
        return x.mean(dim=dims)
    return x.flatten(start_dim=1)


def covariance_and_spectrum(X: torch.Tensor, *, max_dim: int = 1024) -> dict[str, float]:
    """
    Computes covariance trace, top singular values, anisotropy proxies.
    Uses feature matrix X: (N, D).
    """
    X = X.detach().float()
    if X.numel() == 0:
        return {"cov_trace": float("nan"), "sv_max": float("nan"), "sv_sum": float("nan"), "anisotropy": float("nan")}
    # Subsample dims for cost
    if X.shape[1] > max_dim:
        X = X[:, :max_dim]
    X = X - X.mean(dim=0, keepdim=True)
    # Covariance trace
    cov_trace = float((X.var(dim=0, unbiased=False).sum()).item())
    # Spectrum via SVD of centered data matrix (cheaper than cov SVD for N << D sometimes)
    try:
        s = torch.linalg.svdvals(X.to("cpu"))
        sv_max = float(s.max().item()) if s.numel() else float("nan")
        sv_sum = float(s.sum().item()) if s.numel() else float("nan")
        # Feature anisotropy proxy: sv_max / mean(sv)
        anis = float((sv_max / (float(s.mean().item()) + 1e-12)) if s.numel() else float("nan"))
    except Exception:
        sv_max = float("nan")
        sv_sum = float("nan")
        anis = float("nan")
    return {"cov_trace": cov_trace, "sv_max": sv_max, "sv_sum": sv_sum, "anisotropy": anis}

