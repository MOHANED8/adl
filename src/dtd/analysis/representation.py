from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    # X,Y: (N,D)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    XT_Y = X.T @ Y
    hsic = np.sum(XT_Y**2)
    denom = np.sqrt(np.sum((X.T @ X) ** 2) * np.sum((Y.T @ Y) ** 2)) + 1e-12
    return float(hsic / denom)


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(1.0 - float(np.dot(a, b) / (na * nb)))


def pca_project(X: np.ndarray, k: int = 50) -> np.ndarray:
    # Simple SVD PCA
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :k] * S[:k]).astype(np.float32)


def svcca_similarity(X: np.ndarray, Y: np.ndarray, k: int = 50) -> float:
    """
    Practical SVCCA approximation:
    PCA -> CCA on top-k components. Uses numpy SVD whitening.
    """
    Xp = pca_project(X, k=k)
    Yp = pca_project(Y, k=k)
    # whiten
    def _whiten(Z):
        C = np.cov(Z, rowvar=False)
        w, V = np.linalg.eigh(C + 1e-6 * np.eye(C.shape[0]))
        W = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        return Z @ W
    Xw = _whiten(Xp)
    Yw = _whiten(Yp)
    M = Xw.T @ Yw / (Xw.shape[0] - 1)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return float(np.mean(s))


@dataclass(frozen=True)
class RepresentationMonitorConfig:
    module_name: str = "fc"  # which hooked module to treat as embedding source
    feature_mode: str = "flatten"
    sample_batches: int = 2
    max_dim: int = 512
    svcca_k: int = 50


class RepresentationMonitor:
    """
    Tracks inter-epoch representation shift using embeddings captured from a selected module.
    Logs CKA + SVCCA approximations between epoch t and t-1.
    """

    def __init__(self, cfg: RepresentationMonitorConfig = RepresentationMonitorConfig()):
        self.cfg = cfg
        self._prev: Optional[np.ndarray] = None

    def on_step(self, **_):
        return

    @torch.no_grad()
    def on_epoch_end(self, *, trainer, epoch: int, global_step: int, val_metrics: Optional[dict[str, float]], **_):
        model = getattr(trainer, "model_ref", trainer.model)
        if not hasattr(model, "hooks"):
            return
        # Run a few batches in eval mode to capture stable reps
        was_train = model.training
        model.eval()
        reps = []
        loader = trainer.val_loader or trainer.train_loader
        for i, batch in enumerate(loader):
            if i >= self.cfg.sample_batches:
                break
            x, y = batch[:2]
            x = x.to(getattr(trainer, "device", "cpu"))
            if hasattr(model, "hooks"):
                model.hooks.clear()
            _ = model(x)
            acts = model.hooks.activations.get(self.cfg.module_name, [])
            if not acts:
                continue
            v = acts[-1]
            if isinstance(v, torch.Tensor):
                v = v.detach().float().flatten(start_dim=1)
                if v.shape[1] > self.cfg.max_dim:
                    v = v[:, : self.cfg.max_dim]
                reps.append(v.to("cpu").numpy())
        if was_train:
            model.train()

        if not reps:
            return
        X = np.concatenate(reps, axis=0)
        rec = {"kind": "repr_epoch", "epoch": epoch, "global_step": global_step}
        if self._prev is not None and self._prev.shape[0] >= 4 and X.shape[0] >= 4:
            n = min(self._prev.shape[0], X.shape[0])
            A = self._prev[:n]
            B = X[:n]
            rec["cka"] = linear_cka(A, B)
            rec["svcca"] = svcca_similarity(A, B, k=min(self.cfg.svcca_k, A.shape[1], B.shape[1]))
            rec["mean_cosine_drift"] = cosine_dist(A.mean(axis=0), B.mean(axis=0))
        self._prev = X

        log_record = trainer._log_record if hasattr(trainer, "_log_record") else trainer.jsonl.log
        add_scalar = trainer._add_scalar if hasattr(trainer, "_add_scalar") else trainer.tb.add_scalar
        log_record(rec)
        if "cka" in rec:
            add_scalar("repr/cka", float(rec["cka"]), global_step)
            if hasattr(trainer, "update_state"):
                trainer.update_state("repr/cka", float(rec["cka"]))
        if "svcca" in rec:
            add_scalar("repr/svcca", float(rec["svcca"]), global_step)

