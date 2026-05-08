from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MetaSequenceSpec:
    window: int = 10  # timesteps per sample
    horizon: int = 5  # predict collapse within next horizon epochs
    collapse_drop: float = 0.05  # val_acc drop threshold


def _flatten_metrics(record: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in record.items():
        if isinstance(v, (int, float)) and k not in {"epoch", "step", "global_step"}:
            out[k] = float(v)
    return out


class MetaSequenceDataset(Dataset):
    """
    Builds sequences from `events.jsonl` logs.

    Targets:
      - y_overfit: whether val_acc drops by >= collapse_drop within next horizon epochs
      - y_collapse_epoch: estimated epoch of minimum val_acc in horizon window (relative index)
    """

    def __init__(self, events_jsonl: str | Path, spec: MetaSequenceSpec = MetaSequenceSpec()):
        self.path = Path(events_jsonl)
        self.spec = spec
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        # Aggregate per-epoch features from logged events.
        per_epoch: dict[int, dict[str, float]] = {}
        val_acc: dict[int, float] = {}

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                kind = r.get("kind")
                epoch = r.get("epoch")
                if epoch is None:
                    continue
                epoch = int(epoch)
                if kind == "val_epoch":
                    if "val_acc" in r:
                        val_acc[epoch] = float(r["val_acc"])
                    per_epoch.setdefault(epoch, {}).update(_flatten_metrics(r))
                elif kind in {"activation_stats", "gradient_stats", "repr_epoch"}:
                    # For activation_stats, reduce module dict to means.
                    if kind == "activation_stats":
                        mods = r.get("modules", {})
                        if isinstance(mods, dict) and mods:
                            # average a few scalar stats across modules
                            scalars = {}
                            for _, mv in mods.items():
                                if not isinstance(mv, dict):
                                    continue
                                for kk, vv in mv.items():
                                    if isinstance(vv, (int, float)):
                                        scalars.setdefault(kk, []).append(float(vv))
                            for kk, arr in scalars.items():
                                per_epoch.setdefault(epoch, {})[f"act_{kk}_mean"] = float(np.mean(arr))
                        continue
                    per_epoch.setdefault(epoch, {}).update({f"{kind}_{k}": v for k, v in _flatten_metrics(r).items()})

        epochs = sorted(per_epoch.keys())
        if not epochs:
            raise RuntimeError("No usable epochs found in log.")

        # Build feature matrix
        feat_keys = sorted({k for e in epochs for k in per_epoch[e].keys()})
        feats = []
        accs = []
        for e in epochs:
            row = [per_epoch[e].get(k, 0.0) for k in feat_keys]
            feats.append(row)
            accs.append(val_acc.get(e, float("nan")))
        self.feat_keys = feat_keys
        self.X = np.asarray(feats, dtype=np.float32)
        self.val_acc = np.asarray(accs, dtype=np.float32)
        self.epochs = np.asarray(epochs, dtype=np.int64)

        # Build windows
        self.indices = []
        for i in range(0, len(epochs) - spec.window - spec.horizon):
            self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        w = self.spec.window
        h = self.spec.horizon
        Xw = self.X[i : i + w]
        # Determine collapse target from val_acc trajectory
        future = self.val_acc[i + w : i + w + h]
        now = self.val_acc[i + w - 1]
        if np.isnan(now) or np.isnan(future).any():
            y_overfit = 0.0
            y_when = 0.0
        else:
            min_future = float(np.min(future))
            y_overfit = 1.0 if (now - min_future) >= self.spec.collapse_drop else 0.0
            y_when = float(int(np.argmin(future)))  # 0..h-1
        return (
            torch.from_numpy(Xw),
            torch.tensor([y_overfit], dtype=torch.float32),
            torch.tensor([y_when], dtype=torch.float32),
        )

