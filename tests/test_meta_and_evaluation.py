from __future__ import annotations

import json

import numpy as np
import pytest

from dtd.experiments.evaluate import _auc_roc, evaluate_run
from dtd.meta.dataset import MetaSequenceDataset, MetaSequenceSpec


@pytest.mark.integration
def test_meta_sequence_dataset_builds_windows_and_targets(tmp_path):
    path = tmp_path / "events.jsonl"
    val_acc = [0.95, 0.94, 0.93, 0.80, 0.79]
    with path.open("w", encoding="utf-8") as f:
        for epoch, acc in enumerate(val_acc):
            f.write(json.dumps({"kind": "gradient_stats", "epoch": epoch, "grad_norm": epoch + 1.0}) + "\n")
            f.write(json.dumps({"kind": "val_epoch", "epoch": epoch, "val_acc": acc, "val_loss": 1.0 - acc}) + "\n")

    ds = MetaSequenceDataset(path, MetaSequenceSpec(window=2, horizon=2, collapse_drop=0.1))

    x, y_overfit, y_when = ds[0]
    assert len(ds) == 1
    assert x.shape == (2, len(ds.feat_keys))
    assert y_overfit.item() == pytest.approx(1.0)
    assert y_when.item() == pytest.approx(1.0)


@pytest.mark.unit
def test_auc_roc_orders_perfect_classifier_above_bad_classifier():
    y_true = [0, 0, 1, 1]
    good = [0.1, 0.2, 0.8, 0.9]
    bad = [0.9, 0.8, 0.2, 0.1]

    assert _auc_roc(np.array(y_true), np.array(good)) == pytest.approx(1.0)
    assert _auc_roc(np.array(y_true), np.array(bad)) == pytest.approx(0.0)


@pytest.mark.integration
def test_evaluate_run_reports_not_enough_epochs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "val_epoch", "epoch": 0, "val_acc": 0.9}) + "\n")

    result = evaluate_run(run_dir, spec=MetaSequenceSpec(window=2, horizon=2))  # type: ignore[arg-type]

    assert result["run_dir"] == str(run_dir)
    assert result["error"] == "not_enough_epochs_for_window_horizon"
    assert result["needed_epochs"] == 5
