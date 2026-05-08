from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dtd.analysis.representation import RepresentationMonitor, RepresentationMonitorConfig, svcca_similarity
from dtd.models.instrumented import InstrumentedModel
from dtd.training.logging import make_logger


class ScalarSink:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), step))


class TrainerStub:
    def __init__(self, model, loader, tmp_path):
        self.model = model
        self.train_loader = loader
        self.val_loader = loader
        self.cfg = type("Cfg", (), {"device": "cpu"})()
        self.jsonl = make_logger(tmp_path, "events.jsonl")
        self.tb = ScalarSink()


@pytest.mark.unit
def test_svcca_similarity_is_high_for_identical_inputs():
    x = torch.randn(8, 4).numpy()
    assert svcca_similarity(x, x, k=2) > 0.9


@pytest.mark.integration
@pytest.mark.visualization
def test_representation_monitor_logs_epoch_shift(tmp_path, toy_dataset):
    model = InstrumentedModel(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), hook_spec=None)
    loader = DataLoader(toy_dataset, batch_size=4, shuffle=False)
    trainer = TrainerStub(model, loader, tmp_path)
    monitor = RepresentationMonitor(RepresentationMonitorConfig(module_name="0", sample_batches=2, svcca_k=2))

    monitor.on_epoch_end(trainer=trainer, epoch=0, global_step=1, val_metrics=None)
    monitor.on_epoch_end(trainer=trainer, epoch=1, global_step=2, val_metrics=None)

    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "repr_epoch"
    assert rows[1]["kind"] == "repr_epoch"
    assert "cka" in rows[1]
    assert any(name == "repr/cka" for name, _, _ in trainer.tb.scalars)
