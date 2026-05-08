from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dashboard.app import load_events
from dtd.analysis import ActivationMonitor, ActivationMonitorConfig, GradientMonitor, GradientMonitorConfig
from dtd.models.instrumented import InstrumentedModel
from dtd.training.logging import make_logger


class ScalarSink:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), step))


class TrainerStub:
    def __init__(self, model, tmp_path: Path):
        self.model = model
        self.jsonl = make_logger(tmp_path, "events.jsonl")
        self.tb = ScalarSink()


@pytest.mark.visualization
def test_dashboard_load_events_ignores_missing_and_malformed_files(tmp_path, sample_events_file):
    assert load_events(tmp_path / "missing.jsonl").empty

    df = load_events(sample_events_file)
    assert list(df["kind"].dropna())[:2] == ["train_step", "gradient_stats"]
    assert "not-json" not in df.to_string()


@pytest.mark.visualization
@pytest.mark.integration
def test_activation_monitor_logs_module_statistics(tmp_path):
    model = InstrumentedModel(nn.Sequential(nn.Linear(4, 4), nn.ReLU()))
    trainer = TrainerStub(model, tmp_path)

    _ = model(torch.ones(3, 4))
    monitor = ActivationMonitor(ActivationMonitorConfig(feature_mode="flatten", entropy_bins=4))
    monitor.on_step(trainer=trainer, epoch=0, step=0, global_step=0)

    row = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["kind"] == "activation_stats"
    assert row["modules"]
    assert any(name.startswith("activations/") for name, _, _ in trainer.tb.scalars)


@pytest.mark.visualization
@pytest.mark.integration
def test_gradient_monitor_logs_gradient_diagnostics(tmp_path):
    model = nn.Linear(4, 2)
    trainer = TrainerStub(model, tmp_path)
    x = torch.randn(3, 4)
    y = torch.tensor([0, 1, 0])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    monitor = GradientMonitor(GradientMonitorConfig(hutchinson_samples=0, sharpness_eps=1e-4))
    monitor.on_step(trainer=trainer, epoch=0, step=0, global_step=0, x=x, y=y, loss=loss, logits=logits)

    row = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["kind"] == "gradient_stats"
    assert row["grad_norm"] > 0.0
    assert row["exploding"] is False
    assert any(name == "grads/norm" for name, _, _ in trainer.tb.scalars)
