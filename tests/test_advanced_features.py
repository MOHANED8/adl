from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dtd.analysis import (
    CollapseMonitorConfig,
    LayerCollapseMonitor,
    NeuronSpecializationMonitor,
    SelfSupervisedConsistencyMonitor,
    SelfSupervisedMonitorConfig,
    SpecializationMonitorConfig,
    UncertaintyMonitor,
    UncertaintyMonitorConfig,
)
from dtd.analysis.advanced import bayesian_calibration_summary, expected_calibration_error, fit_temperature
from dtd.datasets.synthetic import SyntheticShapesDataset
from dtd.models.instrumented import InstrumentedModel
from dtd.training import OptimConfig, TrainConfig, Trainer


class ScalarSink:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), step))


class TrainerStub:
    def __init__(self, model, loader, tmp_path):
        self.model = model
        self.model_ref = model
        self.train_loader = loader
        self.val_loader = loader
        self.device = torch.device("cpu")
        self.cfg = type("Cfg", (), {"device": "cpu"})()
        self.state = {}
        self.state_history = {}
        self.tb = ScalarSink()
        self.records = []
        self.tmp_path = tmp_path

    def _log_record(self, record):
        self.records.append(dict(record))
        path = self.tmp_path / "events.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _add_scalar(self, name, value, step):
        self.tb.add_scalar(name, value, step)

    def update_state(self, key, value):
        self.state[key] = float(value)
        self.state_history.setdefault(key, []).append(float(value))

    def get_state(self, key, default=None):
        return self.state.get(key, default)


@pytest.mark.unit
def test_calibration_helpers_return_valid_values():
    logits = torch.tensor([[4.0, 1.0], [3.0, 2.0], [1.0, 4.0], [0.5, 2.5]])
    labels = torch.tensor([0, 1, 1, 1])
    probs = logits.softmax(dim=1)

    ece = expected_calibration_error(probs, labels, bins=5)
    temp = fit_temperature(logits, labels, steps=10, lr=0.1)
    bayes = bayesian_calibration_summary(probs, labels, bins=5)

    assert ece >= 0.0
    assert temp > 0.0
    assert bayes["bayes_ece"] >= 0.0
    assert bayes["bayes_ci_width"] >= 0.0


@pytest.mark.dataset
def test_synthetic_shapes_dataset_is_deterministic_and_shaped(tmp_path):
    ds_a = SyntheticShapesDataset(tmp_path, split="train")
    ds_b = SyntheticShapesDataset(tmp_path, split="train")

    image_a, label_a = ds_a[0]
    image_b, label_b = ds_b[0]
    assert image_a.size == image_b.size == (32, 32)
    assert label_a == label_b == 0
    assert len(ds_a) == 128


@pytest.mark.integration
@pytest.mark.visualization
def test_advanced_monitors_log_records_and_state(tmp_path, toy_dataset):
    model = InstrumentedModel(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)),
        hook_spec=None,
    )
    loader = DataLoader(toy_dataset, batch_size=4, shuffle=False)
    trainer = TrainerStub(model, loader, tmp_path)
    x, y = next(iter(loader))

    model.hooks.clear()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    spec = NeuronSpecializationMonitor(SpecializationMonitorConfig(module_name="0"))
    collapse = LayerCollapseMonitor(CollapseMonitorConfig(module_name="0"))
    ssl = SelfSupervisedConsistencyMonitor(SelfSupervisedMonitorConfig(module_name="0", noise_std=0.01))

    spec.on_step(trainer=trainer, epoch=0, step=0, global_step=0)
    collapse.on_step(trainer=trainer, epoch=0, step=0, global_step=0)
    ssl.on_step(trainer=trainer, epoch=0, step=0, global_step=0, x=x)

    uncertainty = UncertaintyMonitor(UncertaintyMonitorConfig(sample_batches=1, bins=4, temperature_steps=5))
    uncertainty.on_epoch_end(trainer=trainer, epoch=0, global_step=1)

    kinds = [record["kind"] for record in trainer.records]
    assert "specialization_stats" in kinds
    assert "collapse_stats" in kinds
    assert "ssl_consistency" in kinds
    assert "uncertainty_stats" in kinds
    assert "collapse/score" in trainer.state
    assert "uncertainty/predictive_entropy" in trainer.state


@pytest.mark.training
def test_trainer_writes_checkpoints(tmp_path, toy_dataset):
    trainer = Trainer(
        model=InstrumentedModel(nn.Linear(4, 2)),
        train_loader=DataLoader(toy_dataset, batch_size=4, shuffle=False),
        val_loader=DataLoader(toy_dataset, batch_size=4, shuffle=False),
        cfg=TrainConfig(
            out_dir=tmp_path,
            epochs=1,
            optim=OptimConfig(name="sgd", lr=0.01, weight_decay=0.0),
            device="cpu",
            checkpoint_enabled=True,
            checkpoint_every_epochs=1,
            save_best=True,
        ),
    )

    trainer.fit()

    assert (tmp_path / "checkpoints" / "epoch_000.pt").exists()
    assert (tmp_path / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "checkpoints" / "best.pt").exists()
