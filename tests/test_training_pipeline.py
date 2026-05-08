from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dtd.models.instrumented import InstrumentedModel
from dtd.training import OptimConfig, TrainConfig, Trainer


class RecorderMonitor:
    def __init__(self):
        self.steps = []
        self.epochs = []

    def on_step(self, **kwargs):
        self.steps.append(kwargs["global_step"])

    def on_epoch_end(self, **kwargs):
        self.epochs.append((kwargs["epoch"], kwargs["val_metrics"]))


class RecorderTracker:
    def __init__(self):
        self.records = []
        self.finished = False

    def log(self, metrics, step):
        self.records.append((step, dict(metrics)))

    def finish(self):
        self.finished = True


@pytest.mark.training
@pytest.mark.integration
def test_trainer_runs_one_epoch_logs_metrics_and_calls_monitors(tmp_path, toy_dataset):
    train_loader = DataLoader(toy_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(toy_dataset, batch_size=4, shuffle=False)
    model = InstrumentedModel(nn.Linear(4, 2))
    monitor = RecorderMonitor()
    tracker = RecorderTracker()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=TrainConfig(
            out_dir=tmp_path,
            epochs=1,
            optim=OptimConfig(name="sgd", lr=0.01, weight_decay=0.0),
            device="cpu",
            log_every_steps=1,
            monitor_every_steps=1,
        ),
        monitors=[monitor],
        tracker=tracker,
    )

    trainer.fit()

    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows].count("train_step") == 2
    assert rows[-1]["kind"] == "val_epoch"
    assert 0.0 <= rows[-1]["val_acc"] <= 1.0
    assert monitor.steps == [0, 1]
    assert monitor.epochs[0][1]["val_loss"] >= 0.0
    assert tracker.finished is True
    assert (tmp_path / "tb").exists()


@pytest.mark.training
@pytest.mark.reproducibility
def test_identical_training_runs_produce_identical_metric_sequence(tmp_path, toy_dataset):
    def run_once(run_dir):
        torch.manual_seed(99)
        model = InstrumentedModel(nn.Linear(4, 2))
        trainer = Trainer(
            model=model,
            train_loader=DataLoader(toy_dataset, batch_size=4, shuffle=False),
            val_loader=DataLoader(toy_dataset, batch_size=4, shuffle=False),
            cfg=TrainConfig(
                out_dir=run_dir,
                epochs=1,
                optim=OptimConfig(name="sgd", lr=0.01, weight_decay=0.0),
                device="cpu",
                log_every_steps=1,
                monitor_every_steps=100,
            ),
        )
        trainer.fit()
        return [
            (row["kind"], round(row.get("loss", row.get("val_loss", 0.0)), 7), row.get("val_acc"))
            for row in (json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines())
        ]

    assert run_once(tmp_path / "a") == run_once(tmp_path / "b")
