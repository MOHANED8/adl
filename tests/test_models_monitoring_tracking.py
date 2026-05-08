from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dtd.hooks import HookSpec
from dtd.meta.models import OverfitPredictorLSTM, OverfitPredictorTransformer
from dtd.models import ModelSpec, build_model
from dtd.models.cnn import SmallCNN
from dtd.models.timm_attention import enable_timm_attention_capture
from dtd.monitoring import EarlyWarningSystem
from dtd.tracking.tracker import CompositeTracker, Tracker, build_tracker


@pytest.mark.unit
def test_small_cnn_and_model_factory_forward_shapes():
    x = torch.randn(2, 3, 16, 16)
    assert SmallCNN(num_classes=5)(x).shape == (2, 5)

    model = build_model(ModelSpec(name="small_cnn", num_classes=5, hook_spec=HookSpec(module_globs=("classifier",))))
    assert model(x).shape == (2, 5)
    assert "classifier" in model.hooks.activations


@pytest.mark.unit
def test_model_factory_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(ModelSpec(name="unknown", num_classes=2))


@pytest.mark.unit
def test_meta_prediction_models_return_expected_shapes():
    x = torch.randn(3, 4, 6)

    lstm = OverfitPredictorLSTM(in_features=6, hidden=8, layers=1)
    p, when = lstm(x)
    assert p.shape == (3, 1)
    assert when.shape == (3, 1)
    assert torch.all((p >= 0.0) & (p <= 1.0))

    transformer = OverfitPredictorTransformer(in_features=6, d_model=8, nhead=2, depth=1)
    p, when = transformer(x)
    assert p.shape == (3, 1)
    assert when.shape == (3, 1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.scale = 1.0
        self.qkv = nn.Linear(8, 24)
        self.attn_drop = nn.Identity()
        self.proj = nn.Linear(8, 8)
        self.proj_drop = nn.Identity()

    def forward(self, x):
        return x


@pytest.mark.hooks
def test_timm_attention_patch_exposes_last_attention_and_is_idempotent():
    model = nn.Sequential(Attention())
    patched = enable_timm_attention_capture(model)
    patched_again = enable_timm_attention_capture(model)

    out = patched_again(torch.randn(2, 3, 8))
    attn = patched_again[0]._last_attn
    assert out.shape == (2, 3, 8)
    assert attn.shape == (2, 2, 3, 3)
    assert patched is model
    assert patched_again[0]._dtd_patched is True


class CapturingLogger:
    def __init__(self):
        self.records = []

    def log(self, record):
        self.records.append(dict(record))


@pytest.mark.integration
def test_early_warning_logs_epoch_recommendation(capsys):
    trainer = SimpleNamespace(jsonl=CapturingLogger())
    monitor = EarlyWarningSystem(mode="console")

    monitor.on_epoch_end(trainer=trainer, epoch=2, global_step=8, val_metrics={"val_loss": 0.42})

    assert trainer.jsonl.records == [
        {
            "kind": "early_warning_epoch",
            "epoch": 2,
            "global_step": 8,
            "adaptive_risk": 0.0,
            "recommendation": "continue",
            "val_loss": 0.42,
        }
    ]
    assert "recommendation=continue" in capsys.readouterr().out


class RecorderTracker(Tracker):
    def __init__(self):
        self.logged = []
        self.artifacts = []
        self.finished = False

    def log(self, metrics, step):
        self.logged.append((step, metrics))

    def log_artifact(self, path, name=None):
        self.artifacts.append((path, name))

    def finish(self):
        self.finished = True


@pytest.mark.unit
def test_composite_tracker_fans_out_calls(tmp_path):
    a = RecorderTracker()
    b = RecorderTracker()
    tracker = CompositeTracker([a, b])

    tracker.log({"loss": 1.0}, step=3)
    tracker.log_artifact(tmp_path / "x.txt", name="x")
    tracker.finish()

    assert a.logged == b.logged == [(3, {"loss": 1.0})]
    assert a.artifacts == b.artifacts == [(tmp_path / "x.txt", "x")]
    assert a.finished and b.finished


@pytest.mark.unit
def test_build_tracker_returns_null_tracker_when_integrations_disabled(tmp_path):
    cfg = SimpleNamespace(
        wandb=SimpleNamespace(enabled=False),
        mlflow=SimpleNamespace(enabled=False),
    )
    tracker = build_tracker(cfg, out_dir=tmp_path, run_name="run")

    assert isinstance(tracker, Tracker)
    assert not isinstance(tracker, CompositeTracker)
    tracker.log({"metric": 1.0}, step=0)
    tracker.log_artifact(tmp_path / "missing.txt")
    tracker.finish()
