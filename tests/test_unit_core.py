from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from dtd.analysis.representation import cosine_dist, linear_cka
from dtd.analysis.stats import approx_tensor_entropy, covariance_and_spectrum, dead_neuron_ratio, moments, sparsity
from dtd.training.logging import make_logger
from dtd.training.optim import OptimConfig, build_optimizer


@pytest.mark.unit
def test_stats_functions_return_expected_values():
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    got = moments(x)
    assert got["mean"] == pytest.approx(0.5)
    assert got["var"] == pytest.approx(0.25)
    assert sparsity(x) == pytest.approx(0.5)
    assert dead_neuron_ratio(x, channel_dim=1) == pytest.approx(0.0)
    assert approx_tensor_entropy(torch.ones(4)) == pytest.approx(0.0)

    spectrum = covariance_and_spectrum(x)
    assert spectrum["cov_trace"] == pytest.approx(0.5)
    assert spectrum["sv_max"] >= 0.0


@pytest.mark.unit
def test_representation_metrics_are_deterministic_and_bounded():
    X = torch.eye(4).numpy()
    assert linear_cka(X, X) == pytest.approx(1.0)
    assert cosine_dist(X[0], X[0]) == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "cls"),
    [("adamw", torch.optim.AdamW), ("adam", torch.optim.Adam), ("sgd", torch.optim.SGD)],
)
def test_optimizer_factory_builds_supported_optimizers(name, cls):
    model = nn.Linear(2, 2)
    opt = build_optimizer(model, OptimConfig(name=name, lr=0.01))
    assert isinstance(opt, cls)


@pytest.mark.unit
def test_optimizer_factory_rejects_unknown_name():
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(model, OptimConfig(name="rmsprop"))  # type: ignore[arg-type]


@pytest.mark.unit
def test_jsonl_logger_writes_valid_records(tmp_path):
    logger = make_logger(tmp_path, "events.jsonl")
    logger.log({"kind": "unit", "value": 3})

    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "unit"
    assert rows[0]["value"] == 3
    assert "ts" in rows[0]
