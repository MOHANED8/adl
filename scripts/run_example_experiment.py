from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtd.analysis import (  # noqa: E402
    ActivationMonitor,
    ActivationMonitorConfig,
    CollapseMonitorConfig,
    LayerCollapseMonitor,
    GradientMonitor,
    GradientMonitorConfig,
    NeuronSpecializationMonitor,
    RepresentationMonitor,
    RepresentationMonitorConfig,
    SelfSupervisedConsistencyMonitor,
    SelfSupervisedMonitorConfig,
    SpecializationMonitorConfig,
    UncertaintyMonitor,
    UncertaintyMonitorConfig,
)
from dtd.datasets import DatasetSpec, build_dataloaders  # noqa: E402
from dtd.experiments.evaluate import EvalSpec, evaluate_run  # noqa: E402
from dtd.hooks import HookSpec  # noqa: E402
from dtd.models import ModelSpec, build_model  # noqa: E402
from dtd.monitoring import EarlyWarningSystem  # noqa: E402
from dtd.training import OptimConfig, RuntimeConfig, TrainConfig, Trainer  # noqa: E402
from dtd.training.logging import configure_logging, get_logger  # noqa: E402


def main() -> None:
    """Run a fast synthetic experiment that generates checkpoints and example logs."""

    configure_logging("INFO")
    logger = get_logger(__name__)
    out_dir = ROOT / "runs" / "example_synthetic_cnn"
    loaders = build_dataloaders(
        train=DatasetSpec(
            name="synthetic_shapes",
            root="data",
            split="train",
            batch_size=32,
            num_workers=0,
            image_size=32,
            aug="none",
            cache_dir="data/cache/synthetic_shapes",
            shuffle=True,
        ),
        val=DatasetSpec(
            name="synthetic_shapes",
            root="data",
            split="val",
            batch_size=64,
            num_workers=0,
            image_size=32,
            aug="none",
            cache_dir="data/cache/synthetic_shapes",
            shuffle=False,
        ),
    )
    model = build_model(
        ModelSpec(
            name="cnn",
            num_classes=3,
            hook_spec=HookSpec(
                module_globs=("features.*", "classifier"),
                capture_outputs=True,
                capture_output_grads=True,
                capture_attention=True,
                max_per_module=1,
            ),
        )
    )
    monitors = [
        ActivationMonitor(ActivationMonitorConfig(feature_mode="gap")),
        GradientMonitor(GradientMonitorConfig(hutchinson_samples=0)),
        RepresentationMonitor(RepresentationMonitorConfig(module_name="classifier", sample_batches=2, svcca_k=3)),
        UncertaintyMonitor(UncertaintyMonitorConfig(sample_batches=2, bins=8, temperature_steps=20)),
        NeuronSpecializationMonitor(SpecializationMonitorConfig(module_name="classifier", topk_fraction=0.25)),
        LayerCollapseMonitor(CollapseMonitorConfig(module_name="classifier", collapse_rank_threshold=0.35)),
        SelfSupervisedConsistencyMonitor(SelfSupervisedMonitorConfig(module_name="classifier", noise_std=0.02)),
        EarlyWarningSystem(mode="console"),
    ]
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        cfg=TrainConfig(
            out_dir=out_dir,
            epochs=5,
            optim=OptimConfig(name="adamw", lr=3e-4, weight_decay=1e-4),
            runtime=RuntimeConfig(device="cpu", amp=False, multi_gpu=False, distributed=False),
            log_every_steps=2,
            monitor_every_steps=2,
            checkpoint_enabled=True,
            checkpoint_every_epochs=1,
            save_best=True,
        ),
        monitors=monitors,
    )
    trainer.fit()

    summary = evaluate_run(out_dir, spec=EvalSpec(window=2, horizon=2, collapse_drop=0.02, granger_maxlag=1))
    summary_path = out_dir / "example_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Example experiment finished. Results written to %s", summary_path)


if __name__ == "__main__":
    main()
