from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtd.analysis import (
    ActivationMonitor,
    ActivationMonitorConfig,
    GradientMonitor,
    GradientMonitorConfig,
    RepresentationMonitor,
    RepresentationMonitorConfig,
)
from dtd.datasets import DatasetSpec, build_dataloaders
from dtd.hooks import HookSpec
from dtd.models import ModelSpec, build_model
from dtd.monitoring import EarlyWarningSystem
from dtd.training import OptimConfig, RuntimeConfig, TrainConfig, Trainer


def main():
    loaders = build_dataloaders(
        train=DatasetSpec(
            name="cifar10",
            root="data",
            split="train",
            batch_size=64,
            num_workers=0,
            image_size=224,
            aug="imagenet_basic",
            cache_dir="data/cache/cifar10_224_det",
        ),
        val=DatasetSpec(
            name="cifar10",
            root="data",
            split="test",
            batch_size=128,
            num_workers=0,
            image_size=224,
            aug="imagenet_basic",
            cache_dir="data/cache/cifar10_224_det",
            shuffle=False,
        ),
    )

    hook_spec = HookSpec(module_globs=("layer1.*", "layer2.*", "layer3.*", "layer4.*", "fc"), max_per_module=1)
    model = build_model(ModelSpec(name="resnet18", num_classes=10, pretrained=False, hook_spec=hook_spec))

    monitors = [
        ActivationMonitor(ActivationMonitorConfig(feature_mode="gap")),
        GradientMonitor(GradientMonitorConfig()),
        RepresentationMonitor(RepresentationMonitorConfig(module_name="fc")),
        EarlyWarningSystem(mode="console"),
    ]

    cfg = TrainConfig(
        out_dir="runs/cifar10_resnet18_monitoring",
        epochs=2,
        optim=OptimConfig(name="adamw", lr=3e-4, weight_decay=1e-4),
        log_every_steps=20,
        monitor_every_steps=20,
        runtime=RuntimeConfig(device="cuda" if torch.cuda.is_available() else "cpu"),
    )
    trainer = Trainer(model=model, train_loader=loaders["train"], val_loader=loaders["val"], cfg=cfg, monitors=monitors)
    trainer.fit()

    print(f"Done. Logs written to: {cfg.out_dir}")


if __name__ == "__main__":
    main()

