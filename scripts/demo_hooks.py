from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow running without installing the package (repo root assumed).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dtd.datasets import DatasetSpec, build_dataloaders
from dtd.hooks import HookSpec
from dtd.models import ModelSpec, build_model


def main():
    loaders = build_dataloaders(
        train=DatasetSpec(
            name="cifar10",
            root="data",
            split="train",
            batch_size=16,
            num_workers=0,
            image_size=224,
            aug="imagenet_basic",
            cache_dir="data/cache/cifar10_224_deterministic",
            shuffle=False,
        )
    )
    x, y = next(iter(loaders["train"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hook_spec = HookSpec(
        module_globs=(
            "layer1.*",
            "layer2.*",
            "layer3.*",
            "layer4.*",
            "fc",
        ),
        capture_outputs=True,
        capture_output_grads=True,
        capture_attention=True,
        detach=True,
        cpu=True,
        max_per_module=1,
    )

    model = build_model(ModelSpec(name="resnet18", num_classes=10, pretrained=False, hook_spec=hook_spec)).to(device)
    model.train()

    x = x.to(device)
    y = y.to(device)

    model.hooks.clear()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    print("Captured activations:", len(model.hooks.activations), "modules")
    print("Captured gradients:", len(model.hooks.gradients), "modules")
    print("Captured attention:", len(model.hooks.attention), "modules")
    print("Example activation keys:", list(model.hooks.activations.keys())[:5])


if __name__ == "__main__":
    main()

