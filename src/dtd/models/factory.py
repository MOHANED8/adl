from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from .instrumented import InstrumentedModel
from dtd.hooks import HookSpec


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_classes: int
    pretrained: bool = False
    image_size: int = 224
    hook_spec: Optional[HookSpec] = None


def build_model(spec: ModelSpec) -> InstrumentedModel:
    """
    Supported models (spec.name, case-insensitive):
      CNN:
        - resnet18, resnet34, resnet50
        - densenet121
        - efficientnet_b0
      Transformers:
        - vit (alias: vit_base_patch16_224 via timm)
        - deit_small (timm)
      Optional advanced:
        - convnext_tiny (torchvision)
        - swin_t (torchvision)
    """
    name = spec.name.lower().strip()

    if name in {"cnn", "small_cnn", "cnn_small"}:
        from .cnn import SmallCNN

        model = SmallCNN(num_classes=spec.num_classes)
        return InstrumentedModel(model, hook_spec=spec.hook_spec)

    if name in {
        "resnet18",
        "resnet34",
        "resnet50",
        "densenet121",
        "efficientnet_b0",
        "convnext",
        "convnext_tiny",
        "swin",
        "swin_t",
    }:
        model = _build_torchvision(name, num_classes=spec.num_classes, pretrained=spec.pretrained)
        return InstrumentedModel(model, hook_spec=spec.hook_spec)

    if name in {"vit", "vit_base", "vit_base_patch16_224", "deit_small", "deit_small_patch16_224"}:
        model = _build_timm(name, num_classes=spec.num_classes, pretrained=spec.pretrained)
        return InstrumentedModel(model, hook_spec=spec.hook_spec)

    raise ValueError(
        f"Unknown model '{spec.name}'. "
        f"Supported: resnet18/34/50, densenet121, efficientnet_b0, vit, deit_small, convnext_tiny, swin_t."
    )


def _build_torchvision(name: str, *, num_classes: int, pretrained: bool) -> nn.Module:
    if name == "convnext":
        name = "convnext_tiny"
    if name == "swin":
        name = "swin_t"

    from torchvision import models

    def _w(obj_name: str):
        if not pretrained:
            return None
        w = getattr(models, obj_name, None)
        return None if w is None else w.DEFAULT

    if name == "resnet18":
        m = models.resnet18(weights=_w("ResNet18_Weights"))
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "resnet34":
        m = models.resnet34(weights=_w("ResNet34_Weights"))
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "resnet50":
        m = models.resnet50(weights=_w("ResNet50_Weights"))
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "densenet121":
        m = models.densenet121(weights=_w("DenseNet121_Weights"))
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=_w("EfficientNet_B0_Weights"))
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=_w("ConvNeXt_Tiny_Weights"))
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if name == "swin_t":
        m = models.swin_t(weights=_w("Swin_T_Weights"))
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m

    raise ValueError(name)


def _build_timm(name: str, *, num_classes: int, pretrained: bool) -> nn.Module:
    try:
        import timm
    except Exception as e:
        raise RuntimeError("Missing dependency 'timm'. Install with `pip install timm`.") from e

    if name in {"vit", "vit_base"}:
        name = "vit_base_patch16_224"
    if name == "deit_small":
        name = "deit_small_patch16_224"

    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    from .timm_attention import enable_timm_attention_capture

    enable_timm_attention_capture(model)
    return model

