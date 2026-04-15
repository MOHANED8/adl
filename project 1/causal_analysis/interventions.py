import torch.nn as nn


def apply_dropout_intervention(model: nn.Module, p: float = 0.3) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p


def freeze_layers(model: nn.Module, layer_prefixes: list[str]) -> None:
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in layer_prefixes):
            param.requires_grad = False
