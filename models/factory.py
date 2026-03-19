import torch
from config import ModelType
from models.mlp import mlp_small, mlp_large
from models.resnet import get_resnet18, get_resnet34
from typing import Tuple
from torch.utils.data import Dataset

# Assuming DatasetName enum and relevant dataset imports will be added later
# For now, we'll use a placeholder for DatasetName and raise NotImplementedError

# from config import DatasetName # This import would be needed for DatasetName

def get_model(model_type: ModelType, num_classes: int = 10, dropout: float = 0.0):
    """
    Factory function to retrieve a model based on the specified type.

    Args:
        model_type (ModelType): The enum value of the model architecture.
        num_classes (int): The number of output classes for the model.
        dropout (float): The dropout rate to apply in the model, if applicable.

    Returns:
        torch.nn.Module: The instantiated model.

    Raises:
        ValueError: If an unknown model type is provided.
    """
    if model_type == ModelType.MLP_SMALL:
        return mlp_small(num_classes=num_classes, dropout=dropout)
    elif model_type == ModelType.MLP_LARGE:
        return mlp_large(num_classes=num_classes, dropout=dropout)
    elif model_type == ModelType.RESNET18:
        return get_resnet18(num_classes=num_classes)
    elif model_type == ModelType.RESNET34:
        return get_resnet34(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
