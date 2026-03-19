import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class DatasetName(Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IWILDCAM = "iwildcam"

class ModelType(Enum):
    MLP_SMALL = "mlp_small"
    MLP_LARGE = "mlp_large"
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"

class NoiseType(Enum):
    NONE = "none"
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"

@dataclass
class ExperimentConfig:
    dataset: DatasetName
    data_fraction: float  # [0.05, 0.1, 0.25, 0.5, 1.0]
    model_type: ModelType
    noise_type: NoiseType
    noise_level: float    # [0.0, 0.2, 0.4]
    weight_decay: float
    dropout_rate: float
    batch_size: int = 128
    learning_rate: float = 0.01
    epochs: int = 80
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Default configuration for smoke tests
SMOKE_CONFIG = ExperimentConfig(
    dataset=DatasetName.CIFAR10,
    data_fraction=0.05,
    model_type=ModelType.MLP_SMALL,
    noise_type=NoiseType.SYMMETRIC,
    noise_level=0.4,
    weight_decay=0.0,
    dropout_rate=0.0,
    batch_size=128,
    learning_rate=0.01,
    epochs=20
)
