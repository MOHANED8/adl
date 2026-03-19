import unittest
import torch
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.factory import get_model
from config import ExperimentConfig, ModelType, DatasetName, NoiseType

class TestModels(unittest.TestCase):
    def test_factory_mlp_small(self):
        config = ExperimentConfig(
            dataset=DatasetName.CIFAR10,
            data_fraction=1.0,
            model_type=ModelType.MLP_SMALL,
            noise_type=NoiseType.NONE,
            noise_level=0.0,
            weight_decay=0.0,
            dropout_rate=0.0
        )
        model = get_model(config.model_type, num_classes=10, dropout=config.dropout_rate)
        self.assertIsInstance(model, torch.nn.Module)
        # Check forward pass
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

    def test_factory_resnet18(self):
        config = ExperimentConfig(
            dataset=DatasetName.CIFAR10,
            data_fraction=1.0,
            model_type=ModelType.RESNET18,
            noise_type=NoiseType.NONE,
            noise_level=0.0,
            weight_decay=0.0,
            dropout_rate=0.0
        )
        model = get_model(config.model_type, num_classes=10, dropout=config.dropout_rate)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (2, 10))

if __name__ == "__main__":
    unittest.main()
