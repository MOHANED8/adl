import unittest
import numpy as np
import torch
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.noise import inject_label_noise
from config import NoiseType

class MockDataset:
    def __init__(self, targets):
        self.targets = list(targets)

class TestData(unittest.TestCase):
    def test_symmetric_noise(self):
        targets = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        ds = MockDataset(targets)
        noise_level = 0.5
        noisy_ds = inject_label_noise(ds, NoiseType.SYMMETRIC, noise_level, num_classes=5)
        
        # Check that approximately 50% changed
        changed = np.sum(np.array(targets) != np.array(noisy_ds.targets))
        self.assertGreaterEqual(changed, 2)
        self.assertLessEqual(changed, 8)

    def test_asymmetric_noise(self):
        # truck(9) -> automobile(1)
        targets = [9, 9, 9] 
        ds = MockDataset(targets)
        noise_level = 1.0
        noisy_ds = inject_label_noise(ds, NoiseType.ASYMMETRIC, noise_level, num_classes=10)
        self.assertTrue(np.all(np.array(noisy_ds.targets) == 1))

if __name__ == "__main__":
    unittest.main()
