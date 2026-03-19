import unittest
import numpy as np
import os
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.metrics import center_kernel_alignment

class TestMetrics(unittest.TestCase):
    def test_cka_identity(self):
        """CKA(X, X) should be 1.0."""
        x = np.random.randn(32, 64)
        sim = center_kernel_alignment(x, x)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_cka_orthogonal(self):
        """CKA(X, Y) should be low for distinct random matrices."""
        x = np.random.randn(100, 50)
        y = np.random.randn(100, 50)
        sim = center_kernel_alignment(x, y)
        self.assertLess(sim, 0.5)

if __name__ == "__main__":
    unittest.main()
