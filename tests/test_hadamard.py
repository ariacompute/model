import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import hadamard
from common.errors import QuantError


class TestHadamard(unittest.TestCase):
    def test_orthogonal(self):
        H = hadamard.hadamard_matrix(8)
        I = H @ H.T
        self.assertTrue(np.allclose(I, np.eye(8), atol=1e-5))

    def test_preserves_norm(self):
        rng = np.random.default_rng(0)
        W = rng.normal(size=(16, 7))
        Wr, meta = hadamard.hadamard_rotate(W, seed=1)
        self.assertTrue(meta["applied"])
        n_in = np.linalg.norm(W)
        n_out = np.linalg.norm(Wr)
        self.assertLess(abs(n_in - n_out) / (n_in + 1e-12), 1e-4)

    def test_non_pow2_pad(self):
        rng = np.random.default_rng(2)
        W = rng.normal(size=(10, 4))
        Wr, meta = hadamard.hadamard_rotate(W)
        self.assertEqual(meta["row_pad"], 6)
        self.assertEqual(Wr.shape, W.shape)

    def test_rejects_bad_n(self):
        with self.assertRaises(QuantError):
            hadamard.hadamard_matrix(3)

    def test_randomized_seed(self):
        H0 = hadamard.hadamard_matrix(8, seed=None)
        H1 = hadamard.hadamard_matrix(8, seed=42)
        self.assertFalse(np.allclose(H0, H1))
        self.assertTrue(np.allclose(H1 @ H1.T, np.eye(8), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
