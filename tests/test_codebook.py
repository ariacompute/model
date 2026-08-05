import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import codebook
from common.errors import QuantError


class TestCodebook(unittest.TestCase):
    def test_lloyd_max_recovers_clusters(self):
        rng = np.random.default_rng(0)
        a = rng.normal(-2.0, 0.1, size=50)
        b = rng.normal(2.0, 0.1, size=50)
        x = np.concatenate([a, b])
        cb = codebook.lloyd_max(x, k=2, seed=0)
        self.assertEqual(cb.shape, (2,))
        self.assertLess(min(abs(cb[0] + 2), abs(cb[1] + 2)), 0.5)
        self.assertLess(min(abs(cb[0] - 2), abs(cb[1] - 2)), 0.5)

    def test_quantize_group(self):
        cb = np.array([-1.0, 1.0])
        idx = codebook.quantize_group(np.array([-0.9, 0.8, -0.1]), cb)
        np.testing.assert_array_equal(idx, np.array([0, 1, 0], dtype=np.uint8))

    def test_constant_vector(self):
        cb = codebook.lloyd_max(np.ones(16), k=4)
        self.assertTrue(np.allclose(cb, 1.0))

    def test_empty_raises(self):
        with self.assertRaises(QuantError):
            codebook.lloyd_max(np.array([]), k=2)

    def test_lloyd_max_batched_torch_cpu(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        rng = np.random.default_rng(0)
        # Two groups with well-separated clusters.
        batch = np.stack(
            [
                np.concatenate(
                    [rng.normal(-2.0, 0.05, 64), rng.normal(2.0, 0.05, 64)]
                ).astype(np.float32),
                np.concatenate(
                    [rng.normal(-1.0, 0.05, 64), rng.normal(1.0, 0.05, 64)]
                ).astype(np.float32),
            ],
            axis=0,
        )
        cbs, idx = codebook.lloyd_max_batched_torch(
            batch, k=2, max_iter=30, seed=0, device="cpu"
        )
        self.assertEqual(cbs.shape, (2, 2))
        self.assertEqual(idx.shape, batch.shape)
        # Reconstruct and check error is small vs data scale.
        recon = cbs[np.arange(2)[:, None], idx]
        err = float(np.sqrt(np.mean((batch - recon) ** 2)))
        self.assertLess(err, 0.2)


if __name__ == "__main__":
    unittest.main()
