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

    def test_fwht_matches_matrix_small(self):
        rng = np.random.default_rng(0)
        W = rng.normal(size=(8, 5)).astype(np.float32)
        H = hadamard.hadamard_matrix(8)
        expected = (H @ W.astype(np.float64)).astype(np.float32)
        got, meta = hadamard.hadamard_rotate(W, seed=None)
        self.assertTrue(meta["applied"])
        self.assertTrue(np.allclose(got, expected, atol=1e-4, rtol=1e-4))

    def test_large_dim_chunked_still_applies(self):
        """Tight RAM budget must chunk columns, not skip Hadamard."""
        import os
        from common import runtime as rt

        rt.max_work_elems.cache_clear()
        # target=next_pow2(1000)=1024; allow only a few columns per chunk
        os.environ["ARIA_QUANT_MAX_ELEMS"] = str(1024 * 4)
        try:
            W = np.ones((1000, 64), dtype=np.float32)
            Wr, meta = hadamard.hadamard_rotate(W)
            self.assertEqual(Wr.shape, W.shape)
            self.assertTrue(meta["applied"])
            self.assertTrue(meta["chunked"])
        finally:
            os.environ.pop("ARIA_QUANT_MAX_ELEMS", None)
            rt.max_work_elems.cache_clear()


if __name__ == "__main__":
    unittest.main()
