import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import hadamard
from common.errors import QuantError


class TestHadamard(unittest.TestCase):
    def test_tile_sizes(self):
        self.assertEqual(hadamard.pow2_tile_sizes(10), [8, 2])
        self.assertEqual(hadamard.pow2_tile_sizes(3072), [2048, 1024])
        self.assertEqual(hadamard.pow2_tile_sizes(64), [64])
        self.assertEqual(hadamard.pow2_tile_sizes(1), [1])

    def test_orthogonal(self):
        H = hadamard.hadamard_matrix(8)
        I = H @ H.T
        self.assertTrue(np.allclose(I, np.eye(8), atol=1e-5))

    def test_preserves_norm(self):
        rng = np.random.default_rng(0)
        W = rng.normal(size=(16, 7))
        Wr, meta = hadamard.hadamard_rotate(W, seed=1)
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["mode"], "blocked")
        n_in = np.linalg.norm(W)
        n_out = np.linalg.norm(Wr)
        self.assertLess(abs(n_in - n_out) / (n_in + 1e-12), 1e-4)

    def test_non_pow2_no_pad(self):
        rng = np.random.default_rng(2)
        W = rng.normal(size=(10, 4)).astype(np.float32)
        Wr, meta = hadamard.hadamard_rotate(W, seed=0)
        self.assertEqual(meta["row_pad"], 0)
        self.assertEqual(Wr.shape, W.shape)
        self.assertEqual(meta["blocks"], [{"start": 0, "size": 8}, {"start": 8, "size": 2}])

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
        self.assertEqual(meta["mode"], "blocked")
        self.assertTrue(np.allclose(got, expected, atol=1e-4, rtol=1e-4))

    def test_blocked_roundtrip_non_pow2(self):
        rng = np.random.default_rng(3)
        W = rng.normal(size=(48, 17)).astype(np.float32)
        Wr, meta = hadamard.hadamard_rotate(W, seed=11)
        self.assertEqual(meta["mode"], "blocked")
        self.assertEqual(meta["row_pad"], 0)
        back, meta_inv = hadamard.hadamard_unrotate(Wr, seed=11)
        self.assertTrue(meta_inv.get("inverse"))
        self.assertTrue(np.allclose(back, W, atol=1e-4, rtol=1e-4))

    def test_signed_unrotate_not_second_forward(self):
        rng = np.random.default_rng(4)
        W = rng.normal(size=(32, 9)).astype(np.float32)
        Wr, _ = hadamard.hadamard_rotate(W, seed=5)
        wrong, _ = hadamard.hadamard_rotate(Wr, seed=5)
        right, _ = hadamard.hadamard_unrotate(Wr, seed=5)
        err_wrong = float(np.linalg.norm(wrong - W) / (np.linalg.norm(W) + 1e-12))
        err_right = float(np.linalg.norm(right - W) / (np.linalg.norm(W) + 1e-12))
        self.assertGreater(err_wrong, 0.5)
        self.assertLess(err_right, 1e-3)

    def test_quant_orig_rmse_blocked(self):
        from common import audit, quant

        rng = np.random.default_rng(6)
        W = rng.normal(size=(48, 32)).astype(np.float32)
        t = quant.quantize_weight(W, bits=4, group_size=16, seed=0)
        self.assertEqual(t.hadamard_meta.get("mode"), "blocked")
        row = audit.audit_one_tensor("blk.0.ffn_up.weight", t, W, seed=0)
        self.assertAlmostEqual(row["rel_rmse_orig"], row["rel_rmse_rot"], places=5)
        self.assertTrue(row["pass"])

    def test_large_dim_chunked_still_applies(self):
        """Tight RAM budget must chunk columns, not skip Hadamard."""
        import os
        from common import runtime as rt

        rt.max_work_elems.cache_clear()
        # largest block for 1000 = 512; allow only a few columns per chunk
        os.environ["ARIA_QUANT_MAX_ELEMS"] = str(512 * 4)
        try:
            W = np.ones((1000, 64), dtype=np.float32)
            Wr, meta = hadamard.hadamard_rotate(W)
            self.assertEqual(Wr.shape, W.shape)
            self.assertTrue(meta["applied"])
            self.assertEqual(meta["mode"], "blocked")
            self.assertTrue(meta["chunked"])
        finally:
            os.environ.pop("ARIA_QUANT_MAX_ELEMS", None)
            rt.max_work_elems.cache_clear()


if __name__ == "__main__":
    unittest.main()
