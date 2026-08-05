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

    def test_signed_roundtrip_pow2(self):
        """Forward T=H@S then unrotate S@H recovers W when rows are power-of-two."""
        rng = np.random.default_rng(3)
        W = rng.normal(size=(64, 17)).astype(np.float32)
        Wr, meta = hadamard.hadamard_rotate(W, seed=11)
        self.assertTrue(meta["applied"])
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

    def test_quant_orig_rmse_after_unrotate(self):
        from common import audit, quant

        rng = np.random.default_rng(6)
        W = rng.normal(size=(64, 32)).astype(np.float32)
        t = quant.quantize_weight(W, bits=4, group_size=32, seed=0)
        row = audit.audit_one_tensor("blk.0.attn_q.weight", t, W, seed=0)
        self.assertLess(row["rel_rmse_rot"], 0.35)
        self.assertLess(row["rel_rmse_orig"], 0.35)
        self.assertTrue(row["pass"])
        self.assertEqual(row["row_pad"], 0)
        self.assertEqual(row["pad_mode"], "none")

    def test_non_pow2_ref_fill_matches_rot_error(self):
        """Pad-aware audit inverse isolates quant error (no zeropad leak)."""
        from common import audit, quant

        rng = np.random.default_rng(7)
        W = rng.normal(size=(48, 16)).astype(np.float32)  # pad 16 → 64
        t = quant.quantize_weight(W, bits=4, group_size=16, seed=0)
        row = audit.audit_one_tensor("blk.0.ffn_up.weight", t, W, seed=0)
        self.assertGreater(row["row_pad"], 0)
        self.assertEqual(row["pad_mode"], "ref_fill")
        # Orthogonality is on the padded domain; cropped orig can be ≤ rot.
        self.assertLessEqual(row["rel_rmse_orig"], row["rel_rmse_rot"] + 1e-5)
        self.assertLess(abs(row["rel_rmse_orig"] - row["rel_rmse_rot"]), 0.05)
        self.assertTrue(row["pass"])
        self.assertGreater(row["rel_rmse_orig_zeropad"], row["rel_rmse_orig"] + 0.05)

    def test_perfect_dequant_ref_fill_recovers(self):
        rng = np.random.default_rng(8)
        W = rng.normal(size=(10, 4)).astype(np.float32)
        Wr, _ = hadamard.hadamard_rotate(W, seed=2)
        back, meta = hadamard.hadamard_unrotate_with_ref(Wr, W, seed=2)
        self.assertEqual(meta["pad_mode"], "ref_fill")
        self.assertTrue(np.allclose(back, W, atol=1e-4, rtol=1e-4))

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
