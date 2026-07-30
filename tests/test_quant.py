import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import quant, hadamard
from common.errors import QuantError


def _rel_rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rmse = np.sqrt(np.mean((a - b) ** 2))
    rms = np.sqrt(np.mean(a ** 2)) + 1e-12
    return rmse / rms


class TestQuant(unittest.TestCase):
    def test_parse_bits(self):
        self.assertEqual(quant.parse_bits(4), 4.0)
        self.assertEqual(quant.parse_bits("2.54"), 2.54)
        with self.assertRaises(QuantError):
            quant.parse_bits(5)

    def test_dequant_error_bounds(self):
        rng = np.random.default_rng(0)
        W = rng.normal(size=(64, 16))
        # group-share is slightly looser than per-channel; keep Spec-ish bands.
        bounds = {4: 0.45, 3: 0.60, 2: 0.85, 1: 1.20}
        W_rot, _ = hadamard.hadamard_rotate(W, seed=0)
        for bits, lim in bounds.items():
            t = quant.quantize_weight(W, bits=bits, group_size=32, seed=0, codebook_share="group")
            recon = quant.dequantize(t)
            err = _rel_rmse(W_rot, recon)
            self.assertLessEqual(err, lim, msg=f"q{bits} group rel_rmse={err}")
        # per-channel still meets tighter band for q4
        t = quant.quantize_weight(W, bits=4, group_size=32, seed=0, codebook_share="channel")
        err = _rel_rmse(W_rot, quant.dequantize(t))
        self.assertLessEqual(err, 0.35, msg=f"q4 channel rel_rmse={err}")

    def test_group_share_much_smaller(self):
        rng = np.random.default_rng(0)
        W = rng.normal(size=(256, 128)).astype(np.float32)
        g = quant.quantize_weight(W, bits=4, group_size=32, codebook_share="group")
        c = quant.quantize_weight(W, bits=4, group_size=32, codebook_share="channel")
        self.assertLess(g.codebook.nbytes, c.codebook.nbytes // 8)
        self.assertEqual(g.codebook.ndim, 2)
        self.assertEqual(c.codebook.ndim, 3)

    def test_mixed_254(self):
        names = [f"blk.{i}.ffn_up.weight" for i in range(10)]
        names += [f"blk.{i}.attn_q.weight" for i in range(10)]
        assign = quant.allocate_mixed_bits(names, 2.54)
        avg = sum(assign.values()) / len(assign)
        self.assertGreaterEqual(avg, 2.45)
        self.assertLessEqual(avg, 2.65)

    def test_mixed_326(self):
        names = [f"layer.{i}.dense" for i in range(20)]
        names += ["token_embd.weight", "lm_head.weight", "blk.0.attn_q.weight"]
        assign = quant.allocate_mixed_bits(names, 3.26)
        avg = sum(assign.values()) / len(assign)
        self.assertGreaterEqual(avg, 3.15)
        self.assertLessEqual(avg, 3.40)

    def test_label(self):
        self.assertEqual(quant.quantization_label(4), "q4")
        self.assertEqual(quant.quantization_label(2.54), "q2.54")


if __name__ == "__main__":
    unittest.main()
