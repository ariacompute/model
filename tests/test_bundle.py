import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import bundle, quant, hf_utils


class TestBundle(unittest.TestCase):
    def test_roundtrip(self):
        weights = hf_utils.make_tiny_state_dict(seed=1)
        subset = {
            "token_embd.weight": weights["token_embd.weight"],
            "blk.0.attn_norm.weight": weights["blk.0.attn_norm.weight"],
            "blk.0.attn_q.weight": weights["blk.0.attn_q.weight"],
        }
        q = {
            "token_embd.weight": quant.quantize_weight(
                subset["token_embd.weight"], bits=4, group_size=32, seed=0
            ),
            "blk.0.attn_norm.weight": subset["blk.0.attn_norm.weight"],
            "blk.0.attn_q.weight": quant.quantize_weight(
                subset["blk.0.attn_q.weight"], bits=2, group_size=32, seed=0
            ),
        }
        with tempfile.TemporaryDirectory() as td:
            bundle.write_bundle(
                td,
                q,
                hf_utils.tiny_model_config(),
                quantization="q4",
                group_size_default=32,
                hadamard_seed=0,
            )
            cfg, loaded = bundle.load_bundle(td)
            self.assertEqual(cfg["format"], "aria-quant-bundle")
            self.assertEqual(cfg["quantization"], "q4")
            self.assertIn("token_embd.weight", loaded)
            self.assertEqual(loaded["token_embd.weight"].bits, 4)
            recon = quant.dequantize(loaded["token_embd.weight"])
            self.assertEqual(recon.shape, subset["token_embd.weight"].shape)
            self.assertEqual(loaded["blk.0.attn_norm.weight"].ndim, 1)


if __name__ == "__main__":
    unittest.main()
