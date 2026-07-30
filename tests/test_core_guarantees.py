import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import bundle, cli, hadamard, hf_utils, quant, runtime
from common.errors import QuantError
from common.quant import QuantTensor


class TestCoreGuarantees(unittest.TestCase):
    def test_chunked_hadamard_equals_full(self):
        """Column-chunked FWHT must match a single H @ W (axis=0)."""
        rng = np.random.default_rng(0)
        W = rng.normal(size=(48, 33)).astype(np.float32)
        runtime.max_work_elems.cache_clear()
        os.environ["ARIA_QUANT_MAX_ELEMS"] = str(64 * 8)  # force many chunks
        try:
            got, meta = hadamard.hadamard_rotate(W, axis=0, seed=7)
            self.assertTrue(meta["applied"])
            self.assertTrue(meta["chunked"])
        finally:
            os.environ.pop("ARIA_QUANT_MAX_ELEMS", None)
            runtime.max_work_elems.cache_clear()

        runtime.max_work_elems.cache_clear()
        os.environ["ARIA_QUANT_MAX_ELEMS"] = str(1 << 30)
        try:
            full, meta2 = hadamard.hadamard_rotate(W, axis=0, seed=7)
            self.assertTrue(meta2["applied"])
            self.assertFalse(meta2["chunked"])
        finally:
            os.environ.pop("ARIA_QUANT_MAX_ELEMS", None)
            runtime.max_work_elems.cache_clear()

        self.assertTrue(np.allclose(got, full, atol=1e-4, rtol=1e-4))

    def test_axis1_rejected(self):
        W = np.ones((8, 4), dtype=np.float32)
        with self.assertRaises(QuantError):
            hadamard.hadamard_rotate(W, axis=1)

    def test_quantize_always_hadamard_and_codebook(self):
        rng = np.random.default_rng(1)
        W = rng.normal(size=(64, 16)).astype(np.float32)
        t = quant.quantize_weight(W, bits=4, group_size=32, seed=0)
        self.assertTrue(t.hadamard_meta.get("applied"))
        self.assertEqual(t.codebook.shape[-1], 16)  # 2^4
        self.assertGreater(len(t.packed_indices), 0)

    def test_streaming_bundle_preserves_core(self):
        """Stream-style BundleWriter path still applies Hadamard + Lloyd-Max on 2D."""
        weights = hf_utils.make_tiny_state_dict(seed=2)
        names_2d = [n for n, w in weights.items() if w.ndim == 2]
        bit_map = {n: 4 for n in names_2d}
        with tempfile.TemporaryDirectory() as td:
            with bundle.BundleWriter(
                td,
                hf_utils.tiny_model_config(),
                "q4",
                group_size_default=32,
                hadamard_seed=0,
            ) as writer:
                for name, arr in weights.items():
                    obj = cli._process_one(name, arr, bit_map, 32, 0, workers=1)
                    writer.add(name, obj)
                writer.close()
            cfg, loaded = bundle.load_bundle(td)
            self.assertEqual(cfg["quantization"], "q4")
            for name in names_2d:
                t = loaded[name]
                self.assertIsInstance(t, QuantTensor)
                self.assertTrue(t.hadamard_meta.get("applied"), msg=name)
                self.assertEqual(t.codebook.shape[-1], 16, msg=name)
                recon = quant.dequantize(t)
                self.assertEqual(recon.shape, weights[name].shape)


if __name__ == "__main__":
    unittest.main()
