import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import cli, bundle, quant


class TestCliTiny(unittest.TestCase):
    def test_tiny_export_and_load(self):
        family = Path(ROOT) / "gemma" / "gemma-4-e2b-it"
        with tempfile.TemporaryDirectory() as td:
            args = cli.build_parser().parse_args(
                ["--tiny", "--bits", "4", "--out", td, "--seed", "0"]
            )
            out = cli.run_quantize(args, str(family), label="test")
            cfg, tensors = bundle.load_bundle(out)
            self.assertEqual(cfg["quantization"], "q4")
            q_names = [n for n, t in tensors.items() if hasattr(t, "bits")]
            self.assertGreaterEqual(len(q_names), 1)
            for n in q_names:
                quant.dequantize(tensors[n])

    def test_tiny_mixed(self):
        family = Path(ROOT) / "qwen" / "qwen3.5-2b"
        with tempfile.TemporaryDirectory() as td:
            args = cli.build_parser().parse_args(
                ["--tiny", "--bits", "2.54", "--out", td]
            )
            out = cli.run_quantize(args, str(family), label="test")
            cfg, _ = bundle.load_bundle(out)
            self.assertEqual(cfg["quantization"], "q2.54")


if __name__ == "__main__":
    unittest.main()
