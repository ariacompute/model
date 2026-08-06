import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import audit, cli, gen_compare, hf_utils
from common.audit_cli import main as audit_main
from common.errors import ConfigError


class TestAudit(unittest.TestCase):
    def test_infer_kind(self):
        self.assertEqual(audit.infer_kind(family="qwen3.5-2b"), "text")
        self.assertEqual(audit.infer_kind(family="gemma-4-e2b-it"), "text")
        self.assertEqual(audit.infer_kind(family="lfm2-350m"), "text")
        self.assertEqual(audit.infer_kind(family="inkling-small"), "text")
        self.assertEqual(audit.infer_kind(family="openvla-7b"), "vla")
        self.assertEqual(audit.infer_kind(family="openpi-pi0-3b"), "vla")
        self.assertEqual(audit.infer_kind(family="lingbot-vla-v2-6b"), "vla")
        self.assertEqual(audit.infer_kind("./out/openpi-pi0.5-3b_q4"), "vla")

    def test_threshold_table(self):
        self.assertEqual(audit.threshold_orig_rmse(8), 0.15)
        self.assertEqual(audit.threshold_orig_rmse(4), 0.35)
        self.assertEqual(audit.threshold_orig_rmse(2, "blk.0.ffn_up.weight"), 0.80)
        self.assertEqual(audit.threshold_orig_rmse(3, "blk.0.ffn_up.weight"), 0.50)

    def test_resolve_family_base_model(self):
        self.assertEqual(audit.resolve_family_base_model("qwen3-0.6b"), "Qwen/Qwen3-0.6B")
        self.assertEqual(
            audit.resolve_family_base_model("qwen/qwen3-0.6b"), "Qwen/Qwen3-0.6B"
        )
        self.assertIsNone(audit.resolve_family_base_model("no-such-family-xyz"))

    def test_tiny_ref_rejects_hf_tensor_names(self):
        with self.assertRaises(ConfigError) as cm:
            audit.load_ref_weights(
                ["model.embed_tokens.weight"],
                ref_tiny=True,
                tiny_seed=0,
            )
        msg = str(cm.exception)
        self.assertIn("--ref tiny", msg)
        self.assertIn("--model", msg)
        self.assertNotIn("model.embed_tokens.weight", hf_utils.make_tiny_state_dict(0))

    def test_layer_audit_tiny_cli_exit_zero(self):
        family = Path(ROOT) / "gemma" / "gemma-4-e2b-it"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bundle"
            args = cli.build_parser().parse_args(
                ["--tiny", "--bits", "4", "--out", str(out), "--seed", "0"]
            )
            cli.run_quantize(args, str(family), label="test")
            report_path = Path(td) / "audit_layer.json"
            rc = audit_main(
                [
                    "layer",
                    "--bundle",
                    str(out),
                    "--ref",
                    "tiny",
                    "--sample",
                    "4",
                    "--family",
                    "gemma-4-e2b-it",
                    "--report",
                    str(report_path),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "layer")
            self.assertEqual(report["kind"], "text")
            self.assertFalse(report["ci_fail"])
            self.assertGreaterEqual(report["sample"], 1)
            self.assertTrue(all("rel_rmse_orig" in r for r in report["layers"]))

    def test_gen_compare_skips_without_forcing_fail(self):
        family = Path(ROOT) / "openvla" / "openvla-7b"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "bundle"
            args = cli.build_parser().parse_args(
                ["--tiny", "--bits", "4", "--out", str(out), "--seed", "0"]
            )
            cli.run_quantize(args, str(family), label="test")
            report = gen_compare.run_gen_compare(
                out,
                "nonexistent/aria-audit-missing-model",
                kind="vla",
                family="openvla-7b",
            )
            self.assertEqual(report["kind"], "vla")
            self.assertFalse(report.get("ci_fail", True))
            self.assertIn(report.get("status"), ("skipped", "ok"))

    def test_exact_prefix_match_helpers(self):
        m = gen_compare.exact_prefix_match([1, 2, 3, 4], [1, 2, 9, 4])
        self.assertEqual(m["exact_prefix_len"], 2)
        self.assertAlmostEqual(m["exact_prefix_frac"], 0.5)
        self.assertFalse(m["exact_match"])
        full = gen_compare.exact_prefix_match([7, 8], [7, 8])
        self.assertTrue(full["exact_match"])
        self.assertEqual(full["exact_prefix_frac"], 1.0)
        empty = gen_compare.exact_prefix_match([], [1])
        self.assertEqual(empty["exact_prefix_len"], 0)
        self.assertFalse(empty["exact_match"])

    def test_default_prompts_are_completion_style(self):
        self.assertGreaterEqual(len(gen_compare.DEFAULT_TEXT_PROMPTS), 2)
        for p in gen_compare.DEFAULT_TEXT_PROMPTS:
            self.assertNotIn("how are you", p.lower())

    def test_min_max_token_validation(self):
        with self.assertRaises(ConfigError):
            gen_compare.run_text_gen_compare(
                ".",
                "x",
                max_new_tokens=4,
                min_new_tokens=8,
            )

if __name__ == "__main__":
    unittest.main()
