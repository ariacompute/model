"""Ensure registered family scaffolds match requirements §1.1."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

EXPECTED = {
    "qwen/qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen/qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen/qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen/qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "gemma/gemma-3-270m-it": "google/gemma-3-270m-it",
    "gemma/gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma/gemma-3n-e2b-it": "google/gemma-3n-E2B-it",
    "gemma/gemma-3n-e4b-it": "google/gemma-3n-E4B-it",
    "gemma/gemma-4-e2b-it": "google/gemma-4-E2B-it",
    "gemma/gemma-4-e4b-it": "google/gemma-4-E4B-it",
    "lfm/lfm2-350m": "LiquidAI/LFM2-350M",
    "lfm/lfm2-700m": "LiquidAI/LFM2-700M",
    "lfm/lfm2-1.2b": "LiquidAI/LFM2-1.2B",
    "lfm/lfm2-2.6b": "LiquidAI/LFM2-2.6B",
    "lfm/lfm2-8b-a1b": "LiquidAI/LFM2-8B-A1B",
    "lfm/lfm2-vl-450m": "LiquidAI/LFM2-VL-450M",
    "lfm/lfm2.5-350m": "LiquidAI/LFM2.5-350M",
    "lfm/lfm2.5-1.2b-instruct": "LiquidAI/LFM2.5-1.2B-Instruct",
    "lfm/lfm2.5-1.2b-thinking": "LiquidAI/LFM2.5-1.2B-Thinking",
    "lfm/lfm2.5-2.6b": "LiquidAI/LFM2.5-2.6B",
    "lfm/lfm2.5-vl-1.6b": "LiquidAI/LFM2.5-VL-1.6B",
    "nanbeige/nanbeige4.2-3b": "Nanbeige/Nanbeige4.2-3B",
    "bonsai/bonsai-27b": "prism-ml/Bonsai-27B-unpacked",
    "inkling/inkling-small": "thinkingmachines/Inkling-Small",
    "openvla/openvla-7b": "openvla/openvla-7b",
    "openpi/openpi-pi0-3b": "lerobot/pi0_base",
    "openpi/openpi-pi0.5-3b": "lerobot/pi05_base",
    "lingbot/lingbot-vla-v2-6b": "robbyant/lingbot-vla-v2-6b",
}


class TestFamilies(unittest.TestCase):
    def test_all_families_present(self):
        for rel, repo in EXPECTED.items():
            d = ROOT / rel
            self.assertTrue((d / "quantize.py").is_file(), msg=rel)
            cfg = d / "config.yaml"
            self.assertTrue(cfg.is_file(), msg=rel)
            text = cfg.read_text(encoding="utf-8")
            self.assertIn(f"base_model: {repo}", text)


if __name__ == "__main__":
    unittest.main()
