"""Quantize qwen3.5-0.8b with Hadamard + codebook quantization.

Usage:
  python qwen/qwen3.5-0.8b/quantize.py --tiny --bits 4
  python qwen/qwen3.5-0.8b/quantize.py --bits 8
  python qwen/qwen3.5-0.8b/quantize.py --model Qwen/Qwen3.5-0.8B --bits 4
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from common import cli


def main():
    family = os.path.dirname(os.path.abspath(__file__))
    args = cli.build_parser().parse_args()
    cli.run_quantize(args, family, label="qwen3.5-0.8b")


if __name__ == "__main__":
    main()
