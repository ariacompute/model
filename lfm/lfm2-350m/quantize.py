"""Quantize lfm2-350m with Hadamard + codebook quantization.

Usage:
  python lfm/lfm2-350m/quantize.py --tiny --bits 4
  python lfm/lfm2-350m/quantize.py --bits 8
  python lfm/lfm2-350m/quantize.py --model LiquidAI/LFM2-350M --bits 4
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
    cli.run_quantize(args, family, label="lfm2-350m")


if __name__ == "__main__":
    main()
