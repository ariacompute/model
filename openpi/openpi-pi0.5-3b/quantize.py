"""Quantize openpi-pi0.5-3b with Hadamard + codebook quantization.

Usage:
  python openpi/openpi-pi0.5-3b/quantize.py --tiny --bits 4 --out ./out/openpi-pi0.5-3b_tiny_q4
  python openpi/openpi-pi0.5-3b/quantize.py --bits 4 --out ./out/openpi-pi0.5-3b_q4
  python openpi/openpi-pi0.5-3b/quantize.py --bits 8 --out ./out/openpi-pi0.5-3b_q8
  python openpi/openpi-pi0.5-3b/quantize.py --model lerobot/pi05_base --bits 4 --out ./out/openpi-pi0.5-3b_q4
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
    cli.run_quantize(args, family, label="openpi-pi0.5-3b")


if __name__ == "__main__":
    main()
