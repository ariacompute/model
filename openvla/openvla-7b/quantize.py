"""Quantize openvla-7b with Hadamard + codebook quantization.

Usage:
  python openvla/openvla-7b/quantize.py --tiny --bits 4 --out ./out/openvla-7b_tiny_q4
  python openvla/openvla-7b/quantize.py --bits 4 --out ./out/openvla-7b_q4
  python openvla/openvla-7b/quantize.py --bits 8 --out ./out/openvla-7b_q8
  python openvla/openvla-7b/quantize.py --model openvla/openvla-7b --bits 4 --out ./out/openvla-7b_q4
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
    cli.run_quantize(args, family, label="openvla-7b")


if __name__ == "__main__":
    main()
