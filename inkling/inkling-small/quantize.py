"""Quantize inkling-small with Hadamard + codebook quantization.

Usage:
  python inkling/inkling-small/quantize.py --tiny --bits 4
  python inkling/inkling-small/quantize.py --bits 8
  python inkling/inkling-small/quantize.py --model thinkingmachines/Inkling-Small --bits 4
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
    cli.run_quantize(args, family, label="inkling-small")


if __name__ == "__main__":
    main()
