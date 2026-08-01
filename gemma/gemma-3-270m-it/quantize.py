"""Quantize gemma-3-270m-it with Hadamard + codebook quantization.

Usage:
  python gemma/gemma-3-270m-it/quantize.py --tiny --bits 4
  python gemma/gemma-3-270m-it/quantize.py --bits 8
  python gemma/gemma-3-270m-it/quantize.py --model google/gemma-3-270m-it --bits 4
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
    cli.run_quantize(args, family, label="gemma-3-270m-it")


if __name__ == "__main__":
    main()
