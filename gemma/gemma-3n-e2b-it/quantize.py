"""Quantize gemma-3n-e2b-it with Hadamard + codebook quantization.

Usage:
  python gemma/gemma-3n-e2b-it/quantize.py --tiny --bits 4
  python gemma/gemma-3n-e2b-it/quantize.py --bits 8
  python gemma/gemma-3n-e2b-it/quantize.py --model google/gemma-3n-E2B-it --bits 4
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
    cli.run_quantize(args, family, label="gemma-3n-e2b-it")


if __name__ == "__main__":
    main()
