"""Quantize nanbeige4.2-3b with Hadamard + codebook quantization.

Usage:
  python nanbeige/nanbeige4.2-3b/quantize.py --tiny --bits 4
  python nanbeige/nanbeige4.2-3b/quantize.py --bits 8
  python nanbeige/nanbeige4.2-3b/quantize.py --model Nanbeige/Nanbeige4.2-3B --bits 4
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
    cli.run_quantize(args, family, label="nanbeige4.2-3b")


if __name__ == "__main__":
    main()
