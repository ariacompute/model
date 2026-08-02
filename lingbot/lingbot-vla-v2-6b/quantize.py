"""Quantize lingbot-vla-v2-6b with Hadamard + codebook quantization.

Usage:
  python lingbot/lingbot-vla-v2-6b/quantize.py --tiny --bits 4 --out ./out/lingbot-vla-v2-6b_tiny_q4
  python lingbot/lingbot-vla-v2-6b/quantize.py --bits 4 --out ./out/lingbot-vla-v2-6b_q4
  python lingbot/lingbot-vla-v2-6b/quantize.py --bits 8 --out ./out/lingbot-vla-v2-6b_q8
  python lingbot/lingbot-vla-v2-6b/quantize.py --model robbyant/lingbot-vla-v2-6b --bits 4 --out ./out/lingbot-vla-v2-6b_q4
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
    cli.run_quantize(args, family, label="lingbot-vla-v2-6b")


if __name__ == "__main__":
    main()
