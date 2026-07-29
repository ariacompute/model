import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import quant, bundle
from common.errors import FormatError, QuantError
import numpy as np


class TestErrors(unittest.TestCase):
    def test_invalid_bits(self):
        with self.assertRaises(QuantError):
            quant.parse_bits(8)

    def test_bad_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FormatError):
                bundle.load_bundle(td)

    def test_nonfinite_weight(self):
        W = np.ones((32, 4))
        W[0, 0] = np.nan
        with self.assertRaises(QuantError):
            quant.quantize_weight(W, bits=4, group_size=32)


if __name__ == "__main__":
    unittest.main()
