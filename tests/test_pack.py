import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common import pack
from common.errors import QuantError


class TestPack(unittest.TestCase):
    def test_roundtrip_all_bits(self):
        rng = np.random.default_rng(0)
        for bits in (1, 2, 3, 4):
            max_v = (1 << bits) - 1
            idx = rng.integers(0, max_v + 1, size=100, dtype=np.uint8)
            packed = pack.pack_indices(idx, bits)
            self.assertEqual(len(packed), pack.packed_size(idx.size, bits))
            out = pack.unpack_indices(packed, idx.size, bits)
            np.testing.assert_array_equal(out, idx)

    def test_bad_bits(self):
        with self.assertRaises(QuantError):
            pack.pack_indices(np.array([0]), bits=5)


if __name__ == "__main__":
    unittest.main()
