"""Bit-pack / unpack codebook indices."""

from __future__ import annotations

import numpy as np

from .errors import QuantError, ShapeMismatchError

INTEGER_BITS = (1, 2, 3, 4, 8)


def packed_size(count: int, bits: int) -> int:
    if count < 0:
        raise QuantError(f"count must be >=0, got {count}")
    if bits not in INTEGER_BITS:
        raise QuantError(f"bits must be one of {INTEGER_BITS}, got {bits}")
    return (count * bits + 7) // 8


def pack_indices(indices: np.ndarray, bits: int) -> bytes:
    """Pack indices LSB-first within each byte (8-bit = raw uint8 bytes)."""
    if bits not in INTEGER_BITS:
        raise QuantError(f"bits must be one of {INTEGER_BITS}, got {bits}")
    idx = np.asarray(indices, dtype=np.uint8).ravel()
    max_val = (1 << bits) - 1
    if idx.size and int(idx.max()) > max_val:
        raise QuantError(f"index {int(idx.max())} exceeds max for {bits}-bit")
    if bits == 8:
        return idx.tobytes(order="C")

    out = bytearray(packed_size(idx.size, bits))
    bit_pos = 0
    for v in idx:
        v = int(v) & max_val
        for b in range(bits):
            if v & (1 << b):
                byte_i = bit_pos // 8
                bit_i = bit_pos % 8
                out[byte_i] |= 1 << bit_i
            bit_pos += 1
    return bytes(out)


def unpack_indices(data: bytes, count: int, bits: int) -> np.ndarray:
    if bits not in INTEGER_BITS:
        raise QuantError(f"bits must be one of {INTEGER_BITS}, got {bits}")
    need = packed_size(count, bits)
    if len(data) < need:
        raise ShapeMismatchError(
            f"packed data length {len(data)} < required {need} for count={count} bits={bits}"
        )
    if bits == 8:
        return np.frombuffer(data[:need], dtype=np.uint8).copy()

    max_val = (1 << bits) - 1
    out = np.zeros(count, dtype=np.uint8)
    bit_pos = 0
    for i in range(count):
        v = 0
        for b in range(bits):
            byte_i = bit_pos // 8
            bit_i = bit_pos % 8
            if data[byte_i] & (1 << bit_i):
                v |= 1 << b
            bit_pos += 1
        out[i] = v & max_val
    return out
