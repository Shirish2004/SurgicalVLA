"""Compact mask encoding for overlays: a mask downsampled to 32 x 32 and bit-packed (128 bytes).

The feature cache stores full-resolution masks as run lengths (``pseudolabels.cache``); this
encoding is only for drawing and quick inspection.
"""

from __future__ import annotations

import numpy as np


def mask_to_bits(mask: np.ndarray, size: int = 32) -> np.ndarray:
    """Downsample a boolean mask to ``size x size`` (nearest) and bit-pack it."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    ys = (np.arange(size) * h / size + h / (2 * size)).astype(int).clip(0, h - 1)
    xs = (np.arange(size) * w / size + w / (2 * size)).astype(int).clip(0, w - 1)
    return np.packbits(mask[np.ix_(ys, xs)].reshape(-1))


def bits_to_mask(bits: np.ndarray, size: int = 32) -> np.ndarray:
    return np.unpackbits(np.asarray(bits, dtype=np.uint8))[: size * size].reshape(size, size).astype(bool)


__all__ = ["bits_to_mask", "mask_to_bits"]
