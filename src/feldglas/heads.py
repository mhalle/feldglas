"""Heads for encoders that do not bring one."""
from __future__ import annotations

import numpy as np


class MeanPoolHead:
    """The vector of a gate is the (weighted) mean of its tokens, at unit length.

    The honest baseline for any encoder with no learned pooling. Its vectors live in the
    encoder's TOKEN space; they are not comparable with a learned head's, which projects first
    (to see how far RADAR's attention is from uniform, use ``RadarHead.pool(..., weights=...)``,
    which keeps RADAR's own value and projection path: cosine 0.46-0.49, EXPLORATION section 2).
    ``bias`` is read as log-weights, so a soft gate's ``log(occupancy)`` weights each token by
    how much of it lies in the region.
    """

    queries: tuple[str, ...] = ()

    def prepare(self, tokens: np.ndarray):
        return np.asarray(tokens, np.float32)

    def pool(self, prepared, index, query=None, bias=None) -> np.ndarray:
        t = prepared[index]
        if bias is None:
            v = t.mean(0)
        else:
            w = np.exp(np.asarray(bias, np.float32) - np.max(bias))
            v = (t * (w / w.sum())[:, None]).sum(0)
        return v / np.linalg.norm(v)
