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


class LatticeMeanHead:
    """For a field whose lattices differ in width (``Field.all_tokens`` puts each in its own
    block of channels): each lattice's tokens in the gate are averaged within its block and set to
    unit length, and the blocks stand side by side, the whole at unit length.

    So every lattice the gate reaches counts ONCE, whatever its token count. A plain mean over the
    blocks would weight lattices by how many tokens they put in the gate - on the null model's
    3 mm grid a 32 mm box holds ~27 fine tokens to one deep one - and the vector would be the fine
    lattice's. RADAR's attention, for comparison, gives its deep lattice 38-45 % of the weight
    with 4-6 % of the tokens (EXPLORATION section 2). Equal weight is the choice with no fitted
    number in it. The block's mean needs no row offsets: rows of other lattices are zero in it,
    so its direction is the mean of its own tokens whatever it is divided by.
    """

    queries: tuple[str, ...] = ()

    def __init__(self, widths):
        self.widths = tuple(int(w) for w in widths)
        edges = np.cumsum((0,) + self.widths)
        self._blocks = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]

    @classmethod
    def for_field(cls, field) -> "LatticeMeanHead":
        return cls(field.widths)

    def prepare(self, tokens: np.ndarray):
        tokens = np.asarray(tokens, np.float32)
        if tokens.shape[1] != sum(self.widths):
            raise ValueError(f"{tokens.shape[1]} channels for lattices of widths {self.widths}")
        return tokens

    def pool(self, prepared, index, query=None, bias=None) -> np.ndarray:
        t = prepared[index]
        if bias is None:
            s = t.sum(0)
        else:
            w = np.exp(np.asarray(bias, np.float32) - np.max(bias))
            s = (t * w[:, None]).sum(0)
        v = np.zeros_like(s)
        for b in self._blocks:
            n = np.linalg.norm(s[b])
            if n > 0:                                  # a lattice the gate did not reach stays zero
                v[b] = s[b] / n
        return v / np.linalg.norm(v)
