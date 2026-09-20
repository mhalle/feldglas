"""Gates: which tokens a region owns, and how much of each.

The region is a query-time choice - the result everything else rests on (REPORT, Results J). A
gate here is a boolean mask on the field's MODEL grid turned into token indices, and the turning
is where a decision hides. RADAR was trained with the raster rule: a token counts, fully, if ONE
voxel of its box carries the label. Measured on 2026-09-20 (EXPLORATION section 10), that rule
puts 32 % of a liver vector's attention, 62 % of a spleen's and 57 % of a kidney's on tokens
that are less than half that organ, and only the liver and the lung own a single token entirely
inside them on the deep lattice. So a gate keeps each token's OCCUPANCY, and the rule is a
parameter:

  "any"        occupancy > 0: the raster rule, what the encoder's own training used
  "interior"   occupancy == 1: clean tokens only - a sample of the organ and nothing else
  a float t    occupancy >= t

and a soft gate is the same selection with ``log(occupancy)`` handed to the head as a bias.
Where masks come from (an encoder's own labels, haversack's class and distance fields, a click)
is not this module's business; a view over haversack's stores resolves to a mask upstream of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import Field


@dataclass(frozen=True)
class Gate:
    """``index`` into the concatenated lattices (``Field.offsets``), with each selected token's
    ``occupancy`` in (0, 1] and the ``lattice`` it belongs to."""

    index: np.ndarray
    occupancy: np.ndarray
    lattice: np.ndarray

    def __len__(self) -> int:
        return int(self.index.size)

    def soft_bias(self, floor: float = 1e-6) -> np.ndarray:
        """``log(occupancy)``: added to a head's pooling logits, a token half inside the region
        counts half as much as one fully inside, instead of exactly as much."""
        return np.log(np.maximum(self.occupancy, floor))

    def only(self, *lattices: int) -> "Gate":
        """The same gate on some lattices only - the lattice as a scale knob (a fine lattice sees
        ~20 mm around a token, a deep one ~80-120 mm; EXPLORATION section 2)."""
        keep = np.isin(self.lattice, lattices)
        return Gate(self.index[keep], self.occupancy[keep], self.lattice[keep])


def occupancy(mask: np.ndarray, kernel) -> np.ndarray:
    """Fraction of each token's box inside ``mask``, flat in lattice order. ``mask`` is
    ``(Z, Y, X)`` on the model grid, boolean or a probability in [0, 1]."""
    m = np.asarray(mask, np.float32)
    z, y, x = (s // k for s, k in zip(m.shape, kernel))
    if (z * kernel[0], y * kernel[1], x * kernel[2]) != m.shape:
        raise ValueError(f"kernel {tuple(kernel)} does not tile a mask of {m.shape}")
    return m.reshape(z, kernel[0], y, kernel[1], x, kernel[2]).mean((1, 3, 5)).ravel()


def occupancies(field: Field, mask: np.ndarray) -> list[np.ndarray]:
    """``mask``'s occupancy on every lattice - computed ONCE for an organ and handed to
    :func:`select` or :func:`box_gate` as ``within``, because a pass over the model grid costs
    tens of milliseconds and a sweep asks the same question of the same organ thousands of times."""
    if tuple(mask.shape) != tuple(field.grid.shape):
        raise ValueError(f"mask {mask.shape} is not on the model grid {field.grid.shape}")
    return [occupancy(mask, k) for k in field.kernels]


def _rule(rule):
    if rule == "any":
        return lambda o: o > 0
    if rule == "interior":
        return lambda o: o >= 1.0 - 1e-6
    if isinstance(rule, (int, float)) and 0 < float(rule) <= 1:
        return lambda o, t=float(rule): o >= t
    raise ValueError(f"rule must be 'any', 'interior' or a fraction in (0, 1]; got {rule!r}")


def select(field: Field, mask: np.ndarray, rule="any", within=None) -> Gate:
    """The tokens ``mask`` owns under ``rule``. ``within`` (a mask, or :func:`occupancies` of
    one) restricts them to tokens that also touch a second region - a box read on an organ's
    tokens only, as every box in the study was."""
    if tuple(mask.shape) != tuple(field.grid.shape):
        raise ValueError(f"mask {mask.shape} is not on the model grid {field.grid.shape}")
    keep = _rule(rule)
    if within is not None and not isinstance(within, list):
        within = occupancies(field, within)
    idx, occ, lat = [], [], []
    for j, (k, off) in enumerate(zip(field.kernels, field.offsets[:-1])):
        o = occupancy(mask, k)
        sel = keep(o)
        if within is not None:
            sel &= within[j] > 0
        hit = np.flatnonzero(sel)
        idx.append(hit + off); occ.append(o[hit]); lat.append(np.full(hit.size, j))
    return Gate(np.concatenate(idx).astype(np.int64), np.concatenate(occ).astype(np.float32),
                np.concatenate(lat).astype(np.int8))


def _box_slices(field: Field, centre, size_mm: float):
    half = [size_mm / s / 2.0 for s in field.grid.spacing]
    return tuple(slice(max(int(c - h), 0), min(int(c + h) + 1, n)) for c, h, n in zip(centre, half, field.grid.shape))


def box(field: Field, centre, size_mm: float) -> np.ndarray:
    """A cube of ``size_mm`` about ``centre`` (a model-grid index, Z Y X), as a mask on the model
    grid - clipped at the grid's edge. Sizes are millimetres because the model grid is
    anisotropic (RADAR's is 5 x 1 x 1 mm) and a cube in voxels is a slab in the patient."""
    m = np.zeros(field.grid.shape, bool)
    m[_box_slices(field, centre, size_mm)] = True
    return m


def box_gate(field: Field, centre, size_mm: float, rule="any", within=None) -> Gate:
    """``select(field, box(...))`` without touching the model grid: a box's occupancy of a token
    is the product of three one-dimensional overlaps, so only the tokens it reaches are visited.
    This is the interactive path - a box under the cursor, a sweep of thousands - and with
    ``within`` precomputed (:func:`occupancies`) it costs microseconds where ``select`` costs a
    pass over ~10 M voxels per lattice. Identical to ``select`` by test."""
    keep = _rule(rule)
    if within is not None and not isinstance(within, list):
        within = occupancies(field, within)
    sl = _box_slices(field, centre, size_mm)
    idx, occ, lat = [], [], []
    for j, (k, off) in enumerate(zip(field.kernels, field.offsets[:-1])):
        shape = field.lattice_shape(j)
        axes, fracs = [], []
        for e, kk in zip(sl, k):
            t = np.arange(e.start // kk, (max(e.stop, e.start + 1) - 1) // kk + 1)
            axes.append(t)
            fracs.append((np.minimum((t + 1) * kk, e.stop) - np.maximum(t * kk, e.start)).clip(0) / kk)
        o = np.einsum("a,b,c->abc", *fracs).ravel()
        flat = np.ravel_multi_index(np.meshgrid(*axes, indexing="ij"), shape).ravel()
        sel = keep(o)
        if within is not None:
            sel &= within[j][flat] > 0
        idx.append(flat[sel] + off); occ.append(o[sel]); lat.append(np.full(int(sel.sum()), j))
    order = [np.argsort(i) for i in idx]                               # select() returns tokens in index order
    return Gate(np.concatenate([i[o_] for i, o_ in zip(idx, order)]).astype(np.int64),
                np.concatenate([c[o_] for c, o_ in zip(occ, order)]).astype(np.float32),
                np.concatenate(lat).astype(np.int8))
