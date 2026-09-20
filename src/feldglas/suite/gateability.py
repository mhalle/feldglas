"""Gateability: how deep a structure is, against how many tokens lie ENTIRELY inside it.

An organ shallower than a token's half-width owns no clean token on that lattice, whatever the
head does afterwards - a fact about the instrument that a distance transform states before
anything is pooled. For RADAR (2026-09-20, six scans, its own mask): only the liver (5 of 101
admitted) and the lung (5 of 143) own any token on the deep lattice, which draws 38-45 % of the
attention; the kidney, the pancreas and the aorta own none on the mid lattice either; the
adrenal, the ribs and the iliac vessels own none anywhere. The table this returns is the
specification for an encoder designed here: a lattice fine enough, with meaning, for the small
structures to own tokens (EXPLORATION sections 10 and 11).
"""
from __future__ import annotations

import numpy as np

from ..contract import Field
from ..gate import occupancy


def gateability(field: Field, mask: np.ndarray | None = None, labels=None, min_voxels: int = 300) -> dict:
    """``{label: {"voxels", "inscribed_depth_mm", "inside": [...], "admitted": [...]}}`` per
    structure of ``mask`` (default: the encoder's own), lattices in the field's order."""
    try:
        from scipy import ndimage
    except ImportError as e:
        raise ImportError("gateability needs a distance transform: install feldglas[suite]") from e
    mask = field.native_mask if mask is None else mask
    labels = field.native_labels if labels is None else labels
    if mask is None:
        raise ValueError("this field has no native mask: pass one on the model grid")
    out = {}
    for v in np.unique(mask):
        if v == 0:
            continue
        m = mask == v
        if m.sum() < min_voxels:
            continue
        box = tuple(slice(max(a.min() - 2, 0), a.max() + 3) for a in np.nonzero(m))
        depth = float(ndimage.distance_transform_edt(m[box], sampling=field.grid.spacing).max())
        occ = [occupancy(m, k) for k in field.kernels]
        name = labels[int(v) - 1] if 0 < int(v) <= len(labels) else str(int(v))
        out[name] = {"voxels": int(m.sum()), "inscribed_depth_mm": round(depth, 1),
                     "inside": [int((o >= 1.0 - 1e-6).sum()) for o in occ],
                     "admitted": [int((o > 0).sum()) for o in occ]}
    return out
