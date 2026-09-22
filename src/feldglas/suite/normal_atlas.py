"""The normal atlas: tell it what is unremarkable, and it shows what is not.

Six healthy donors' 32 mm liver boxes, from another collection and with no label used, found
liver tumors better than RADAR's own finding score did (0.909 pooled against 0.816; 2026-09-20,
EXPLORATION section 5). This module is that experiment as a probe any encoder can be put
through: sweep an organ with boxes, pool each, fit the healthy people's cloud, and score
everybody else by distance from it - reported PER LESION, because a per-region AUC is flattered
by the big lesions that fill many boxes.

The pieces, each its own function so the tool that runs a cohort and a viewer that runs one
scan share them:

  sweep / pool_sweep   boxes of one size over one organ, at half-box steps on a regular lattice
                       (the lattice index is kept: it is what makes "local maximum" mean something)
  scatter_mask         an expert mask on ITS grid -> occupancy on the model grid, by scattering
                       sub-voxel samples; nearest-neighbour resampling loses a 6 mm lesion between
                       5 mm slices, and the occupancy is also what a volume in ml is summed from
  PositionedNormal     a normal model whose MEAN depends on where in the organ a box sits (and how
                       full of organ it is): the dome, the hilum and a box half outside the organ
                       are each unremarkable in their own way
  auc, local_maxima, froc   per-lesion sensitivity against false positives per scan

Rules inherited from the study (EXPLORATION 3.3): patient-disjoint fits, both positive rules,
per-patient and pooled figures, lesions counted rather than regions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract import Field
from ..gate import box_gate, occupancies
from ..observe import shrunk_covariance

ML_PER_MM3 = 1e-3


# -- boxes ------------------------------------------------------------------------------------
def _edges(centres: np.ndarray, size_mm: float, spacing, shape):
    """The same slices ``gate._box_slices`` cuts, for many centres at once."""
    half = np.asarray([size_mm / s / 2.0 for s in spacing])
    lo = np.maximum((centres - half).astype(np.int64), 0)
    hi = np.minimum((centres + half).astype(np.int64) + 1, np.asarray(shape))
    return lo, hi


def integral(volume: np.ndarray) -> np.ndarray:
    """Summed-volume table, padded so a box sum is eight lookups."""
    return np.pad(np.asarray(volume, np.float64).cumsum(0).cumsum(1).cumsum(2), ((1, 0),) * 3)


def box_sums(table: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    z0, y0, x0 = lo.T; z1, y1, x1 = hi.T
    t = table
    return (t[z1, y1, x1] - t[z0, y1, x1] - t[z1, y0, x1] - t[z1, y1, x0]
            + t[z0, y0, x1] + t[z0, y1, x0] + t[z1, y0, x0] - t[z0, y0, x0])


@dataclass
class Sweep:
    """Boxes of one size over one organ. ``ijk`` is each box's place on the sweep's own regular
    lattice; ``coords`` its centre in the organ's bounding box, 0..1 along (Z, Y, X)."""

    size_mm: float
    centre: np.ndarray          # (N, 3) model-grid index
    ijk: np.ndarray             # (N, 3)
    lo: np.ndarray
    hi: np.ndarray
    fill: np.ndarray            # fraction of the box's voxels that are organ
    organ_ml: np.ndarray
    coords: np.ndarray


def sweep(field: Field, organ: np.ndarray, size_mm: float, min_fill: float = 0.25, step: float = 0.5) -> Sweep:
    """Every box of ``size_mm`` on a half-box lattice over ``organ``'s bounding box that is at
    least ``min_fill`` organ - the study's rule, which keeps a box hanging off the capsule and
    drops one that only grazes it."""
    sp = field.grid.spacing
    nz = np.nonzero(organ)
    if not len(nz[0]):
        z = np.zeros((0, 3))
        return Sweep(size_mm, z.astype(int), z.astype(int), z.astype(int), z.astype(int), z[:, 0], z[:, 0], z)
    b0 = np.array([a.min() for a in nz]); b1 = np.array([a.max() for a in nz])
    stride = np.maximum(np.round(size_mm * step / np.asarray(sp)), 1).astype(int)
    axes = [np.arange(a, b + 1, s) for a, b, s in zip(b0, b1, stride)]
    ijk = np.stack(np.meshgrid(*[np.arange(len(a)) for a in axes], indexing="ij"), -1).reshape(-1, 3)
    centre = np.stack([axes[d][ijk[:, d]] for d in range(3)], 1)
    lo, hi = _edges(centre, size_mm, sp, organ.shape)
    vox = np.prod(hi - lo, 1)
    n = box_sums(integral(organ), lo, hi)
    keep = n >= min_fill * vox
    span = np.maximum(b1 - b0, 1)
    return Sweep(size_mm, centre[keep], ijk[keep], lo[keep], hi[keep], (n / vox)[keep],
                 n[keep] * float(np.prod(sp)) * ML_PER_MM3, ((centre - b0) / span)[keep])


def pool_sweep(field: Field, head, prepared, sw: Sweep, organ: np.ndarray, query, rule="any") -> np.ndarray:
    """One vector per box, pooled on the ORGAN's tokens inside it. A box with no token under
    ``rule`` gets a row of NaN, so rows stay aligned with the sweep."""
    within = occupancies(field, organ)
    out = np.full((len(sw.centre), field.channels), np.nan, np.float32)
    for i, c in enumerate(sw.centre):
        g = box_gate(field, c, sw.size_mm, rule=rule, within=within)
        if len(g):
            out[i] = head.pool(prepared, g.index, query)
    return out


def normal_atlas(field: Field, head, organ: np.ndarray, query, normal, size_mm: float = 32.0, rule="any",
                 prepared=None) -> tuple[Sweep, np.ndarray]:
    """One scan against a normal model: the sweep and each box's distance from the unremarkable
    - the numbers a deviation map is drawn from. ``normal`` is an ``observe.NormalModel`` or a
    :class:`PositionedNormal`; the cohort tool (``tools/radar_atlas_modal.py``) fits them. (The
    six-donor pilot of this was medseg's radar_donor_atlas.py: 0.909 pooled against 0.816.)"""
    prepared = head.prepare(field.all_tokens()) if prepared is None else prepared
    sw = sweep(field, organ, size_mm)
    X = pool_sweep(field, head, prepared, sw, organ, query, rule=rule)
    ok = np.isfinite(X[:, 0])
    d = np.full(len(X), np.nan)
    if ok.any():
        d[ok] = (normal.distance(X[ok], sw.coords[ok], sw.fill[ok]) if isinstance(normal, PositionedNormal)
                 else normal.distance(X[ok]))
    return sw, d


# -- an expert mask, on the model grid --------------------------------------------------------
def scatter_mask(field: Field, labels: np.ndarray, affine_lps: np.ndarray, sub: int = 2):
    """``labels`` (an integer volume, 0 = nothing, on a grid whose index -> LPS mm map is the 4x4
    ``affine_lps``) as ``(occupancy, owner)`` on the model grid: the fraction of each model voxel
    that is labelled, and which label gave it most. Each source voxel is split into ``sub``^3
    samples and each sample dropped into the model voxel it falls in."""
    shape = field.grid.shape
    occ = np.zeros(int(np.prod(shape)), np.float64)
    best = np.zeros_like(occ); owner = np.zeros(occ.size, np.int32)
    A = np.asarray(affine_lps, np.float64)
    inv = np.linalg.inv(np.asarray(field.grid.directions, np.float64))          # world = origin + index @ directions
    origin = np.asarray(field.grid.origin, np.float64)
    src_mm3 = abs(np.linalg.det(A[:3, :3])); dst_mm3 = float(np.prod(field.grid.spacing))
    off = (np.arange(sub) + 0.5) / sub - 0.5
    offs = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    for v in np.unique(labels):
        if v == 0:
            continue
        idx = np.argwhere(labels == v).astype(np.float64)
        pts = (idx[:, None, :] + offs[None]).reshape(-1, 3)
        world = pts @ A[:3, :3].T + A[:3, 3]
        m = np.rint((world - origin) @ inv).astype(np.int64)
        ok = np.all((m >= 0) & (m < np.asarray(shape)), 1)
        flat, cnt = np.unique(np.ravel_multi_index(m[ok].T, shape), return_counts=True)
        part = cnt * (src_mm3 / len(offs)) / dst_mm3
        occ[flat] += part
        win = part > best[flat]
        best[flat[win]] = part[win]; owner[flat[win]] = int(v)
    return np.minimum(occ, 1.0).reshape(shape).astype(np.float32), owner.reshape(shape)


def lesion_overlaps(sw: Sweep, occ: np.ndarray, owner: np.ndarray, voxel_ml: float):
    """``(box, lesion, ml)`` rows for every box that holds any of a lesion."""
    rows = []
    for i, (lo, hi) in enumerate(zip(sw.lo, sw.hi)):
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        o = occ[sl]
        if not o.any():
            continue
        w = np.bincount(owner[sl].ravel(), weights=o.ravel())
        rows += [(i, int(l), float(w[l] * voxel_ml)) for l in np.flatnonzero(w) if l]
    return np.asarray(rows, np.float64).reshape(-1, 3)


# -- a normal model that knows where it is ----------------------------------------------------
def _features(coords: np.ndarray, fill: np.ndarray | None) -> np.ndarray:
    c = np.asarray(coords, np.float64)
    cols = [np.ones(len(c)), *c.T, *(c ** 2).T, c[:, 0] * c[:, 1], c[:, 0] * c[:, 2], c[:, 1] * c[:, 2]]
    if fill is not None:
        f = np.asarray(fill, np.float64)
        cols += [f, f ** 2]
    return np.stack(cols, 1)


@dataclass
class PositionedNormal:
    """Mean = a quadratic in the box's place in the organ (and, optionally, its fill); precision
    = the shrunk covariance of what that leaves. Distance is Mahalanobis in the residual."""

    coef: np.ndarray
    precision: np.ndarray
    uses_fill: bool
    n: int
    shrinkage: float

    @classmethod
    def fit(cls, X, coords, fill=None, ridge: float = 1e-3) -> "PositionedNormal":
        X = np.asarray(X, np.float64); F = _features(coords, fill)
        coef = np.linalg.solve(F.T @ F + ridge * len(F) * np.eye(F.shape[1]), F.T @ X)
        C, s = shrunk_covariance(X - F @ coef)
        return cls(coef, np.linalg.inv(C), fill is not None, len(X), s)

    def distance(self, X, coords, fill=None) -> np.ndarray:
        D = np.asarray(X, np.float64) - _features(coords, fill if self.uses_fill else None) @ self.coef
        # ((D @ P) * D).sum(1), not einsum("ij,jk,ik->i"): unoptimized, einsum runs the three
        # operands as one single-threaded C loop with no BLAS - 156x slower at 704 dimensions
        # (5.6 s against 0.036 s for 8,000 boxes; equal to 2e-15), and it was the whole of the
        # null model's atlas analysis (py-spy, 2026-09-22)
        return np.sqrt(np.maximum(((D @ self.precision) * D).sum(1), 0.0))


# -- scoring ----------------------------------------------------------------------------------
def auc(scores, positive) -> float:
    """Mann-Whitney AUC with ties at half, in numpy. NaN when a side is empty."""
    s = np.asarray(scores, np.float64); y = np.asarray(positive, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, ranks) / cnt)[inv]                                  # average rank within ties
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def local_maxima(ijk: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Which boxes score at least as high as all 26 neighbours on the sweep's lattice (a missing
    neighbour never wins). These are the DETECTIONS: a threshold crossed by a ridge of forty
    overlapping boxes is one finding, not forty."""
    if not len(ijk):
        return np.zeros(0, bool)
    ijk = np.asarray(ijk, np.int64) - np.asarray(ijk, np.int64).min(0) + 1
    grid = np.full(tuple(ijk.max(0) + 2), -np.inf)
    grid[tuple(ijk.T)] = scores
    top = np.ones(len(ijk), bool)
    for d in np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij"), -1).reshape(-1, 3):
        if d.any():
            top &= scores >= grid[tuple((ijk + d).T)]
    return top & np.isfinite(scores)


def froc(scans: list[dict], fp_per_scan=(0.5, 1.0, 2.0, 4.0), bins_mm=(0, 10, 20, 40, 1e9),
         hit_lesion: float = 0.25, hit_box: float = 0.25) -> dict:
    """Per-lesion sensitivity at fixed false positives per scan.

    Each scan: ``{"score", "ijk", "tumor_ml", "organ_ml", "overlaps" (box, lesion, ml),
    "lesions" {id: {"ml", "diameter_mm"}}}``. A box HITS a lesion when it holds ``hit_lesion`` of
    the lesion or the lesion is ``hit_box`` of the box's organ; a lesion's score is its best
    hitting box. A FALSE POSITIVE is a local maximum of the score on a box with no tumor in it -
    boxes that merely touch a lesion are neither. Thresholds are set on the pooled false
    positives: one threshold for everybody, as a tool has."""
    fps, lesions = [], []
    for s in scans:
        sc = np.asarray(s["score"], np.float64)
        clean = np.asarray(s["tumor_ml"]) <= 0
        fps.append(sc[local_maxima(s["ijk"], np.where(np.isfinite(sc), sc, -np.inf)) & clean])
        best = {}
        for b, l, ml in np.asarray(s["overlaps"]).reshape(-1, 3):
            b, l = int(b), int(l)
            L = s["lesions"].get(l)
            if L is None or not np.isfinite(sc[b]):
                continue
            if ml >= hit_lesion * L["ml"] or ml >= hit_box * max(float(s["organ_ml"][b]), 1e-9):
                best[l] = max(best.get(l, -np.inf), sc[b])
        lesions += [(L["diameter_mm"], best.get(l, -np.inf)) for l, L in s["lesions"].items()]
    fp = np.sort(np.concatenate(fps))[::-1] if fps else np.zeros(0)
    d = np.array([x[0] for x in lesions]); ls = np.array([x[1] for x in lesions])
    out = {"scans": len(scans), "lesions": int(len(d)), "never_hit": int(np.isneginf(ls).sum()), "at": {}}
    for f in fp_per_scan:
        k = int(round(f * len(scans)))
        t = fp[k - 1] if 0 < k <= len(fp) else (-np.inf if k > len(fp) else np.inf)
        hit = (ls >= t) & np.isfinite(ls)                                         # a lesion no box hits is never found
        row = {"threshold": float(t) if np.isfinite(t) else None, "all": round(float(hit.mean()), 3) if len(ls) else None}
        for a, b in zip(bins_mm[:-1], bins_mm[1:]):
            m = (d >= a) & (d < b)
            row[f"{a}-{b if b < 1e8 else 'up'}mm"] = [round(float(hit[m].mean()), 3) if m.any() else None, int(m.sum())]
        out["at"][str(f)] = row
    return out
