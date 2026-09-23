"""The receiving end of an embedding field: open one, gate it with a mask on any grid, pool it,
and ask whether a region is unlike normal tissue - numpy and zarr, nothing of the encoder.

This is the field's packed README (``field_readme.md``) as code, and it reads the FILE, not the
encoder contract: no model grid, no ``Field``, no rankfield geometry - world coordinates
throughout, as a client in any language would work. The TypeScript client planned beside it
(docs/embedding-field.md, Not built 9) implements the same README; conformance fixtures will hold
the two together. Started 2026-09-23.

    f = open_field("series.zarr.zip")                         # decoded lattices, centers, extents
    labels = read_seg_nrrd("series.seg.nrrd")                 # haversack's, or 3D Slicer's (layered)
    liver = structure_mask(labels, "liver", optional=["liver_tumor"])   # organ + lesion segments, by name
    vecs = organ_vectors(f, labels)                           # one vector per structure, shared mean off
    ref = Reference.build([(g, mask_g) for g, mask_g in normals])
    s = ref.score(f, liver)                                   # per-token distance, flagged sites

What is measured behind the defaults is in EXPLORATION 5.14 (medseg) and the README: the fine
lattice; tokens at least half inside the region; the 5 nearest normal tokens; the threshold at the
largest distance a held-out normal token reached; boundary tokens left out; sites within 15 mm.

Four adversarial reviews (2026-09-23) shaped what follows: occupancy is sampled INSIDE each token's
box (sending the mask's voxels to tokens at a stride missed 3-33 % of an organ's touching tokens
and let a cropped mask report slivers as full); the reader checks everything the store's reader
checks (a reversed member list made ``fine`` the coarsest lattice); and nothing non-finite, empty
or integer-without-a-transform passes as an answer.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field as _field

import numpy as np

from .labels import LabelMap, read_seg_nrrd

FORMAT, FORMAT_VERSIONS, EXTENSION, EXTENSION_VERSION = "feldglas-field", {"0.2"}, "embedding", "0.1"
AXIS_LINEAR = "embedding.linear_along_axis"
PLACEMENT_TOLERANCE_MM = 1e-3

__all__ = ["open_field", "FieldView", "Lattice", "Mask", "structure_mask", "seg_mask", "read_seg_nrrd", "labels_at",
           "occupancy", "tokens_in", "unit", "organ_vectors", "knn_distance", "Reference", "Scores", "Site"]


def _round(x):
    """Half away from the lower index - ``floor(x + 0.5)`` - the same rule everywhere: ``np.rint``
    rounds halves to EVEN, which sent voxel centers on token faces to alternate tokens."""
    return np.floor(np.asarray(x) + 0.5).astype(np.int64)


# -- the file ------------------------------------------------------------------------------------
@dataclass
class Lattice:
    """One lattice as the file states it: ``tokens[n]`` is the token at ``index[n]`` (C order over
    ``shape``); every geometric fact is world (LPS mm)."""

    name: str
    tokens: np.ndarray              # (N, C) float32, decoded
    shape: tuple[int, int, int]
    steps: np.ndarray               # (3, 3): row a is space_direction a, a full step in mm
    origin: np.ndarray              # the center of token (0, 0, 0)
    extent: tuple[tuple[int, int, int], tuple[int, int, int]] | None
    kernel: tuple[int, int, int] | None
    thickness: tuple | None         # per space axis, mm (a token's measured reach, full width)
    support_offset: tuple | None    # evidence center minus drawn center, mm along each axis
    space: dict                     # model, weights, layer, stage: the comparability key
    metric: str = "cosine"
    normalized: bool = False

    @property
    def index(self) -> np.ndarray:
        """``(N, 3)`` token indices, in token order."""
        return np.stack(np.unravel_index(np.arange(len(self.tokens)), self.shape), -1)

    @property
    def centers(self) -> np.ndarray:
        """``(N, 3)`` token centers, LPS mm."""
        return self.origin + self.index @ self.steps

    def continuous_index(self, points) -> np.ndarray:
        """Where world points fall in token-index units: solve ``steps.T @ i = p - origin``."""
        return np.linalg.solve(self.steps.T, (np.atleast_2d(points) - self.origin).T).T

    def index_of(self, points) -> np.ndarray:
        """The token index at each world point (possibly outside the lattice)."""
        return _round(self.continuous_index(points))

    @property
    def in_extent(self) -> np.ndarray:
        """``(N,)``: tokens whose box holds scanned image (all of them when the file states no extent)."""
        if self.extent is None:
            return np.ones(len(self.tokens), bool)
        lo, hi = self.extent
        i = self.index
        return np.all((i >= lo) & (i < hi), axis=1)

    @property
    def interior(self) -> np.ndarray:
        """In the extent and not on its boundary (index ``lo`` or ``hi - 1`` along any axis): the
        README's rule for tokens that saw only image."""
        if self.extent is None:
            return self.in_extent
        lo, hi = (np.asarray(b) for b in self.extent)
        i = self.index
        return np.all((i > lo) & (i < hi - 1), axis=1)

    def key(self) -> tuple:
        """What two vectors must share to be compared. A file that names no layer is keyed by the
        lattice's kernel and shape instead - otherwise every lattice of it would share one key, and a
        coarse lattice could be scored against a fine reference."""
        layer = self.space.get("layer") or f"kernel {self.kernel} of {self.shape}"
        return (self.space.get("model", ""), self.space.get("weights", ""), layer, self.space.get("stage", ""))


@dataclass
class FieldView:
    lattices: list[Lattice]
    provenance: dict
    data_box: dict | None
    path: str = ""

    @property
    def encoder(self) -> str:
        return self.provenance.get("encoder", "")

    @property
    def license(self) -> str:
        return self.provenance.get("license", "")

    @property
    def input_grid(self) -> dict | None:
        """The input CT's grid as delivered (``shape``, ``directions`` one LPS row per array axis,
        ``origin``) - compare it with your CT's before trusting a mask drawn on that CT."""
        return (self.provenance.get("input") or {}).get("grid")

    @property
    def fine(self) -> Lattice:
        """The lattice with the smallest token - by its step volume, never by its place in the list."""
        return min(self.lattices, key=lambda l: abs(float(np.linalg.det(l.steps))))


def _need(d: dict, key: str, where: str):
    if key not in d:
        raise ValueError(f"{where}: no {key!r} - not a field this reader understands")
    return d[key]


def _decode(values: np.ndarray, transforms, where: str) -> np.ndarray:
    if not transforms:
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"{where}: {values.dtype} values with no value transform - integers are not tokens")
        out = np.asarray(values, np.float32)
    else:
        if len(transforms) != 1 or transforms[0].get("name") != AXIS_LINEAR:
            raise ValueError(f"{where}: stored through {[t.get('name') for t in transforms]}; only {AXIS_LINEAR!r} is "
                             "decoded here - these values are not tokens")
        p = transforms[0].get("parameters") or {}
        slope, intercept = (np.asarray(p.get(k, []), np.float32) for k in ("slope", "intercept"))
        if p.get("axis") != values.ndim - 1 or len(slope) != values.shape[-1] or len(intercept) != values.shape[-1]:
            raise ValueError(f"{where}: {AXIS_LINEAR} does not fit the channel axis")
        out = values.astype(np.float32) * slope + intercept
    if not np.isfinite(out).all():
        raise ValueError(f"{where}: {int((~np.isfinite(out)).sum())} token values are not finite")
    return out


def _model_grid(lat: Lattice):
    """The model grid a lattice implies (README "The model grid"): its step and first voxel's center."""
    k = np.asarray(lat.kernel, float)
    rows = lat.steps / k[:, None]
    return rows, lat.origin - ((k - 1) / 2.0) @ rows


def open_field(path, token: str | None = None) -> FieldView:
    """A ``<series>.zarr.zip`` field - a path, or an ``http(s)://`` URL - every lattice decoded - after the checks the store's own reader
    makes: one group, in order; extents inside their lattices and agreeing with the data box; every
    lattice placing the same model grid; finite, non-singular geometry; no integer read as tokens.
    Refuses a format, version or value transform it does not know, rather than guess at it."""
    import zarr
    from .fetch import local
    path = str(local(path, token, expect="zip"))              # an http(s) URL is fetched into the cache
    if not pathlib.Path(path).is_file():
        raise FileNotFoundError(f"{path}: no such field")
    root = zarr.open_group(store=zarr.storage.ZipStore(path, mode="r"), mode="r")
    ext = ((root.attrs.asdict().get("duckn") or {}).get("extensions") or {}).get(EXTENSION)
    if not ext:
        raise ValueError(f"{path}: no {EXTENSION!r} extension on the root - not an embedding field")
    if ext.get("format") != FORMAT or ext.get("format_version") not in FORMAT_VERSIONS or ext.get("version") != EXTENSION_VERSION:
        raise ValueError(f"{path}: {ext.get('format')!r} {ext.get('format_version')!r}, extension {ext.get('version')!r}; "
                         f"this reader knows {FORMAT} {sorted(FORMAT_VERSIONS)} with {EXTENSION} {EXTENSION_VERSION}")
    group = _need(ext, "group", path)
    members = list(_need(group, "members", path))
    box = ext.get("data_box")
    for name in members:
        if name not in root:
            raise ValueError(f"{path}:{name}: the root lists this lattice and the file has no such array")
    lattices = []
    for j, name in enumerate(members):
        where = f"{path}:{name}"
        arr = root[name]
        d = _need(arr.attrs.asdict(), "duckn", where)
        axes = _need(d, "axes", where)
        if [a.get("kind") for a in axes] != ["space", "space", "space", "list"]:
            raise ValueError(f"{where}: not three space axes and a list axis")
        if d.get("space") != "left-posterior-superior":
            raise ValueError(f"{where}: in {d.get('space')!r}, not left-posterior-superior")
        e = (d.get("extensions") or {}).get(EXTENSION) or {}
        g = e.get("group") or {}
        if (g.get("id"), g.get("member"), g.get("members")) != (group.get("id"), j, len(members)):
            raise ValueError(f"{where}: says it is member {g.get('member')} of {g.get('members')} in {g.get('id')!r}; "
                             f"the root lists it as {j} of {len(members)} in {group.get('id')!r}")
        steps = np.array([_need(a, "space_direction", where) for a in axes[:3]], float)
        origin = np.asarray(_need(d, "space_origin", where), float)
        if not (np.isfinite(steps).all() and np.isfinite(origin).all()) or abs(np.linalg.det(steps)) < 1e-9:
            raise ValueError(f"{where}: its placement is not finite or its steps are singular")
        v = _decode(arr[:], d.get("value_transforms"), where)
        shape = tuple(int(s) for s in v.shape[:3])
        extent = None
        if "extent" in e:
            lo, hi = tuple(int(x) for x in e["extent"]["lo"]), tuple(int(x) for x in e["extent"]["hi"])
            if not all(0 <= l < h <= n for l, h, n in zip(lo, hi, shape)):
                raise ValueError(f"{where}: extent {lo}..{hi} is not a box inside the lattice {shape}")
            extent = (lo, hi)
        kernel = tuple(int(k) for k in e["kernel"]) if "kernel" in e else None
        if box is not None and kernel is not None:
            want = (tuple(l // k for l, k in zip(box["lo"], kernel)), tuple(-(-h // k) for h, k in zip(box["hi"], kernel)))
            if extent != want:
                raise ValueError(f"{where}: extent {extent} is not what the data box gives ({want})")
        th = [a.get("thickness") for a in axes[:3]]
        lattices.append(Lattice(
            name=name, tokens=v.reshape(-1, v.shape[-1]), shape=shape, steps=steps, origin=origin, extent=extent,
            kernel=kernel, thickness=None if all(t is None for t in th) else tuple(th),
            support_offset=tuple(e["support"]["offset"]) if "support" in e else None,
            space=dict(e.get("space") or {}), metric=e.get("metric", "cosine"), normalized=bool(e.get("normalized", False))))
    if not lattices:
        raise ValueError(f"{path}: the group lists no lattices")
    if all(l.kernel is not None for l in lattices):          # every lattice must imply the same model grid
        rows0, first0 = _model_grid(lattices[0])
        for l in lattices[1:]:
            rows, first = _model_grid(l)
            if not (np.allclose(rows, rows0, atol=1e-6) and np.allclose(first, first0, atol=PLACEMENT_TOLERANCE_MM)):
                raise ValueError(f"{path}: {l.name} is not placed where {lattices[0].name} puts the model grid - "
                                 "the lattices disagree about where the patient is")
    return FieldView(lattices, dict(ext.get("provenance") or {}), box, path)


# -- masks on any grid ---------------------------------------------------------------------------
@dataclass
class Mask:
    """``values[i, j, k]`` - boolean, or a fraction in [0, 1] - with ``world_lps = affine @ (i, j, k, 1)``:
    the form ``feldglas.labels.LabelMap`` has, so a segmentation, a drawn ROI or a CT-grid array all
    gate alike. A label map is not a mask: take structures from it with :func:`structure_mask`."""

    values: np.ndarray
    affine_lps: np.ndarray

    def __post_init__(self):
        self.values = np.asarray(self.values)
        self.affine_lps = np.asarray(self.affine_lps, float)
        if self.values.ndim != 3:
            raise ValueError(f"a mask is a 3-D array, not {self.values.ndim}-D")
        if self.affine_lps.shape != (4, 4) or not np.isfinite(self.affine_lps).all() \
                or abs(np.linalg.det(self.affine_lps[:3, :3])) < 1e-12:
            raise ValueError("a mask's affine must be a finite, non-singular 4 x 4")
        if self.values.dtype != bool and self.values.size and (self.values.min() < 0 or self.values.max() > 1):
            raise ValueError(f"mask values run {self.values.min()}..{self.values.max()}; a mask is boolean or a "
                             "fraction in [0, 1] (a label map: use structure_mask)")

    @classmethod
    def from_grid(cls, values, directions, origin) -> "Mask":
        """From duckn's form: one LPS row per array axis (a full step), and the first sample's place."""
        A = np.eye(4); A[:3, :3] = np.asarray(directions, float).T; A[:3, 3] = np.asarray(origin, float)
        return cls(np.asarray(values), A)

    def at(self, points) -> np.ndarray:
        """The mask's value at world points (the voxel whose center is nearest); 0 outside the array."""
        inv = np.linalg.inv(self.affine_lps)
        ijk = _round(np.asarray(points) @ inv[:3, :3].T + inv[:3, 3])
        ok = np.all((ijk >= 0) & (ijk < self.values.shape), axis=-1)
        out = np.zeros(ijk.shape[:-1], np.float32)
        out[ok] = self.values[tuple(np.moveaxis(ijk[ok], -1, 0))]
        return out


def structure_mask(labels: LabelMap, *structures: str, optional=()) -> Mask:
    """The union of named structures from a label map - usually a ``.seg.nrrd`` read by
    :func:`read_seg_nrrd`, haversack's (one layer) or 3D Slicer's (layered, when segments overlap) -
    by NAME (a trailing ``*`` is a prefix), never by label value, and across layers. Every name in
    ``structures`` must be there; names in ``optional`` join when present and are skipped when not -
    a lesion segment is absent from a scan with no lesion (haversack lists only the structures a
    scan has). The organ with its lesion segments:
    ``structure_mask(labels, "kidney_left", optional=["kidney_cyst_left"])``."""
    names = list(structures) + [n for n in optional
                                if (any(m.startswith(n[:-1]) for m in labels.names) if n.endswith("*") else n in labels.names)]
    if not names:
        raise ValueError("no structure to make a mask of: every name was optional and none is in this map")
    return Mask(labels.mask(*names), np.asarray(labels.affine_lps, float))


def seg_mask(path, *structures: str, optional=(), names=None, token: str | None = None) -> Mask:
    """``structure_mask(read_seg_nrrd(path, names), *structures, optional=...)``: a region straight
    from a ``.seg.nrrd`` (``names`` only for a plain labelmap with no segment table)."""
    return structure_mask(read_seg_nrrd(path, names, token=token), *structures, optional=optional)


def grid_mismatch(f: FieldView, labels) -> str | None:
    """Why a label map is NOT on the CT this field was computed from - or None when its grid is the
    field's recorded input grid (or the field recorded none: then nothing can be checked, and the
    caller should say so). A mask from another scan gates the wrong anatomy silently: a review
    scored one patient's field with another's labels - 680 tokens "flagged", exit 0 (2026-09-23)."""
    g = f.input_grid
    if not g:
        return None
    shape = tuple(int(v) for v in np.asarray(labels.values).shape[-3:])
    A = np.asarray(labels.affine_lps, float)
    D = np.asarray(g["directions"], float)
    if shape != tuple(g["shape"]):
        return f"the labels are {shape} voxels and the field's CT was {tuple(g['shape'])}"
    if not np.allclose(A[:3, :3].T, D, atol=1e-3) or not np.allclose(A[:3, 3], g["origin"], atol=0.05):
        return (f"the labels are placed elsewhere than the field's CT (origin {np.round(A[:3, 3], 2).tolist()} "
                f"against {np.round(g['origin'], 2).tolist()})")
    return None


def _layers_at(lattice: Lattice, labels: LabelMap) -> np.ndarray:
    """``(layers, N)``: the label under each token's center on every layer of the map (0 outside)."""
    stack = labels.values if labels.layered else labels.values[None]
    inv = np.linalg.inv(np.asarray(labels.affine_lps, float))
    ijk = _round(lattice.centers @ inv[:3, :3].T + inv[:3, 3])
    ok = np.all((ijk >= 0) & (ijk < stack.shape[1:]), axis=1)
    out = np.zeros((len(stack), len(ijk)), stack.dtype)
    out[:, ok] = stack[(slice(None), *ijk[ok].T)]
    return out


def labels_at(lattice: Lattice, labels) -> np.ndarray:
    """The label (or mask value) under each token's CENTER; 0 outside the map. A layered map has no
    single label per voxel (its segments overlap): use :func:`structure_mask` with it."""
    if isinstance(labels, LabelMap) and labels.layered:
        raise ValueError("a layered segmentation has no single label per voxel: use structure_mask")
    values = np.asarray(labels.values)
    inv = np.linalg.inv(np.asarray(labels.affine_lps, float))
    ijk = _round(lattice.centers @ inv[:3, :3].T + inv[:3, 3])
    ok = np.all((ijk >= 0) & (ijk < values.shape), axis=1)
    out = np.zeros(len(ijk), values.dtype)
    out[ok] = values[tuple(ijk[ok].T)]
    return out


def occupancy(lattice: Lattice, mask: Mask, points_per_axis=None, max_points: int = 32) -> np.ndarray:
    """``(N,)``: the fraction of each token's BOX inside ``mask`` - its volume, estimated at a regular
    grid of points inside the box (sub-cell centers) where the mask is read (0 outside its array).
    Sampling the token, not the mask, is what makes every denominator the box: a mask array cropped to
    an ROI, or coarser than the tokens, reads right. The default puts points at most 0.75 of the mask's
    finest spacing apart (at most ``max_points`` per axis), so a single mask voxel inside a box is
    always hit; with ``points_per_axis`` a multiple of a model-grid kernel, points sit at voxel centers
    and the result is exact. Only tokens whose box can reach the mask's nonzero voxels are visited."""
    v = mask.values
    A = mask.affine_lps
    out = np.zeros(len(lattice.tokens), np.float32)
    nz = np.nonzero(v)
    if not len(nz[0]):
        return out
    step = np.linalg.norm(lattice.steps, axis=1)
    if points_per_axis is None:
        fine = float(np.linalg.norm(A[:3, :3], axis=0).min())
        s = np.minimum(np.maximum(np.ceil(step / (0.75 * fine) - 1e-9), 2), max_points).astype(int)
    else:
        s = np.broadcast_to(np.asarray(points_per_axis, int), (3,))
    lo = np.array([a.min() for a in nz]) - 0.5
    hi = np.array([a.max() for a in nz]) + 0.5
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    ci = lattice.continuous_index(corners @ A[:3, :3].T + A[:3, 3])
    t_lo = np.maximum(np.floor(ci.min(0) - 0.5).astype(int), 0)
    t_hi = np.minimum(np.ceil(ci.max(0) + 0.5).astype(int) + 1, lattice.shape)
    if np.any(t_hi <= t_lo):
        return out
    cand = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(t_lo, t_hi)], indexing="ij"), -1).reshape(-1, 3)
    grid = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) / n - 0.5 for n in s], indexing="ij"), -1).reshape(-1, 3)
    offsets = grid @ lattice.steps                          # sub-cell centers, world, around a token's center
    flat = np.ravel_multi_index(tuple(cand.T), lattice.shape)
    centers = lattice.origin + cand @ lattice.steps
    chunk = max(1, int(4_000_000 // len(offsets)))
    for a in range(0, len(cand), chunk):
        pts = centers[a:a + chunk, None, :] + offsets[None]
        out[flat[a:a + chunk]] = mask.at(pts).mean(1)
    return out


def tokens_in(lattice: Lattice, mask: Mask, rule="center", min_occupancy: float = 0.5,
              drop_boundary: bool = False, points_per_axis=None) -> np.ndarray:
    """Indices of in-extent tokens the mask owns: ``"center"`` - the mask at the token's center is at
    least a half; ``"touch"`` - any of its box is inside (the gate RADAR's head was trained on);
    ``"occupancy"`` - at least ``min_occupancy`` of its box is."""
    keep = lattice.interior if drop_boundary else lattice.in_extent
    if rule == "center":
        sel = mask.at(lattice.centers) >= 0.5
    elif rule == "touch":
        sel = occupancy(lattice, mask, points_per_axis) > 0
    elif rule == "occupancy":
        sel = occupancy(lattice, mask, points_per_axis) >= min_occupancy
    else:
        raise ValueError(f"rule {rule!r}: center, touch or occupancy")
    return np.flatnonzero(keep & sel)


# -- vectors -------------------------------------------------------------------------------------
def unit(x) -> np.ndarray:
    """Rows at unit length; a zero or non-finite row is refused - its direction is undefined, and one
    NaN once made a reference's threshold NaN, so that nothing was ever flagged."""
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    bad = ~np.isfinite(n) | (n == 0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} vector(s) are zero or not finite - they have no direction")
    return x / n


def organ_vectors(f: FieldView, labels: LabelMap, lattice: int | None = None, min_tokens: int = 5,
                  structures=None, counts: dict | None = None) -> dict[str, np.ndarray]:
    """One vector per structure on one lattice (default: the finest), the README's recipe exactly:
    in-extent tokens whose center is in the structure (at least ``min_tokens``), unit length,
    averaged (not re-normalized), minus the mean unit token of the body - every labeled structure -
    (not re-normalized), then unit length. A structure indistinguishable from the body's mean has no
    direction and is left out. Compare two vectors by their dot product. ``counts``, if given, is
    filled with every structure's token count (those below ``min_tokens`` included)."""
    lat = f.fine if lattice is None else f.lattices[lattice]
    lab = _layers_at(lat, labels)                        # (layers, N): a layered map's segments may overlap
    t = unit(lat.tokens)
    inside = lat.in_extent
    body = inside & (lab > 0).any(0)
    if not body.any():
        raise ValueError("no in-extent token lies in any labeled structure: is the label map on this CT?")
    ref = t[body].mean(0)
    out = {}
    for name, value in labels.names.items():
        if structures is not None and name not in structures:
            continue
        sel = inside & (lab[labels.layer_of.get(name, 0) if labels.layered else 0] == value)
        if counts is not None:
            counts[name] = int(sel.sum())
        if sel.sum() >= min_tokens:
            d = t[sel].mean(0) - ref
            if np.linalg.norm(d) > 1e-6:
                out[name] = d / np.linalg.norm(d)
    return out


def knn_distance(x: np.ndarray, reference: np.ndarray, k: int = 5) -> np.ndarray:
    """``1 - mean cosine to the k most similar reference tokens``, for unit-length rows. Chunked so a
    chunk's similarity matrix stays near 64 MB whatever the reference's size."""
    if not 1 <= k <= len(reference):
        raise ValueError(f"k = {k} with {len(reference)} reference tokens")
    out = np.empty(len(x), np.float32)
    chunk = max(1, int(16_000_000 // max(len(reference), 1)))
    for a in range(0, len(x), chunk):
        s = x[a:a + chunk] @ reference.T
        top = np.partition(s, len(reference) - k, axis=1)[:, -k:]
        out[a:a + chunk] = 1.0 - top.mean(1)
    return out


# -- normal tissue -------------------------------------------------------------------------------
@dataclass
class Site:
    center: np.ndarray               # LPS mm, the mean of its flagged tokens' centers
    tokens: int
    peak: float                      # the largest distance among them
    members: np.ndarray | None = None  # positions of its tokens in the scores' arrays


@dataclass
class Scores:
    index: np.ndarray                # token indices scored, into the lattice
    centers: np.ndarray              # their centers, LPS mm
    distance: np.ndarray
    threshold: float
    sites: list[Site]

    @property
    def flagged(self) -> np.ndarray:
        return self.distance > self.threshold


def _sites(centers, distance, flagged, radius) -> list[Site]:
    """Flagged tokens joined into sites by SINGLE LINKAGE: two tokens within ``radius`` mm are one
    site (so a site can chain further). A grid of ``radius`` cells keeps it near-linear; strongest
    site first."""
    idx = np.flatnonzero(flagged)
    if not len(idx):
        return []
    c = centers[idx]
    parent = np.arange(len(idx))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    cells: dict[tuple, list[int]] = {}
    for a, key in enumerate(map(tuple, np.floor(c / radius).astype(int))):
        cells.setdefault(key, []).append(a)
    for key, members in cells.items():
        near = [b for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                for b in cells.get((key[0] + dz, key[1] + dy, key[2] + dx), ())]
        near = np.asarray(near)
        for a in members:
            hit = near[np.linalg.norm(c[near] - c[a], axis=1) <= radius]
            for b in hit:
                ra, rb = find(a), find(int(b))
                if ra != rb:
                    parent[rb] = ra
    roots = np.array([find(a) for a in range(len(idx))])
    sites = [Site(c[roots == r].mean(0), int((roots == r).sum()), float(distance[idx[roots == r]].max()), idx[roots == r])
             for r in np.unique(roots)]
    return sorted(sites, key=lambda s: -s.peak)


@dataclass
class Reference:
    """Normal tissue from a few normal scans of the SAME protocol, and the threshold those scans set
    for themselves: each scored against the others, the largest distance a held-out normal token
    reached. The README's "Is a region unlike normal tissue?", token by token."""

    tokens: np.ndarray               # (M, C) unit length
    subject: np.ndarray              # (M,) which normal each came from
    threshold: float
    held_out: np.ndarray             # every held-out normal token's distance
    key: tuple                       # the lattice's comparability key (model, weights, layer, stage)
    lattice: int | None = None       # None: each field's finest lattice
    k: int = 5
    min_occupancy: float = 0.5
    drop_boundary: bool = True
    meta: dict = _field(default_factory=dict)
    centers: np.ndarray | None = None  # (M, 3) LPS mm of each normal token: where a large held-out distance is

    @staticmethod
    def _lattice(f: FieldView, lattice) -> Lattice:
        return f.fine if lattice is None else f.lattices[lattice]

    @classmethod
    def build(cls, normals, lattice: int | None = None, k: int = 5, min_occupancy: float = 0.5,
              drop_boundary: bool = True, meta: dict | None = None) -> "Reference":
        """``normals``: ``(FieldView, Mask)`` pairs - each normal scan and its normal-tissue region
        (an organ from a segmentation, eroded to keep tokens interior, or a drawn ROI)."""
        normals = list(normals)
        if len(normals) < 2:
            raise ValueError("a reference needs at least two normal scans: its threshold is set by leaving one out")
        keys = {cls._lattice(f, lattice).key() for f, _ in normals}
        if len(keys) != 1:
            raise ValueError(f"normals from different encoders or layers cannot make one reference: {sorted(keys)}")
        sources = [f.provenance.get("source") or f.path for f, _ in normals]
        if len(set(sources)) != len(sources):
            raise ValueError(f"the same scan is listed twice among the normals ({sorted({x for x in sources if sources.count(x) > 1})}): "
                             "it would count its own tokens as normal and set the threshold too low")
        parts, where = [], []
        for f, m in normals:
            lat = cls._lattice(f, lattice)
            i = tokens_in(lat, m, "occupancy", min_occupancy, drop_boundary)
            if not len(i):
                raise ValueError(f"{f.path}: no token of its region")
            parts.append(unit(lat.tokens[i])); where.append(lat.centers[i])
        held = [knn_distance(p, np.concatenate([q for j, q in enumerate(parts) if j != s]), k) for s, p in enumerate(parts)]
        allheld = np.concatenate(held)
        return cls(np.concatenate(parts), np.concatenate([np.full(len(p), s) for s, p in enumerate(parts)]),
                   float(allheld.max()), allheld, keys.pop(), lattice, k, min_occupancy, drop_boundary,
                   {"normals": [f.path for f, _ in normals], "sources": sources,
                    "tokens_per_normal": [len(p) for p in parts],
                    "license": sorted({f.license for f, _ in normals}), **(meta or {})}, np.concatenate(where))

    def per_normal(self) -> list[dict]:
        """How each normal's held-out tokens scored: the threshold is ONE token's distance, and one
        normal can set it far above the others (a review found 0.657 from one where the other two
        reached 0.195 and 0.051) - this is how to see that."""
        out = []
        for s_ in range(int(self.subject.max()) + 1):
            d = self.held_out[self.subject == s_]
            i = int(np.argmax(d))
            row = {"normal": (self.meta.get("sources") or self.meta.get("normals") or [None] * (s_ + 1))[s_],
                   "tokens": int(len(d)), "p50": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max())}
            if self.centers is not None:
                row["max_at_lps"] = np.round(self.centers[self.subject == s_][i], 1).tolist()
            out.append(row)
        return out

    # -- the file: a zarr zip like a field's, so one reader stack serves both ------------------------
    FORMAT, FORMAT_VERSION = "feldglas-reference", "0.1"

    def save(self, path) -> pathlib.Path:
        """``<name>.zarr.zip``: the reference tokens (float32), which normal each came from, every
        held-out distance, and - on the root's ``embedding`` extension - the comparability key, the
        threshold and how it was set, and ``meta`` (the normals, their regions, anything the builder
        noted, e.g. the protocol). A reference derives from the normals' fields and inherits their
        license; ``meta["license"]`` says it."""
        import os, shutil, tempfile, zipfile
        import zarr
        from zarr.storage import LocalStore
        path = pathlib.Path(path)
        if not str(path).endswith(".zarr.zip"):
            raise ValueError(f"{path}: a reference is written as <name>.zarr.zip")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = pathlib.Path(tempfile.mkdtemp(prefix=path.name + ".staging-", dir=path.parent))
        try:
            root = zarr.create_group(store=LocalStore(str(staging)))
            arrays = [("tokens", self.tokens.astype(np.float32)), ("subject", self.subject.astype(np.int32)),
                      ("held_out", self.held_out.astype(np.float32))]
            if self.centers is not None:
                arrays.append(("centers", np.asarray(self.centers, np.float64)))
            for name, a in arrays:
                arr = root.create_array(name, shape=a.shape, dtype=a.dtype, compressors=zarr.codecs.ZstdCodec(level=3))
                arr[:] = a
            root.attrs.update({"duckn": {"version": "1.0", "intent": "embedding-reference", "extensions": {EXTENSION: {
                "version": EXTENSION_VERSION, "format": self.FORMAT, "format_version": self.FORMAT_VERSION,
                "key": dict(zip(("model", "weights", "layer", "stage"), self.key)), "threshold": self.threshold,
                "k": self.k, "min_occupancy": self.min_occupancy, "drop_boundary": self.drop_boundary,
                "lattice": self.lattice, "meta": self.meta}}}})
            partial = path.with_name(path.name + ".partial")
            with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
                for fp in sorted(q for q in staging.rglob("*") if q.is_file()):
                    zf.write(fp, fp.relative_to(staging).as_posix())
            os.replace(partial, path)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return path

    @classmethod
    def load(cls, path, token: str | None = None) -> "Reference":
        """A reference written by :meth:`save` - a path or an ``http(s)://`` URL."""
        import zarr
        from .fetch import local
        p = str(local(path, token, expect="zip"))
        root = zarr.open_group(store=zarr.storage.ZipStore(p, mode="r"), mode="r")
        e = ((root.attrs.asdict().get("duckn") or {}).get("extensions") or {}).get(EXTENSION) or {}
        if (e.get("format"), e.get("format_version"), e.get("version")) != (cls.FORMAT, cls.FORMAT_VERSION, EXTENSION_VERSION):
            raise ValueError(f"{path}: {e.get('format')!r} {e.get('format_version')!r}; this reader knows "
                             f"{cls.FORMAT} {cls.FORMAT_VERSION}")
        bad = lambda why: ValueError(f"{path}: not a usable reference - {why}")
        try:
            k = e["key"]
            key = (k["model"], k["weights"], k["layer"], k["stage"])
            threshold, kk, occ, drop, lat = e["threshold"], e["k"], e["min_occupancy"], e["drop_boundary"], e.get("lattice")
            tokens, subject, held = root["tokens"][:], root["subject"][:], root["held_out"][:]
        except KeyError as x:
            raise bad(f"it has no {x.args[0]!r}") from None
        if not (isinstance(threshold, (int, float)) and np.isfinite(threshold) and 0 <= threshold <= 2):
            raise bad(f"threshold {threshold!r} is not a distance in [0, 2]")
        if not (isinstance(occ, (int, float)) and 0 < occ <= 1):
            raise bad(f"min_occupancy {occ!r} is not in (0, 1]")
        if not isinstance(drop, bool) or not (lat is None or isinstance(lat, int)):
            raise bad("drop_boundary must be true or false, and lattice null or an integer")
        if not (tokens.ndim == 2 and len(tokens) == len(subject) == len(held)):
            raise bad(f"{len(tokens)} tokens, {len(subject)} subjects and {len(held)} held-out distances disagree")
        if not (isinstance(kk, int) and 1 <= kk <= len(tokens)):
            raise bad(f"k {kk!r} with {len(tokens)} tokens")
        n = np.linalg.norm(tokens, axis=1)
        if not (np.isfinite(tokens).all() and np.allclose(n, 1, atol=1e-3)):
            raise bad("its tokens are not finite unit vectors")
        centers = root["centers"][:] if "centers" in root else None
        return cls(tokens, subject, float(threshold), held, key, lat, kk, float(occ), drop, dict(e.get("meta") or {}), centers)

    def score(self, f: FieldView, region: Mask, radius: float = 15.0) -> Scores:
        """Every token of ``region`` in ``f``, its distance from normal, and the flagged tokens
        grouped into sites (within ``radius`` mm), strongest first. A site is a place to look. An
        empty region is refused: no token scored is not "normal"."""
        if self.lattice is not None and not -len(f.lattices) <= self.lattice < len(f.lattices):
            raise ValueError(f"the reference is for lattice {self.lattice}; {f.path} has {len(f.lattices)}")
        lat = self._lattice(f, self.lattice)
        if lat.key() != self.key:
            raise ValueError(f"{f.path}: tokens of {lat.key()} cannot be scored against a reference of {self.key}")
        i = tokens_in(lat, region, "occupancy", self.min_occupancy, self.drop_boundary)
        if not len(i):
            raise ValueError(f"{f.path}: no token of the region (in the extent{', off its boundary' if self.drop_boundary else ''})"
                             " - nothing was scored")
        c = lat.centers[i]
        d = knn_distance(unit(lat.tokens[i]), self.tokens, self.k)
        return Scores(i, c, d, self.threshold, _sites(c, d, d > self.threshold, radius))
