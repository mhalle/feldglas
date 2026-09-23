"""The receiving end of an embedding field: open one, gate it with a mask on any grid, pool it,
and ask whether a region is unlike normal tissue - numpy and zarr, nothing of the encoder.

This is the field's packed README (``field_readme.md``) as code, and it reads the FILE, not the
encoder contract: no model grid, no ``Field``, no rankfield geometry - world coordinates
throughout, as a client in any language would work. The TypeScript client planned beside it
(docs/embedding-field.md, Not built 9) implements the same README; conformance fixtures will hold
the two together. Started 2026-09-23.

    f = open_field("series.zarr.zip")                         # decoded lattices, centers, extents
    labels = read_seg_nrrd("series.seg.nrrd")                 # haversack's labels (feldglas.labels)
    liver = structure_mask(labels, "liver")
    vecs = organ_vectors(f, labels)                           # one vector per structure, shared mean off
    ref = Reference.build([(g, mask_g) for g, mask_g in normals])
    s = ref.score(f, liver)                                   # per-token distance, flagged sites

What is measured behind the defaults is in EXPLORATION 5.14 (medseg) and the README: the fine
lattice; tokens at least half inside the region; the 5 nearest normal tokens; the threshold at the
largest distance a held-out normal token reached; boundary tokens left out; sites within 15 mm.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field as _field

import numpy as np

from .labels import LabelMap, read_seg_nrrd

FORMAT, FORMAT_VERSIONS, EXTENSION, EXTENSION_VERSION = "feldglas-field", {"0.2"}, "embedding", "0.1"
AXIS_LINEAR = "embedding.linear_along_axis"

__all__ = ["open_field", "FieldView", "Lattice", "Mask", "structure_mask", "read_seg_nrrd", "labels_at",
           "occupancy", "tokens_in", "unit", "organ_vectors", "knn_distance", "Reference", "Scores", "Site"]


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

    def index_of(self, points) -> np.ndarray:
        """The token index (rounded, possibly outside the lattice) at each world point: solve
        ``steps.T @ i = p - origin``."""
        return np.rint(np.linalg.solve(self.steps.T, (np.atleast_2d(points) - self.origin).T).T).astype(np.int64)

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
        return tuple(self.space.get(k, "") for k in ("model", "weights", "layer", "stage"))


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
        return self.lattices[-1]


def _decode(values: np.ndarray, transforms, where: str) -> np.ndarray:
    if not transforms:
        return np.asarray(values, np.float32)
    if len(transforms) != 1 or transforms[0].get("name") != AXIS_LINEAR:
        raise ValueError(f"{where}: stored through {[t.get('name') for t in transforms]}; only {AXIS_LINEAR!r} is "
                         "decoded here - these values are not tokens")
    p = transforms[0]["parameters"]
    if p.get("axis") != values.ndim - 1 or len(p["slope"]) != values.shape[-1] or len(p["intercept"]) != values.shape[-1]:
        raise ValueError(f"{where}: {AXIS_LINEAR} does not fit the channel axis")
    return values.astype(np.float32) * np.asarray(p["slope"], np.float32) + np.asarray(p["intercept"], np.float32)


def open_field(path) -> FieldView:
    """A ``<series>.zarr.zip`` field, every lattice decoded. Refuses a format, version or value
    transform it does not know, rather than guess at it."""
    import zarr
    path = str(pathlib.Path(path).expanduser())
    root = zarr.open_group(store=zarr.storage.ZipStore(path, mode="r"), mode="r")
    ext = ((root.attrs.asdict().get("duckn") or {}).get("extensions") or {}).get(EXTENSION)
    if not ext:
        raise ValueError(f"{path}: no {EXTENSION!r} extension on the root - not an embedding field")
    if ext.get("format") != FORMAT or ext.get("format_version") not in FORMAT_VERSIONS or ext.get("version") != EXTENSION_VERSION:
        raise ValueError(f"{path}: {ext.get('format')!r} {ext.get('format_version')!r}, extension {ext.get('version')!r}; "
                         f"this reader knows {FORMAT} {sorted(FORMAT_VERSIONS)} with {EXTENSION} {EXTENSION_VERSION}")
    lattices = []
    for name in ext["group"]["members"]:
        arr = root[name]
        d = arr.attrs.asdict()["duckn"]
        axes = d["axes"]
        if [a.get("kind") for a in axes] != ["space", "space", "space", "list"]:
            raise ValueError(f"{path}: {name} is not three space axes and a list axis")
        if d.get("space") != "left-posterior-superior":
            raise ValueError(f"{path}: {name} is in {d.get('space')!r}, not left-posterior-superior")
        e = (d.get("extensions") or {}).get(EXTENSION) or {}
        v = _decode(arr[:], d.get("value_transforms"), f"{path}:{name}")
        th = [a.get("thickness") for a in axes[:3]]
        lattices.append(Lattice(
            name=name, tokens=v.reshape(-1, v.shape[-1]), shape=tuple(int(s) for s in v.shape[:3]),
            steps=np.array([a["space_direction"] for a in axes[:3]], float), origin=np.asarray(d["space_origin"], float),
            extent=(tuple(e["extent"]["lo"]), tuple(e["extent"]["hi"])) if "extent" in e else None,
            kernel=tuple(e["kernel"]) if "kernel" in e else None,
            thickness=None if all(t is None for t in th) else tuple(th),
            support_offset=tuple(e["support"]["offset"]) if "support" in e else None,
            space=dict(e.get("space") or {}), metric=e.get("metric", "cosine"), normalized=bool(e.get("normalized", False))))
    return FieldView(lattices, dict(ext.get("provenance") or {}), ext.get("data_box"), path)


# -- masks on any grid ---------------------------------------------------------------------------
@dataclass
class Mask:
    """``values[i, j, k]`` (boolean, or a fraction in [0, 1]) with ``world_lps = affine @ (i, j, k, 1)``
    - the form ``feldglas.labels.LabelMap`` has, so a segmentation, a drawn ROI or a CT-grid array
    all gate alike."""

    values: np.ndarray
    affine_lps: np.ndarray

    @classmethod
    def from_grid(cls, values, directions, origin) -> "Mask":
        """From duckn's form: one LPS row per array axis (a full step), and the first sample's place."""
        A = np.eye(4); A[:3, :3] = np.asarray(directions, float).T; A[:3, 3] = np.asarray(origin, float)
        return cls(np.asarray(values), A)


def structure_mask(labels: LabelMap, *structures: str) -> Mask:
    """The union of named structures, by NAME (a trailing ``*`` is a prefix), never by label value."""
    vals = []
    for s in structures:
        hit = [v for n, v in labels.names.items() if n.startswith(s[:-1])] if s.endswith("*") else \
              ([labels.names[s]] if s in labels.names else [])
        if not hit:
            raise KeyError(f"no structure matches {s!r} ({len(labels.names)} named)")
        vals += hit
    return Mask(np.isin(labels.values, vals), np.asarray(labels.affine_lps, float))


def labels_at(lattice: Lattice, labels) -> np.ndarray:
    """The label (or mask value) under each token's CENTER; 0 outside the map."""
    values = labels.values
    inv = np.linalg.inv(np.asarray(labels.affine_lps, float))
    ijk = np.rint(lattice.centers @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < values.shape), axis=1)
    out = np.zeros(len(ijk), values.dtype)
    out[ok] = values[tuple(ijk[ok].T)]
    return out


def occupancy(lattice: Lattice, mask: Mask, samples_per_token: int = 4) -> np.ndarray:
    """``(N,)``: the fraction of each token's box inside ``mask``, estimated from the mask's voxel
    centers (sub-sampled to about ``samples_per_token`` per token step along each axis) sent to the
    token they fall in. Only the mask's bounding box, grown by a token, is visited."""
    v = np.asarray(mask.values)
    A = np.asarray(mask.affine_lps, float)
    spacing = np.linalg.norm(A[:3, :3], axis=0)
    step = float(np.linalg.norm(lattice.steps, axis=1).min())
    stride = np.maximum((step / samples_per_token / spacing).astype(int), 1)
    nz = np.nonzero(v)
    if not len(nz[0]):
        return np.zeros(len(lattice.tokens), np.float32)
    grow = np.ceil(np.linalg.norm(lattice.steps, axis=1).max() / spacing).astype(int)
    lo = np.maximum(np.array([a.min() for a in nz]) - grow, 0)
    hi = np.minimum(np.array([a.max() for a in nz]) + grow + 1, v.shape)
    n_all = np.zeros(len(lattice.tokens)); n_in = np.zeros(len(lattice.tokens))
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0], stride[0]), np.arange(lo[1], hi[1], stride[1]), indexing="ij")
    for k in range(lo[2], hi[2], stride[2]):              # one slab at a time: megabytes, not gigabytes
        ijk = np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, k)], 1)
        t = lattice.index_of(ijk @ A[:3, :3].T + A[:3, 3])
        ok = np.all((t >= 0) & (t < lattice.shape), axis=1)
        flat = np.ravel_multi_index(tuple(t[ok].T), lattice.shape)
        n_all += np.bincount(flat, minlength=len(n_all))
        n_in += np.bincount(flat, weights=v[tuple(ijk[ok].T)].astype(float), minlength=len(n_in))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_all > 0, n_in / np.maximum(n_all, 1), 0.0).astype(np.float32)


def tokens_in(lattice: Lattice, mask: Mask, rule="center", min_occupancy: float = 0.5,
              drop_boundary: bool = False) -> np.ndarray:
    """Indices of in-extent tokens the mask owns: ``"center"`` - the token's center is inside;
    ``"touch"`` - any of its box is (the gate RADAR's head was trained on); ``"occupancy"`` - at
    least ``min_occupancy`` of its box is."""
    keep = lattice.interior if drop_boundary else lattice.in_extent
    if rule == "center":
        sel = labels_at(lattice, mask).astype(float) > 0.5
    elif rule == "touch":
        sel = occupancy(lattice, mask) > 0
    elif rule == "occupancy":
        sel = occupancy(lattice, mask) >= min_occupancy
    else:
        raise ValueError(f"rule {rule!r}: center, touch or occupancy")
    return np.flatnonzero(keep & sel)


# -- vectors -------------------------------------------------------------------------------------
def unit(x) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def organ_vectors(f: FieldView, labels: LabelMap, lattice: int = -1, min_tokens: int = 5,
                  structures=None) -> dict[str, np.ndarray]:
    """One vector per structure on one lattice, the README's recipe exactly: tokens whose center is
    in the structure, unit length, averaged (not re-normalized), minus the body's mean unit token
    (not re-normalized), then unit length. Compare two by their dot product."""
    lat = f.lattices[lattice]
    lab = labels_at(lat, labels)
    t = unit(lat.tokens)
    inside = lat.in_extent
    ref = t[inside & (lab > 0)].mean(0)
    out = {}
    for name, value in labels.names.items():
        if structures is not None and name not in structures:
            continue
        sel = inside & (lab == value)
        if sel.sum() >= min_tokens:
            out[name] = unit(t[sel].mean(0) - ref)
    return out


def knn_distance(x: np.ndarray, reference: np.ndarray, k: int = 5, chunk: int = 4096) -> np.ndarray:
    """``1 - mean cosine to the k most similar reference tokens``, for unit-length rows."""
    out = np.empty(len(x), np.float32)
    for a in range(0, len(x), chunk):
        s = x[a:a + chunk] @ reference.T
        out[a:a + chunk] = 1.0 - np.sort(s, axis=1)[:, -k:].mean(1)
    return out


# -- normal tissue -------------------------------------------------------------------------------
@dataclass
class Site:
    center: np.ndarray               # LPS mm, the mean of its flagged tokens
    tokens: int
    peak: float                      # the largest distance among them


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
    idx = np.flatnonzero(flagged)
    label = -np.ones(len(idx), int); n = 0
    for a in range(len(idx)):
        if label[a] >= 0:
            continue
        stack = [a]; label[a] = n
        while stack:
            b = stack.pop()
            near = np.flatnonzero((label < 0) & (np.linalg.norm(centers[idx] - centers[idx[b]], axis=1) <= radius))
            label[near] = n; stack.extend(near.tolist())
        n += 1
    sites = [Site(centers[idx[label == i]].mean(0), int((label == i).sum()), float(distance[idx[label == i]].max()))
             for i in range(n)]
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
    lattice: int = -1
    k: int = 5
    min_occupancy: float = 0.5
    drop_boundary: bool = True
    meta: dict = _field(default_factory=dict)

    @classmethod
    def build(cls, normals, lattice: int = -1, k: int = 5, min_occupancy: float = 0.5,
              drop_boundary: bool = True) -> "Reference":
        """``normals``: ``(FieldView, Mask)`` pairs - each normal scan and its normal-tissue region
        (an organ from a segmentation, eroded to keep tokens interior, or a drawn ROI)."""
        normals = list(normals)
        if len(normals) < 2:
            raise ValueError("a reference needs at least two normal scans: its threshold is set by leaving one out")
        keys = {f.lattices[lattice].key() for f, _ in normals}
        if len(keys) != 1:
            raise ValueError(f"normals from different encoders or layers cannot make one reference: {sorted(keys)}")
        parts = []
        for f, m in normals:
            lat = f.lattices[lattice]
            i = tokens_in(lat, m, "occupancy", min_occupancy, drop_boundary)
            if not len(i):
                raise ValueError(f"{f.path}: no token of its region")
            parts.append(unit(lat.tokens[i]))
        held = [knn_distance(p, np.concatenate([q for j, q in enumerate(parts) if j != s]), k) for s, p in enumerate(parts)]
        allheld = np.concatenate(held)
        return cls(np.concatenate(parts), np.concatenate([np.full(len(p), s) for s, p in enumerate(parts)]),
                   float(allheld.max()), allheld, keys.pop(), lattice, k, min_occupancy, drop_boundary,
                   {"normals": [f.path for f, _ in normals], "tokens_per_normal": [len(p) for p in parts]})

    def score(self, f: FieldView, region: Mask, radius: float = 15.0) -> Scores:
        """Every token of ``region`` in ``f``, its distance from normal, and the flagged tokens
        grouped into sites (within ``radius`` mm), strongest first. A site is a place to look."""
        lat = f.lattices[self.lattice]
        if lat.key() != self.key:
            raise ValueError(f"{f.path}: tokens of {lat.key()} cannot be scored against a reference of {self.key}")
        i = tokens_in(lat, region, "occupancy", self.min_occupancy, self.drop_boundary)
        c = lat.centers[i]
        d = knn_distance(unit(lat.tokens[i]), self.tokens, self.k)
        return Scores(i, c, d, self.threshold, _sites(c, d, d > self.threshold, radius))
