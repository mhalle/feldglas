"""Label maps from elsewhere, as gates: read one, put it on a field's model grid, select BY NAME.

The encoder's own mask is one source of gates and often not the best (RADAR's kidney mask holds
0.79 of the expert tumor volume, and under half in 33 of 148 KiTS scans; EXPLORATION 5.6). The
family's segmentations are the other: haversack writes a `.seg.nrrd` whose header names every
segment, and a consumer selects structures by NAME, never by label value - values differ between
tasks, versions and ecosystems, names are the convention. So a `LabelMap` keeps the names it was
given, and everything downstream asks for `"kidney_left"`, not 3.

A `.seg.nrrd` is a text header, a blank line and a (gzip) block in Fortran order, so it is read with
the standard library: importing this module costs numpy and nothing else, and no SimpleITK is
needed to gate a field. Only what haversack writes is accepted - one 3-D layer, LPS space - and
anything else is refused by name rather than guessed at.
"""
from __future__ import annotations

import gzip
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
from rankfield.geometry import Geometry

_DTYPES = {"unsigned char": np.uint8, "uchar": np.uint8, "uint8": np.uint8, "uint8_t": np.uint8,
           "short": np.int16, "int16": np.int16, "unsigned short": np.uint16, "ushort": np.uint16, "uint16": np.uint16,
           "int": np.int32, "int32": np.int32, "unsigned int": np.uint32, "uint32": np.uint32}


@dataclass
class LabelMap:
    """``values[i, j, k]`` with ``world_lps = affine @ (i, j, k, 1)``, and the name of every value."""

    values: np.ndarray
    affine_lps: np.ndarray
    names: dict[str, int]
    task: str = ""                                   # what produced it, when the header says
    meta: dict = field(default_factory=dict)

    def value_of(self, name: str) -> int:
        try:
            return self.names[name]
        except KeyError:
            near = sorted(n for n in self.names if name.split("_")[0] in n)[:6]
            raise KeyError(f"no structure {name!r} in this label map"
                           + (f" (it has {', '.join(near)})" if near else f" ({len(self.names)} structures)")) from None

    def on_grid(self, grid: Geometry) -> "GridLabels":
        """The label under the CENTER of every voxel of ``grid`` (0 outside the map): a nearest-
        neighbor pull, a slab at a time so a 12 M voxel model grid costs megabytes. Gates take
        each token's OCCUPANCY from this, so a voxel-level nearest choice is averaged over the
        hundreds to thousands of model voxels a token holds."""
        inv = np.linalg.inv(np.asarray(self.affine_lps, np.float64))
        D = np.asarray(grid.directions, np.float64); o = np.asarray(grid.origin, np.float64)
        Z, Y, X = grid.shape
        yx = np.stack(np.meshgrid(np.arange(Y), np.arange(X), indexing="ij"), -1).reshape(-1, 2).astype(np.float64)
        plane = yx[:, :1] * D[1] + yx[:, 1:] * D[2] + o                         # world of (0, y, x)
        out = np.zeros(grid.shape, self.values.dtype)
        hi = np.asarray(self.values.shape)
        for z in range(Z):
            w = plane + z * D[0]
            ijk = np.rint(w @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
            ok = np.all((ijk >= 0) & (ijk < hi), 1)
            row = np.zeros(len(w), self.values.dtype)
            row[ok] = self.values[tuple(ijk[ok].T)]
            out[z] = row.reshape(Y, X)
        return GridLabels(out, dict(self.names), self.task)


@dataclass
class GridLabels:
    """A label map already on a field's model grid."""

    values: np.ndarray
    names: dict[str, int]
    task: str = ""

    def mask(self, *structures: str) -> np.ndarray:
        """The union of the named structures. A trailing ``*`` matches a prefix (``"kidney_*"``)."""
        vals = []
        for s in structures:
            hit = [v for n, v in self.names.items() if n.startswith(s[:-1])] if s.endswith("*") else \
                  ([self.names[s]] if s in self.names else [])
            if not hit:
                raise KeyError(f"no structure matches {s!r} ({len(self.names)} named; e.g. {', '.join(sorted(self.names)[:5])})")
            vals += hit
        return np.isin(self.values, vals)

    def present(self, min_voxels: int = 1) -> list[str]:
        n = np.bincount(self.values.ravel().astype(np.int64), minlength=max(self.names.values(), default=0) + 1)
        return sorted(name for name, v in self.names.items() if n[v] >= min_voxels)


def read_seg_nrrd(path) -> LabelMap:
    raw = pathlib.Path(path).read_bytes()
    cut = raw.find(b"\n\n")
    if not raw.startswith(b"NRRD") or cut < 0:
        raise ValueError(f"{path}: not an NRRD file")
    head: dict[str, str] = {}
    for line in raw[:cut].decode("utf8", "replace").split("\n")[1:]:
        if line.startswith("#"):
            continue
        for sep in (":=", ": "):
            if sep in line:
                a, b = line.split(sep, 1); head[a.strip()] = b.strip(); break
    if head.get("dimension") != "3":
        raise ValueError(f"{path}: dimension {head.get('dimension')} - a layered (4-D) segmentation is not read here; "
                         "haversack writes one layer")
    if head.get("space") != "left-posterior-superior":
        raise ValueError(f"{path}: space {head.get('space')!r}, not left-posterior-superior")
    if head.get("type") not in _DTYPES:
        raise ValueError(f"{path}: voxel type {head.get('type')!r} is not an integer label type")
    enc = head.get("encoding", "raw")
    if enc in ("gzip", "gz"):
        data = gzip.decompress(raw[cut + 2:])
    elif enc == "raw":
        data = raw[cut + 2:]
    else:
        raise ValueError(f"{path}: encoding {enc!r} (gzip and raw are read)")
    sizes = [int(v) for v in head["sizes"].split()]
    dt = np.dtype(_DTYPES[head["type"]]).newbyteorder(">" if head.get("endian") == "big" else "<")
    values = np.frombuffer(data, dt, count=int(np.prod(sizes))).reshape(sizes, order="F")
    vec = lambda s: [float(v) for v in s.strip("()").split(",")]
    A = np.eye(4); A[:3, :3] = np.array([vec(v) for v in head["space directions"].split()]).T; A[:3, 3] = vec(head["space origin"])
    names: dict[str, int] = {}
    for k, name in head.items():
        if k.startswith("Segment") and k.endswith("_Name"):
            v = int(head[k[:-len("_Name")] + "_LabelValue"])
            if name in names and names[name] != v:
                raise ValueError(f"{path}: {name!r} names two label values - a lookup by name would be a silent choice")
            names[name] = v
    if len(set(names.values())) != len(names):
        raise ValueError(f"{path}: two segment names share one label value")
    prov = {}
    for k in ("haversack_provenance",):
        if k in head:
            try:
                prov = json.loads(head[k])
            except ValueError:
                pass
    return LabelMap(values, A, names, task=str(prov.get("task", "")), meta=prov)
