"""A field on disk, with a version gate. Two forms, told apart by the name.

``<series>.zarr.zip`` - ``feldglas-field 0.2`` (2026-09-22), the family's form. A zarr v3 group,
packed into a zip with STORED entries so each chunk is range-addressable, whose arrays are the
lattices - ``lattice_<j>``, shaped ``(Z, Y, X, C)`` in C order, so one token's embedding is
contiguous (pooling gathers tokens) and every chunk holds the whole channel axis. Each array is
placed in the world by duckn alone: three ``space`` axes with ``space_direction`` and ``cell``
centering, ``space_origin`` at the center of the first token's box, and a ``list`` axis for the
channels (a range kind: never resampled). The ``feldglas`` extension (unregistered, 0.1) carries
what duckn does not: on each array its kernel and index, on the root the encoder, the lattice
list and the provenance with the license the embeddings inherit - held here until duckn's own
provenance extension, drafted but not implemented, exists. Tokens are stored as written (fp16
from the exporters) with NO value transform: per-channel int8 needs a transform duckn does not
define yet (``component_linear``, a convention change of its own).

**No mask, and no reference to one** (the user's decision, 2026-09-22): a field is the embeddings
placed in the world and nothing about how they will be gated. A client brings its own mask on any
grid and pools on its own CPU. A 0.2 file never carries ``native_mask``, whatever the Field held.

The model grid is not stored: it is DERIVED from each lattice's own placement and kernel, and the
reader checks that every lattice derives the same one - two lattices that disagree about where
the patient is are refused, never averaged. One fact, one place.

``<series>.npz`` - ``feldglas-field 0.1`` (2026-09-20): ``tokens<j>`` per lattice, ``meta`` as
JSON (kernels, the model grid in rankfield's form, provenance, native labels), ``native_mask``
when the encoder had one. Still read, so the 1,680 fields already on R2 stay usable; written only
when asked for by name.

Readers refuse what they do not know, as the rest of the family does: a field whose geometry is
misread draws a finding on the wrong organ, silently.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
import zipfile

import numpy as np
from rankfield.geometry import Geometry

from .contract import Field, Provenance

FORMAT, VERSION = "feldglas-field", "0.1"
ZARR_VERSION = "0.2"
KNOWN_VERSIONS = {"0.1", "0.2"}
EXTENSION, EXTENSION_VERSION = "feldglas", "0.1"
DUCKN_VERSION = "1.0"                                 # geometry and a list axis: nothing past 1.0
SPACE = "left-posterior-superior"                     # rankfield's and haversack's world
CHUNK = 16                                            # tokens per spatial chunk edge; channels whole
_PROVENANCE_FIELDS = ("encoder", "code", "weights", "preprocessing", "license", "source")


def is_zarr_zip(path) -> bool:
    return str(path).lower().endswith(".zarr.zip")


def write_field(path, field: Field, token_dtype=np.float16) -> pathlib.Path:
    """``<name>.zarr.zip`` writes 0.2 (the duckn form); ``<name>.npz`` writes 0.1."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_zarr_zip(path):
        return _write_zarr(path, field, token_dtype)
    meta = {"format": FORMAT, "version": VERSION,
            "kernels": [list(k) for k in field.kernels],
            "grid": {"shape": list(field.grid.shape), "directions": [list(r) for r in field.grid.directions],
                     "origin": list(field.grid.origin)},
            "exact_geometry": bool(field.exact_geometry),
            "provenance": _provenance_record(field.provenance),
            "native_labels": list(field.native_labels)}
    arrays = {f"tokens{j}": np.asarray(t, token_dtype) for j, t in enumerate(field.tokens)}
    if field.native_mask is not None:
        arrays["native_mask"] = np.asarray(field.native_mask)
    tmp = path.with_name(path.name + ".tmp.npz")             # never leave half a field under the real name
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    tmp.replace(path)
    return path


def read_meta(path) -> dict:
    """The field's description without its tokens: ``format``, ``version``, ``kernels`` and
    ``provenance`` in either form (0.1 adds its grid and native labels)."""
    if is_zarr_zip(path):
        root = _open_group(path)
        ext = _root_extension(root, path)
        return {"format": FORMAT, "version": ext["format_version"], "encoder": ext["encoder"],
                "kernels": [list(_array_extension(root[n], path)["kernel"]) for n in ext["lattices"]],
                "provenance": ext["provenance"]}
    with np.load(path) as z:
        if "meta" not in z.files:
            raise ValueError(f"{path}: no meta block - not a feldglas field")
        meta = json.loads(str(z["meta"]))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: format {meta.get('format')!r}, not {FORMAT!r} "
                         "(a 2026-09-20 pilot file? use adapters.radar.read_pilot_field)")
    if meta.get("version") != VERSION:
        raise ValueError(f"{path}: {FORMAT} version {meta.get('version')!r} in an .npz; this reader knows "
                         f"{VERSION} there (0.2 is a .zarr.zip)")
    return meta


def read_field(path) -> Field:
    if is_zarr_zip(path):
        return _read_zarr(path)
    meta = read_meta(path)
    with np.load(path) as z:
        tokens = [z[f"tokens{j}"] for j in range(len(meta["kernels"]))]
        mask = z["native_mask"] if "native_mask" in z.files else None
    g = meta["grid"]
    return Field(tokens=tokens, kernels=[tuple(k) for k in meta["kernels"]],
                 grid=Geometry(shape=tuple(g["shape"]), directions=tuple(tuple(r) for r in g["directions"]),
                               origin=tuple(g["origin"])),
                 provenance=Provenance(**meta["provenance"]), native_mask=mask,
                 native_labels=tuple(meta.get("native_labels", ())),
                 exact_geometry=bool(meta.get("exact_geometry", True)))


# -- 0.2: the duckn zarr zip ---------------------------------------------------------------------
def _provenance_record(p: Provenance) -> dict:
    return {**{k: getattr(p, k) for k in _PROVENANCE_FIELDS}, "extra": p.extra}


def _lattice_attrs(field: Field, j: int) -> dict:
    from duckn import AxisMetadata, DucknMetadata
    from duckn.models import duckn_attrs
    g = field.lattice_geometry(j)
    axes = [AxisMetadata(kind="space", centering="cell", unit="mm",
                         space_direction=[round(float(v), 9) for v in row]) for row in g.directions]
    axes.append(AxisMetadata(kind="list"))
    return duckn_attrs(DucknMetadata(
        version=DUCKN_VERSION, space=SPACE, space_origin=[round(float(v), 9) for v in g.origin], axes=axes,
        extensions={EXTENSION: {"version": EXTENSION_VERSION, "lattice": j,
                                "kernel": [int(k) for k in field.kernels[j]]}}))


def _root_attrs(field: Field, names: list[str]) -> dict:
    from duckn import DucknMetadata
    from duckn.models import duckn_attrs
    return duckn_attrs(DucknMetadata(version=DUCKN_VERSION, extensions={EXTENSION: {
        "version": EXTENSION_VERSION, "format": FORMAT, "format_version": ZARR_VERSION,
        "encoder": field.provenance.encoder, "lattices": names,
        "provenance": _provenance_record(field.provenance)}}))


def _write_zarr(path: pathlib.Path, field: Field, token_dtype) -> pathlib.Path:
    import zarr
    from zarr.storage import LocalStore
    if not field.exact_geometry:
        raise ValueError(f"{path}: a field without exact geometry (a 2026-09-20 pilot) cannot be written as "
                         f"{FORMAT} {ZARR_VERSION}, which places every lattice in the world - keep it as .npz")
    staging = pathlib.Path(tempfile.mkdtemp(prefix=path.name + ".staging-", dir=path.parent))
    try:
        root = zarr.create_group(store=LocalStore(str(staging)))
        names = []
        for j, t in enumerate(field.tokens):
            shape = (*field.lattice_shape(j), field.widths[j])
            data = np.ascontiguousarray(np.asarray(t, token_dtype).reshape(shape))   # row-major = (Z, Y, X, C)
            name = f"lattice_{j}"
            arr = root.create_array(name, shape=shape, dtype=data.dtype,
                                    chunks=tuple(min(CHUNK, s) for s in shape[:3]) + (shape[3],),
                                    compressors=zarr.codecs.ZstdCodec(level=3),
                                    attributes=_lattice_attrs(field, j))
            arr[:] = data
            names.append(name)
        root.attrs.update(_root_attrs(field, names))
        partial = path.with_name(path.name + ".partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for fp in sorted(q for q in staging.rglob("*") if q.is_file()):
                zf.write(fp, fp.relative_to(staging).as_posix())
        os.replace(partial, path)                     # never half a field under the real name
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return path


def _open_group(path):
    import zarr
    from zarr.storage import ZipStore
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    return zarr.open_group(store=ZipStore(str(p), mode="r"), mode="r")


def _root_extension(root, path) -> dict:
    ext = (root.attrs.asdict().get("duckn") or {}).get("extensions", {}).get(EXTENSION)
    if not ext:
        raise ValueError(f"{path}: no {EXTENSION!r} extension on the root - not a feldglas field")
    if ext.get("format") != FORMAT or ext.get("format_version") not in KNOWN_VERSIONS - {"0.1"}:
        raise ValueError(f"{path}: {ext.get('format')!r} {ext.get('format_version')!r}; this reader knows "
                         f"{FORMAT} {ZARR_VERSION} in a .zarr.zip")
    if ext.get("version") != EXTENSION_VERSION:
        raise ValueError(f"{path}: extension {EXTENSION!r} {ext.get('version')!r}; this reader knows {EXTENSION_VERSION} "
                         "(a 0.x extension may change incompatibly)")
    return ext


def _array_extension(arr, path) -> dict:
    ext = (arr.attrs.asdict().get("duckn") or {}).get("extensions", {}).get(EXTENSION)
    if not ext or ext.get("version") != EXTENSION_VERSION:
        raise ValueError(f"{path}: array {arr.name!r} carries no {EXTENSION} {EXTENSION_VERSION} extension")
    return ext


def _placement(arr, path) -> Geometry:
    """A lattice array's own placement, read from its duckn attributes: three spatial axes, then
    the channel axis."""
    d = arr.attrs.asdict()["duckn"]
    if d.get("space") != SPACE:
        raise ValueError(f"{path}: {arr.name!r} is in space {d.get('space')!r}; this reader places {SPACE!r}")
    axes = d.get("axes") or []
    kinds = [a.get("kind") for a in axes]
    if kinds != ["space", "space", "space", "list"] or any(a.get("centering") != "cell" for a in axes[:3]):
        raise ValueError(f"{path}: {arr.name!r} axes {kinds}; a lattice is three cell-centered space axes then a list axis")
    return Geometry(shape=tuple(int(s) for s in arr.shape[:3]),
                    directions=tuple(tuple(float(v) for v in a["space_direction"]) for a in axes[:3]),
                    origin=tuple(float(v) for v in d["space_origin"]))


def _model_grid(lattice: Geometry, kernel) -> Geometry:
    """The model grid a lattice was pooled from: a step of 1/kernel of the lattice's, the first
    voxel (k - 1) / 2 model steps before the first token's center, kernel times as many samples."""
    rows = np.asarray(lattice.directions, float) / np.asarray(kernel, float)[:, None]
    origin = np.asarray(lattice.origin, float) - ((np.asarray(kernel, float) - 1) / 2.0) @ rows
    return Geometry(shape=tuple(int(s) * int(k) for s, k in zip(lattice.shape, kernel)),
                    directions=tuple(tuple(float(v) for v in r) for r in rows),
                    origin=tuple(float(v) for v in origin))


def _read_zarr(path) -> Field:
    root = _open_group(path)
    ext = _root_extension(root, path)
    tokens, kernels, placed = [], [], []
    for name in ext["lattices"]:
        arr = root[name]
        a = _array_extension(arr, path)
        placed.append(_placement(arr, path)); kernels.append(tuple(int(k) for k in a["kernel"]))
        data = arr[:]
        tokens.append(data.reshape(-1, data.shape[-1]))
    grid = _model_grid(placed[0], kernels[0])
    field = Field(tokens=tokens, kernels=kernels, grid=grid, provenance=Provenance(**ext["provenance"]))
    for j, g in enumerate(placed):                    # every lattice must derive the same model grid
        want = field.lattice_geometry(j)
        if (tuple(want.shape) != tuple(g.shape)
                or not np.allclose(want.directions, g.directions, atol=1e-6)
                or not np.allclose(want.origin, g.origin, atol=1e-4)):
            raise ValueError(f"{path}: lattice {j} is not placed where lattice 0's model grid puts it "
                             f"(stored origin {g.origin}, derived {want.origin}) - refusing a field whose "
                             "lattices disagree about where the patient is")
    return field
