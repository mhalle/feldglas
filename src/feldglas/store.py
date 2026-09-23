"""A field on disk, with a version gate. Two forms, told apart by the name.

``<series>.zarr.zip`` - ``feldglas-field 0.2`` (2026-09-22), the family's form. A zarr v3 group,
packed into a zip with STORED entries so each chunk is range-addressable, whose arrays are the
lattices - ``lattice_<j>``, shaped ``(Z, Y, X, C)`` in C order, so one token's embedding is
contiguous (pooling gathers tokens) and every chunk holds the whole channel axis. Each array is
placed in the world by duckn alone: three ``space`` axes with ``space_direction`` and ``cell``
centering, ``space_origin`` at the center of the first token's box, and a ``list`` axis for the
channels (a range kind: never resampled), with each axis's ``thickness`` the token's measured
extent where the encoder knows it, and ``intent: "embedding-field"``. What duckn core does not say
rides in the ``embedding`` extension (unregistered, 0.1, defined in docs/embedding-field.md - the
ONE design record; change it there first): on each array the comparability key (model, weights,
layer, stage), metric, kernel, support offset and its place in the group; on the root the file's
version, the group's members, and the provenance - license and the input CT's identity and grid
included - until duckn's own provenance extension, drafted but not implemented, exists. Tokens
are stored as written (fp16 from the exporters) with NO value transform: per-channel int8 waits
for a ``linear``-along-an-axis transform in duckn core.

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

from .contract import Embedding, Field, Provenance

FORMAT, VERSION = "feldglas-field", "0.1"
ZARR_VERSION = "0.2"
KNOWN_VERSIONS = {"0.1", "0.2"}
EXTENSION, EXTENSION_VERSION = "embedding", "0.1"   # general, unregistered (duckn 456516e rule)
SCHEMA = "README.md (in this file)"                   # the design record is feldglas docs/embedding-field.md
README = pathlib.Path(__file__).with_name("field_readme.md")   # packed into every field as README.md
INTENT = "embedding-field"
#: int8 tokens decode through a per-channel slope and intercept along the channel axis - a value
#: transform duckn core does not define yet (2026-09-22, the user: "try" it without a standard).
#: duckn's rule makes that SAFE: a reader meeting an unknown transform name treats the value mapping
#: as unknown and offers only the stored integers, never passes them off as the embeddings - where
#: scales kept only in an extension would leave a plain duckn reader taking int8 for the values.
#: Namespaced by the extension that defines it (field_readme.md), so a later duckn transform cannot
#: collide with it. Measured on a RADAR field against its fp16 tokens: token cosine >= 0.9998,
#: pooled organ cosine >= 0.99992, 33.8 -> 15.7 MB; one scale per lattice (duckn's standard
#: ``linear``) was ten times worse (token cosine >= 0.9968).
AXIS_LINEAR = "embedding.linear_along_axis"


def _quantize(t: np.ndarray):
    """``(int8 array, slope, intercept)`` per channel (last axis): each channel's range onto
    -128..127. Decoding is ``slope * q + intercept`` in float32."""
    t = np.asarray(t, np.float32)
    lo, hi = t.min(axis=0), t.max(axis=0)
    slope = ((hi - lo) / np.float32(255)).astype(np.float32)
    slope[slope == 0] = 1.0                          # a constant channel: any slope decodes it exactly
    intercept = (lo + np.float32(128) * slope).astype(np.float32)
    q = np.clip(np.rint((t - intercept) / slope), -128, 127).astype(np.int8)
    return q, slope, intercept


def _decode(q: np.ndarray, transforms, path, name) -> np.ndarray:
    """The embeddings from a lattice's stored values, or a refusal: a transform this reader does
    not know leaves the values' meaning undefined (duckn), and they are never offered as tokens."""
    if not transforms:
        return q
    if len(transforms) != 1 or transforms[0].get("name") != AXIS_LINEAR:
        raise ValueError(f"{path}: {name!r} is stored through {[t.get('name') for t in transforms]}; this reader "
                         f"decodes only {AXIS_LINEAR!r} - its tokens' values are undefined here")
    par = transforms[0]["parameters"]
    if par.get("axis") != q.ndim - 1 or len(par["slope"]) != q.shape[-1] or len(par["intercept"]) != q.shape[-1]:
        raise ValueError(f"{path}: {name!r}'s {AXIS_LINEAR} does not fit its channel axis")
    return q.astype(np.float32) * np.asarray(par["slope"], np.float32) + np.asarray(par["intercept"], np.float32)
DUCKN_VERSION = "1.0"                                 # geometry and a list axis: nothing past 1.0
SPACE = "left-posterior-superior"                     # rankfield's and haversack's world
CHUNK = 16                                            # tokens per spatial chunk edge; channels whole
_PROVENANCE_FIELDS = ("encoder", "code", "weights", "preprocessing", "license", "source")


def is_zarr_zip(path) -> bool:
    return str(path).lower().endswith(".zarr.zip")


def write_field(path, field: Field, token_dtype=np.float16) -> pathlib.Path:
    """``<name>.zarr.zip`` writes 0.2 (the duckn form); ``<name>.npz`` writes 0.1. ``token_dtype``
    int8 stores 0.2's tokens through ``AXIS_LINEAR`` (about half the size of fp16)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_zarr_zip(path):
        return _write_zarr(path, field, token_dtype)
    if np.dtype(token_dtype).kind != "f":
        raise ValueError(f"{path}: 0.1 (.npz) has no value transform - int8 tokens need a .zarr.zip")
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
        arrays = [_array_extension(root[n], path) for n in ext["group"]["members"]]
        return {"format": FORMAT, "version": ext["format_version"], "encoder": ext["provenance"]["encoder"],
                "kernels": [list(a["kernel"]) for a in arrays], "embedding": arrays,
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
    return {**{k: getattr(p, k) for k in _PROVENANCE_FIELDS}, "extra": p.extra, "input": p.input}


def _mm(v) -> list[float]:
    return [round(float(x), 6) for x in v]


def _group_id(field: Field) -> str:
    return f"{field.provenance.encoder}/{field.provenance.source}"


def _lattice_attrs(field: Field, j: int, transform: dict | None = None) -> dict:
    """One lattice array: placed by duckn core, described by the ``embedding`` extension."""
    from duckn import AxisMetadata, DucknMetadata
    from duckn.models import duckn_attrs
    e, p = field.embedding, field.provenance
    g = field.lattice_geometry(j)
    reach = e.lattice("receptive_mm", j)
    axes = [AxisMetadata(kind="space", centering="cell", unit="mm",
                         space_direction=[round(float(v), 9) for v in row],
                         thickness=None if reach is None else round(float(reach[a]), 6))
            for a, row in enumerate(g.directions)]
    axes.append(AxisMetadata(kind="list"))
    ext = {"version": EXTENSION_VERSION, "schema": SCHEMA,
           "space": {"model": p.encoder, "weights": p.weights, "layer": e.lattice("layers", j) or "",
                     "stage": e.stage},
           "metric": e.metric, "normalized": bool(e.normalized),
           "kernel": [int(k) for k in field.kernels[j]],
           "group": {"id": _group_id(field), "member": j, "members": field.lattices}}
    if e.projects_to is not None:
        ext["projects_to"] = e.projects_to
    extent = field.lattice_extent(j)
    if extent is not None:
        ext["extent"] = {"lo": list(extent[0]), "hi": list(extent[1])}
    offset = e.lattice("support_offset_mm", j)
    if offset is not None:
        ext["support"] = {"offset": _mm(offset), "unit": "mm"}
    return duckn_attrs(DucknMetadata(
        version=DUCKN_VERSION, space=SPACE, space_origin=[round(float(v), 9) for v in g.origin], axes=axes,
        intent=INTENT, value_transforms=[transform] if transform else None, extensions={EXTENSION: ext}))


def _root_attrs(field: Field, names: list[str]) -> dict:
    """The group: which arrays are the field's lattices, the file's version, and the provenance
    (the input CT's identity and grid included) until duckn's provenance extension exists."""
    from duckn import DucknMetadata
    from duckn.models import duckn_attrs
    return duckn_attrs(DucknMetadata(version=DUCKN_VERSION, intent=INTENT, extensions={EXTENSION: {
        "version": EXTENSION_VERSION, "schema": SCHEMA, "format": FORMAT, "format_version": ZARR_VERSION,
        "group": {"id": _group_id(field), "members": names},
        **({"data_box": {"lo": list(field.data_box[0]), "hi": list(field.data_box[1])}}
           if field.data_box is not None else {}),
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
            transform = None
            if np.dtype(token_dtype) == np.int8:
                q, slope, intercept = _quantize(t)
                data = np.ascontiguousarray(q.reshape(shape))
                transform = {"name": AXIS_LINEAR, "parameters": {"axis": 3, "slope": slope.tolist(),
                                                                 "intercept": intercept.tolist()}}
            elif np.dtype(token_dtype).kind == "f":
                data = np.ascontiguousarray(np.asarray(t, token_dtype).reshape(shape))   # row-major = (Z, Y, X, C)
            else:
                raise ValueError(f"token dtype {np.dtype(token_dtype)}: a field is float or int8 (with {AXIS_LINEAR})")
            name = f"lattice_{j}"
            arr = root.create_array(name, shape=shape, dtype=data.dtype,
                                    chunks=tuple(min(CHUNK, s) for s in shape[:3]) + (shape[3],),
                                    compressors=zarr.codecs.ZstdCodec(level=3),
                                    attributes=_lattice_attrs(field, j, transform))
            arr[:] = data
            names.append(name)
        root.attrs.update(_root_attrs(field, names))
        partial = path.with_name(path.name + ".partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for fp in sorted(q for q in staging.rglob("*") if q.is_file()):
                zf.write(fp, fp.relative_to(staging).as_posix())
            zf.writestr("README.md", README.read_text())   # the client's guide travels with the field
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
    exts = (root.attrs.asdict().get("duckn") or {}).get("extensions", {})
    ext = exts.get(EXTENSION)
    if not ext:
        hint = (" (a draft 0.2 written before 2026-09-22 with a 'feldglas' extension: rewrite it from its source)"
                if "feldglas" in exts else "")
        raise ValueError(f"{path}: no {EXTENSION!r} extension on the root - not a feldglas field{hint}")
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


def _embedding(arrays: list[dict], thickness: list, root_ext: dict, path) -> Embedding:
    """The field's :class:`Embedding` from its arrays - which must agree on everything that is not
    per lattice, and be the members of ONE group in the order the root lists them."""
    gid, n = root_ext["group"]["id"], len(arrays)
    for j, a in enumerate(arrays):
        g = a.get("group") or {}
        if (g.get("id"), g.get("member"), g.get("members")) != (gid, j, n):
            raise ValueError(f"{path}: lattice {j} says it is member {g.get('member')} of {g.get('members')} "
                             f"in {g.get('id')!r}; the root lists it as {j} of {n} in {gid!r}")
    shared = [(a["space"]["model"], a["space"]["weights"], a["space"]["stage"], a["metric"], a["normalized"],
               json.dumps(a.get("projects_to"), sort_keys=True)) for a in arrays]
    if len(set(shared)) != 1:
        raise ValueError(f"{path}: lattices of one field disagree about model, weights, stage or metric: {shared}")
    first = arrays[0]
    offsets = [tuple(a["support"]["offset"]) if "support" in a else None for a in arrays]
    layers = tuple(a["space"]["layer"] for a in arrays)
    return Embedding(layers=layers if any(layers) else (), stage=first["space"]["stage"],
                     metric=first["metric"], normalized=bool(first["normalized"]),
                     projects_to=first.get("projects_to"),
                     receptive_mm=tuple(thickness) if any(t is not None for t in thickness) else (),
                     support_offset_mm=tuple(offsets) if any(o is not None for o in offsets) else ())


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
    tokens, kernels, placed, described, thickness = [], [], [], [], []
    for name in ext["group"]["members"]:
        arr = root[name]
        a = _array_extension(arr, path)
        placed.append(_placement(arr, path)); kernels.append(tuple(int(k) for k in a["kernel"]))
        described.append(a)
        t = [ax.get("thickness") for ax in arr.attrs.asdict()["duckn"]["axes"][:3]]
        thickness.append(None if all(v is None for v in t) else tuple(float(v) for v in t))
        data = _decode(arr[:], arr.attrs.asdict()["duckn"].get("value_transforms"), path, name)
        tokens.append(data.reshape(-1, data.shape[-1]))
    grid = _model_grid(placed[0], kernels[0])
    box = ext.get("data_box")
    field = Field(tokens=tokens, kernels=kernels, grid=grid, provenance=Provenance(**ext["provenance"]),
                  embedding=_embedding(described, thickness, ext, path),
                  data_box=None if box is None else (tuple(box["lo"]), tuple(box["hi"])))
    for j, a in enumerate(described):                 # each lattice's extent is derived from the box, and checked
        want = field.lattice_extent(j)
        got = a.get("extent")
        if (want is None) != (got is None) or (got is not None and (tuple(got["lo"]), tuple(got["hi"])) != want):
            raise ValueError(f"{path}: lattice {j}'s extent {got} is not what the field's data box gives ({want})")
    if (field.provenance.encoder, field.provenance.weights) != (described[0]["space"]["model"],
                                                                 described[0]["space"]["weights"]):
        raise ValueError(f"{path}: the lattices' space names another model or weights than the provenance")
    for j, g in enumerate(placed):                    # every lattice must derive the same model grid
        want = field.lattice_geometry(j)
        if (tuple(want.shape) != tuple(g.shape)
                or not np.allclose(want.directions, g.directions, atol=1e-6)
                or not np.allclose(want.origin, g.origin, atol=1e-4)):
            raise ValueError(f"{path}: lattice {j} is not placed where lattice 0's model grid puts it "
                             f"(stored origin {g.origin}, derived {want.origin}) - refusing a field whose "
                             "lattices disagree about where the patient is")
    return field
