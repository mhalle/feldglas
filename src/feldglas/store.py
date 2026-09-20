"""A field on disk: one ``.npz`` per series, with a version gate.

Format ``feldglas-field 0.1`` (2026-09-20): ``tokens<j>`` per lattice as written (fp16 from the
exporter), ``native_mask`` when the encoder has one, and ``meta`` - a JSON string holding the
kernels, the model grid's geometry in rankfield's form, the provenance (with the license the
arrays inherit) and the native label names. Readers refuse what they do not know, as the rest of
the family does: a field whose geometry is misread draws a finding on the wrong organ, silently.

This is deliberately the simplest thing that holds the contract. Moving the arrays into the
family's chunked store (so a read-out can be restored by the existing kernels, and a field can be
int8 per channel at ~5 MB) is EXPLORATION section 10, experiment F4 - the on-disk form changes
then, and ``Field`` does not.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from rankfield.geometry import Geometry

from .contract import Field, Provenance

FORMAT, VERSION = "feldglas-field", "0.1"
KNOWN_VERSIONS = {"0.1"}


def write_field(path, field: Field, token_dtype=np.float16) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"format": FORMAT, "version": VERSION,
            "kernels": [list(k) for k in field.kernels],
            "grid": {"shape": list(field.grid.shape), "directions": [list(r) for r in field.grid.directions],
                     "origin": list(field.grid.origin)},
            "exact_geometry": bool(field.exact_geometry),
            "provenance": {**{k: getattr(field.provenance, k) for k in
                              ("encoder", "code", "weights", "preprocessing", "license", "source")},
                           "extra": field.provenance.extra},
            "native_labels": list(field.native_labels)}
    arrays = {f"tokens{j}": np.asarray(t, token_dtype) for j, t in enumerate(field.tokens)}
    if field.native_mask is not None:
        arrays["native_mask"] = np.asarray(field.native_mask)
    tmp = path.with_name(path.name + ".tmp.npz")             # never leave half a field under the real name
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    tmp.replace(path)
    return path


def read_meta(path) -> dict:
    with np.load(path) as z:
        if "meta" not in z.files:
            raise ValueError(f"{path}: no meta block - not a feldglas field")
        meta = json.loads(str(z["meta"]))
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: format {meta.get('format')!r}, not {FORMAT!r} "
                         "(a 2026-09-20 pilot file? use adapters.radar.read_pilot_field)")
    if meta.get("version") not in KNOWN_VERSIONS:
        raise ValueError(f"{path}: {FORMAT} version {meta.get('version')!r}; this reader knows {sorted(KNOWN_VERSIONS)}")
    return meta


def read_field(path) -> Field:
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
