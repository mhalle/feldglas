"""Fields on a shared object store: a MANIFEST here, the blobs in provender.

Token fields are 24-37 MB each, a working cohort is hundreds of them, and the laptop this was
written on had 52 GB free (2026-09-20). So fields live on an object store (the family's R2
bucket), are pooled and swept where they live, and only working subsets come down. Storing
bytes is not this package's business: provender does it - content-addressed, written only if
absent, verified on every fetch - and writing a second, thinner copy of that here is exactly
the drift the family keeps paying for. What IS feldglas's business is the one thing a
content-addressed store cannot do: say which digest is series X. That is the manifest.

A manifest is a small JSON file, tracked in git (it names public series ids and hashes, and
carries no derived data), in the shape provender's 0.2 pointer body will have - so when the
mutable half lands, the migration is one upload:

    {"format": 1,
     "files": {"<series>.npz": {"digest": "sha256:...", "size": 24117248}},
     "body":  {"provenance": {encoder, code, weights, preprocessing, license}, "members": [...]}}

It does two jobs: the lookup, and the live set handed to provender's sweep (which refuses an
empty one - "nothing is live" and "I could not read my manifest" look the same from there).
One manifest holds ONE provenance: a field from other weights or another preprocessing is a
different object under the same series name, and mixing them in one map is how a cached result
comes to mean two things.

LICENSE. Fields derive from an encoder's weights and inherit their license (RADAR's: CC BY-NC-SA
4.0). The store must be PRIVATE, and the prefix feldglas's own: provender scopes every listing
and sweep to a prefix, so another project sharing the bucket can neither read nor collect these.

provender is imported inside the functions: it is the ``remote`` extra, and importing this
module must not require it, nor pull a store client in behind it.
"""
from __future__ import annotations

import json
import os
import pathlib

from .paths import encoder_dir

FORMAT = 1
STORE_ENV = "FELDGLAS_STORE"            # e.g. s3://<bucket>/feldglas  (the encoder's name is appended)
_SAME = ("encoder", "code", "weights", "preprocessing", "license")


def _provender():
    try:
        import provender
    except ImportError as e:
        raise ImportError("fields on an object store need provender: install feldglas[remote]") from e
    return provender


def open_blobs(encoder: str, url: str | None = None, check: bool = True):
    """provender's ``Blobs`` for one encoder's fields, under ``<url>/<encoder>/``. ``url``
    defaults to ``$FELDGLAS_STORE``; credentials come from the environment, as obstore reads
    them. ``check`` asks the store whether it honors create-if-absent before trusting it."""
    url = url or os.environ.get(STORE_ENV)
    if not url:
        raise ValueError(f"no store: pass url= or set ${STORE_ENV} (e.g. s3://bucket/feldglas)")
    pv = _provender()
    store, prefix = pv.open_store(url)
    prefix = f"{prefix}{encoder}/"
    if check:
        pv.check_store(store, prefix, updates=False)       # blobs need create-if-absent only
    return pv.Blobs(store, prefix)


class Manifest:
    """``name -> {digest, size}`` for one export set, with the provenance all its members share."""

    def __init__(self, path, files: dict | None = None, body: dict | None = None):
        self.path = pathlib.Path(path)
        self.files: dict[str, dict] = dict(files or {})
        self.body: dict = {"provenance": {}, "members": [], **(body or {})}

    @classmethod
    def load(cls, path) -> "Manifest":
        """An absent file is an EMPTY manifest (the first export creates it); an unreadable or
        unknown one is an error - never an empty manifest, which a sweep would read as
        "nothing is live"."""
        path = pathlib.Path(path)
        if not path.exists():
            return cls(path)
        doc = json.loads(path.read_text())
        if doc.get("format") != FORMAT or not isinstance(doc.get("files"), dict):
            raise ValueError(f"{path}: not a feldglas manifest of format {FORMAT}")
        return cls(path, doc["files"], doc.get("body"))

    def save(self) -> pathlib.Path:
        self.body["members"] = sorted(self.files)
        doc = {"format": FORMAT, "files": {k: self.files[k] for k in sorted(self.files)}, "body": self.body}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")     # sorted: a diff shows what changed
        tmp.replace(self.path)
        return self.path

    def add(self, name: str, blob: dict, provenance: dict) -> None:
        """Record ``name``. Refuses a member whose provenance differs from the manifest's."""
        mine = {k: provenance.get(k, "") for k in _SAME}
        have = self.body.get("provenance") or {}
        if have and any(have.get(k, "") != mine[k] for k in _SAME):
            diff = {k: (have.get(k, ""), mine[k]) for k in _SAME if have.get(k, "") != mine[k]}
            raise ValueError(f"{self.path.name} holds fields of another provenance {diff}: "
                             "use a separate manifest for a different encoder, weights or preprocessing")
        self.body["provenance"] = mine
        self.files[name] = {"digest": blob["digest"], "size": int(blob["size"])}

    def digests(self) -> set[str]:
        return {f["digest"] for f in self.files.values()}


def _provenance_of(path) -> dict:
    from .store import read_meta
    return read_meta(path)["provenance"]


def push(field_path, blobs, manifest: Manifest, name: str | None = None) -> dict:
    """Upload one field file (a no-op if those bytes are already stored) and record it. The
    caller saves the manifest - once, after a batch, so an interrupted export never leaves a
    manifest naming blobs that were not written."""
    field_path = pathlib.Path(field_path)
    provenance = _provenance_of(field_path)                 # also refuses a file that is not a field
    blob = blobs.put_file(field_path)
    manifest.add(name or field_path.name, blob, provenance)
    return blob


def pull(name: str, blobs, manifest: Manifest, dest_dir=None) -> pathlib.Path:
    """The local copy of ``name``, fetched only if it is missing or is not the bytes the
    manifest names (a 30 MB file hashes in ~0.05 s; trusting a stale or half-written copy
    costs a wrong answer)."""
    if name not in manifest.files:
        raise KeyError(f"{name!r} is not in {manifest.path.name} ({len(manifest.files)} fields)")
    want = manifest.files[name]["digest"]
    encoder = (manifest.body.get("provenance") or {}).get("encoder") or "unknown"
    dest = pathlib.Path(dest_dir) if dest_dir else encoder_dir(encoder) / "fields"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / name
    if out.exists() and _provender().digest_file(out) == want:
        return out
    if not blobs.fetch(want, out):
        raise FileNotFoundError(f"{name}: blob {want} is gone from the store, or did not match its name")
    return out


def sweep(blobs, manifest: Manifest, **kw) -> dict:
    """Delete stored blobs the manifest does not name (provender spares the young and refuses
    an empty live set). Run it from ONE place, with the manifest every exporter writes to."""
    return blobs.sweep(keep=manifest.digests(), **kw)
