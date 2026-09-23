"""``feldglas``: the receiving end on the command line - `feldglas.client` with files or URLs.

    feldglas health [--haversack SERVER]
    feldglas info FIELD|REFERENCE
    feldglas vectors FIELD --labels LABELS [--structure liver ...]
    feldglas reference build -o REF.zarr.zip --normal FIELD LABELS [--normal ...] --structure liver
    feldglas score FIELD --labels LABELS --reference REF.zarr.zip --structure liver --optional liver_tumor

FIELD, LABELS and REF may each be a path or an ``http(s)://`` URL. LABELS is a ``.seg.nrrd`` -
haversack's or 3D Slicer's; from a haversack server, name it by ``--haversack SERVER --series
SOURCE:ID --task TASK`` instead (``<server>/v1/<source>/<id>/<task>/labels.seg.nrrd``), a local
``haversack serve`` included. A token (``--token`` or ``$FELDGLAS_TOKEN``) goes only to its URL's
own origin.

Most callers are agents (2026-09-23), so every command takes ``--json`` and then prints ONE JSON
object on stdout - ``{"ok": true, "warnings": [...], ...}``, or on failure ``{"ok": false,
"error": "...", "exit_code": 1}`` - and never a traceback; progress goes to stderr. Exit codes:
0 done, 1 refused or failed (the message names why), 2 a usage error. Whatever a result cannot
vouch for - labels from another scan, a reference used outside its terms, a field that was one of
the reference's own normals - is refused or named in ``warnings``, never passed over in silence.
"""
from __future__ import annotations

import functools
import json
import math
import pathlib
import sys

import click
import numpy as np

EXIT_CODES = {0: "done (health: nothing wrong)", 1: "refused or failed; the message names why (health: a problem found)",
              2: "usage: a missing, conflicting or out-of-range option"}
# what each dependency makes possible: a missing one is reported by what it costs
_FEATURES = {"zarr": "open fields and references", "obstore": "files by http(s) URL", "click": "this command",
             "scipy": "regions (erosion, depth)", "numpy": "everything", "rankfield": "feldglas's internals",
             "duckn": "writing fields (feldglas.store) - not needed to read"}
FLAGGED_FRACTION_WARNING = 0.05


# -- output -----------------------------------------------------------------------------------------
def _clean(o):
    """JSON that is JSON: NaN and infinities become null, arrays lists."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def _out(data, as_json, text):
    data = {"ok": True, "warnings": [], **data}
    if as_json:
        click.echo(json.dumps(_clean(data), indent=1, allow_nan=False))
    else:
        click.echo(text)
        for w in data["warnings"]:
            click.echo(f"warning: {w}", err=True)


def _one_line(e: BaseException) -> str:
    if isinstance(e, click.ClickException):
        return e.message
    if isinstance(e, KeyError) and e.args:
        return str(e.args[0])
    m = str(e).splitlines()[0] if str(e) else ""
    return m if isinstance(e, (ValueError, OSError)) and m else f"{type(e).__name__}: {m}"


def agentic(fn):
    """Every failure is one line - and with --json, one JSON object on stdout - never a traceback."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (click.UsageError, click.exceptions.Exit, click.exceptions.Abort):
            raise
        except Exception as e:                                           # noqa: BLE001 - the point
            msg = _one_line(e)
            if kwargs.get("as_json"):
                click.echo(json.dumps({"ok": False, "error": msg, "exit_code": 1}))
                sys.exit(1)
            raise click.ClickException(msg) from None
    return wrapper


def _progress(msg):
    click.echo(f"feldglas: {msg}", err=True)


# -- inputs -----------------------------------------------------------------------------------------
def _labels_path(labels, haversack, series, task):
    if labels and (haversack or series or task):
        raise click.UsageError("give --labels, or --haversack/--series/--task, not both")
    if labels:
        return labels
    if not (haversack and series and task):
        raise click.UsageError("labels are needed: --labels PATH|URL, or --haversack SERVER --series SOURCE:ID --task TASK")
    source, _, ident = series.partition(":")
    if not ident or not source:
        raise click.UsageError(f"--series {series!r}: SOURCE:ID, e.g. idc:<crdc_series_uuid>")
    return f"{haversack.rstrip('/')}/v1/{source}/{ident}/{task}/labels.seg.nrrd"


def labels_options(f):
    for opt in reversed((
            click.option("--labels", help="the scan's segmentation: a .seg.nrrd path or URL"),
            click.option("--haversack", metavar="SERVER", help="a haversack server (https://..., or http://localhost:8790)"),
            click.option("--series", metavar="SOURCE:ID", help="with --haversack: the input, e.g. idc:<uuid>"),
            click.option("--task", help="with --haversack: the segmentation task, e.g. ts.v2:total"),
            click.option("--names", "names_json", help="for a plain labelmap .nrrd with no segment table: JSON {name: value}"),
            click.option("--skip-grid-check", is_flag=True,
                         help="use labels whose grid is not the field's recorded CT grid (they gate other anatomy)"))):
        f = opt(f)
    return f


def _read_labels(ctx, labels, haversack, series, task, names_json):
    from . import client as fc
    names = None
    if names_json:
        try:
            names = {str(k): int(v) for k, v in json.loads(names_json).items()}
        except (ValueError, AttributeError, TypeError) as e:
            raise click.UsageError(f"--names must be a JSON object {{name: label value}}: {e}")
    return fc.read_seg_nrrd(_labels_path(labels, haversack, series, task), names, token=ctx.obj["token"])


def _check_grid(f, lm, where, skip, warnings):
    """Labels must be of the field's own CT: refused otherwise (or warned of under --skip-grid-check)."""
    from . import client as fc
    why = fc.grid_mismatch(f, lm)
    if why and not skip:
        raise ValueError(f"{where}: {why} - these labels are not of the scan this field was computed from "
                         "(--skip-grid-check to use them anyway)")
    if why:
        warnings.append(f"{where}: {why} (used anyway: --skip-grid-check)")
    elif not f.input_grid:
        warnings.append(f"{where}: the field records no input CT grid - nothing checks that these labels are of its scan")


class Region:
    """A region of named structures on the labels' grid - optional ones when present - eroded by
    ``erode_mm``, with the depth of every point inside the uneroded region. The distance transform
    runs on the region's bounding box only (a whole CT took 13 of a score's 17 s), padded with
    outside so an organ cut by the scan's edge erodes from that edge too."""

    def __init__(self, labels, structures, optional, erode_mm):
        from scipy.ndimage import distance_transform_edt
        from . import client as fc
        if erode_mm < 0:
            raise click.UsageError("--erode is a distance in mm: 0 or more")
        for pat in optional:
            if pat.strip("*") == "":
                raise click.UsageError(f"--optional {pat!r} would add every structure")
        self.asked = list(optional)
        self.matched = sorted({n for n in labels.names for pat in optional
                               if (n.startswith(pat[:-1]) if pat.endswith("*") else n == pat)})
        whole = fc.structure_mask(labels, *structures, optional=optional)
        v = whole.values
        self.affine = whole.affine_lps
        self.spacing = np.linalg.norm(self.affine[:3, :3], axis=0)
        nz = np.nonzero(v)
        if not len(nz[0]):
            raise ValueError(f"the region {list(structures) + self.matched} is empty in these labels")
        margin = np.ceil(erode_mm / self.spacing).astype(int) + 2
        self.lo = np.maximum(np.array([a.min() for a in nz]) - margin, 0)
        hi = np.minimum(np.array([a.max() for a in nz]) + margin + 1, v.shape)
        box = tuple(slice(a, b) for a, b in zip(self.lo, hi))
        sub = np.pad(v[box], 1)                                          # outside, beyond the array's edge
        self.depth = distance_transform_edt(sub, sampling=self.spacing)[1:-1, 1:-1, 1:-1]
        eroded = np.zeros(v.shape, bool)
        eroded[box] = self.depth > erode_mm
        if not eroded.any():
            raise ValueError(f"the region {list(structures) + self.matched} is empty after a {erode_mm:g} mm erosion")
        self.mask = fc.Mask(eroded, self.affine)
        self.voxels = int(eroded.sum())

    def depth_at(self, points) -> np.ndarray:
        """mm inside the UNERODED region at world points (0 outside it)."""
        inv = np.linalg.inv(self.affine)
        ijk = np.floor(np.atleast_2d(points) @ inv[:3, :3].T + inv[:3, 3] + 0.5).astype(int) - self.lo
        ok = np.all((ijk >= 0) & (ijk < self.depth.shape), axis=1)
        out = np.zeros(len(ijk))
        out[ok] = self.depth[tuple(ijk[ok].T)]
        return out


# -- the command ------------------------------------------------------------------------------------
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--token", envvar="FELDGLAS_TOKEN", default=None, help="bearer token for URLs (only ever to their own origin)")
@click.pass_context
def main(ctx, token):
    """Read embedding fields: describe them, pool organs, and find what is unlike normal tissue.
    Start with `feldglas health`; every command takes --json (one JSON object on stdout)."""
    ctx.obj = {"token": token}


def _version(dist):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _importable(module):
    import importlib
    try:
        importlib.import_module(module)
        return None
    except Exception as e:                                               # noqa: BLE001 - reported, not raised
        return f"{type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}"


@main.command()
@click.option("--haversack", metavar="SERVER", help="also check a haversack server (its /v1/health; 5 s, no retries)")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
@agentic
def health(ctx, haversack, as_json):
    """Is this installation ready, and what can it read? Exit 0 when nothing is wrong, 1 otherwise -
    for an agent to check before it drives anything else. Never prints a token."""
    import os, platform, time
    from .paths import cache_dir
    problems = []
    deps = {}
    for d, what in _FEATURES.items():
        err = _importable(d)
        deps[d] = {"version": _version(d), "importable": err is None, "enables": what}
        if err and d != "duckn":
            problems.append(f"{d} cannot be imported ({err}): no {what} - pip install feldglas[client]")
    from . import client as fc
    cache = cache_dir()
    http = cache / "http"
    fetched = [q for q in http.glob("*") if q.is_file() and not q.name.endswith((".json", ".partial", ".lock"))] \
        if http.is_dir() else []
    probe = cache if cache.exists() else next((p for p in cache.parents if p.exists()), pathlib.Path("/"))
    writable = os.access(probe, os.W_OK)
    if not writable:
        problems.append(f"the cache {cache} is not writable: URLs cannot be fetched (set $FELDGLAS_CACHE)")
    token = ctx.obj["token"]
    server = None
    if haversack:
        t0 = time.time()
        try:
            import datetime
            from obstore import get
            from obstore.store import HTTPStore
            from .fetch import origin
            scheme, host, port, path = origin(haversack.rstrip("/") + "/v1/health")
            netloc = f"[{host}]" if ":" in host else host
            st = HTTPStore.from_url(f"{scheme}://{netloc}" + (f":{port}" if port else ""),
                                    client_options={"allow_http": scheme == "http", "timeout": "5s", "connect_timeout": "3s",
                                                    "user_agent": "feldglas"},
                                    retry_config={"max_retries": 0, "retry_timeout": datetime.timedelta(seconds=5),
                                                  "backoff": {"init_backoff": datetime.timedelta(seconds=1),
                                                              "max_backoff": datetime.timedelta(seconds=1), "base": 2}})
            body = json.loads(bytes(get(st, path).bytes()))
            server = {"url": haversack, "reachable": True, "seconds": round(time.time() - t0, 3), "health": body}
        except Exception as e:                                           # noqa: BLE001 - reported
            server = {"url": haversack, "reachable": False, "error": _one_line(e)[:200]}
            problems.append(f"the haversack server {haversack} did not answer /v1/health within 5 s ({server['error'][:80]})")
    data = {
        "ok": not problems, "problems": problems,
        "feldglas": _version("feldglas"), "python": platform.python_version(),
        "dependencies": deps,
        "reads": {"field": {"format": fc.FORMAT, "versions": sorted(fc.FORMAT_VERSIONS),
                            "extension": f"{fc.EXTENSION} {fc.EXTENSION_VERSION}", "value_transforms": [fc.AXIS_LINEAR]},
                  "reference": {"format": fc.Reference.FORMAT, "version": fc.Reference.FORMAT_VERSION},
                  "labels": {"seg.nrrd": ["labelmap style (one layer)", "layered style (3D Slicer)", "LPS", "RAS"],
                             "plain labelmap .nrrd": "with --names"},
                  "urls": ["http", "https"] if deps["obstore"]["importable"] else []},
        "cache": {"path": str(cache), "exists": cache.exists(), "writable": writable, "fetched_files": len(fetched),
                  "fetched_bytes": sum(q.stat().st_size for q in fetched), "env": "FELDGLAS_CACHE"},
        "token": {"set": bool(token), "from": "--token or $FELDGLAS_TOKEN", "sent_to": "each URL's own origin only"},
        "server": server,
        "commands": sorted(main.commands), "exit_codes": EXIT_CODES,
    }
    text = [f"feldglas {data['feldglas']} (python {data['python']}): {'ok' if data['ok'] else 'PROBLEMS'}"]
    text += [f"  problem: {p}" for p in problems]
    text += [f"  {d:10} {v['version'] or '-':10} {'ok' if v['importable'] else 'NOT IMPORTABLE':15} {v['enables']}"
             for d, v in deps.items()]
    text.append(f"  reads fields {fc.FORMAT} {sorted(fc.FORMAT_VERSIONS)}, references {fc.Reference.FORMAT} "
                f"{fc.Reference.FORMAT_VERSION}, .seg.nrrd (labelmap and layered, LPS/RAS), URLs {data['reads']['urls']}")
    text.append(f"  cache {cache} ({'writable' if writable else 'NOT writable'}; {len(fetched)} fetched files, "
                f"{data['cache']['fetched_bytes'] / 1e6:.1f} MB); token {'set' if token else 'not set'}")
    if server:
        text.append(f"  server {haversack}: " + (f"up in {server['seconds']} s, version {server['health'].get('version')}"
                                                   if server["reachable"] else f"NOT reachable - {server['error']}"))
    if as_json:
        click.echo(json.dumps(_clean({"warnings": [], **data}), indent=1, allow_nan=False))
    else:
        click.echo("\n".join(text))
    if problems:
        sys.exit(1)


def _kind(path, token):
    """'field' or 'reference', by the file's own root."""
    import zarr
    from .fetch import local
    from . import client as fc
    p = local(path, token, expect="zip")
    if not p.is_file():
        raise FileNotFoundError(f"{path}: no such file")
    try:
        root = zarr.open_group(store=zarr.storage.ZipStore(str(p), mode="r"), mode="r")
    except Exception:                                                     # noqa: BLE001
        raise ValueError(f"{path}: not a zarr zip - neither a field nor a reference") from None
    fmt = (((root.attrs.asdict().get("duckn") or {}).get("extensions") or {}).get(fc.EXTENSION) or {}).get("format")
    return {fc.FORMAT: "field", fc.Reference.FORMAT: "reference"}.get(fmt, fmt)


def _threshold_warning(per, warnings):
    maxes = sorted((p["max"] for p in per), reverse=True)
    if len(maxes) > 1 and maxes[1] > 0 and maxes[0] > 2 * maxes[1]:
        top = max(per, key=lambda p: p["max"])
        warnings.append(f"one normal sets the threshold: {top['normal']} reaches {top['max']:.3f} at LPS "
                        f"{top.get('max_at_lps')}, the next normal {maxes[1]:.3f} - check that scan (and that it is normal)")


@main.command()
@click.argument("path")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
@agentic
def info(ctx, path, as_json):
    """What a field or a reference is: encoder, license, lattices, extents, the input CT's grid - or a
    reference's normals, how its threshold was set, and its terms (structures, erosion, k)."""
    from . import client as fc
    from .fetch import local
    kind = _kind(path, ctx.obj["token"])
    size = local(path, ctx.obj["token"], expect="zip").stat().st_size
    if kind == "reference":
        r = fc.Reference.load(path, ctx.obj["token"])
        per, warnings = r.per_normal(), []
        _threshold_warning(per, warnings)
        data = {"kind": "reference", "path": path, "bytes": size, "key": list(r.key), "threshold": r.threshold,
                "k": r.k, "min_occupancy": r.min_occupancy, "drop_boundary": r.drop_boundary, "lattice": r.lattice,
                "tokens": len(r.tokens), "per_normal": per,
                "terms": {k: r.meta.get(k) for k in ("structures", "erode_mm", "note")},
                "license": r.meta.get("license"), "normals": r.meta.get("normals"), "warnings": warnings}
        text = [f"{path}: a reference ({size / 1e6:.1f} MB) for {r.meta.get('structures')} eroded {r.meta.get('erode_mm')} mm, "
                f"layer {r.key[2]}, k {r.k}; threshold {r.threshold:.4f}; license {r.meta.get('license')}"]
        text += [f"  normal {p['normal']}: {p['tokens']} tokens, held-out distance p50 {p['p50']:.3f} p99 {p['p99']:.3f} "
                 f"max {p['max']:.3f} at LPS {p.get('max_at_lps')}" for p in per]
        _out(data, as_json, "\n".join(text))
        return
    if kind != "field":
        raise ValueError(f"{path}: a {kind!r} file - neither a field nor a reference")
    f = fc.open_field(path, ctx.obj["token"])
    rows = []
    for l in f.lattices:
        c = l.centers[l.in_extent]
        rows.append({"name": l.name, "shape": l.shape, "step_mm": np.round(np.linalg.norm(l.steps, axis=1), 3).tolist(),
                     "kernel": l.kernel, "extent": l.extent, "in_extent": int(l.in_extent.sum()), "tokens": len(l.tokens),
                     "channels": int(l.tokens.shape[1]), "reach_mm": l.thickness, "key": list(l.key()),
                     "extent_lps_mm": [np.round(c.min(0), 1).tolist(), np.round(c.max(0), 1).tolist()] if len(c) else None,
                     "finest": l is f.fine})
    warnings = [] if f.input_grid else ["the field records no input CT grid: labels given with it cannot be checked"]
    data = {"kind": "field", "path": path, "bytes": size, "encoder": f.encoder, "license": f.license, "lattices": rows,
            "data_box": f.data_box, "input_grid": f.input_grid, "source": f.provenance.get("source"), "warnings": warnings}
    text = [f"{path} ({size / 1e6:.1f} MB)", f"  encoder {f.encoder}   license {f.license}   source {data['source']}"]
    for r in rows:
        text.append(f"  {r['name']:10} {tuple(r['shape'])} x {r['channels']}   step {r['step_mm']} mm   kernel {r['kernel']}   "
                    f"in extent {r['in_extent']} of {r['tokens']}   reach {r['reach_mm']}   layer {r['key'][2]}"
                    + ("   (finest: used by vectors and score)" if r["finest"] else ""))
    g = f.input_grid
    text.append(f"  input CT grid: {g['shape']}, origin {np.round(g['origin'], 1).tolist()} (labels are checked against it)"
                if g else "  input CT grid: not recorded")
    _out(data, as_json, "\n".join(text))


@main.command()
@click.argument("field")
@labels_options
@click.option("--structure", "structures", multiple=True, help="only these structures (repeatable); default: all")
@click.option("--top", default=3, show_default=True, type=click.IntRange(min=1), help="nearest structures to list for each")
@click.option("--min-tokens", default=5, show_default=True, type=click.IntRange(min=1))
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
@agentic
def vectors(ctx, field, labels, haversack, series, task, names_json, skip_grid_check, structures, top, min_tokens, as_json):
    """One vector per structure (finest lattice, shared mean off), and each one's nearest structures."""
    from . import client as fc
    from .labels import no_structure
    f = fc.open_field(field, ctx.obj["token"])
    lm = _read_labels(ctx, labels, haversack, series, task, names_json)
    warnings = []
    _check_grid(f, lm, "labels", skip_grid_check, warnings)
    for s in structures:
        if s not in lm.names:
            raise KeyError(no_structure(s, lm.names))
    counts = {}
    vec = fc.organ_vectors(f, lm, min_tokens=min_tokens, counts=counts)
    names = sorted(vec)
    skipped = {n: counts.get(n, 0) for n in (structures or lm.names) if n not in vec}
    for n in structures:
        if n in skipped:
            warnings.append(f"{n}: {skipped[n]} token(s), below --min-tokens {min_tokens} (or no direction of its own)")
    M = np.array([vec[n] for n in names]) if names else np.zeros((0, 1))
    S = M @ M.T
    shown = [n for n in names if not structures or n in structures]
    out = {n: {"tokens": counts[n], "nearest": [{"structure": names[j], "cosine": round(float(S[i, j]), 4)}
                                                 for j in np.argsort(-S[i]) if j != i][:top]}
           for i, n in enumerate(names) if n in shown}
    lat = f.fine
    _out({"field": field, "lattice": lat.name, "layer": lat.key()[2], "structures": out, "skipped": skipped,
          "warnings": warnings}, as_json,
         "\n".join([f"{len(names)} structures with at least {min_tokens} tokens on {lat.name} ({lat.key()[2]}); "
                    "nearest by cosine (shared mean off):"]
                   + [f"  {n:28} ({out[n]['tokens']:4} tokens)  " + ", ".join(f"{m['structure']} {m['cosine']:.2f}"
                                                                           for m in out[n]["nearest"]) for n in shown]))


@main.group()
def reference():
    """Normal tissue from a few normal scans of the same protocol."""


@reference.command("build")
@click.option("-o", "--out", required=True, help="the reference to write (<name>.zarr.zip)")
@click.option("--normal", "normals", nargs=2, multiple=True, required=True, metavar="FIELD LABELS",
              help="a normal scan's field and its .seg.nrrd (paths or URLs); repeat for each (at least two)")
@click.option("--structure", "structures", multiple=True, required=True, help="the normal tissue: an organ (repeatable)")
@click.option("--erode", default=15.0, show_default=True, type=click.FloatRange(min=0),
              help="mm to erode the region: interior tokens only")
@click.option("--k", default=5, show_default=True, type=click.IntRange(min=1), help="nearest normal tokens a score averages over")
@click.option("--note", default="", help="what the builder wants recorded, e.g. the protocol")
@click.option("--skip-grid-check", is_flag=True, help="accept labels whose grid is not their field's recorded CT grid")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
@agentic
def reference_build(ctx, out, normals, structures, erode, k, note, skip_grid_check, as_json):
    """Build a normal-tissue reference: its threshold is the largest distance a held-out normal token
    reached, with each normal scored against the others."""
    from . import client as fc
    if not str(out).endswith(".zarr.zip"):
        raise click.UsageError(f"-o {out}: a reference is written as <name>.zarr.zip")
    if len(normals) < 2:
        raise click.UsageError("at least two --normal FIELD LABELS pairs: the threshold is set by leaving one out")
    pairs, warnings = [], []
    for i, (field, labels) in enumerate(normals):
        _progress(f"normal {i + 1}/{len(normals)}: {field}")
        f = fc.open_field(field, ctx.obj["token"])
        lm = fc.read_seg_nrrd(labels, token=ctx.obj["token"])
        _check_grid(f, lm, f"normal {i + 1} ({labels})", skip_grid_check, warnings)
        pairs.append((f, Region(lm, structures, (), erode).mask))
    _progress("scoring each normal against the others")
    ref = fc.Reference.build(pairs, k=k, meta={"structures": list(structures), "erode_mm": erode, "note": note,
                                               "labels": [l for _, l in normals]})
    ref.save(out)
    per = ref.per_normal()
    _threshold_warning(per, warnings)
    _out({"reference": out, "threshold": ref.threshold, "per_normal": per, "key": list(ref.key),
          "terms": {"structures": list(structures), "erode_mm": erode, "k": k}, "license": ref.meta["license"],
          "warnings": warnings}, as_json,
         "\n".join([f"{out}: {len(ref.tokens)} normal tokens, threshold {ref.threshold:.4f} ({list(structures)} eroded "
                    f"{erode:g} mm, k {k}; layer {ref.key[2]}); license {ref.meta['license']}"]
                   + [f"  normal {p['normal']}: {p['tokens']} tokens, held-out p50 {p['p50']:.3f} p99 {p['p99']:.3f} "
                      f"max {p['max']:.3f} at LPS {p.get('max_at_lps')}" for p in per]))


@main.command()
@click.argument("field")
@labels_options
@click.option("--reference", "ref_path", required=True, help="a reference built by `feldglas reference build`")
@click.option("--structure", "structures", multiple=True, required=True, help="the organ to test (repeatable)")
@click.option("--optional", "optional", multiple=True,
              help="lesion segments to union in when present (repeatable; a trailing * is a prefix)")
@click.option("--erode", type=click.FloatRange(min=0), default=None, help="mm to erode the region (default: as the reference was)")
@click.option("--radius", default=15.0, show_default=True, type=click.FloatRange(min=0, min_open=True),
              help="mm within which flagged tokens join one site")
@click.option("--allow-other-structure", is_flag=True, help="score structures the reference was not built for")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
@agentic
def score(ctx, field, labels, haversack, series, task, names_json, skip_grid_check, ref_path, structures, optional,
          erode, radius, allow_other_structure, as_json):
    """Score a region against a reference: flagged sites, strongest first, each with its depth inside
    the organ. A site is a place to look, not a finding."""
    from . import client as fc
    ref = fc.Reference.load(ref_path, ctx.obj["token"])
    warnings = []
    built_for = ref.meta.get("structures") or []
    if built_for and set(structures) != set(built_for) and not allow_other_structure:
        raise ValueError(f"the reference is for {built_for} and this asks for {list(structures)} - build a reference for "
                         "that tissue (or --allow-other-structure)")
    ref_erode = float(ref.meta.get("erode_mm", 0.0))
    erode = ref_erode if erode is None else erode
    if erode < ref_erode:
        warnings.append(f"eroded {erode:g} mm, less than the reference's {ref_erode:g} mm: tokens nearer the surface than "
                        "any normal token will look unlike normal")
    f = fc.open_field(field, ctx.obj["token"])
    lm = _read_labels(ctx, labels, haversack, series, task, names_json)
    _check_grid(f, lm, "labels", skip_grid_check, warnings)
    if f.provenance.get("source") and f.provenance.get("source") in (ref.meta.get("sources") or []):
        warnings.append(f"{f.provenance['source']} is one of the reference's own normals: its tokens are in the reference, "
                        "so this score is in-sample and will look normal")
    region = Region(lm, structures, optional, erode)
    s = ref.score(f, region.mask, radius=radius)
    lat = f.fine if ref.lattice is None else f.lattices[ref.lattice]
    frac = float(s.flagged.mean())
    if frac > FLAGGED_FRACTION_WARNING:
        warnings.append(f"{frac:.0%} of the region is flagged - more than a lesion would be: suspect a protocol, scanner or "
                        "phase different from the reference's normals before suspecting disease")
    sites = []
    for x in s.sites:
        c = s.centers[x.members]
        sites.append({"center_lps": np.round(x.center, 1).tolist(), "tokens": x.tokens, "peak": round(x.peak, 4),
                      "depth_mm": round(float(region.depth_at(x.center)[0]), 1),
                      "bbox_lps": [np.round(c.min(0), 1).tolist(), np.round(c.max(0), 1).tolist()]})
    q = np.percentile(s.distance, [50, 90, 99])
    data = {"field": field, "reference": ref_path, "lattice": lat.name, "layer": lat.key()[2], "threshold": s.threshold,
            "scored": len(s.distance), "flagged": int(s.flagged.sum()), "flagged_fraction": round(frac, 4),
            "distance": {"p50": q[0], "p90": q[1], "p99": q[2], "max": float(s.distance.max())},
            "region": {"structures": list(structures), "optional_asked": region.asked, "optional_matched": region.matched,
                       "erode_mm": erode, "voxels": region.voxels}, "sites": sites, "warnings": warnings}
    text = [f"{len(s.distance)} tokens of {list(structures) + region.matched} (eroded {erode:g} mm) scored on {lat.name} "
            f"({lat.key()[2]}); threshold {s.threshold:.4f}; {int(s.flagged.sum())} flagged in {len(sites)} site(s)"]
    text += [f"  site {i + 1}: LPS {x['center_lps']}  {x['tokens']} token(s)  peak {x['peak']:.3f}  {x['depth_mm']:.0f} mm deep"
             for i, x in enumerate(sites)]
    _out(data, as_json, "\n".join(text))


if __name__ == "__main__":
    sys.exit(main())
