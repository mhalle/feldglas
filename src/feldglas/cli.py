"""``feldglas``: the receiving end on the command line - `feldglas.client` with files or URLs.

    feldglas info FIELD
    feldglas vectors FIELD --labels LABELS [--structure liver ...]
    feldglas reference build -o REF.zarr.zip --normal FIELD LABELS [--normal ...] --structure liver
    feldglas score FIELD --labels LABELS --reference REF.zarr.zip --structure liver --optional liver_tumor

FIELD, LABELS and REF may each be a path or an ``http(s)://`` URL. LABELS is a ``.seg.nrrd`` -
haversack's or 3D Slicer's; from a haversack server, name it by ``--haversack SERVER --series
SOURCE:ID --task TASK`` instead (``<server>/v1/<source>/<id>/<task>/labels.seg.nrrd``), a local
``haversack serve`` included. A token (``--token`` or ``$FELDGLAS_TOKEN``) goes only to its URL's
own origin. Every command takes ``--json``.
"""
from __future__ import annotations

import json
import pathlib
import sys

import click
import numpy as np


def _labels_path(labels, haversack, series, task):
    if labels and (haversack or series or task):
        raise click.UsageError("give --labels, or --haversack/--series/--task, not both")
    if labels:
        return labels
    if not (haversack and series and task):
        raise click.UsageError("labels are needed: --labels PATH|URL, or --haversack SERVER --series SOURCE:ID --task TASK")
    source, _, ident = series.partition(":")
    if not ident:
        raise click.UsageError(f"--series {series!r}: SOURCE:ID, e.g. idc:<crdc_series_uuid>")
    return f"{haversack.rstrip('/')}/v1/{source}/{ident}/{task}/labels.seg.nrrd"


def labels_options(f):
    for opt in reversed((
            click.option("--labels", help="the scan's segmentation: a .seg.nrrd path or URL"),
            click.option("--haversack", metavar="SERVER", help="a haversack server (https://..., or http://localhost:8790)"),
            click.option("--series", metavar="SOURCE:ID", help="with --haversack: the input, e.g. idc:<uuid>"),
            click.option("--task", help="with --haversack: the segmentation task, e.g. ts.v2:total"))):
        f = opt(f)
    return f


def _region(labels, structures, optional, erode_mm):
    """The union of named structures (optional ones when present), eroded by ``erode_mm``."""
    from . import client as fc
    m = fc.structure_mask(labels, *structures, optional=optional)
    if erode_mm and erode_mm > 0:
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            raise click.ClickException("--erode needs scipy: install feldglas[client]") from None
        spacing = np.linalg.norm(m.affine_lps[:3, :3], axis=0)
        m = fc.Mask(distance_transform_edt(m.values, sampling=spacing) > erode_mm, m.affine_lps)
    if not m.values.any():
        raise click.ClickException(f"the region {list(structures) + list(optional)} is empty"
                                   + (f" after a {erode_mm:g} mm erosion" if erode_mm else ""))
    return m


def _out(data, as_json, text):
    if as_json:
        click.echo(json.dumps(data, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    else:
        click.echo(text)


class _Group(click.Group):
    """A refusal is a message, not a traceback: every error feldglas raises names its problem, so the
    command line prints that one line and exits 1."""

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.exceptions.Exit, click.exceptions.Abort):
            raise
        except (ValueError, KeyError, FileNotFoundError, OSError) as e:
            raise click.ClickException(str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)) from None


@click.group(cls=_Group, context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--token", envvar="FELDGLAS_TOKEN", default=None, help="bearer token for URLs (only ever to their own origin)")
@click.pass_context
def main(ctx, token):
    """Read embedding fields: describe them, pool organs, and find what is unlike normal tissue."""
    ctx.obj = {"token": token}


EXIT_CODES = {0: "success (health: nothing wrong)", 1: "refused or failed; the message names why (health: a problem found)",
              2: "usage: a missing or conflicting option"}
# what each optional dependency makes possible, so a missing one is reported by what it costs
_FEATURES = {"zarr": "open fields and references", "obstore": "files by http(s) URL", "click": "this command",
             "scipy": "--erode", "duckn": "writing fields (feldglas.store)", "rankfield": "feldglas's internals"}


def _version(dist):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


@main.command()
@click.option("--haversack", metavar="SERVER", help="also check a haversack server (its /v1/health)")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def health(ctx, haversack, as_json):
    """Is this installation ready, and what can it read? Exit 0 when nothing is wrong, 1 otherwise -
    for an agent to check before it drives anything else. Never prints a token."""
    import os, platform, time
    from . import client as fc
    from .paths import cache_dir
    problems = []
    deps = {d: _version(d) for d in _FEATURES}
    deps["numpy"] = _version("numpy")
    for d, what in _FEATURES.items():
        if deps[d] is None and d not in ("duckn",):
            problems.append(f"{d} is not installed: no {what} (pip install feldglas[client])")
    cache = cache_dir()
    http = cache / "http"
    fetched = [q for q in http.glob("*") if q.is_file() and not q.name.endswith((".json", ".partial"))] if http.is_dir() else []
    writable = os.access(cache if cache.exists() else cache.parent if cache.parent.exists() else pathlib.Path.home(), os.W_OK)
    if not writable:
        problems.append(f"the cache {cache} is not writable: URLs cannot be fetched (set $FELDGLAS_CACHE)")
    token = ctx.obj["token"]
    server = None
    if haversack:
        from obstore import get
        from obstore.store import HTTPStore
        import urllib.parse
        u = urllib.parse.urlsplit(haversack.rstrip("/"))
        t0 = time.time()
        try:
            st = HTTPStore.from_url(f"{u.scheme}://{u.netloc}", client_options={"allow_http": u.scheme == "http",
                                                                               "timeout": "30s", "user_agent": "feldglas"})
            body = json.loads(bytes(get(st, (u.path.strip("/") + "/v1/health").lstrip("/")).bytes()))
            server = {"url": haversack, "reachable": True, "seconds": round(time.time() - t0, 3), "health": body}
        except Exception as e:
            server = {"url": haversack, "reachable": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            problems.append(f"the haversack server {haversack} did not answer /v1/health ({type(e).__name__})")
    from .labels import read_seg_nrrd  # noqa: F401 - importable is part of healthy
    data = {
        "ok": not problems, "problems": problems,
        "feldglas": _version("feldglas"), "python": platform.python_version(),
        "dependencies": {d: {"version": v, "enables": _FEATURES.get(d, "arrays")} for d, v in deps.items()},
        "reads": {"field": {"format": fc.FORMAT, "versions": sorted(fc.FORMAT_VERSIONS),
                            "extension": f"{fc.EXTENSION} {fc.EXTENSION_VERSION}", "value_transforms": [fc.AXIS_LINEAR]},
                  "reference": {"format": fc.Reference.FORMAT, "version": fc.Reference.FORMAT_VERSION},
                  "labels": {"seg.nrrd": ["labelmap style (one layer)", "layered style (3D Slicer)", "LPS", "RAS"],
                             "plain labelmap .nrrd": "with names= / --names"},
                  "urls": ["http", "https"] if deps["obstore"] else []},
        "cache": {"path": str(cache), "exists": cache.exists(), "writable": writable, "fetched_files": len(fetched),
                  "fetched_bytes": sum(q.stat().st_size for q in fetched), "env": "FELDGLAS_CACHE"},
        "token": {"set": bool(token), "from": "--token or $FELDGLAS_TOKEN", "sent_to": "each URL's own origin only"},
        "server": server,
        "commands": sorted(main.commands), "exit_codes": EXIT_CODES,
    }
    text = [f"feldglas {data['feldglas']} (python {data['python']}): {'ok' if data['ok'] else 'PROBLEMS'}"]
    text += [f"  problem: {p}" for p in problems]
    text += [f"  {d:10} {v['version'] or 'MISSING':10} {v['enables']}" for d, v in data["dependencies"].items()]
    text.append(f"  reads fields {fc.FORMAT} {sorted(fc.FORMAT_VERSIONS)}, references {fc.Reference.FORMAT} "
                f"{fc.Reference.FORMAT_VERSION}, .seg.nrrd (labelmap and layered, LPS/RAS), URLs {data['reads']['urls']}")
    text.append(f"  cache {cache} ({'writable' if writable else 'NOT writable'}; {len(fetched)} fetched files, "
                f"{data['cache']['fetched_bytes'] / 1e6:.1f} MB); token {'set' if token else 'not set'}")
    if server:
        text.append(f"  server {haversack}: " + (f"up in {server['seconds']} s, version {server['health'].get('version')}"
                                                   if server["reachable"] else f"NOT reachable - {server['error']}"))
    _out(data, as_json, "\n".join(text))
    if problems:
        ctx.exit(1)


@main.command()
@click.argument("field")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def info(ctx, field, as_json):
    """What a field is: encoder, license, lattices, extents, the input CT's grid."""
    from . import client as fc
    f = fc.open_field(field, ctx.obj["token"])
    rows = [{"name": l.name, "shape": l.shape, "step_mm": np.round(np.linalg.norm(l.steps, axis=1), 3).tolist(),
             "kernel": l.kernel, "extent": l.extent, "in_extent": int(l.in_extent.sum()), "tokens": len(l.tokens),
             "channels": int(l.tokens.shape[1]), "reach_mm": l.thickness, "key": list(l.key())} for l in f.lattices]
    data = {"field": field, "encoder": f.encoder, "license": f.license, "lattices": rows,
            "data_box": f.data_box, "input_grid": f.input_grid, "source": f.provenance.get("source")}
    text = [f"{field}", f"  encoder {f.encoder}   license {f.license}   source {data['source']}"]
    for r in rows:
        text.append(f"  {r['name']:10} {tuple(r['shape'])} x {r['channels']}   step {r['step_mm']} mm   kernel {r['kernel']}   "
                    f"in extent {r['in_extent']} of {r['tokens']}   reach {r['reach_mm']}   layer {r['key'][2]}")
    g = f.input_grid
    text.append(f"  input CT grid: {g['shape']} (compare with your CT before trusting a mask drawn on it)" if g else
                "  input CT grid: not recorded")
    _out(data, as_json, "\n".join(text))


@main.command()
@click.argument("field")
@labels_options
@click.option("--structure", "structures", multiple=True, help="only these structures (repeatable); default: all")
@click.option("--names", "names_json", help="for a plain labelmap .nrrd: its names as JSON {name: value}")
@click.option("--top", default=3, show_default=True, help="nearest structures to list for each")
@click.option("--min-tokens", default=5, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def vectors(ctx, field, labels, haversack, series, task, structures, names_json, top, min_tokens, as_json):
    """One vector per structure (finest lattice, shared mean off), and each one's nearest structures."""
    from . import client as fc
    f = fc.open_field(field, ctx.obj["token"])
    lm = fc.read_seg_nrrd(_labels_path(labels, haversack, series, task), json.loads(names_json) if names_json else None,
                          token=ctx.obj["token"])
    vec = fc.organ_vectors(f, lm, min_tokens=min_tokens)
    names = sorted(vec)
    M = np.array([vec[n] for n in names])
    S = M @ M.T
    shown = [n for n in names if not structures or n in structures]
    near = {n: [(names[j], round(float(S[i, j]), 4)) for j in np.argsort(-S[i]) if j != i][:top]
            for i, n in enumerate(names) if n in shown}
    _out({"field": field, "structures": len(names), "nearest": near}, as_json,
         "\n".join([f"{len(names)} structures with at least {min_tokens} tokens; nearest by cosine (shared mean off):"]
                   + [f"  {n:28} " + ", ".join(f"{m} {c:.2f}" for m, c in near[n]) for n in shown]))


@main.group(cls=_Group)
def reference():
    """Normal tissue from a few normal scans of the same protocol."""


@reference.command("build")
@click.option("-o", "--out", required=True, help="the reference to write (<name>.zarr.zip)")
@click.option("--normal", "normals", nargs=2, multiple=True, required=True, metavar="FIELD LABELS",
              help="a normal scan's field and its .seg.nrrd (paths or URLs); repeat for each (at least two)")
@click.option("--structure", "structures", multiple=True, required=True, help="the normal tissue: an organ (repeatable)")
@click.option("--erode", default=15.0, show_default=True, help="mm to erode the region: interior tokens only")
@click.option("--k", default=5, show_default=True, help="nearest normal tokens a score averages over")
@click.option("--note", default="", help="what the builder wants recorded, e.g. the protocol")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def reference_build(ctx, out, normals, structures, erode, k, note, as_json):
    """Build a normal-tissue reference: its threshold is the largest distance a held-out normal token reached."""
    from . import client as fc
    pairs = []
    for field, labels in normals:
        lm = fc.read_seg_nrrd(labels, token=ctx.obj["token"])
        pairs.append((fc.open_field(field, ctx.obj["token"]), _region(lm, structures, (), erode)))
    ref = fc.Reference.build(pairs, k=k, meta={"structures": list(structures), "erode_mm": erode, "note": note,
                                               "labels": [l for _, l in normals]})
    ref.save(out)
    _out({"reference": out, "threshold": ref.threshold, "tokens_per_normal": ref.meta["tokens_per_normal"],
          "key": list(ref.key)}, as_json,
         f"{out}: {len(ref.tokens)} normal tokens {ref.meta['tokens_per_normal']}, threshold {ref.threshold:.4f}, "
         f"{list(structures)} eroded {erode:g} mm, k {k}; license {ref.meta['license']}")


@main.command()
@click.argument("field")
@labels_options
@click.option("--reference", "ref_path", required=True, help="a reference built by `feldglas reference build`")
@click.option("--structure", "structures", multiple=True, required=True, help="the organ to test (repeatable)")
@click.option("--optional", "optional", multiple=True,
              help="lesion segments to union in when present (repeatable; a trailing * is a prefix)")
@click.option("--erode", type=float, default=None, help="mm to erode the region (default: as the reference was)")
@click.option("--radius", default=15.0, show_default=True, help="mm within which flagged tokens join one site")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def score(ctx, field, labels, haversack, series, task, ref_path, structures, optional, erode, radius, as_json):
    """Score a region against a reference: flagged sites, strongest first. A site is a place to look."""
    from . import client as fc
    ref = fc.Reference.load(ref_path, ctx.obj["token"])
    erode = float(ref.meta.get("erode_mm", 0.0)) if erode is None else erode
    f = fc.open_field(field, ctx.obj["token"])
    lm = fc.read_seg_nrrd(_labels_path(labels, haversack, series, task), token=ctx.obj["token"])
    s = ref.score(f, _region(lm, structures, optional, erode), radius=radius)
    sites = [{"center_lps": np.round(x.center, 1).tolist(), "tokens": x.tokens, "peak": round(x.peak, 4)} for x in s.sites]
    data = {"field": field, "reference": ref_path, "threshold": s.threshold, "scored": len(s.distance),
            "flagged": int(s.flagged.sum()), "region": {"structures": list(structures), "optional": list(optional),
                                                         "erode_mm": erode}, "sites": sites}
    text = [f"{len(s.distance)} tokens scored ({list(structures) + list(optional)}, eroded {erode:g} mm); "
            f"threshold {s.threshold:.4f}; {int(s.flagged.sum())} flagged in {len(sites)} site(s)"]
    text += [f"  site {i + 1}: LPS {x['center_lps']}  {x['tokens']} token(s)  peak {x['peak']:.3f}" for i, x in enumerate(sites)]
    _out(data, as_json, "\n".join(text))


if __name__ == "__main__":
    sys.exit(main())
