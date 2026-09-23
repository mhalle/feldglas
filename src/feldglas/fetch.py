"""Files by URL: wherever feldglas takes a field or a ``.seg.nrrd``, an ``http(s)://`` URL does too.

The family's server is the point (2026-09-23): a cached haversack result is addressable without a
job - ``<server>/v1/<source>/<identifier>/<task>/labels.seg.nrrd`` - on its anonymous twin, on a
Modal deployment, or on a ``haversack serve`` on this machine (plain ``http://``). Its ETag is the
digest of the bytes, so a copy fetched once is revalidated with ``If-None-Match`` and never
fetched twice.

obstore does the HTTP (its generic HTTP store; the family already uses obstore for object stores):
fast, retried, and range-capable for the day a field is read chunk by chunk instead of whole.
Imported inside the function: a caller with only local files needs no obstore.

Fetched files live under ``<cache>/http/`` (``$FELDGLAS_CACHE``; a field inherits its weights'
license, so the cache, like every feldglas cache, is outside any repository). A token - ``token=``
or ``$FELDGLAS_TOKEN`` - is sent only to the URL's own origin, and never over plain http to a
host that is not this machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import urllib.parse

from .paths import cache_dir

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def is_url(x) -> bool:
    return str(x).startswith(("http://", "https://"))


def _cached(url: str, cache: pathlib.Path | None) -> tuple[pathlib.Path, pathlib.Path]:
    root = pathlib.Path(cache) if cache else cache_dir() / "http"
    name = pathlib.PurePosixPath(urllib.parse.urlsplit(url).path).name or "file"
    stem = hashlib.sha256(url.encode()).hexdigest()[:20]
    return root / f"{stem}-{name}", root / f"{stem}-{name}.json"


def local(path_or_url, token: str | None = None, cache=None) -> pathlib.Path:
    """A local path for ``path_or_url``: a path is returned as it is; a URL is fetched (or
    revalidated) into the cache and its copy returned."""
    if not is_url(path_or_url):
        return pathlib.Path(path_or_url).expanduser()
    url = str(path_or_url)
    u = urllib.parse.urlsplit(url)
    if u.query or u.fragment:
        raise ValueError(f"{url}: a URL with a query or fragment is not a file this reads")
    token = token if token is not None else os.environ.get("FELDGLAS_TOKEN") or None
    if token and u.scheme == "http" and (u.hostname or "") not in _LOCAL_HOSTS:
        raise ValueError(f"{url}: a token is never sent over plain http to another host - use https")
    from obstore import get
    from obstore.exceptions import NotModifiedError
    from obstore.store import HTTPStore
    opts = {"user_agent": "feldglas", "timeout": "300s"}
    if u.scheme == "http":
        opts["allow_http"] = True
    if token:
        opts["default_headers"] = {"authorization": f"Bearer {token}"}
    store = HTTPStore.from_url(f"{u.scheme}://{u.netloc}", client_options=opts)
    path = urllib.parse.unquote(u.path.lstrip("/"))
    body, meta = _cached(url, cache)
    known = json.loads(meta.read_text()) if meta.exists() and body.exists() else {}
    try:
        options = {"if_none_match": known["e_tag"]} if known.get("e_tag") else None
        got = get(store, path, options=options)
    except NotModifiedError:
        return body
    except FileNotFoundError:
        raise FileNotFoundError(f"{url}: not found (404). On a haversack server a result path answers 404 when the "
                                "result is not cached - compute it first (an authorized request with Prefer, or "
                                "POST /v1/jobs), then ask again") from None
    except Exception as e:                                   # a network failure: a known copy still serves
        if known:
            print(f"feldglas: {url} could not be revalidated ({type(e).__name__}); using the copy fetched before",
                  file=sys.stderr)
            return body
        raise OSError(f"{url}: {e}") from None
    body.parent.mkdir(parents=True, exist_ok=True)
    tmp = body.with_name(body.name + ".partial")
    tmp.write_bytes(bytes(got.bytes()))
    tmp.replace(body)                                        # never half a file under the real name
    meta.write_text(json.dumps({"url": url, "e_tag": got.meta.get("e_tag"), "size": got.meta.get("size")}))
    return body
