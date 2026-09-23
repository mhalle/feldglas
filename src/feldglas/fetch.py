"""Files by URL: wherever feldglas takes a field, a reference or a ``.seg.nrrd``, an ``http(s)://``
URL does too.

The family's server is the point (2026-09-23): a cached haversack result is addressable without a
job - ``<server>/v1/<source>/<identifier>/<task>/labels.seg.nrrd`` - on its anonymous twin, on a
Modal deployment, or on a ``haversack serve`` on this machine (plain ``http://``). Its ETag is the
digest of the bytes, so a copy fetched once is revalidated with ``If-None-Match`` and never
fetched twice. obstore does the HTTP (its generic HTTP store): fast, retried, streamed.

What an adversarial review (2026-09-23) found and this now guarantees:
- ONE parse of the URL decides both where the request goes and whether a token may go with it:
  the store is built from the parsed host and port, never from the raw netloc (``\\`` in a netloc
  once made Python check one host while the request went to another, token included). Userinfo,
  backslashes and other oddities in the authority are refused.
- A token goes to its own origin only, never over plain http to a host that is not this machine,
  and never through a proxy over plain http (local hosts bypass any proxy).
- The cache cannot pin the wrong version: one fetch of a URL at a time (a lock), a private temp
  file per fetch, and the body's sha256 in the sidecar, checked before a cached copy is used.
- A body is kept only if it is the kind of file the caller expects (``expect``: a zip, an NRRD):
  haversack answers 202 with a JSON note while it computes, and obstore calls every 2xx success.
- A stale copy is served only when the network fails - never after the server REFUSED (401, 403,
  409, 410, 429, 5xx), which may mean the old copy is wrong, not merely old.
- Bodies stream to disk under a size cap (``$FELDGLAS_MAX_FETCH_BYTES``, default 4 GB).
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import urllib.parse

from .paths import cache_dir

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_SIGNATURES = {"zip": (b"PK\x03\x04",), "nrrd": (b"NRRD",)}
_STATUS = re.compile(r"status(?: code)?:\s*(\d{3})")


class FetchError(OSError):
    """A URL that did not yield the file, with the reason in one line."""


def is_url(x) -> bool:
    return str(x).lower().startswith(("http://", "https://"))


def origin(url: str) -> tuple[str, str, int | None, str]:
    """``(scheme, host, port, path)`` of a URL - the ONE parse both the token rule and the request
    use. Refuses what could be read two ways."""
    u = urllib.parse.urlsplit(url)
    scheme = u.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"{url}: not an http(s) URL")
    if u.query or u.fragment:
        raise ValueError(f"{url}: a URL with a query or fragment is not a file this reads")
    if "@" in u.netloc or "\\" in url or any(c in u.netloc for c in " \t\r\n%"):
        raise ValueError(f"{url}: userinfo, backslashes or escapes in the host part are refused - they can make one "
                         "URL mean two hosts")
    if not u.hostname:
        raise ValueError(f"{url}: no host")
    try:
        port = u.port
    except ValueError:
        raise ValueError(f"{url}: a port that is not a number") from None
    if not u.path or u.path.endswith("/"):
        raise ValueError(f"{url}: names a directory, not a file")
    if re.search(r"%2[fF]|%5[cC]", u.path):
        raise ValueError(f"{url}: an encoded slash in the path would name another file")
    return scheme, u.hostname.lower(), port, urllib.parse.unquote(u.path.lstrip("/"))


def token_allowed(scheme: str, host: str) -> bool:
    """A token may go over https anywhere, and over plain http only to this machine."""
    return scheme == "https" or host in _LOCAL_HOSTS


@contextlib.contextmanager
def _no_proxy_for(host: str):
    """A local server is never reached through a proxy: the HTTP client takes proxies from the
    environment when a store is built and honors NO_PROXY there (obstore's ``proxy_excludes`` covers
    only a proxy it was given, and a review saw a token for 127.0.0.1 go to $HTTP_PROXY in clear)."""
    if host not in _LOCAL_HOSTS:
        yield
        return
    saved = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
    extra = ",".join(sorted(_LOCAL_HOSTS))
    try:
        for k, v in saved.items():
            os.environ[k] = f"{v},{extra}" if v else extra
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _paths(url: str, cache) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    root = pathlib.Path(cache) if cache else cache_dir() / "http"
    name = (pathlib.PurePosixPath(urllib.parse.urlsplit(url).path).name or "file")[-100:]
    stem = hashlib.sha256(url.encode()).hexdigest()[:20]
    return root / f"{stem}-{name}", root / f"{stem}-{name}.json", root / f"{stem}.lock"


@contextlib.contextmanager
def _locked(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except ImportError:                                  # no flock (Windows): the temp file + rename still hold
            pass
        yield


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _message(url: str, status: int | None, detail: str) -> str:
    known = {401: "the server wants a token (--token or $FELDGLAS_TOKEN)",
             403: "the server refused this token",
             404: "not found (404) - check the URL; on a haversack result path a wrong source, series or task "
                  "name is also a 404, and so is a result that is not cached yet (an authorized request with "
                  "Prefer, or POST /v1/jobs, computes it)",
             409: "conflict (409) - on haversack, a pinned version the server does not have",
             410: "gone (410) - on haversack, a result that was purged: compute it again",
             429: "too many requests (429) - retry later",
             503: "unavailable (503) - on haversack, busy or still computing: retry later"}
    if status in known:
        return f"{url}: {known[status]}"
    if status:
        return f"{url}: the server answered {status}"
    return f"{url}: {detail.splitlines()[0][:200] if detail else 'the request failed'}"


def local(path_or_url, token: str | None = None, cache=None, expect: str | None = None) -> pathlib.Path:
    """A local path for ``path_or_url``: a path is returned as it is; a URL is fetched (or
    revalidated) into the cache and its copy returned. ``expect`` ("zip", "nrrd") is the kind of
    file the caller needs: anything else the server sends is refused and never cached."""
    if not is_url(path_or_url):
        return pathlib.Path(path_or_url).expanduser()
    url = str(path_or_url)
    scheme, host, port, path = origin(url)
    token = token if token is not None else os.environ.get("FELDGLAS_TOKEN") or None
    if token and not token_allowed(scheme, host):
        raise ValueError(f"{url}: a token is never sent over plain http to another host - use https")
    from obstore import get
    from obstore import exceptions as ox
    from obstore.store import HTTPStore
    opts = {"user_agent": "feldglas", "timeout": "300s", "connect_timeout": "10s"}
    if scheme == "http":
        opts["allow_http"] = True
    if token:
        opts["default_headers"] = {"authorization": f"Bearer {token}"}
    netloc = f"[{host}]" if ":" in host else host
    with _no_proxy_for(host):
        store = HTTPStore.from_url(f"{scheme}://{netloc}" + (f":{port}" if port else ""), client_options=opts,
                                   retry_config={"max_retries": 2, "retry_timeout": datetime.timedelta(seconds=30),
                                                 "backoff": {"init_backoff": datetime.timedelta(seconds=1),
                                                             "max_backoff": datetime.timedelta(seconds=5), "base": 2}})
    body, meta, lock = _paths(url, cache)
    cap = int(os.environ.get("FELDGLAS_MAX_FETCH_BYTES", 4 << 30))
    with _locked(lock):
        known = {}
        if meta.exists() and body.exists():
            try:
                known = json.loads(meta.read_text())
            except ValueError:
                known = {}
            if known.get("sha256") != _sha256(body):         # a copy that is not what was fetched: forget it
                known = {}
        try:
            got = get(store, path, options={"if_none_match": known["e_tag"]} if known.get("e_tag") else None)
        except ox.NotModifiedError:
            if known:
                return body
            raise FetchError(_message(url, None, "the server said 'not modified' for a file never fetched")) from None
        except (ox.NotFoundError, FileNotFoundError):           # obstore raises the builtin for a 404
            raise FileNotFoundError(_message(url, 404, "")) from None
        except (ox.UnauthenticatedError, ox.PermissionDeniedError, ox.AlreadyExistsError, ox.PreconditionError) as e:
            status = {ox.UnauthenticatedError: 401, ox.PermissionDeniedError: 403, ox.AlreadyExistsError: 409,
                      ox.PreconditionError: 412}[type(e)]
            raise FetchError(_message(url, status, "")) from None
        except Exception as e:
            m = _STATUS.search(str(e))
            status = int(m.group(1)) if m else None
            if status is None and known:                     # the NETWORK failed: a verified copy still serves
                print(f"feldglas: {url} could not be reached; using the copy fetched before", file=sys.stderr)
                return body
            raise FetchError(_message(url, status, str(e))) from None
        size = got.meta.get("size")
        if size is not None and size > cap:
            raise FetchError(f"{url}: {size} bytes, past the {cap}-byte cap ($FELDGLAS_MAX_FETCH_BYTES)")
        body.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=body.name[:40] + ".", suffix=".partial", dir=body.parent)
        tmp = pathlib.Path(tmp)
        try:
            n = 0
            with os.fdopen(fd, "wb") as out:
                for chunk in got.stream():
                    n += len(chunk)
                    if n > cap:
                        raise FetchError(f"{url}: more than the {cap}-byte cap ($FELDGLAS_MAX_FETCH_BYTES)")
                    out.write(bytes(chunk))
            if expect:
                with open(tmp, "rb") as fh:
                    head = fh.read(512)
                if not head.startswith(_SIGNATURES[expect]):
                    hint = ""
                    try:
                        detail = json.loads(head.decode("utf8", "replace")).get("detail")
                        hint = f" - the server says {detail!r} (on haversack: still computing? retry shortly)" if detail else ""
                    except (ValueError, AttributeError):
                        pass
                    raise FetchError(f"{url}: the answer is not a {expect} file{hint}")
            digest = _sha256(tmp)
            os.replace(tmp, body)                            # the old copy stays until a good one replaces it
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        meta.write_text(json.dumps({"url": url, "e_tag": got.meta.get("e_tag"), "size": n, "sha256": digest}))
    return body
