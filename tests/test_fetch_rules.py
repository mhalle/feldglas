"""feldglas.fetch against what the network review (2026-09-23) did to it: one parse deciding host and
token, proxies, 202 "still computing" answers, refusals that must not fall back to a stale copy,
a cache that must not pin the wrong version, size caps. Pure rules are tested as tables; the rest
against a local server that can change what it answers, or go away."""
import hashlib, http.server, json, os, pathlib, tempfile, threading, unittest

import pytest

pytest.importorskip("obstore")

from feldglas import fetch


class Server:
    """Serves files from ``root``; ``status`` forces an answer, ``body202`` answers 202 with a JSON
    note (haversack while it computes); records what it saw; can be stopped."""

    def __init__(self, root: pathlib.Path):
        self.root, self.seen, self.status = root, [], None
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                   "inm": self.headers.get("If-None-Match")})
                if outer.status == 202:
                    b = json.dumps({"detail": "materializing"}).encode()
                    self.send_response(202); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
                    return
                if outer.status:
                    self.send_response(outer.status); self.send_header("Content-Length", "0"); self.end_headers(); return
                f = outer.root / self.path.lstrip("/")
                if not f.is_file():
                    self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
                body = f.read_bytes()
                tag = '"sha256:' + hashlib.sha256(body).hexdigest() + '"'
                if self.headers.get("If-None-Match") == tag:
                    self.send_response(304); self.send_header("ETag", tag); self.end_headers(); return
                self.send_response(200); self.send_header("ETag", tag); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown(); self.httpd.server_close()


class Case(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.root = self.d / "served"; self.root.mkdir()
        self.s = Server(self.root); self.addCleanup(self.s.stop)
        self.env = {k: os.environ.get(k) for k in ("FELDGLAS_CACHE", "FELDGLAS_TOKEN", "FELDGLAS_MAX_FETCH_BYTES",
                                                   "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")}
        self.addCleanup(self.restore)
        for k in self.env:
            os.environ.pop(k, None)
        os.environ["FELDGLAS_CACHE"] = str(self.d / "cache")

    def restore(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def put(self, name, data: bytes):
        p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)


class Origin(Case):
    def test_one_parse_refuses_what_could_mean_two_hosts(self):
        for url, why in (("http://evil.example\\@127.0.0.1/x", "backslash"), ("http://u:p@127.0.0.1/x", "userinfo"),
                         ("http://127.0.0.1:8/dir/", "directory"), ("http://127.0.0.1:8/a%2Fb", "encoded slash"),
                         ("http://127.0.0.1:8/x?y=1", "query"), ("http://127.0.0.1:8/x#f", "query or fragment"),
                         ("http://127.0.0.1:port/x", "port")):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, why):
                fetch.origin(url)
        self.assertEqual(fetch.origin("HTTP://LocalHost:8790/v1/idc/a/ts.v2:total/labels.seg.nrrd"),
                         ("http", "localhost", 8790, "v1/idc/a/ts.v2:total/labels.seg.nrrd"))
        self.assertEqual(fetch.origin("http://[::1]:9/a%20b")[1:], ("::1", 9, "a b"))

    def test_the_token_rule(self):
        for scheme, host, ok in (("https", "example.org", True), ("http", "127.0.0.1", True), ("http", "localhost", True),
                                 ("http", "::1", True), ("http", "example.org", False), ("http", "127.0.0.2", False)):
            self.assertEqual(fetch.token_allowed(scheme, host), ok, (scheme, host))

    def test_the_token_goes_to_localhost_and_the_flag_outranks_the_environment(self):
        self.put("a.bin", b"x")
        fetch.local(f"http://localhost:{self.s.port}/a.bin", token="flag")
        self.assertEqual(self.s.seen[-1]["auth"], "Bearer flag")
        os.environ["FELDGLAS_TOKEN"] = "env"
        self.put("b.bin", b"y")
        fetch.local(f"{self.s.url}/b.bin")
        self.assertEqual(self.s.seen[-1]["auth"], "Bearer env")
        self.put("c.bin", b"z")
        fetch.local(f"{self.s.url}/c.bin", token="flag")
        self.assertEqual(self.s.seen[-1]["auth"], "Bearer flag")

    def test_a_local_server_is_never_reached_through_a_proxy(self):
        proxy = Server(self.d); self.addCleanup(proxy.stop)
        os.environ["HTTP_PROXY"] = os.environ["http_proxy"] = proxy.url
        self.put("a.bin", b"x")
        self.assertEqual(fetch.local(f"{self.s.url}/a.bin", token="secret").read_bytes(), b"x")
        self.assertEqual(proxy.seen, [])


class Answers(Case):
    def test_still_computing_is_refused_and_never_replaces_a_good_copy(self):
        self.put("l.seg.nrrd", b"NRRD0004\n\nfirst")
        good = fetch.local(f"{self.s.url}/l.seg.nrrd", expect="nrrd")
        self.s.status = 202
        self.put("l.seg.nrrd", b"NRRD0004\n\nsecond")                     # so a revalidation is not a 304
        with self.assertRaisesRegex(fetch.FetchError, "materializing.*still computing"):
            fetch.local(f"{self.s.url}/l.seg.nrrd", expect="nrrd")
        self.assertEqual(good.read_bytes(), b"NRRD0004\n\nfirst")
        self.assertEqual(list((self.d / "cache" / "http").glob("*.partial")), [])

    def test_refusals_do_not_fall_back_but_a_network_failure_does(self):
        self.put("a.bin", b"cached")
        fetch.local(f"{self.s.url}/a.bin")
        self.put("a.bin", b"changed")
        for status, words in ((401, "wants a token"), (403, "refused this token"), (409, "conflict"), (410, "purged"),
                              (429, "retry later"), (500, "answered 500"), (503, "still computing")):
            self.s.status = status
            with self.subTest(status=status), self.assertRaisesRegex(fetch.FetchError, words):
                fetch.local(f"{self.s.url}/a.bin")
        self.s.status = None
        self.s.stop()
        import contextlib, io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(fetch.local(f"{self.s.url}/a.bin").read_bytes(), b"cached")
        self.assertIn("could not be reached", err.getvalue())
        with self.assertRaises(fetch.FetchError):
            fetch.local(f"{self.s.url}/never.bin")                       # nothing cached: nothing to fall back on

    def test_a_tampered_or_vanished_copy_is_fetched_again(self):
        self.put("a.bin", b"right")
        p = fetch.local(f"{self.s.url}/a.bin")
        p.write_bytes(b"wrong")                                           # the copy no longer matches its sidecar
        self.assertEqual(fetch.local(f"{self.s.url}/a.bin").read_bytes(), b"right")
        self.assertIsNone(self.s.seen[-1]["inm"])                         # not revalidated against a copy it cannot trust
        p.unlink()
        self.assertEqual(fetch.local(f"{self.s.url}/a.bin").read_bytes(), b"right")
        self.assertIsNone(self.s.seen[-1]["inm"])

    def test_the_size_cap_and_the_expected_kind(self):
        self.put("big.bin", b"x" * 5000)
        os.environ["FELDGLAS_MAX_FETCH_BYTES"] = "1000"
        with self.assertRaisesRegex(fetch.FetchError, "cap"):
            fetch.local(f"{self.s.url}/big.bin")
        os.environ.pop("FELDGLAS_MAX_FETCH_BYTES")
        self.put("f.zarr.zip", b"not a zip")
        with self.assertRaisesRegex(fetch.FetchError, "not a zip"):
            fetch.local(f"{self.s.url}/f.zarr.zip", expect="zip")

    def test_one_fetch_of_a_url_at_a_time_and_the_copy_is_the_sidecars(self):
        self.put("a.bin", b"v0")
        errors = []

        def worker(i):
            try:
                for _ in range(5):
                    fetch.local(f"{self.s.url}/a.bin")
            except Exception as e:                                       # noqa: BLE001
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for v in range(1, 6):
            self.put("a.bin", f"v{v}".encode())
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        body, meta, _ = fetch._paths(f"{self.s.url}/a.bin", None)
        self.assertEqual(json.loads(meta.read_text())["sha256"], hashlib.sha256(body.read_bytes()).hexdigest())
        self.assertEqual(fetch.local(f"{self.s.url}/a.bin").read_bytes(), b"v5")   # and the next fetch is current

    def test_a_long_name_is_cached_under_a_short_one(self):
        name = "n" * 240 + ".bin"                                        # the cache name adds 21; the limit is 255
        self.put(name, b"x")
        p = fetch.local(f"{self.s.url}/{name}")
        self.assertLessEqual(len(p.name), 130)
        self.assertEqual(p.read_bytes(), b"x")
