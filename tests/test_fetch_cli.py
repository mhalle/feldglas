"""Files by URL (feldglas.fetch) and the `feldglas` command. A local HTTP server stands in for a
haversack server - content-digest ETags, 304 on If-None-Match, 404 for what is not cached - so the
tests run offline, over plain http as a local `haversack serve` would."""
import hashlib, http.server, json, os, pathlib, tempfile, threading, unittest

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("obstore")
pytest.importorskip("click")

from click.testing import CliRunner

from feldglas import client as fc
from feldglas import fetch
from feldglas.cli import main
from feldglas.store import write_field
from test_client import cluster_field
from test_labels_session import write_seg


class _Server:
    """Serves a directory with ETags (sha256 of the bytes), 304s and 404s; records what it saw."""

    def __init__(self, root: pathlib.Path, redirect_to: str | None = None):
        self.root, self.seen, self.redirect_to = root, [], redirect_to
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                   "inm": self.headers.get("If-None-Match")})
                if outer.redirect_to and self.path.startswith("/moved/"):
                    self.send_response(302); self.send_header("Location", outer.redirect_to + self.path[len("/moved"):])
                    self.end_headers(); return
                f = outer.root / self.path.lstrip("/")
                if not f.is_file():
                    self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers(); return
                body = f.read_bytes()
                tag = '"sha256:' + hashlib.sha256(body).hexdigest() + '"'
                if self.headers.get("If-None-Match") == tag:
                    self.send_response(304); self.send_header("ETag", tag); self.end_headers(); return
                self.send_response(200); self.send_header("ETag", tag); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)

            do_HEAD = do_GET

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()


class Base(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.served = self.d / "served"; self.served.mkdir()
        self.server = _Server(self.served); self.addCleanup(self.server.close)
        old = os.environ.get("FELDGLAS_CACHE")
        os.environ["FELDGLAS_CACHE"] = str(self.d / "cache")
        self.addCleanup(lambda: os.environ.__setitem__("FELDGLAS_CACHE", old) if old else os.environ.pop("FELDGLAS_CACHE", None))
        os.environ.pop("FELDGLAS_TOKEN", None)


class Fetch(Base):
    def test_a_path_is_itself_and_a_url_is_fetched_once_then_revalidated(self):
        self.assertEqual(fetch.local(self.d / "x"), self.d / "x")
        (self.served / "a.bin").write_bytes(b"first")
        p = fetch.local(self.server.url + "/a.bin")
        self.assertEqual(p.read_bytes(), b"first")
        q = fetch.local(self.server.url + "/a.bin")
        self.assertEqual((q, q.read_bytes()), (p, b"first"))
        self.assertIsNone(self.server.seen[0]["inm"])
        self.assertTrue(self.server.seen[1]["inm"].startswith('"sha256:'))       # asked "has it changed?", got a 304
        (self.served / "a.bin").write_bytes(b"second")                            # a recompute republished it
        self.assertEqual(fetch.local(self.server.url + "/a.bin").read_bytes(), b"second")

    def test_two_urls_with_one_file_name_are_two_files(self):
        (self.served / "a").mkdir(); (self.served / "b").mkdir()
        (self.served / "a/labels.seg.nrrd").write_bytes(b"one"); (self.served / "b/labels.seg.nrrd").write_bytes(b"two")
        self.assertEqual(fetch.local(self.server.url + "/a/labels.seg.nrrd").read_bytes(), b"one")
        self.assertEqual(fetch.local(self.server.url + "/b/labels.seg.nrrd").read_bytes(), b"two")
        self.assertEqual(fetch.local(self.server.url + "/a/labels.seg.nrrd").read_bytes(), b"one")
        tag_a = '"sha256:' + hashlib.sha256(b"one").hexdigest() + '"'
        self.assertEqual(self.server.seen[-1]["inm"], tag_a)                      # revalidated ITS copy: a 304, no body

    def test_not_cached_on_the_server_says_so(self):
        with self.assertRaisesRegex(FileNotFoundError, r"not found \(404\).*wrong source, series or task.*not cached yet"):
            fetch.local(self.server.url + "/v1/idc/x/ts.v2:total/labels.seg.nrrd")

    def test_a_token_goes_to_its_origin_only_and_never_over_http_elsewhere(self):
        (self.served / "a.bin").write_bytes(b"x")
        fetch.local(self.server.url + "/a.bin", token="secret")
        self.assertEqual(self.server.seen[-1]["auth"], "Bearer secret")
        with self.assertRaisesRegex(ValueError, "never sent over plain http"):
            fetch.local("http://example.invalid/a.bin", token="secret")
        other = _Server(self.served); self.addCleanup(other.close)                 # another origin (port)
        front = _Server(self.served, redirect_to=other.url); self.addCleanup(front.close)
        fetch.local(front.url + "/moved/a.bin", token="secret")
        self.assertEqual(front.seen[-1]["auth"], "Bearer secret")
        self.assertEqual([s["auth"] for s in other.seen], [None])                  # not carried across the redirect
        with self.assertRaisesRegex(ValueError, "query"):
            fetch.local(self.server.url + "/a.bin?x=1")

    def test_fields_labels_and_references_open_from_urls(self):
        f = cluster_field(0)
        write_field(self.served / "f.zarr.zip", f)
        g = f.grid
        A = np.eye(4); A[:3, :3] = np.asarray(g.directions, float).T; A[:3, 3] = g.origin
        lab = np.zeros(g.shape, np.uint8); lab[2:12, 8:56, 8:56] = 1
        write_seg(self.served / "l.seg.nrrd", lab, A, {"liver": 1})
        v = fc.open_field(self.server.url + "/f.zarr.zip")
        np.testing.assert_allclose(v.fine.tokens, fc.open_field(self.served / "f.zarr.zip").fine.tokens)
        np.testing.assert_array_equal(fc.seg_mask(self.server.url + "/l.seg.nrrd", "liver").values, lab == 1)


class Command(Base):
    def run_cli(self, *args):
        r = CliRunner().invoke(main, [str(a) for a in args])
        return r

    def setUp(self):
        super().setUp()
        g = cluster_field(0).grid
        A = np.eye(4); A[:3, :3] = np.asarray(g.directions, float).T; A[:3, 3] = g.origin
        lab = np.zeros(g.shape, np.uint8); lab[:, :, :] = 1; lab[:, :, 48:] = 2
        for i in range(4):
            anomaly = [(z, y, x) for z in (3, 4) for y in (2, 3) for x in (2, 3)] if i == 3 else None
            write_field(self.served / f"s{i}.zarr.zip", cluster_field(i, anomaly=anomaly))
            write_seg(self.served / f"s{i}.seg.nrrd", lab, A, {"liver": 1, "spleen": 2})
        (self.served / "v1/idc/s3/ts.v2:total").mkdir(parents=True)
        (self.served / "v1/idc/s3/ts.v2:total/labels.seg.nrrd").write_bytes((self.served / "s3.seg.nrrd").read_bytes())

    def test_info(self):
        r = self.run_cli("info", self.server.url + "/s0.zarr.zip", "--json")
        self.assertEqual(r.exit_code, 0, r.output)
        d = json.loads(r.output)
        self.assertEqual((d["encoder"], len(d["lattices"])), ("synthetic", 3))

    def test_vectors_by_labels_url(self):
        r = self.run_cli("vectors", self.server.url + "/s0.zarr.zip", "--labels", self.server.url + "/s0.seg.nrrd", "--json")
        self.assertEqual(r.exit_code, 0, r.output)
        self.assertEqual(set(json.loads(r.output)["structures"]), {"liver", "spleen"})

    def test_reference_build_then_score_a_planted_anomaly_from_a_haversack_path(self):
        u = self.server.url
        ref = self.d / "ref.zarr.zip"
        args = ["reference", "build", "-o", ref, "--structure", "liver", "--erode", "0", "--note", "synthetic"]
        for i in range(3):
            args += ["--normal", f"{u}/s{i}.zarr.zip", f"{u}/s{i}.seg.nrrd"]
        r = self.run_cli(*args)
        self.assertEqual(r.exit_code, 0, r.output)
        loaded = fc.Reference.load(ref)
        self.assertEqual((loaded.meta["erode_mm"], loaded.meta["note"], loaded.meta["structures"]), (0.0, "synthetic", ["liver"]))
        r = self.run_cli("score", f"{u}/s3.zarr.zip", "--haversack", u, "--series", "idc:s3", "--task", "ts.v2:total",
                         "--reference", ref, "--structure", "liver", "--optional", "liver_tumor", "--json")
        self.assertEqual(r.exit_code, 0, r.output)
        d = json.loads(r.output)
        planted = fc.open_field(self.served / "s3.zarr.zip").fine
        idx = np.ravel_multi_index(tuple(np.array([(z, y, x) for z in (3, 4) for y in (2, 3) for x in (2, 3)]).T), planted.shape)
        self.assertLess(float(np.linalg.norm(np.asarray(d["sites"][0]["center_lps"]) - planted.centers[idx].mean(0))), 5.0)
        self.assertEqual(d["region"]["erode_mm"], 0.0)                          # as the reference was

    def test_errors_are_one_line(self):
        r = self.run_cli("vectors", self.server.url + "/s0.zarr.zip", "--haversack", self.server.url,
                         "--series", "idc:nothing", "--task", "ts.v2:total")
        self.assertEqual(r.exit_code, 1)
        self.assertIn("not cached yet", r.output); self.assertNotIn("Traceback", r.output)
        r = self.run_cli("vectors", self.server.url + "/s0.zarr.zip")
        self.assertEqual(r.exit_code, 2); self.assertIn("labels are needed", r.output)
        r = self.run_cli("score", self.server.url + "/s0.zarr.zip", "--labels", self.server.url + "/s0.seg.nrrd",
                         "--reference", self.d / "missing.zarr.zip", "--structure", "liver")
        self.assertEqual(r.exit_code, 1); self.assertNotIn("Traceback", r.output)

    def test_health_reports_readiness_and_a_servers_health(self):
        (self.served / "v1").mkdir(exist_ok=True)
        (self.served / "v1/health").write_text(json.dumps({"status": "ok", "version": "0.12.9"}))
        r = self.run_cli("health", "--json", "--haversack", self.server.url)
        self.assertEqual(r.exit_code, 0, r.output)
        d = json.loads(r.output)
        self.assertTrue(d["ok"]); self.assertEqual(d["problems"], [])
        self.assertEqual(d["server"]["health"]["version"], "0.12.9")
        self.assertIn("score", d["commands"]); self.assertEqual(set(d["exit_codes"]), {"0", "1", "2"})
        self.assertEqual(d["reads"]["field"]["versions"], ["0.2"])
        r = CliRunner().invoke(main, ["--token", "s3cret", "health", "--json", "--haversack", "http://127.0.0.1:9"])
        self.assertEqual(r.exit_code, 1)
        d = json.loads(r.output)
        self.assertFalse(d["ok"]); self.assertFalse(d["server"]["reachable"]); self.assertTrue(d["token"]["set"])
        self.assertNotIn("s3cret", r.output)
