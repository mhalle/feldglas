"""The manifest, and fields moving through provender - against its in-memory store."""
import json, pathlib, tempfile, unittest
import unittest.mock

import numpy as np
import pytest

pytest.importorskip("provender"); obstore = pytest.importorskip("obstore")
from obstore.store import MemoryStore                                   # noqa: E402
from provender import Blobs, EmptyKeepSet                               # noqa: E402

from feldglas import Provenance                                         # noqa: E402
from feldglas.remote import Manifest, open_blobs, pull, push, sweep    # noqa: E402
from feldglas.store import read_field, write_field                     # noqa: E402
from conftest import make_field                                         # noqa: E402

RADAR = dict(encoder="radar", code="c1", weights="w1", preprocessing="p1", license="CC-BY-NC-SA-4.0")


class _Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(); self.tmp = pathlib.Path(self._tmp.name)
        self.blobs = Blobs(MemoryStore(), "feldglas/radar/")
        self.manifest = Manifest.load(self.tmp / "manifests" / "radar.json")

    def tearDown(self):
        self._tmp.cleanup()

    def field_file(self, name, seed=0, **prov):
        f = make_field(shape=(8, 32, 32), channels=8, seed=seed)
        f.provenance = Provenance(**{**RADAR, **prov}, source=name)
        return write_field(self.tmp / "src" / f"{name}.npz", f)


class Manifests(_Fixture):
    def test_push_records_the_digest_and_the_shared_provenance_in_the_pointer_bodys_shape(self):
        a = push(self.field_file("series-a"), self.blobs, self.manifest)
        push(self.field_file("series-b", seed=1), self.blobs, self.manifest)
        doc = json.loads(self.manifest.save().read_text())
        self.assertEqual(sorted(doc), ["body", "files", "format"])            # what provender 0.2 will read
        self.assertEqual(doc["files"]["series-a.npz"], {"digest": a["digest"], "size": a["size"]})
        self.assertEqual(doc["body"]["members"], ["series-a.npz", "series-b.npz"])
        self.assertEqual(doc["body"]["provenance"]["license"], "CC-BY-NC-SA-4.0")
        again = Manifest.load(self.manifest.path)
        self.assertEqual(again.files, self.manifest.files); self.assertEqual(again.digests(), self.manifest.digests())

    def test_one_manifest_holds_one_provenance(self):
        push(self.field_file("series-a"), self.blobs, self.manifest)
        with self.assertRaises(ValueError) as cm:
            push(self.field_file("series-b", seed=1, weights="w2"), self.blobs, self.manifest)
        self.assertIn("weights", str(cm.exception)); self.assertNotIn("series-b.npz", self.manifest.files)

    def test_an_unreadable_manifest_is_an_error_never_an_empty_one(self):
        self.manifest.path.parent.mkdir(parents=True); self.manifest.path.write_text('{"format": 99, "files": {}}')
        with self.assertRaises(ValueError):
            Manifest.load(self.manifest.path)
        self.assertEqual(Manifest.load(self.tmp / "absent.json").files, {})   # absent = the first export

    def test_a_file_that_is_not_a_field_is_not_pushed(self):
        p = self.tmp / "junk.npz"; np.savez(p, x=np.zeros(3))
        with self.assertRaises(ValueError):
            push(p, self.blobs, self.manifest)
        self.assertEqual(self.blobs.entries(), [])


class Pull(_Fixture):
    def test_round_trip_and_a_second_pull_touches_no_store(self):
        push(self.field_file("series-a"), self.blobs, self.manifest)
        out = pull("series-a.npz", self.blobs, self.manifest, self.tmp / "cache")
        self.assertEqual(read_field(out).provenance.source, "series-a")
        with unittest.mock.patch.object(obstore, "get", side_effect=AssertionError("fetched again")):
            self.assertEqual(pull("series-a.npz", self.blobs, self.manifest, self.tmp / "cache"), out)

    def test_a_local_copy_that_is_not_the_named_bytes_is_replaced(self):
        push(self.field_file("series-a"), self.blobs, self.manifest)
        out = pull("series-a.npz", self.blobs, self.manifest, self.tmp / "cache")
        out.write_bytes(b"half a download")
        self.assertEqual(read_field(pull("series-a.npz", self.blobs, self.manifest, self.tmp / "cache")).provenance.source, "series-a")

    def test_unknown_names_and_vanished_blobs_say_so(self):
        with self.assertRaises(KeyError):
            pull("nope.npz", self.blobs, self.manifest, self.tmp / "cache")
        blob = push(self.field_file("series-a"), self.blobs, self.manifest)
        obstore.delete(self.blobs.store, self.blobs.path(blob["digest"]))
        with self.assertRaises(FileNotFoundError):
            pull("series-a.npz", self.blobs, self.manifest, self.tmp / "cache")


class Sweep(_Fixture):
    def test_sweep_keeps_what_the_manifest_names_and_refuses_an_empty_manifest(self):
        kept = push(self.field_file("series-a"), self.blobs, self.manifest)
        stray = self.blobs.put_bytes(b"an export that never reached the manifest")
        self.assertEqual(sweep(self.blobs, self.manifest)["deleted"], 0)       # a day's grace by default
        self.assertEqual(sweep(self.blobs, self.manifest, grace_s=0)["deleted"], 1)
        self.assertTrue(self.blobs.has(kept["digest"])); self.assertFalse(self.blobs.has(stray["digest"]))
        with self.assertRaises(EmptyKeepSet):
            sweep(self.blobs, Manifest.load(self.tmp / "absent.json"), grace_s=0)


class Opening(unittest.TestCase):
    def test_the_encoder_owns_its_own_prefix_and_a_store_must_be_named(self):
        self.assertEqual(open_blobs("radar", "memory://bucket/feldglas").prefix, "bucket/feldglas/radar/")
        with unittest.mock.patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            open_blobs("radar")
