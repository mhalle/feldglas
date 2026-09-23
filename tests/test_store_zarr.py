"""feldglas-field 0.2: a field as a duckn zarr zip (2026-09-22). What is checked: the tokens
round-trip in C order, duckn's OWN geometry code places every token where Field says it is, no
mask is ever written, the model grid is derived and cross-checked, and the file is a stored zip."""
import json, pathlib, tempfile, unittest, zipfile

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
duckn = pytest.importorskip("duckn")

from rankfield.geometry import Geometry

from feldglas import Field, Provenance
from feldglas.store import read_field, read_meta, write_field
from conftest import KERNELS, make_field


def oblique_field(widths=(128, 256, 320)) -> Field:
    """Mixed widths on an oblique, anisotropic grid - the case a placement bug would show in."""
    rng = np.random.default_rng(4)
    th = 0.4
    R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    rows = tuple(tuple(r) for r in (R * np.array([5.0, 1.0, 1.0])[:, None]))
    shape = (16, 64, 64)
    grid = Geometry(shape=shape, directions=rows, origin=(-120.0, 35.5, 410.25))
    toks = [rng.standard_normal((int(np.prod([s // k for s, k in zip(shape, kk)])), w)).astype(np.float32)
            for kk, w in zip(KERNELS, widths)]
    m = np.zeros(shape, np.uint8); m[2:10, 8:40, 8:40] = 1
    return Field(tokens=toks, kernels=KERNELS, grid=grid, native_mask=m, native_labels=("liver",),
                 provenance=Provenance(encoder="radar", code="9319f36", weights="ckpt", preprocessing="p",
                                       license="CC-BY-NC-SA-4.0", source="series-1", extra={"crop": [1, 2]}))


class ZarrStore(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.addCleanup(self.d.cleanup)
        self.f = oblique_field()
        self.p = write_field(pathlib.Path(self.d.name) / "s.zarr.zip", self.f)

    def test_round_trip_keeps_tokens_geometry_and_provenance(self):
        g = read_field(self.p)
        self.assertEqual(g.widths, self.f.widths)
        self.assertEqual(g.kernels, self.f.kernels)
        self.assertEqual(tuple(g.grid.shape), tuple(self.f.grid.shape))
        np.testing.assert_allclose(g.grid.directions, self.f.grid.directions, atol=1e-9)
        np.testing.assert_allclose(g.grid.origin, self.f.grid.origin, atol=1e-6)
        self.assertEqual(g.provenance, self.f.provenance)
        for a, b in zip(g.tokens, self.f.tokens):
            np.testing.assert_array_equal(a, b.astype(np.float16))
        self.assertEqual(read_meta(self.p)["provenance"]["license"], "CC-BY-NC-SA-4.0")

    def test_no_mask_is_ever_written(self):
        self.assertIsNotNone(self.f.native_mask)                 # the Field had one
        with zipfile.ZipFile(self.p) as z:
            names = z.namelist()
            self.assertFalse(any("mask" in n for n in names), names)
            text = " ".join(z.read(n).decode() for n in names if n.endswith("zarr.json"))
        self.assertNotIn("native", text); self.assertNotIn("liver", text)
        g = read_field(self.p)
        self.assertIsNone(g.native_mask); self.assertEqual(g.native_labels, ())

    def test_duckn_itself_places_every_token_where_the_field_says(self):
        from duckn import DucknMetadata
        from duckn.models import validate_against_shape
        from duckn.spatial import VolumeGeometry
        root = zarr.open_group(store=zarr.storage.ZipStore(str(self.p), mode="r"), mode="r")
        for j in range(self.f.lattices):
            arr = root[f"lattice_{j}"]
            meta = DucknMetadata(**arr.attrs.asdict()["duckn"])
            validate_against_shape(meta, arr.shape)
            vg = VolumeGeometry.from_metadata(meta, arr.shape)
            idx = np.stack(np.meshgrid(*[np.arange(s) for s in arr.shape[:3]], indexing="ij"), -1).reshape(-1, 3)
            np.testing.assert_allclose(vg.index_to_world(idx), self.f.token_centers(j), atol=1e-6,
                                       err_msg=f"lattice {j}")

    def test_c_order_puts_one_tokens_embedding_contiguous(self):
        root = zarr.open_group(store=zarr.storage.ZipStore(str(self.p), mode="r"), mode="r")
        arr = root["lattice_2"]
        self.assertEqual(arr.shape, (*self.f.lattice_shape(2), 320))
        self.assertEqual(arr.chunks[-1], 320)                    # a chunk never splits a token
        z, y, x = 1, 3, 5
        flat = np.ravel_multi_index((z, y, x), self.f.lattice_shape(2))
        np.testing.assert_array_equal(arr[z, y, x, :], self.f.tokens[2][flat].astype(np.float16))

    def test_a_stored_zip_and_duckn_metadata_through_its_own_models(self):
        with zipfile.ZipFile(self.p) as z:
            self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist()))
            root = json.loads(z.read("zarr.json"))
        ext = root["attributes"]["duckn"]["extensions"]["feldglas"]
        self.assertEqual((ext["version"], ext["format_version"], ext["encoder"]), ("0.1", "0.2", "radar"))
        self.assertEqual(ext["lattices"], ["lattice_0", "lattice_1", "lattice_2"])

    def test_lattices_that_disagree_about_the_patient_are_refused(self):
        bad = pathlib.Path(self.d.name) / "bad.zarr.zip"
        with zipfile.ZipFile(self.p) as zin, zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as zout:
            for info in zin.infolist():
                data = zin.read(info)
                if info.filename == "lattice_1/zarr.json":
                    doc = json.loads(data)
                    doc["attributes"]["duckn"]["space_origin"][2] += 2.5      # half a slice off
                    data = json.dumps(doc).encode()
                zout.writestr(info.filename, data)
        with self.assertRaisesRegex(ValueError, "disagree about where the patient is"):
            read_field(bad)

    def test_a_field_without_exact_geometry_is_not_written_as_0_2(self):
        f = make_field(); f.exact_geometry = False
        with self.assertRaises(ValueError):
            write_field(pathlib.Path(self.d.name) / "pilot.zarr.zip", f)

    def test_the_npz_form_still_reads_and_writes(self):
        f = make_field()
        q = write_field(pathlib.Path(self.d.name) / "old.npz", f)
        self.assertEqual(read_meta(q)["version"], "0.1")
        self.assertEqual(read_field(q).widths, f.widths)
