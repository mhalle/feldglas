"""feldglas-field 0.2: a field as a duckn zarr zip (2026-09-22). What is checked: the tokens
round-trip in C order, duckn's OWN geometry code places every token where Field says it is, no
mask is ever written, the model grid is derived and cross-checked, the file is a stored zip, and
the metadata is docs/embedding-field.md's: ``thickness``, ``intent``, the ``embedding`` extension,
the input CT in the provenance."""
import json, pathlib, tempfile, unittest, zipfile

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
duckn = pytest.importorskip("duckn")

from rankfield.geometry import Geometry

from feldglas import Embedding, Field, Provenance
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
                 data_box=((0, 0, 0), (13, 60, 64)),                  # padded at the end, as the encoders pad
                 provenance=Provenance(encoder="radar", code="9319f36", weights="ckpt", preprocessing="p",
                                       license="CC-BY-NC-SA-4.0", source="series-1", extra={"crop": [1, 2]},
                                       input={"identity": {"series": "series-1"},
                                              "grid": {"shape": [512, 512, 90], "origin": [1.0, 2.0, 3.0],
                                                       "directions": [[0.7, 0, 0], [0, 0.7, 0], [0, 0, 2.5]]}}),
                 embedding=Embedding(layers=("deep", "mid", "fine"), receptive_mm=((80.0, 80.0, 80.0), None, (20.0, 21.0, 22.0)),
                                     support_offset_mm=((6.6, 6.3, 7.6), (2.8, 2.2, 2.7), None)))


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

    def test_a_stored_zip_and_the_records_metadata(self):
        with zipfile.ZipFile(self.p) as z:
            self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist()))
            root = json.loads(z.read("zarr.json"))["attributes"]["duckn"]
            arr = json.loads(z.read("lattice_2/zarr.json"))["attributes"]["duckn"]
        self.assertEqual(set(root["extensions"]), {"embedding"})
        ext = root["extensions"]["embedding"]
        self.assertEqual((ext["version"], ext["format_version"], root["intent"]), ("0.1", "0.2", "embedding-field"))
        self.assertEqual(ext["group"], {"id": "radar/series-1", "members": ["lattice_0", "lattice_1", "lattice_2"]})
        self.assertEqual(ext["provenance"]["input"]["grid"]["shape"], [512, 512, 90])
        a = arr["extensions"]["embedding"]
        self.assertEqual(arr["intent"], "embedding-field")
        self.assertEqual(a["space"], {"model": "radar", "weights": "ckpt", "layer": "fine", "stage": "raw"})
        self.assertEqual((a["metric"], a["normalized"], a["kernel"]), ("cosine", False, [2, 8, 8]))
        self.assertEqual(a["group"], {"id": "radar/series-1", "member": 2, "members": 3})
        self.assertNotIn("support", a)                           # not known for this lattice: not stated
        self.assertNotIn("projects_to", a)
        self.assertEqual([x.get("thickness") for x in arr["axes"]], [20.0, 21.0, 22.0, None])

    def test_each_lattice_says_which_tokens_saw_the_scan(self):
        root = zarr.open_group(store=zarr.storage.ZipStore(str(self.p), mode="r"), mode="r")
        ext = root.attrs.asdict()["duckn"]["extensions"]["embedding"]
        self.assertEqual(ext["data_box"], {"lo": [0, 0, 0], "hi": [13, 60, 64]})
        # kernels (8,32,32), (4,16,16), (2,8,8): a token counts if its box holds ANY scanned voxel
        want = [([0, 0, 0], [2, 2, 2]), ([0, 0, 0], [4, 4, 4]), ([0, 0, 0], [7, 8, 8])]
        for j, (lo, hi) in enumerate(want):
            e = root[f"lattice_{j}"].attrs.asdict()["duckn"]["extensions"]["embedding"]["extent"]
            self.assertEqual((e["lo"], e["hi"]), (lo, hi), f"lattice {j}")
        self.assertEqual(read_field(self.p).data_box, self.f.data_box)

    def test_a_data_box_outside_the_model_grid_is_refused(self):
        f = oblique_field()
        for box in (((0, 0, 0), (17, 64, 64)), ((5, 0, 0), (5, 64, 64)), ((-1, 0, 0), (4, 4, 4))):
            with self.assertRaisesRegex(ValueError, "data box"):
                Field(tokens=f.tokens, kernels=f.kernels, grid=f.grid, provenance=f.provenance, data_box=box)

    def test_an_extent_that_disagrees_with_the_box_is_refused(self):
        def wider(d): d["extensions"]["embedding"]["extent"]["hi"][0] += 1
        with self.assertRaisesRegex(ValueError, "extent"):
            read_field(self._rewrite("lattice_2/zarr.json", wider))

    def test_the_client_guide_travels_in_the_zip(self):
        with zipfile.ZipFile(self.p) as z:
            text = z.read("README.md").decode()
            self.assertEqual(z.getinfo("README.md").compress_type, zipfile.ZIP_STORED)
        self.assertIn("Which tokens saw the scan", text)
        root = json.loads(zipfile.ZipFile(self.p).read("zarr.json"))["attributes"]["duckn"]
        self.assertTrue(root["extensions"]["embedding"]["schema"].startswith("README.md"))
        read_field(self.p)                                        # zarr ignores it

    def test_duckn_reads_thickness_on_every_space_axis(self):
        from duckn import DucknMetadata
        root = zarr.open_group(store=zarr.storage.ZipStore(str(self.p), mode="r"), mode="r")
        meta = DucknMetadata(**root["lattice_0"].attrs.asdict()["duckn"])
        self.assertEqual([a.thickness for a in meta.axes], [80.0, 80.0, 80.0, None])
        self.assertEqual(meta.intent, "embedding-field")

    def test_the_embedding_and_input_round_trip(self):
        g = read_field(self.p)
        self.assertEqual(g.embedding, self.f.embedding)
        self.assertEqual(g.provenance.input, self.f.provenance.input)
        self.assertEqual(read_meta(self.p)["embedding"][0]["support"]["offset"], [6.6, 6.3, 7.6])

    def test_a_field_that_knows_nothing_more_still_round_trips(self):
        f = make_field(mask=False)
        g = read_field(write_field(pathlib.Path(self.d.name) / "plain.zarr.zip", f))
        self.assertEqual(g.embedding, f.embedding)
        self.assertEqual(g.provenance, f.provenance)

    def _rewrite(self, member, edit) -> pathlib.Path:
        bad = pathlib.Path(self.d.name) / "bad.zarr.zip"
        with zipfile.ZipFile(self.p) as zin, zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as zout:
            for info in zin.infolist():
                data = zin.read(info)
                if info.filename == member:
                    doc = json.loads(data); edit(doc["attributes"]["duckn"]); data = json.dumps(doc).encode()
                zout.writestr(info.filename, data)
        return bad

    def test_a_lattice_from_another_group_or_place_is_refused(self):
        def other_id(d): d["extensions"]["embedding"]["group"]["id"] = "radar/series-2"
        def other_member(d): d["extensions"]["embedding"]["group"]["member"] = 0
        for edit in (other_id, other_member):
            with self.assertRaisesRegex(ValueError, "member"):
                read_field(self._rewrite("lattice_1/zarr.json", edit))

    def test_lattices_from_other_weights_are_refused(self):
        def other(d): d["extensions"]["embedding"]["space"]["weights"] = "another-ckpt"
        with self.assertRaisesRegex(ValueError, "disagree about model"):
            read_field(self._rewrite("lattice_1/zarr.json", other))

    def test_a_draft_with_the_old_feldglas_extension_is_refused_by_name(self):
        def old(d): d["extensions"] = {"feldglas": d["extensions"]["embedding"]}
        with self.assertRaisesRegex(ValueError, "draft 0.2"):
            read_field(self._rewrite("zarr.json", old))

    def test_lattices_that_disagree_about_the_patient_are_refused(self):
        def shift(d): d["space_origin"][2] += 2.5                    # half a slice off
        with self.assertRaisesRegex(ValueError, "disagree about where the patient is"):
            read_field(self._rewrite("lattice_1/zarr.json", shift))

    def test_a_field_without_exact_geometry_is_not_written_as_0_2(self):
        f = make_field(); f.exact_geometry = False
        with self.assertRaises(ValueError):
            write_field(pathlib.Path(self.d.name) / "pilot.zarr.zip", f)

    def test_the_npz_form_still_reads_and_writes(self):
        f = make_field()
        q = write_field(pathlib.Path(self.d.name) / "old.npz", f)
        self.assertEqual(read_meta(q)["version"], "0.1")
        self.assertEqual(read_field(q).widths, f.widths)
