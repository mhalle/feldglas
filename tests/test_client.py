"""feldglas.client: the receiving end, reading the FILE (world coordinates, no Field, no model
grid). Each piece is held to the package's own internals - the store, the gates, the labels - so
two readers of one format cannot drift apart without a test saying so."""
import json, pathlib, tempfile, unittest, zipfile

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("duckn")

from rankfield.geometry import Geometry

from feldglas import Field, Provenance
from feldglas import client as fc
from feldglas.gate import occupancy as gate_occupancy
from feldglas.labels import LabelMap
from feldglas.store import write_field
from conftest import KERNELS
from test_labels_session import write_seg
from test_store_zarr import oblique_field


def model_mask(field: Field, values) -> fc.Mask:
    """A mask on the field's own model grid, as a client would hold one on any grid."""
    return fc.Mask.from_grid(values, field.grid.directions, field.grid.origin)


class Open(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.f = oblique_field()

    def test_the_client_reads_what_the_store_writes_float_and_int8(self):
        for dtype, atol in ((np.float16, 1e-2), (np.int8, None)):
            v = fc.open_field(write_field(self.d / f"{np.dtype(dtype).name}.zarr.zip", self.f, token_dtype=dtype))
            self.assertEqual((v.encoder, v.license), ("radar", "CC-BY-NC-SA-4.0"))
            self.assertEqual(v.input_grid["shape"], [512, 512, 90])
            for j, lat in enumerate(v.lattices):
                ref = self.f.tokens[j]
                tol = atol if atol is not None else float(((ref.max(0) - ref.min(0)) / 255).max())
                np.testing.assert_allclose(lat.tokens, ref, atol=tol, err_msg=f"{dtype} lattice {j}")
                np.testing.assert_allclose(lat.centers, self.f.token_centers(j), atol=1e-6)
                lo, hi = self.f.lattice_extent(j)
                np.testing.assert_array_equal(lat.in_extent, np.all((lat.index >= lo) & (lat.index < hi), 1))
                self.assertEqual(lat.kernel, self.f.kernels[j])
                self.assertEqual(lat.key(), ("radar", "ckpt", ("deep", "mid", "fine")[j], "raw"))
            np.testing.assert_allclose(v.fine.index_of(v.fine.centers), v.fine.index)

    def test_boundary_tokens_are_the_extents_first_and_last_rows(self):
        lat = fc.open_field(write_field(self.d / "f.zarr.zip", self.f)).fine
        lo, hi = (np.asarray(b) for b in lat.extent)
        i = lat.index
        edge = lat.in_extent & np.any((i == lo) | (i == hi - 1), axis=1)
        np.testing.assert_array_equal(lat.interior, lat.in_extent & ~edge)
        self.assertTrue(edge.any() and lat.interior.any())

    def test_an_unknown_transform_or_format_is_refused(self):
        p = write_field(self.d / "q.zarr.zip", self.f, token_dtype=np.int8)
        for member, edit, msg in (("lattice_0/zarr.json", lambda d: d["value_transforms"][0].update(name="other"), "not tokens"),
                                  ("zarr.json", lambda d: d["extensions"]["embedding"].update(format_version="9.9"), "knows")):
            bad = self.d / "bad.zarr.zip"
            with zipfile.ZipFile(p) as zin, zipfile.ZipFile(bad, "w", zipfile.ZIP_STORED) as zout:
                for info in zin.infolist():
                    data = zin.read(info)
                    if info.filename == member:
                        doc = json.loads(data); edit(doc["attributes"]["duckn"]); data = json.dumps(doc).encode()
                    zout.writestr(info.filename, data)
            with self.assertRaisesRegex(ValueError, msg):
                fc.open_field(bad)


class Gates(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.f = oblique_field()
        self.v = fc.open_field(write_field(self.d / "f.zarr.zip", self.f))
        rng = np.random.default_rng(1)
        self.m = np.zeros(self.f.grid.shape, bool)
        self.m[3:11, 10:45, 5:30] = True
        self.m &= rng.random(self.m.shape) < 0.9                 # ragged, so fractions are not all 0 or 1

    def test_occupancy_on_any_grid_is_the_gates_on_the_model_grid(self):
        for j, lat in enumerate(self.v.lattices):
            got = fc.occupancy(lat, model_mask(self.f, self.m), samples_per_token=1000)   # every voxel
            np.testing.assert_allclose(got, gate_occupancy(self.m, self.f.kernels[j]), atol=1e-6, err_msg=f"lattice {j}")

    def test_a_mask_on_a_finer_transposed_grid_gives_nearly_the_same_occupancy(self):
        g = self.f.grid
        D = np.asarray(g.directions, float)
        A = np.eye(4); A[:3, :3] = np.array([D[2] / 2, D[1] / 2, D[0] / 2]).T       # (x, y, z), twice as fine
        A[:3, 3] = np.asarray(g.origin) - (D[0] + D[1] + D[2]) / 4
        fine = np.repeat(np.repeat(np.repeat(self.m.transpose(2, 1, 0), 2, 0), 2, 1), 2, 2)
        want = gate_occupancy(self.m, self.f.kernels[-1])
        exact = fc.occupancy(self.v.fine, fc.Mask(fine, A), samples_per_token=1000)   # every voxel: the geometry
        np.testing.assert_allclose(exact, want, atol=1e-6)
        sampled = fc.occupancy(self.v.fine, fc.Mask(fine, A))                        # the default sub-sample
        self.assertLess(float(np.abs(sampled - want).mean()), 0.02)

    def test_rules_center_touch_occupancy(self):
        lat = self.v.fine
        occ = gate_occupancy(self.m.astype(float), self.f.kernels[-1])
        mask = model_mask(self.f, self.m)
        touch = set(fc.tokens_in(lat, mask, "touch"))
        half = set(fc.tokens_in(lat, mask, "occupancy", 0.5))
        self.assertEqual(touch, set(np.flatnonzero(lat.in_extent & (occ > 0))))
        self.assertEqual(half, set(np.flatnonzero(lat.in_extent & (occ >= 0.5))))
        self.assertTrue(half <= touch)
        solid = np.zeros(self.f.grid.shape, np.uint8); solid[2:12, 8:48, 8:32] = 1     # token-aligned: no center on a face
        occ = gate_occupancy(solid.astype(float), self.f.kernels[-1])
        center = set(fc.tokens_in(lat, model_mask(self.f, solid), "center"))
        self.assertEqual(center, set(np.flatnonzero(lat.in_extent & (occ == 1))))


class Vectors(unittest.TestCase):
    def test_organ_vectors_follow_the_readme_recipe(self):
        d = pathlib.Path(tempfile.mkdtemp())
        f = oblique_field()
        v = fc.open_field(write_field(d / "f.zarr.zip", f))
        A = np.eye(4); A[:3, :3] = np.asarray(f.grid.directions, float).T; A[:3, 3] = f.grid.origin
        lab = np.zeros(f.grid.shape, np.uint8); lab[2:8, 8:40, 8:40] = 1; lab[8:14, 30:60, 20:50] = 2
        labels = LabelMap(lab, A, {"liver": 1, "spleen": 2})
        vec = fc.organ_vectors(v, labels, min_tokens=3)
        lat = v.fine
        at = fc.labels_at(lat, labels)
        t = fc.unit(lat.tokens)
        ref = t[lat.in_extent & (at > 0)].mean(0)
        for name, value in labels.names.items():
            want = t[lat.in_extent & (at == value)].mean(0) - ref
            np.testing.assert_allclose(vec[name], want / np.linalg.norm(want), atol=1e-5)

    def test_a_seg_nrrd_and_a_structure_by_name(self):
        d = pathlib.Path(tempfile.mkdtemp())
        f = oblique_field()
        A = np.eye(4); A[:3, :3] = np.asarray(f.grid.directions, float).T; A[:3, 3] = f.grid.origin
        lab = np.zeros(f.grid.shape, np.uint8); lab[2:8, 8:40, 8:40] = 3; lab[8:14, 30:60, 20:50] = 7
        labels = fc.read_seg_nrrd(write_seg(d / "l.seg.nrrd", lab, A, {"kidney_left": 3, "kidney_right": 7}))
        m = fc.structure_mask(labels, "kidney_*")
        np.testing.assert_array_equal(m.values, lab > 0)
        with self.assertRaises(KeyError):
            fc.structure_mask(labels, "liver")


def cluster_field(seed, anomaly=None) -> Field:
    """Tokens drawn around a few "tissue" prototypes, the same for every scan; ``anomaly`` replaces a
    block of fine tokens with another direction."""
    rng = np.random.default_rng(seed)
    protos = np.random.default_rng(99).standard_normal((4, 32))
    shape = (16, 64, 64)
    grid = Geometry(shape=shape, directions=((0.0, 0.0, 5.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)), origin=(-30.0, -40.0, 100.0))
    toks = []
    for kk in KERNELS:
        n = int(np.prod([s // k for s, k in zip(shape, kk)]))
        toks.append((protos[rng.integers(0, 4, n)] + 0.15 * rng.standard_normal((n, 32))).astype(np.float32))
    if anomaly is not None:
        lshape = tuple(s // k for s, k in zip(shape, KERNELS[-1]))
        idx = np.ravel_multi_index(tuple(np.array(anomaly).T), lshape)
        toks[-1][idx] = np.random.default_rng(7).standard_normal(32).astype(np.float32) * 3
    return Field(tokens=toks, kernels=KERNELS, grid=grid,
                 provenance=Provenance(encoder="synthetic", weights="w", license="none", source=f"s{seed}"))


class NormalTissue(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.normals = [fc.open_field(write_field(self.d / f"n{i}.zarr.zip", cluster_field(i))) for i in range(3)]
        g = cluster_field(0).grid
        self.region = fc.Mask.from_grid(np.ones(g.shape, bool), g.directions, g.origin)
        self.block = [(z, y, x) for z in (3, 4) for y in (3, 4) for x in (3, 4)]

    def test_a_planted_anomaly_is_the_strongest_site_and_normal_tissue_passes(self):
        ref = fc.Reference.build([(n, self.region) for n in self.normals])
        self.assertEqual(ref.threshold, float(ref.held_out.max()))
        u = fc.open_field(write_field(self.d / "u.zarr.zip", cluster_field(10, anomaly=self.block)))
        s = ref.score(u, self.region)
        lat = u.fine
        planted = np.ravel_multi_index(tuple(np.array(self.block).T), lat.shape)
        where = np.isin(s.index, planted)
        self.assertTrue(where.any() and s.flagged[where].all())
        center = lat.centers[planted].mean(0)
        self.assertLess(float(np.linalg.norm(s.sites[0].center - center)), 5.0)
        self.assertEqual(s.sites[0].tokens, int(where.sum()))
        clean = ref.score(fc.open_field(write_field(self.d / "c.zarr.zip", cluster_field(11))), self.region)
        self.assertLessEqual(len(clean.sites), 1)                   # the held-out maximum: rarely, never many

    def test_a_reference_refuses_one_normal_and_mixed_encoders(self):
        with self.assertRaisesRegex(ValueError, "two normal"):
            fc.Reference.build([(self.normals[0], self.region)])
        other = cluster_field(5); other.provenance = Provenance(encoder="other", weights="w", license="none", source="x")
        o = fc.open_field(write_field(self.d / "o.zarr.zip", other))
        with self.assertRaisesRegex(ValueError, "different encoders"):
            fc.Reference.build([(self.normals[0], self.region), (o, self.region)])
        ref = fc.Reference.build([(n, self.region) for n in self.normals])
        with self.assertRaisesRegex(ValueError, "cannot be scored"):
            ref.score(o, self.region)
