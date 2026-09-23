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

    def test_a_missing_file_says_so(self):
        with self.assertRaisesRegex(FileNotFoundError, "no such field"):
            fc.open_field(self.d / "nothing.zarr.zip")

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

    def exact(self, j):
        return tuple(2 * k for k in self.f.kernels[j])          # points at voxel centers: occupancy is exact

    def test_occupancy_on_any_grid_is_the_gates_on_the_model_grid(self):
        for j, lat in enumerate(self.v.lattices):
            got = fc.occupancy(lat, model_mask(self.f, self.m), points_per_axis=self.exact(j))
            np.testing.assert_allclose(got, gate_occupancy(self.m, self.f.kernels[j]), atol=1e-6, err_msg=f"lattice {j}")

    def test_a_mask_on_a_finer_transposed_grid_is_read_exactly(self):
        g = self.f.grid
        D = np.asarray(g.directions, float)
        A = np.eye(4); A[:3, :3] = np.array([D[2] / 2, D[1] / 2, D[0] / 2]).T       # (x, y, z), twice as fine
        A[:3, 3] = np.asarray(g.origin) - (D[0] + D[1] + D[2]) / 4
        fine = np.repeat(np.repeat(np.repeat(self.m.transpose(2, 1, 0), 2, 0), 2, 1), 2, 2)
        want = gate_occupancy(self.m, self.f.kernels[-1])
        np.testing.assert_allclose(fc.occupancy(self.v.fine, fc.Mask(fine, A), points_per_axis=self.exact(2)), want, atol=1e-6)
        self.assertLess(float(np.abs(fc.occupancy(self.v.fine, fc.Mask(fine, A)) - want).mean()), 0.02)

    def test_touch_never_misses_a_touched_token_at_default_sampling(self):
        """The first sampler sent mask voxels to tokens at a stride and missed 3-33 % of an organ's
        touching tokens (a single voxel was often assigned to none)."""
        occ = gate_occupancy(self.m, self.f.kernels[-1])
        self.assertEqual(set(fc.tokens_in(self.v.fine, model_mask(self.f, self.m), "touch")),
                         set(np.flatnonzero(self.v.fine.in_extent & (occ > 0))))
        rng = np.random.default_rng(3)
        for _ in range(25):                                    # a single voxel, anywhere, on every lattice
            one = np.zeros(self.f.grid.shape, bool)
            one[tuple(rng.integers(0, n) for n in one.shape)] = True
            for j, lat in enumerate(self.v.lattices):
                want = set(np.flatnonzero(lat.in_extent & (gate_occupancy(one, self.f.kernels[j]) > 0)))
                self.assertEqual(set(fc.tokens_in(lat, model_mask(self.f, one), "touch")), want, f"lattice {j}")

    def test_a_cropped_mask_array_counts_the_box_outside_it_as_outside(self):
        """A mask array covering only part of a token's box: the rest is NOT the mask. The first
        sampler divided by the samples it found, and a sliver read as 1.0."""
        box = np.zeros(self.f.grid.shape, bool); box[4:7, 10:17, 20:23] = True
        want = gate_occupancy(box, self.f.kernels[-1])
        g = self.f.grid
        D = np.asarray(g.directions, float)
        crop = fc.Mask.from_grid(np.ones((3, 7, 3), bool), D, np.asarray(g.origin) + np.array([4, 10, 20]) @ D)
        got = fc.occupancy(self.v.fine, crop, points_per_axis=self.exact(2))
        np.testing.assert_allclose(got, want, atol=1e-6)
        self.assertLess(got.max(), 0.5)

    def test_a_mask_coarser_than_the_tokens_reads_the_tokens_inside_it(self):
        g = self.f.grid
        D = np.asarray(g.directions, float)
        coarse = fc.Mask.from_grid(np.ones((3, 5, 5), bool), D * 6, np.asarray(g.origin) + 2.5 * D.sum(0))   # 6-voxel cells
        occ = fc.occupancy(self.v.fine, coarse)
        truth = np.zeros(g.shape, bool); truth[:18, :30, :30] = True                                        # the same region
        want = gate_occupancy(truth, self.f.kernels[-1])
        np.testing.assert_allclose(occ[want == 1], 1.0)
        self.assertTrue((occ[want == 0] == 0).all())

    def test_rules_center_touch_occupancy(self):
        lat = self.v.fine
        occ = gate_occupancy(self.m.astype(float), self.f.kernels[-1])
        mask = model_mask(self.f, self.m)
        touch = set(fc.tokens_in(lat, mask, "touch"))
        half = set(fc.tokens_in(lat, mask, "occupancy", 0.5, points_per_axis=self.exact(2)))
        self.assertEqual(touch, set(np.flatnonzero(lat.in_extent & (occ > 0))))
        self.assertEqual(half, set(np.flatnonzero(lat.in_extent & (occ >= 0.5))))
        self.assertTrue(half <= touch)
        solid = np.zeros(self.f.grid.shape, np.uint8); solid[2:12, 8:48, 8:32] = 1     # token-aligned: no center on a face
        occ = gate_occupancy(solid.astype(float), self.f.kernels[-1])
        center = set(fc.tokens_in(lat, model_mask(self.f, solid), "center"))
        self.assertEqual(center, set(np.flatnonzero(lat.in_extent & (occ == 1))))

    def test_exactly_half_a_token_is_owned_at_min_occupancy_one_half(self):
        half = np.zeros(self.f.grid.shape, bool); half[6:8, 16:20, 16:24] = True       # 2 x 4 x 8 of a 2 x 8 x 8 token
        lat = self.v.fine
        t = int(np.ravel_multi_index((3, 2, 2), lat.shape))
        mask = model_mask(self.f, half)
        self.assertAlmostEqual(float(fc.occupancy(lat, mask, points_per_axis=self.exact(2))[t]), 0.5, places=6)
        self.assertEqual(list(fc.tokens_in(lat, mask, "occupancy", 0.5, points_per_axis=self.exact(2))), [t])
        self.assertEqual(list(fc.tokens_in(lat, mask, "occupancy", 0.51, points_per_axis=self.exact(2))), [])

    def test_a_fractional_mask_is_averaged_and_a_label_map_is_refused(self):
        frac = np.where(self.m, 0.3, 0.0)
        occ = fc.occupancy(self.v.fine, model_mask(self.f, frac), points_per_axis=self.exact(2))
        np.testing.assert_allclose(occ, 0.3 * gate_occupancy(self.m, self.f.kernels[-1]), atol=1e-6)
        self.assertEqual(len(fc.tokens_in(self.v.fine, model_mask(self.f, frac), "center")), 0)   # 0.3 < a half
        with self.assertRaisesRegex(ValueError, "structure_mask"):
            model_mask(self.f, self.m.astype(np.uint8) * 7)
        with self.assertRaisesRegex(ValueError, "3-D"):
            fc.Mask(np.ones((4, 4)), np.eye(4))
        with self.assertRaisesRegex(ValueError, "non-singular"):
            fc.Mask(np.ones((4, 4, 4)), np.diag([1.0, 1.0, 0.0, 1.0]))

    def test_a_point_on_a_voxel_face_goes_to_the_higher_voxel_everywhere(self):
        """floor(x + 0.5), not numpy's rint, which sends halves to the EVEN neighbor - alternate
        voxels, a half-voxel skew that depended on the index."""
        values = np.zeros((8, 8, 8), np.float32); values[3, 3, 3] = 1; values[5, 5, 5] = 1
        m = fc.Mask(values, np.eye(4))
        self.assertEqual(float(m.at(np.array([[2.5, 2.5, 2.5]]))[0]), 1.0)     # rint -> 2 (empty)
        self.assertEqual(float(m.at(np.array([[4.5, 4.5, 4.5]]))[0]), 1.0)     # rint -> 4 (empty)

    def test_labels_at_rounds_and_answers_zero_outside_the_map(self):
        lat = self.v.fine
        c = lat.centers[0]
        D = np.eye(3)
        values = np.zeros((6, 6, 6), np.uint8); values[3, 3, 3] = 5
        m = LabelMap(values, np.block([[D, (c - 2.7 * np.ones(3))[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]), {"x": 5})
        self.assertEqual(int(fc.labels_at(lat, m)[0]), 5)          # center at index 2.7 -> voxel 3
        self.assertTrue((fc.labels_at(lat, m)[1:] == 0).all())     # every other token: outside this small map


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


def rewrite(src: pathlib.Path, member: str, edit, dst=None) -> pathlib.Path:
    """A copy of a field with one zip member's duckn attributes edited (``edit`` gets the attrs)."""
    dst = dst or src.with_name("edited-" + src.name)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            data = zin.read(info)
            if info.filename == member:
                doc = json.loads(data); edit(doc["attributes"]["duckn"]); data = json.dumps(doc).encode()
            zout.writestr(info.filename, data)
    return dst


def box_field() -> Field:
    f = oblique_field()
    f.data_box = ((0, 0, 0), (13, 60, 64))
    return f


class Refusals(unittest.TestCase):
    """Every lie the store's own reader refuses, the client refuses too - by a message that names it.
    (The first client read a reversed member list and made its COARSEST lattice ``fine``.)"""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.p = write_field(self.d / "f.zarr.zip", box_field(), token_dtype=np.int8)

    def refused(self, member, edit, message):
        bad = rewrite(self.p, member, edit)
        with self.assertRaisesRegex(ValueError, message):
            fc.open_field(bad)
        return bad

    def test_format_extension_and_version_gates(self):
        self.refused("zarr.json", lambda d: d["extensions"]["embedding"].update(version="0.2"), "knows")
        self.refused("zarr.json", lambda d: d["extensions"]["embedding"].update(format="other"), "knows")
        self.refused("zarr.json", lambda d: d["extensions"].pop("embedding"), "no 'embedding' extension")

    def test_axes_space_and_placement(self):
        self.refused("lattice_1/zarr.json", lambda d: d["axes"][3].update(kind="space"), "three space axes")
        self.refused("lattice_1/zarr.json", lambda d: d.update(space="right-anterior-superior"), "left-posterior-superior")
        self.refused("lattice_1/zarr.json", lambda d: d["axes"][0].pop("space_direction"), "space_direction")
        self.refused("lattice_1/zarr.json", lambda d: d["axes"][0].update(space_direction=[0.0, 0.0, 0.0]), "singular")
        self.refused("lattice_1/zarr.json", lambda d: d.update(space_origin=[float("nan"), 0.0, 0.0]), "not finite")
        self.refused("lattice_2/zarr.json", lambda d: d["space_origin"].__setitem__(2, d["space_origin"][2] + 30), "disagree")

    def test_the_value_transform(self):
        self.refused("lattice_0/zarr.json", lambda d: d["value_transforms"][0]["parameters"].update(axis=0), "does not fit")
        self.refused("lattice_0/zarr.json", lambda d: d["value_transforms"][0]["parameters"]["slope"].pop(), "does not fit")
        self.refused("lattice_0/zarr.json", lambda d: d["value_transforms"][0]["parameters"]["intercept"].pop(), "does not fit")
        self.refused("lattice_0/zarr.json", lambda d: d["value_transforms"][0]["parameters"]["slope"].__setitem__(0, float("nan")),
                     "not finite")
        self.refused("lattice_0/zarr.json", lambda d: d.pop("value_transforms"), "not tokens")

    def test_members_extents_and_the_data_box(self):
        def reverse(d):
            d["extensions"]["embedding"]["group"]["members"].reverse()
        self.refused("zarr.json", reverse, "member")
        self.refused("zarr.json", lambda d: d["extensions"]["embedding"]["group"]["members"].pop(), "member")
        self.refused("zarr.json", lambda d: d["extensions"]["embedding"]["group"]["members"].append("lattice_9"), "no such array")
        self.refused("lattice_2/zarr.json", lambda d: d["extensions"]["embedding"]["extent"].update(hi=[99, 99, 99]), "inside the lattice")
        self.refused("lattice_2/zarr.json", lambda d: d["extensions"]["embedding"]["extent"].update(lo=[7, 0, 0], hi=[7, 8, 8]), "inside the lattice")
        self.refused("lattice_2/zarr.json", lambda d: d["extensions"]["embedding"]["extent"]["hi"].__setitem__(0, 6), "data box")
        self.refused("lattice_2/zarr.json", lambda d: d["extensions"]["embedding"].pop("extent"), "data box")

    def test_the_store_refuses_the_same(self):
        from feldglas.store import read_field
        for member, edit, message in (
                ("zarr.json", lambda d: d["extensions"]["embedding"].update(version="9"), "extension"),
                ("lattice_1/zarr.json", lambda d: d["extensions"]["embedding"].update(version="9"), "extension"),
                ("zarr.json", lambda d: d["extensions"]["embedding"].update(format_version="0.1"), "knows"),
                ("lattice_0/zarr.json", lambda d: d["value_transforms"][0]["parameters"]["intercept"].pop(), "does not fit"),
                ("lattice_0/zarr.json", lambda d: d["value_transforms"][0]["parameters"]["slope"].__setitem__(0, float("inf")), "not finite"),
                ("lattice_0/zarr.json", lambda d: d.pop("value_transforms"), "not tokens"),
                ("zarr.json", lambda d: d["extensions"]["embedding"]["provenance"].update(encoder="other"), "model or weights"),
                ("lattice_2/zarr.json", lambda d: d["extensions"]["embedding"].pop("extent"), "extent"),
                ("zarr.json", lambda d: d["extensions"]["embedding"]["group"]["members"].append("lattice_9"), "no such array"),
                ("lattice_1/zarr.json", lambda d: d["axes"][0].pop("space_direction"), "space_direction")):
            with self.subTest(member=member, message=message), self.assertRaisesRegex(ValueError, message):
                read_field(rewrite(self.p, member, edit))

    def test_the_store_never_writes_what_it_could_not_read_back(self):
        f = oblique_field(); f.tokens[1][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "not finite"):
            write_field(self.d / "nan.zarr.zip", f)
        g = oblique_field(); g.tokens[0][0, 0] = 1e5
        with self.assertRaisesRegex(ValueError, "range"):
            write_field(self.d / "big.zarr.zip", g)                  # fp16 would store inf
        write_field(self.d / "big32.zarr.zip", g, token_dtype=np.float32)


class Metadata(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())

    def test_what_each_lattice_says_about_itself(self):
        f = box_field()
        v = fc.open_field(write_field(self.d / "f.zarr.zip", f))
        e = f.embedding
        for j, lat in enumerate(v.lattices):
            self.assertEqual(lat.thickness, e.receptive_mm[j])
            self.assertEqual(lat.support_offset, e.support_offset_mm[j])
            self.assertEqual((lat.metric, lat.normalized), (e.metric, e.normalized))
        self.assertEqual(v.data_box, {"lo": [0, 0, 0], "hi": [13, 60, 64]})

    def test_fine_is_the_smallest_token_and_unnamed_layers_are_told_apart(self):
        from conftest import make_field
        v = fc.open_field(write_field(self.d / "m.zarr.zip", make_field(mask=False)))
        self.assertIs(v.fine, v.lattices[2])
        self.assertEqual(len({l.key() for l in v.lattices}), 3)       # no layer names: kernels keep them apart


class Boundaries(unittest.TestCase):
    def test_drop_boundary_leaves_the_extents_outer_rows_out_of_a_reference(self):
        d = pathlib.Path(tempfile.mkdtemp())
        normals = []
        for i in range(3):
            f = cluster_field(i); f.data_box = ((0, 0, 0), (14, 56, 56))
            normals.append(fc.open_field(write_field(d / f"n{i}.zarr.zip", f)))
        g = cluster_field(0).grid
        region = fc.Mask.from_grid(np.ones(g.shape, bool), g.directions, g.origin)
        lat = normals[0].fine
        kept = fc.tokens_in(lat, region, "occupancy", 0.5, drop_boundary=True)
        self.assertEqual(set(kept), set(np.flatnonzero(lat.interior)))
        self.assertLess(len(kept), int(lat.in_extent.sum()))
        ref = fc.Reference.build([(n, region) for n in normals])
        self.assertEqual(ref.meta["tokens_per_normal"], [int(n.fine.interior.sum()) for n in normals])


class Pieces(unittest.TestCase):
    def test_knn_distance_by_hand(self):
        ref = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
        x = np.array([[1.0, 0.0]], np.float32)
        self.assertAlmostEqual(float(fc.knn_distance(x, ref, k=1)[0]), 0.0)
        self.assertAlmostEqual(float(fc.knn_distance(x, ref, k=2)[0]), 0.5)
        for k in (0, 3):
            with self.assertRaisesRegex(ValueError, "k ="):
                fc.knn_distance(x, ref, k=k)

    def test_unit_refuses_what_has_no_direction(self):
        with self.assertRaisesRegex(ValueError, "no direction"):
            fc.unit(np.array([[1.0, 0.0], [0.0, 0.0]]))
        with self.assertRaisesRegex(ValueError, "no direction"):
            fc.unit(np.array([[np.nan, 1.0]]))

    def test_sites_are_single_linkage_strongest_first_with_their_peak(self):
        c = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [100, 0, 0], [105, 0, 0]], float)
        dist = np.array([0.5, 0.6, 0.4, 0.9, 0.8])
        sites = fc._sites(c, dist, np.ones(5, bool), 15.0)
        self.assertEqual([s.tokens for s in sites], [2, 3])            # strongest (0.9) first; the chain 0-10-20 is one
        self.assertEqual([s.peak for s in sites], [0.9, 0.6])
        np.testing.assert_allclose(sites[1].center, [10, 0, 0])
        self.assertEqual(len(fc._sites(c, dist, np.ones(5, bool), 9.0)), 4)          # 0 | 10 | 20 | 100-105


class OrganVectorRules(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.f = box_field()
        self.v = fc.open_field(write_field(self.d / "f.zarr.zip", self.f))
        self.A = np.eye(4); self.A[:3, :3] = np.asarray(self.f.grid.directions, float).T; self.A[:3, 3] = self.f.grid.origin

    def test_tokens_past_the_extent_are_not_the_organs_or_the_bodys(self):
        lab = np.zeros(self.f.grid.shape, np.uint8)
        lab[2:10, 8:40, 8:40] = 1
        lab[14:16, 8:40, 8:40] = 2                                     # past data_box hi (13) in z: padding only
        lab[2:10, 44:60, 44:60] = 3                                    # a second in-extent structure: the body is not the liver alone
        labels = LabelMap(lab, self.A, {"liver": 1, "padding": 2, "spleen": 3})
        vec = fc.organ_vectors(self.v, labels, min_tokens=1)
        lat = self.v.fine
        at = fc.labels_at(lat, labels)
        self.assertFalse((lat.in_extent & (at == 2)).any())
        self.assertNotIn("padding", vec)
        t = fc.unit(lat.tokens)                                        # and the body's mean is in-extent only
        want = t[lat.in_extent & (at == 1)].mean(0) - t[lat.in_extent & (at > 0)].mean(0)
        np.testing.assert_allclose(vec["liver"], want / np.linalg.norm(want), atol=1e-5)

    def test_min_tokens_is_inclusive_and_a_lone_structure_has_no_direction(self):
        lat = self.v.fine
        lab = np.zeros(self.f.grid.shape, np.uint8); lab[2:10, 8:40, 8:40] = 1; lab[2:4, 40:48, 40:48] = 2
        labels = LabelMap(lab, self.A, {"liver": 1, "small": 2})
        n = int((lat.in_extent & (fc.labels_at(lat, labels) == 2)).sum())
        self.assertIn("small", fc.organ_vectors(self.v, labels, min_tokens=n))
        self.assertNotIn("small", fc.organ_vectors(self.v, labels, min_tokens=n + 1))
        only = LabelMap((lab == 1).astype(np.uint8), self.A, {"liver": 1})
        self.assertEqual(fc.organ_vectors(self.v, only), {})             # it IS the body: no direction, not NaN
        with self.assertRaisesRegex(ValueError, "no in-extent token"):
            fc.organ_vectors(self.v, LabelMap(np.zeros_like(lab), self.A, {"liver": 1}))


class ScoringEdges(unittest.TestCase):
    def test_an_empty_region_is_refused_not_called_normal(self):
        d = pathlib.Path(tempfile.mkdtemp())
        normals = [fc.open_field(write_field(d / f"n{i}.zarr.zip", cluster_field(i))) for i in range(3)]
        g = cluster_field(0).grid
        region = fc.Mask.from_grid(np.ones(g.shape, bool), g.directions, g.origin)
        ref = fc.Reference.build([(n, region) for n in normals])
        far = fc.Mask.from_grid(np.ones((2, 2, 2), bool), np.eye(3), (5000.0, 5000.0, 5000.0))
        with self.assertRaisesRegex(ValueError, "nothing was scored"):
            ref.score(normals[0], far)


class SegNrrdInputs(unittest.TestCase):
    """Masks from .seg.nrrd files as they arrive: haversack's (one layer, LPS) and 3D Slicer's
    (layered when segments overlap; RAS or LPS) - the same regions and vectors either way."""

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.f = box_field()
        self.v = fc.open_field(write_field(self.d / "f.zarr.zip", self.f))
        g = self.f.grid
        D = np.asarray(g.directions, float)
        self.A = np.eye(4); self.A[:3, :3] = np.array([D[2], D[1], D[0]]).T; self.A[:3, 3] = g.origin     # (x, y, z) axes
        organ = np.zeros((64, 64, 16), np.uint8); organ[8:40, 8:50, 2:12] = 1; organ[44:60, 10:40, 2:12] = 2
        lesion = np.zeros_like(organ); lesion[16:26, 20:30, 4:8] = 1                                     # inside organ 1
        self.organ, self.lesion = organ, lesion

    def test_a_lesion_on_its_own_layer_joins_its_organ_by_name(self):
        p = write_seg(self.d / "s.seg.nrrd", np.stack([self.organ, self.lesion]), self.A,
                      {"liver": 1, "spleen": 2, "liver_lesion": 1}, layers={"liver": 0, "spleen": 0, "liver_lesion": 1},
                      space="right-anterior-superior")
        region = fc.seg_mask(p, "liver", "liver_lesion")
        np.testing.assert_array_equal(region.values, (self.organ == 1) | (self.lesion == 1))
        lat = self.v.fine
        with self.assertRaisesRegex(ValueError, "layered"):
            fc.labels_at(lat, fc.read_seg_nrrd(p))
        vec = fc.organ_vectors(self.v, fc.read_seg_nrrd(p), min_tokens=1)
        self.assertEqual(set(vec), {"liver", "spleen", "liver_lesion"})

    def test_haversacks_one_layer_file_and_slicers_layered_one_agree(self):
        one = fc.read_seg_nrrd(write_seg(self.d / "one.seg.nrrd", self.organ, self.A, {"liver": 1, "spleen": 2}))
        two = fc.read_seg_nrrd(write_seg(self.d / "two.seg.nrrd", np.stack([self.organ, self.lesion]), self.A,
                                         {"liver": 1, "spleen": 2, "liver_lesion": 1},
                                         layers={"liver": 0, "spleen": 0, "liver_lesion": 1}, space="right-anterior-superior"))
        np.testing.assert_allclose(two.affine_lps, one.affine_lps)
        a = fc.organ_vectors(self.v, one, min_tokens=1)
        b = fc.organ_vectors(self.v, two, min_tokens=1, structures={"liver", "spleen"})
        c = fc.organ_vectors(self.v, one, min_tokens=1, structures={"liver", "spleen"})
        np.testing.assert_allclose(b["liver"], c["liver"], atol=1e-6)       # the lesion is inside the organ: same body
        self.assertEqual(set(a), {"liver", "spleen"})
        lat = self.v.fine
        np.testing.assert_array_equal(fc.tokens_in(lat, fc.structure_mask(one, "liver"), "touch"),
                                      fc.tokens_in(lat, fc.structure_mask(two, "liver"), "touch"))

    def test_a_lesion_segment_the_scan_does_not_have_is_optional_the_organ_is_not(self):
        p = write_seg(self.d / "h.seg.nrrd", self.organ, self.A, {"liver": 1, "spleen": 2})    # no lesion listed
        m = fc.seg_mask(p, "liver", optional=["liver_lesion", "liver_cyst*"])
        np.testing.assert_array_equal(m.values, self.organ == 1)
        with self.assertRaises(KeyError):
            fc.seg_mask(p, "liver", "liver_lesion")                        # required and absent: refused by name
        with self.assertRaisesRegex(ValueError, "none is in this map"):
            fc.seg_mask(p, optional=["liver_lesion"])
        both = write_seg(self.d / "b.seg.nrrd", np.stack([self.organ, self.lesion]), self.A,
                         {"liver": 1, "spleen": 2, "liver_lesion": 1}, layers={"liver": 0, "spleen": 0, "liver_lesion": 1})
        np.testing.assert_array_equal(fc.seg_mask(both, "liver", optional=["liver_lesion"]).values,
                                      (self.organ == 1) | (self.lesion == 1))

    def test_a_plain_labelmap_takes_its_names_from_the_caller(self):
        p = write_seg(self.d / "plain.nrrd", self.organ, self.A, {})                  # no segment table at all
        self.assertEqual(fc.read_seg_nrrd(p).names, {})
        m = fc.seg_mask(p, "liver", names={"liver": 1, "spleen": 2})
        np.testing.assert_array_equal(m.values, self.organ == 1)
        named = write_seg(self.d / "named.seg.nrrd", self.organ, self.A, {"liver": 1, "spleen": 2})
        with self.assertRaisesRegex(ValueError, "names its segments"):
            fc.read_seg_nrrd(named, names={"liver": 1})
        layered = write_seg(self.d / "lay.seg.nrrd", np.stack([self.organ, self.lesion]), self.A, {}, layers={})
        with self.assertRaisesRegex(ValueError, "layered"):
            fc.read_seg_nrrd(layered, names={"liver": 1})

    def test_organ_vectors_read_each_structure_on_its_own_layer_and_the_body_on_all(self):
        outside = np.zeros_like(self.organ); outside[44:60, 44:60, 2:12] = 1          # a lesion OUTSIDE every layer-0 structure
        lesion = np.maximum(self.lesion, outside)
        p = write_seg(self.d / "v.seg.nrrd", np.stack([self.organ, lesion]), self.A,
                      {"liver": 1, "spleen": 2, "liver_lesion": 1}, layers={"liver": 0, "spleen": 0, "liver_lesion": 1})
        lm = fc.read_seg_nrrd(p)
        vec = fc.organ_vectors(self.v, lm, min_tokens=1)
        lat = self.v.fine
        t = fc.unit(lat.tokens)
        at = lambda m: fc.Mask(m.astype(bool), lm.affine_lps).at(lat.centers) >= 0.5
        body = lat.in_extent & (at(self.organ > 0) | at(lesion > 0))                    # the body is every layer's
        for name, m in (("liver", self.organ == 1), ("liver_lesion", lesion == 1)):
            want = t[lat.in_extent & at(m)].mean(0) - t[body].mean(0)
            np.testing.assert_allclose(vec[name], want / np.linalg.norm(want), atol=1e-5, err_msg=name)

    def test_a_malformed_layered_file_is_refused(self):
        p = write_seg(self.d / "m.seg.nrrd", np.stack([self.organ, self.lesion]), self.A,
                      {"liver": 1, "liver_lesion": 1}, layers={"liver": 0, "liver_lesion": 1})
        raw = p.read_bytes()
        for old, new, msg in ((b"kinds: list domain", b"kinds: domain domain", "list"),
                              (b"Segment1_Layer:=1", b"Segment1_Layer:=2", "layer 2 of 2")):
            bad = self.d / "bad.seg.nrrd"; bad.write_bytes(raw.replace(old, new, 1))
            with self.assertRaisesRegex(ValueError, msg):
                fc.read_seg_nrrd(bad)
