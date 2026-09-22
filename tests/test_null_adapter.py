"""The null model's pieces that need no weights: mixed-width fields, the lattice-by-lattice head,
the aligned tiling and its blend, the organ mapping and the geometry hand-over from haversack."""
import pathlib, tempfile, unittest

import numpy as np
from rankfield.geometry import Geometry

from feldglas import Field, Provenance
from feldglas.adapters import null, radar
from feldglas.heads import LatticeMeanHead, MeanPoolHead
from feldglas.store import read_field, write_field

SHAPE = (32, 48, 64)


def mixed_field(seed=0, widths=null.WIDTHS) -> Field:
    rng = np.random.default_rng(seed)
    grid = Geometry.aligned(SHAPE, (3.0, 3.0, 3.0))
    toks = [rng.standard_normal((int(np.prod([s // k for s, k in zip(SHAPE, kk)])), w)).astype(np.float32)
            for kk, w in zip(null.KERNELS, widths)]
    return Field(tokens=toks, kernels=list(null.KERNELS), grid=grid, provenance=null.provenance(source="t"))


class MixedWidths(unittest.TestCase):
    def test_each_lattice_gets_its_own_block_of_channels(self):
        f = mixed_field()
        self.assertFalse(f.uniform)
        self.assertEqual(f.widths, (128, 256, 320))
        self.assertEqual(f.channels, 704)
        A = f.all_tokens()
        self.assertEqual(A.shape, (int(f.offsets[-1]), 704))
        for j in range(3):
            rows = slice(f.offsets[j], f.offsets[j + 1])
            np.testing.assert_array_equal(A[rows, f.block(j)], f.tokens[j])
            others = np.delete(A[rows], np.r_[f.block(j)], axis=1)
            self.assertFalse(others.any(), f"lattice {j} leaked into another block")

    def test_a_uniform_field_is_unchanged(self):
        f = mixed_field(widths=(64, 64, 64))
        self.assertTrue(f.uniform)
        self.assertEqual(f.channels, 64)
        np.testing.assert_array_equal(f.all_tokens(), np.concatenate(f.tokens))

    def test_a_mixed_field_round_trips_through_the_store(self):
        f = mixed_field()
        with tempfile.TemporaryDirectory() as d:
            g = read_field(write_field(pathlib.Path(d) / "f.npz", f))
        self.assertEqual(g.widths, f.widths)
        self.assertEqual(g.provenance.license, "Apache-2.0")
        self.assertEqual(g.provenance.encoder, null.NAME)


class LatticeMean(unittest.TestCase):
    def setUp(self):
        self.f = mixed_field()
        self.h = LatticeMeanHead.for_field(self.f)
        self.P = self.h.prepare(self.f.all_tokens())

    def expected(self, index, weights=None):
        parts = []
        for j in range(3):
            rows = [i - self.f.offsets[j] for i in index if self.f.offsets[j] <= i < self.f.offsets[j + 1]]
            if not rows:
                parts.append(np.zeros(self.f.widths[j])); continue
            t = self.f.tokens[j][rows].astype(np.float64)
            w = np.ones(len(rows)) if weights is None else np.asarray(
                [weights[list(index).index(r + self.f.offsets[j])] for r in rows])
            m = (t * w[:, None]).sum(0)
            parts.append(m / np.linalg.norm(m))
        v = np.concatenate(parts)
        return v / np.linalg.norm(v)

    def test_each_lattice_counts_once_whatever_its_token_count(self):
        o = self.f.offsets
        index = np.r_[o[0]:o[0] + 27, o[1]:o[1] + 3, o[2]]          # 27 fine, 3 mid, 1 deep
        v = self.h.pool(self.P, index)
        np.testing.assert_allclose(v, self.expected(index), atol=1e-5)
        for j in range(3):                                          # every block at 1/sqrt(3)
            self.assertAlmostEqual(float(np.linalg.norm(v[self.f.block(j)])), 1 / np.sqrt(3), places=5)
        # a plain mean over the blocks is the fine lattice's vector - the reason this head exists
        m = MeanPoolHead().pool(self.P, index)
        self.assertGreater(float(np.linalg.norm(m[self.f.block(0)])), 0.8)

    def test_bias_weights_tokens_within_their_lattice(self):
        o = self.f.offsets
        index = np.r_[o[0]:o[0] + 5, o[2]:o[2] + 2]
        bias = np.log(np.linspace(0.1, 1.0, len(index)))
        np.testing.assert_allclose(self.h.pool(self.P, index, bias=bias),
                                   self.expected(index, np.exp(bias - bias.max())), atol=1e-5)

    def test_a_lattice_the_gate_does_not_reach_stays_zero(self):
        o = self.f.offsets
        v = self.h.pool(self.P, np.r_[o[2]:o[2] + 2])
        self.assertFalse(v[self.f.block(0)].any())
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)

    def test_a_head_refuses_a_field_of_other_widths(self):
        with self.assertRaises(ValueError):
            LatticeMeanHead((128, 256)).prepare(self.f.all_tokens())


class Tiling(unittest.TestCase):
    PATCH = (112, 112, 128)

    def test_padding_and_starts_are_aligned_and_cover_everything(self):
        for n in (1, 40, 112, 113, 170, 257, 400):
            for p in self.PATCH:
                m = null.padded_extent(n, p)
                self.assertGreaterEqual(m, max(n, p)); self.assertEqual(m % 16, 0)
                s = null.tile_starts(m, p)
                self.assertTrue(all(v % 16 == 0 for v in s))
                self.assertEqual(s[0], 0); self.assertEqual(s[-1], m - p)
                covered = np.zeros(m, bool)
                for v in s:
                    covered[v:v + p] = True
                self.assertTrue(covered.all())
                self.assertTrue(all(b - a <= p for a, b in zip(s, s[1:])), "a gap between tiles")

    def test_an_unpadded_extent_is_refused(self):
        with self.assertRaises(ValueError):
            null.tile_starts(170, 112)

    def test_the_blend_places_every_tiles_tokens_exactly(self):
        # a stand-in encoder whose token is a known function of its place: whatever the weights,
        # a correct placement blends back to that function exactly
        padded = tuple(null.padded_extent(n, p) for n, p in zip((150, 130, 200), self.PATCH))
        truth = [np.add.outer(np.add.outer(np.arange(padded[0] // k[0]) * 1e4, np.arange(padded[1] // k[1]) * 1e2),
                              np.arange(padded[2] // k[2])) for k in null.KERNELS]
        acc = [np.zeros_like(t) for t in truth]; wsum = [np.zeros_like(t) for t in truth]
        g = np.random.default_rng(0).random(self.PATCH) + 0.1
        W = [null.token_weights(g, k) for k in null.KERNELS]
        for vox, toks in null.tile_slices(padded, self.PATCH):
            for j, sl in enumerate(toks):
                out = truth[j][sl]                                     # what the tile would output
                self.assertEqual(out.shape, W[j].shape)
                acc[j][sl] += out * W[j]; wsum[j][sl] += W[j]
        for j in range(3):
            self.assertTrue((wsum[j] > 0).all())
            np.testing.assert_allclose(acc[j] / wsum[j], truth[j], rtol=1e-12)

    def test_token_weights_are_box_means_of_the_gaussian(self):
        g = np.random.default_rng(1).random((16, 32, 48))
        w = null.token_weights(g, (8, 8, 16))
        self.assertEqual(w.shape, (2, 4, 3))
        self.assertAlmostEqual(w[1, 2, 0], g[8:16, 16:24, 0:16].mean())


class OrgansAndGeometry(unittest.TestCase):
    def test_structures_map_into_radars_organ_scheme(self):
        lm = {1: "spleen", 2: "kidney_right", 3: "kidney_left", 5: "liver", 90: "prostate", 117: "costal_cartilages"}
        labels = np.array([0, 1, 2, 3, 5, 90, 117, 4])
        got = null.organ_mask(labels, lm)
        name = lambda v: radar.ORGANS[v - 1] if v else None
        self.assertEqual([name(int(v)) for v in got],
                         [None, "spleen", "kidney", "kidney", "liver", None, None, None])

    def test_haversack_geometry_lands_where_itk_puts_a_voxel(self):
        # an oblique canonical frame: ITK world of index (x, y, z) = origin + D @ (x sx, y sy, z sz)
        th = 0.3
        D = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        eff = (3.1, 2.9, 3.0)                                          # (Z, Y, X)
        origin = (-120.0, 80.0, 400.0)
        g = null.grid_from_haversack((20, 30, 40), eff, origin, D.reshape(-1).tolist())
        for zyx in [(0, 0, 0), (5, 7, 11), (19, 29, 39)]:
            z, y, x = zyx
            itk = np.asarray(origin) + D @ np.array([x * eff[2], y * eff[1], z * eff[0]])
            np.testing.assert_allclose(g.world(np.array(zyx, float)), itk, atol=1e-9)

    def test_an_export_becomes_a_field_with_the_license_it_inherits(self):
        f = mixed_field()
        meta = {"u": "series-1", "grid": {"shape": list(SHAPE), "directions": [list(r) for r in f.grid.directions],
                                          "origin": list(f.grid.origin)}, "tiles": 12, "haversack": "651aff7"}
        arrays = {f"tokens{j}": t for j, t in enumerate(f.tokens)}
        arrays["native_mask"] = np.zeros(SHAPE, np.uint8)
        g = null.field_from_export(arrays, meta)
        self.assertEqual(g.widths, null.WIDTHS)
        self.assertEqual(g.provenance.license, "Apache-2.0")
        self.assertEqual(g.provenance.code, "651aff7")
        self.assertEqual(g.native_labels, radar.ORGANS)
