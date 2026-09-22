import unittest

import numpy as np

from feldglas.gate import box, occupancy, select
from conftest import make_field


class Gates(unittest.TestCase):
    def test_occupancy_is_the_fraction_of_each_box(self):
        m = np.zeros((4, 16, 16), bool); m[:2, :8, :8] = True
        np.testing.assert_allclose(occupancy(m, (4, 16, 16)), [0.125])
        np.testing.assert_allclose(occupancy(m, (2, 8, 8)), [1, 0, 0, 0, 0, 0, 0, 0])

    def test_the_raster_rule_admits_what_the_interior_rule_refuses(self):
        f = make_field()
        big = f.native_mask == 1
        any_, inside, half = select(f, big), select(f, big, "interior"), select(f, big, 0.5)
        self.assertGreater(len(any_), len(half)); self.assertGreater(len(half), len(inside))
        self.assertTrue(np.all(inside.occupancy >= 1 - 1e-6))
        self.assertTrue(set(inside.index) <= set(half.index) <= set(any_.index))
        # the big organ owns NO deep token entirely - the gateability fact, in miniature
        self.assertEqual(int((inside.lattice == 0).sum()), 0)
        self.assertGreater(int((any_.lattice == 0).sum()), 0)

    def test_indices_address_the_concatenated_lattices(self):
        f = make_field()
        g = select(f, f.native_mask == 2)
        for j in range(3):
            idx = g.index[g.lattice == j]
            self.assertTrue(np.all((idx >= f.offsets[j]) & (idx < f.offsets[j + 1])))

    def test_soft_bias_and_lattice_restriction(self):
        f = make_field()
        g = select(f, f.native_mask == 1)
        np.testing.assert_allclose(g.soft_bias(), np.log(g.occupancy))
        self.assertTrue(np.all(g.only(2).lattice == 2)); self.assertEqual(len(g.only(0, 1, 2)), len(g))

    def test_a_box_is_millimeters_not_voxels(self):
        f = make_field()                                   # 5 x 1 x 1 mm voxels
        m = box(f, (8, 32, 32), 20.0)
        z, y, x = (np.ptp(np.nonzero(m)[a]) + 1 for a in range(3))
        self.assertEqual((z, y, x), (5, 21, 21))           # 25 mm of slices, 21 mm in plane: one cube in the patient
        inside = select(f, m, within=f.native_mask == 1)
        self.assertGreater(len(inside), 0)

    def test_a_mask_off_the_model_grid_is_refused(self):
        with self.assertRaises(ValueError):
            select(make_field(), np.zeros((4, 4, 4), bool))


class FastBoxes(unittest.TestCase):
    def test_box_gate_is_select_of_the_box_exactly(self):
        from feldglas.gate import box_gate, occupancies
        f = make_field()
        organ = f.native_mask == 1
        within = occupancies(f, organ)
        rng = np.random.default_rng(0)
        for _ in range(40):
            c = [rng.uniform(0, s) for s in f.grid.shape]            # off-lattice centers, some at the edge
            mm = float(rng.choice([8.0, 16.0, 32.0, 64.0]))
            for rule in ("any", "interior", 0.5):
                for w_slow, w_fast in ((None, None), (organ, within)):
                    a, b = select(f, box(f, c, mm), rule, within=w_slow), box_gate(f, c, mm, rule, within=w_fast)
                    np.testing.assert_array_equal(a.index, b.index)
                    np.testing.assert_allclose(a.occupancy, b.occupancy, atol=1e-6)
                    np.testing.assert_array_equal(a.lattice, b.lattice)


class BoxCenters(unittest.TestCase):
    """A box is centered where it says it is (2026-09-22). The old rule floored the lower edge, so
    along any axis whose half size is not a whole number of voxels every box took an extra voxel
    below: half a voxel low (2.5 mm in z on RADAR's 5 x 1 x 1 mm grid, every axis on the null
    model's) and a voxel too big."""

    GRIDS = (((64, 256, 256), (5.0, 1.0, 1.0)), ((96, 96, 96), (3.0, 3.0, 3.0)), ((192, 192, 192), (1.5, 1.5, 1.5)))

    def test_a_box_about_a_voxel_is_centered_on_it_and_holds_exactly_the_voxels_within_it(self):
        from feldglas.gate import box_bounds
        rng = np.random.default_rng(0)
        for shape, sp in self.GRIDS:
            for size in (16.0, 32.0, 64.0):
                half = np.array([size / s / 2 for s in sp])
                for _ in range(20):
                    c = np.array([rng.integers(int(h) + 2, n - int(h) - 2) for h, n in zip(half, shape)])
                    lo, hi = box_bounds(c, size, sp, shape)
                    np.testing.assert_array_equal(lo + hi - 1, 2 * c, err_msg=f"{sp} {size} mm about {c}")
                    for d in range(3):                           # exactly the voxel centers within +-h
                        want = [i for i in range(shape[d]) if abs(i - c[d]) <= half[d] + 1e-9]
                        self.assertEqual((lo[d], hi[d] - 1), (want[0], want[-1]), f"{sp} axis {d} {size} mm")

    def test_fractional_centers_are_unbiased(self):
        from feldglas.gate import box_bounds
        rng = np.random.default_rng(1)
        for shape, sp in self.GRIDS:
            c = np.stack([rng.uniform(n // 4, n - n // 4, 2000) for n in shape], 1)
            lo, hi = box_bounds(c, 32.0, sp, shape)
            bias = ((lo + hi - 1) / 2.0 - c).mean(0)
            self.assertTrue(np.all(np.abs(bias) < 0.05), f"{sp}: mean offset {bias}")

    def test_the_mask_the_gate_and_the_sweep_cut_the_same_box(self):
        from feldglas.gate import box, box_bounds
        from feldglas.suite.normal_atlas import _edges
        f = make_field()
        c = np.array([7, 30, 25])
        lo, hi = box_bounds(c, 32.0, f.grid.spacing, f.grid.shape)
        m = box(f, c, 32.0)
        idx = np.argwhere(m)
        np.testing.assert_array_equal(idx.min(0), lo); np.testing.assert_array_equal(idx.max(0), hi - 1)
        l2, h2 = _edges(c[None], 32.0, f.grid.spacing, f.grid.shape)
        np.testing.assert_array_equal(l2[0], lo); np.testing.assert_array_equal(h2[0], hi)
