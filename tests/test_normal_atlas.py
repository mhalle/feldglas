"""The normal-atlas probe on synthetic fields: each piece against a slower, obvious version."""
import unittest

import numpy as np

from conftest import make_field
from feldglas.gate import box, select
from feldglas.heads import MeanPoolHead
from feldglas.suite import normal_atlas as na


class TestSweep(unittest.TestCase):
    def setUp(self):
        self.field = make_field(shape=(16, 64, 64))
        self.organ = self.field.native_mask == 1

    def test_fill_and_volume_match_a_box_drawn_the_slow_way(self):
        sw = na.sweep(self.field, self.organ, 32.0)
        self.assertGreater(len(sw.center), 4)
        for i in range(len(sw.center)):
            b = box(self.field, sw.center[i], 32.0)
            self.assertAlmostEqual(sw.fill[i], (b & self.organ).sum() / b.sum(), places=9)
            self.assertAlmostEqual(sw.organ_ml[i], (b & self.organ).sum() * 5.0 / 1000.0, places=9)
            self.assertGreaterEqual(sw.fill[i], 0.25)

    def test_every_box_the_rule_admits_is_there(self):
        sw = na.sweep(self.field, self.organ, 32.0, min_fill=0.0)
        kept = na.sweep(self.field, self.organ, 32.0, min_fill=0.5)
        self.assertEqual(len(kept.center), int((sw.fill >= 0.5).sum()))
        self.assertTrue((kept.coords >= 0).all() and (kept.coords <= 1).all())

    def test_pooled_vectors_are_the_ones_select_gives(self):
        head = MeanPoolHead(); prepared = head.prepare(self.field.all_tokens())
        sw = na.sweep(self.field, self.organ, 32.0)
        X = na.pool_sweep(self.field, head, prepared, sw, self.organ, None)
        for i in (0, len(sw.center) // 2, len(sw.center) - 1):
            g = select(self.field, box(self.field, sw.center[i], 32.0), within=self.organ)
            np.testing.assert_allclose(X[i], head.pool(prepared, g.index), atol=1e-6)

    def test_one_scan_against_a_normal_model(self):
        from feldglas.observe import NormalModel
        head = MeanPoolHead(); prepared = head.prepare(self.field.all_tokens())
        sw = na.sweep(self.field, self.organ, 32.0)
        X = na.pool_sweep(self.field, head, prepared, sw, self.organ, None)
        rng = np.random.default_rng(0)
        cloud = X.mean(0) + 0.05 * rng.standard_normal((600, X.shape[1]))
        for model in (NormalModel.fit(cloud), na.PositionedNormal.fit(cloud, rng.random((600, 3)), rng.random(600))):
            sw2, d = na.normal_atlas(self.field, head, self.organ, None, model, 32.0)
            self.assertEqual(len(d), len(sw2.center)); self.assertTrue(np.isfinite(d).all() and (d > 0).all())

    def test_an_empty_organ_is_an_empty_sweep(self):
        sw = na.sweep(self.field, np.zeros(self.field.grid.shape, bool), 32.0)
        self.assertEqual(len(sw.center), 0)


class TestScatter(unittest.TestCase):
    def setUp(self):
        self.field = make_field(shape=(16, 64, 64))
        g = self.field.grid
        self.own = np.eye(4); self.own[:3, :3] = np.asarray(g.directions).T; self.own[:3, 3] = g.origin

    def test_a_mask_on_the_model_grid_comes_back_as_itself(self):
        lab = np.zeros(self.field.grid.shape, np.int32)
        lab[4:7, 10:20, 12:30] = 1; lab[10:12, 40:44, 40:44] = 2
        occ, owner = na.scatter_mask(self.field, lab, self.own, sub=1)
        np.testing.assert_array_equal(occ > 0, lab > 0)
        np.testing.assert_array_equal(owner, lab)
        np.testing.assert_allclose(occ[lab > 0], 1.0)

    def test_a_finer_mask_keeps_its_volume_and_gives_partial_voxels(self):
        fine = self.own.copy(); fine[:3, :3] = self.own[:3, :3] @ np.diag([0.5, 0.5, 0.5])   # 2.5 x 0.5 x 0.5 mm
        lab = np.zeros((32, 128, 128), np.int32)
        lab[9:14, 41:63, 41:63] = 1                                   # odd edges: partial model voxels
        occ, _ = na.scatter_mask(self.field, lab, fine, sub=2)
        self.assertAlmostEqual(occ.sum() * 5.0, lab.sum() * 2.5 * 0.25, delta=0.02 * lab.sum() * 0.625)
        self.assertTrue(((occ > 0.05) & (occ < 0.95)).any())

    def test_overlaps_name_the_lesion_and_its_ml(self):
        lab = np.zeros(self.field.grid.shape, np.int32); lab[6:8, 24:32, 24:32] = 7
        occ, owner = na.scatter_mask(self.field, lab, self.own, sub=1)
        sw = na.sweep(self.field, self.field.native_mask == 1, 32.0)
        rows = na.lesion_overlaps(sw, occ, owner, 5.0 / 1000.0)
        self.assertTrue(len(rows) and set(rows[:, 1]) == {7.0})
        self.assertAlmostEqual(rows[:, 2].max(), (lab > 0).sum() * 0.005, places=6)


class TestScoring(unittest.TestCase):
    def test_auc_is_sklearns_with_ties(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(1)
        y = rng.random(300) < 0.3
        s = np.round(rng.standard_normal(300) + y, 1)                  # rounding makes ties
        self.assertAlmostEqual(na.auc(s, y), roc_auc_score(y, s), places=12)
        self.assertTrue(np.isnan(na.auc(s, np.zeros(300, bool))))

    def test_local_maxima_are_one_per_ridge(self):
        ijk = np.stack(np.meshgrid(range(5), range(5), range(5), indexing="ij"), -1).reshape(-1, 3)
        s = -np.linalg.norm(ijk - np.array([1, 1, 1]), axis=1)
        s = np.maximum(s, -np.linalg.norm(ijk - np.array([4, 4, 4]), axis=1) - 0.5)
        top = na.local_maxima(ijk, s)
        self.assertEqual({tuple(x) for x in ijk[top]}, {(1, 1, 1), (4, 4, 4)})

    def _scan(self, lesion_score, clean_peak):
        ijk = np.stack(np.meshgrid(range(4), range(4), range(1), indexing="ij"), -1).reshape(-1, 3)
        score = np.zeros(16); score[5] = lesion_score; score[15] = clean_peak
        tumor = np.zeros(16); tumor[5] = 1.0
        return {"score": score, "ijk": ijk, "tumor_ml": tumor, "organ_ml": np.full(16, 30.0),
                "overlaps": np.array([[5, 1, 1.0]]), "lesions": {1: {"ml": 1.0, "diameter_mm": 12.4}}}

    def test_froc_counts_lesions_against_clean_peaks(self):
        scans = [self._scan(5.0, 1.0), self._scan(0.5, 2.0)]           # one lesion above every clean peak, one below
        r = na.froc(scans, fp_per_scan=(0.5, 1.0))
        self.assertEqual((r["scans"], r["lesions"], r["never_hit"]), (2, 2, 0))
        self.assertEqual(r["at"]["0.5"]["threshold"], 2.0)             # one false positive in two scans
        self.assertEqual(r["at"]["0.5"]["all"], 0.5)
        self.assertEqual(r["at"]["1.0"]["10-20mm"], [0.5, 2])

    def test_a_lesion_no_box_hits_is_never_found(self):
        s = self._scan(5.0, 1.0); s["overlaps"] = np.array([[5, 1, 0.01]]); s["organ_ml"] = np.full(16, 30.0)
        r = na.froc([s], fp_per_scan=(50.0,))                          # more false positives than exist: threshold -inf
        self.assertEqual((r["never_hit"], r["at"]["50.0"]["all"]), (1, 0.0))


class TestPositioned(unittest.TestCase):
    def test_a_mean_that_moves_with_place_is_followed(self):
        rng = np.random.default_rng(3)
        n, p = 3000, 24
        c = rng.random((n, 3)); fill = 0.25 + 0.75 * rng.random(n)
        drift = rng.standard_normal((3, p)) * 3.0
        X = c @ drift + np.outer(fill, rng.standard_normal(p)) + 0.3 * rng.standard_normal((n, p))
        m = na.PositionedNormal.fit(X[:2500], c[:2500], fill[:2500])
        d = m.distance(X[2500:], c[2500:], fill[2500:])
        self.assertAlmostEqual(float(np.mean(d ** 2)), p, delta=0.2 * p)           # whitened: chi-square with p
        # the outlier a flat model cannot see at all: tissue that is normal SOMEWHERE ELSE in the
        # organ. (A shift along a quiet axis is no test - whitening by the total covariance already
        # finds it, because a low-rank drift leaves every other direction quiet.)
        odd, there = X[2500:], 1.0 - c[2500:]
        truth = np.r_[np.zeros(500), np.ones(500)] > 0
        self.assertGreater(na.auc(np.r_[d, m.distance(odd, there, fill[2500:])], truth), 0.98)
        flat = na.PositionedNormal.fit(X[:2500], np.zeros((2500, 3)))
        f = flat.distance(odd, np.zeros((500, 3)))
        self.assertAlmostEqual(na.auc(np.r_[f, f], truth), 0.5, places=6)

if __name__ == "__main__":
    unittest.main()
