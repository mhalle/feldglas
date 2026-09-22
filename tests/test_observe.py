import tempfile, pathlib, unittest

import numpy as np
import pytest

from feldglas.observe import NormalModel, bank_max, hotelling_template, shrunk_covariance

P, LOUD = 32, 4


def cloud(n=600, seed=0):
    """Normal "tissue": four LOUD directions (sd 3) and twenty-eight QUIET ones (sd 0.2), in a
    rotated basis so nothing is axis-aligned for the estimator. Returns the rows and the basis -
    column j of ``R`` is direction j. The shape of the real thing: region vectors vary mostly
    along a dozen dimensions that patients, phases and scanners share, and a lesion moves them
    along quiet ones (EXPLORATION 5.3)."""
    rng = np.random.default_rng(seed)
    R = np.linalg.qr(rng.standard_normal((P, P)))[0]
    sd = np.r_[np.full(LOUD, 3.0), np.full(P - LOUD, 0.2)]
    return (rng.standard_normal((n, P)) * sd) @ R.T, R


auc = lambda pos, neg: float((np.asarray(pos)[:, None] > np.asarray(neg)[None, :]).mean())


class Shrinkage(unittest.TestCase):
    def test_same_estimator_as_scikit_learn(self):
        cov = pytest.importorskip("sklearn.covariance")
        X, _ = cloud(); X = X - X.mean(0)
        C, s = shrunk_covariance(X)
        lw = cov.LedoitWolf(assume_centered=True).fit(X)
        np.testing.assert_allclose(C, lw.covariance_, rtol=1e-8, atol=1e-10)
        self.assertAlmostEqual(s, float(lw.shrinkage_), places=10)

    def test_invertible_when_there_are_fewer_rows_than_dimensions(self):
        X, _ = cloud(n=20); X = X - X.mean(0)
        C, s = shrunk_covariance(X)
        self.assertTrue(0 < s <= 1)
        self.assertGreater(np.linalg.eigvalsh(C).min(), 0)


class Normal(unittest.TestCase):
    def test_whitening_finds_a_quiet_direction_that_euclid_cannot(self):
        # the lesson of 2026-09-20 in miniature: the signal lives on a LOW-variance direction
        X, R = cloud(n=700)
        model = NormalModel.fit(X[:500])
        clean, odd = X[500:], X[500:600] + 1.6 * R[:, 10]                  # eight quiet sds, half a loud one
        self.assertGreater(auc(model.distance(odd), model.distance(clean)), 0.97)
        euclid = lambda A: np.linalg.norm(A - model.mean, axis=1)
        self.assertLess(auc(euclid(odd), euclid(clean)), 0.65)             # all but invisible without the covariance

    def test_within_patient_covariance_ignores_where_each_patient_sits(self):
        rng = np.random.default_rng(3)
        groups = np.repeat(np.arange(20), 25)
        X = rng.standard_normal((500, 16)) * 0.1 + rng.standard_normal((20, 16))[groups] * 3.0
        within, total = NormalModel.fit(X, groups=groups), NormalModel.fit(X)
        self.assertTrue(within.within_groups); self.assertFalse(total.within_groups)
        ref = X[groups == 0].mean(0)
        self.assertGreater(np.median(within.distance(X[groups == 0], center=ref)),
                           3 * np.median(total.distance(X[groups == 0], center=ref)))

    def test_save_and_load(self):
        m = NormalModel.fit(cloud()[0], label="liver box32 venous")
        with tempfile.TemporaryDirectory() as d:
            g = NormalModel.load(m.save(pathlib.Path(d) / "atlas" / "liver.npz"))
        np.testing.assert_allclose(g.precision, m.precision); self.assertEqual(g.label, m.label); self.assertEqual(g.n, m.n)


class Templates(unittest.TestCase):
    def test_hotelling_beats_the_raw_displacement_and_a_bank_handles_an_unknown_parameter(self):
        X, R = cloud(n=900, seed=5)
        model = NormalModel.fit(X[:600])
        clean = X[600:]
        # two signals ("small" and "large" lesions): each mostly along its own QUIET direction, with
        # a larger component along a LOUD one - which is what a raw mean difference latches onto
        d_small, d_large = 1.0 * R[:, 10] + 2.0 * R[:, 0], 1.0 * R[:, 20] + 2.0 * R[:, 1]
        small, large = clean[:100] + d_small, clean[100:200] + d_large
        w_s, w_l = hotelling_template(model, d_small), hotelling_template(model, d_large)
        self.assertGreater(auc(small @ w_s, clean @ w_s), 0.97)
        self.assertLess(auc(small @ d_small, clean @ d_small), 0.85)       # the unwhitened template
        z, z0 = bank_max(np.concatenate([small, large]), [w_s, w_l], clean), bank_max(clean, [w_s, w_l], clean)
        self.assertGreater(auc(z, z0), 0.95)                               # size unknown: the bank finds both
        self.assertLess(auc(large @ w_s, clean @ w_s), 0.75)               # the wrong single template does not
