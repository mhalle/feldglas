"""Observers: detection theory on an encoder's channels.

A region's vector is a set of CHANNELS; normal tissue is a BACKGROUND (mean, covariance); a lesion
or a variant is a SIGNAL (the displacement it causes). What the user knows decides the observer
(EXPLORATION section 3.2):

  knows the signal               Hotelling template  w = S^-1 d        :func:`hotelling_template`
  knows it up to a parameter     the maximum over a bank of templates  :func:`bank_max`
  knows only what normal is      Mahalanobis distance from normal      :meth:`NormalModel.distance`

What 2026-09-20's tests established, and this module encodes:

- The failures of "novelty" in the RADAR study were failures of the CENTRE, not of the idea.
  Distance from a POPULATION's normal mean repaired every one (kidney at 64 mm: 0.000 -> 1.00
  per patient), and a normal model from six healthy donors of another collection found
  tumor-bearing liver regions at 0.909 pooled AUC against 0.816 for RADAR's own finding score.
- Whitening helps a user-supplied reference (one click, kidney 32 mm: 0.76 -> 0.91) and needs a
  CLEAN covariance: estimated with lesions in it, it whitens the signal away (0.54-0.70). Fit
  normal on normal.
- Report pooled figures beside per-patient ones. A tool has one threshold for everybody;
  own-organ novelty was 0.844 per patient and 0.569 pooled.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np


def shrunk_covariance(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf covariance of CENTRED rows, and the shrinkage it chose. Region vectors are
    unit length in a few hundred dimensions, so the sample covariance is rank-deficient by
    construction and the shrinkage is what makes it invertible. Same estimator as scikit-learn's
    (tests hold the two together), written out so the core needs only numpy."""
    X = np.asarray(X, np.float64)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    delta_ = (S ** 2).sum()
    beta_ = ((X ** 2).sum(1) ** 2).sum()
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * np.trace(S) + p * mu ** 2) / p
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else float(beta / delta)
    C = (1.0 - shrinkage) * S
    C.flat[::p + 1] += shrinkage * mu
    return C, shrinkage


@dataclass
class NormalModel:
    """The unremarkable, for one kind of region (an organ, a gate size, a phase): a mean and a
    precision. ``groups`` (patient ids) makes the covariance WITHIN-patient - what is left once a
    patient-specific reference has been subtracted, the right whitening for a click - while
    without them it is the population's total, the right one for distance from ``mean``."""

    mean: np.ndarray
    precision: np.ndarray
    n: int
    shrinkage: float
    within_groups: bool = False
    label: str = ""

    @classmethod
    def fit(cls, X, groups=None, label: str = "") -> "NormalModel":
        X = np.asarray(X, np.float64)
        mean = X.mean(0)
        if groups is None:
            R = X - mean
        else:
            g = np.asarray(groups)
            R = np.concatenate([X[g == u] - X[g == u].mean(0) for u in np.unique(g) if (g == u).sum() >= 2])
        C, s = shrunk_covariance(R)
        return cls(mean=mean, precision=np.linalg.inv(C), n=int(len(X)), shrinkage=s,
                   within_groups=groups is not None, label=label)

    def distance(self, X, centre=None) -> np.ndarray:
        """Mahalanobis distance from ``centre`` - the model's own mean, or a reference the user
        supplies (the mean of a few "this is normal" regions of THIS scan)."""
        D = np.atleast_2d(np.asarray(X, np.float64)) - (self.mean if centre is None else np.asarray(centre, np.float64))
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", D, self.precision, D), 0.0))

    def save(self, path) -> pathlib.Path:
        path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, precision=self.precision, n=self.n, shrinkage=self.shrinkage,
                 within_groups=self.within_groups, label=np.array(self.label))
        return path

    @classmethod
    def load(cls, path) -> "NormalModel":
        with np.load(path) as z:
            return cls(mean=z["mean"], precision=z["precision"], n=int(z["n"]), shrinkage=float(z["shrinkage"]),
                       within_groups=bool(z["within_groups"]), label=str(z["label"]))


def hotelling_template(normal: NormalModel, displacements) -> np.ndarray:
    """The matched filter for a KNOWN signal: ``S^-1 d``, with ``d`` the mean displacement the
    signal causes (lesion region minus that patient's clean centre; or painted minus unpainted).
    Score a region as ``(v - reference) @ template``."""
    return normal.precision @ np.asarray(displacements, np.float64).reshape(-1, normal.mean.size).mean(0)


def bank_max(X, templates, clean) -> np.ndarray:
    """A parameter left unknown (lesion size): the maximum over a bank of templates, each
    standardized on CLEAN vectors so the bands compare. On the blind body-gated boxes this matched
    the size-matched template in every size band (0.842 / 0.989 / 0.984 / 0.967) - and so, within
    0.02, did one direction fitted on all sizes; what a size-tuned template buys is selectivity."""
    X, clean = np.asarray(X, np.float64), np.asarray(clean, np.float64)
    z = []
    for w in np.atleast_2d(np.asarray(templates, np.float64)):
        c = clean @ w
        z.append((X @ w - c.mean()) / (c.std() + 1e-12))
    return np.max(np.stack(z), 0)
