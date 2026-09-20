"""Synthetic fields and heads: nothing in the default suite touches an encoder's weights or any
file derived from them (those are license-bound and live outside the repository)."""
import numpy as np
import pytest
from rankfield.geometry import Geometry

from feldglas import Field, Provenance

KERNELS = [(8, 32, 32), (4, 16, 16), (2, 8, 8)]


def make_field(shape=(16, 64, 64), channels=256, seed=0, mask=True) -> Field:
    rng = np.random.default_rng(seed)
    grid = Geometry(shape=shape, directions=((0.0, 0.0, 5.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
                    origin=(-30.0, -40.0, 100.0))
    tokens = [rng.standard_normal((int(np.prod([s // k for s, k in zip(shape, kk)])), channels)).astype(np.float32)
              for kk in KERNELS]
    m = None
    if mask:
        m = np.zeros(shape, np.uint8)
        m[2:14, 8:56, 8:40] = 1                  # a big organ
        m[6:8, 24:32, 48:56] = 2                 # a small one: exactly one fine token (2 x 8 x 8), part of a mid one
    return Field(tokens=tokens, kernels=KERNELS, grid=grid,
                 provenance=Provenance(encoder="synthetic", license="none", source="test"),
                 native_mask=m, native_labels=("big", "small"))


@pytest.fixture
def field():
    return make_field()
