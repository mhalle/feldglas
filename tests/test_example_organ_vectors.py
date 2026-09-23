"""examples/organ_vectors.py reads a field with zarr and numpy alone, as a client would: hold its
reader to the store's, so the example cannot drift from the format it demonstrates."""
import importlib.util, pathlib, tempfile, unittest

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("duckn")

from feldglas.store import write_field
from test_store_zarr import oblique_field

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "organ_vectors.py"


def example():
    spec = importlib.util.spec_from_file_location("organ_vectors", EXAMPLE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class ExampleReader(unittest.TestCase):
    def test_tokens_centers_and_extent_match_the_store(self):
        f = oblique_field()
        for dtype in (np.float16, np.int8):
            with tempfile.TemporaryDirectory() as d:
                p = write_field(pathlib.Path(d) / "f.zarr.zip", f, token_dtype=dtype)
                lattices, ext = example().read_field(str(p))
            self.assertEqual(ext["provenance"]["encoder"], "radar")
            for j, (tokens, centers, inside, D, o, shape) in enumerate(lattices):
                step = (f.tokens[j].max(0) - f.tokens[j].min(0)) / 255
                np.testing.assert_allclose(tokens, f.tokens[j], atol=float(step.max()) if dtype == np.int8 else 1e-2,
                                           err_msg=f"{dtype.__name__} lattice {j}")
                np.testing.assert_allclose(centers, f.token_centers(j), atol=1e-6)
                lo, hi = f.lattice_extent(j)
                idx = np.stack(np.unravel_index(np.arange(len(tokens)), shape), -1)
                np.testing.assert_array_equal(inside, np.all((idx >= lo) & (idx < hi), axis=1))
