import json, tempfile, pathlib, unittest

import numpy as np
from rankfield.geometry import Geometry

from feldglas import Field, Provenance
from feldglas.store import read_field, read_meta, write_field
from conftest import KERNELS, make_field


class Contract(unittest.TestCase):
    def test_lattice_shapes_and_offsets(self):
        f = make_field()
        self.assertEqual([f.lattice_shape(j) for j in range(3)], [(2, 2, 2), (4, 4, 4), (8, 8, 8)])
        self.assertEqual(f.offsets.tolist(), [0, 8, 72, 584])
        self.assertEqual(f.all_tokens().shape, (584, 256))

    def test_a_token_center_is_the_center_of_its_box_in_the_world(self):
        # the geometry is the part that fails silently: check it two independent ways
        f = make_field()
        for j, k in enumerate(KERNELS):
            centers = f.token_centers(j).reshape(*f.lattice_shape(j), 3)
            a, b, c = (s - 1 for s in f.lattice_shape(j))                       # the LAST token
            lo = np.array([a * k[0], b * k[1], c * k[2]], float)
            box_center = lo + (np.array(k) - 1) / 2.0                           # model index of its box's center
            np.testing.assert_allclose(centers[a, b, c], f.grid.world(box_center), atol=1e-9)
        # and the spacing of a lattice is the model grid's times the kernel
        np.testing.assert_allclose(f.lattice_geometry(0).spacing, (40.0, 32.0, 32.0))

    def test_a_kernel_that_does_not_tile_the_grid_is_refused(self):
        f = make_field()
        with self.assertRaises(ValueError):
            Field(tokens=f.tokens, kernels=KERNELS, grid=Geometry.aligned((16, 64, 60), (5, 1, 1)), provenance=f.provenance)

    def test_token_count_must_match_the_lattice(self):
        f = make_field()
        with self.assertRaises(ValueError):
            Field(tokens=[f.tokens[0][:-1], *f.tokens[1:]], kernels=KERNELS, grid=f.grid, provenance=f.provenance)


class Store(unittest.TestCase):
    def test_round_trip_keeps_geometry_provenance_and_license(self):
        f = make_field()
        f.provenance = Provenance(encoder="radar", code="x", weights="w", preprocessing="p",
                                  license="CC-BY-NC-SA-4.0", source="series-1", extra={"crop": {"lo": [1, 2, 3]}})
        with tempfile.TemporaryDirectory() as d:
            p = write_field(pathlib.Path(d) / "a" / "f.npz", f)
            g = read_field(p)
            self.assertEqual(read_meta(p)["provenance"]["license"], "CC-BY-NC-SA-4.0")
        self.assertEqual(g.grid, f.grid)
        self.assertEqual(g.provenance, f.provenance)
        self.assertEqual(g.native_labels, f.native_labels)
        np.testing.assert_array_equal(g.native_mask, f.native_mask)
        for a, b in zip(g.tokens, f.tokens):                                    # fp16 on disk
            np.testing.assert_allclose(a.astype(np.float32), b, atol=2e-3, rtol=1e-3)

    def test_readers_refuse_what_they_do_not_know(self):
        f = make_field()
        with tempfile.TemporaryDirectory() as d:
            p = write_field(pathlib.Path(d) / "f.npz", f)
            with np.load(p) as z:
                arrays = {k: z[k] for k in z.files}
            meta = json.loads(str(arrays["meta"]))
            for change in ({"version": "9.0"}, {"format": "something-else"}):
                np.savez(pathlib.Path(d) / "bad.npz", **{**arrays, "meta": np.array(json.dumps({**meta, **change}))})
                with self.assertRaises(ValueError):
                    read_field(pathlib.Path(d) / "bad.npz")
