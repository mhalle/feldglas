"""Label maps as gates, and the session that asks a field things by structure name."""
import gzip
import unittest

import numpy as np

from conftest import make_field
from feldglas.adapters import radar
from feldglas.gate import select
from feldglas.heads import MeanPoolHead
from feldglas.labels import GridLabels, read_seg_nrrd
from feldglas.session import Session


def write_seg(path, values, affine, names, extra="", dimension=None, space="left-posterior-superior", layers=None):
    """A .seg.nrrd as haversack writes it (3-D, one layer) or, with ``values`` 4-D (layer first) and
    ``layers`` {name: layer}, as 3D Slicer writes overlapping segments. ``affine`` is LPS; with
    ``space`` RAS the file's directions and origin are written in RAS, as such a file would hold them."""
    values = np.asarray(values)
    layered = values.ndim == 4
    dimension = dimension or values.ndim
    A = np.array(affine, float)
    if space == "right-anterior-superior":
        A[:2, :] *= -1
    d = A[:3, :3].T
    dirs = " ".join("(" + ",".join(repr(float(x)) for x in row) + ")" for row in d)
    head = ["NRRD0004", "# a test file", "type: unsigned char", f"dimension: {dimension}", f"space: {space}",
            "sizes: " + " ".join(str(v) for v in values.shape),
            "space directions: " + ("none " if layered else "") + dirs,
            "kinds: " + ("list " if layered else "") + "domain domain domain", "encoding: gzip",
            "space origin: (" + ",".join(repr(float(x)) for x in A[:3, 3]) + ")"]
    for i, (n, v) in enumerate(names.items()):
        head += [f"Segment{i}_LabelValue:={v}", f"Segment{i}_Name:={n}"]
        if layers is not None:
            head.append(f"Segment{i}_Layer:={layers[n]}")
    if extra:
        head.append(extra)
    path.write_bytes(("\n".join(head) + "\n\n").encode() + gzip.compress(np.asarray(values, np.uint8).tobytes(order="F")))
    return path


class TestLabels(unittest.TestCase):
    def setUp(self):
        import tempfile, pathlib
        self.dir = pathlib.Path(tempfile.mkdtemp())
        self.field = make_field(shape=(16, 64, 64))
        g = self.field.grid                                   # a label volume on (x, y, z) axes, as a scanner writes it:
        self.A = np.eye(4)                                    # finer than the model grid, and transposed against it
        self.A[:3, :3] = np.array([g.directions[2], g.directions[1], np.asarray(g.directions[0]) / 2]).T
        self.A[:3, 3] = np.asarray(g.origin) - np.asarray(g.directions[0]) / 4
        v = np.zeros((64, 64, 32), np.uint8)                  # (x, y, z)
        v[8:40, 8:56, 4:28] = 1                               # the conftest's "big" organ, in these axes
        v[48:56, 24:32, 12:16] = 2
        self.values, self.names = v, {"liver": 1, "kidney_left": 2}

    def test_a_seg_nrrd_is_read_with_its_names_and_its_place(self):
        p = write_seg(self.dir / "a.seg.nrrd", self.values, self.A, self.names,
                      extra='haversack_provenance:={"task": "ts.v2:total"}')
        lm = read_seg_nrrd(p)
        np.testing.assert_array_equal(lm.values, self.values)
        np.testing.assert_allclose(lm.affine_lps, self.A)
        self.assertEqual((lm.names, lm.task), (self.names, "ts.v2:total"))
        with self.assertRaises(KeyError) as e:
            lm.value_of("kidney_right")
        self.assertIn("kidney_left", str(e.exception))        # the refusal names what IS there

    def test_on_the_model_grid_it_is_the_fields_own_mask(self):
        gl = read_seg_nrrd(write_seg(self.dir / "b.seg.nrrd", self.values, self.A, self.names)).on_grid(self.field.grid)
        np.testing.assert_array_equal(gl.mask("liver"), self.field.native_mask == 1)
        np.testing.assert_array_equal(gl.mask("kidney_*"), self.field.native_mask == 2)
        self.assertEqual(gl.present(), ["kidney_left", "liver"])
        with self.assertRaises(KeyError):
            gl.mask("spleen")

    def test_ras_and_layered_files_are_read_and_the_rest_refused_by_name(self):
        lps = read_seg_nrrd(write_seg(self.dir / "l.seg.nrrd", self.values, self.A, self.names))
        ras = read_seg_nrrd(write_seg(self.dir / "r.seg.nrrd", self.values, self.A, self.names, space="right-anterior-superior"))
        np.testing.assert_allclose(ras.affine_lps, lps.affine_lps)
        np.testing.assert_array_equal(ras.values, lps.values)
        lesion = np.zeros_like(self.values); lesion[20:30, 20:30, 10:20] = 1          # drawn OVER the liver
        lay = read_seg_nrrd(write_seg(self.dir / "y.seg.nrrd", np.stack([self.values, lesion]), self.A,
                                      {**self.names, "liver_lesion": 1}, layers={"liver": 0, "kidney_left": 0, "liver_lesion": 1}))
        self.assertTrue(lay.layered)
        np.testing.assert_array_equal(lay.mask("liver_lesion"), lesion == 1)
        np.testing.assert_array_equal(lay.mask("liver", "liver_lesion"), (self.values == 1) | (lesion == 1))
        self.assertTrue(((self.values == 1) & (lesion == 1)).any())                 # they overlap: two layers needed
        with self.assertRaisesRegex(ValueError, "layered"):
            lay.on_grid(self.field.grid)

    def test_what_is_not_haversacks_is_refused_by_name(self):
        for kw, word in (({"dimension": 4}, "layered"), ({"space": "scanner-xyz"}, "left-posterior-superior")):
            with self.assertRaises(ValueError) as e:
                read_seg_nrrd(write_seg(self.dir / "c.seg.nrrd", self.values, self.A, self.names, **kw))
            self.assertIn(word, str(e.exception))
        p = write_seg(self.dir / "d.seg.nrrd", self.values, self.A, {"liver": 1, "hepar": 1})
        with self.assertRaises(ValueError):
            read_seg_nrrd(p)


class TestQueries(unittest.TestCase):
    def test_structures_find_their_organ_by_name(self):
        for s, q in (("kidney_left", "kidney"), ("vertebrae_L3", "lumbar vertebrae"), ("vertebrae_T12", "thoracic vertebrae"),
                     ("colon", "large bowel"), ("autochthon_right", "erector spinae muscle"), ("urinary_bladder", "bladder"),
                     ("lung_lower_lobe_left", "lung"), ("rib_left_7", "rib"), ("liver", "liver")):
            self.assertEqual(radar.query_for(s), q)
            self.assertIn(q, radar.ORGANS)
        for s in ("sternum", "spinal_cord", "vertebrae_S1", "costal_cartilages"):
            with self.assertRaises(KeyError):
                radar.query_for(s)
        self.assertEqual({q for _, q in radar.TS_QUERY_RULES} - set(radar.ORGANS), set())


class FakeHead(MeanPoolHead):
    """A head with queries: the query shifts the vector, so a wrong one shows."""
    queries = ("liver", "kidney")

    def pool(self, prepared, index, query=None, bias=None):
        v = super().pool(prepared, index, None, bias)
        return v + (0.0 if query == "liver" else 1.0)

    def score(self, vector, pair):
        return float(1 / (1 + np.exp(-(pair[1] - pair[0]) @ vector)))


class TestSession(unittest.TestCase):
    def setUp(self):
        self.field = make_field(shape=(16, 64, 64))
        self.labels = GridLabels(self.field.native_mask.copy(), {"liver": 1, "kidney_left": 2})
        self.s = Session.open(self.field, FakeHead(), self.labels, query_of=radar.query_for)

    def test_a_vector_by_name_is_the_gate_and_the_right_question(self):
        head = FakeHead(); prepared = head.prepare(self.field.all_tokens())
        for name, q in (("liver", "liver"), ("kidney_left", "kidney")):
            g = select(self.field, self.field.native_mask == self.labels.names[name])
            np.testing.assert_allclose(self.s.vector(name), head.pool(prepared, g.index, q), atol=1e-6)
            np.testing.assert_array_equal(self.s.gate(name).index, g.index)
        soft = self.s.vector("kidney_left", soft=True)
        self.assertFalse(np.allclose(soft, self.s.vector("kidney_left")))

    def test_two_structures_under_two_questions_need_one_named(self):
        with self.assertRaises(ValueError):
            self.s.vector("liver", "kidney_left")
        self.assertEqual(self.s.vector("liver", "kidney_left", query="liver").shape, (256,))

    def test_findings_are_the_organs_own_and_sorted(self):
        rng = np.random.default_rng(0)
        text = {f"{o}_{f}": rng.standard_normal((2, 256)) for o in ("Liver", "Kidney") for f in ("Cyst", "Mass", "Stone")}
        rows = self.s.findings("kidney_left", text=text)
        self.assertEqual({n.split("_")[0] for n, _ in rows}, {"Kidney"})
        self.assertEqual([p for _, p in rows], sorted((p for _, p in rows), reverse=True))
        t = self.s.table(text=text, min_voxels=1)
        self.assertEqual(set(t), {"liver", "kidney_left"}); self.assertEqual(t["kidney_left"]["query"], "kidney")

    def test_a_map_is_painted_where_the_boxes_are(self):
        sw, X = self.s.sweep("liver", size_mm=32.0)
        score = np.arange(len(sw.center), dtype=float)
        vol = self.s.volume(sw, score, within=self.s.mask("liver"))
        self.assertEqual(vol.shape, tuple(self.field.grid.shape))
        self.assertTrue(np.isnan(vol[~self.s.mask("liver")]).all())
        self.assertEqual(np.nanmax(vol), score.max())
        self.assertTrue(np.isfinite(vol[self.s.mask("liver")]).mean() > 0.9)

    def test_a_field_without_the_encoders_mask_is_a_field(self):
        arrays = {f"tokens{j}": t for j, t in enumerate(self.field.tokens)}
        g = self.field.grid
        meta = {"u": "x", "grid": {"shape": list(g.shape), "directions": [list(r) for r in g.directions], "origin": list(g.origin)}}
        f = radar.field_from_export(arrays, meta)
        self.assertIsNone(f.native_mask)
        self.assertEqual(Session.open(f, MeanPoolHead(), self.labels).vector("liver").shape, (256,))


if __name__ == "__main__":
    unittest.main()
