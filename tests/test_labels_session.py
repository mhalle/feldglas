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


def write_seg(path, values, affine, names, extra="", dimension=3, space="left-posterior-superior"):
    d = affine[:3, :3].T
    head = ["NRRD0004", "# a test file", "type: unsigned char", f"dimension: {dimension}", f"space: {space}",
            "sizes: " + " ".join(str(v) for v in values.shape),
            "space directions: " + " ".join("(" + ",".join(repr(float(x)) for x in row) + ")" for row in d),
            "kinds: domain domain domain", "encoding: gzip",
            "space origin: (" + ",".join(repr(float(x)) for x in affine[:3, 3]) + ")"]
    for i, (n, v) in enumerate(names.items()):
        head += [f"Segment{i}_LabelValue:={v}", f"Segment{i}_Name:={n}"]
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

    def test_what_is_not_haversacks_is_refused_by_name(self):
        for kw, word in (({"dimension": 4}, "layered"), ({"space": "right-anterior-superior"}, "left-posterior-superior")):
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
