"""The feldglas command, the reference file and the .seg.nrrd rules, against what the 2026-09-23
reviews found: a scan scored with another scan's labels, references used outside their terms,
tampered references, erosion that did not run in any test, one fixture's easy values hiding whole
branches. Fixtures here vary on purpose: a real erosion, two anomalies of different strength, a
lesion outside its organ."""
import gzip, json, os, pathlib, tempfile, unittest

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("click")
pytest.importorskip("scipy")

from click.testing import CliRunner

from feldglas import client as fc
from feldglas.cli import Region, _labels_path, main
from feldglas.labels import LabelMap, read_seg_nrrd
from feldglas.store import write_field
from test_client import cluster_field, rewrite
from test_labels_session import write_seg

BLOCK_A = [(z, y, x) for z in (3, 4) for y in (2, 3) for x in (2, 3)]          # strong
BLOCK_B = [(z, y, x) for z in (5, 6) for y in (5, 6) for x in (5, 6)]          # weaker, far from A (fine lattice 8^3)


def planted(seed, blocks):
    """A cluster field with blocks of fine tokens moved off the normal cloud by different amounts."""
    f = cluster_field(seed)
    lat_shape = tuple(s // k for s, k in zip(f.grid.shape, f.kernels[-1]))
    rng = np.random.default_rng(7)
    for block, scale in blocks:
        idx = np.ravel_multi_index(tuple(np.array(block).T), lat_shape)
        f.tokens[-1][idx] = f.tokens[-1][idx] * (1 - scale) + scale * 3 * rng.standard_normal(32).astype(np.float32)
    return f


class Base(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        old = os.environ.get("FELDGLAS_CACHE")
        os.environ["FELDGLAS_CACHE"] = str(self.d / "cache")
        self.addCleanup(lambda: os.environ.__setitem__("FELDGLAS_CACHE", old) if old else os.environ.pop("FELDGLAS_CACHE", None))
        g = cluster_field(0).grid
        self.A = np.eye(4); self.A[:3, :3] = np.asarray(g.directions, float).T; self.A[:3, 3] = g.origin
        lab = np.zeros(g.shape, np.uint8); lab[:, :, :] = 1; lab[:, :, 48:] = 2
        self.lab = lab
        for i in range(3):
            write_field(self.d / f"n{i}.zarr.zip", cluster_field(i))
            write_seg(self.d / f"n{i}.seg.nrrd", lab, self.A, {"liver": 1, "spleen": 2})
        write_field(self.d / "u.zarr.zip", planted(10, [(BLOCK_A, 1.0), (BLOCK_B, 0.6)]))
        write_seg(self.d / "u.seg.nrrd", lab, self.A, {"liver": 1, "spleen": 2})

    def cli(self, *args):
        return CliRunner().invoke(main, [str(a) for a in args])

    def build(self, erode="5", *extra):
        args = ["reference", "build", "-o", self.d / "ref.zarr.zip", "--structure", "liver", "--erode", erode, *extra]
        for i in range(3):
            args += ["--normal", self.d / f"n{i}.zarr.zip", self.d / f"n{i}.seg.nrrd"]
        return self.cli(*args, "--json")


class Build(Base):
    def test_a_build_reports_how_its_threshold_was_set(self):
        r = self.build()
        self.assertEqual(r.exit_code, 0, r.output)
        d = json.loads(r.stdout)
        self.assertEqual([p["normal"] for p in d["per_normal"]], ["s0", "s1", "s2"])
        self.assertAlmostEqual(d["threshold"], max(p["max"] for p in d["per_normal"]), places=6)
        ref = fc.Reference.load(self.d / "ref.zarr.zip")
        self.assertEqual((ref.meta["erode_mm"], ref.meta["structures"], ref.meta["sources"]), (5.0, ["liver"], ["s0", "s1", "s2"]))
        self.assertEqual(ref.meta["labels"], [str(self.d / f"n{i}.seg.nrrd") for i in range(3)])
        self.assertEqual(len(ref.centers), len(ref.tokens))

    def test_refusals_before_and_during_a_build(self):
        r = self.cli("reference", "build", "-o", self.d / "ref.zip", "--structure", "liver",
                     "--normal", self.d / "n0.zarr.zip", self.d / "n0.seg.nrrd", "--normal", self.d / "n1.zarr.zip", self.d / "n1.seg.nrrd")
        self.assertEqual(r.exit_code, 2); self.assertIn("zarr.zip", r.output)
        r = self.cli("reference", "build", "-o", self.d / "r.zarr.zip", "--structure", "liver", "--erode", "-1",
                     "--normal", self.d / "n0.zarr.zip", self.d / "n0.seg.nrrd", "--normal", self.d / "n1.zarr.zip", self.d / "n1.seg.nrrd")
        self.assertEqual(r.exit_code, 2)
        r = self.cli("reference", "build", "-o", self.d / "r.zarr.zip", "--structure", "liver", "--json",
                     "--normal", self.d / "n0.zarr.zip", self.d / "n0.seg.nrrd", "--normal", self.d / "n0.zarr.zip", self.d / "n0.seg.nrrd")
        self.assertEqual(r.exit_code, 1)
        self.assertEqual(json.loads(r.stdout)["ok"], False); self.assertIn("twice", json.loads(r.stdout)["error"])


class Score(Base):
    def score(self, *extra, labels=None):
        return self.cli("score", self.d / "u.zarr.zip", "--labels", labels or self.d / "u.seg.nrrd",
                        "--reference", self.d / "ref.zarr.zip", "--structure", "liver", "--json", *extra)

    def test_two_sites_strongest_first_matching_the_library(self):
        self.assertEqual(self.build("0").exit_code, 0)
        r = self.score()
        self.assertEqual(r.exit_code, 0, r.output)
        d = json.loads(r.stdout)
        ref = fc.Reference.load(self.d / "ref.zarr.zip")
        s = ref.score(fc.open_field(self.d / "u.zarr.zip"), fc.structure_mask(read_seg_nrrd(self.d / "u.seg.nrrd"), "liver"))
        self.assertEqual((d["flagged"], d["scored"]), (int(s.flagged.sum()), len(s.distance)))
        peaks = [x["peak"] for x in d["sites"]]
        self.assertGreaterEqual(len(peaks), 2); self.assertEqual(peaks, sorted(peaks, reverse=True))
        self.assertAlmostEqual(peaks[0], float(s.distance.max()), places=4)
        lat = fc.open_field(self.d / "u.zarr.zip").fine
        a = lat.centers[np.ravel_multi_index(tuple(np.array(BLOCK_A).T), lat.shape)].mean(0)
        self.assertLess(float(np.linalg.norm(np.asarray(d["sites"][0]["center_lps"]) - a)), 5.0)
        self.assertEqual((d["lattice"], d["layer"]), ("lattice_2", "kernel (2, 8, 8) of (8, 8, 8)"))

    def test_the_erosion_defaults_to_the_references_and_less_is_warned(self):
        self.assertEqual(self.build("5").exit_code, 0)
        d = json.loads(self.score().stdout)
        self.assertEqual(d["region"]["erode_mm"], 5.0)
        self.assertFalse(any("erode" in w for w in d["warnings"]))
        d = json.loads(self.score("--erode", "0").stdout)
        self.assertEqual(d["region"]["erode_mm"], 0.0)
        self.assertTrue(any("less than the reference's" in w for w in d["warnings"]))

    def test_other_structures_other_scans_and_in_sample_scores_are_named(self):
        self.assertEqual(self.build("0").exit_code, 0)
        r = self.cli("score", self.d / "u.zarr.zip", "--labels", self.d / "u.seg.nrrd", "--reference", self.d / "ref.zarr.zip",
                     "--structure", "spleen", "--json")
        self.assertEqual(r.exit_code, 1); self.assertIn("reference is for ['liver']", json.loads(r.stdout)["error"])
        r = self.cli("score", self.d / "n0.zarr.zip", "--labels", self.d / "n0.seg.nrrd", "--reference", self.d / "ref.zarr.zip",
                     "--structure", "liver", "--json")
        self.assertTrue(any("in-sample" in w for w in json.loads(r.stdout)["warnings"]))
        self.assertTrue(any("no input CT grid" in w for w in json.loads(r.stdout)["warnings"]))   # synthetic fields record none

    def test_labels_of_another_scan_are_refused(self):
        self.assertEqual(self.build("0").exit_code, 0)
        f = planted(10, [(BLOCK_A, 1.0)])
        from feldglas import Provenance
        f.provenance = Provenance(encoder="synthetic", weights="w", license="none", source="u2",
                                  input={"grid": {"shape": list(self.lab.shape), "directions": self.A[:3, :3].T.tolist(),
                                                  "origin": self.A[:3, 3].tolist()}})
        write_field(self.d / "u2.zarr.zip", f)
        ok = self.cli("score", self.d / "u2.zarr.zip", "--labels", self.d / "u.seg.nrrd", "--reference", self.d / "ref.zarr.zip",
                      "--structure", "liver", "--json")
        self.assertEqual(ok.exit_code, 0, ok.output); self.assertEqual(json.loads(ok.stdout)["warnings"], [])
        B = self.A.copy(); B[:3, 3] += 7.0
        write_seg(self.d / "other.seg.nrrd", self.lab, B, {"liver": 1, "spleen": 2})
        r = self.cli("score", self.d / "u2.zarr.zip", "--labels", self.d / "other.seg.nrrd", "--reference", self.d / "ref.zarr.zip",
                     "--structure", "liver", "--json")
        self.assertEqual(r.exit_code, 1); self.assertIn("not of the scan", json.loads(r.stdout)["error"])
        r = self.cli("score", self.d / "u2.zarr.zip", "--labels", self.d / "other.seg.nrrd", "--reference", self.d / "ref.zarr.zip",
                     "--structure", "liver", "--json", "--skip-grid-check")
        self.assertEqual(r.exit_code, 0); self.assertTrue(any("--skip-grid-check" in w for w in json.loads(r.stdout)["warnings"]))

    def test_optional_segments_report_what_they_matched(self):
        self.assertEqual(self.build("0").exit_code, 0)
        d = json.loads(self.score("--optional", "liver_tumor", "--optional", "spl*").stdout)
        self.assertEqual((d["region"]["optional_asked"], d["region"]["optional_matched"]), (["liver_tumor", "spl*"], ["spleen"]))
        r = self.score("--optional", "*")
        self.assertEqual(r.exit_code, 2)


class Vectors(Base):
    def test_structures_narrow_and_unknown_ones_are_refused_with_a_suggestion(self):
        r = self.cli("vectors", self.d / "u.zarr.zip", "--labels", self.d / "u.seg.nrrd", "--structure", "liver", "--json")
        d = json.loads(r.stdout)
        self.assertEqual(set(d["structures"]), {"liver"})
        near = d["structures"]["liver"]["nearest"]
        self.assertEqual(near[0]["structure"], "spleen"); self.assertGreater(d["structures"]["liver"]["tokens"], 0)
        r = self.cli("vectors", self.d / "u.zarr.zip", "--labels", self.d / "u.seg.nrrd", "--structure", "livr", "--json")
        self.assertEqual(r.exit_code, 1); self.assertIn("did you mean liver", json.loads(r.stdout)["error"])
        r = self.cli("vectors", self.d / "u.zarr.zip", "--labels", self.d / "u.seg.nrrd", "--names", "{bad")
        self.assertEqual(r.exit_code, 2); self.assertIn("--names", r.output)


class Info(Base):
    def test_info_describes_references_as_well_as_fields(self):
        self.assertEqual(self.build("0").exit_code, 0)
        d = json.loads(self.cli("info", self.d / "ref.zarr.zip", "--json").stdout)
        self.assertEqual((d["kind"], d["terms"]["structures"], len(d["per_normal"])), ("reference", ["liver"], 3))
        d = json.loads(self.cli("info", self.d / "u.zarr.zip", "--json").stdout)
        self.assertEqual(d["kind"], "field"); self.assertTrue(d["lattices"][2]["finest"])
        (self.d / "junk.zarr.zip").write_bytes(b"not a zip")
        r = self.cli("info", self.d / "junk.zarr.zip", "--json")
        self.assertEqual(r.exit_code, 1); self.assertNotIn("Traceback", r.output); self.assertFalse(json.loads(r.stdout)["ok"])


class Health(Base):
    def test_a_dependency_that_cannot_be_imported_is_a_problem(self):
        import feldglas.cli as cli
        real = cli._importable
        cli._importable = lambda m: "ImportError: blocked" if m == "scipy" else real(m)
        self.addCleanup(lambda: setattr(cli, "_importable", real))
        r = self.cli("health", "--json")
        self.assertEqual(r.exit_code, 1)
        d = json.loads(r.stdout)
        self.assertFalse(d["dependencies"]["scipy"]["importable"])
        self.assertTrue(any("scipy" in p for p in d["problems"]))


class Refs(Base):
    def test_a_reference_round_trips_every_field(self):
        normals = [(fc.open_field(self.d / f"n{i}.zarr.zip"),
                    fc.structure_mask(read_seg_nrrd(self.d / f"n{i}.seg.nrrd"), "liver")) for i in range(3)]
        ref = fc.Reference.build(normals, lattice=1, k=3, min_occupancy=0.7, drop_boundary=False, meta={"note": "x"})
        ref.save(self.d / "r.zarr.zip")
        back = fc.Reference.load(self.d / "r.zarr.zip")
        for name in ("threshold", "key", "lattice", "k", "min_occupancy", "drop_boundary", "meta"):
            self.assertEqual(getattr(back, name), getattr(ref, name), name)
        for name in ("tokens", "subject", "held_out", "centers"):
            np.testing.assert_array_equal(getattr(back, name), np.asarray(getattr(ref, name), getattr(back, name).dtype), name)
        self.assertEqual(sorted(p.name for p in self.d.iterdir() if p.name.startswith("r.")), ["r.zarr.zip"])
        with self.assertRaisesRegex(ValueError, "zarr.zip"):
            ref.save(self.d / "r.zip")

    def test_a_tampered_reference_is_refused(self):
        normals = [(fc.open_field(self.d / f"n{i}.zarr.zip"),
                    fc.structure_mask(read_seg_nrrd(self.d / f"n{i}.seg.nrrd"), "liver")) for i in range(3)]
        p = fc.Reference.build(normals).save(self.d / "r.zarr.zip")
        ext = lambda d: d["extensions"]["embedding"]
        for edit, why in ((lambda d: ext(d).update(threshold=float("nan")), "threshold"),
                          (lambda d: ext(d).update(threshold=5.0), "threshold"),
                          (lambda d: ext(d).update(min_occupancy=0), "min_occupancy"),
                          (lambda d: ext(d).update(drop_boundary="false"), "drop_boundary"),
                          (lambda d: ext(d).update(k=10 ** 6), "k"),
                          (lambda d: ext(d).update(format_version="9"), "this reader knows"),
                          (lambda d: ext(d).update(version="9"), "this reader knows"),
                          (lambda d: ext(d).pop("key"), "no 'key'")):
            with self.subTest(why=why), self.assertRaisesRegex(ValueError, why):
                fc.Reference.load(rewrite(p, "zarr.json", edit))


class Regions(unittest.TestCase):
    def test_erosion_is_in_mm_on_anisotropic_voxels_and_from_the_arrays_edge(self):
        A = np.diag([1.0, 1.0, 4.0, 1.0])                              # 1 x 1 x 4 mm voxels
        v = np.zeros((40, 40, 10), np.uint8); v[:, 5:35, 1:9] = 1        # touches the array's edge along x
        lm = LabelMap(v, A, {"organ": 1})
        r = Region(lm, ["organ"], (), 6.0)
        m = r.mask.values
        self.assertFalse(m[:6].any())                                    # the array's edge erodes like any surface
        self.assertTrue(m[10:30, 12:28, 3:7].all())
        self.assertFalse(m[:, :, 1].any() or m[:, :, 8].any())           # 4 mm slices: the outer ones are within 6 mm
        self.assertAlmostEqual(float(r.depth_at(np.array([[20.0, 20.0, 4 * 4.5]]))[0]), 15.0, places=4)
        self.assertEqual(float(r.depth_at(np.array([[20.0, 2.0, 16.0]]))[0]), 0.0)
        with self.assertRaisesRegex(ValueError, "empty after"):
            Region(lm, ["organ"], (), 50.0)


class LabelsPath(unittest.TestCase):
    def test_the_haversack_grammar_and_its_usage_errors(self):
        import click
        self.assertEqual(_labels_path(None, "https://s.example/", "idc:a:b", "ts.v2:total"),
                         "https://s.example/v1/idc/a:b/ts.v2:total/labels.seg.nrrd")
        for args in (("x", "h", None, None), (None, "h", None, "t"), (None, "h", "idc", "t"), (None, None, None, None)):
            with self.subTest(args=args), self.assertRaises(click.UsageError):
                _labels_path(*args)


class SegNrrdHeaders(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.v = np.zeros((4, 5, 6), np.uint8); self.v[1:3, 1:4, 2:5] = 1

    def write(self, head_edit=None, **kw):
        p = write_seg(self.d / "x.seg.nrrd", kw.pop("values", self.v), np.eye(4), kw.pop("names", {"organ": 1}), **kw)
        if head_edit:
            raw = p.read_bytes(); cut = raw.find(b"\n\n")
            p.write_bytes(head_edit(raw[:cut]) + raw[cut:])
        return p

    def test_each_header_rule(self):
        lay = np.stack([self.v, self.v])
        cases = [(dict(values=lay, names={"a": 1, "b": 1}, layers={"a": 0, "b": 1}),
                  lambda h: h.replace(b"space directions: none", b"space directions: (1,0,0)"), "list"),
                 (dict(), lambda h: h + b"\nSegment0_Layer:=1", "one-layer"),
                 (dict(values=lay, names={"a": 1, "b": 1}, layers={"a": 0, "b": 1}),
                  lambda h: h + b"\nSegment2_Name:=a\nSegment2_LabelValue:=1\nSegment2_Layer:=1", "two segments"),
                 (dict(), lambda h: h.replace(b"sizes: 4 5 6", b"sizes: 4 5 6 1"), "dimension 3")]
        for kw, edit, why in cases:
            with self.subTest(why=why), self.assertRaisesRegex(ValueError, why):
                read_seg_nrrd(self.write(edit, **kw))

    def test_raw_big_endian_and_a_named_zero(self):
        v16 = (self.v.astype("u2") * 300).astype(">u2")                      # big-endian bytes, as the header says
        head = ["NRRD0004", "type: unsigned short", "dimension: 3", "space: left-posterior-superior", "sizes: 4 5 6",
                "space directions: (1,0,0) (0,1,0) (0,0,1)", "kinds: domain domain domain", "endian: big", "encoding: raw",
                "space origin: (0,0,0)"]
        p = self.d / "raw.nrrd"
        p.write_bytes(("\n".join(head) + "\n\n").encode() + v16.tobytes(order="F"))
        lm = read_seg_nrrd(p, names={"background": 0, "organ": 300})
        np.testing.assert_array_equal(lm.values, self.v.astype(int) * 300)
        self.assertEqual(lm.names, {"background": 0, "organ": 300})


class SegMaskEdges(unittest.TestCase):
    def test_an_optional_segment_outside_the_organ_joins_and_a_present_prefix_matches(self):
        d = pathlib.Path(tempfile.mkdtemp())
        organ = np.zeros((10, 10, 10), np.uint8); organ[1:5, 1:5, 1:5] = 1
        cyst = np.zeros_like(organ); cyst[6:9, 6:9, 6:9] = 1                  # OUTSIDE the organ
        p = write_seg(d / "c.seg.nrrd", np.stack([organ, cyst]), np.eye(4), {"liver": 1, "liver_cyst_1": 1},
                      layers={"liver": 0, "liver_cyst_1": 1})
        m = fc.seg_mask(p, "liver", optional=["liver_cyst*"])
        np.testing.assert_array_equal(m.values, (organ == 1) | (cyst == 1))
        np.testing.assert_array_equal(fc.seg_mask(p, "liver").values, organ == 1)
