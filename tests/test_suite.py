import unittest

import pytest

from feldglas import suite
from conftest import make_field


class Registry(unittest.TestCase):
    def test_every_probe_says_what_it_asks_what_it_costs_and_where_it_stands(self):
        names = [p.name for p in suite.PROBES]
        self.assertEqual(len(names), len(set(names)))
        for p in suite.PROBES:
            self.assertTrue(p.question.endswith("?") or len(p.question) > 20, p.name)
            self.assertIn(p.needs, {"fields", "intervention", "pairs", "labels", "stores"}, p.name)
            self.assertIn(p.status, {"here", "medseg", "planned"}, p.name)
            self.assertTrue(p.where, p.name)

    def test_a_probe_marked_here_exists_here(self):
        import importlib
        for p in suite.by_status("here"):
            mod = importlib.import_module(p.where)
            self.assertTrue(callable(getattr(mod, p.name)), p.name)

    def test_the_variant_axis_is_part_of_the_suite(self):
        for name in ("mirror", "accessory_spleen", "identity_in_context", "correspondence"):
            self.assertEqual(suite.probe(name).status, "planned")
        with self.assertRaises(KeyError):
            suite.probe("no-such-probe")


class Gateability(unittest.TestCase):
    def test_a_small_structure_owns_no_coarse_token(self):
        pytest.importorskip("scipy")
        from feldglas.suite.gateability import gateability
        t = gateability(make_field(), min_voxels=10)
        self.assertEqual(set(t), {"big", "small"})
        self.assertEqual(t["small"]["inside"], [0, 0, 1])              # exactly one fine token, nothing coarser
        self.assertEqual(t["big"]["inside"][0], 0)                     # even the big organ owns no deep token
        self.assertGreater(t["big"]["admitted"][0], 0)
        self.assertGreater(t["big"]["inscribed_depth_mm"], t["small"]["inscribed_depth_mm"])
