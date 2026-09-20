"""The RADAR head in numpy against the real thing.

Two checks, neither of which ships a weight. The first builds torch's own MultiheadAttention with
RANDOM weights, copies them into the head's file format and demands the same pooled vector - so
the arithmetic (in-projection split, per-head scaling, bias placement, output projection, the
per-organ Linear, the normalization) is pinned against the layer upstream uses. The second runs
only where the 2026-09-20 pilot's files exist (they derive from CC BY-NC-SA weights and are not
distributed): the GPU's own vectors for the same gates, stored beside the tokens, to 1e-5.
"""
import json, unittest

import numpy as np
import pytest

from feldglas import Head
from feldglas.adapters import radar
from feldglas.heads import MeanPoolHead
from feldglas.paths import pilot_dir


def random_head(seed=0):
    rng = np.random.default_rng(seed)
    n = len(radar.ORGANS)
    return {"in_proj_weight": rng.standard_normal((768, 256)) * 0.05, "in_proj_bias": rng.standard_normal(768) * 0.05,
            "out_proj_weight": rng.standard_normal((256, 256)) * 0.05, "out_proj_bias": rng.standard_normal(256) * 0.05,
            "query_tokens": rng.standard_normal((n, 256)), "temp": np.float32(0.0383),
            "vision_proj_weight": rng.standard_normal((n, 256, 256)) * 0.05,
            "vision_proj_bias": rng.standard_normal((n, 256)) * 0.05}


class Arithmetic(unittest.TestCase):
    def test_matches_torch_multihead_attention(self):
        torch = pytest.importorskip("torch")
        w = random_head()
        head = radar.RadarHead(w)
        mha = torch.nn.MultiheadAttention(256, 4, batch_first=True).eval()
        with torch.no_grad():
            mha.in_proj_weight.copy_(torch.as_tensor(w["in_proj_weight"], dtype=torch.float32))
            mha.in_proj_bias.copy_(torch.as_tensor(w["in_proj_bias"], dtype=torch.float32))
            mha.out_proj.weight.copy_(torch.as_tensor(w["out_proj_weight"], dtype=torch.float32))
            mha.out_proj.bias.copy_(torch.as_tensor(w["out_proj_bias"], dtype=torch.float32))
        rng = np.random.default_rng(1)
        tokens = rng.standard_normal((500, 256)).astype(np.float32)
        prepared = head.prepare(tokens)
        for organ in ("liver", "adrenal gland", 35):
            o = radar.ORGANS.index(organ) if isinstance(organ, str) else organ
            idx = np.sort(rng.choice(500, 60, replace=False))
            kv = torch.as_tensor(tokens[idx])[None]
            q = torch.as_tensor(w["query_tokens"][o], dtype=torch.float32)[None, None]
            with torch.no_grad():
                upd, att = mha(q, kv, kv, average_attn_weights=False)
                f = upd[0, 0] @ torch.as_tensor(w["vision_proj_weight"][o], dtype=torch.float32).T \
                    + torch.as_tensor(w["vision_proj_bias"][o], dtype=torch.float32)
                want = (f / f.norm()).numpy()
            np.testing.assert_allclose(head.pool(prepared, idx, organ), want, atol=2e-5)
            np.testing.assert_allclose(head.attention(prepared, idx, organ), att[0, :, 0].T.numpy(), atol=1e-5)

    def test_a_soft_gate_moves_weight_toward_the_tokens_it_favours(self):
        head = radar.RadarHead(random_head())
        tokens = np.random.default_rng(2).standard_normal((40, 256)).astype(np.float32)
        prepared, idx = head.prepare(tokens), np.arange(40)
        bias = np.where(idx < 5, 0.0, np.log(1e-3))
        a = head.attention(prepared, idx, "liver", bias).mean(1)
        self.assertGreater(a[:5].sum(), 0.9)
        np.testing.assert_allclose(np.linalg.norm(head.pool(prepared, idx, "liver", bias)), 1.0, atol=1e-5)

    def test_both_heads_satisfy_the_contract_and_radar_needs_a_query(self):
        head = radar.RadarHead(random_head())
        self.assertIsInstance(head, Head); self.assertIsInstance(MeanPoolHead(), Head)
        with self.assertRaises(ValueError):
            head.pool(head.prepare(np.zeros((3, 256), np.float32)), np.arange(3))

    def test_score_is_the_sigmoid_of_the_pair_at_the_heads_temperature(self):
        head = radar.RadarHead(random_head())
        v = np.zeros(256, np.float32); v[0] = 1.0
        pair = np.zeros((2, 256), np.float32); pair[1, 0] = 0.0383          # pos leads by exactly one temperature
        self.assertAlmostEqual(head.score(v, pair), 1 / (1 + np.exp(-1.0)), places=5)


class MeanPool(unittest.TestCase):
    def test_uniform_and_weighted(self):
        t = np.eye(4, dtype=np.float32)
        h = MeanPoolHead(); p = h.prepare(t)
        np.testing.assert_allclose(h.pool(p, np.arange(4)), np.full(4, 0.5), atol=1e-6)
        v = h.pool(p, np.arange(4), bias=np.log(np.array([1.0, 1e-9, 1e-9, 1e-9])))
        self.assertGreater(v[0], 0.999)


class AgainstThePilot(unittest.TestCase):
    """Opt-in by presence: the pilot's fields carry the GPU's own vectors for a handful of gates."""

    def test_cpu_head_reproduces_the_gpu_vectors(self):
        head_file, fields = pilot_dir() / "head" / "radar_head.npz", sorted((pilot_dir() / "fields").glob("*.npz"))
        if not head_file.exists() or not fields:
            self.skipTest("the 2026-09-20 pilot's files are not on this machine")
        head = radar.RadarHead.load(head_file)
        f = radar.read_pilot_field(fields[0])
        self.assertFalse(f.exact_geometry)
        prepared = head.prepare(f.all_tokens())
        with np.load(fields[0]) as z:
            gates = json.loads(str(z["meta"]))["gates"]
            self.assertGreater(len(gates), 3)
            for g in gates:
                got = head.pool(prepared, z[f"gate_{g['name']}_idx"], g["organ"])
                np.testing.assert_allclose(got, z[f"gate_{g['name']}_gpu16"], atol=1e-5)


class Weights(unittest.TestCase):
    def test_given_weights_replace_the_attention_and_keep_the_path(self):
        head = radar.RadarHead(random_head())
        tokens = np.random.default_rng(4).standard_normal((30, 256)).astype(np.float32)
        prepared, idx = head.prepare(tokens), np.arange(30)
        a = head.attention(prepared, idx, "liver")
        np.testing.assert_allclose(head.pool(prepared, idx, "liver", weights=a), head.pool(prepared, idx, "liver"), atol=1e-6)
        one = np.zeros(30, np.float32); one[7] = 1.0
        np.testing.assert_allclose(head.pool(prepared, idx, "liver", weights=one),
                                   head.pool(prepared, idx[7:8], "liver"), atol=1e-6)
