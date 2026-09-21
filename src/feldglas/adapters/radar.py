"""RADAR (alibaba-damo-academy/damo-radar, Science 2026) behind the contract.

What RADAR is once its interface is removed (read from upstream at 9319f36, measured
2026-09-20; EXPLORATION section 2): a plain convolutional encoder whose skips at strides
(8, 32, 32), (4, 16, 16) and (2, 8, 8) are the three token lattices, and - the whole of what
follows the encoder - ONE 4-head attention layer with a learned query per organ, one Linear per
organ into the space the text shares, and a normalization. That head is 2.6 M parameters, and
this numpy version of it matched the GPU's vectors to 3e-8 on the exported fields, at 0.2 ms per
32 mm box gate. So the encode is the only GPU step (``tools/radar_export_modal.py``) and
everything a user does afterwards runs here.

LICENSE. RADAR's weights are CC BY-NC-SA 4.0. The head weights, every token field and every
vector pooled from one derive from them: research use only, share-alike, and never in this
repository (``paths.encoder_dir("radar")``). Upstream's code is Apache-2.0 and none of it is
copied here; the finding names are read out of the user's own clone (``english_names``).

Input facts an adapter's user has to know, because upstream does not check them: the scan must
be LAS (scores move by up to 0.72 otherwise), axial, and is resampled to 1 x 1 x 5 mm WITHOUT
anti-aliasing and clipped to [-300, 400] HU - so lung parenchyma and ground glass are flat, and
heavy noise reaches the encoder intact.
"""
from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
from rankfield.geometry import Geometry

from ..contract import Field, Provenance

NAME = "radar"
LICENSE = "CC-BY-NC-SA-4.0"
CODE = "damo-radar 9319f36"
WEIGHTS = "checkpoint_radar_pretrain.pth (HF radar-generalist/RADAR)"
KERNELS = [(8, 32, 32), (4, 16, 16), (2, 8, 8)]          # deep, mid, fine: model voxels per token
# upstream's 36 structures, in ITS order (the index is the query and the projection); label
# value v in RADAR's own mask is ORGANS[v - 1]
ORGANS = ("adrenal gland", "aorta", "erector spinae muscle", "brain", "clavicle", "large bowel", "duodenum",
          "esophagus", "face", "femur", "gallbladder", "gluteus muscle", "heart", "hip joint", "humerus",
          "iliac artery", "iliac vein", "iliopsoas muscle", "inferior vena cava", "kidney", "liver", "lung",
          "pancreas", "portal vein", "pulmonary artery", "rib", "sacrum", "scapula", "small bowel", "spleen",
          "stomach", "trachea", "bladder", "cervical vertebrae", "lumbar vertebrae", "thoracic vertebrae")
#: Where a token LOOKS, against the centre of its own box: the centroid of the token change a
#: painted sphere causes, minus the sphere's place, in mm along the model grid's (Z, Y, X) =
#: patient (superior, anterior, left). Measured 2026-09-20 (tools/radar_export_modal.py::phantom,
#: 11 spheres in 3 livers; spreads 0.3-0.4 / 1.0-1.3 / 3.6-4.9 mm). Every offset is POSITIVE: a
#: strided convolution centres its output on the first of the voxels it replaces, so a token sees
#: a point a little toward index 0 of where its box is drawn - 14-18 % of a token's width. Not
#: applied anywhere yet; a map that wants to be sharper than that should shift each lattice by it.
#: (The same check put the GRID itself within 0.10 mm of where a sphere was painted, so the
#: 1-2.5 mm by which RADAR's own mask sits superior of haversack's is the mask's, not the field's.)
LOOK_OFFSET_MM = {"deep": (6.6, 6.3, 7.6), "mid": (2.8, 2.2, 2.7), "fine": (1.1, 1.6, 1.4)}
#: Which of RADAR's 36 organ QUERIES a named structure is pooled under, for TotalSegmentator's
#: names (haversack `ts.v2:*`). By NAME and by rule, never by label value: values differ between
#: tasks and versions. Derived from upstream's own merge table (`process_img_mask.merged_organ_id`,
#: Apache-2.0) as the validation study used it, and checked name by name against that study's
#: value-indexed table on the 70 structures of an abdominal `ts.v2:total_fast` result (2026-09-20).
#: First match wins; a trailing ``*`` is a prefix. Structures with no RADAR organ (sternum, spinal
#: cord, costal cartilages, S1 ...) are absent on purpose: `query_for` raises rather than guess.
TS_QUERY_RULES = (
    ("adrenal_gland_*", "adrenal gland"), ("aorta", "aorta"), ("autochthon_*", "erector spinae muscle"),
    ("brain", "brain"), ("clavicula_*", "clavicle"), ("colon", "large bowel"), ("duodenum", "duodenum"),
    ("esophagus", "esophagus"), ("femur_*", "femur"), ("gallbladder", "gallbladder"), ("gluteus_*", "gluteus muscle"),
    ("heart", "heart"), ("hip_*", "hip joint"), ("humerus_*", "humerus"), ("iliac_artery_*", "iliac artery"),
    ("iliac_vena_*", "iliac vein"), ("iliopsoas_*", "iliopsoas muscle"), ("inferior_vena_cava", "inferior vena cava"),
    ("kidney_*", "kidney"), ("liver", "liver"), ("lung_*", "lung"), ("pancreas", "pancreas"),
    ("portal_vein_and_splenic_vein", "portal vein"), ("rib_*", "rib"), ("sacrum", "sacrum"), ("scapula_*", "scapula"),
    ("small_bowel", "small bowel"), ("spleen", "spleen"), ("stomach", "stomach"), ("trachea", "trachea"),
    ("urinary_bladder", "bladder"), ("vertebrae_C*", "cervical vertebrae"), ("vertebrae_L*", "lumbar vertebrae"),
    ("vertebrae_T*", "thoracic vertebrae"),
)


def query_for(structure: str, rules=TS_QUERY_RULES) -> str:
    """The organ query a named structure is pooled under. RADAR has one learned query and one
    projection per organ, so a gate is not enough - the question has to be the right organ's."""
    for pat, organ in rules:
        if structure == pat or (pat.endswith("*") and structure.startswith(pat[:-1])):
            return organ
    raise KeyError(f"RADAR has no organ query for {structure!r}: pass query= explicitly, or gate a structure it knows")


_E, _H = 256, 4


class RadarHead:
    """The attention layer and the per-organ projection, in numpy. Satisfies ``contract.Head``."""

    queries = ORGANS

    def __init__(self, weights: dict):
        w = {k: np.asarray(v, np.float32) for k, v in weights.items()}
        self._wq, self._wk, self._wv = (w["in_proj_weight"][i * _E:(i + 1) * _E] for i in range(3))
        self._bq, self._bk, self._bv = (w["in_proj_bias"][i * _E:(i + 1) * _E] for i in range(3))
        self._wo, self._bo = w["out_proj_weight"], w["out_proj_bias"]
        self._query, self._pw, self._pb = w["query_tokens"], w["vision_proj_weight"], w["vision_proj_bias"]
        self.temperature = float(w["temp"])

    @classmethod
    def load(cls, path=None) -> "RadarHead":
        from ..paths import encoder_dir
        path = pathlib.Path(path) if path else encoder_dir(NAME) / "head.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path}: no RADAR head - export one with tools/radar_export_modal.py "
                                    "(it derives from CC BY-NC-SA weights and is not distributed)")
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def _organ(self, query) -> int:
        if query is None:
            raise ValueError("RADAR pools under an organ's query: pass query='liver' (or its index)")
        return ORGANS.index(query) if isinstance(query, str) else int(query)

    def prepare(self, tokens: np.ndarray):
        """Keys and values for every token, once per scan (~0.1 s for ~120,000 tokens)."""
        t = np.asarray(tokens, np.float32)
        return ((t @ self._wk.T + self._bk).reshape(-1, _H, _E // _H),
                (t @ self._wv.T + self._bv).reshape(-1, _H, _E // _H))

    def attention(self, prepared, index, query, bias=None) -> np.ndarray:
        """``(n, heads)`` weights over the gate's tokens - where the vector was assembled from
        (upstream computes these and discards them; REPORT, Results I)."""
        q = (self._query[self._organ(query)] @ self._wq.T + self._bq).reshape(_H, _E // _H)
        a = np.einsum("nhd,hd->nh", prepared[0][index], q) / np.sqrt(_E // _H)
        if bias is not None:
            a = a + np.asarray(bias, np.float32)[:, None]
        a = np.exp(a - a.max(0))
        return a / a.sum(0)

    def pool(self, prepared, index, query=None, bias=None, weights=None) -> np.ndarray:
        """``weights`` (one per token, or per token and head) REPLACE the attention and keep the
        rest of the path - uniform weights are how the suite measures how much the attention
        matters (cosine 0.46-0.49 to the attention-pooled vector of the same gate)."""
        o = self._organ(query)
        if weights is None:
            a = self.attention(prepared, index, o, bias)
        else:
            a = np.asarray(weights, np.float32)
            a = np.repeat(a[:, None], _H, 1) if a.ndim == 1 else a
            a = a / a.sum(0)
        ctx = np.einsum("nh,nhd->hd", a, prepared[1][index]).reshape(_E)
        f = (ctx @ self._wo.T + self._bo) @ self._pw[o].T + self._pb[o]
        return f / np.linalg.norm(f)

    def score(self, vector: np.ndarray, pair: np.ndarray) -> float:
        """A finding's probability: ``pair`` is upstream's (negative, positive) text pair."""
        lg = (np.asarray(pair, np.float32) @ vector) / self.temperature
        return float(1.0 / (1.0 + np.exp(-(lg[1] - lg[0]))))


def load_text_table(path) -> dict[str, np.ndarray]:
    """Upstream's ``ckpt/infer_text_embedding_radar.pt`` as ``{"<organ>_<finding>": (2, 256)}``,
    keys as upstream spells them. The ``.pt`` needs torch (the ``radar`` extra); convert it
    ONCE with :func:`convert_text_table` and read the ``.npz`` afterwards without it. The table
    derives from RADAR's text tower, so the ``.npz`` belongs in the cache like the head."""
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    try:
        import torch
    except ImportError as e:
        raise ImportError(f"reading {path.name} needs torch: install feldglas[radar], or convert it once "
                          "with adapters.radar.convert_text_table") from e
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return {(k if isinstance(k, str) else "_".join(k)): v.detach().float().numpy() for k, v in raw.items()}


def convert_text_table(pt_path, npz_path) -> pathlib.Path:
    npz_path = pathlib.Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **load_text_table(pt_path))
    return npz_path


def english_names(inference_demo_py) -> dict[str, str]:
    """``{upstream key: "Organ_Finding"}``, PARSED out of the user's own clone of upstream
    (``RADAR_inference/inference_demo.py``) rather than copied here or imported - importing it
    would pull torch and MONAI for a dictionary literal."""
    tree = ast.parse(pathlib.Path(inference_demo_py).read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Attribute) and t.attr == "english_mapping" for t in node.targets)):
            return {str(k): str(v) for k, v in ast.literal_eval(node.value).items()}
    raise ValueError(f"{inference_demo_py}: no english_mapping dictionary found")


def provenance(source: str = "", **extra) -> Provenance:
    return Provenance(encoder=NAME, code=CODE, weights=WEIGHTS, preprocessing="radar-prep 0.1 (upstream's, on the GPU)",
                      license=LICENSE, source=source, extra=extra)


def field_from_export(arrays, meta: dict) -> Field:
    """The ``Field`` for what ``tools/radar_export_modal.py``'s GPU worker hands back: three
    token arrays, RADAR's own mask, and a ``meta`` holding the model grid (rankfield's form) and
    how the scan got there. ONE function, because two places build a field from it - the
    laptop, and the CPU worker that writes a field straight to the object store - and the
    format must not have two authors."""
    g = meta["grid"]
    extra = {k: meta[k] for k in ("crop", "resample_target", "image_shape", "image_affine_ras",
                                  "prep_max_abs_vs_upstream", "encode_s") if k in meta}
    return Field(tokens=[arrays[f"tokens{j}"] for j in range(len(KERNELS))], kernels=KERNELS,
                 grid=Geometry(shape=tuple(g["shape"]), directions=tuple(tuple(row) for row in g["directions"]),
                               origin=tuple(g["origin"])),
                 provenance=provenance(source=meta["u"], **extra),
                 native_mask=arrays.get("native_mask"), native_labels=ORGANS)   # None: an encoder-only encode (fp16 on MPS)


def read_pilot_field(path) -> Field:
    """A token field from the 2026-09-20 pilot (medseg's radar_instrument_modal.py), written
    before the exporter recorded its crop and resample: the grid has the right shape and spacing
    and an ARBITRARY origin and orientation (``exact_geometry`` False) - good for gating and
    pooling, not for drawing on the patient."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        tokens = [z[f"tokens{j}"] for j in range(3)]
        mask = z["own_mask"]
    return Field(tokens=tokens, kernels=KERNELS, grid=Geometry.aligned(tuple(meta["grid"]), tuple(meta["vox_mm"])),
                 provenance=provenance(source=meta.get("u", ""), pid=meta.get("pid", ""), pilot=True),
                 native_mask=mask, native_labels=ORGANS, exact_geometry=False)
