"""The null model: an encoder that was never shown a report.

Why it exists (EXPLORATION section 11). RADAR's representation cares about pathology because
400,000 radiology reports taught it to. Whether that can be had another way decides how seriously
to take an encoder of our own that has no such reports - and the cheapest way to find out is to
put a network that has only ever learned ANATOMY through the same suite: the encoder of the
TotalSegmentator network haversack already runs. If a donor atlas on those features finds tumors
nearly as well as RADAR's (0.966 / 0.936 / 0.946 pooled AUC whitened, EXPLORATION 5.6), report
supervision matters less than it looks; if it lands near chance, the reports are what made the
features pathology-aware. It is also the second adapter, which is what keeps the contract honest.

What was chosen, and why (2026-09-22):

- **The network is ``ts.v2:total_fast``** (Dataset 297, one PlainConvUNet, 3 mm isotropic, all
  117 classes): one encoder, where ``total`` is five part models at 1.5 mm each with its own.
  Apache-2.0 weights, so these fields are NOT bound by RADAR's CC BY-NC-SA - they still stay out
  of git (``.gitignore`` refuses every ``.npz``), but they may go anywhere.
- **The lattices are encoder stages 2, 3, 4**: strides 4 / 8 / 16 on the 3 mm grid, so token boxes
  of 12 / 24 / 48 mm - RADAR's fine / mid / deep are 10x8x8 / 20x16x16 / 40x32x32 mm, so the two
  are compared at nearly the same token sizes. Their widths differ (128 / 256 / 320) and nothing
  learned projects them to one: ``Field`` takes mixed widths (each lattice in its own channel
  block) and :func:`head` pools lattice by lattice (``heads.LatticeMeanHead``).
- **Tiled as the network was trained**, at its 112 x 112 x 128 patch - nnU-Net normalizes by
  instance, so a whole-volume pass would hand the encoder statistics it never saw. Every tile
  starts on a multiple of 16 voxels (the deepest stride), so each tile's tokens land exactly on
  the field's lattices, and overlapping tiles are blended with nnU-Net's own Gaussian, averaged
  over each token's box (:func:`token_weights`). The step is half a patch rounded DOWN to 16
  (48 / 48 / 64), a little more overlap than nnU-Net's 0.5. A token near a tile's edge has seen
  less context than one at its center; the blend favors centers, and the suite's point-spread
  probe should measure what is left before a map is trusted.
- **The native mask is TotalSegmentator's own labels** from the same tiles (argmax of the blended
  logits), mapped into RADAR's 36-organ scheme through ``adapters.radar.TS_QUERY_RULES`` so the
  atlas tools gate both encoders by the same organ names. Structures RADAR has no organ for
  (prostate, thyroid, the skull ...) are background here. The tiling is ours, not haversack's
  (aligned starts, another step), so these labels can differ from a haversack result by a few
  boundary voxels; they are a gate, and 5.11 measured that the gate source barely matters.
- **Pooling is by mean** (no learned query): vectors live in the encoder's token space and are
  not comparable with RADAR's. Only the SUITE's numbers are compared.

The encode (:func:`load_model`, :func:`encode`, :func:`check_against_haversack`) needs haversack
and so torch, imported inside those functions only. It uses weights already INSTALLED where
haversack keeps them - locally ``~/.totalsegmentator/nnunet/results`` (Dataset297 is there), on
Modal the global ``haversack-weights`` volume - and fetches nothing. Two callers, one encode:
``tools/null_encode_local.py`` (this machine, MPS) and ``tools/null_export_modal.py`` (the cohort).
"""
from __future__ import annotations

import math

import numpy as np
from rankfield.geometry import Geometry

from ..contract import Field, Provenance
from ..heads import LatticeMeanHead

NAME = "null-totalsegmentator"
TASK = "ts.v2:total_fast"
DATASET = 297
STAGES = (2, 3, 4)                                   # encoder stages kept, shallow to deep
KERNELS = ((4, 4, 4), (8, 8, 8), (16, 16, 16))        # their strides on the model grid
WIDTHS = (128, 256, 320)                              # their channels (min(32 * 2**stage, 320))
ALIGN = 16                                            # the deepest kept stride: tiles start on it
LICENSE = "Apache-2.0"                                # TotalSegmentator's open weights (haversack's attribution record)
PREPROCESSING = "null-1"                              # this adapter's tiling and blending rule


def head() -> LatticeMeanHead:
    return LatticeMeanHead(WIDTHS)


def padded_extent(n: int, patch: int, align: int = ALIGN) -> int:
    """The model grid's extent along one axis once padded at its END: at least a patch, and a
    multiple of ``align`` - so every tile start and every lattice fits exactly."""
    if patch % align:
        raise ValueError(f"patch {patch} is not a multiple of the alignment {align}")
    return max(patch, int(math.ceil(n / align)) * align)


def tile_starts(n: int, patch: int, align: int = ALIGN) -> list[int]:
    """Tile starts along one axis of a PADDED extent ``n``: every ``step`` (half a patch rounded
    down to ``align``) from 0, and one flush with the end. All are multiples of ``align``."""
    if n % align or n < patch:
        raise ValueError(f"extent {n} is not padded for patch {patch} and alignment {align}")
    step = max(align, (patch // 2) // align * align)
    starts = list(range(0, n - patch + 1, step))
    if starts[-1] != n - patch:
        starts.append(n - patch)
    return starts


def tile_slices(padded_shape, patch, kernels=KERNELS, align: int = ALIGN):
    """Every tile of a padded grid: ``(voxels, [tokens per lattice])``, each a tuple of three
    slices - where the tile's input comes from, and where each lattice's output tokens go. A tile
    starting at voxel ``s`` puts lattice ``j``'s token ``t`` at field token ``s // k + t``, exactly,
    because ``s`` is a multiple of every kernel."""
    if any(align % k for kk in kernels for k in kk):
        raise ValueError(f"alignment {align} is not a multiple of every kernel {kernels}")
    axes = [tile_starts(int(n), int(p), align) for n, p in zip(padded_shape, patch)]
    out = []
    for z in axes[0]:
        for y in axes[1]:
            for x in axes[2]:
                s = (z, y, x)
                vox = tuple(slice(a, a + int(p)) for a, p in zip(s, patch))
                toks = [tuple(slice(a // k, (a + int(p)) // k) for a, p, k in zip(s, patch, kk)) for kk in kernels]
                out.append((vox, toks))
    return out


def token_weights(gaussian: np.ndarray, kernel) -> np.ndarray:
    """nnU-Net's patch Gaussian averaged over each token's box: how much a tile's token counts
    in the blend. ``gaussian`` is the patch-shaped weight nnU-Net blends logits with."""
    g = np.asarray(gaussian, np.float64)
    k = tuple(int(v) for v in kernel)
    if any(s % kk for s, kk in zip(g.shape, k)):
        raise ValueError(f"kernel {k} does not tile the patch {g.shape}")
    s = [x // kk for x, kk in zip(g.shape, k)]
    return g.reshape(s[0], k[0], s[1], k[1], s[2], k[2]).mean(axis=(1, 3, 5))


def organ_mask(labels: np.ndarray, label_map: dict) -> np.ndarray:
    """TotalSegmentator labels (``label_map``: value -> structure name) -> RADAR's organ scheme
    (value v is ``radar.ORGANS[v - 1]``, 0 background), so both encoders gate by one vocabulary.
    A structure RADAR has no organ for becomes background."""
    from .radar import ORGANS, query_for
    lut = np.zeros(max(int(v) for v in label_map) + 1, np.uint8)
    for v, name in label_map.items():
        try:
            lut[int(v)] = ORGANS.index(query_for(name)) + 1
        except KeyError:
            pass
    labels = np.asarray(labels)
    return lut[np.clip(labels, 0, len(lut) - 1)] * (labels < len(lut))


def grid_from_haversack(shape, eff_zyx, origin_xyz, direction_xyz) -> Geometry:
    """haversack's model-grid geometry (``ranked_output.model_grid_geometry``: true spacing
    (Z, Y, X), first-voxel-center origin and ITK direction cosines, both LPS) -> rankfield's form,
    one direction row per array axis (Z, Y, X). ITK's direction matrix holds index axis i's
    direction in COLUMN i, with i over (x, y, z)."""
    D = np.asarray(direction_xyz, float).reshape(3, 3)
    rows = tuple(tuple(float(v) for v in D[:, 2 - a] * float(eff_zyx[a])) for a in range(3))
    return Geometry(shape=tuple(int(s) for s in shape), directions=rows,
                    origin=tuple(float(v) for v in origin_xyz))


def provenance(source: str = "", code: str = "", weights: str = "", **extra) -> Provenance:
    return Provenance(encoder=NAME, code=code, weights=weights or f"{TASK} (Dataset{DATASET})",
                      preprocessing=PREPROCESSING, license=LICENSE, source=source,
                      extra={"stages": list(STAGES), **extra})


# -- the encode: needs haversack (and so torch), imported only here ----------------------------
def load_model(weights_root=None, device: str = "auto", dtype: str = "fp16"):
    """``(haversack TorchModel, label map)`` for :data:`TASK`, from weights ALREADY installed -
    haversack's TotalSegmentator layout (``$TOTALSEG_WEIGHTS_PATH``, else
    ``~/.totalsegmentator/nnunet/results``; on Modal the global ``haversack-weights`` volume).
    Nothing is fetched: a missing model is haversack's ``ModelNotFound``, naming where it looked."""
    from haversack.network import TorchModel
    from haversack.tasks import TaskCatalog, resolve_model_folder, weights_root as _root
    spec = TaskCatalog("ts").get(TASK.split(":", 1)[1])
    root = _root("ts", weights_root)
    folder = resolve_model_folder(DATASET, model_root=root, **spec.model_choice(DATASET))
    m = TorchModel(folder, device=device, dtype=dtype)
    if m.transpose_forward != (0, 1, 2):
        raise RuntimeError(f"{folder.name}: a transposed model - the tiling here assumes none")
    if tuple(m.patch) and any(p % ALIGN for p in m.patch):
        raise RuntimeError(f"{folder.name}: patch {m.patch} is not a multiple of {ALIGN}")
    return m, {int(k): str(v) for k, v in spec.label_map.items()}


def encode(path, model, device=None):
    """CT -> ``(tokens per lattice (N, C) fp16, TS labels on the PADDED model grid, meta)``.

    haversack reads (canonical RAS), resamples (3 mm, corner rule, cubic) and normalizes exactly as
    for a segmentation, and places the model grid (``ranked_output.model_grid_geometry``); this
    tiles the grid padded at its end (:func:`tile_slices`), runs the network once per tile, keeps
    :data:`STAGES` through a forward hook on the encoder and blends with nnU-Net's Gaussian - per
    token box for the lattices (:func:`token_weights`), per voxel for the logits whose argmax is the
    native mask. ONE implementation for the Modal cohort tool and the local one."""
    import time
    import torch
    from haversack.io import read
    from haversack.preprocess import normalize_for, to_model_grid
    from haversack.ranked_output import model_grid_geometry
    m = model
    dev = torch.device(device) if device is not None else m.device
    arr, geom, orient = read(str(path))
    grid = to_model_grid(arr, geom, m.spacing_zyx, convention="corner", device=str(dev.type),
                         original_orientation=orient)
    x = normalize_for(grid, m)[0]                                        # (Z, Y, X) float32, normalized
    shape = tuple(int(s) for s in x.shape)
    padded = tuple(padded_extent(n, p) for n, p in zip(shape, m.patch))
    xp = torch.zeros(padded, dtype=m.dtype, device=dev)                  # nnU-Net pads with 0 after normalization
    xp[:shape[0], :shape[1], :shape[2]] = x.to(dev, m.dtype)
    gauss = m._gaussian_cpu.double().numpy()
    W = [torch.as_tensor(token_weights(gauss, k), dtype=torch.float32, device=dev) for k in KERNELS]
    G = torch.as_tensor(gauss, dtype=torch.float32, device=dev)
    acc = [torch.zeros((w, *(p // k for p, k in zip(padded, kk))), device=dev) for w, kk in zip(WIDTHS, KERNELS)]
    wsum = [torch.zeros(tuple(p // k for p, k in zip(padded, kk)), device=dev) for kk in KERNELS]
    logits = torch.zeros((m.K, *padded), dtype=torch.float16, device=dev)
    seen = {}
    hook = m.net.encoder.register_forward_hook(lambda mod, inp, out: seen.__setitem__("skips", out))
    tiles = tile_slices(padded, m.patch)
    t0 = time.time()
    try:
        with torch.inference_mode():
            for vox, toks in tiles:
                out = m.net(xp[vox][None, None])[0]                      # (K, *patch)
                for j, stage in enumerate(STAGES):
                    f = seen["skips"][stage][0].float()
                    if tuple(f.shape) != (WIDTHS[j], *W[j].shape):
                        raise RuntimeError(f"stage {stage}: {tuple(f.shape)}, expected {(WIDTHS[j], *W[j].shape)}")
                    acc[j][(slice(None), *toks[j])] += f * W[j]
                    wsum[j][toks[j]] += W[j]
                logits[(slice(None), *vox)] += (out.float() * G).half()  # argmax needs no division
            if dev.type == "cuda":
                torch.cuda.synchronize()
            elif dev.type == "mps":
                torch.mps.synchronize()
    finally:
        hook.remove()
    encode_s = time.time() - t0
    tokens = [(a / s).permute(1, 2, 3, 0).reshape(-1, a.shape[0]).half().cpu().numpy() for a, s in zip(acc, wsum)]
    labels = logits.argmax(0).to(torch.int16).cpu().numpy()
    labels[shape[0]:] = 0; labels[:, shape[1]:] = 0; labels[:, :, shape[2]:] = 0
    eff, origin, direction, centering = model_grid_geometry(
        {"frame": grid.frame.to_meta(), "model_grid": list(shape), "convention": "corner"})
    g = grid_from_haversack(padded, eff, origin, direction)
    meta = {"grid": {"shape": list(g.shape), "directions": [list(r) for r in g.directions], "origin": list(g.origin)},
            "model_shape": list(shape), "patch": list(m.patch), "tiles": len(tiles), "encode_s": round(encode_s, 2),
            "image_shape": [int(v) for v in arr.shape], "dtype": str(m.dtype).replace("torch.", ""),
            "centering": centering, "model": m.folder.name}
    return tokens, labels, meta


def check_against_haversack(path, labels, grid_meta: dict, label_map: dict, weights_root=None,
                            device: str = "auto", min_ml: float = 20.0) -> dict:
    """The independent test of the grid. Per organ of ``min_ml`` and up: the world centroid of OUR
    native mask through the field's geometry, against haversack's own ``segment`` of :data:`TASK`
    through SimpleITK's geometry of its label volume - mm apart (LPS), and the volume ratio. Two
    code paths meet only at the CT: a geometry error here is tens of mm, tiling differences a few
    boundary voxels."""
    import SimpleITK as sitk
    from haversack.pipeline import segment
    from .radar import ORGANS
    g = Geometry(shape=tuple(grid_meta["shape"]), directions=tuple(tuple(r) for r in grid_meta["directions"]),
                 origin=tuple(grid_meta["origin"]))
    ours = organ_mask(labels, label_map)
    img = segment(str(path), TASK, weights=weights_root, device=device).labels
    theirs = organ_mask(sitk.GetArrayFromImage(img), label_map)          # (Z, Y, X) on the source grid
    v_ours = abs(float(np.linalg.det(np.asarray(g.directions, float)))) * 1e-3
    v_theirs = float(np.prod(img.GetSpacing())) * 1e-3
    out = {}
    for v in np.unique(ours):
        if v == 0 or (ours == v).sum() * v_ours < min_ml or not (theirs == v).any():
            continue
        a = np.argwhere(ours == v).mean(0)
        b = np.argwhere(theirs == v).mean(0)
        pa = np.asarray(g.world(a))
        pb = np.asarray(img.TransformContinuousIndexToPhysicalPoint([float(b[2]), float(b[1]), float(b[0])]))
        out[ORGANS[int(v) - 1]] = {"mm": round(float(np.linalg.norm(pa - pb)), 2), "lps_mm": (pa - pb).round(2).tolist(),
                                   "volume_ratio": round(float((ours == v).sum() * v_ours / ((theirs == v).sum() * v_theirs)), 4)}
    return out


def field_from_export(arrays, meta: dict) -> Field:
    """The ``Field`` for what ``tools/null_export_modal.py`` hands back: one token array per kept
    stage, the organ-scheme mask, and a ``meta`` holding the grid in rankfield's form. One
    function, as for RADAR, because the format must not have two authors."""
    from .radar import ORGANS
    g = meta["grid"]
    extra = {k: meta[k] for k in ("model_shape", "patch", "tiles", "encode_s", "haversack", "image_shape",
                                  "dtype", "model", "centering", "device") if k in meta}
    return Field(tokens=[arrays[f"tokens{j}"] for j in range(len(KERNELS))], kernels=list(KERNELS),
                 grid=Geometry(shape=tuple(g["shape"]), directions=tuple(tuple(r) for r in g["directions"]),
                               origin=tuple(g["origin"])),
                 provenance=provenance(source=meta["u"], code=meta.get("haversack", ""), **extra),
                 native_mask=arrays.get("native_mask"), native_labels=ORGANS)
