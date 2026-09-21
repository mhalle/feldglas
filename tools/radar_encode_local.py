"""Encode a CT with RADAR on THIS machine - Apple MPS, CUDA or CPU - and write a token field.

`radar_export_modal.py` is the cohort path (an L40S, 0.2-0.5 s a scan). This is the one-scan path
a viewer needs: no Modal round trip, no cold start. Measured 2026-09-20 on a 16 GB M2 with the
REAL checkpoint, against the field the L40S wrote for the same series (model grid 96 x 320 x 384,
105,120 tokens; the stored field is fp16, which is the 1e-3 floor of the last column):

    whole model, fp32   18.3 s, 11.3 GB   tokens AND RADAR's own mask (agreement 0.99999)   token cosine >= 0.999995
    encoder only, fp32   8.1 s,  9.3 GB   tokens only                                        token cosine >= 0.999995
    encoder only, fp16   3.3 s,  4.8 GB   tokens only                                        token cosine >= 0.99967 (median 0.999998)

and pooled organ vectors (liver, spleen, kidney, pancreas; same gate, numpy head) at cosine
>= 0.999996 in every mode. The grid's origin is identical (0.0 mm): preprocessing runs on the CPU.
The checkpoint loads with ``weights_only=True`` - no pickle is executed. With random weights the
whole model equals the CPU to 1e-6 (the CPU itself takes ~100 s).

The decoder is what RADAR's own 36-structure mask needs, and its ConvTranspose3d is refused by MPS
in fp16 - so fp16 is encoder-only here, and a field written that way has NO native mask: gate it
with haversack's labels (`feldglas.labels.read_seg_nrrd`, `feldglas.session.Session`). (haversack's ShuffleUp3d rewrite of transposed convolutions would lift
that; not needed yet.) The allocator is capped at the recommended working set, because past it
Metal returns ZEROS silently (haversack, 2026-09-03) - a real shortfall then raises instead.

Preprocessing and the grid are `radar_export_modal.py`'s, imported rather than restated, so the two
paths cannot drift. Upstream's own script is not used: it hard-codes `.cuda()` and pins torch 2.4.

Needs: a clone of upstream at 9319f36 (``$RADAR_REPO``, default ``../medseg/upstream/damo-radar``),
the checkpoint (``checkpoint_radar_pretrain.pth``; CC BY-NC-SA 4.0 - it stays in the cache, never in
a repository), torch and nibabel (haversack's dev environment has both).

Run:  PYTHONPATH=src ../haversack/.venv/bin/python tools/radar_encode_local.py CT.nii.gz [--device mps] [--fp16]
      ... --check ~/.cache/feldglas/radar/fields/<series>.npz     compare with a field the L40S wrote
"""
import argparse, json, os, pathlib, sys, time, types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = pathlib.Path(os.environ.get("RADAR_REPO", HERE.parents[1] / "medseg" / "upstream" / "damo-radar"))


def vision_branch(checkpoint, device, dtype):
    import torch
    if "monai" not in sys.modules:                      # upstream imports monai for ONE decorator; do not install it for that
        m, mu = types.ModuleType("monai"), types.ModuleType("monai.utils")
        mu.deprecated_arg = lambda *a, **k: (lambda f: f); m.utils = mu
        sys.modules["monai"], sys.modules["monai.utils"] = m, mu
    sys.path.insert(0, str(REPO / "RADAR_inference"))
    from dynamic_network_architectures.vision_branch import VisionBranch
    try:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as e:                              # a pickled checkpoint: say so rather than load it quietly
        print(f"note: weights_only load refused ({type(e).__name__}); loading the pickle - the same file the Modal runs load")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = {k[len("visual_encoder."):]: v for k, v in ck["model"].items() if k.startswith("visual_encoder.")}
    vb = VisionBranch().eval()
    missing, unexpected = vb.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"{len(missing)} vision weights missing from the checkpoint, e.g. {missing[:3]}")
    return vb.to(device, dtype)


def encode(vb, base, with_mask=True):
    """Tokens (deep, mid, fine) and, with the decoder, RADAR's own mask - as the export does it."""
    import torch, torch.nn.functional as F
    with torch.inference_mode():
        if with_mask:
            skips, segs = vb.UNet(base)
        else:
            skips, segs = vb.UNet.encoder(base), None
        toks = [g[0].flatten(1).T for g in (vb.proj1(skips[-1]), vb.proj2(skips[-2]), vb.proj3(skips[-3]))]
        own = None
        if segs is not None:
            lg = segs[0]
            lg = F.interpolate(lg, size=[lg.shape[-3], lg.shape[-2] * 2, lg.shape[-1] * 2], mode="trilinear", align_corners=False)
            own = lg.softmax(1).argmax(1)[0]
    return toks, own


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ct"); ap.add_argument("--device", default="auto"); ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-mask", action="store_true", help="encoder only: no native mask, about half the time and memory")
    ap.add_argument("--checkpoint", default=""); ap.add_argument("--out", default=""); ap.add_argument("--check", default="")
    a = ap.parse_args()
    import importlib.util
    import nibabel as nib, torch
    from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
    from feldglas.adapters import radar
    from feldglas.paths import encoder_dir
    from feldglas.store import read_field, write_field
    spec = importlib.util.spec_from_file_location("radar_export_modal", HERE / "radar_export_modal.py")
    ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)         # its _prep and _grid, not copies of them
    Export = ex.Export._get_user_cls() if hasattr(ex.Export, "_get_user_cls") else ex.Export

    dev = a.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if dev == "mps":
        torch.mps.set_per_process_memory_fraction(1.0)   # past the working set Metal returns zeros, silently
    dtype = torch.float16 if a.fp16 else torch.float32
    with_mask = not (a.no_mask or (a.fp16 and dev == "mps"))
    if a.fp16 and dev == "mps" and not a.no_mask:
        print("note: fp16 on MPS is encoder-only (ConvTranspose3d is refused in fp16): the field will have no native mask")
    ckpt = a.checkpoint or encoder_dir(radar.NAME) / "checkpoint_radar_pretrain.pth"
    t0 = time.time(); vb = vision_branch(ckpt, dev, dtype); t_load = time.time() - t0

    ct = nib.load(a.ct)
    ct = ct.as_reoriented(ornt_transform(io_orientation(ct.affine), axcodes2ornt(("L", "A", "S"))))
    arr = torch.as_tensor(np.asarray(ct.dataobj, np.float32)); aff = np.asarray(ct.affine, float)
    t0 = time.time()
    base, (lo, hi), tgt = Export._prep(None, arr, aff)   # on the CPU: one interpolate, and the same bits on every machine
    if base is None:
        raise SystemExit(f"upstream's own rule refuses this series (resampled shape {tgt})")
    t_prep = time.time() - t0
    x = base.to(dev, dtype)
    sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    t0 = time.time(); toks, own = encode(vb, x, with_mask); sync(); t_first = time.time() - t0
    t0 = time.time(); toks, own = encode(vb, x, with_mask); sync(); t_enc = time.time() - t0
    toks = [t.float().cpu().numpy() for t in toks]
    if not all(np.isfinite(t).all() for t in toks) or any(np.abs(t).max() == 0 for t in toks):
        raise SystemExit("the encode returned zeros or non-finite tokens: out of memory on this device (try --fp16 or --no-mask)")
    mem = torch.mps.driver_allocated_memory() / 1e9 if dev == "mps" else torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
    print(f"{dev} {str(dtype)[6:]}{'' if with_mask else ' (encoder only)'}: model grid {tuple(base.shape[2:])}, load {t_load:.1f} s, prep {t_prep:.1f} s, "
          f"encode {t_enc:.2f} s (first call {t_first:.2f} s), {mem:.1f} GB, {sum(len(t) for t in toks)} tokens")

    u = pathlib.Path(a.ct).name.split(".")[0]
    meta = {"u": u, "grid": Export._grid(aff, arr.shape, tgt, lo, base.shape[2:]), "crop": {"lo": lo, "hi": hi}, "resample_target": tgt,
            "image_shape": list(arr.shape), "image_affine_ras": aff.tolist(), "encode_s": round(t_enc, 2)}
    arrays = {f"tokens{j}": t.astype(np.float16) for j, t in enumerate(toks)}
    if own is not None:
        arrays["native_mask"] = own.to(torch.uint8).cpu().numpy()
    if a.check:
        ref = read_field(a.check)
        rep = {"grid_shape_equal": tuple(ref.grid.shape) == tuple(base.shape[2:]),
               "origin_max_abs_mm": float(np.abs(np.asarray(ref.grid.origin) - np.asarray(meta["grid"]["origin"])).max())}
        for j, (t, r) in enumerate(zip(toks, ref.tokens)):
            r = np.asarray(r, np.float32)
            cos = (t * r).sum(1) / (np.linalg.norm(t, axis=1) * np.linalg.norm(r, axis=1) + 1e-12)
            rep[f"lattice{j}"] = {"min_cosine": round(float(cos.min()), 6), "median_cosine": round(float(np.median(cos)), 6),
                                  "max_abs_diff / max_abs": round(float(np.abs(t - r).max() / np.abs(r).max()), 5)}
        if own is not None and ref.native_mask is not None:
            rep["mask_agreement"] = round(float((arrays["native_mask"] == ref.native_mask).mean()), 6)
        head = radar.RadarHead.load()                    # what a user sees: pooled vectors of whole organs, both fields
        pa, pb = head.prepare(np.concatenate(toks)), head.prepare(ref.all_tokens())
        from feldglas.gate import select
        vec = {}
        for organ in ("liver", "spleen", "kidney", "pancreas"):
            m = ref.native_mask == radar.ORGANS.index(organ) + 1
            if m.sum() > 500:
                g = select(ref, m)
                vec[organ] = round(float(head.pool(pa, g.index, organ) @ head.pool(pb, g.index, organ)), 6)
        rep["pooled_organ_vector_cosine (same gate, this encode vs the stored field)"] = vec
        print(json.dumps(rep, indent=1))
    if a.out or not a.check:
        field = radar.field_from_export(arrays, meta)     # no native mask after an encoder-only encode: gate it by name
        p = write_field(a.out or encoder_dir(radar.NAME) / "fields" / f"{u}.local.npz", field)
        print(f"-> {p}")


if __name__ == "__main__":
    main()
