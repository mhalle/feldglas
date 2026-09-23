"""Encode CTs with RADAR on Modal and bring back token FIELDS - the one GPU step.

Writes, OUTSIDE the repository (``$FELDGLAS_CACHE`` or ``~/.cache/feldglas``):

    radar/head.npz                 the pooling head (one attention layer, a Linear per organ; 10.6 MB)
    radar/fields/<series>.npz      one field per series, ``feldglas-field 0.1``: three token lattices
                                   (fp16), RADAR's own 36-structure mask, and the model grid's geometry

Both derive from RADAR's CC BY-NC-SA 4.0 weights: research use only, never committed anywhere.

Ported from medseg's radar_instrument_modal.py and radar_response_modal.py (2026-09-20), which
established the pieces used here: the whole-volume encode (0.2-0.5 s on an L40S), upstream's
preprocessing re-implemented on the GPU so the crop and the resample are KNOWN (checked against
upstream's own DataFolder on every scan; about 1e-5), and the head's exactness in numpy (3e-8).
What is new is the geometry: the pilot's fields could be gated and pooled but not drawn on the
patient. Here the model grid is placed in the world from the LAS image's affine, the resample
(``align_corners=False``: output o reads input (o + 0.5) * in / out - 0.5) and the crop, written
in rankfield's form - LPS millimeters, one direction row per array axis.

Needs: a clone of upstream at 9319f36 (``$RADAR_REPO``, default ``../medseg/upstream/damo-radar``),
the Modal volumes of the RADAR study (``radar-idc-validation`` holds the CTs as ``/ct/<uuid>.nii.gz``,
``radar-probe-weights`` the checkpoint), and this package installed locally with the ``modal`` extra.

Run:  uv run --extra modal modal run tools/radar_export_modal.py::main --collection colorectal_liver_metastases --limit 4
      uv run --extra modal modal run tools/radar_export_modal.py::main --uuids <uuid>,<uuid>

TO THE OBJECT STORE (``--store s3://bucket/feldglas``). A cohort is hundreds of 24-37 MB fields
and must not travel through a laptop, so with ``--store`` the GPU worker hands its arrays to a
small CPU worker (Python 3.12, where feldglas and rankfield install - RADAR's own image is
3.10), which writes the field and puts it into provender's blobs under ``<store>/radar/``. Only
``{digest, size}`` comes back, into the MANIFEST (``manifests/radar.json``, tracked in git),
which is the one map from a series to its blob. The CPU worker reads the store's credentials
from the Modal secret ``$FELDGLAS_MODAL_SECRET`` (default ``feldglas-r2``: AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT, AWS_REGION=auto) - attached ONLY when ``--store`` names a
real bucket, so every other run works without the secret existing. ``--store memory://x``
exercises the whole path (image, field writer, blob put) with no bucket and no credentials;
what it stores dies with the container, so nothing is written to the manifest.

      FELDGLAS_STORE=s3://<bucket>/feldglas uv run --extra modal modal run tools/radar_export_modal.py::main --limit 4 --store env
"""
import csv, io, json, os, pathlib, sys, time

import modal

HERE = pathlib.Path(__file__).resolve().parent


def _up(n):
    return HERE.parents[n] if len(HERE.parents) > n else HERE          # /root/x.py inside the container


MEDSEG = _up(1) / "medseg"
REPO = pathlib.Path(os.environ.get("RADAR_REPO", MEDSEG / "upstream" / "damo-radar"))
CHOICE = MEDSEG / "docs" / "radar-idc-validation" / "results" / "validation" / "choice.csv"
# the SAME image definition as the study's scripts, so Modal reuses the image it already built
image = (modal.Image.debian_slim(python_version="3.10")
         .pip_install("torch==2.4.1", "numpy<2", "monai==1.4.0", "transformers==4.25.1", "SimpleITK",
                      "nibabel", "pandas", "tqdm", "huggingface_hub", "scipy", "httpx",
                      "idc-index", "highdicom>=0.23", "pydicom>=3", "pylibjpeg", "pylibjpeg-libjpeg")
         .add_local_dir(REPO / "RADAR_inference", "/radar/RADAR_inference")
         .add_local_dir(REPO / "ckpt", "/radar/ckpt"))
# Whether the store's secret is attached is decided ONCE, on the laptop, and BAKED INTO THE
# IMAGE's environment, so a container re-importing this file decides the same way. The first
# version read the variable on the laptop only and reasoned that a container's empty list was
# harmless: Modal checks a function's dependencies against the ones it was defined with, found
# 1 where the laptop had declared 2, and crash-looped the worker with the GPU containers still
# attached (2026-09-20). A Modal object under a conditional must evaluate alike on both sides.
_STORE = os.environ.get("FELDGLAS_STORE", "")
_SECRET = os.environ.get("FELDGLAS_MODAL_SECRET", "feldglas-r2")
_SECRETS = [modal.Secret.from_name(_SECRET)] if _STORE.startswith(("s3://", "gs://", "az://")) else []
# the finishing worker: where a field is WRITTEN and stored. feldglas is mounted from this
# checkout; rankfield and provender come from the same tags pyproject pins.
finish_image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
                .pip_install("numpy>=1.24", "obstore>=0.11",
                             "rankfield @ git+https://github.com/mhalle/rankfield.git@v0.3.2",
                             "provender @ git+https://github.com/mhalle/provender.git@v0.1.1")
                .env({"FELDGLAS_STORE": _STORE, "FELDGLAS_MODAL_SECRET": _SECRET})
                .add_local_python_source("feldglas"))
vol = modal.Volume.from_name("radar-idc-validation")
rweights = modal.Volume.from_name("radar-probe-weights")
app = modal.App("feldglas-radar-export")


@app.function(image=finish_image, secrets=_SECRETS, memory=4096, timeout=900, max_containers=12)
def finish(raw: bytes, store_url: str) -> str:
    """Raw export -> a field file -> provender's blobs. Returns ``{name, digest, size, provenance}``."""
    import tempfile
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.remote import open_blobs
    from feldglas.store import read_meta, write_field
    z = np.load(io.BytesIO(raw))
    meta = json.loads(str(z["meta"]))
    field = radar.field_from_export({k: z[k] for k in z.files if k != "meta"}, meta)
    with tempfile.TemporaryDirectory() as d:
        p = write_field(pathlib.Path(d) / f"{meta['u']}.npz", field)
        blobs = open_blobs(radar.NAME, store_url)
        blob = blobs.put_file(p)
        assert blobs.has(blob["digest"])
        return json.dumps({"name": p.name, **blob, "provenance": read_meta(p)["provenance"],
                           "tokens": int(field.offsets[-1]), "encode_s": meta["encode_s"],
                           "prep_max_abs_vs_upstream": meta["prep_max_abs_vs_upstream"]})


@app.cls(image=image, gpu="L40S", volumes={"/vol": vol, "/weights": rweights}, memory=65536,
         timeout=3600, max_containers=6)
class Export:
    @modal.enter()
    def load(self):
        os.environ["MODEL_ROOT"] = os.environ["CONFIGS_ROOT"] = "/weights"
        os.chdir("/radar/RADAR_inference"); sys.path.insert(0, ".")
        import inference_demo as D
        self.D = D
        _, self.model = D.initialize()

    @modal.method()
    def head(self) -> bytes:
        import numpy as np
        m = self.model
        f = lambda t: t.detach().float().cpu().numpy()
        buf = io.BytesIO()
        np.savez(buf, in_proj_weight=f(m.attention.in_proj_weight), in_proj_bias=f(m.attention.in_proj_bias),
                 out_proj_weight=f(m.attention.out_proj.weight), out_proj_bias=f(m.attention.out_proj.bias),
                 query_tokens=f(m.query_tokens), temp=f(m.temp),
                 vision_proj_weight=np.stack([f(p.weight) for p in m.vision_projs]),
                 vision_proj_bias=np.stack([f(p.bias) for p in m.vision_projs]))
        return buf.getvalue()

    # -- the pieces `field` and `phantom` share -------------------------------------------
    def _prep(self, arr, aff, crop=None):
        """Upstream's DataFolder on the GPU: ``(tensor, (lo, hi), resample target)``. ``crop``
        imposes another pass's crop, so two encodes of one scan share a grid exactly."""
        import numpy as np, torch, torch.nn.functional as F
        sp = np.abs(np.diag(aff)[:3])                           # upstream reads spacing off the diagonal
        h, w, dd = arr.shape
        tgt = [int(h * sp[1] / 1.0), int(w * sp[0] / 1.0), int(dd * sp[2] / 5.0)]   # upstream's x/y swap, kept
        x = F.interpolate(arr[None, None], size=tgt, mode="trilinear", align_corners=False)[0].permute(0, 3, 2, 1)
        x = x.clamp(-300, 400)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        if any(s > 1000 for s in x.shape[1:]):
            return None, (None, None), list(x.shape[1:])
        if crop is None:
            nz = x[0] > 0
            lo, hi = [], []
            for ax in range(3):
                idx = torch.nonzero(nz.any(dim=tuple(a for a in range(3) if a != ax))).flatten()
                lo.append(int(idx.min())); hi.append(int(idx.max()))
            lo = [max(l - e, 0) for l, e in zip(lo, (5, 20, 20))]
            hi = [min(m + e, s) for m, e, s in zip(hi, (5, 20, 20), x.shape[1:])]
        else:
            lo, hi = crop
        x = x[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        want = [int(np.ceil(max(s, m) / 32) * 32) for s, m in zip(x.shape[1:], (96, 256, 384))]
        pad = []
        for s, t in zip(reversed(x.shape[1:]), reversed(want)):
            pad += [0, t - s]
        return F.pad(x, pad)[None].float(), (lo, hi), tgt

    def _encode(self, base):
        import torch, torch.nn.functional as F
        vb = self.model.visual_encoder
        with torch.inference_mode():
            skips, segs = vb.UNet(base)
            toks = [g[0].flatten(1).T for g in (vb.proj1(skips[-1]), vb.proj2(skips[-2]), vb.proj3(skips[-3]))]
            lg = segs[0]
            lg = F.interpolate(lg, size=[lg.shape[-3], lg.shape[-2] * 2, lg.shape[-1] * 2], mode="trilinear",
                               align_corners=False)
            return toks, lg.softmax(1).argmax(1)[0]

    @staticmethod
    def _grid(aff, image_shape, tgt, lo, grid_shape) -> dict:
        """The model grid in the world, in rankfield's form (LPS mm, a direction row per array
        axis). Model axes (Z, Y, X) are the LAS image's axes (2, 1, 0); one model voxel is
        in/out image voxels (``align_corners=False``: output o reads input (o + 0.5) * in / out
        - 0.5); the first one sits at the crop's corner."""
        import numpy as np
        h, w, dd = image_shape
        step = [dd / tgt[2], w / tgt[1], h / tgt[0]]
        first = [(lo[2] + 0.5) * step[2] - 0.5, (lo[1] + 0.5) * step[1] - 0.5, (lo[0] + 0.5) * step[0] - 0.5]   # image (i, j, k)
        ras_to_lps = np.array([-1.0, -1.0, 1.0])
        return {"shape": [int(v) for v in grid_shape],
                "directions": [(aff[:3, 2] * step[0] * ras_to_lps).tolist(), (aff[:3, 1] * step[1] * ras_to_lps).tolist(),
                               (aff[:3, 0] * step[2] * ras_to_lps).tolist()],
                "origin": ((aff @ np.array([*first, 1.0]))[:3] * ras_to_lps).tolist()}

    @modal.method()
    def phantom(self, u: str, n_sites: int = 4) -> str:
        """Where does a sphere painted at a KNOWN place land? Two answers per site:

        grid    the sphere's centroid in the PREPROCESSED image (painted minus unpainted, on the
                model grid), carried to the world through the exported geometry, against where it
                was painted. No network involved: this is the affine, the resample and the crop.
        tokens  per lattice, the centroid of the token CHANGE against the site, along the model
                axes. A strided convolution (kernel 3, stride 2, padding 1) centers output i on
                input 2i, not on the middle of the two voxels it replaces, so a token may look at
                a point offset from the center of its own box - which a map has to know.
        """
        import nibabel as nib, numpy as np, torch
        from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
        from scipy import ndimage
        out = {"u": u, "sites": []}
        try:
            vol.reload()
            ct = nib.load(f"/vol/ct/{u}.nii.gz")
            ct = ct.as_reoriented(ornt_transform(io_orientation(ct.affine), axcodes2ornt(("L", "A", "S"))))
            arr = torch.as_tensor(np.asarray(ct.dataobj, np.float32)).cuda()
            aff = np.asarray(ct.affine, float)
            sp = np.abs(np.diag(aff)[:3]); h, w, dd = arr.shape
            base, crop, tgt = self._prep(arr, aff)
            toks, own = self._encode(base)
            grid = self._grid(aff, (h, w, dd), tgt, crop[0], base.shape[2:])
            G, origin = np.array(grid["directions"]), np.array(grid["origin"])
            vox = np.linalg.norm(G, axis=1)
            liver = ndimage.binary_erosion((own == 21).cpu().numpy(), np.ones((5, 25, 25), bool))
            cand = np.argwhere(liver)
            rng = np.random.default_rng(0)
            cand = cand[rng.choice(len(cand), min(2000, len(cand)), replace=False)]
            sites = [cand[0]]
            for c in cand:
                if min(np.linalg.norm((c - s_) * vox) for s_ in sites) > 60:
                    sites.append(c)
                if len(sites) >= n_sites:
                    break
            step = np.array([dd / tgt[2], w / tgt[1], h / tgt[0]])
            KERN = [(8, 32, 32), (4, 16, 16), (2, 8, 8)]
            shapes = [tuple(s_ // k for s_, k in zip(base.shape[2:], kk)) for kk in KERN]
            for m in sites:
                # the site as an INTEGER image voxel (i, j, k), so where it is in the world is exact
                img = np.round([(m[2] + crop[0][2] + 0.5) * step[2] - 0.5, (m[1] + crop[0][1] + 0.5) * step[1] - 0.5,
                                (m[0] + crop[0][0] + 0.5) * step[0] - 0.5]).astype(int)
                true_lps = (aff @ np.array([*img, 1.0]))[:3] * np.array([-1.0, -1.0, 1.0])
                r = 8.0
                sl = tuple(slice(max(int(c - r / s_) - 1, 0), int(c + r / s_) + 2) for c, s_ in zip(img, sp))
                g = torch.meshgrid(*[(torch.arange(e.start, min(e.stop, n), device=arr.device) - c) * float(s_)
                                     for e, n, c, s_ in zip(sl, arr.shape, img, sp)], indexing="ij")
                a2 = arr.clone()
                a2[sl] += 80.0 * (torch.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) <= r)
                painted, _, _ = self._prep(a2, aff, crop)
                d = (painted - base)[0, 0].abs().cpu().numpy()
                idx = np.argwhere(d > 0); wts = d[d > 0]
                centroid = (idx * wts[:, None]).sum(0) / wts.sum()
                found_lps = origin + centroid @ G
                site_model = np.array([(img[2] + 0.5) / step[0] - 0.5 - crop[0][0], (img[1] + 0.5) / step[1] - 0.5 - crop[0][1],
                                       (img[0] + 0.5) / step[2] - 0.5 - crop[0][2]])
                t2, _ = self._encode(painted)
                lat = []
                for j, (kk, shp) in enumerate(zip(KERN, shapes)):
                    ch = (torch.linalg.norm(t2[j] - toks[j], dim=1) ** 2).cpu().numpy()
                    ti = np.stack(np.unravel_index(np.arange(ch.size), shp), -1)
                    center = (ti + 0.5) * np.array(kk) - 0.5                      # each token's box center, model index
                    c = (center * ch[:, None]).sum(0) / ch.sum()
                    lat.append(((c - site_model) * vox).round(2).tolist())        # mm along model (Z, Y, X)
                out["sites"].append({"grid_error_mm": round(float(np.linalg.norm(found_lps - true_lps)), 3),
                                     "grid_error_lps_mm": (found_lps - true_lps).round(3).tolist(),
                                     "token_offset_mm_zyx": lat})
            out["voxel_mm_zyx"] = vox.round(3).tolist()
        except Exception as e:
            import traceback
            out["err"] = f"{type(e).__name__}: {e}"[:300]; out["trace"] = traceback.format_exc()[-1500:]
        return json.dumps(out)

    @modal.method()
    def field(self, u: str, store_url: str = "") -> bytes:
        import shutil, tempfile
        import nibabel as nib, numpy as np, torch, torch.nn.functional as F
        from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
        meta, arrays = {"u": u}, {}
        work = tempfile.mkdtemp()
        try:
            vol.reload()
            ct0 = nib.load(f"/vol/ct/{u}.nii.gz")                # the grid AS DELIVERED goes in the provenance
            ct = ct0.as_reoriented(ornt_transform(io_orientation(ct0.affine), axcodes2ornt(("L", "A", "S"))))
            arr = torch.as_tensor(np.asarray(ct.dataobj, np.float32)).cuda()
            aff = np.asarray(ct.affine, float)
            h, w, dd = arr.shape
            base, (lo, hi), tgt = self._prep(arr, aff)
            if base is None:                                    # upstream's own rule (non-axial series land here)
                return json.dumps({**meta, "skipped": tgt}).encode()

            # the check that keeps this honest: upstream's own preprocessing of the same scan
            os.makedirs(f"{work}/img"); nib.save(ct, f"{work}/img/{u}.nii.gz")
            ref, _, _ = self.D.DataFolder(f"{work}/img")[0]
            ref = ref.as_tensor() if hasattr(ref, "as_tensor") else ref
            d_, y_, x_ = (min(a, b) for a, b in zip(ref.shape[1:], base.shape[2:]))
            meta["prep_max_abs_vs_upstream"] = float((ref[0, :d_, :y_, :x_].cuda() - base[0, 0, :d_, :y_, :x_]).abs().max())

            t0 = time.time()
            toks, own = self._encode(base)
            torch.cuda.synchronize(); meta["encode_s"] = round(time.time() - t0, 2)
            assert tuple(own.shape) == tuple(base.shape[2:]), (own.shape, base.shape)
            meta.update(grid=self._grid(aff, (h, w, dd), tgt, lo, base.shape[2:]),
                        crop={"lo": lo, "hi": hi}, resample_target=tgt, image_shape=[h, w, dd],
                        image_affine_ras=aff.tolist(), input_shape=[int(v) for v in ct0.shape[:3]],
                        input_affine_ras=np.asarray(ct0.affine, float).tolist())
            for j, t in enumerate(toks):
                arrays[f"tokens{j}"] = t.half().cpu().numpy()
            arrays["native_mask"] = own.to(torch.uint8).cpu().numpy()
        except Exception as e:
            import traceback
            meta["err"] = f"{type(e).__name__}: {e}"[:300]; meta["trace"] = traceback.format_exc()[-1500:]
        finally:
            shutil.rmtree(work, ignore_errors=True)
        buf = io.BytesIO()
        if store_url and not meta.get("err"):
            np.savez(buf, meta=np.array(json.dumps(meta)), **arrays)     # uncompressed: it goes one hop, inside Modal
            return finish.remote(buf.getvalue(), store_url).encode()      # a small JSON record, not the field
        np.savez_compressed(buf, meta=np.array(json.dumps(meta)), **arrays)
        return buf.getvalue()


@app.local_entrypoint()
def main(uuids: str = "", collection: str = "", phase: str = "", limit: int = 4, choice: str = "",
         store: str = "", manifest: str = ""):
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.paths import encoder_dir
    from feldglas.remote import Manifest
    from feldglas.store import write_field

    store = _STORE if store == "env" else store
    if store.startswith(("s3://", "gs://", "az://")) and store != _STORE:
        raise SystemExit(f"--store {store} needs its credentials attached when the app is DEFINED: run with "
                         f"FELDGLAS_STORE={store} in the environment and pass --store env")
    mf = Manifest.load(manifest or HERE.parent / "manifests" / f"{radar.NAME}.json") if store else None

    if uuids:
        jobs = [u.strip() for u in uuids.split(",") if u.strip()]
    else:
        wanted = {c.strip() for c in collection.split(",") if c.strip()}           # one collection, or several
        rows = [r for r in csv.DictReader(open(choice or CHOICE)) if r["chosen"]
                and (not wanted or r["coll"] in wanted) and (not phase or r["phase"] == phase)]
        jobs = [r["chosen"] for r in rows]
        if mf is not None:                       # resumable: what the manifest already names is done
            done = {n[:-4] for n in mf.files}
            jobs = [u for u in jobs if u not in done]
            print(f"{len(done)} series already in the manifest; {len(jobs)} to go")
        jobs = jobs[:limit]
    out = encoder_dir(radar.NAME)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    ex = Export()
    (out / "head.npz").write_bytes(ex.head.remote())
    print(f"head -> {out / 'head.npz'}")
    t = time.time()
    for u, blob in zip(jobs, ex.field.starmap([(u, store) for u in jobs], return_exceptions=True)):
        if not isinstance(blob, (bytes, bytearray)):
            print(f"  {u}: {blob!r}"[:200]); continue
        if blob[:1] == b"{":
            rec = json.loads(blob.decode())
            if "digest" not in rec:
                print(f"  {u}: {blob.decode()[:200]}"); continue
            if not store.startswith("memory://"):            # a memory store died with its container
                mf.add(rec["name"], rec, rec["provenance"])
                if len(mf.files) % 25 == 0:                   # a long run that dies must not leave blobs
                    mf.save()                                 # the manifest does not name
            print(f"  {u}: {rec['tokens']} tokens -> {rec['digest'][:19]}... {rec['size'] / 1e6:.1f} MB, encode "
                  f"{rec['encode_s']} s, preprocessing vs upstream {rec['prep_max_abs_vs_upstream']:.1e}")
            continue
        z = np.load(io.BytesIO(blob))
        meta = json.loads(str(z["meta"]))
        if meta.get("err"):
            print(f"  {u}: {meta['err']}\n{meta.get('trace', '')}"); continue
        field = radar.field_from_export({k: z[k] for k in z.files if k != "meta"}, meta)
        p = write_field(out / "fields" / f"{u}.npz", field)
        print(f"  {u}: {int(field.offsets[-1])} tokens, {p.stat().st_size / 1e6:.1f} MB, encode {meta['encode_s']} s, "
              f"preprocessing vs upstream {meta['prep_max_abs_vs_upstream']:.1e}")
    if mf is not None and mf.files and not store.startswith("memory://"):
        print(f"manifest -> {mf.save()}  ({len(mf.files)} fields; commit it - it is the only map to these blobs)")
    print(f"{len(jobs)} series in {time.time() - t:.0f} s -> {store or out / 'fields'}")


@app.local_entrypoint()
def phantom(limit: int = 3, collection: str = "colorectal_liver_metastases", choice: str = ""):
    """Settle where a field IS with a painted sphere instead of another segmentation."""
    import numpy as np
    rows = [r for r in csv.DictReader(open(choice or CHOICE)) if r["chosen"] and r["coll"] == collection
            and r["phase"] == "venous"][:limit]
    got = [json.loads(x) for x in Export().phantom.map([r["chosen"] for r in rows])]
    sites = [s_ for g in got for s_ in g.get("sites", [])]
    for g in got:
        if g.get("err"):
            print(f"  {g['u']}: {g['err']}\n{g.get('trace', '')}")
    if not sites:
        return
    ge = np.array([s_["grid_error_mm"] for s_ in sites]); gv = np.array([s_["grid_error_lps_mm"] for s_ in sites])
    print(f"{len(sites)} spheres over {len(got)} scans; model voxel (Z, Y, X) mm: {got[0].get('voxel_mm_zyx')}")
    print(f"GRID   painted place vs the exported geometry: median {np.median(ge):.2f} mm, worst {ge.max():.2f} mm; "
          f"mean error vector (L, P, S) {gv.mean(0).round(2).tolist()} mm")
    for j, name in enumerate(("deep 8x32x32", "mid 4x16x16", "fine 2x8x8")):
        off = np.array([s_["token_offset_mm_zyx"][j] for s_ in sites])
        print(f"TOKENS {name:13} centroid of change minus the site, mm along model (Z, Y, X): "
              f"median {np.median(off, 0).round(1).tolist()}  (spread {off.std(0).round(1).tolist()})")
    from feldglas.paths import encoder_dir
    p = encoder_dir("radar") / "phantom.json"
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(got, indent=1))
    print(f"-> {p}")
