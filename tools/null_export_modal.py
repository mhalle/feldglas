"""Encode CTs with the NULL model on Modal - TotalSegmentator's ``total_fast`` encoder, through
haversack - and write token FIELDS. The null model's one GPU step; see ``feldglas.adapters.null``
for what is kept and why.

One container does everything: haversack reads the CT (canonical RAS), resamples it to 3 mm and
normalizes it exactly as for a segmentation; ``null.encode`` tiles the padded model grid at the
network's patch with every start on a multiple of 16, runs the network once per tile, keeps
encoder stages 2 / 3 / 4 through a forward hook, and blends tiles with nnU-Net's Gaussian (per
token box for the lattices, per voxel for the logits whose argmax is the native mask). The
model grid is placed in the world by haversack's own derivation
(``ranked_output.model_grid_geometry``) - never re-derived here.

Unlike RADAR's export there is no second worker: haversack runs on Python 3.12, where feldglas,
rankfield and provender install too, so the field is written and stored where it was encoded.
The weights are TotalSegmentator's open ones (Apache-2.0); the fields inherit that license and
say so, and they are not bound by RADAR's NC-SA boundary - they still never go into git.

Needs: the RADAR study's CT volume (``radar-idc-validation``, ``/ct/<uuid>.nii.gz``), haversack's
global weights volume (``haversack-weights``, which already holds Dataset297; mounted READ-ONLY -
nothing is fetched or installed), and ``$FELDGLAS_STORE`` for the object store, as for RADAR. The
encode itself is ``feldglas.adapters.null.encode``, shared with ``tools/null_encode_local.py``.

Run:  uv run --extra modal modal run tools/null_export_modal.py::main --uuids <uuid>,<uuid> --check
      uv run --extra modal modal run tools/null_export_modal.py::main --collection colorectal_liver_metastases --limit 2 --store memory://x
      FELDGLAS_STORE=s3://<bucket>/feldglas uv run --extra modal modal run tools/null_export_modal.py::main \
          --collection pancreas_ct,colorectal_liver_metastases,hcc_tace_seg,c4kc_kits --limit 1000 --store env

The cohort is RADAR's 560 (those four collections); ``choice.csv`` also names others, so pass them.

``--check`` also runs haversack's own ``segment`` of ``total_fast`` on each scan and compares, per
organ, where our native mask sits in the world (through the field's geometry) with where
haversack's label volume sits (through SimpleITK's): the independent test of the grid.
"""
import csv, io, json, os, pathlib, time

import modal

HERE = pathlib.Path(__file__).resolve().parent


def _up(n):
    return HERE.parents[n] if len(HERE.parents) > n else HERE


MEDSEG = _up(1) / "medseg"
CHOICE = MEDSEG / "docs" / "radar-idc-validation" / "results" / "validation" / "choice.csv"
HAVERSACK = "651aff741329e6b9542c3f88e025be0510b821e1"          # main, 2026-09-22 (rankfield v0.3.3 pin)
DTYPE = "fp16"                                                  # haversack's default for a segmentation
_STORE = os.environ.get("FELDGLAS_STORE", "")
_SECRET = os.environ.get("FELDGLAS_MODAL_SECRET", "feldglas-r2")
# decided on the laptop and baked into the image, so a container re-importing this decides alike
# (radar_export_modal.py has the story of the crash loop the other way cost)
_SECRETS = [modal.Secret.from_name(_SECRET)] if _STORE.startswith(("s3://", "gs://", "az://")) else []
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install(f"haversack @ git+https://github.com/mhalle/haversack.git@{HAVERSACK}",
                      "obstore>=0.11",
                      "rankfield @ git+https://github.com/mhalle/rankfield.git@v0.3.3",
                      "provender @ git+https://github.com/mhalle/provender.git@v0.1.1")
         .env({"FELDGLAS_STORE": _STORE, "FELDGLAS_MODAL_SECRET": _SECRET,
               "FELDGLAS_GPU": os.environ.get("FELDGLAS_GPU", "")})
         .add_local_python_source("feldglas"))
ctvol = modal.Volume.from_name("radar-idc-validation")
# the weights are already there (Dataset297 among them): read-only, so a run can never write into
# the volume every haversack deployment reads from
wvol = modal.Volume.from_name("haversack-weights").with_mount_options(read_only=True)
app = modal.App("feldglas-null-export")


# A fallback list: Modal takes the first with capacity. The first smoke (2026-09-22) waited on an
# L40S and never ran; total_fast is one 3 mm network, and the logit accumulator for a whole-body grid
# is ~2 GB, so any of these holds it. $FELDGLAS_GPU (comma-separated) overrides, baked in like the store.
GPUS = [g.strip() for g in os.environ.get("FELDGLAS_GPU", "L40S,A10,L4,A100-40GB").split(",") if g.strip()]


@app.cls(image=image, gpu=GPUS, volumes={"/vol": ctvol, "/weights": wvol}, secrets=_SECRETS,
         memory=49152, timeout=3600, max_containers=6)
class Export:
    @modal.enter()
    def load(self):
        from feldglas.adapters import null
        self.m, self.label_map = null.load_model("/weights", device="cuda", dtype=DTYPE)

    @modal.method()
    def field(self, u: str, store_url: str = "", check: bool = False) -> bytes:
        import tempfile
        import numpy as np
        from feldglas.adapters import null
        meta, arrays = {"u": u}, {}
        try:
            ctvol.reload()
            path = f"/vol/ct/{u}.nii.gz"
            tokens, labels, m = null.encode(path, self.m)
            import torch
            meta.update(m, haversack=HAVERSACK, device=f"cuda:{torch.cuda.get_device_name(0)}")
            for j, t in enumerate(tokens):
                arrays[f"tokens{j}"] = t
            arrays["native_mask"] = null.organ_mask(labels, self.label_map).astype(np.uint8)
            if check:
                meta["check"] = null.check_against_haversack(path, labels, meta["grid"], self.label_map,
                                                             "/weights", device="cuda")
        except Exception as e:
            import traceback
            meta["err"] = f"{type(e).__name__}: {e}"[:300]; meta["trace"] = traceback.format_exc()[-1500:]
        if store_url and not meta.get("err"):
            from feldglas.remote import open_blobs
            from feldglas.store import read_meta, write_field
            field = null.field_from_export(arrays, meta)
            with tempfile.TemporaryDirectory() as d:
                p = write_field(pathlib.Path(d) / f"{u}.npz", field)
                blobs = open_blobs(null.NAME, store_url)
                blob = blobs.put_file(p)
                assert blobs.has(blob["digest"])
                return json.dumps({"name": p.name, **blob, "provenance": read_meta(p)["provenance"],
                                   "tokens": int(field.offsets[-1]), "encode_s": meta["encode_s"],
                                   "tiles": meta["tiles"], "device": meta["device"],
                                   "check": meta.get("check")}).encode()
        buf = io.BytesIO()
        np.savez_compressed(buf, meta=np.array(json.dumps(meta)), **arrays)
        return buf.getvalue()


@app.local_entrypoint()
def main(uuids: str = "", collection: str = "", phase: str = "", limit: int = 4, choice: str = "",
         store: str = "", manifest: str = "", check: bool = False):
    import numpy as np
    from feldglas.adapters import null
    from feldglas.paths import encoder_dir
    from feldglas.remote import Manifest
    from feldglas.store import write_field

    store = _STORE if store == "env" else store
    if store.startswith(("s3://", "gs://", "az://")) and store != _STORE:
        raise SystemExit(f"--store {store} needs its credentials attached when the app is DEFINED: run with "
                         f"FELDGLAS_STORE={store} in the environment and pass --store env")
    mf = Manifest.load(manifest or HERE.parent / "manifests" / f"{null.NAME}.json") if store else None
    if uuids:
        jobs = [u.strip() for u in uuids.split(",") if u.strip()]
    else:
        wanted = {c.strip() for c in collection.split(",") if c.strip()}
        rows = [r for r in csv.DictReader(open(choice or CHOICE)) if r["chosen"]
                and (not wanted or r["coll"] in wanted) and (not phase or r["phase"] == phase)]
        jobs = [r["chosen"] for r in rows]
        if mf is not None:                       # resumable: what the manifest already names is done
            done = {n[:-4] for n in mf.files}
            jobs = [u for u in jobs if u not in done]
            print(f"{len(done)} series already in the manifest; {len(jobs)} to go")
        jobs = jobs[:limit]
    out = encoder_dir(null.NAME) / "fields"
    t = time.time()
    checks = {}
    for u, blob in zip(jobs, Export().field.starmap([(u, store, check) for u in jobs], return_exceptions=True)):
        if not isinstance(blob, (bytes, bytearray)):
            print(f"  {u}: {blob!r}"[:300]); continue
        if blob[:1] == b"{":
            rec = json.loads(blob.decode())
            if "digest" not in rec:
                print(f"  {u}: {blob.decode()[:300]}"); continue
            if not store.startswith("memory://"):
                mf.add(rec["name"], {k: rec[k] for k in ("digest", "size")}, rec["provenance"])
                if len(mf.files) % 25 == 0:
                    mf.save()
            if rec.get("check"):
                checks[u] = rec["check"]
            print(f"  {u}: {rec['tokens']} tokens, {rec['tiles']} tiles -> {rec['digest'][:19]}... "
                  f"{rec['size'] / 1e6:.1f} MB, encode {rec['encode_s']} s on {rec['device']}")
            continue
        z = np.load(io.BytesIO(blob))
        meta = json.loads(str(z["meta"]))
        if meta.get("err"):
            print(f"  {u}: {meta['err']}\n{meta.get('trace', '')}"); continue
        field = null.field_from_export({k: z[k] for k in z.files if k != "meta"}, meta)
        p = write_field(out / f"{u}.npz", field)
        if meta.get("check"):
            checks[u] = meta["check"]
        print(f"  {u}: {int(field.offsets[-1])} tokens, {meta['tiles']} tiles, {p.stat().st_size / 1e6:.1f} MB, "
              f"encode {meta['encode_s']} s, grid {meta['model_shape']} -> {list(field.grid.shape)}")
    for u, c in checks.items():
        mm = sorted((v["mm"], k) for k, v in c.items())
        vr = [v["volume_ratio"] for v in c.values()]
        print(f"  check {u}: {len(c)} organs, centroid gap median {np.median([a for a, _ in mm]):.2f} mm, "
              f"worst {mm[-1][0]:.2f} mm ({mm[-1][1]}); volume ratio {min(vr):.3f}-{max(vr):.3f}")
    if checks:
        p = encoder_dir(null.NAME) / "geometry_check.json"
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(checks, indent=1))
        print(f"-> {p}")
    if mf is not None and mf.files and not store.startswith("memory://"):
        print(f"manifest -> {mf.save()}  ({len(mf.files)} fields; commit it - it is the only map to these blobs)")
    print(f"{len(jobs)} series in {time.time() - t:.0f} s -> {store or out}")
