"""WP2 on Modal, beside the fields: region vectors for every scan, then the normal atlas and FROC.

The cohort's fields are 20 GB on the object store and the laptop this was written on had 52 GB
free, so nothing here brings a field home. Two CPU steps, no GPU:

  ::main      per scan - fetch the field from the store (verified), sweep every organ of haversack's
              `ts.v2:total` labels with 64 and 32 mm boxes (and 16 mm for the small solid organs), pool each
              box on the CPU head, and, where the series has an expert SEG in IDC, scatter its
              "Mass" segments onto the model grid as per-lesion occupancy. One ``regions/<series>.npz``
              per scan on the Modal volume ``feldglas-radar-work``: vectors (fp16), the box table,
              box x lesion overlaps, the lesion table. Resumable: a scan whose file exists is skipped.
  ::analysis  one container reads them all - the atlas from the 80 healthy donors (total, within-
              donor, positioned), every detector on every patient, AUC under both positive rules
              (per patient and pooled), per-lesion FROC by lesion size, the donor-count curve, phase
              match and mismatch, the gate rule, a donor-held-out control - and hands back JSON.

Everything on the work volume derives from RADAR's CC BY-NC-SA 4.0 weights (the head, the text
table, region vectors, atlas models): research use only, never in a repository. The JSON of
measurements is what comes home.

Needs the manifest (``manifests/radar.json``), medseg's cohort tables (``choice.csv``,
``expert_segs.csv``, ``radar_validation_series_all.csv``), a local head and text table
(``tools/radar_export_modal.py`` wrote them to the cache), and ``$FELDGLAS_STORE`` - baked into
the image so the container opens the same store (see radar_export_modal.py for why).

Run:  FELDGLAS_STORE=s3://<bucket>/feldglas uv run --no-sync modal run tools/radar_atlas_modal.py::main --limit 6
      FELDGLAS_STORE=s3://<bucket>/feldglas uv run --no-sync modal run tools/radar_atlas_modal.py::main
      FELDGLAS_STORE=s3://<bucket>/feldglas uv run --no-sync modal run tools/radar_atlas_modal.py::analysis

**The gate is haversack's, never the encoder's** (2026-09-22, the user's decision: a field carries no
mask). Labels come BY PATH from the read-only twin of the study server (``$HAVERSACK_TWIN``, cache
hits only, no credentials; ``ts.v2:total``, else ``ts.v2:total_fast``), are read with
``feldglas.labels`` and pulled onto the field's model grid. An organ's gate is the union of the
structures ``adapters.radar.TS_QUERY_RULES`` files under it - kidney = ``kidney_left``,
``kidney_right`` and ``kidney_cyst_*`` - so every encoder is swept over the SAME organs from the SAME
segmentation. EXPLORATION 5.11 measured that the gate's source barely matters; before this, each
encoder was swept over its own mask. A series the twin has no labels for is an error, not a skip.

The NULL model's fields (``tools/null_export_modal.py``) go through the same steps with
``FELDGLAS_ENCODER=null-totalsegmentator``: its own manifest, work volume (``feldglas-null-work``)
and results (``normal_atlas_null.json``), the mean-per-lattice head, and no RADAR-finding
detectors. The organ names are RADAR's either way, so boxes are drawn over the same organs by the
same rules.
"""
import csv, functools, io, json, os, pathlib, time

import modal

HERE = pathlib.Path(__file__).resolve().parent


def _up(n):
    return HERE.parents[n] if len(HERE.parents) > n else HERE


MEDSEG = _up(1) / "medseg" / "docs" / "radar-idc-validation"
_STORE = os.environ.get("FELDGLAS_STORE", "")
# WHICH ENCODER's fields (2026-09-22, the null model): "radar" or "null-totalsegmentator". Baked
# into the image as the store is, so a container decides alike. Each encoder has its own work
# volume and results file; RADAR keeps the names it had. The null model has no head of its own
# (mean per lattice) and no vocabulary, so the two RADAR-finding detectors run for RADAR only.
ENCODER = os.environ.get("FELDGLAS_ENCODER", "radar")
# variant -> (whose fields, its work volume, its results' suffix, how a gate is pooled). RADAR
# keeps the names it had. "radar-mean" (2026-09-22) is RADAR's own stored fields pooled WITHOUT its
# head - the mean per lattice, as the null models are - the pooling control 5.12 left open: can a
# normal atlas see disease in raw tokens, with no learned attention or projection?
ENCODERS = {"radar": ("radar", "feldglas-radar-work", "", "head"),
            "radar-mean": ("radar", "feldglas-radarmean-work", "_radar_mean", "mean"),
            "null-totalsegmentator": ("null-totalsegmentator", "feldglas-null-work", "_null", "mean"),
            "null-totalsegmentator-1.5mm": ("null-totalsegmentator-1.5mm", "feldglas-null15-work", "_null15", "mean")}
if ENCODER not in ENCODERS:
    raise SystemExit(f"FELDGLAS_ENCODER={ENCODER!r}: one of {sorted(ENCODERS)}")
FIELDS, _VOLUME, SUFFIX, POOL = ENCODERS[ENCODER]
TWIN = os.environ.get("HAVERSACK_TWIN", "")          # the haversack server holding the labels (its read-only twin will do)
LABEL_TASKS = ("ts.v2:total", "ts.v2:total_fast")    # the first the twin has, in this order
_SECRET = os.environ.get("FELDGLAS_MODAL_SECRET", "feldglas-r2")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("numpy>=1.24", "scipy", "obstore>=0.11", "idc-index", "highdicom>=0.23", "pydicom>=3",
                      "pylibjpeg", "pylibjpeg-libjpeg",
                      "rankfield @ git+https://github.com/mhalle/rankfield.git@v0.3.2",
                      "provender @ git+https://github.com/mhalle/provender.git@v0.1.1")
         .env({"FELDGLAS_STORE": _STORE, "FELDGLAS_ENCODER": ENCODER})
         .add_local_python_source("feldglas"))
work = modal.Volume.from_name(_VOLUME, create_if_missing=True)
app = modal.App("feldglas-radar-atlas" if ENCODER == "radar" else f"feldglas-{ENCODER.replace('totalsegmentator', 'ts').replace('.', '')}-atlas")

# cohort -> (the organ its expert SEG marks, RADAR's finding for it)
TARGET = {"hcc_tace_seg": ("liver", "Liver_Hepatocellular carcinoma"),
          "colorectal_liver_metastases": ("liver", "Liver_Metastasis"),
          "c4kc_kits": ("kidney", "Kidney_Renal cell carcinoma")}
DONORS = "pancreas_ct"
SIZES = (64, 32)
SMALL = {"liver", "kidney", "spleen", "pancreas", "adrenal gland", "gallbladder"}     # these get 16 mm boxes too
HALF_RULE = {"liver", "kidney"}                     # and these a second, majority gate at 32 mm
MIN_ORGAN_ML, MAX_BOXES = 5.0, 9000


@app.function(image=image, secrets=[modal.Secret.from_name(_SECRET)], volumes={"/work": work}, cpu=2,
              memory=8192, timeout=2400, max_containers=40)
def regions(u: str, digest: str, pid: str, coll: str, phase: str, seg_series: str, force: bool = False,
            twin: str = TWIN) -> str:
    import glob, tempfile
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.remote import open_blobs
    from feldglas.store import read_field
    from feldglas.suite import normal_atlas as na
    out_path = pathlib.Path(f"/work/regions/{u}.npz")
    if out_path.exists() and not force:
        return json.dumps({"u": u, "cached": True})
    t0 = time.time()
    meta = {"u": u, "pid": pid, "coll": coll, "phase": phase, "seg_series": seg_series}
    try:
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / f"{u}.npz"
            if not open_blobs(FIELDS, check=False).fetch(digest, p):
                raise FileNotFoundError(f"blob {digest} is gone, or did not match its name")
            field = read_field(p)
            organs, meta["labels_task"] = _organ_gates(field, u, d, twin)
            head = _head(field)
            prepared = head.prepare(field.all_tokens(blocks=True) if POOL == "mean" else field.all_tokens())
            voxel_ml = float(np.prod(field.grid.spacing)) * 1e-3

            occ = owner = None; lesions = np.zeros((0, 6))
            if seg_series:
                import highdicom as hd
                from scipy import ndimage
                _idc().download_from_selection(seriesInstanceUID=[seg_series], downloadDir=f"{d}/seg", quiet=True)
                sg = hd.seg.segread(glob.glob(f"{d}/seg/**/*.dcm", recursive=True)[0])
                v = sg.get_volume(combine_segments=False)
                arr = np.asarray(v.array)
                keep = [i for i, n in enumerate(sg.segment_numbers)
                        if str(sg.get_segment_description(n).segmented_property_type.meaning) == "Mass"]
                if not keep:
                    raise RuntimeError("the SEG has no Mass segment")
                tumor = np.any(np.stack([arr[..., i] > 0 for i in keep]), axis=0)
                lab, n = ndimage.label(tumor, structure=np.ones((3, 3, 3)))
                A = np.asarray(v.affine, float)
                occ, owner = na.scatter_mask(field, lab, A)
                src_ml = abs(np.linalg.det(A[:3, :3])) * 1e-3
                rows = []
                for l in range(1, n + 1):
                    ml = float((lab == l).sum() * src_ml)
                    c = A[:3, :3] @ np.argwhere(lab == l).mean(0) + A[:3, 3]
                    rows.append((l, ml, (6e3 * ml / np.pi) ** (1 / 3), *c))
                lesions = np.asarray(rows).reshape(-1, 6)
                meta["tumor_ml"] = float(tumor.sum() * src_ml)
                meta["tumor_ml_on_model_grid"] = float(occ.sum() * voxel_ml)

            # "center" since 2026-09-22 (American spelling); the region files already on both work
            # volumes say "centre" (spelling: allow centre). Nothing reads the key back today - a reader that will
            # must accept both.
            T = {"organ": [], "size": [], "rule": [], "center": [], "ijk": [], "fill": [], "organ_ml": [],
                 "coords": [], "tumor_ml": [], "vec": []}
            overlaps, base, skipped = [], 0, []
            target = TARGET.get(coll, (None,))[0]
            occ_table = na.integral(occ) if occ is not None else None
            for name, organ in organs.items():
                val = radar.ORGANS.index(name) + 1          # RADAR's numbering, so the analysis reads it as before
                if organ.sum() * voxel_ml < MIN_ORGAN_ML:
                    continue
                if name == target and occ is not None:
                    meta["tumor_in_organ_gate"] = float(occ[organ].sum() / max(occ.sum(), 1e-9))
                for size in SIZES + ((16,) if name in SMALL else ()):
                    sw = na.sweep(field, organ, float(size))
                    if not len(sw.center) or len(sw.center) > MAX_BOXES:
                        if len(sw.center):
                            skipped.append([name, size, len(sw.center)])
                        continue
                    tum = (na.box_sums(occ_table, sw.lo, sw.hi) * voxel_ml if name == target
                           else np.zeros(len(sw.center)))
                    ov = na.lesion_overlaps(sw, occ, owner, voxel_ml) if name == target else np.zeros((0, 3))
                    for r, rule in enumerate(("any", 0.5)[:2 if (name in HALF_RULE and size == 32) else 1]):
                        X = na.pool_sweep(field, head, prepared, sw, organ, name, rule=rule)
                        ok = np.isfinite(X[:, 0])
                        new = np.full(len(ok), -1); new[ok] = base + np.arange(int(ok.sum()))
                        if len(ov):
                            o2 = ov[ok[ov[:, 0].astype(int)]].copy()
                            o2[:, 0] = new[o2[:, 0].astype(int)]
                            overlaps.append(o2)
                        n_ok = int(ok.sum()); base += n_ok
                        T["organ"].append(np.full(n_ok, int(val), np.int8)); T["size"].append(np.full(n_ok, size, np.int16))
                        T["rule"].append(np.full(n_ok, r, np.int8)); T["center"].append(sw.center[ok].astype(np.int16))
                        T["ijk"].append(sw.ijk[ok].astype(np.int16)); T["fill"].append(sw.fill[ok].astype(np.float32))
                        T["organ_ml"].append(sw.organ_ml[ok].astype(np.float32)); T["coords"].append(sw.coords[ok].astype(np.float32))
                        T["tumor_ml"].append(tum[ok].astype(np.float32)); T["vec"].append(X[ok].astype(np.float16))
            if not T["vec"]:
                raise RuntimeError("no organ of haversack's labels was large enough to sweep")
            meta.update(boxes=base, skipped=skipped, seconds=round(time.time() - t0, 1),
                        lesions=int(len(lesions)), grid=list(field.grid.shape))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp.npz")
            np.savez(tmp, meta=np.array(json.dumps(meta)), lesions=lesions,
                     overlaps=np.concatenate(overlaps) if overlaps else np.zeros((0, 3)),
                     **{k: np.concatenate(v) for k, v in T.items()})
            tmp.replace(out_path)
            work.commit()
    except Exception as e:
        import traceback
        return json.dumps({**meta, "err": f"{type(e).__name__}: {e}"[:300], "trace": traceback.format_exc()[-1200:]})
    return json.dumps(meta)


# _read_whole (2026-09-22): np.load on a volume path reads each array in 256 KB pieces through
# zipfile, and every piece is a round trip on the Modal volume's network mount - ~5 MB/s. The null
# model's region files (704-wide vectors, 5.8 GB for 475 scans) took 23 minutes to load that way
# (py-spy: zipfile.read under np.load). One read_bytes per file is one round trip.


def _organ_gates(field, u: str, d: str, twin: str):
    """``({RADAR organ: mask on the model grid}, the task the labels came from)``: haversack's labels
    for series ``u``, fetched by path from the read-only twin and pulled onto ``field``'s grid, each
    organ the union of the structures ``TS_QUERY_RULES`` files under it. Structures RADAR has no
    organ for (prostate, thyroid, ...) gate nothing."""
    if not twin:
        raise RuntimeError("set $HAVERSACK_TWIN: the haversack server whose cached labels gate these fields")
    import urllib.error, urllib.request
    from feldglas.adapters import radar
    from feldglas.labels import read_seg_nrrd
    misses = []
    for task in LABEL_TASKS:
        try:
            with urllib.request.urlopen(f"{twin}/v1/idc/{u}/{task}/labels.seg.nrrd", timeout=180) as r:
                (pathlib.Path(d) / "labels.seg.nrrd").write_bytes(r.read())
            break
        except urllib.error.HTTPError as e:
            misses.append(f"{task}: {e.code}")
    else:
        raise RuntimeError(f"the twin has no haversack labels for this series ({'; '.join(misses)})")
    gl = read_seg_nrrd(pathlib.Path(d) / "labels.seg.nrrd").on_grid(field.grid)
    by_organ: dict[str, list[str]] = {}
    for name in gl.present():
        try:
            by_organ.setdefault(radar.query_for(name), []).append(name)
        except KeyError:
            pass
    return {organ: gl.mask(*names) for organ, names in sorted(by_organ.items())}, task


@functools.lru_cache(maxsize=1)
def _idc():
    """One IDC client per container: constructing it loads its parquet index, which a local
    profile (2026-09-22) showed costing more per scan than the sweep's own pooling."""
    from idc_index import IDCClient
    return IDCClient()


def _head(field=None):
    from feldglas.adapters import radar
    from feldglas.heads import LatticeMeanHead
    return radar.RadarHead.load("/work/head.npz") if POOL == "head" else LatticeMeanHead.for_field(field)


@app.function(image=image, volumes={"/work": work}, cpu=8, memory=49152, timeout=7200)
def analyze(seed: int = 0) -> str:
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.observe import NormalModel
    from feldglas.suite import normal_atlas as na
    rng = np.random.default_rng(seed)
    text = {}
    if POOL == "head":
        head = radar.RadarHead.load("/work/head.npz")
        with np.load("/work/text_en.npz") as z:
            text = {k: z[k].astype(np.float64) for k in z.files}
    scans = []
    for p in sorted(pathlib.Path("/work/regions").glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        with np.load(io.BytesIO(p.read_bytes())) as z:     # whole: see _read_whole
            s = {k: z[k] for k in z.files if k != "meta"}
            s["meta"] = json.loads(str(z["meta"]))
        scans.append(s)
    donors = [s for s in scans if s["meta"]["coll"] == DONORS]
    out = {"encoder": ENCODER, "scans": len(scans), "donors": len(donors), "cohorts": {}, "atlas": {}, "coverage": {}}
    print(f"{len(scans)} scans, {len(donors)} donors", flush=True)

    def take(s, organ_val, size, rule=0):
        m = (s["organ"] == organ_val) & (s["size"] == size) & (s["rule"] == rule)
        return np.flatnonzero(m)

    def finding(X, name):
        lg = X @ text[name].T / head.temperature
        return 1.0 / (1.0 + np.exp(-(lg[:, 1] - lg[:, 0])))

    def donor_set(organ_val, size, rule=0, which=None):
        X, g, c, f, ph = [], [], [], [], []
        for i, s in enumerate(donors if which is None else [donors[j] for j in which]):
            k = take(s, organ_val, size, rule)
            X.append(s["vec"][k].astype(np.float64)); g.append(np.full(len(k), i)); c.append(s["coords"][k])
            f.append(s["fill"][k]); ph += [s["meta"]["phase"]] * len(k)
        return np.concatenate(X), np.concatenate(g), np.concatenate(c), np.concatenate(f), np.asarray(ph)

    # -- the atlas, every organ and size: how big, how shrunk, and does normal depend on place ----
    pathlib.Path("/work/atlas").mkdir(parents=True, exist_ok=True)
    for val, name in enumerate(radar.ORGANS, 1):
        for size in (64, 32, 16):
            X, g, c, f, _ = donor_set(val, size)
            if len(X) < 300 or len(np.unique(g)) < 20:
                continue
            total = NormalModel.fit(X, label=f"{name}|{size}")
            total.save(f"/work/atlas/{name.replace(' ', '_')}_{size}.npz")
            pos = na.PositionedNormal.fit(X, c, f)
            resid = X - na._features(c, f) @ pos.coef
            folds = np.array_split(rng.permutation(np.unique(g)), 5)
            held = np.concatenate([NormalModel.fit(X[~np.isin(g, fo)]).distance(X[np.isin(g, fo)]) for fo in folds])
            out["atlas"][f"{name}|{size}"] = {
                "boxes": int(len(X)), "donors": int(len(np.unique(g))), "shrinkage": round(total.shrinkage, 4),
                "variance_explained_by_place_and_fill": round(float(1 - resid.var(0).sum() / X.var(0).sum()), 3),
                "held_out_donor_distance_q50_q95_q99": [round(float(q), 2) for q in np.quantile(held, [0.5, 0.95, 0.99])]}
    work.commit()
    print(f"atlas: {len(out['atlas'])} organ x size cells", flush=True)

    # -- detection, per cohort --------------------------------------------------------------------
    def evaluate(patients, organ_val, size, fname, rule=0, which=None, detectors=None):
        X, g, c, f, _ = donor_set(organ_val, size, rule, which)
        total = NormalModel.fit(X); within = NormalModel.fit(X, groups=g)
        pos = na.PositionedNormal.fit(X, c); posfill = na.PositionedNormal.fit(X, c, f)
        organ_name = radar.ORGANS[organ_val - 1]
        family = [k for k in text if k.lower().startswith(organ_name + "_")]
        D = {"euclid_from_donor_mean": lambda V, s, k: np.linalg.norm(V - total.mean, axis=1),
             "maha_from_donor_mean": lambda V, s, k: total.distance(V),
             "maha_positioned": lambda V, s, k: pos.distance(V, s["coords"][k]),
             "maha_positioned_fill": lambda V, s, k: posfill.distance(V, s["coords"][k], s["fill"][k]),
             "euclid_own_mean": lambda V, s, k: np.linalg.norm(V - V.mean(0), axis=1),
             "maha_own_median_donor_within": lambda V, s, k: within.distance(V, center=np.median(V, 0)),
             "radar_target_finding": lambda V, s, k: finding(V, fname),
             "radar_any_finding_of_organ": lambda V, s, k: np.max([finding(V, n) for n in family], 0)}
        if not text:                                      # an encoder with no vocabulary
            D = {k: v for k, v in D.items() if not k.startswith("radar_")}
        D = {k: v for k, v in D.items() if detectors is None or k in detectors}
        per = {d: {"A": [], "B": []} for d in D}; pool = {d: {"s": [], "A": [], "B": [], "clean": []} for d in D}
        fscans = {d: [] for d in D}
        for s in patients:
            k = take(s, organ_val, size, rule)
            if len(k) < 6 or not len(s["lesions"]):
                continue
            V = s["vec"][k].astype(np.float64)
            tum, oml = s["tumor_ml"][k].astype(np.float64), s["organ_ml"][k].astype(np.float64)
            clean, A, B = tum <= 0, tum >= 0.25 * np.maximum(oml, 1e-6), tum > 0.05
            ov = s["overlaps"]; pos_of = np.full(len(s["organ"]), -1); pos_of[k] = np.arange(len(k))
            ov = ov[pos_of[ov[:, 0].astype(int)] >= 0].copy(); ov[:, 0] = pos_of[ov[:, 0].astype(int)]
            les = {int(r[0]): {"ml": float(r[1]), "diameter_mm": float(r[2])} for r in s["lesions"]}
            for d, fn in D.items():
                sc = fn(V, s, k)
                for ruleN, P in (("A", A), ("B", B)):
                    if P.sum() >= 3 and clean.sum() >= 3:
                        per[d][ruleN].append(na.auc(sc[P | clean], P[P | clean]))
                pool[d]["s"].append(sc); pool[d]["A"].append(A); pool[d]["B"].append(B); pool[d]["clean"].append(clean)
                fscans[d].append({"score": sc, "ijk": s["ijk"][k], "tumor_ml": tum, "organ_ml": oml, "overlaps": ov, "lesions": les})
        res = {"donor_boxes": int(len(X)), "patients": len(fscans[next(iter(D))])}
        for d in D:
            if not pool[d]["s"]:
                continue
            sc, cl = np.concatenate(pool[d]["s"]), np.concatenate(pool[d]["clean"])
            r = {}
            for ruleN, label in (("A", "fill25"), ("B", "any_overlap")):
                P = np.concatenate(pool[d][ruleN])
                r[f"pooled_auc_{label}"] = round(na.auc(sc[P | cl], P[P | cl]), 3)
                r[f"median_patient_auc_{label}"] = [round(float(np.median(per[d][ruleN])), 3) if per[d][ruleN] else None,
                                                    len(per[d][ruleN])]
            r["froc"] = na.froc(fscans[d])
            res[d] = r
        return res, (total, fscans)

    by_coll = {c: [s for s in scans if s["meta"]["coll"] == c] for c in TARGET}
    for coll, (organ_name, fname) in TARGET.items():
        pts = by_coll[coll]
        if not pts:
            continue
        val = radar.ORGANS.index(organ_name) + 1
        cov = [s["meta"].get("tumor_in_organ_gate") for s in pts if s["meta"].get("tumor_in_organ_gate") is not None]
        onm = [s["meta"]["tumor_ml_on_model_grid"] / max(s["meta"]["tumor_ml"], 1e-9) for s in pts if "tumor_ml" in s["meta"]]
        out["coverage"][coll] = {"scans": len(pts),
                                 "tumor_volume_inside_the_organ_gate_q10_q50_q90": [round(float(q), 3) for q in np.quantile(cov, [0.1, 0.5, 0.9])] if cov else None,
                                 "organ_gate": "haversack ts.v2:total (else total_fast), by TS_QUERY_RULES",
                                 "tumor_volume_kept_by_the_scatter_q10_q50": [round(float(q), 3) for q in np.quantile(onm, [0.1, 0.5])] if onm else None}
        C = out["cohorts"][coll] = {"organ": organ_name, "finding": fname, "sizes": {}}
        for size in (64, 32, 16):
            C["sizes"][str(size)], (total, fsc) = evaluate(pts, val, size, fname)
            # healthy people at the patients' thresholds: false positives per DONOR scan, held out by donor
            thr = {f: C["sizes"][str(size)]["maha_from_donor_mean"]["froc"]["at"][f]["threshold"] for f in ("1.0", "4.0")}
            X, g, *_ = donor_set(val, size)
            folds = np.array_split(rng.permutation(np.unique(g)), 5); fp = {f: 0 for f in thr}
            for fo in folds:
                m = NormalModel.fit(X[~np.isin(g, fo)])
                for i in fo:
                    k = take(donors[i], val, size)
                    sc = m.distance(donors[i]["vec"][k].astype(np.float64))
                    top = na.local_maxima(donors[i]["ijk"][k], sc)
                    for f in thr:
                        if thr[f] is not None:                     # None: fewer false positives exist than were asked for
                            fp[f] += int((sc[top] >= thr[f]).sum())
            C["sizes"][str(size)]["donor_false_positives_per_scan_at_patient_thresholds"] = {
                f: round(fp[f] / len(np.unique(g)), 2) for f in thr}
            print(f"{coll} {size} mm done", flush=True)
        one = ("maha_from_donor_mean", "euclid_from_donor_mean")
        C["gate_rule_half_32mm"], _ = evaluate(pts, val, 32, fname, rule=1, detectors=one)
        phases = np.array([s["meta"]["phase"] for s in donors])
        C["phase"] = {}
        for ph in sorted({s["meta"]["phase"] for s in pts}):
            sub = [s for s in pts if s["meta"]["phase"] == ph]
            row = {"patients": len(sub)}
            for label, which in (("all_donors", None), ("matched", np.flatnonzero(phases == ph)), ("mismatched", np.flatnonzero(phases != ph))):
                if which is not None and len(which) < 10:
                    continue
                r, _ = evaluate(sub, val, 32, fname, which=which, detectors=("maha_from_donor_mean",))
                if "maha_from_donor_mean" in r:
                    row[label] = {"donors": len(donors) if which is None else int(len(which)),
                                  "pooled_auc_fill25": r["maha_from_donor_mean"]["pooled_auc_fill25"],
                                  "sens_at_1fp": r["maha_from_donor_mean"]["froc"]["at"]["1.0"]["all"]}
            C["phase"][ph] = row
        C["donor_count_32mm"] = {}
        for n in (3, 6, 12, 25, 50, len(donors)):
            vals = []
            for _ in range(1 if n == len(donors) else 5):
                r, _ = evaluate(pts, val, 32, fname, which=rng.choice(len(donors), n, replace=False), detectors=one)
                vals.append([r[d]["pooled_auc_fill25"] for d in one] + [r[d]["froc"]["at"]["1.0"]["all"] for d in one])
            C["donor_count_32mm"][str(n)] = dict(zip(["maha_auc", "euclid_auc", "maha_sens_1fp", "euclid_sens_1fp"],
                                                     [round(float(v), 3) for v in np.mean(vals, 0)]))
        print(f"{coll} done", flush=True)
    pathlib.Path("/work/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"/work/results/normal_atlas{SUFFIX}.json").write_text(json.dumps(out, indent=1))
    work.commit()
    return json.dumps(out)


@app.function(image=image, volumes={"/work": work}, cpu=8, memory=49152, timeout=3600)
def tails(size: int = 32) -> str:
    """Why a detector with the best AUC can have the worst FROC (2026-09-20: liver metastases at
    32 mm, Mahalanobis from the donor mean 0.966 pooled and 17 % of lesions at 1 FP/scan; Euclidean
    0.920 and 84 %). AUC is about the bulk of the clean boxes and FROC about their extreme tail, so
    this asks what the tail IS: whose boxes, where in the organ, along which of the donors'
    directions - and whether truncating the whitening, or reading each box against its own scan,
    removes it."""
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.suite import normal_atlas as na
    scans = []
    for p in sorted(pathlib.Path("/work/regions").glob("*.npz")):
        if not p.name.endswith(".tmp.npz"):
            with np.load(io.BytesIO(p.read_bytes())) as z:     # whole: see _read_whole
                s = {k: z[k] for k in z.files if k != "meta"}; s["meta"] = json.loads(str(z["meta"]))
            scans.append(s)
    out = {}
    for coll, (organ_name, _) in TARGET.items():
        val = radar.ORGANS.index(organ_name) + 1
        sel = lambda s: np.flatnonzero((s["organ"] == val) & (s["size"] == size) & (s["rule"] == 0))
        X = np.concatenate([s["vec"][sel(s)].astype(np.float64) for s in scans if s["meta"]["coll"] == DONORS])
        mu = X.mean(0); w, U = np.linalg.eigh(np.cov((X - mu).T)); w, U = w[::-1], U[:, ::-1]
        pts = [s for s in scans if s["meta"]["coll"] == coll and len(s["lesions"]) and len(sel(s)) >= 6]
        P = [(s, sel(s), (s["vec"][sel(s)].astype(np.float64) - mu) @ U) for s in pts]          # donors' principal axes

        cur = {"i": 0}                                     # which scan `run` is scoring, for fold-aware detectors

        def run(score):
            fs, pooled, lab = [], [], []
            for i, (s, k, Z) in enumerate(P):
                cur["i"] = i
                sc = score(Z); tum = s["tumor_ml"][k].astype(np.float64); oml = s["organ_ml"][k].astype(np.float64)
                pos_of = np.full(len(s["organ"]), -1); pos_of[k] = np.arange(len(k))
                ov = s["overlaps"]; ov = ov[pos_of[ov[:, 0].astype(int)] >= 0].copy(); ov[:, 0] = pos_of[ov[:, 0].astype(int)]
                fs.append({"score": sc, "ijk": s["ijk"][k], "tumor_ml": tum, "organ_ml": oml, "overlaps": ov,
                           "lesions": {int(r[0]): {"ml": float(r[1]), "diameter_mm": float(r[2])} for r in s["lesions"]}})
                A, cl = tum >= 0.25 * np.maximum(oml, 1e-6), tum <= 0
                pooled.append(sc[A | cl]); lab.append(A[A | cl])
            f = na.froc(fs)
            return {"pooled_auc": round(na.auc(np.concatenate(pooled), np.concatenate(lab)), 3),
                    "sens_at_0.5_1_2_4": [f["at"][k]["all"] for k in ("0.5", "1.0", "2.0", "4.0")]}, fs

        scanwise = lambda d: (d - np.median(d)) / (1.4826 * np.median(np.abs(d - np.median(d))) + 1e-12)
        R = {"eigen_share_top_4_16_64": [round(float(w[:k].sum() / w.sum()), 3) for k in (4, 16, 64)], "detectors": {}}
        for k in (2, 4, 8, 16, 32, 64, 128, 256):
            R["detectors"][f"maha_top{k}"], _ = run(lambda Z, k=k: np.sqrt((Z[:, :k] ** 2 / w[:k]).sum(1)))
            R["detectors"][f"maha_top{k}_scanwise"], _ = run(lambda Z, k=k: scanwise(np.sqrt((Z[:, :k] ** 2 / w[:k]).sum(1))))
        for a, b in ((16, 64), (64, 256)):
            R["detectors"][f"maha_axes{a}-{b}"], _ = run(lambda Z, a=a, b=b: np.sqrt((Z[:, a:b] ** 2 / w[a:b]).sum(1)))
        # SITE-ADAPTED whitening, label-free and held out by patient: the donors' mean and axes, but
        # each axis no quieter than the target site's own boxes make it (robust spread of ALL boxes
        # of the OTHER half of the patients - tumors are a minority and no mask is read)
        fold = np.arange(len(P)) % 2
        site = {}
        for h in (0, 1):
            Zo = np.concatenate([Z for j, (_, _, Z) in enumerate(P) if fold[j] != h])
            med = np.median(Zo, 0)
            site[h] = (med, np.maximum(w, (1.4826 * np.median(np.abs(Zo), 0)) ** 2),
                       np.maximum(w, (1.4826 * np.median(np.abs(Zo - med), 0)) ** 2))
        R["site_to_donor_spread_ratio_axes_0-16_16-64_64-256"] = [
            round(float(np.sqrt(site[0][1][a:b] / w[a:b]).mean()), 2) for a, b in ((0, 16), (16, 64), (64, 256))]
        R["detectors"]["maha_site_scaled_donor_center"], _ = run(lambda Z: np.sqrt((Z ** 2 / site[fold[cur["i"]]][1]).sum(1)))
        R["detectors"]["maha_site_scaled_site_center"], _ = run(
            lambda Z: np.sqrt(((Z - site[fold[cur["i"]]][0]) ** 2 / site[fold[cur["i"]]][2]).sum(1)))
        R["detectors"]["euclid"], _ = run(lambda Z: np.linalg.norm(Z, axis=1))
        R["detectors"]["euclid_scanwise"], _ = run(lambda Z: scanwise(np.linalg.norm(Z, axis=1)))
        R["detectors"]["maha_full_centered_on_scan_median"], _ = run(lambda Z: np.sqrt((((Z - np.median(Z, 0)) ** 2) / w).sum(1)))
        # the tail itself, for the full Mahalanobis: the top (1 per scan) clean local maxima
        _, fs = run(lambda Z: np.sqrt((Z ** 2 / w).sum(1)))
        rows = []
        for i, (f, (s, k, Z)) in enumerate(zip(fs, P)):
            top = na.local_maxima(f["ijk"], f["score"]) & (f["tumor_ml"] <= 0)
            for j in np.flatnonzero(top):
                rows.append((f["score"][j], i, s["fill"][k][j], *s["coords"][k][j], *(Z[j] ** 2 / w)))
        rows = np.asarray(sorted(rows, key=lambda r: -r[0])[:len(P)])
        per_scan = np.bincount(rows[:, 1].astype(int), minlength=len(P))
        share = rows[:, 6:] / rows[:, 6:].sum(1, keepdims=True)
        allfill = np.concatenate([s["fill"][k] for s, k, _ in P])
        scan_med = np.array([np.median(f["score"]) for f in fs])
        R["tail"] = {"false_positives_examined": int(len(rows)), "scans_they_come_from": int((per_scan > 0).sum()),
                     "share_from_the_5_worst_scans": round(float(np.sort(per_scan)[::-1][:5].sum() / len(rows)), 3),
                     "their_fill_q25_q50": [round(float(q), 2) for q in np.quantile(rows[:, 2], [0.25, 0.5])],
                     "all_boxes_fill_q25_q50": [round(float(q), 2) for q in np.quantile(allfill, [0.25, 0.5])],
                     "share_of_their_distance_on_axes_0-16_16-64_64-256": [round(float(share[:, a:b].sum(1).mean()), 3) for a, b in ((0, 16), (16, 64), (64, 256))],
                     "scan_median_distance_q10_q50_q90_max": [round(float(q), 1) for q in np.quantile(scan_med, [0.1, 0.5, 0.9, 1.0])],
                     "phases_of_the_5_worst": [P[i][0]["meta"]["phase"] for i in np.argsort(per_scan)[::-1][:5]],
                     "worst_scans": [P[i][0]["meta"]["u"] for i in np.argsort(per_scan)[::-1][:5]]}
        out[coll] = R
        print(coll, "done", flush=True)
    pathlib.Path("/work/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"/work/results/normal_atlas{SUFFIX}_tails_{size}.json").write_text(json.dumps(out, indent=1)); work.commit()
    return json.dumps(out)


def _jobs(collection: str, limit: int):
    from feldglas.remote import Manifest
    mf = Manifest.load(HERE.parent / "manifests" / f"{FIELDS}.json")
    V, I = MEDSEG / "results" / "validation", MEDSEG / "results" / "idc"
    se_of = {r["crdc_series_uuid"]: r["se"] for r in csv.DictReader(open(I / "radar_validation_series_all.csv"))}
    seg = {(r["pid"], r["ref_series"]): r["seg_series"] for r in csv.DictReader(open(I / "expert_segs.csv"))
           if not r["ar"] and r["coll"] in TARGET}
    wanted = {c.strip() for c in collection.split(",") if c.strip()} or {DONORS, *TARGET}
    jobs = []
    for r in csv.DictReader(open(V / "choice.csv")):
        u = r["chosen"]
        if not u or r["coll"] not in wanted or f"{u}.npz" not in mf.files:
            continue
        s = seg.get((r["pid"], se_of.get(u)), "")
        if r["coll"] in TARGET and not s:
            continue                                   # a patient without an expert mask on THIS series tells nothing here
        jobs.append((u, mf.files[f"{u}.npz"]["digest"], r["pid"], r["coll"], r["phase"], s))
    by = {}
    for j in jobs:
        by.setdefault(j[3], []).append(j)
    if limit:
        per = max(1, limit // len(by))
        jobs = [j for v in by.values() for j in v[:per]]
    return jobs


@app.local_entrypoint()
def main(collection: str = "", limit: int = 0, force: bool = False):
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.paths import encoder_dir
    if not _STORE.startswith(("s3://", "gs://", "az://")):
        raise SystemExit("set FELDGLAS_STORE (e.g. s3://<bucket>/feldglas): the fields are read from it")
    if POOL == "head":
        cache = encoder_dir(radar.NAME)
        names = radar.english_names(MEDSEG.parents[1] / "upstream" / "damo-radar" / "RADAR_inference" / "inference_demo.py")
        table = radar.load_text_table(cache / "text_table.npz")
        buf = io.BytesIO(); np.savez(buf, **{names[k]: v for k, v in table.items()})
        with work.batch_upload(force=True) as up:        # 11 MB, once per run: the container's head and vocabulary
            up.put_file(str(cache / "head.npz"), "/head.npz")
            up.put_file(io.BytesIO(buf.getvalue()), "/text_en.npz")
    jobs = _jobs(collection, limit)
    print(f"{len(jobs)} scans: " + ", ".join(f"{c} {sum(j[3] == c for j in jobs)}" for c in sorted({j[3] for j in jobs})))
    t = time.time(); n = {"ok": 0, "cached": 0, "err": 0}
    for j, r in zip(jobs, regions.starmap([(*j, force, TWIN) for j in jobs], return_exceptions=True)):
        r = json.loads(r) if isinstance(r, str) else {"err": repr(r)[:300]}
        if r.get("err"):
            n["err"] += 1; print(f"  {j[0]} {j[3]}: {r['err']}\n{r.get('trace', '')}")
        elif r.get("cached"):
            n["cached"] += 1
        else:
            n["ok"] += 1
            print(f"  {j[0]} {j[3]:28} {r['boxes']:6} boxes {r['lesions']:3} lesions {r['seconds']:6.1f} s"
                  + (f"  tumor in the {TARGET[j[3]][0]} gate {r.get('tumor_in_organ_gate', float('nan')):.2f}" if j[3] in TARGET else "")
                  + (f"  [{r.get('labels_task')}]" if r.get('labels_task') != LABEL_TASKS[0] else "")
                  + (f"  skipped {r['skipped']}" if r.get("skipped") else ""))
    print(f"{n} in {time.time() - t:.0f} s")


@app.local_entrypoint()
def analysis(out: str = ""):
    r = json.loads(analyze.remote())
    p = pathlib.Path(out) if out else MEDSEG / "results" / "validation" / f"normal_atlas{SUFFIX}.json"
    p.write_text(json.dumps(r, indent=1))
    print(f"-> {p}")
    for coll, C in r["cohorts"].items():
        print(f"\n{coll} ({C['organ']}; {r['coverage'][coll]})")
        for size, S in C["sizes"].items():
            print(f"  {size} mm, {S['patients']} patients, {S['donor_boxes']} donor boxes"
                  f"   donor FP/scan at patients' 1 and 4 FP thresholds: {S['donor_false_positives_per_scan_at_patient_thresholds']}")
            print(f"    {'detector':34} {'pooled':>7} {'patient':>8} {'any-ov':>7}   sens @0.5 / 1 / 2 / 4 FP per scan   never hit")
            for d, v in S.items():
                if not isinstance(v, dict) or "froc" not in v:
                    continue
                at = v["froc"]["at"]
                print(f"    {d:34} {v['pooled_auc_fill25']:7.3f} {v['median_patient_auc_fill25'][0] or float('nan'):8.3f} "
                      f"{v['pooled_auc_any_overlap']:7.3f}   " + " / ".join(f"{at[f]['all']:.2f}" for f in ("0.5", "1.0", "2.0", "4.0"))
                      + f"   {v['froc']['never_hit']}/{v['froc']['lesions']}")


@app.local_entrypoint()
def tail_study(size: int = 32, out: str = ""):
    r = json.loads(tails.remote(size))
    p = pathlib.Path(out) if out else MEDSEG / "results" / "validation" / f"normal_atlas{SUFFIX}_tails_{size}.json"
    p.write_text(json.dumps(r, indent=1)); print(f"-> {p}")
    for coll, R in r.items():
        print(f"\n{coll}   donor variance in the top 4 / 16 / 64 axes: {R['eigen_share_top_4_16_64']}   site/donor spread by axis band: {R.get('site_to_donor_spread_ratio_axes_0-16_16-64_64-256')}")
        for d, v in R["detectors"].items():
            print(f"  {d:36} {v['pooled_auc']:.3f}   " + " / ".join(f"{x:.2f}" for x in v["sens_at_0.5_1_2_4"]))
        print("  tail:", json.dumps(R["tail"]))
