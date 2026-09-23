"""Does the SOURCE of the kidney gate matter? RADAR's own mask against haversack's, on KiTS.

5.6 found RADAR's own kidney mask holding a median 0.79 of the expert tumor volume and under half
in 33 of 148 KiTS scans - the one organ where a box sweep gated by the encoder's own mask could be
missing the very tissue it is asked about. This runs the SAME atlas and the SAME per-lesion FROC
under three gates, everything else held fixed (field, head, kidney query, 32 mm boxes, donors):

  own     RADAR's own mask (what 5.6 used)
  hv      haversack `kidney_left` + `kidney_right` (`ts.v2:total`, else `ts.v2:total_fast`)
  hv+     the same plus `kidney_cyst_*` - TotalSegmentator splits cysts out of the kidney

Labels come BY PATH from the read-only twin of the redeployed study server
(``$HAVERSACK_TWIN``: cache hits only, no credentials, ~1 s a
series), are read with `feldglas.labels` and pulled onto the model grid. The healthy donors are
swept under each gate too, so each gate is judged against an atlas built the same way.

Run:  FELDGLAS_STORE=s3://<bucket>/feldglas uv run --no-sync modal run tools/radar_kidney_gates_modal.py
"""
import csv, json, os, pathlib, time

import modal

HERE = pathlib.Path(__file__).resolve().parent
MEDSEG = HERE.parents[1] / "medseg" / "docs" / "radar-idc-validation" if len(HERE.parents) > 1 else HERE
_SECRET = os.environ.get("FELDGLAS_MODAL_SECRET", "feldglas-r2")
TWIN = os.environ.get("HAVERSACK_TWIN", "")          # the haversack server holding the labels (its read-only twin will do)
# the SAME image definition as radar_atlas_modal.py, so Modal reuses the one it built
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("numpy>=1.24", "scipy", "obstore>=0.11", "idc-index", "highdicom>=0.23", "pydicom>=3",
                      "pylibjpeg", "pylibjpeg-libjpeg",
                      "rankfield @ git+https://github.com/mhalle/rankfield.git@v0.3.2",
                      "provender @ git+https://github.com/mhalle/provender.git@v0.1.1")
         .env({"FELDGLAS_STORE": os.environ.get("FELDGLAS_STORE", "")})
         .add_local_python_source("feldglas"))
work = modal.Volume.from_name("feldglas-radar-work", create_if_missing=True)
app = modal.App("feldglas-radar-kidney-gates")
GATES = ("own", "hv", "hv+")
FINDING = "Kidney_Renal cell carcinoma"


@app.function(image=image, secrets=[modal.Secret.from_name(_SECRET)], volumes={"/work": work}, cpu=2, memory=8192,
              timeout=2400, max_containers=40)
def sweep_one(u: str, digest: str, pid: str, coll: str, phase: str, seg_series: str, twin: str, force: bool = False) -> str:
    if not twin:
        raise RuntimeError("set $HAVERSACK_TWIN: the haversack server whose cached labels gate these fields")
    import glob, tempfile, urllib.error, urllib.request
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.labels import read_seg_nrrd
    from feldglas.remote import open_blobs
    from feldglas.store import read_field
    from feldglas.suite import normal_atlas as na
    out_path = pathlib.Path(f"/work/kidney_gates/{u}.npz")
    if out_path.exists() and not force:
        return json.dumps({"u": u, "cached": True})
    meta = {"u": u, "pid": pid, "coll": coll, "phase": phase}
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / f"{u}.npz"
            if not open_blobs(radar.NAME, check=False).fetch(digest, p):
                raise FileNotFoundError(f"blob {digest} is gone, or did not match its name")
            field = read_field(p)
            lm = None
            for task in ("ts.v2:total", "ts.v2:total_fast"):
                try:
                    with urllib.request.urlopen(f"{twin}/v1/idc/{u}/{task}/labels.seg.nrrd", timeout=180) as r:
                        (pathlib.Path(d) / "labels.seg.nrrd").write_bytes(r.read())
                    lm = read_seg_nrrd(pathlib.Path(d) / "labels.seg.nrrd"); meta["labels_task"] = task
                    break
                except urllib.error.HTTPError as e:
                    meta.setdefault("label_misses", []).append(f"{task}: {e.code}")
            if lm is None:
                raise RuntimeError(f"the twin has no labels for this series ({meta.get('label_misses')})")
            gl = lm.on_grid(field.grid)
            kid = [n for n in gl.names if n in ("kidney_left", "kidney_right")]
            cyst = [n for n in gl.names if n.startswith("kidney_cyst")]
            masks = {"own": field.native_mask == radar.ORGANS.index("kidney") + 1,
                     "hv": gl.mask(*kid) if kid else np.zeros(field.grid.shape, bool)}
            masks["hv+"] = masks["hv"] | (gl.mask(*cyst) if cyst else False)
            voxel_ml = float(np.prod(field.grid.spacing)) * 1e-3
            meta["kidney_ml"] = {g: round(float(m.sum()) * voxel_ml, 1) for g, m in masks.items()}
            meta["cyst_ml"] = round(float((masks["hv+"] & ~masks["hv"]).sum()) * voxel_ml, 1)
            inter = (masks["own"] & masks["hv"]).sum()
            meta["dice_own_hv"] = round(float(2 * inter / max(masks["own"].sum() + masks["hv"].sum(), 1)), 3)

            occ = owner = None; lesions = np.zeros((0, 3))
            if seg_series:                                        # the same lesion ids radar_atlas_modal.py wrote
                import highdicom as hd
                from idc_index import IDCClient
                from scipy import ndimage
                IDCClient().download_from_selection(seriesInstanceUID=[seg_series], downloadDir=f"{d}/seg", quiet=True)
                sg = hd.seg.segread(glob.glob(f"{d}/seg/**/*.dcm", recursive=True)[0])
                v = sg.get_volume(combine_segments=False); arr = np.asarray(v.array)
                keep = [i for i, n in enumerate(sg.segment_numbers)
                        if str(sg.get_segment_description(n).segmented_property_type.meaning) == "Mass"]
                tumor = np.any(np.stack([arr[..., i] > 0 for i in keep]), axis=0)
                lab, n = ndimage.label(tumor, structure=np.ones((3, 3, 3)))
                A = np.asarray(v.affine, float); src_ml = abs(np.linalg.det(A[:3, :3])) * 1e-3
                occ, owner = na.scatter_mask(field, lab, A)
                lesions = np.array([(l, (lab == l).sum() * src_ml, (6e3 * (lab == l).sum() * src_ml / np.pi) ** (1 / 3)) for l in range(1, n + 1)]).reshape(-1, 3)
                tot = max(float(occ.sum()), 1e-9)
                meta["tumor_inside_gate"] = {g: round(float(occ[m].sum()) / tot, 3) for g, m in masks.items()}
            head = radar.RadarHead.load("/work/head.npz")
            prepared = head.prepare(field.all_tokens())
            arrays = {"lesions": lesions}
            for g, m in masks.items():
                if m.sum() * voxel_ml < 5.0:
                    continue
                sw = na.sweep(field, m, 32.0)
                X = na.pool_sweep(field, head, prepared, sw, m, "kidney")
                ok = np.isfinite(X[:, 0]); new = np.full(len(ok), -1); new[ok] = np.arange(int(ok.sum()))
                tum = (na.box_sums(na.integral(occ), sw.lo, sw.hi) * voxel_ml) if occ is not None else np.zeros(len(ok))
                ov = na.lesion_overlaps(sw, occ, owner, voxel_ml) if occ is not None else np.zeros((0, 3))
                if len(ov):
                    ov = ov[ok[ov[:, 0].astype(int)]].copy(); ov[:, 0] = new[ov[:, 0].astype(int)]
                arrays.update({f"{g}|vec": X[ok].astype(np.float16), f"{g}|ijk": sw.ijk[ok].astype(np.int16),
                               f"{g}|tumor_ml": tum[ok].astype(np.float32), f"{g}|organ_ml": sw.organ_ml[ok].astype(np.float32),
                               f"{g}|overlaps": ov})
            meta["boxes"] = {g: int(len(arrays[f"{g}|vec"])) for g in masks if f"{g}|vec" in arrays}
            meta["seconds"] = round(time.time() - t0, 1)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp.npz")
            np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays); tmp.replace(out_path); work.commit()
    except Exception as e:
        import traceback
        return json.dumps({**meta, "err": f"{type(e).__name__}: {e}"[:300], "trace": traceback.format_exc()[-900:]})
    return json.dumps(meta)


@app.function(image=image, volumes={"/work": work}, cpu=8, memory=32768, timeout=3600)
def compare() -> str:
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.observe import NormalModel
    from feldglas.suite import normal_atlas as na
    head = radar.RadarHead.load("/work/head.npz")
    with np.load("/work/text_en.npz") as z:
        pair = z[FINDING].astype(np.float64)
    S = []
    for p in sorted(pathlib.Path("/work/kidney_gates").glob("*.npz")):
        if not p.name.endswith(".tmp.npz"):
            with np.load(p) as z:
                S.append({**{k: z[k] for k in z.files if k != "meta"}, "meta": json.loads(str(z["meta"]))})
    donors = [s for s in S if s["meta"]["coll"] == "pancreas_ct"]
    pts = [s for s in S if s["meta"]["coll"] == "c4kc_kits" and len(s["lesions"])]
    q = lambda xs: [round(float(v), 3) for v in np.quantile(xs, [0.1, 0.5, 0.9])] if len(xs) else None
    out = {"donors": len(donors), "patients": len(pts), "labels_task": {}, "gates": {},
           "dice_own_vs_hv_q10_q50_q90": {"donors": q([s["meta"]["dice_own_hv"] for s in donors]), "patients": q([s["meta"]["dice_own_hv"] for s in pts])},
           "cyst_ml_patients_q50_q90_max": [round(float(v), 1) for v in np.quantile([s["meta"]["cyst_ml"] for s in pts], [0.5, 0.9, 1.0])]}
    for s in S:
        t = s["meta"].get("labels_task", "?"); out["labels_task"][t] = out["labels_task"].get(t, 0) + 1
    for g in GATES:
        D = [s for s in donors if f"{g}|vec" in s]
        X = np.concatenate([s[f"{g}|vec"].astype(np.float64) for s in D])
        nm = NormalModel.fit(X); mu = X.mean(0)
        w, U = np.linalg.eigh(np.cov((X - mu).T)); w, U = np.maximum(w[::-1], 1e-12), U[:, ::-1]
        det = {"distance from the donors' mean": lambda V: np.linalg.norm(V - mu, axis=1),
               "the same, whitened": lambda V: nm.distance(V),
               "whitened, top 16 axes": lambda V: np.sqrt(((((V - mu) @ U[:, :16]) ** 2) / w[:16]).sum(1)),
               "RADAR's finding score": lambda V: 1 / (1 + np.exp(-((V @ pair.T)[:, 1] - (V @ pair.T)[:, 0]) / head.temperature))}
        P = [s for s in pts if f"{g}|vec" in s and len(s[f"{g}|vec"]) >= 6]
        cov = [s["meta"]["tumor_inside_gate"][g] for s in pts if "tumor_inside_gate" in s["meta"]]
        R = {"donor_boxes": int(len(X)), "patients_swept": len(P), "patient_boxes_median": int(np.median([len(s[f"{g}|vec"]) for s in P])),
             "tumor_volume_inside_gate_q10_q50_q90": q(cov), "scans_under_half": int(np.sum(np.asarray(cov) < 0.5)), "detectors": {}}
        for name, fn in det.items():
            fs, sc_, lab = [], [], []
            for s in P:
                V = s[f"{g}|vec"].astype(np.float64); sc = fn(V)
                tum, oml = s[f"{g}|tumor_ml"].astype(np.float64), s[f"{g}|organ_ml"].astype(np.float64)
                fs.append({"score": sc, "ijk": s[f"{g}|ijk"], "tumor_ml": tum, "organ_ml": oml, "overlaps": s[f"{g}|overlaps"],
                           "lesions": {int(r[0]): {"ml": float(r[1]), "diameter_mm": float(r[2])} for r in s["lesions"]}})
                A, cl = tum >= 0.25 * np.maximum(oml, 1e-6), tum <= 0
                sc_.append(sc[A | cl]); lab.append(A[A | cl])
            f = na.froc(fs)
            R["detectors"][name] = {"pooled_auc": round(na.auc(np.concatenate(sc_), np.concatenate(lab)), 3),
                                    "lesions_at_0.5_1_2_4_fp": [f["at"][k]["all"] for k in ("0.5", "1.0", "2.0", "4.0")],
                                    "never_hit": f["never_hit"], "lesions": f["lesions"],
                                    "at_1_fp_by_size": {k: v for k, v in f["at"]["1.0"].items() if k.endswith("mm")}}
        out["gates"][g] = R
        print(g, "done", flush=True)
    pathlib.Path("/work/results/kidney_gates.json").write_text(json.dumps(out, indent=1)); work.commit()
    return json.dumps(out)


@app.local_entrypoint()
def main(limit: int = 0, force: bool = False):
    from feldglas.remote import Manifest
    if not os.environ.get("FELDGLAS_STORE", "").startswith(("s3://", "gs://", "az://")):
        raise SystemExit("set FELDGLAS_STORE (e.g. s3://<bucket>/feldglas): the fields are read from it")
    mf = Manifest.load(HERE.parent / "manifests" / "radar.json")
    V, I = MEDSEG / "results" / "validation", MEDSEG / "results" / "idc"
    se_of = {r["crdc_series_uuid"]: r["se"] for r in csv.DictReader(open(I / "radar_validation_series_all.csv"))}
    seg = {(r["pid"], r["ref_series"]): r["seg_series"] for r in csv.DictReader(open(I / "expert_segs.csv")) if not r["ar"] and r["coll"] == "c4kc_kits"}
    jobs = []
    for r in csv.DictReader(open(V / "choice.csv")):
        u = r["chosen"]
        if not u or f"{u}.npz" not in mf.files or r["coll"] not in ("pancreas_ct", "c4kc_kits"):
            continue
        s = seg.get((r["pid"], se_of.get(u)), "")
        if r["coll"] == "c4kc_kits" and not s:
            continue
        jobs.append((u, mf.files[f"{u}.npz"]["digest"], r["pid"], r["coll"], r["phase"], s, TWIN, force))
    if limit:
        jobs = [j for j in jobs if j[3] == "pancreas_ct"][:limit // 2] + [j for j in jobs if j[3] == "c4kc_kits"][:limit - limit // 2]
    print(f"{len(jobs)} scans: donors {sum(j[3] == 'pancreas_ct' for j in jobs)}, KiTS {sum(j[3] == 'c4kc_kits' for j in jobs)}; labels from {TWIN}")
    t = time.time(); n = {"ok": 0, "cached": 0, "err": 0}
    for j, r in zip(jobs, sweep_one.starmap(jobs, return_exceptions=True)):
        r = json.loads(r) if isinstance(r, str) else {"err": repr(r)[:300]}
        n["err" if r.get("err") else "cached" if r.get("cached") else "ok"] += 1
        if r.get("err"):
            print(f"  {j[0]} {j[3]}: {r['err']}")
    print(f"{n} in {time.time() - t:.0f} s")
    if limit:
        return
    out = json.loads(compare.remote())
    p = V / "kidney_gates.json"; p.write_text(json.dumps(out, indent=1)); print(f"-> {p}")
    print(f"labels: {out['labels_task']}; Dice own vs hv (q10/q50/q90): {out['dice_own_vs_hv_q10_q50_q90']}; cyst ml in patients q50/q90/max: {out['cyst_ml_patients_q50_q90_max']}")
    for g, R in out["gates"].items():
        print(f"\n{g}: tumor volume inside the gate {R['tumor_volume_inside_gate_q10_q50_q90']}, {R['scans_under_half']} scans under half; "
              f"{R['patients_swept']} patients, {R['patient_boxes_median']} boxes each, {R['donor_boxes']} donor boxes")
        for d, v in R["detectors"].items():
            print(f"   {d:32} AUC {v['pooled_auc']:.3f}  lesions at 0.5/1/2/4 FP {v['lesions_at_0.5_1_2_4_fp']}  never hit {v['never_hit']}/{v['lesions']}")
