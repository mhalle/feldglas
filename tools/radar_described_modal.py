"""Does a DESCRIBED lesion land where real lesions of that description do?

The user's query "look for lesions with this scale and falloff" needs a road from an IMAGE-domain
description to a direction in the space, and the only such road is intervention: paint it, encode,
take the displacement (EXPLORATION 3.3). medseg's `radar_response_modal.py::falloff` painted a bank
into twelve healthy donors' livers - radius 3-13 mm x contrast -60..+30 HU (below the response's
saturation) x edge width 0.5-8 mm - and kept the pooled 32 mm box vector at each site, with and
without the lesion. This asks whether that bank means anything for REAL lesions:

  ::describe   per expert-outlined liver lesion, from the CT and the SEG (CPU): diameter, contrast
               (median HU of the core minus median of a 3-10 mm shell of parenchyma) and edge width
               (an erf fitted to HU against signed distance from the outline) - the same three numbers
               the bank is indexed by. Lesion ids are the ones `radar_atlas_modal.py` wrote.
  ::compare    the bank's own geometry (can RADAR tell a soft edge from a sharp one at all?); each real
               lesion's displacement (its best box minus the patient's clean boxes) against the bank -
               direction, and whether the best-matching painted lesion ESTIMATES the real one's size,
               contrast and edge; and detection with a painted template as the whole query: no word,
               no example - beside the text direction and the real-example template of 5.9

Run:  FELDGLAS_STORE=s3://<bucket>/feldglas uv run --no-sync modal run tools/radar_described_modal.py::describe
      uv run --no-sync modal run tools/radar_described_modal.py::compare
"""
import csv, json, os, pathlib, time

import modal

HERE = pathlib.Path(__file__).resolve().parent


def _up(n):
    return HERE.parents[n] if len(HERE.parents) > n else HERE


MEDSEG = _up(1) / "medseg" / "docs" / "radar-idc-validation"
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("numpy>=1.24", "scipy", "nibabel", "obstore>=0.11", "idc-index", "highdicom>=0.23", "pydicom>=3",
                      "pylibjpeg", "pylibjpeg-libjpeg",
                      "rankfield @ git+https://github.com/mhalle/rankfield.git@v0.3.2",
                      "provender @ git+https://github.com/mhalle/provender.git@v0.1.1")
         .add_local_python_source("feldglas"))
work = modal.Volume.from_name("feldglas-radar-work", create_if_missing=True)
cts = modal.Volume.from_name("radar-idc-validation").read_only()              # the study's CTs: /ct/<series>.nii.gz
app = modal.App("feldglas-radar-described")

LIVER_COHORTS = {"colorectal_liver_metastases": "Liver_Metastasis", "hcc_tace_seg": "Liver_Hepatocellular carcinoma"}
WIDTHS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)


@app.function(image=image, volumes={"/work": work, "/vol": cts}, cpu=2, memory=16384, timeout=1800, max_containers=40)
def describe_one(u: str, pid: str, coll: str, seg_series: str, force: bool = False) -> str:
    import glob, tempfile
    import highdicom as hd, nibabel as nib, numpy as np
    from idc_index import IDCClient
    from scipy import ndimage
    from scipy.special import erf
    out_path = pathlib.Path(f"/work/described/{u}.json")
    if out_path.exists() and not force:
        return json.dumps({"u": u, "cached": True})
    try:
        ct = nib.load(f"/vol/ct/{u}.nii.gz")
        hu = np.asarray(ct.dataobj, np.float32); sp = np.sqrt((np.asarray(ct.affine)[:3, :3] ** 2).sum(0))
        with tempfile.TemporaryDirectory() as d:
            IDCClient().download_from_selection(seriesInstanceUID=[seg_series], downloadDir=d, quiet=True)
            sg = hd.seg.segread(glob.glob(f"{d}/**/*.dcm", recursive=True)[0])
            v = sg.get_volume(combine_segments=False); arr = np.asarray(v.array)
            keep = [i for i, n in enumerate(sg.segment_numbers)
                    if str(sg.get_segment_description(n).segmented_property_type.meaning) == "Mass"]
            tumor = np.any(np.stack([arr[..., i] > 0 for i in keep]), axis=0)
        lab, n = ndimage.label(tumor, structure=np.ones((3, 3, 3)))              # the SAME ids radar_atlas_modal.py wrote
        M = np.linalg.inv(np.asarray(v.affine, float)) @ np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(ct.affine, float)   # CT index -> SEG index
        lab_ct = ndimage.affine_transform(lab.astype(np.int16), M[:3, :3], offset=M[:3, 3], output_shape=hu.shape, order=0)
        pad = np.ceil(12.0 / sp).astype(int)
        rows = []
        for l, sl in enumerate(ndimage.find_objects(lab_ct, n), 1):
            if sl is None:
                continue
            sl = tuple(slice(max(s.start - p, 0), min(s.stop + p, dim)) for s, p, dim in zip(sl, pad, hu.shape))
            m = lab_ct[sl] == l; other = (lab_ct[sl] > 0) & ~m; H = hu[sl]
            if m.sum() < 8:
                continue
            dist = ndimage.distance_transform_edt(~m, sampling=sp) - ndimage.distance_transform_edt(m, sampling=sp)   # + outside
            tissue = (H > -50) & (H < 300) & ~other
            core = m & (dist <= -1.0) if (m & (dist <= -1.0)).sum() >= 10 else m
            shell = (dist >= 3.0) & (dist <= 10.0) & tissue
            if shell.sum() < 30:
                continue
            c_in, c_out = float(np.median(H[core])), float(np.median(H[shell]))
            con = c_in - c_out
            ds = np.arange(-8, 9); prof = []
            for a in ds:
                b = (dist >= a - 0.5) & (dist < a + 0.5) & (m | tissue)
                prof.append(float(np.median(H[b])) if b.sum() >= 5 else np.nan)
            prof = np.asarray(prof); ok = np.isfinite(prof)
            sse = [float(((prof[ok] - (c_out + con * 0.5 * (1 - erf(ds[ok] / w)))) ** 2).mean()) for w in WIDTHS]
            ml = float(m.sum() * np.prod(sp) / 1000.0)
            rows.append({"lesion": l, "ml_on_ct": round(ml, 3), "diameter_mm": round((6e3 * ml / np.pi) ** (1 / 3), 1), "contrast_hu": round(con, 1),
                         "core_hu": round(c_in, 1), "shell_hu": round(c_out, 1), "core_sd_hu": round(float(H[core].std()), 1),
                         "edge_mm": WIDTHS[int(np.argmin(sse))], "edge_fit_rms_hu": round(float(np.sqrt(min(sse))), 1)})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"u": u, "pid": pid, "coll": coll, "lesions": rows})); work.commit()
        return json.dumps({"u": u, "lesions": len(rows), "of": int(n)})
    except Exception as e:
        import traceback
        return json.dumps({"u": u, "err": f"{type(e).__name__}: {e}"[:300], "trace": traceback.format_exc()[-800:]})


@app.function(image=image, volumes={"/work": work}, cpu=8, memory=65536, timeout=7200)
def compare_all(size: int = 32, seed: int = 0) -> str:
    import numpy as np
    from scipy.stats import spearmanr
    from feldglas.adapters import radar
    from feldglas.suite import normal_atlas as na
    rng = np.random.default_rng(seed)
    unit = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
    with np.load("/work/falloff_vectors.npz") as z:
        radii, cons, edges = z["radii_mm"], z["contrast_hu"], z["edge_mm"]
        hosts = sorted({k.split("|")[0] for k in z.files if "|" in k})
        D = np.stack([(z[f"{h}|lesion"][..., 0, :].astype(np.float64) - z[f"{h}|base"][:, 0].astype(np.float64)).mean(3) for h in hosts])
    Tb = D.mean(0)                                                               # (radius, contrast, edge, 256): the BANK
    out = {"bank": {"hosts": len(hosts), "radii_mm": radii.tolist(), "contrast_hu": cons.tolist(), "edge_mm": edges.tolist()}}
    U = unit(Tb); nr, nc, ne = Tb.shape[:3]
    B = out["bank"]
    B["norm_by_radius (mean over contrast, edge)"] = [round(float(np.linalg.norm(Tb[i], axis=-1).mean()), 3) for i in range(nr)]
    B["norm_by_contrast"] = [round(float(np.linalg.norm(Tb[:, i], axis=-1).mean()), 3) for i in range(nc)]
    B["norm_by_edge"] = [round(float(np.linalg.norm(Tb[:, :, i], axis=-1).mean()), 3) for i in range(ne)]
    B["cosine_between_radii (same contrast and edge)"] = np.round(np.mean([[[U[i, c, e] @ U[j, c, e] for j in range(nr)] for i in range(nr)] for c in range(nc) for e in range(ne)], 0), 2).tolist()
    B["cosine_between_edges (same radius and contrast)"] = np.round(np.mean([[[U[r, c, i] @ U[r, c, j] for j in range(ne)] for i in range(ne)] for r in range(nr) for c in range(nc)], 0), 2).tolist()
    B["cosine_between_contrasts (same radius and edge)"] = np.round(np.mean([[[U[r, i, e] @ U[r, j, e] for j in range(nc)] for i in range(nc)] for r in range(nr) for e in range(ne)], 0), 2).tolist()
    B["host_to_host_cosine_of_one_cell (how repeatable a painted displacement is)"] = round(float(np.mean([unit(D[a]).reshape(-1, 256)[i] @ unit(D[b]).reshape(-1, 256)[i]
                                                                                        for a in range(len(hosts)) for b in range(a + 1, len(hosts)) for i in range(nr * nc * ne)])), 3)
    # a paired test of the edge: within one host, does a change of EDGE move the vector more than host-to-host noise of the same cell?
    B["edge_effect: |T(sharp) - T(softest)| / |T(sharp)| by radius"] = [round(float(np.mean([np.linalg.norm(Tb[r, c, 0] - Tb[r, c, -1]) / np.linalg.norm(Tb[r, c, 0]) for c in range(nc)])), 2) for r in range(nr)]

    val = radar.ORGANS.index("liver") + 1
    with np.load("/work/text_en.npz") as z:
        text = {k: z[k].astype(np.float64) for k in z.files}
    donors, scans = [], []
    for p in sorted(pathlib.Path("/work/regions").glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            if meta["coll"] != "pancreas_ct" and meta["coll"] not in LIVER_COHORTS:
                continue
            k = np.flatnonzero((z["size"] == size) & (z["rule"] == 0) & (z["organ"] == val))
            d = {"meta": meta, "k": k, "vec": z["vec"][k].astype(np.float64), "tumor_ml": z["tumor_ml"][k].astype(np.float64),
                 "organ_ml": z["organ_ml"][k].astype(np.float64), "ijk": z["ijk"][k], "overlaps": z["overlaps"], "lesions": z["lesions"], "n_rows": len(z["organ"])}
        (donors if meta["coll"] == "pancreas_ct" else scans).append(d)
    dmu = np.concatenate([d["vec"] for d in donors]).mean(0)

    L = []                                                                        # one row per described real lesion
    for d in scans:
        desc_p = pathlib.Path(f"/work/described/{d['meta']['u']}.json")
        clean = d["tumor_ml"] <= 0
        if not desc_p.exists() or clean.sum() < 3 or not len(d["lesions"]):
            continue
        desc = {r["lesion"]: r for r in json.loads(desc_p.read_text())["lesions"]}
        pos_of = np.full(d["n_rows"], -1); pos_of[d["k"]] = np.arange(len(d["k"]))
        ov = d["overlaps"]; ov = ov[pos_of[ov[:, 0].astype(int)] >= 0].copy(); ov[:, 0] = pos_of[ov[:, 0].astype(int)]
        d["ov"] = ov
        cm = d["vec"][clean].mean(0)
        for l, ml, diam, *_ in d["lesions"]:
            o = ov[ov[:, 1] == l]
            if int(l) not in desc or not len(o):
                continue
            b = int(o[np.argmax(o[:, 2]), 0])
            L.append({**desc[int(l)], "coll": d["meta"]["coll"], "pid": d["meta"]["pid"], "delta": d["vec"][b] - cm,
                      "box_tumor_fraction": float(d["tumor_ml"][b] / max(d["organ_ml"][b], 1e-6)), "lesion_share_in_box": float(o[:, 2].max() / max(ml, 1e-9))})
    out["real_lesions"] = {"described": len(L)}
    for c in LIVER_COHORTS:
        X = [x for x in L if x["coll"] == c]
        out["real_lesions"][c] = {"n": len(X), **{f"{k}_q10_q50_q90": [round(float(q), 1) for q in np.quantile([x[k] for x in X], [0.1, 0.5, 0.9])]
                                                  for k in ("diameter_mm", "contrast_hu", "edge_mm", "core_sd_hu")},
                                  "hypodense_share": round(float(np.mean([x["contrast_hu"] < 0 for x in X])), 2)}

    # -- direction: a real lesion's displacement against the bank ------------------------------------
    flat = U.reshape(-1, 256); idx = np.stack(np.unravel_index(np.arange(len(flat)), (nr, nc, ne)), 1)
    use = [x for x in L if abs(x["contrast_hu"]) >= 10 and x["lesion_share_in_box"] >= 0.25]
    dl = unit(np.stack([x["delta"] for x in use]))
    cos = dl @ flat.T
    near = lambda grid, v: int(np.argmin(np.abs(np.asarray(grid) - v)))
    matched = np.array([np.ravel_multi_index((near(radii, x["diameter_mm"] / 2), near(cons, x["contrast_hu"]), near(edges, x["edge_mm"])), (nr, nc, ne)) for x in use])
    pids = np.array([x["pid"] for x in use]); colls = np.array([x["coll"] for x in use])
    loo = np.array([dl[i] @ unit(np.stack([x["delta"] for j, x in enumerate(use) if pids[j] != pids[i]]).mean(0)) for i in range(len(use))])
    bankmean = unit(Tb.reshape(-1, 256).mean(0))
    out["direction"] = {"lesions (|contrast| >= 10 HU, >= 25 % of the lesion in its best box)": len(use),
                        "median_cosine_with": {"the bank's mean (any painted lesion)": round(float(np.median(dl @ bankmean)), 3),
                                               "the DESCRIPTION-MATCHED painted lesion": round(float(np.median(cos[np.arange(len(use)), matched])), 3),
                                               "a RANDOM painted lesion": round(float(np.median(cos[np.arange(len(use)), rng.integers(0, len(flat), len(use))])), 3),
                                               "the BEST painted lesion in the bank": round(float(np.median(cos.max(1))), 3),
                                               "the mean of OTHER patients' real lesions (ceiling)": round(float(np.median(loo)), 3)},
                        "matched_minus_random_by_cohort": {c: round(float(np.median(cos[colls == c][np.arange(int((colls == c).sum())), matched[colls == c]]) - np.median(cos[colls == c])), 3) for c in LIVER_COHORTS}}

    # -- estimation: does the best-matching painted lesion say how big, how dark, how sharp? ----------
    best = idx[cos.argmax(1)]
    soft = np.exp(cos / 0.05); soft /= soft.sum(1, keepdims=True)
    est = {"radius": (radii[best[:, 0]], soft @ radii[idx[:, 0]], np.array([x["diameter_mm"] / 2 for x in use])),
           "contrast": (cons[best[:, 1]], soft @ cons[idx[:, 1]], np.array([x["contrast_hu"] for x in use])),
           "edge": (edges[best[:, 2]], soft @ edges[idx[:, 2]], np.array([x["edge_mm"] for x in use]))}
    out["estimation (Spearman with the measured value: best cell, softmax-weighted cell)"] = {
        k: [round(float(spearmanr(a, t).statistic), 3), round(float(spearmanr(b, t).statistic), 3)] for k, (a, b, t) in est.items()}
    out["estimation"] = {"norm_of_displacement_vs_diameter (Spearman)": round(float(spearmanr(np.linalg.norm([x["delta"] for x in use], axis=1), est["radius"][2]).statistic), 3),
                         "norm_vs_abs_contrast": round(float(spearmanr(np.linalg.norm([x["delta"] for x in use], axis=1), np.abs(est["contrast"][2])).statistic), 3),
                         "norm_vs_box_tumor_fraction": round(float(spearmanr(np.linalg.norm([x["delta"] for x in use], axis=1), [x["box_tumor_fraction"] for x in use]).statistic), 3),
                         "sign_of_contrast_agrees (best cell)": round(float(np.mean(np.sign(est["contrast"][0]) == np.sign(est["contrast"][2]))), 3),
                         "share_of_real_lesions_that_are_hypodense": round(float(np.mean(est["contrast"][2] < 0)), 3)}

    # -- detection: a painted template as the whole query -------------------------------------------
    out["detection"] = {}
    for c, fname in LIVER_COHORTS.items():
        P = [d for d in scans if d["meta"]["coll"] == c and "ov" in d and len(d["k"]) >= 6]

        def run(t):
            fs, sc_, lab = [], [], []
            for d in P:
                sc = (d["vec"] - dmu) @ t; tum, oml = d["tumor_ml"], d["organ_ml"]
                fs.append({"score": sc, "ijk": d["ijk"], "tumor_ml": tum, "organ_ml": oml, "overlaps": d["ov"],
                           "lesions": {int(r[0]): {"ml": float(r[1]), "diameter_mm": float(r[2])} for r in d["lesions"]}})
                A, cl = tum >= 0.25 * np.maximum(oml, 1e-6), tum <= 0
                sc_.append(sc[A | cl]); lab.append(A[A | cl])
            f = na.froc(fs)["at"]["1.0"]
            return {"pooled_auc": round(na.auc(np.concatenate(sc_), np.concatenate(lab)), 3), "lesions_at_1_fp": f["all"],
                    "by_size_mm": {k: v for k, v in f.items() if k.endswith("mm")}}
        X = [x for x in L if x["coll"] == c]
        med = (near(radii, np.median([x["diameter_mm"] for x in X]) / 2), near(cons, np.median([x["contrast_hu"] for x in X])), near(edges, np.median([x["edge_mm"] for x in X])))
        R = {"patients": len(P), "text direction": run(text[fname][1] - text[fname][0]),
             "painted: the whole bank's mean": run(Tb.reshape(-1, 256).mean(0)),
             f"painted: the cohort's median description (r {radii[med[0]]}, {cons[med[1]]} HU, edge {edges[med[2]]})": run(Tb[med]),
             "painted: hypodense only": run(Tb[:, cons < 0].reshape(-1, 256).mean(0)),
             "painted: hyperdense only": run(Tb[:, cons > 0].reshape(-1, 256).mean(0))}
        for i, r in enumerate(radii):
            R[f"painted: radius {r} mm (all contrasts and edges)"] = run(Tb[i].reshape(-1, 256).mean(0))
        for i, e in enumerate(edges):
            R[f"painted: edge {e} mm"] = run(Tb[:, :, i].reshape(-1, 256).mean(0))
        R["real: the mean displacement of ALL described lesions of the other cohort"] = run(np.mean([x["delta"] for x in L if x["coll"] != c], 0))
        out["detection"][c] = R
        print(c, "done", flush=True)
    pathlib.Path("/work/results/described.json").write_text(json.dumps(out, indent=1)); work.commit()
    return json.dumps(out)


@app.local_entrypoint()
def describe(limit: int = 0, force: bool = False):
    V, I = MEDSEG / "results" / "validation", MEDSEG / "results" / "idc"
    se_of = {r["crdc_series_uuid"]: r["se"] for r in csv.DictReader(open(I / "radar_validation_series_all.csv"))}
    seg = {(r["pid"], r["ref_series"]): r["seg_series"] for r in csv.DictReader(open(I / "expert_segs.csv")) if not r["ar"] and r["coll"] in LIVER_COHORTS}
    jobs = [(r["chosen"], r["pid"], r["coll"], seg[(r["pid"], se_of.get(r["chosen"]))], force)
            for r in csv.DictReader(open(V / "choice.csv")) if r["chosen"] and (r["pid"], se_of.get(r["chosen"])) in seg]
    jobs = jobs[:limit] if limit else jobs
    t = time.time(); n = {"ok": 0, "cached": 0, "err": 0}; les = 0
    for j, r in zip(jobs, describe_one.starmap(jobs, return_exceptions=True)):
        r = json.loads(r) if isinstance(r, str) else {"err": repr(r)[:300]}
        n["err" if r.get("err") else "cached" if r.get("cached") else "ok"] += 1; les += r.get("lesions", 0)
        if r.get("err"):
            print(f"  {j[0]}: {r['err']}\n{r.get('trace', '')}")
    print(f"{n}, {les} lesions described, in {time.time() - t:.0f} s")


@app.local_entrypoint()
def compare(size: int = 32):
    bank = MEDSEG / "results" / "validation" / "falloff_vectors.npz"
    if not bank.exists():
        raise SystemExit(f"{bank}: paint the bank first - medseg scripts/radar_response_modal.py::falloff")
    with work.batch_upload(force=True) as up:
        up.put_file(str(bank), "/falloff_vectors.npz")
    r = json.loads(compare_all.remote(size))
    p = MEDSEG / "results" / "validation" / "described.json"
    p.write_text(json.dumps(r, indent=1)); print(f"-> {p}")
