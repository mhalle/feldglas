"""What is in RADAR's vector space? A first map, from region vectors already on the work volume.

`radar_atlas_modal.py::main` left, for 475 scans of four collections, one pooled vector per 32 mm
box of every organ in RADAR's own mask - so the same ten organs, mostly healthy, seen at four
SITES (the healthy donors of `pancreas_ct`, and the livers of kidney patients, the spleens of
liver patients, everybody's muscle and spine). That is what makes site separable from disease
here, which a lesion cohort alone never allows. One CPU container, numpy only, JSON home:

  partition    per organ, how a clean box's variance splits: site (collection), patient within
               site, place in the organ and fill, and what is left
  read-out     what a linear probe reads from a box, patient-disjoint: collection; phase, slice
               thickness and manufacturer WITHIN a collection (or they would be the site again);
               place in the organ; the side of a kidney
  bands        where each factor lives in the healthy donors' own principal axes (0-16, 16-64,
               64-256), in variance share and in donor sigmas per axis - beside the LESION signal.
               (5.6 found whitening's false positives in the quiet band; this says what else is there)
  shared site  is a scan's shift from the donors the same vector in its liver, its spleen, its
               muscle? If so the acquisition can be read off organs nobody is asking about and
               subtracted - tested directly: per-lesion FROC with the shift estimated from OTHER
               organs of the same scan, and with an oracle shift (the target organ's own clean mean)
  place        nearest neighbor of a box in OTHER patients: how far away in the organ does it
               land, against a random box? within the donors, and from another site into them
  axes         the donors' leading liver axes against everything known about a box

Run:  uv run --no-sync modal run tools/radar_space_modal.py
"""
import csv, json, os, pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent


def _up(n):
    return HERE.parents[n] if len(HERE.parents) > n else HERE


MEDSEG = _up(1) / "medseg" / "docs" / "radar-idc-validation"
# the SAME image definition as radar_atlas_modal.py, so Modal reuses the one it built
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("numpy>=1.24", "scipy", "obstore>=0.11", "idc-index", "highdicom>=0.23", "pydicom>=3",
                      "pylibjpeg", "pylibjpeg-libjpeg",
                      "rankfield @ git+https://github.com/mhalle/rankfield.git@v0.3.2",
                      "provender @ git+https://github.com/mhalle/provender.git@v0.1.1")
         .env({"FELDGLAS_STORE": os.environ.get("FELDGLAS_STORE", "")})
         .add_local_python_source("feldglas"))
work = modal.Volume.from_name("feldglas-radar-work", create_if_missing=True)
app = modal.App("feldglas-radar-space")

DONORS = "pancreas_ct"
TARGET = {"hcc_tace_seg": "liver", "colorectal_liver_metastases": "liver", "c4kc_kits": "kidney"}
ORGANS = ("liver", "spleen", "kidney", "pancreas", "stomach", "erector spinae muscle", "lumbar vertebrae",
          "aorta", "large bowel", "lung")
BANDS = ((0, 16), (16, 64), (64, 256))


@app.function(image=image, volumes={"/work": work}, cpu=8, memory=65536, timeout=7200)
def space(series: dict, size: int = 32, seed: int = 0) -> str:
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.suite import normal_atlas as na
    rng = np.random.default_rng(seed)
    try:                                                   # manufacturer and model, from IDC's own index
        from idc_index import IDCClient
        ix = IDCClient().index
        ix = ix[ix["SeriesInstanceUID"].isin({v["se"] for v in series.values()})]
        maker = {r.SeriesInstanceUID: (str(r.Manufacturer), str(r.ManufacturerModelName)) for r in ix.itertuples()}
    except Exception as e:                                  # the map is still worth having without it
        print("no IDC index:", repr(e)[:200], flush=True); maker = {}

    val_of = {n: radar.ORGANS.index(n) + 1 for n in ORGANS}
    S, raw = [], []                                         # scan table; per-scan arrays
    for p in sorted(pathlib.Path("/work/regions").glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            k = np.flatnonzero((z["size"] == size) & (z["rule"] == 0) & np.isin(z["organ"], list(val_of.values())))
            d = {"k": k, "organ": z["organ"][k], "vec": z["vec"][k].astype(np.float32), "coords": z["coords"][k],
                 "fill": z["fill"][k], "tumor_ml": z["tumor_ml"][k], "organ_ml": z["organ_ml"][k], "ijk": z["ijk"][k],
                 "overlaps": z["overlaps"], "lesions": z["lesions"], "n_rows": len(z["organ"])}
        info = series.get(meta["u"], {})
        mk = maker.get(info.get("se"), ("", ""))
        S.append({**{a: meta[a] for a in ("u", "pid", "coll", "phase")}, "th": info.get("th", ""), "maker": mk[0], "model": mk[1]})
        raw.append(d)
    coll = np.array([s["coll"] for s in S]); pid = np.array([s["pid"] for s in S])
    print(f"{len(S)} scans; makers: { {m: int(sum(s['maker'] == m for s in S)) for m in sorted({s['maker'] for s in S})} }", flush=True)
    out = {"scans": len(S), "size_mm": size, "organs": {}, "collections": {c: int((coll == c).sum()) for c in sorted(set(coll))}}

    def organ_set(name, clean=True, cap=None):
        """All boxes of one organ: vectors, scan index, coords, fill. ``clean`` drops boxes with tumor."""
        V, si, C, F = [], [], [], []
        for i, d in enumerate(raw):
            m = d["organ"] == val_of[name]
            if clean:
                m &= d["tumor_ml"] <= 0
            j = np.flatnonzero(m)
            if cap and len(j) > cap:
                j = rng.choice(j, cap, replace=False)
            V.append(d["vec"][j]); si.append(np.full(len(j), i)); C.append(d["coords"][j]); F.append(d["fill"][j])
        return np.concatenate(V).astype(np.float64), np.concatenate(si), np.concatenate(C).astype(np.float64), np.concatenate(F).astype(np.float64)

    def group_means(V, g):
        u, inv = np.unique(g, return_inverse=True)
        M = np.zeros((len(u), V.shape[1])); np.add.at(M, inv, V)
        n = np.bincount(inv); return u, M / n[:, None], n, inv

    def folds(groups, k=5):
        u = rng.permutation(np.unique(groups)); return [np.isin(groups, f) for f in np.array_split(u, k)]

    def ridge(Xa, Ya, Xb, lam=1e-2):
        mu = Xa.mean(0); A = Xa - mu
        W = np.linalg.solve(A.T @ A + lam * np.trace(A.T @ A) / A.shape[1] * np.eye(A.shape[1]), A.T @ (Ya - Ya.mean(0)))
        return (Xb - mu) @ W + Ya.mean(0)

    def classify(X, y, groups):
        """Balanced accuracy of a least-squares classifier, folds disjoint in ``groups``; chance = 1/K."""
        cls = np.unique(y)
        if len(cls) < 2:
            return None
        Y = (y[:, None] == cls[None]).astype(float)
        Y = Y / Y.mean(0)                                   # class-balanced targets
        pred = np.empty(len(y), int)
        for te in folds(groups):
            if len(np.unique(y[~te])) < len(cls):
                return None
            pred[te] = ridge(X[~te], Y[~te], X[te]).argmax(1)
        return {"classes": [str(c) for c in cls], "n": [int((y == c).sum()) for c in cls],
                "balanced_accuracy": round(float(np.mean([(pred[y == c] == i).mean() for i, c in enumerate(cls)])), 3),
                "chance": round(1 / len(cls), 3)}

    def r2(X, Y, groups):
        P = np.empty_like(Y)
        for te in folds(groups):
            P[te] = ridge(X[~te], Y[~te], X[te])
        return [round(float(1 - ((Y[:, j] - P[:, j]) ** 2).sum() / ((Y[:, j] - Y[:, j].mean()) ** 2).sum()), 3) for j in range(Y.shape[1])]

    donor_axes = {}
    shifts = {}                                             # organ -> {scan index: clean mean - donor mean}
    for name in ORGANS:
        V, si, C, F = organ_set(name)
        if len(V) < 2000:
            continue
        R = out["organs"][name] = {"clean_boxes": int(len(V)), "scans": int(len(np.unique(si)))}
        # -- partition -------------------------------------------------------------------------
        mu = V.mean(0); tot = ((V - mu) ** 2).sum()
        su, SM, sn, sinv = group_means(V, si)
        cu, CM, cn, cinv = group_means(V, coll[si])
        ss_coll = (cn[:, None] * (CM - mu) ** 2).sum()
        ss_pat = (sn[:, None] * (SM - CM[np.unique(coll[su], return_inverse=True)[1]]) ** 2).sum()
        W = V - SM[sinv]                                     # within-scan
        Fx = na._features(C, F); Fx = Fx - group_means(Fx, si)[1][sinv]
        coef = np.linalg.lstsq(Fx[:, 1:], W, rcond=None)[0]
        ss_place = ((Fx[:, 1:] @ coef) ** 2).sum()
        R["variance_share"] = {"site": round(float(ss_coll / tot), 3), "patient_within_site": round(float(ss_pat / tot), 3),
                               "place_and_fill_within_scan": round(float(ss_place / tot), 3),
                               "within_scan_residual": round(float(1 - (ss_coll + ss_pat + ss_place) / tot), 3)}
        # -- donors' axes, and where the factors live in them ------------------------------------
        dm = coll[si] == DONORS
        dmu = V[dm].mean(0)
        w, U = np.linalg.eigh(np.cov((V[dm] - dmu).T)); w, U = np.maximum(w[::-1], 1e-12), U[:, ::-1]
        donor_axes[name] = (dmu, w, U)
        R["donor_participation_ratio"] = round(float(w.sum() ** 2 / (w ** 2).sum()), 1)

        def bands(D):
            Z2 = ((np.atleast_2d(D) @ U) ** 2).mean(0)
            return {"share": [round(float(Z2[a:b].sum() / Z2.sum()), 3) for a, b in BANDS],
                    "donor_sigmas_per_axis": [round(float(np.sqrt((Z2[a:b] / w[a:b]).mean())), 2) for a, b in BANDS],
                    "norm": round(float(np.sqrt(Z2.sum())), 3)}
        nd = coll[su] != DONORS
        shifts[name] = {int(s): SM[i] - dmu for i, s in enumerate(su)}
        B = R["bands_0-16_16-64_64-256"] = {
            "site_shift (scan mean - donor mean, other collections)": bands(SM[nd] - dmu),
            "donor_to_donor (donor scan mean - donor mean)": bands(SM[~nd] - dmu),
            "place_and_fill (fitted, within scan)": bands(Fx[:, 1:] @ coef),
            "within_scan_residual": bands(W - Fx[:, 1:] @ coef)}
        for c in sorted(set(coll)):
            if c != DONORS and (coll[su] == c).sum() >= 10:
                B[f"site_shift: {c}"] = bands(SM[coll[su] == c] - dmu)
        # -- read-outs ---------------------------------------------------------------------------
        keep = np.concatenate([rng.choice(np.flatnonzero(si == s), min(60, int((si == s).sum())), replace=False) for s in su])
        X, g = V[keep], pid[si[keep]]
        RO = R["read_out"] = {"collection": classify(X, coll[si[keep]], g)}
        for c in sorted(set(coll)):
            m = coll[si[keep]] == c
            for fact in ("phase", "th", "maker"):
                y = np.array([S[i][fact] for i in si[keep][m]])
                ok = np.isin(y, [v for v in np.unique(y) if v and (y == v).sum() >= 300])
                if ok.sum() and len(np.unique(y[ok])) > 1:
                    r = classify(X[m][ok], y[ok], g[m][ok])
                    if r:
                        RO[f"{fact} within {c}"] = r
        dk = keep[coll[si[keep]] == DONORS]
        RO["place_r2_zyx_donors"] = r2(V[dk], C[dk], pid[si[dk]])
        RO["fill_r2_donors"] = r2(V[dk], F[dk][:, None], pid[si[dk]])[0]
        ok_ = keep[coll[si[keep]] != DONORS]
        RO["place_r2_zyx_trained_on_donors_tested_elsewhere"] = [
            round(float(1 - ((C[ok_][:, j] - P) ** 2).sum() / ((C[ok_][:, j] - C[ok_][:, j].mean()) ** 2).sum()), 3)
            for j, P in enumerate(ridge(V[dk], C[dk], V[ok_]).T)]
        if name == "kidney":
            RO["side (x < 0.5) donors"] = classify(V[dk], (C[dk][:, 2] < 0.5).astype(int), pid[si[dk]])
        # -- place: nearest neighbor in OTHER patients ------------------------------------------
        def nn_place(q, ref):
            sim = V[q] @ V[ref].T
            sim[pid[si[q]][:, None] == pid[si[ref]][None]] = -2
            j = ref[sim.argmax(1)]
            err = np.linalg.norm(C[q] - C[j], axis=1)
            rand = np.linalg.norm(C[q] - C[rng.choice(ref, len(q))], axis=1)
            return {"median_error": round(float(np.median(err)), 3), "random_pair": round(float(np.median(rand)), 3),
                    "within_0.15": round(float((err < 0.15).mean()), 3), "random_within_0.15": round(float((rand < 0.15).mean()), 3)}
        ref = rng.choice(dk, min(20000, len(dk)), replace=False)
        R["nearest_neighbor_place"] = {"donor_box -> other donors": nn_place(rng.choice(dk, min(3000, len(dk)), replace=False), ref)}
        if len(ok_):
            R["nearest_neighbor_place"]["other_site_box -> donors"] = nn_place(rng.choice(ok_, min(3000, len(ok_)), replace=False), ref)
        # -- the donors' leading axes against what is known about a box ---------------------------
        if name in ("liver", "kidney"):
            Z = (V[dm] - dmu) @ U[:, :8]
            known = {"z": C[dm][:, 0], "y": C[dm][:, 1], "x": C[dm][:, 2], "fill": F[dm],
                     **{f"phase={ph}": (np.array([S[i]["phase"] for i in si[dm]]) == ph).astype(float) for ph in ("arterial", "venous", "late")},
                     "slice_mm": np.array([float(S[i]["th"] or "nan") for i in si[dm]])}
            R["donor_axes_0-7"] = {"variance_share": [round(float(x), 3) for x in w[:8] / w.sum()],
                                   "abs_correlation": {kname: [round(float(abs(np.corrcoef(Z[np.isfinite(kv), a], kv[np.isfinite(kv)])[0, 1])), 2)
                                                               for a in range(8)] for kname, kv in known.items() if np.nanstd(kv) > 0}}
        print(name, "done", flush=True)

    # -- is the site shift ONE vector per scan, whatever the organ? -------------------------------
    names = [n for n in ORGANS if n in shifts]
    nd_scans = [i for i in range(len(S)) if coll[i] != DONORS]
    cosm = np.full((len(names), len(names)), np.nan)
    for a, A in enumerate(names):
        for b, Bn in enumerate(names):
            both = [i for i in nd_scans if i in shifts[A] and i in shifts[Bn]]
            if len(both) >= 30:
                x = np.stack([shifts[A][i] for i in both]); y = np.stack([shifts[Bn][i] for i in both])
                cosm[a, b] = np.mean((x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12))
    out["site_shift_shared_across_organs"] = {
        "organs": names, "mean_cosine_of_a_scans_shift_in_two_organs": [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in cosm],
        "note": "each organ's vector comes through its own query and projection; a cosine near 1 means the acquisition moves them together anyway"}

    # -- and can it be subtracted? per-lesion FROC with the shift taken from OTHER organs ----------
    out["site_correction"] = {}
    for c, tname in TARGET.items():
        dmu, w, U = donor_axes[tname]
        P = []
        for i, d in enumerate(raw):
            if coll[i] != c or not len(d["lesions"]):
                continue
            j = np.flatnonzero(d["organ"] == val_of[tname])
            if len(j) < 6:
                continue
            others = [shifts[o][i] for o in names if o != tname and o not in ("kidney", "liver") and i in shifts[o]]
            clean = d["tumor_ml"][j] <= 0
            P.append((d, j, d["vec"][j].astype(np.float64), np.mean(others, 0) if others else np.zeros(256),
                      d["vec"][j][clean].astype(np.float64).mean(0) - dmu if clean.sum() >= 3 else np.zeros(256)))

        def run(score):
            fs, pooled, lab = [], [], []
            for d, j, Vt, off, oracle in P:
                sc = score(Vt, off, oracle); tum = d["tumor_ml"][j].astype(np.float64); oml = d["organ_ml"][j].astype(np.float64)
                pos_of = np.full(d["n_rows"], -1); pos_of[d["k"][j]] = np.arange(len(j))
                ov = d["overlaps"]; ov = ov[pos_of[ov[:, 0].astype(int)] >= 0].copy(); ov[:, 0] = pos_of[ov[:, 0].astype(int)]
                fs.append({"score": sc, "ijk": d["ijk"][j], "tumor_ml": tum, "organ_ml": oml, "overlaps": ov,
                           "lesions": {int(r[0]): {"ml": float(r[1]), "diameter_mm": float(r[2])} for r in d["lesions"]}})
                A, cl = tum >= 0.25 * np.maximum(oml, 1e-6), tum <= 0
                pooled.append(sc[A | cl]); lab.append(A[A | cl])
            f = na.froc(fs)
            return {"pooled_auc": round(na.auc(np.concatenate(pooled), np.concatenate(lab)), 3),
                    "sens_at_0.5_1_2_4": [f["at"][k]["all"] for k in ("0.5", "1.0", "2.0", "4.0")]}
        maha = lambda D: np.sqrt((((D) @ U) ** 2 / w).sum(1))
        cosines = [float(off @ orc / (np.linalg.norm(off) * np.linalg.norm(orc) + 1e-12)) for *_, off, orc in P]
        out["site_correction"][c] = {
            "patients": len(P), "cosine_other_organs_shift_vs_target_organs_own": [round(float(q), 2) for q in np.quantile(cosines, [0.1, 0.5, 0.9])],
            "plain": run(lambda Vt, off, orc: np.linalg.norm(Vt - dmu, axis=1)),
            "plain, shift from other organs removed": run(lambda Vt, off, orc: np.linalg.norm(Vt - dmu - off, axis=1)),
            "plain, oracle shift removed": run(lambda Vt, off, orc: np.linalg.norm(Vt - dmu - orc, axis=1)),
            "whitened": run(lambda Vt, off, orc: maha(Vt - dmu)),
            "whitened, shift from other organs removed": run(lambda Vt, off, orc: maha(Vt - dmu - off)),
            "whitened, oracle shift removed": run(lambda Vt, off, orc: maha(Vt - dmu - orc))}
        # the lesion signal in the donors' axes, beside the site's (bands above)
        disp = []
        for d, j, Vt, off, orc in P:
            tum, oml = d["tumor_ml"][j], d["organ_ml"][j]
            A, cl = tum >= 0.25 * np.maximum(oml, 1e-6), tum <= 0
            if A.sum() >= 1 and cl.sum() >= 3:
                disp.append(Vt[A].mean(0) - Vt[cl].mean(0))
        Z2 = ((np.stack(disp) @ U) ** 2).mean(0)
        out["site_correction"][c]["lesion_displacement_bands"] = {
            "share": [round(float(Z2[a:b].sum() / Z2.sum()), 3) for a, b in BANDS],
            "donor_sigmas_per_axis": [round(float(np.sqrt((Z2[a:b] / w[a:b]).mean())), 2) for a, b in BANDS],
            "norm": round(float(np.sqrt(Z2.sum())), 3), "patients": len(disp)}
        print(c, "correction done", flush=True)
    pathlib.Path("/work/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"/work/results/vector_space_{size}.json").write_text(json.dumps(out, indent=1)); work.commit()
    return json.dumps(out)


_SECRET = os.environ.get("FELDGLAS_MODAL_SECRET", "feldglas-r2")
PER_LABEL = 40


@app.function(image=image, secrets=[modal.Secret.from_name(_SECRET)], volumes={"/work": work}, cpu=2, memory=8192,
              timeout=1200, max_containers=40)
def token_sample(u: str, digest: str, pid: str, coll: str, phase: str, force: bool = False) -> str:
    """RAW tokens - the encoder's features BEFORE the organ's query and projection - sampled from
    one field: up to PER_LABEL per structure of RADAR's own mask (and background) per lattice,
    each with the structure that fills most of its box, how much of it, and where it is."""
    import tempfile
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.gate import occupancy
    from feldglas.remote import open_blobs
    from feldglas.store import read_field
    out_path = pathlib.Path(f"/work/tokens/{u}.npz")
    if out_path.exists() and not force:
        return json.dumps({"u": u, "cached": True})
    try:
        rng = np.random.default_rng(abs(hash(u)) % (2 ** 32))
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / f"{u}.npz"
            if not open_blobs(radar.NAME, check=False).fetch(digest, p):
                raise FileNotFoundError(f"blob {digest} is gone, or did not match its name")
            field = read_field(p)
        mask = field.native_mask
        vals = [int(v) for v in np.unique(mask) if v]
        liver = np.argwhere(mask == radar.ORGANS.index("liver") + 1)
        frame = field.grid.world(liver.mean(0)) if len(liver) else None          # the body frame's origin: the liver's centroid
        T = {"lattice": [], "label": [], "purity": [], "world": [], "organ_coords": [], "vec": []}
        bbox = {v: (np.array([a.min() for a in np.nonzero(mask == v)]), np.array([a.max() for a in np.nonzero(mask == v)])) for v in vals}
        for j, k in enumerate(field.kernels):
            occ = np.stack([occupancy(mask == v, k) for v in vals])                 # (labels, tokens)
            best = occ.argmax(0); pur = occ.max(0)
            bg = 1.0 - occ.sum(0)
            label = np.where(bg > pur, 0, np.asarray(vals)[best]); pur = np.maximum(pur, bg)
            centers = field.token_centers(j)
            shape = field.lattice_shape(j)
            idx3 = np.stack(np.unravel_index(np.arange(len(label)), shape), 1) * np.asarray(k) + (np.asarray(k) - 1) / 2.0
            for v in [0] + vals:
                cand = np.flatnonzero(label == v)
                if not len(cand):
                    continue
                take = rng.choice(cand, min(PER_LABEL * (3 if v == 0 else 1), len(cand)), replace=False)
                T["lattice"].append(np.full(len(take), j, np.int8)); T["label"].append(np.full(len(take), v, np.int8))
                T["purity"].append(pur[take].astype(np.float32)); T["world"].append(centers[take].astype(np.float32))
                oc = ((idx3[take] - bbox[v][0]) / np.maximum(bbox[v][1] - bbox[v][0], 1)) if v else np.full((len(take), 3), np.nan)
                T["organ_coords"].append(oc.astype(np.float32)); T["vec"].append(np.asarray(field.tokens[j][take], np.float16))
        meta = {"u": u, "pid": pid, "coll": coll, "phase": phase, "tokens": int(sum(len(x) for x in T["label"])),
                "frame_origin_lps": None if frame is None else [float(x) for x in frame]}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez(tmp, meta=np.array(json.dumps(meta)), **{a: np.concatenate(b) for a, b in T.items()})
        tmp.replace(out_path); work.commit()
        return json.dumps(meta)
    except Exception as e:
        import traceback
        return json.dumps({"u": u, "err": f"{type(e).__name__}: {e}"[:300], "trace": traceback.format_exc()[-1000:]})


@app.function(image=image, volumes={"/work": work}, cpu=8, memory=65536, timeout=7200)
def token_space(series: dict, seed: int = 0) -> str:
    """What one RAW token knows, per lattice, read by linear probes that never see a patient twice:
    which structure it is in, where it is in the body (mm from the liver's centroid) and in its
    organ, which side, which site, which phase - and whether its nearest neighbor in ANOTHER
    patient is the same structure at the same place."""
    import numpy as np
    from feldglas.adapters import radar
    rng = np.random.default_rng(seed)
    L, LAB, PUR, POS, OC, V, SI, S = [], [], [], [], [], [], [], []
    for p in sorted(pathlib.Path("/work/tokens").glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            if meta["frame_origin_lps"] is None:
                continue
            n = len(z["label"])
            L.append(z["lattice"]); LAB.append(z["label"]); PUR.append(z["purity"]); OC.append(z["organ_coords"])
            POS.append(z["world"] - np.asarray(meta["frame_origin_lps"], np.float32)); V.append(z["vec"].astype(np.float32))
            SI.append(np.full(n, len(S))); S.append({**meta, "th": series.get(meta["u"], {}).get("th", "")})
    L, LAB, PUR, POS, OC, V, SI = (np.concatenate(x) for x in (L, LAB, PUR, POS, OC, V, SI))
    coll = np.array([s["coll"] for s in S])[SI]; pid = np.array([s["pid"] for s in S])[SI]; phase = np.array([s["phase"] for s in S])[SI]
    print(f"{len(S)} scans, {len(V)} tokens", flush=True)

    def folds(groups, k=5):
        u = rng.permutation(np.unique(groups)); return [np.isin(groups, f) for f in np.array_split(u, k)]

    def ridge(Xa, Ya, Xb, lam=1e-3):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        A = (Xa - mu) / sd
        W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Ya - Ya.mean(0)))
        return ((Xb - mu) / sd) @ W + Ya.mean(0)

    def classify(X, y, groups, train=None, test=None):
        cls = np.unique(y); Y = (y[:, None] == cls[None]).astype(np.float64); Y = Y / Y.mean(0)
        pred = np.full(len(y), -1)
        if train is not None:
            pred[test] = ridge(X[train], Y[train], X[test]).argmax(1); m = test
        else:
            for te in folds(groups):
                pred[te] = ridge(X[~te], Y[~te], X[te]).argmax(1)
            m = np.ones(len(y), bool)
        per = {str(c): round(float((pred[m & (y == c)] == i).mean()), 3) for i, c in enumerate(cls) if (m & (y == c)).sum()}
        return {"balanced_accuracy": round(float(np.mean(list(per.values()))), 3), "chance": round(1 / len(cls), 3), "per_class": per}

    out = {"scans": len(S), "tokens": int(len(V)), "lattices": {}}
    for j, lname in enumerate(("deep (8,32,32)", "mid (4,16,16)", "fine (2,8,8)")):
        m = L == j
        X, lab, pur, pos, oc, g, c_, ph = V[m].astype(np.float64), LAB[m], PUR[m], POS[m].astype(np.float64), OC[m], pid[m], coll[m], phase[m]
        R = out["lattices"][lname] = {"tokens": int(m.sum()), "median_purity": round(float(np.median(pur)), 2)}
        common = [v for v in np.unique(lab) if (lab == v).sum() >= 2000]
        k = np.isin(lab, common)
        names = {0: "background", **{i + 1: n for i, n in enumerate(radar.ORGANS)}}
        r = classify(X[k], lab[k], g[k])
        r["per_class"] = {names[int(a)]: b for a, b in r["per_class"].items()}
        R["structure"] = r
        don = c_ == "pancreas_ct"
        r = classify(X[k], lab[k], g[k], train=don[k], test=~don[k])
        R["structure, trained on donors, tested at the other sites"] = {a: r[a] for a in ("balanced_accuracy", "chance")}
        hi = k & (pur >= 0.9)
        if hi.sum() > 5000 and len(np.unique(lab[hi])) > 2:
            r = classify(X[hi], lab[hi], g[hi]); R["structure, tokens >= 0.9 one structure"] = {a: r[a] for a in ("balanced_accuracy", "chance")}
        # where in the body: mm from the liver's centroid, along L, P, S
        P = np.empty_like(pos)
        for te in folds(g):
            P[te] = ridge(X[~te], pos[~te], X[te])
        R["position_from_liver_centroid_LPS"] = {
            "r2": [round(float(1 - ((pos[:, a] - P[:, a]) ** 2).sum() / ((pos[:, a] - pos[:, a].mean()) ** 2).sum()), 3) for a in range(3)],
            "median_abs_error_mm": [round(float(np.median(np.abs(pos[:, a] - P[:, a]))), 1) for a in range(3)],
            "spread_mm (median abs deviation)": [round(float(np.median(np.abs(pos[:, a] - np.median(pos[:, a])))), 1) for a in range(3)]}
        R["side (left of the liver centroid's sagittal plane + 60 mm)"] = classify(X, (pos[:, 0] > 60).astype(int), g)["balanced_accuracy"]
        for organ in ("liver", "kidney", "lung"):
            o = lab == radar.ORGANS.index(organ) + 1
            if o.sum() >= 3000:
                Po = np.empty((int(o.sum()), 3))
                for te in folds(g[o]):
                    Po[te] = ridge(X[o][~te], oc[o][~te].astype(np.float64), X[o][te])
                R[f"place within {organ} (r2 z, y, x)"] = [round(float(1 - ((oc[o][:, a] - Po[:, a]) ** 2).sum() / ((oc[o][:, a] - oc[o][:, a].mean()) ** 2).sum()), 3) for a in range(3)]
        R["collection"] = {a: b for a, b in classify(X, c_, g).items() if a != "per_class"}
        liv = lab == radar.ORGANS.index("liver") + 1
        R["collection, liver tokens only"] = {a: b for a, b in classify(X[liv], c_[liv], g[liv]).items() if a != "per_class"}
        kk = c_ == "c4kc_kits"
        R["phase within c4kc_kits"] = classify(X[kk], ph[kk], g[kk])
        # nearest neighbor in ANOTHER patient: same structure? how far away in the body?
        q = rng.choice(np.flatnonzero(k), min(4000, int(k.sum())), replace=False)
        ref = rng.choice(np.flatnonzero(k), min(40000, int(k.sum())), replace=False)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        sim = Xn[q] @ Xn[ref].T
        sim[g[q][:, None] == g[ref][None]] = -2
        nn = ref[sim.argmax(1)]
        rnd = rng.choice(ref, len(q))
        R["nearest_neighbor_in_another_patient"] = {
            "same_structure": round(float((lab[nn] == lab[q]).mean()), 3), "random": round(float((lab[rnd] == lab[q]).mean()), 3),
            "median_distance_mm_in_the_liver_frame": round(float(np.median(np.linalg.norm(pos[nn] - pos[q], axis=1))), 1),
            "random_mm": round(float(np.median(np.linalg.norm(pos[rnd] - pos[q], axis=1))), 1)}
        # how the variance of a token splits
        mu = X.mean(0); tot = ((X - mu) ** 2).sum()
        def between(y):
            u, inv = np.unique(y, return_inverse=True); M = np.zeros((len(u), X.shape[1])); np.add.at(M, inv, X)
            n = np.bincount(inv); return float((n[:, None] * (M / n[:, None] - mu) ** 2).sum() / tot)
        R["variance_share_between"] = {"structures": round(between(lab), 3), "collections": round(between(c_), 3), "scans": round(between(SI[m]), 3)}
        print(lname, "done", flush=True)
    pathlib.Path("/work/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path("/work/results/token_space.json").write_text(json.dumps(out, indent=1)); work.commit()
    return json.dumps(out)


def _series():
    return {r["crdc_series_uuid"]: {"se": r["se"], "th": r["th"]}
            for r in csv.DictReader(open(MEDSEG / "results" / "idc" / "radar_validation_series_all.csv"))}


@app.local_entrypoint()
def tokens(limit: int = 0, force: bool = False):
    """Sample raw tokens from every field in the manifest, then map them. Needs FELDGLAS_STORE."""
    import time
    from feldglas.remote import Manifest
    if not os.environ.get("FELDGLAS_STORE", "").startswith(("s3://", "gs://", "az://")):
        raise SystemExit("set FELDGLAS_STORE (e.g. s3://<bucket>/feldglas): the fields are read from it")
    mf = Manifest.load(HERE.parent / "manifests" / "radar.json")
    rows = [r for r in csv.DictReader(open(MEDSEG / "results" / "validation" / "choice.csv"))
            if r["chosen"] and f"{r['chosen']}.npz" in mf.files]
    jobs = [(r["chosen"], mf.files[f"{r['chosen']}.npz"]["digest"], r["pid"], r["coll"], r["phase"], force) for r in rows]
    jobs = jobs[:limit] if limit else jobs
    t = time.time(); n = {"ok": 0, "cached": 0, "err": 0}
    for j, r in zip(jobs, token_sample.starmap(jobs, return_exceptions=True)):
        r = json.loads(r) if isinstance(r, str) else {"err": repr(r)[:300]}
        n["err" if r.get("err") else "cached" if r.get("cached") else "ok"] += 1
        if r.get("err"):
            print(f"  {j[0]}: {r['err']}")
    print(f"{n} in {time.time() - t:.0f} s")
    if not limit:
        out = json.loads(token_space.remote(_series()))
        p = MEDSEG / "results" / "validation" / "token_space.json"
        p.write_text(json.dumps(out, indent=1)); print(f"-> {p}")


@app.local_entrypoint()
def main(size: int = 32, out: str = ""):
    series = {r["crdc_series_uuid"]: {"se": r["se"], "th": r["th"]}
              for r in csv.DictReader(open(MEDSEG / "results" / "idc" / "radar_validation_series_all.csv"))}
    r = json.loads(space.remote(series, size))
    p = pathlib.Path(out) if out else MEDSEG / "results" / "validation" / f"vector_space_{size}.json"
    p.write_text(json.dumps(r, indent=1)); print(f"-> {p}")
