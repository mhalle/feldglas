"""The unnamed half: what RADAR's vocabulary spans, and what the vectors carry outside it.

A finding's score is a sigmoid of ``v . (t_pos - t_neg) / T``: all the vocabulary can say about a
box is its projection on that finding's DIRECTION. So the 146 directions span a NAMED subspace of
the 256 (an organ's own findings a smaller one), and everything else in a vector is, to upstream's
interface, invisible. This asks, on the pooled 32 mm boxes already on the work volume:

  geometry     how many dimensions the vocabulary really has; how alike an organ's findings are;
               where the directions lie in the healthy donors' own axes; whether a finding's direction
               points where real lesions of that kind actually go
  shares       how much of each kind of variation - another person, another site, place in the organ,
               within-scan texture, a LESION - falls inside the named subspace, against a random
               subspace of the same size (k / 256) and the best possible one (the data's own top k axes)
  disease      detection with the vector cut down to: the organ's findings, all findings, and the
               complement of all findings - unsupervised (distance from the donors) and supervised
               (a fitted direction, patient-disjoint): is there disease the vocabulary cannot express?
  read-outs    place, side, site, slice thickness from each part: what is the vocabulary blind to?
  kinds        HCC against metastasis from tumor boxes, in each part, beside the same classifier on
               the same patients' CLEAN boxes (the two diseases are two sites; only the excess counts)
  handles      the leading axes of what is left when every named direction is removed, against
               everything known about a box - the ones that match nothing are the unnamed handles

  ::shots      how many example patients an EMPIRICAL lesion direction needs to beat the text direction

Run:  uv run --no-sync modal run tools/radar_unnamed_modal.py::main
      uv run --no-sync modal run tools/radar_unnamed_modal.py::shots
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
app = modal.App("feldglas-radar-unnamed")

DONORS = "pancreas_ct"
COHORTS = {"liver": {"colorectal_liver_metastases": "Liver_Metastasis", "hcc_tace_seg": "Liver_Hepatocellular carcinoma"},
           "kidney": {"c4kc_kits": "Kidney_Renal cell carcinoma"}}


@app.function(image=image, volumes={"/work": work}, cpu=8, memory=65536, timeout=7200)
def unnamed(series: dict, size: int = 32, seed: int = 0) -> str:
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.suite import normal_atlas as na
    rng = np.random.default_rng(seed)
    with np.load("/work/text_en.npz") as z:
        text = {k: z[k].astype(np.float64) for k in z.files}
    D_all = {k: v[1] - v[0] for k, v in text.items()}                       # a finding's direction: positive minus negative

    def basis(names):
        M = np.stack([D_all[n] for n in names])
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        keep = s > 1e-6 * s[0]
        return Vt[keep].T, s[keep]                                          # (256, k) orthonormal

    Q_all, s_all = basis(sorted(D_all))
    out = {"size_mm": size, "vocabulary": {
        "findings": len(D_all), "rank": int(Q_all.shape[1]),
        "effective_dimensions (participation ratio of singular values^2)": round(float((s_all ** 2).sum() ** 2 / (s_all ** 4).sum()), 1),
        "dimensions_for_90_and_99_percent": [int(np.searchsorted(np.cumsum(s_all ** 2) / (s_all ** 2).sum(), q) + 1) for q in (0.9, 0.99)],
        "direction_norm_q10_q50_q90": [round(float(q), 3) for q in np.quantile([np.linalg.norm(d) for d in D_all.values()], [0.1, 0.5, 0.9])]}, "organs": {}}

    S, raw = [], []
    for p in sorted(pathlib.Path("/work/regions").glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            k = np.flatnonzero((z["size"] == size) & (z["rule"] == 0) & np.isin(z["organ"], [radar.ORGANS.index(o) + 1 for o in COHORTS]))
            raw.append({"k": k, "organ": z["organ"][k], "vec": z["vec"][k].astype(np.float32), "coords": z["coords"][k], "fill": z["fill"][k],
                        "tumor_ml": z["tumor_ml"][k], "organ_ml": z["organ_ml"][k], "ijk": z["ijk"][k], "overlaps": z["overlaps"],
                        "lesions": z["lesions"], "n_rows": len(z["organ"])})
        S.append({**{a: meta[a] for a in ("u", "pid", "coll", "phase")}, "th": series.get(meta["u"], {}).get("th", "")})
    coll = np.array([s["coll"] for s in S]); pid = np.array([s["pid"] for s in S])
    print(len(S), "scans", flush=True)

    def folds(groups, k=5):
        u = rng.permutation(np.unique(groups)); return [np.isin(groups, f) for f in np.array_split(u, k)]

    def ridge(Xa, Ya, Xb, lam=1e-2):
        mu = Xa.mean(0); A = Xa - mu
        W = np.linalg.solve(A.T @ A + (lam * np.trace(A.T @ A) / A.shape[1] + 1e-12) * np.eye(A.shape[1]), A.T @ (Ya - Ya.mean(0)))
        return (Xb - mu) @ W + Ya.mean(0)

    def cv_scores(X, y, g):
        """Out-of-fold least-squares score of a binary label."""
        sc = np.empty(len(y))
        for te in folds(g):
            sc[te] = ridge(X[~te], y[~te].astype(float)[:, None], X[te])[:, 0]
        return sc

    def cv_accuracy(X, y, g):
        cls = np.unique(y); Y = (y[:, None] == cls[None]).astype(float); Y = Y / Y.mean(0)
        pred = np.empty(len(y), int)
        for te in folds(g):
            pred[te] = ridge(X[~te], Y[~te], X[te]).argmax(1)
        return round(float(np.mean([(pred[y == c] == i).mean() for i, c in enumerate(cls)])), 3)

    def cv_r2(X, Y, g):
        P = np.empty_like(Y)
        for te in folds(g):
            P[te] = ridge(X[~te], Y[~te], X[te])
        return [round(float(1 - ((Y[:, j] - P[:, j]) ** 2).sum() / ((Y[:, j] - Y[:, j].mean()) ** 2).sum()), 3) for j in range(Y.shape[1])]

    for organ, cohorts in COHORTS.items():
        val = radar.ORGANS.index(organ) + 1
        own = sorted(n for n in D_all if n.lower().startswith(organ + "_"))
        Q_own, s_own = basis(own)
        # everything about a box of this organ, all scans
        V, si, C, F, T, O = [], [], [], [], [], []
        for i, d in enumerate(raw):
            j = np.flatnonzero(d["organ"] == val)
            V.append(d["vec"][j]); si.append(np.full(len(j), i)); C.append(d["coords"][j]); F.append(d["fill"][j])
            T.append(d["tumor_ml"][j]); O.append(d["organ_ml"][j])
        V, si, C, F, T, O = (np.concatenate(x).astype(np.float64) for x in (V, si, C, F, T, O)); si = si.astype(int)
        frac = T / np.maximum(O, 1e-6)
        clean, tumor = T <= 0, frac >= 0.25
        dm = (coll[si] == DONORS) & clean
        dmu = V[dm].mean(0)
        w, U = np.linalg.eigh(np.cov((V[dm] - dmu).T)); w, U = np.maximum(w[::-1], 1e-12), U[:, ::-1]
        R = out["organs"][organ] = {"boxes": int(len(V)), "findings": own, "own_rank": int(Q_own.shape[1])}

        # -- geometry of the organ's vocabulary --------------------------------------------------------
        Dn = np.stack([D_all[n] / np.linalg.norm(D_all[n]) for n in own])
        G = Dn @ Dn.T
        R["vocabulary_geometry"] = {
            "mean_cosine_between_two_findings": round(float(G[np.triu_indices(len(own), 1)].mean()), 3),
            "share_of_first_principal_direction": round(float(s_own[0] ** 2 / (s_own ** 2).sum()), 3),
            "effective_dimensions": round(float((s_own ** 2).sum() ** 2 / (s_own ** 4).sum()), 1),
            "share_of_the_directions_in_donor_axes_0-16_16-64_64-256": [round(float(((Dn @ U[:, a:b]) ** 2).sum() / len(own)), 3) for a, b in ((0, 16), (16, 64), (64, 256))],
            "share_of_the_directions_inside_the_span_of_real_boxes (top 64 axes of all boxes)": None}
        wa, Ua = np.linalg.eigh(np.cov((V - V.mean(0)).T)); Ua = Ua[:, ::-1]
        R["vocabulary_geometry"]["share_of_the_directions_inside_the_span_of_real_boxes (top 64 axes of all boxes)"] = round(float(((Dn @ Ua[:, :64]) ** 2).sum() / len(own)), 3)

        # -- shares: how much of each kind of variation is named ---------------------------------------
        su, inv = np.unique(si[clean], return_inverse=True)
        SM = np.zeros((len(su), 256)); np.add.at(SM, inv, V[clean]); SM /= np.bincount(inv)[:, None]
        Wres = V[clean] - SM[inv]
        Fx = na._features(C[clean], F[clean]); Fm = np.zeros((len(su), Fx.shape[1])); np.add.at(Fm, inv, Fx); Fx = Fx - (Fm / np.bincount(inv)[:, None])[inv]
        place = Fx[:, 1:] @ np.linalg.lstsq(Fx[:, 1:], Wres, rcond=None)[0]
        kinds = {"another healthy person (donor scan means)": SM[coll[su] == DONORS] - dmu,
                 "another site (scan means of the other collections)": SM[coll[su] != DONORS] - dmu,
                 "place in the organ and fill": place, "within-scan residual": Wres - place}
        disp = {}
        for c in cohorts:
            rows = []
            for s in np.unique(si[coll[si] == c]):
                m = si == s
                if (m & tumor).sum() >= 1 and (m & clean).sum() >= 3:
                    rows.append(V[m & tumor].mean(0) - V[m & clean].mean(0))
            if rows:
                disp[c] = np.stack(rows); kinds[f"a LESION ({c}: tumor boxes - the patient's clean boxes)"] = disp[c]

        def share(X, Q):
            return float(((X @ Q) ** 2).sum() / (X ** 2).sum())

        def best(X, k):
            s = np.linalg.svd(X[rng.choice(len(X), min(len(X), 20000), replace=False)], compute_uv=False) ** 2
            return float(s[:k].sum() / s.sum())
        R["share_inside_the_named_subspace"] = {"chance": {"own findings": round(Q_own.shape[1] / 256, 3), "all findings": round(Q_all.shape[1] / 256, 3)}}
        for name, X in kinds.items():
            R["share_inside_the_named_subspace"][name] = {
                "own findings": round(share(X, Q_own), 3), "all findings": round(share(X, Q_all), 3),
                "best possible with as many axes (own, all)": [round(best(X, Q_own.shape[1]), 3), round(best(X, Q_all.shape[1]), 3)]}

        # -- does a finding's direction point where real lesions go? -----------------------------------
        R["finding_direction_vs_real_displacement"] = {}
        for c, fname in cohorts.items():
            if c not in disp:
                continue
            m = disp[c].mean(0); mn = m / np.linalg.norm(m)
            cos = {n: float(mn @ D_all[n] / np.linalg.norm(D_all[n])) for n in own}
            top = sorted(cos.items(), key=lambda kv: -kv[1])[:4]
            R["finding_direction_vs_real_displacement"][c] = {
                "patients": int(len(disp[c])), "cosine_with_its_own_finding": round(cos[fname], 3),
                "best_aligned_findings": [[n, round(v, 3)] for n, v in top],
                "share_of_the_mean_displacement_inside_own_findings": round(share(m[None], Q_own), 3),
                "consistency (mean cosine of a patient's displacement with the mean)": round(float(np.mean(disp[c] @ mn / np.linalg.norm(disp[c], axis=1))), 3)}

        # -- the vector cut into parts -----------------------------------------------------------------
        Pn = Q_all @ Q_all.T
        parts = {"whole vector (256)": lambda X: X, f"own findings ({Q_own.shape[1]})": lambda X: X @ Q_own,
                 f"all findings ({Q_all.shape[1]})": lambda X: X @ Q_all, f"complement of all findings ({256 - Q_all.shape[1]})": lambda X: X - X @ Pn}
        R["disease_by_part"] = {}
        for c, fname in cohorts.items():
            scans = [s for s in np.unique(si[coll[si] == c]) if len(raw[s]["lesions"]) and (si == s).sum() >= 6]
            res = {}
            for pname, cut in parts.items():
                Z = cut(V - dmu)
                fs, pooled, lab = [], [], []
                for s in scans:
                    m = np.flatnonzero(si == s); d = raw[s]; j = np.flatnonzero(d["organ"] == val)
                    sc = np.linalg.norm(Z[m], axis=1)
                    pos_of = np.full(d["n_rows"], -1); pos_of[d["k"][j]] = np.arange(len(j))
                    ov = d["overlaps"]; ov = ov[pos_of[ov[:, 0].astype(int)] >= 0].copy(); ov[:, 0] = pos_of[ov[:, 0].astype(int)]
                    fs.append({"score": sc, "ijk": d["ijk"][j], "tumor_ml": T[m], "organ_ml": O[m], "overlaps": ov,
                               "lesions": {int(r[0]): {"ml": float(r[1]), "diameter_mm": float(r[2])} for r in d["lesions"]}})
                    a, b = tumor[m], clean[m]
                    pooled.append(sc[a | b]); lab.append(a[a | b])
                f = na.froc(fs)
                res[pname] = {"distance_from_donors: pooled_auc": round(na.auc(np.concatenate(pooled), np.concatenate(lab)), 3),
                              "distance_from_donors: lesions_at_1_fp": f["at"]["1.0"]["all"]}
                mm = np.isin(si, scans) & (tumor | clean)
                keep = np.concatenate([rng.choice(np.flatnonzero(mm & (si == s)), min(80, int((mm & (si == s)).sum())), replace=False) for s in scans])
                res[pname]["fitted_direction: pooled_auc (patient-disjoint)"] = round(na.auc(cv_scores(Z[keep], tumor[keep], pid[si[keep]]), tumor[keep]), 3)
            lg = (V - 0) @ text[fname].T / 1.0
            res["the finding's own score"] = {"pooled_auc": round(na.auc((lg[:, 1] - lg[:, 0])[np.isin(si, scans) & (tumor | clean)],
                                                                          tumor[np.isin(si, scans) & (tumor | clean)]), 3)}
            R["disease_by_part"][c] = res
            print(organ, c, "disease by part done", flush=True)

        # -- read-outs by part -------------------------------------------------------------------------
        keep = np.concatenate([rng.choice(np.flatnonzero(clean & (si == s)), min(60, int((clean & (si == s)).sum())), replace=False)
                               for s in np.unique(si[clean])])
        dk = keep[coll[si[keep]] == DONORS]
        R["read_out_by_part"] = {}
        for pname, cut in parts.items():
            Z = cut(V - dmu)
            row = {"place r2 (z, y, x), donors": cv_r2(Z[dk], C[dk], pid[si[dk]]),
                   "collection (chance 0.25)": cv_accuracy(Z[keep], coll[si[keep]], pid[si[keep]])}
            if organ == "kidney":
                row["side (chance 0.5)"] = cv_accuracy(Z[dk], (C[dk][:, 2] < 0.5).astype(int), pid[si[dk]])
            cm = keep[coll[si[keep]] == "colorectal_liver_metastases"]
            th = np.array([S[i]["th"] for i in si[cm]]); ok = np.isin(th, ["2.5", "5.0"])
            if ok.sum() > 600:
                row["slice thickness 2.5 vs 5 mm within one collection (chance 0.5)"] = cv_accuracy(Z[cm][ok], th[ok], pid[si[cm]][ok])
            R["read_out_by_part"][pname] = row

        # -- HCC against metastasis, beside the same patients' clean boxes ------------------------------
        if len(cohorts) == 2:
            a, b = list(cohorts)
            withles = np.array([len(d["lesions"]) > 0 for d in raw])[si]
            R["kinds_of_lesion"] = {"note": f"{a} against {b}: the two diseases are two sites, so the clean-box column is the site alone"}
            for pname, cut in parts.items():
                Z = cut(V - dmu); row = {}
                for label, sel in (("tumor boxes", tumor), ("clean boxes of the same patients", clean)):
                    m = sel & np.isin(coll[si], [a, b]) & withles
                    kk = np.concatenate([rng.choice(np.flatnonzero(m & (si == s)), min(40, int((m & (si == s)).sum())), replace=False)
                                         for s in np.unique(si[m])])
                    row[label] = cv_accuracy(Z[kk], coll[si[kk]], pid[si[kk]])
                R["kinds_of_lesion"][pname] = row

        # -- the handles: what is left when every named direction is removed ----------------------------
        Zc = (V - V.mean(0)) - (V - V.mean(0)) @ Pn
        wr, Ur = np.linalg.eigh(np.cov(Zc[rng.choice(len(Zc), min(len(Zc), 200000), replace=False)].T)); wr, Ur = wr[::-1], Ur[:, ::-1]
        A = Zc @ Ur[:, :12]
        known = {"z": C[:, 0], "y": C[:, 1], "x": C[:, 2], "fill": F, "tumor fraction": frac,
                 **{f"site={c}": (coll[si] == c).astype(float) for c in sorted(set(coll))},
                 **{f"phase={ph}": np.array([S[i]["phase"] == ph for i in si], float) for ph in ("arterial", "venous", "late")},
                 "slice_mm": np.array([float(S[i]["th"] or "nan") for i in si])}
        tot = float(np.trace(np.cov((V - V.mean(0)).T)))
        H = []
        for a_ in range(12):
            cors = {kn: abs(float(np.corrcoef(A[np.isfinite(kv), a_], kv[np.isfinite(kv)])[0, 1])) for kn, kv in known.items()}
            bestk = max(cors, key=cors.get)
            su_, inv_ = np.unique(si, return_inverse=True); mean_s = np.bincount(inv_, A[:, a_]) / np.bincount(inv_)
            Xk = np.c_[np.ones(len(A)), C, C ** 2, F, frac, *[known[k] for k in known if k.startswith(("site=", "phase="))]]
            ok = np.isfinite(Xk).all(1)
            fit = Xk[ok] @ np.linalg.lstsq(Xk[ok], A[ok, a_], rcond=None)[0]
            H.append({"axis": a_, "share_of_total_variance": round(float(wr[a_] / tot), 4),
                      "between_scan_share (a trait of the scan, not a pattern within it)": round(float((np.bincount(inv_) * mean_s ** 2).sum() / (A[:, a_] ** 2).sum()), 3),
                      "best_known_correlate": [bestk, round(cors[bestk], 2)],
                      "r2_from_everything_known": round(float(1 - ((A[ok, a_] - fit) ** 2).sum() / ((A[ok, a_] - A[ok, a_].mean()) ** 2).sum()), 3),
                      "tumor_vs_clean_auc": round(max(na.auc(A[tumor | clean, a_], tumor[tumor | clean]), 1 - na.auc(A[tumor | clean, a_], tumor[tumor | clean])), 3)})
        R["unnamed_axes"] = {"share_of_total_variance_outside_all_findings": round(float(wr.sum() / tot), 3), "axes": H}
        print(organ, "done", flush=True)
    pathlib.Path("/work/results/unnamed.json").write_text(json.dumps(out, indent=1)); work.commit()
    return json.dumps(out)


@app.function(image=image, volumes={"/work": work}, cpu=8, memory=65536, timeout=7200)
def fewshot(size: int = 32, seed: int = 0, draws: int = 10) -> str:
    """How many example patients does an EMPIRICAL lesion direction need to beat the text?

    `unnamed` found a finding's direction at cosine 0.31-0.41 with where real lesions go, while
    patients agree with each other (0.61-0.80). So: take n patients with an expert mask, average
    (tumor boxes - that patient's clean boxes), and use the result as the template on everybody
    else, scoring a box as ``(v - donor mean) . template`` - no label at test time. Plain, and
    whitened by the healthy donors' covariance on its leading axes (the Hotelling form; 5.6 says
    never the full precision). Beside it: the text direction (zero examples) and the template
    carried between the two liver diseases."""
    import numpy as np
    from feldglas.adapters import radar
    from feldglas.suite import normal_atlas as na
    rng = np.random.default_rng(seed)
    with np.load("/work/text_en.npz") as z:
        text = {k: z[k].astype(np.float64) for k in z.files}
    raw, coll = [], []
    for p in sorted(pathlib.Path("/work/regions").glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            k = np.flatnonzero((z["size"] == size) & (z["rule"] == 0))
            raw.append({"k": k, "organ": z["organ"][k], "vec": z["vec"][k].astype(np.float32), "tumor_ml": z["tumor_ml"][k],
                        "organ_ml": z["organ_ml"][k], "ijk": z["ijk"][k], "overlaps": z["overlaps"], "lesions": z["lesions"], "n_rows": len(z["organ"])})
        coll.append(meta["coll"])
    coll = np.array(coll)
    out = {}
    templates = {}
    for organ, cohorts in COHORTS.items():
        val = radar.ORGANS.index(organ) + 1
        D = np.concatenate([d["vec"][d["organ"] == val] for d, c in zip(raw, coll) if c == DONORS]).astype(np.float64)
        dmu = D.mean(0); w, U = np.linalg.eigh(np.cov((D - dmu).T)); w, U = np.maximum(w[::-1], 1e-12), U[:, ::-1]
        white = lambda t, k=32: U[:, :k] @ ((U[:, :k].T @ t) / w[:k])                # S^-1 d on the donors' k loudest axes
        for c, fname in cohorts.items():
            P = []
            for d, cc in zip(raw, coll):
                j = np.flatnonzero(d["organ"] == val)
                if cc != c or not len(d["lesions"]) or len(j) < 6:
                    continue
                V = d["vec"][j].astype(np.float64); tum = d["tumor_ml"][j].astype(np.float64); oml = d["organ_ml"][j].astype(np.float64)
                A, cl = tum >= 0.25 * np.maximum(oml, 1e-6), tum <= 0
                pos_of = np.full(d["n_rows"], -1); pos_of[d["k"][j]] = np.arange(len(j))
                ov = d["overlaps"]; ov = ov[pos_of[ov[:, 0].astype(int)] >= 0].copy(); ov[:, 0] = pos_of[ov[:, 0].astype(int)]
                P.append({"V": V - dmu, "A": A, "cl": cl, "disp": V[A].mean(0) - V[cl].mean(0) if A.sum() and cl.sum() >= 3 else None,
                          "f": {"ijk": d["ijk"][j], "tumor_ml": tum, "organ_ml": oml, "overlaps": ov,
                                "lesions": {int(r[0]): {"ml": float(r[1]), "diameter_mm": float(r[2])} for r in d["lesions"]}}})
            usable = [i for i, x in enumerate(P) if x["disp"] is not None]

            def evaluate(t, test):
                fs, sc_, lab = [], [], []
                for i in test:
                    x = P[i]; sc = x["V"] @ t
                    fs.append({**x["f"], "score": sc}); m = x["A"] | x["cl"]
                    sc_.append(sc[m]); lab.append(x["A"][m])
                return na.auc(np.concatenate(sc_), np.concatenate(lab)), na.froc(fs)["at"]["1.0"]["all"]

            everyone = list(range(len(P)))
            tdir = text[fname][1] - text[fname][0]
            R = {"patients": len(P), "with_a_usable_displacement": len(usable),
                 "text direction (0 examples)": dict(zip(("pooled_auc", "lesions_at_1_fp"), [round(float(v), 3) for v in evaluate(tdir, everyone)])),
                 "text direction, whitened": dict(zip(("pooled_auc", "lesions_at_1_fp"), [round(float(v), 3) for v in evaluate(white(tdir), everyone)])),
                 "examples": {}}
            for n in (1, 2, 3, 5, 10, 20, 40):
                if n > len(usable) - 20:
                    continue
                rows = {"plain": [], "whitened": []}
                for _ in range(draws):
                    tr = rng.choice(usable, n, replace=False); te = [i for i in everyone if i not in set(tr)]
                    t = np.mean([P[i]["disp"] for i in tr], 0)
                    rows["plain"].append(evaluate(t, te)); rows["whitened"].append(evaluate(white(t), te))
                R["examples"][str(n)] = {k: {"pooled_auc_mean_min": [round(float(np.mean([a for a, _ in v])), 3), round(float(np.min([a for a, _ in v])), 3)],
                                             "lesions_at_1_fp_mean_min": [round(float(np.mean([b for _, b in v])), 3), round(float(np.min([b for _, b in v])), 3)]}
                                         for k, v in rows.items()}
            templates[c] = (np.mean([P[i]["disp"] for i in usable], 0), P, everyone, white)
            out[c] = R
            print(c, "done", flush=True)
    a, b = "colorectal_liver_metastases", "hcc_tace_seg"
    for src, dst in ((a, b), (b, a)):
        t, _, _, white = templates[src]; _, P, everyone, _ = templates[dst]
        def evaluate(t_):
            fs, sc_, lab = [], [], []
            for i in everyone:
                x = P[i]; sc = x["V"] @ t_
                fs.append({**x["f"], "score": sc}); m = x["A"] | x["cl"]; sc_.append(sc[m]); lab.append(x["A"][m])
            return [round(float(na.auc(np.concatenate(sc_), np.concatenate(lab))), 3), na.froc(fs)["at"]["1.0"]["all"]]
        out[dst][f"template from ALL {src} patients (another disease, another site): auc, lesions_at_1_fp"] = {"plain": evaluate(t), "whitened": evaluate(white(t))}
    ta, tb = templates[a][0], templates[b][0]
    out["cosine between the metastasis and the HCC template"] = round(float(ta @ tb / np.linalg.norm(ta) / np.linalg.norm(tb)), 3)
    pathlib.Path("/work/results/fewshot.json").write_text(json.dumps(out, indent=1)); work.commit()
    return json.dumps(out)


@app.local_entrypoint()
def shots(size: int = 32):
    r = json.loads(fewshot.remote(size))
    p = MEDSEG / "results" / "validation" / "fewshot.json"
    p.write_text(json.dumps(r, indent=1)); print(f"-> {p}")


@app.local_entrypoint()
def main(size: int = 32):
    series = {r["crdc_series_uuid"]: {"se": r["se"], "th": r["th"]}
              for r in csv.DictReader(open(MEDSEG / "results" / "idc" / "radar_validation_series_all.csv"))}
    r = json.loads(unnamed.remote(series, size))
    p = MEDSEG / "results" / "validation" / "unnamed.json"
    p.write_text(json.dumps(r, indent=1)); print(f"-> {p}")
