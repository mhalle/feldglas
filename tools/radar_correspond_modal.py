"""Can RAW tokens carry a place from one patient to another? Tested against INDEPENDENT landmarks.

`radar_space_modal.py` found that a raw token reads its structure at 0.77-0.82 and lands, by
nearest neighbor, ~5 cm from itself in another patient - but against RADAR's OWN mask, which the
same encoder produced. Here the truth is haversack's: `ts.v2:total` label maps of the same series,
left by the validation study on the Modal volume `haversack-radar-val-cache` (read-only here, and
read with the standard library - a `.seg.nrrd` is a text header and a gzip block). Their centroids
are the landmarks: single vertebrae, kidneys and adrenals by side, ribs by number, gallbladder,
spleen, pancreas ... Structures cut by the field of view, and tubes whose centroid is wherever the
scan stops, are left out.

  ::main        index the cache, write `landmarks/<series>.json` for every series with a label map,
                copy a balanced subset of fields to the work volume (`dense/`), then
  match         every query scan against every reference scan: carry each landmark across by
                  translation / affine   from the OTHER shared landmarks only (leave-one-out) - what
                                         segmentation alone gives, no embedding
                  deep / mid / fine      the token holding the landmark in A -> its most similar token
                                         anywhere in B (cosine; raw, and standardized within each scan)
                  coarse-to-fine         deep anywhere, mid within 60 mm of that, fine within 30 mm
                  affine + mid / fine    BOTH: the most similar token within 40 / 25 mm of the affine guess
                  floor                  the nearest token center to B's landmark - what the lattice allows
                error = distance to B's own landmark, in mm
  ::bodycoord   a body coordinate read off one sampled token: vertebral level (T8 = 8 ... L1 = 13 ...
                S1 = 18, continuous) and mm left and posterior of the spine at that level

Run:  FELDGLAS_STORE=s3://<bucket>/feldglas uv run --no-sync modal run tools/radar_correspond_modal.py::main
      uv run --no-sync modal run tools/radar_correspond_modal.py::bodycoord
"""
import csv, json, os, pathlib, time

import modal

HERE = pathlib.Path(__file__).resolve().parent


def _up(n):
    return HERE.parents[n] if len(HERE.parents) > n else HERE


MEDSEG = _up(1) / "medseg" / "docs" / "radar-idc-validation"
_SECRET = os.environ.get("FELDGLAS_MODAL_SECRET", "feldglas-r2")
# the SAME image definition as radar_atlas_modal.py, so Modal reuses the one it built
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("numpy>=1.24", "scipy", "obstore>=0.11", "idc-index", "highdicom>=0.23", "pydicom>=3",
                      "pylibjpeg", "pylibjpeg-libjpeg",
                      "rankfield @ git+https://github.com/mhalle/rankfield.git@v0.3.2",
                      "provender @ git+https://github.com/mhalle/provender.git@v0.1.1")
         .env({"FELDGLAS_STORE": os.environ.get("FELDGLAS_STORE", "")})
         .add_local_python_source("feldglas"))
work = modal.Volume.from_name("feldglas-radar-work", create_if_missing=True)
hv = modal.Volume.from_name("haversack-radar-val-cache").read_only()       # another project's results: never written
app = modal.App("feldglas-radar-correspond")

LEVELS = {**{f"vertebrae_T{i}": i for i in range(1, 13)}, **{f"vertebrae_L{i}": 12 + i for i in range(1, 6)}, "vertebrae_S1": 18}
TUBES = ("aorta", "inferior_vena_cava", "colon", "small_bowel", "spinal_cord", "esophagus", "costal_cartilages",
         "autochthon", "iliopsoas", "portal_vein", "sternum", "iliac_", "gluteus", "trachea", "pulmonary", "duodenum")


def _read_seg_nrrd(path):
    """``(labels[i, j, k], affine_lps 4x4, {name: value})`` of a single-layer .seg.nrrd."""
    import gzip
    import numpy as np
    raw = pathlib.Path(path).read_bytes()
    cut = raw.index(b"\n\n")
    head = {}
    for line in raw[:cut].decode("utf8", "replace").split("\n")[1:]:
        if line.startswith("#"):
            continue
        for sep in (":=", ": "):
            if sep in line:
                a, b = line.split(sep, 1); head[a.strip()] = b.strip(); break
    if head.get("dimension") != "3" or head.get("space") != "left-posterior-superior" or head.get("encoding") not in ("gzip", "gz"):
        raise ValueError(f"not a single-layer LPS gzip seg.nrrd: {head.get('dimension')}, {head.get('space')}, {head.get('encoding')}")
    sizes = [int(x) for x in head["sizes"].split()]
    dt = {"unsigned char": np.uint8, "uint8": np.uint8, "uchar": np.uint8, "short": np.int16, "unsigned short": np.uint16}[head["type"]]
    arr = np.frombuffer(gzip.decompress(raw[cut + 2:]), dt).reshape(sizes, order="F")
    vec = lambda s: [float(x) for x in s.strip("()").split(",")]
    A = np.eye(4); A[:3, :3] = np.array([vec(v) for v in head["space directions"].split()]).T; A[:3, 3] = vec(head["space origin"])
    names = {head[k]: int(head[k.replace("_Name", "_LabelValue")]) for k in head if k.endswith("_Name") and k.startswith("Segment")}
    return arr, A, names


@app.function(image=image, volumes={"/hv": hv}, timeout=3600)
def index_cache() -> str:
    """``{series: {task: path of labels.seg.nrrd}}`` for every entry whose input is one IDC series."""
    out = {}
    for key in pathlib.Path("/hv").iterdir():
        cur = key / "current"
        if not cur.is_file():
            continue
        g = key / f"g-{cur.read_text().strip()}"
        try:
            meta = json.loads((g / "meta.json").read_text())
        except (OSError, ValueError):
            continue
        ident = meta.get("identity") or []
        if len(ident) == 1 and ident[0].startswith("idc:") and (g / "labels.seg.nrrd").is_file():
            out.setdefault(ident[0][4:], {})[meta.get("task", "")] = str(g / "labels.seg.nrrd")
    return json.dumps(out)


@app.function(image=image, secrets=[modal.Secret.from_name(_SECRET)], volumes={"/work": work, "/hv": hv}, cpu=2,
              memory=8192, timeout=1200, max_containers=40)
def prepare(u: str, digest: str, label_path: str, task: str, pid: str, coll: str, phase: str, dense: bool) -> str:
    import numpy as np
    from scipy import ndimage
    try:
        lm_path = pathlib.Path(f"/work/landmarks/{u}.json")
        if not lm_path.exists():
            arr, A, names = _read_seg_nrrd(label_path)
            idx = np.arange(1, int(arr.max()) + 1)
            n = ndimage.sum_labels(np.ones(arr.shape, np.uint8), arr, idx)
            com = np.asarray(ndimage.center_of_mass(np.ones(arr.shape, np.uint8), arr, idx))
            sl = ndimage.find_objects(arr, int(arr.max()))
            L = {}
            for name, v in names.items():
                if v < 1 or v > len(idx) or n[v - 1] < 30 or sl[v - 1] is None:
                    continue
                cut = any(s.start == 0 or s.stop == dim for s, dim in zip(sl[v - 1], arr.shape))
                L[name] = {"lps": [float(x) for x in A[:3, :3] @ com[v - 1] + A[:3, 3]], "voxels": int(n[v - 1]), "cut": bool(cut)}
            lm_path.parent.mkdir(parents=True, exist_ok=True)
            lm_path.write_text(json.dumps({"u": u, "pid": pid, "coll": coll, "phase": phase, "task": task, "landmarks": L}))
        if dense and not pathlib.Path(f"/work/dense/{u}.npz").exists():
            from feldglas.adapters import radar
            from feldglas.remote import open_blobs
            pathlib.Path("/work/dense").mkdir(parents=True, exist_ok=True)
            tmp = pathlib.Path(f"/work/dense/{u}.tmp")
            if not open_blobs(radar.NAME, check=False).fetch(digest, tmp):
                raise FileNotFoundError(f"blob {digest} is gone, or did not match its name")
            tmp.replace(f"/work/dense/{u}.npz")
        work.commit()
        return json.dumps({"u": u, "ok": True})
    except Exception as e:
        import traceback
        return json.dumps({"u": u, "err": f"{type(e).__name__}: {e}"[:300], "trace": traceback.format_exc()[-800:]})


def _usable(L):
    return {k: v["lps"] for k, v in L.items() if not v["cut"] and not any(t in k for t in TUBES)}


@app.function(image=image, volumes={"/work": work}, cpu=4, memory=16384, timeout=3600, max_containers=40)
def match(q: str, refs: list) -> bytes:
    """One query scan against every reference: per shared landmark, the error in mm of each way
    of carrying it across. Returns an ``.npz`` of rows."""
    import io
    import numpy as np
    from feldglas.store import read_field

    def load(u):
        f = read_field(f"/work/dense/{u}.npz")
        inv = np.linalg.inv(np.asarray(f.grid.directions, np.float64)); org = np.asarray(f.grid.origin, np.float64)
        lat = []
        for j, k in enumerate(f.kernels):
            T = np.asarray(f.tokens[j], np.float32)
            Z = (T - T.mean(0)) / (T.std(0) + 1e-6)                                   # standardized WITHIN the scan
            lat.append({"raw": T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9),
                        "std": Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9),
                        "centers": f.token_centers(j).astype(np.float32), "shape": f.lattice_shape(j), "kernel": np.asarray(k)})
        meta = json.loads(pathlib.Path(f"/work/landmarks/{u}.json").read_text())
        return {"lat": lat, "inv": inv, "org": org, "lm": _usable(meta["landmarks"]), "coll": meta["coll"], "pid": meta["pid"]}

    def token_at(S, j, x):
        idx = np.floor(((np.asarray(x) - S["org"]) @ S["inv"] + 0.5) / S["lat"][j]["kernel"]).astype(int)
        if np.any(idx < 0) or np.any(idx >= np.asarray(S["lat"][j]["shape"])):
            return None
        return int(np.ravel_multi_index(idx, S["lat"][j]["shape"]))

    A = load(q)
    names_all = sorted(A["lm"])
    rows = []
    for r in refs:
        if r == q:
            continue
        B = load(r)
        if B["pid"] == A["pid"]:
            continue
        shared = [n for n in names_all if n in B["lm"]]
        if len(shared) < 8:
            continue
        XA = np.array([A["lm"][n] for n in shared]); XB = np.array([B["lm"][n] for n in shared])
        for i, n in enumerate(shared):
            o = np.arange(len(shared)) != i
            e = {"translation": np.linalg.norm(XA[i] + (XB[o] - XA[o]).mean(0) - XB[i])}
            H = np.c_[XA[o], np.ones(o.sum())]                                        # affine from the OTHER landmarks
            M = np.linalg.solve(H.T @ H + 1e-3 * np.eye(4), H.T @ XB[o])
            guess = np.r_[XA[i], 1.0] @ M
            e["affine"] = np.linalg.norm(guess - XB[i])
            prev = None
            for j, lname in ((0, "deep"), (1, "mid"), (2, "fine")):
                t = token_at(A, j, XA[i])
                if t is None:
                    e[f"{lname}_raw"] = e[f"{lname}_std"] = e[f"c2f_{lname}"] = e[f"floor_{lname}"] = np.nan
                    if j:
                        e[f"affine+{lname}"] = np.nan
                    continue
                cen = B["lat"][j]["centers"]
                for kind in ("raw", "std"):
                    sim = B["lat"][j][kind] @ A["lat"][j][kind][t]
                    e[f"{lname}_{kind}"] = np.linalg.norm(cen[sim.argmax()] - XB[i])
                sim = B["lat"][j]["std"] @ A["lat"][j]["std"][t]
                e[f"floor_{lname}"] = np.linalg.norm(cen - XB[i], axis=1).min()            # the best ANY token of this lattice could do
                if j:                                                                  # BOTH: geometry says which one, the embedding where
                    near = np.linalg.norm(cen - guess, axis=1) <= (40.0 if j == 1 else 25.0)
                    e[f"affine+{lname}"] = np.linalg.norm(cen[np.where(near, sim, -2).argmax()] - XB[i]) if near.any() else np.nan
                if prev is not None:
                    sim = np.where(np.linalg.norm(cen - prev, axis=1) <= (60.0 if j == 1 else 30.0), sim, -2)
                prev = cen[sim.argmax()]
                e[f"c2f_{lname}"] = np.linalg.norm(prev - XB[i])
            rows.append((n, A["coll"] == B["coll"], len(shared), e))
    keys = sorted(rows[0][3]) if rows else []
    buf = io.BytesIO()
    np.savez_compressed(buf, landmark=np.array([r[0] for r in rows]), same_site=np.array([r[1] for r in rows]),
                        shared=np.array([r[2] for r in rows]), methods=np.array(keys),
                        err=np.array([[r[3][k] for k in keys] for r in rows], np.float32).reshape(len(rows), len(keys)))
    return buf.getvalue()


@app.function(image=image, volumes={"/work": work}, cpu=8, memory=65536, timeout=7200)
def body_coordinate(seed: int = 0) -> str:
    import numpy as np
    rng = np.random.default_rng(seed)
    lat, tgt, vec, grp, col = [], [], [], [], []
    n_scans = 0
    for p in sorted(pathlib.Path("/work/tokens").glob("*.npz")):
        lm = pathlib.Path(f"/work/landmarks/{p.stem}.json")
        if p.name.endswith(".tmp.npz") or not lm.exists():
            continue
        meta = json.loads(lm.read_text())
        V = sorted((LEVELS[k], v["lps"]) for k, v in meta["landmarks"].items() if k in LEVELS and not v["cut"])
        if len(V) < 4:
            continue
        lev = np.array([a for a, _ in V], float); P = np.array([b for _, b in V])
        o = np.argsort(P[:, 2]); lev, P = lev[o], P[o]
        if np.any(np.diff(P[:, 2]) <= 0):
            continue
        with np.load(p) as z:
            W, X, Lt = z["world"].astype(np.float64), z["vec"], z["lattice"]
        inside = (W[:, 2] >= P[0, 2]) & (W[:, 2] <= P[-1, 2])                         # never extrapolate the spine
        s = np.interp(W[:, 2], P[:, 2], lev)
        dl = W[:, 0] - np.interp(W[:, 2], P[:, 2], P[:, 0]); dp = W[:, 1] - np.interp(W[:, 2], P[:, 2], P[:, 1])
        lat.append(Lt[inside]); tgt.append(np.c_[s, dl, dp][inside]); vec.append(X[inside].astype(np.float32))
        grp += [meta["pid"]] * int(inside.sum()); col += [meta["coll"]] * int(inside.sum()); n_scans += 1
    lat, tgt, vec, grp, col = np.concatenate(lat), np.concatenate(tgt), np.concatenate(vec), np.array(grp), np.array(col)
    print(f"{n_scans} scans, {len(lat)} tokens inside their spine's range", flush=True)

    def ridge(Xa, Ya, Xb, lam=1e-3):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
        A = (Xa - mu) / sd
        W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Ya - Ya.mean(0)))
        return ((Xb - mu) / sd) @ W + Ya.mean(0)

    out = {"scans": n_scans, "targets": ["vertebral level", "mm left of the spine", "mm posterior of the spine"], "lattices": {}}
    for j, lname in enumerate(("deep", "mid", "fine")):
        m = np.flatnonzero(lat == j)
        if len(m) > 150000:
            m = rng.choice(m, 150000, replace=False)
        X, Y, g, c = vec[m].astype(np.float64), tgt[m], grp[m], col[m]
        u = rng.permutation(np.unique(g)); P = np.empty_like(Y); K = np.empty_like(Y)
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-6); Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-9
        for f in np.array_split(u, 5):
            te = np.isin(g, f)
            P[te] = ridge(X[~te], Y[~te], X[te])
            tr = np.flatnonzero(~te); tr = rng.choice(tr, min(40000, len(tr)), replace=False)
            for a in np.array_split(np.flatnonzero(te), max(1, te.sum() // 4000)):
                nn = np.argpartition(-(Xn[a] @ Xn[tr].T), 10, axis=1)[:, :10]
                K[a] = np.median(Y[tr][nn], axis=1)                                   # 10 most similar tokens of OTHER patients
        don = c == "pancreas_ct"
        T = ridge(X[don], Y[don], X[~don])
        mae = lambda a, b: [round(float(np.median(np.abs(a[:, i] - b[:, i]))), 2) for i in range(3)]
        out["lattices"][lname] = {"tokens": int(len(m)),
                                  "median_abs_error": {"guess_the_median": mae(np.tile(np.median(Y, 0), (len(Y), 1)), Y), "linear": mae(P, Y),
                                                       "10_nearest_tokens_of_other_patients": mae(K, Y),
                                                       "linear, donors -> other sites": mae(T, Y[~don])},
                                  "r2_linear": [round(float(1 - ((Y[:, i] - P[:, i]) ** 2).sum() / ((Y[:, i] - Y[:, i].mean()) ** 2).sum()), 3) for i in range(3)]}
        print(lname, "done", flush=True)
    pathlib.Path("/work/results/body_coordinate.json").write_text(json.dumps(out, indent=1)); work.commit()
    return json.dumps(out)


@app.local_entrypoint()
def main(queries_per_site: int = 30, refs_per_site: int = 10, skip_prepare: bool = False):
    import io
    import numpy as np
    from feldglas.remote import Manifest
    if not os.environ.get("FELDGLAS_STORE", "").startswith(("s3://", "gs://", "az://")):
        raise SystemExit("set FELDGLAS_STORE (e.g. s3://<bucket>/feldglas): the fields are read from it")
    mf = Manifest.load(HERE.parent / "manifests" / "radar.json")
    cache = json.loads(index_cache.remote())
    rows = [r for r in csv.DictReader(open(MEDSEG / "results" / "validation" / "choice.csv"))
            if r["chosen"] and f"{r['chosen']}.npz" in mf.files and r["chosen"] in cache]
    print(f"{len(cache)} series in haversack's cache; {len(rows)} of the {len(mf.files)} stored fields have a label map; tasks: "
          + str({t: sum(t in cache[r['chosen']] for r in rows) for t in sorted({t for r in rows for t in cache[r['chosen']]})}))
    by = {}
    for r in rows:
        by.setdefault(r["coll"], []).append(r["chosen"])
    dense = {u for v in by.values() for u in v[:queries_per_site]}
    refs = [u for v in by.values() for u in v[:refs_per_site]]
    pick = lambda u: next((t, cache[u][t]) for t in ("ts.v2:total", "ts.v2:total_fast", *cache[u]) if t in cache[u])
    if not skip_prepare:
        jobs = [(r["chosen"], mf.files[f"{r['chosen']}.npz"]["digest"], pick(r["chosen"])[1], pick(r["chosen"])[0],
                 r["pid"], r["coll"], r["phase"], r["chosen"] in dense) for r in rows]
        t = time.time(); bad = 0
        for x in prepare.starmap(jobs, return_exceptions=True):
            x = json.loads(x) if isinstance(x, str) else {"err": repr(x)[:300]}
            if x.get("err"):
                bad += 1; print("  ", x.get("u"), x["err"])
        print(f"prepared {len(jobs) - bad}/{len(jobs)} in {time.time() - t:.0f} s; {len(dense)} dense, {len(refs)} references")
    t = time.time(); parts = []
    for x in match.starmap([(q, refs) for q in sorted(dense)], return_exceptions=True):
        if isinstance(x, (bytes, bytearray)):
            z = np.load(io.BytesIO(x))
            if len(z["landmark"]):
                parts.append({k: z[k] for k in z.files})
        else:
            print("  match failed:", repr(x)[:300])
    print(f"matched {len(parts)} queries in {time.time() - t:.0f} s")
    methods = [str(m) for m in parts[0]["methods"]]
    lm = np.concatenate([p["landmark"] for p in parts]); same = np.concatenate([p["same_site"] for p in parts])
    err = np.concatenate([p["err"] for p in parts])
    med = lambda m: [None if not m.any() else round(float(np.nanmedian(err[m, i])), 1) for i in range(len(methods))]
    groups = {"vertebrae": "vertebrae_", "ribs": "rib_", "kidneys": "kidney_", "adrenals": "adrenal_", "spleen": "spleen",
              "gallbladder": "gallbladder", "pancreas": "pancreas", "liver": "liver", "stomach": "stomach", "hips and femurs": ("hip_", "femur_", "sacrum"),
              "bladder": "urinary_bladder", "heart and lungs": ("heart", "lung_")}
    out = {"pairs_rows": int(len(lm)), "queries": len(parts), "references": len(refs), "methods": methods,
           "median_error_mm": {"all": med(np.ones(len(lm), bool)), "same site": med(same), "other site": med(~same)},
           "within_20mm": [round(float(np.nanmean(err[:, i] <= 20)), 3) for i in range(len(methods))],
           "within_40mm": [round(float(np.nanmean(err[:, i] <= 40)), 3) for i in range(len(methods))],
           "by_landmark_group": {g: {"n": int(np.char.startswith(lm, pre).sum() if isinstance(pre, str) else sum(np.char.startswith(lm, x).sum() for x in pre)),
                                     "median_error_mm": med(np.char.startswith(lm, pre) if isinstance(pre, str) else np.any([np.char.startswith(lm, x) for x in pre], 0))}
                                 for g, pre in groups.items()}}
    p = MEDSEG / "results" / "validation" / "correspondence.json"
    p.write_text(json.dumps(out, indent=1)); print(f"-> {p}")
    print(f"{'':18}" + " ".join(f"{m[:11]:>11}" for m in methods))
    for k, v in out["median_error_mm"].items():
        print(f"{k:18}" + " ".join(f"{(x if x is not None else float('nan')):11.1f}" for x in v))
    for g, v in out["by_landmark_group"].items():
        print(f"{g[:18]:18}" + " ".join(f"{(x if x is not None else float('nan')):11.1f}" for x in v["median_error_mm"]) + f"   n={v['n']}")


@app.local_entrypoint()
def bodycoord():
    r = json.loads(body_coordinate.remote())
    p = MEDSEG / "results" / "validation" / "body_coordinate.json"
    p.write_text(json.dumps(r, indent=1)); print(f"-> {p}")
    for l, R in r["lattices"].items():
        print(l, R["tokens"], "tokens; r2", R["r2_linear"])
        for k, v in R["median_abs_error"].items():
            print(f"   {k:40} level {v[0]:.2f}   left {v[1]:.1f} mm   posterior {v[2]:.1f} mm")
