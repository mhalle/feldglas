"""Three normal scans, one unknown: is a region of the unknown unlike normal tissue? A demonstration
with a PAINTED lesion, all local (2026-09-22).

The question a user asked: with a region of normal tissue drawn in one organ of three normal
subjects, can regions of a fourth, unknown scan be tested against it? This does it end to end on
public full-dose CTs of one study (``ldct_and_projection_data``, one protocol) already in the cache:

  1. paint a lesion into the unknown: a sphere of ``--diameter`` mm whose HU drop by ``--contrast``
     (the scan's own texture and noise kept), deep in the liver;
  2. encode all five CTs (three normals, the unknown, the unknown painted) with RADAR on MPS;
  3. the ROI: each liver (haversack's ``ts.v2:total_fast`` labels of the unpainted scans) eroded
     by ``--margin`` mm - "normal tissue", interior, so every box compared is interior too;
  4. sweep the ROI with ``--box`` mm boxes at half-box steps, pool each on the tokens touching the
     ROI inside it - with RADAR's head (liver query) and without it (mean per lattice);
  5. the null: build the normal (mean vector) from two normals, score every box of the third,
     rotate - the distances normal tissue at this site gives when it was not in the reference;
  6. score the unknown, painted and not, against the normal of all three. A box is flagged beyond
     the largest held-out normal distance (no false alarm on the normals).

Plain distance, not whitened: the reference is other people (EXPLORATION 5.6). Vectors derive
from RADAR's CC BY-NC-SA weights and stay in ``--work``; only numbers and a figure come out.

Run (haversack's environment: torch, nibabel, SimpleITK):
    PYTHONPATH=src ../haversack/.venv/bin/python tools/roi_atlas_demo.py --work DIR
"""
import argparse, json, pathlib, subprocess, sys, time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CT = pathlib.Path.home() / ".cache" / "radar-idc-validation" / "ldct"
LABELS = pathlib.Path.home() / ".cache" / "radar-idc-validation" / "antialias" / "labels"
NORMALS = ("996adc64-84ac-42dc-b485-4622e123b699", "b769e358-005f-4643-b17b-a85c4aa271ec",
           "ba1e8e10-6218-40ca-911c-9827aa22e073")
UNKNOWN = "159dff32-8c60-4e38-964b-d78dd423ee15"


def liver_mask_on_ct(u):
    """haversack's liver on the CT's own voxel grid, as nibabel indexes it (i, j, k)."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(LABELS / f"{u}.seg.nrrd"))
    value = next(int(img.GetMetaData(k.replace("_Name", "_LabelValue"))) for k in img.GetMetaDataKeys()
                 if k.endswith("_Name") and img.GetMetaData(k) == "liver")
    return sitk.GetArrayFromImage(img).transpose(2, 1, 0) == value


def paint(u, out, diameter, contrast):
    """The unknown with a sphere of ``diameter`` mm whose HU drop by ``contrast``, centered on the
    liver voxel farthest from the liver's surface; the edge ramps over one voxel. Returns its LPS
    center (mm) and depth."""
    import nibabel as nib
    from scipy.ndimage import distance_transform_edt
    ct = nib.load(str(CT / f"{u}.nii.gz"))
    zooms = np.asarray(ct.header.get_zooms()[:3], float)
    liver = liver_mask_on_ct(u)
    depth = distance_transform_edt(liver, sampling=zooms)
    c = np.array(np.unravel_index(np.argmax(depth), depth.shape))
    a = np.asarray(ct.dataobj, np.float32)
    h = np.ceil((diameter / 2 + 2) / zooms).astype(int)                 # a cube around the sphere only
    lo, hi = np.maximum(c - h, 0), np.minimum(c + h + 1, a.shape)
    ijk = np.stack(np.meshgrid(*[np.arange(l, u) for l, u in zip(lo, hi)], indexing="ij"), -1)
    r = np.linalg.norm((ijk - c) * zooms, axis=-1)
    w = np.clip((diameter / 2 + zooms.min() / 2 - r) / zooms.min(), 0, 1)
    a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] -= contrast * w
    img = nib.Nifti1Image(a, ct.affine)                  # float32, no scaling: only the sphere changes
    img.set_qform(ct.affine, code=1); img.set_sform(ct.affine, code=1)
    nib.save(img, str(out))
    ras = ct.affine[:3, :3] @ c + ct.affine[:3, 3]
    return (ras * [-1, -1, 1]).tolist(), float(depth[tuple(c)]), int((w > 0.5).sum())


def encode(ct_path, series, out):
    if not out.exists():
        subprocess.run([sys.executable, str(HERE / "radar_encode_local.py"), str(ct_path), "--series", series,
                        "--device", "mps", "--fp16", "--out", str(out)], check=True, capture_output=True)
    from feldglas.store import read_field
    return read_field(out)


def boxes(field, u, margin, size):
    """The ROI (liver eroded by ``margin`` mm) on the model grid, and its sweep."""
    from scipy.ndimage import distance_transform_edt
    from feldglas.labels import read_seg_nrrd
    from feldglas.suite import normal_atlas as na
    liver = read_seg_nrrd(LABELS / f"{u}.seg.nrrd").on_grid(field.grid).mask("liver")
    roi = distance_transform_edt(liver, sampling=field.grid.spacing) > margin
    return roi, na.sweep(field, roi, size)


def roi_tokens(field, roi, lattice=-1, min_occupancy=0.5):
    """The lattice's tokens at least ``min_occupancy`` inside the ROI, unit length, with their
    centers (LPS mm) - the head-free unit of comparison: a box's mean dilutes a local lesion
    (a 32 mm box moved 0.06-0.17 for spheres whose own fine tokens changed by up to 111 %)."""
    from feldglas.gate import occupancies
    j = lattice % field.lattices
    keep = occupancies(field, roi)[j] >= min_occupancy
    t = field.tokens[j].astype(np.float32)[keep]
    return t / np.linalg.norm(t, axis=1, keepdims=True), field.token_centers(j)[keep]


def knn_distance(x, ref, k=5):
    """1 - mean cosine to the ``k`` most similar normal tokens: normal tissue is many kinds of
    token (parenchyma, vessels, fissures), and a mean is none of them."""
    s = x @ ref.T
    return 1 - np.sort(s, 1)[:, -k:].mean(1)


def sites(centers, flagged, radius=15.0):
    """Flagged tokens grouped into sites: tokens within ``radius`` mm are one site."""
    idx = np.flatnonzero(flagged)
    label = -np.ones(len(idx), int); n = 0
    for a in range(len(idx)):
        if label[a] >= 0:
            continue
        stack = [a]; label[a] = n
        while stack:
            b = stack.pop()
            near = np.flatnonzero((label < 0) & (np.linalg.norm(centers[idx] - centers[idx[b]], axis=1) <= radius))
            label[near] = n; stack.extend(near.tolist())
        n += 1
    return [centers[idx[label == i]].mean(0) for i in range(n)]


def pooled(field, roi, sw, heads):
    from feldglas.suite import normal_atlas as na
    out = {}
    for name, (head, blocks, query) in heads.items():
        prepared = head.prepare(field.all_tokens(blocks=blocks))
        out[name] = na.pool_sweep(field, head, prepared, sw, roi, query)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, help="where fields and the painted CT go (license-bound: not a repository)")
    ap.add_argument("--diameter", type=float, default=15.0); ap.add_argument("--contrast", type=float, default=40.0)
    ap.add_argument("--margin", type=float, default=15.0); ap.add_argument("--box", type=float, default=32.0)
    ap.add_argument("--figure", default="")
    a = ap.parse_args()
    from feldglas.adapters import radar
    from feldglas.heads import LatticeMeanHead
    work = pathlib.Path(a.work); work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    painted = work / f"{UNKNOWN}.painted-{a.diameter:g}mm-{a.contrast:g}HU.nii.gz"
    center, depth, voxels = paint(UNKNOWN, painted, a.diameter, a.contrast)
    print(f"painted a {a.diameter:g} mm sphere, -{a.contrast:g} HU, {voxels} voxels, at LPS {np.round(center, 1)} "
          f"({depth:.0f} mm inside the liver)")
    scans = {u: encode(CT / f"{u}.nii.gz", u, work / f"{u}.zarr.zip") for u in (*NORMALS, UNKNOWN)}
    scans["painted"] = encode(painted, UNKNOWN, work / painted.name.replace(".nii.gz", ".zarr.zip"))
    print(f"five fields in {time.time() - t0:.0f} s")

    rh = radar.RadarHead.load()
    heads = {"RADAR head": (rh, False, "liver")}
    V, S, ROI = {}, {}, {}
    for key, f in scans.items():
        u = UNKNOWN if key == "painted" else key
        heads["no head (mean per lattice)"] = (LatticeMeanHead([f.widths[0]] * f.lattices), True, None)
        ROI[key], S[key] = boxes(f, u, a.margin, a.box)
        V[key] = pooled(f, ROI[key], S[key], heads)
    same = np.array_equal(S["painted"].center, S[UNKNOWN].center)
    f = scans["painted"]
    lesion = np.linalg.solve(np.asarray(f.grid.directions, float).T, np.asarray(center) - np.asarray(f.grid.origin))
    sp = np.asarray(f.grid.spacing)
    hit = np.all(np.abs(S["painted"].center - lesion) * sp <= a.box / 2, axis=1)   # boxes holding its center

    report = {"lesion": {"diameter_mm": a.diameter, "contrast_hu": -a.contrast, "center_lps": center, "depth_mm": depth},
              "boxes": {k: int(len(s.center)) for k, s in S.items()}, "lesion_boxes": int(hit.sum()), "heads": {}}
    print(f"boxes of {a.box:g} mm per scan: " + ", ".join(f"{k[:8]} {len(s.center)}" for k, s in S.items())
          + f"; {int(hit.sum())} hold the lesion's center; unknown and painted swept alike: {same}")
    for name in heads:
        clean = lambda X: X[np.isfinite(X).all(1)]
        null = []
        for i, u in enumerate(NORMALS):                          # leave one subject out
            ref = np.concatenate([clean(V[n][name]) for n in NORMALS if n != u]).mean(0)
            null.append(np.linalg.norm(clean(V[u][name]) - ref, axis=1))
        ref = np.concatenate([clean(V[n][name]) for n in NORMALS]).mean(0)
        d = {k: np.linalg.norm(V[k][name] - ref, axis=1) for k in (UNKNOWN, "painted")}
        allnull = np.concatenate(null)
        T = max(float(x.max()) for x in null)
        pct = lambda x: float((allnull < x).mean() * 100)
        dl = d["painted"][hit]
        r = {"null_median": float(np.median(allnull)), "threshold": T,
             "unknown_flagged": int((d[UNKNOWN] > T).sum()), "painted_flagged": int((d["painted"] > T).sum()),
             "lesion_box_max": float(np.nanmax(dl)), "lesion_box_percentile": pct(np.nanmax(dl)),
             "same_boxes_unpainted": float(np.nanmax(d[UNKNOWN][hit])) if same else None,
             "lesion_rank_in_painted": int((d["painted"] > np.nanmax(dl)).sum()) + 1,
             "unknown_max": float(np.nanmax(d[UNKNOWN])), "unknown_max_percentile": pct(np.nanmax(d[UNKNOWN]))}
        report["heads"][name] = r
        print(f"\n{name}:\n  held-out normal boxes: median distance {r['null_median']:.3f}, largest {T:.3f} (the threshold)"
              f"\n  unknown, as scanned: largest {r['unknown_max']:.3f} (percentile {r['unknown_max_percentile']:.1f}), "
              f"{r['unknown_flagged']} of {len(d[UNKNOWN])} boxes over"
              f"\n  painted: the lesion's boxes reach {r['lesion_box_max']:.3f} (percentile {r['lesion_box_percentile']:.1f}; "
              f"the same boxes unpainted {r['same_boxes_unpainted']:.3f}); rank {r['lesion_rank_in_painted']} of "
              f"{len(d['painted'])}; {r['painted_flagged']} boxes over")
        V[f"_d_{name}"] = d
    # -- head-free, token by token: each fine token of the unknown's ROI against the normals' ----------
    T = {k: roi_tokens(scans[k], ROI[k]) for k in scans}
    null = [knn_distance(T[u][0], np.concatenate([T[n][0] for n in NORMALS if n != u])) for u in NORMALS]
    thr = max(float(x.max()) for x in null)
    ref = np.concatenate([T[n][0] for n in NORMALS])
    tok = {"null_median": float(np.median(np.concatenate(null))), "threshold": thr}
    print(f"\nno head, token by token (fine lattice, 5 nearest normal tokens): held-out normal tokens "
          f"median {tok['null_median']:.3f}, largest {thr:.3f} (the threshold)")
    for k in (UNKNOWN, "painted"):
        dd = knn_distance(T[k][0], ref)
        found = sites(T[k][1], dd > thr)
        near = np.linalg.norm(T[k][1] - center, axis=1) <= a.diameter / 2 + 5
        hit_site = [s for s in found if np.linalg.norm(s - center) <= a.diameter / 2 + 10]
        tok[k] = {"tokens": len(dd), "flagged": int((dd > thr).sum()), "sites": [np.round(s, 1).tolist() for s in found],
                  "lesion_found": bool(hit_site), "lesion_token_max": float(dd[near].max()) if near.any() else None}
        T[k] = (*T[k], dd)
        print(f"  {'unknown, as scanned' if k == UNKNOWN else 'painted'}: {tok[k]['flagged']} of {len(dd)} tokens over, "
              f"{len(found)} site(s)" + (f"; the lesion's tokens reach {tok[k]['lesion_token_max']:.3f} - "
              f"{'FOUND' if hit_site else 'missed'}, other sites {len(found) - len(hit_site)}" if k == "painted" else "")
              + "".join(f"\n      site at LPS {np.round(s, 0).astype(int).tolist()}, {np.linalg.norm(s - center):.0f} mm from the lesion" for s in found))
    report["tokens"] = tok
    if a.figure:
        figure(a.figure, painted, scans["painted"], S["painted"], V, heads, center, report, a.box, T["painted"], thr)
    (work / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\n{time.time() - t0:.0f} s; -> {work / 'report.json'}")


def figure(path, painted, field, sw, V, heads, center, report, size, tokens=None, thr=None):
    """An axial slice through the lesion: the painted CT, and every box through that slice colored
    by its distance from normal, the threshold marked - one panel per head."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from matplotlib.patches import Rectangle
    ct = nib.load(str(painted)); a = np.asarray(ct.dataobj, np.float32)
    inv = np.linalg.inv(ct.affine)
    to_vox = lambda lps: (inv[:3, :3] @ (np.asarray(lps) * [-1, -1, 1]).T).T + inv[:3, 3]
    k = int(round(to_vox(center)[2]))
    fig, axes = plt.subplots(1, len(heads) + (tokens is not None), figsize=(7 * (len(heads) + 1), 7))
    for ax, name in zip(np.atleast_1d(axes), heads):
        d = V[f"_d_{name}"]["painted"]; T = report["heads"][name]["threshold"]
        ax.imshow(a[:, :, k].T, cmap="gray", vmin=-160, vmax=240, origin="lower")
        if ct.affine[1, 1] > 0:                  # j runs anterior (RAS): anterior up, as a radiologist reads
            pass
        else:
            ax.invert_yaxis()
        lo, hi = field.grid.world(sw.lo.astype(float) - 0.5), field.grid.world(sw.hi.astype(float) - 0.5)
        zlo, zhi = np.minimum(lo[:, 2], hi[:, 2]), np.maximum(lo[:, 2], hi[:, 2])
        on = (zlo <= center[2]) & (zhi >= center[2]) & np.isfinite(d)
        norm = matplotlib.colors.Normalize(report["heads"][name]["null_median"], max(T, np.nanmax(d)))
        for i in np.flatnonzero(on):
            p0, p1 = to_vox(lo[i]), to_vox(hi[i])
            x0, x1 = sorted((p0[0], p1[0])); y0, y1 = sorted((p0[1], p1[1]))
            over = d[i] > T
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2.5 if over else 0.8,
                                   ec=plt.cm.inferno(norm(d[i])) if not over else "cyan", alpha=0.9 if over else 0.6))
        c = to_vox(center)
        ax.add_patch(plt.Circle((c[0], c[1]), report["lesion"]["diameter_mm"] / 2 / float(ct.header.get_zooms()[0]),
                                fill=False, ec="lime", lw=1.5, ls="--"))
        r = report["heads"][name]
        ax.set_title(f"{name}: {size:g} mm boxes through the lesion's slice\n"
                     f"cyan = beyond every held-out normal box ({r['painted_flagged']} in the scan); lesion rank {r['lesion_rank_in_painted']}",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    if tokens is not None:
        ax = np.atleast_1d(axes)[-1]
        ax.imshow(a[:, :, k].T, cmap="gray", vmin=-160, vmax=240, origin="lower")
        if ct.affine[1, 1] <= 0:
            ax.invert_yaxis()
        _, c, dd = tokens
        on = np.abs(c[:, 2] - center[2]) <= 5.1                # the fine lattice's slab through the lesion
        p = to_vox(c[on])
        sc = ax.scatter(p[:, 0], p[:, 1], c=dd[on], s=60, cmap="inferno", vmin=0, vmax=max(thr, dd.max()), alpha=0.8)
        over = on & (dd > thr)
        if over.any():
            q = to_vox(c[over]); ax.scatter(q[:, 0], q[:, 1], s=140, facecolors="none", edgecolors="cyan", lw=2)
        cc = to_vox(center)
        ax.add_patch(plt.Circle((cc[0], cc[1]), report["lesion"]["diameter_mm"] / 2 / float(ct.header.get_zooms()[0]),
                                fill=False, ec="lime", lw=1.5, ls="--"))
        t = report["tokens"]["painted"]
        ax.set_title(f"no head, token by token: fine tokens of the ROI in this slab\ncyan = beyond every held-out normal "
                     f"token; lesion {'FOUND' if t['lesion_found'] else 'missed'}, {len(t['sites']) - t['lesion_found']} other site(s)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(sc, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=110)
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
