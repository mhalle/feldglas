"""The simplest use of an embedding field: ONE vector per organ, then (1) which organs look alike
and (2) - with the encoder's head - which findings each organ's vector points to.

A client's script: it reads ``field.zarr.zip`` with zarr and numpy alone, as its packed README.md
describes, and a segmentation of the same CT (``labels.seg.nrrd``, e.g. haversack's
``ts.v2:total_fast``) with SimpleITK. Nothing of feldglas is needed for part 1. Part 2 needs RADAR's
head - the attention layer and per-organ projections that turn raw tokens into RADAR's text space,
and the finding-text table - which ships separately from the field, once per encoder (CC BY-NC-SA,
not distributed with this code); its numpy implementation is feldglas's, imported only there.

    python organ_vectors.py field.zarr.zip labels.seg.nrrd
    python organ_vectors.py field.zarr.zip labels.seg.nrrd --head head.npz --findings findings_en.npz

Part 1, any encoder: an organ's vector is the mean of its unit-normalized tokens (one lattice,
tokens whose center lies in the organ and inside the lattice's extent), minus the body's mean, then
unit-normalized - raw cosines are high everywhere, so without the shared mean every organ looks
like every other. Cosine between two such vectors is how alike the encoder finds them.

Part 2, RADAR: an organ's vector is RADAR's own - its learned query attends over the organ's tokens
on all three lattices, and the organ's projection puts the result in the text space where each
finding is a (negative, positive) sentence pair; a finding's score is RADAR's probability. One scan
and no reference population: these are the model's readings, not diagnoses.

The two parts gate differently, on purpose. Part 1 takes tokens whose CENTER is in the organ -
mostly the organ, a purer average. Part 2 takes every token whose box TOUCHES the organ, because
that is the gate RADAR's head was trained on; fed centers only, small organs read wrongly (on one
normal scan: duodenum "diverticulum" 0.91 from 5 tokens against 0.02 from its 20 touching ones;
esophagus "hiatal hernia" 0.97 against 0.12). Use the head's own gate with a head.
"""
import argparse
import json

import numpy as np


def read_field(path):
    """Every lattice as ``(tokens (N, C) float32, centers (N, 3) LPS mm, in_extent (N,) bool,
    steps (3, 3), origin, shape)``, decoding int8 through the field's own transform."""
    import zarr
    root = zarr.open_group(store=zarr.storage.ZipStore(path, mode="r"), mode="r")
    ext = root.attrs.asdict()["duckn"]["extensions"]["embedding"]
    out = []
    for name in ext["group"]["members"]:
        arr = root[name]
        d = arr.attrs.asdict()["duckn"]
        v = arr[:]
        for t in d.get("value_transforms") or []:
            if t["name"] != "embedding.linear_along_axis":
                raise SystemExit(f"{name}: unknown value transform {t['name']!r} - its values are undefined")
            v = v.astype(np.float32) * np.asarray(t["parameters"]["slope"], np.float32) \
                + np.asarray(t["parameters"]["intercept"], np.float32)
        shape = v.shape[:3]
        idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
        D = np.array([a["space_direction"] for a in d["axes"][:3]], float)
        centers = np.asarray(d["space_origin"], float) + idx @ D
        e = d["extensions"]["embedding"]["extent"]
        inside = np.all((idx >= e["lo"]) & (idx < e["hi"]), axis=1)
        out.append((v.reshape(-1, v.shape[-1]).astype(np.float32), centers, inside, D,
                    np.asarray(d["space_origin"], float), shape))
    return out, ext


def read_labels(path):
    """``(label map (i, j, k), world -> index, index -> world, spacing, {value: name})`` from a .seg.nrrd."""
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)             # SimpleITK is (k, j, i)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3) * np.asarray(img.GetSpacing(), float)
    o = np.asarray(img.GetOrigin(), float)                           # SimpleITK is LPS, like the field
    names = {}
    for k in img.GetMetaDataKeys():
        if k.startswith("Segment") and k.endswith("_Name"):
            n = k[len("Segment"):-len("_Name")]
            names[int(img.GetMetaData(f"Segment{n}_LabelValue"))] = img.GetMetaData(k)
    to_index = lambda p: np.rint(np.linalg.solve(D, (p - o).T).T).astype(int)
    to_world = lambda ijk: o + ijk @ D.T
    return arr, to_index, to_world, img.GetSpacing(), names


def labels_at(points, labels, to_index):
    """The label under each world point (0 outside the label map)."""
    i = to_index(points)
    ok = np.all((i >= 0) & (i < labels.shape), axis=1)
    out = np.zeros(len(points), int)
    out[ok] = labels[tuple(i[ok].T)]
    return out


def touching(lattices, labels, to_world, spacing):
    """For each lattice, ``{label value: token indices}`` of the in-extent tokens whose box holds
    any voxel of that label - found by sending labeled voxels (a sub-sample finer than half the
    smallest token step) to the token they fall in."""
    steps = min(float(np.linalg.norm(D, axis=1).min()) for D in (l[3] for l in lattices))
    stride = [max(1, int(steps / 2 / sp)) for sp in spacing]
    sub = labels[::stride[0], ::stride[1], ::stride[2]]
    ijk = np.argwhere(sub > 0) * stride
    values = sub[sub > 0]
    points = to_world(ijk)
    out = []
    for tokens, centers, inside, D, o, shape in lattices:
        t = np.rint(np.linalg.solve(D.T, (points - o).T).T).astype(int)
        ok = np.all((t >= 0) & (t < shape), axis=1)
        flat = np.ravel_multi_index(tuple(t[ok].T), shape)
        keep = inside[flat]
        pairs = np.unique(np.stack([values[ok][keep], flat[keep]]), axis=1)
        out.append({int(v): pairs[1][pairs[0] == v] for v in np.unique(pairs[0])})
    return out


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def organ_similarity(lattice, labels, to_index, names, min_tokens=5):
    tokens, centers, inside = lattice[:3]
    lab = labels_at(centers, labels, to_index)
    body = inside & (lab > 0)
    ref = unit(tokens[body]).mean(0)                                 # the shared mean to take off
    vec = {}
    for value, name in names.items():
        sel = inside & (lab == value)
        if sel.sum() >= min_tokens:
            vec[name] = unit(unit(tokens[sel]).mean(0) - ref)
    order = sorted(vec)
    M = np.array([vec[n] for n in order])
    return order, M @ M.T, {n: int((inside & (lab == v)).sum()) for v, n in names.items() if n in vec}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("field"); ap.add_argument("labels")
    ap.add_argument("--lattice", type=int, default=-1, help="which lattice for part 1 (default: the last, finest)")
    ap.add_argument("--organs", default="liver,spleen,kidney_left,pancreas,stomach,heart,aorta,lung_lower_lobe_left,vertebrae_L1",
                    help="organs to report neighbors for (part 1)")
    ap.add_argument("--head", help="RADAR's head.npz (part 2)")
    ap.add_argument("--findings", help="RADAR's finding-text table in English, {'Organ_Finding': (2, 256)} (part 2)")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--json", help="also write the results here")
    a = ap.parse_args()

    lattices, ext = read_field(a.field)
    labels, to_index, to_world, spacing, names = read_labels(a.labels)
    print(f"{a.field}: {ext['provenance']['encoder']} ({ext['provenance']['license']}), "
          f"{len(lattices)} lattices; {len(names)} structures in {a.labels}")

    # -- part 1: which organs look alike ---------------------------------------------------------
    order, S, counts = organ_similarity(lattices[a.lattice], labels, to_index, names)
    result = {"similarity": {}}
    print(f"\n1. One vector per organ (lattice {a.lattice % len(lattices)}, body mean removed): nearest organs by cosine")
    for organ in [o for o in a.organs.split(",") if o in order]:
        i = order.index(organ)
        near = [(order[j], float(S[i, j])) for j in np.argsort(-S[i]) if j != i][:a.top]
        result["similarity"][organ] = near
        print(f"   {organ:22} ({counts[organ]:4} tokens)  " + ", ".join(f"{n} {c:.2f}" for n, c in near))

    # -- part 2: which findings each organ's vector points to (RADAR's head) -------------------
    if a.head and a.findings:
        from feldglas.adapters.radar import RadarHead, query_for     # the head's numpy code
        head = RadarHead.load(a.head)
        with np.load(a.findings) as z:
            findings = {k: z[k] for k in z.files}
        tokens = np.concatenate([l[0] for l in lattices])            # RADAR attends over every lattice
        offsets = np.cumsum([0] + [len(l[0]) for l in lattices])
        touch = touching(lattices, labels, to_world, spacing)
        groups = {}                                                  # TotalSegmentator names -> RADAR's organs
        for value, name in names.items():
            try:
                groups.setdefault(query_for(name), []).append(value)
            except KeyError:
                pass
        prepared = head.prepare(tokens)
        result["findings"] = {}
        print(f"\n2. One RADAR vector per organ, scored against its findings (top {a.top}; model readings, not diagnoses)")
        for organ in sorted(groups):
            index = np.unique(np.concatenate([touch[j].get(v, np.empty(0, int)) + offsets[j]
                                              for j in range(len(lattices)) for v in groups[organ]]))
            pairs = {k.split("_", 1)[1]: p for k, p in findings.items() if k.lower().startswith(organ + "_")}
            if len(index) < 5 or not pairs:
                continue
            v = head.pool(prepared, index, organ)
            scores = sorted(((f, head.score(v, p)) for f, p in pairs.items()), key=lambda x: -x[1])
            result["findings"][organ] = scores
            print(f"   {organ:18} ({len(index):5} tokens)  " + ", ".join(f"{f} {s:.2f}" for f, s in scores[:a.top]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
