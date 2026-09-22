"""Encode a CT with the NULL model on THIS machine - Apple MPS, CUDA or CPU - and write a token field.

``null_export_modal.py`` is the cohort path; this is the one-scan path, with the weights haversack
already has here (``~/.totalsegmentator/nnunet/results/Dataset297_*``, or ``$TOTALSEG_WEIGHTS_PATH``)
and nothing fetched. The encode is ``feldglas.adapters.null.encode`` - the same function the Modal
tool calls - so the two paths cannot drift.

The fields are derived from TotalSegmentator's open weights (Apache-2.0) and say so; they are
written to the cache (``$FELDGLAS_CACHE`` or ``~/.cache/feldglas``), never into the repository.

Run:  PYTHONPATH=src ../haversack/.venv/bin/python tools/null_encode_local.py CT.nii.gz [--device mps] [--check]

``--check`` also runs haversack's own ``segment`` of ``ts.v2:total_fast`` and compares, per organ,
where the field's native mask sits in the world with where haversack's labels sit.
"""
import argparse, json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ct", nargs="+", type=pathlib.Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "fp32"))
    ap.add_argument("--weights", default=None, help="haversack's TotalSegmentator weights root (default: its own)")
    ap.add_argument("--variant", default="null-totalsegmentator",
                    help="null-totalsegmentator (total_fast, 3 mm) or null-totalsegmentator-1.5mm (Dataset 291)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="directory for the fields")
    a = ap.parse_args(argv)

    import numpy as np
    from feldglas.adapters import null
    from feldglas.paths import encoder_dir
    from feldglas.store import write_field
    v = null.variant(a.variant)
    out = a.out or encoder_dir(v.name) / "fields"
    t = time.time()
    m, label_map = null.load_model(a.weights, device=a.device, dtype=a.dtype, v=v)
    print(f"{m.folder.parent.name}/{m.folder.name} on {m.device}, {a.dtype}: loaded in {time.time() - t:.1f} s")
    checks = {}
    for ct in a.ct:
        u = ct.name.split(".")[0]
        tokens, labels, meta = null.encode(ct, m, v)
        meta.update(u=u, device=str(m.device))
        arrays = {f"tokens{j}": tk for j, tk in enumerate(tokens)}
        arrays["native_mask"] = null.organ_mask(labels, label_map).astype(np.uint8)
        field = null.field_from_export(arrays, meta)
        p = write_field(out / f"{u}.npz", field)
        print(f"  {u}: grid {meta['model_shape']} -> {list(field.grid.shape)}, {meta['tiles']} tiles in "
              f"{meta['encode_s']} s, {int(field.offsets[-1])} tokens {field.widths}, {p.stat().st_size / 1e6:.1f} MB -> {p}")
        if a.check:
            c = checks[u] = null.check_against_haversack(ct, labels, meta["grid"], label_map, a.weights,
                                                         device=a.device, v=v)
            mm = sorted((r["mm"], k) for k, r in c.items())
            vr = [r["volume_ratio"] for r in c.values()]
            print(f"  check: {len(c)} organs, centroid gap median {np.median([x for x, _ in mm]):.2f} mm, worst "
                  f"{mm[-1][0]:.2f} mm ({mm[-1][1]}); volume ratio {min(vr):.3f}-{max(vr):.3f}")
            for k, r in sorted(c.items()):
                print(f"    {k:24} {r['mm']:6.2f} mm  LPS {r['lps_mm']}  volume x{r['volume_ratio']}")
    if checks:
        p = encoder_dir(v.name) / "geometry_check_local.json"
        p.write_text(json.dumps(checks, indent=1))
        print(f"-> {p}")


if __name__ == "__main__":
    sys.exit(main())
