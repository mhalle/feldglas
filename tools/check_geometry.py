"""Is a field where it says it is? Its own organs' centroids against haversack's, in the patient.

Geometry bugs are silent: a flipped axis or a forgotten crop offset gates, pools and scores
perfectly well and then draws the finding on the wrong organ. The exporter places the model grid
in the world from the image's affine, the resample and the crop; this checks that placement
against an INDEPENDENT source - the centroids haversack's `statistics.json` reports (RAS mm) for
the same series, from a different network, on a different grid, through a different resampler.
Two segmentations of one organ disagree by millimetres; a geometry error is tens of them.

Run:  uv run python tools/check_geometry.py [field.npz ...]
      (default: every field under the radar cache; statistics from medseg's study folder)
"""
import json, pathlib, sys

import numpy as np

from feldglas.paths import encoder_dir
from feldglas.store import read_field

STATS = pathlib.Path(__file__).resolve().parents[2] / "medseg" / "docs" / "radar-idc-validation" / "results" / "validation" / "total_stats"
# RADAR's merged structure -> the ts.v2:total structures it covers
SAME = {"liver": ["liver"], "spleen": ["spleen"], "stomach": ["stomach"], "pancreas": ["pancreas"],
        "gallbladder": ["gallbladder"], "kidney": ["kidney_left", "kidney_right"]}


def main(paths):
    paths = [pathlib.Path(p) for p in paths] or sorted((encoder_dir("radar") / "fields").glob("*.npz"))
    worst = 0.0
    for p in paths:
        f = read_field(p)
        s = STATS / f"{f.provenance.source}.json"
        if not f.exact_geometry or not s.exists():
            print(f"{p.name}: skipped ({'no exact geometry' if not f.exact_geometry else 'no haversack statistics'})"); continue
        st = {x["structure"]: x for x in json.loads(s.read_text())["structures"] if x.get("voxels", 0) > 0}
        print(f"{f.provenance.source}")
        for organ, names in SAME.items():
            v = f.native_labels.index(organ) + 1
            idx = np.argwhere(f.native_mask == v)
            have = [st[n] for n in names if n in st]
            if len(idx) < 500 or len(have) != len(names):
                continue
            lps = f.grid.world(idx.mean(0))
            ras = np.array([-lps[0], -lps[1], lps[2]])
            w = np.array([h["volume_ml"] for h in have])
            ref = (np.array([h["centroid_ras_mm"] for h in have]) * w[:, None]).sum(0) / w.sum()
            d = float(np.linalg.norm(ras - ref)); worst = max(worst, d)
            print(f"   {organ:12} field {np.round(ras, 1)}   haversack {np.round(ref, 1)}   apart {d:5.1f} mm")
    print(f"worst disagreement {worst:.1f} mm" + ("  <- look at this before trusting any map" if worst > 15 else ""))
    return 0 if worst <= 15 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
