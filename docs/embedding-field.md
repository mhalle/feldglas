# The embedding field: one design record

*Written 2026-09-22 to stop re-designing it. This product was designed at least five times -
EXPLORATION section 10 and F4 (medseg, 2026-09-20), haversack `docs/result-references.md` steps 2-3
(2026-09-20), the "RADAR capabilities summary" session (2026-09-22, 13:33-14:57; transcript
`~/.claude/projects/-Users-halazar-Dropbox-development-haversack/46e8dbfb-4705-42a6-8058-956f4dffd386.jsonl`),
and twice in the session that built store 0.2. This file supersedes them. Change it, do not
start another.*

## What it is

A CT's **embedding field**: an encoder's token lattices, placed in the patient's world by duckn,
returned to a client like a rankfield store is, and read by that client **with a mask of its own
choosing, on its own CPU** - vectors, finding scores, deviation maps, templates, change between
timepoints. The field is encoder-agnostic: RADAR is the first encoder, TotalSegmentator's
encoder (the null model) the second, and anything behind feldglas's contract fits.

Analogy with the family's class field:

| | rankfield store (haversack) | embedding field |
| --- | --- | --- |
| stored | ranks and gaps of the class field, on the model grid | token lattices, each on its own grid |
| placed by | duckn geometry | duckn geometry, one array per lattice |
| tied to the input | its frame: source grid, orientation, crop | the input CT's content identity and grid, as provenance |
| client operation | restore labels onto any grid | pool under any mask on any grid |
| needs besides the file | nothing | the encoder's head, shipped once per encoder |

## Decisions (settled - do not relitigate)

1. **No mask, and no reference to one** (the user, 2026-09-22). A field is the embeddings placed in
   the world and nothing about how they will be gated - not even in provenance. Every gate is the
   client's (haversack labels by structure name, a drawn ROI, any mask on any grid). Measured basis:
   EXPLORATION 5.11 (the gate's source barely matters), 5.12 (both encoders gated alike).
2. **A zarr v3 group, shipped as a zip with STORED entries** so every chunk is range-addressable
   (on R2, inside the zip, through ZMP). One field per series: `<series>.zarr.zip`.
3. **One duckn array per lattice**, shaped `(Z, Y, X, C)` in **C order**: three `kind: "space"` axes
   with `space_direction` and `centering: "cell"`, then one `kind: "list"` axis for channels (a range
   kind: never resampled). `space_origin` is the center of the first token's box (duckn puts
   `space_origin` at the first SAMPLE for cell and node alike). Chunks tile space and always hold
   the whole channel axis - a token is read whole. Lattices may differ in width (the null model:
   128 / 256 / 320). They are distinct encoder layers, NOT a pyramid (duckn §4.5's pyramid rule
   does not apply): sibling arrays of one group.
4. **The head ships separately**, once per encoder version, referenced by digest from each field:
   RADAR's attention layer, per-organ projections and finding-text table (~11 MB, CC BY-NC-SA).
5. **Clients pool on a CPU**, numpy only (`feldglas.session`, `gate`, `heads`): 0.2-0.3 ms a box.
6. **Geometry comes from duckn's models, never hand-written dicts**, and the model grid is DERIVED
   from the lattices, with every lattice checked against it (one fact, one place).
7. **int8 goes through a value transform in duckn core**, never sibling scale arrays (duckn §4.3
   warns against metadata and bytes kept in agreement by hand; zarr's scale-offset codec is a
   numcodecs extension JavaScript readers lack). Until duckn has it, fields are fp16 with no
   transform.
8. **License travels on the file.** The embeddings inherit the weights' license (RADAR: CC BY-NC-SA
   4.0; null model: Apache-2.0).

## Metadata, in duckn's terms

**Core fields, per lattice array** (all exist in duckn today):
- `space` (`left-posterior-superior`), `space_origin`, per-axis `space_direction`, `centering`,
  `unit: "mm"` - placement.
- `thickness` on each space axis - **the token's measured receptive field, as a full WIDTH** (about
  the full width at half maximum of the measured point spread; the spec defines thickness as the
  extent of the region measured to produce each sample). RADAR: 20 mm fine, 40 mm mid, 80 mm deep
  (EXPLORATION section 2). A viewer then knows a map's real resolution.
- `intent: "embedding-field"` - documentation only; the vocabulary is open.

**The `embedding` extension, 0.1** - an unregistered extension under duckn's rule (`456516e`: its own
name, 0.x while unstable, JSON types, `schema` pointing at this file). It is general: it must fit
RADAR, the null model's unequal widths, a ViT's patch tokens, a pathology model's tiles, and a
pooled vector with no space axes.

```json
"extensions": {
  "embedding": {
    "version": "0.1",
    "schema": "<this document>",
    "space": {"model": "radar", "weights": "sha256:...", "layer": "encoder.stage5", "stage": "raw"},
    "projects_to": {"space": "radar:text-256", "head": "sha256:..."},
    "metric": "cosine", "normalized": false,
    "kernel": [8, 32, 32],
    "support": {"offset": [-6.6, -6.3, -7.6], "unit": "mm"},
    "extent": {"lo": [0, 0, 0], "hi": [8, 11, 12]},
    "group": {"id": "radar/<series>", "member": 0, "members": 3}
  }
}
```

- **`space`** is the comparability key: two vectors compare only if model, weights digest, layer and
  stage agree. It is what stops RADAR's vectors being compared with the null model's, or raw tokens
  with projected ones.
- **`stage`**: `raw` tokens need a head; `projected` vectors are already in a named shared space.
  (The lesson of the study's sign-flip bug: raw tokens are not in the text space.)
- **`support.offset`**: where a token's evidence is centered MINUS where its box is drawn, mm along
  the lattice's axes, positive toward increasing index. RADAR's measured look offset (phantom,
  feldglas `LOOK_OFFSET_MM`) points toward index 0, so its stored offsets are negative. Extent is
  core `thickness`. Moving `space_origin` to absorb the offset would misstate the grid, which is exact.
- **`extent`** (per lattice) and **`data_box`** (root): encoders pad their input - RADAR its crop to
  multiples of 32, the null model to its tiles - so a lattice extends past the scan. `data_box` is the
  half-open box of model voxels that held the scanned image; each lattice's `extent` is the half-open
  box of tokens whose box holds any of them, derived from `data_box` and checked on read. Clients
  drop tokens outside it: a client test (below) found a third of a RADAR field padded, the deep
  padded tokens with larger norms than the body's.
- **`kernel`**: model-grid voxels per token along the lattice's axes. The model grid is derived from
  it (decision 6); a pooled vector with no space axes omits it.
- **`group`**: which arrays are the layers of one field. The root's `embedding` extension lists the
  members in order (`group.members`) with `format: "feldglas-field"`, `format_version: "0.2"` and the
  provenance; each array names the group id and its place, and the reader refuses a lattice whose
  group, place, model, weights, stage or metric disagree.
- Facts not known are **not stated** (no `thickness`, no `support`, no `projects_to`), never guessed.

**Provenance** (duckn's provenance extension is drafted but NOT implemented; until it is, these facts
ride in the `embedding` extension or the group's attributes, and move when it lands):
- `sources`: the input CT - content digest (haversack's identity), DICOM UIDs, collection, license,
  citation; its **grid** in duckn's form (shape, directions, origin) **as the file was delivered**,
  so a client with a mask on the CT's grid knows it is the same image - never the encoder's
  reoriented copy (RADAR's LAS grid runs the other way in y: matching by index would swap anterior
  and posterior). Stored today as `provenance.input` (`identity`, `grid`). Acquisition facts (phase and how it was decided,
  manufacturer, kernel, dose) belong here too: RADAR's vectors carry them and they cannot be removed
  afterwards (5.3-5.5), and duckn §4.5 drops the CT's own metadata on derivation, so the field
  restates them.
- `processing`: the encoder's input conventions as parameters of the encode step - RADAR: reorient
  to LAS, clip to [-300, 400] HU, trilinear resample to 1 x 1 x 5 mm with no prefilter (5.13: a
  prefilter is a distribution shift of about a patient step), the crop, window or whole-volume mode.
- `attribution`: the license and citation.

## What is built, and what is not

Built (feldglas main, 2026-09-22):
- **Store 0.2** (`f4e4831`, aligned to this record the same evening, `src/feldglas/store.py`): the
  group, the C-order lattice arrays placed by duckn and checked by duckn's own `VolumeGeometry`, no
  mask, fp16, the model grid derived and cross-checked; `intent`, `thickness`, the `embedding`
  extension as above, and `provenance.input` (the input CT's identity and grid). A draft written
  with the first day's `feldglas` extension is refused by name. In the contract: `Embedding` on
  every `Field`, `Provenance.input`. RADAR states its layers, measured reach (20 / 40 / 80 mm,
  EXPLORATION section 2's point spread) and look offsets, and its input grid from the exporter's
  `image_affine_ras` (the LAS-reoriented image's grid: the CT's voxels, axes permuted and flipped);
  the null model states its layers (`encoder.stages.N`) and nothing it has not measured.
- Mixed-width lattices and `LatticeMeanHead` (`d5d8d09`); the atlas gates by haversack labels
  (`1e91eb2`); boxes centered correctly (`d4eaba1`).

Not built:
1. The null model's reach and look offset (a phantom run like RADAR's), and its input grid: its
   exporters record `image_shape` but no affine.
2. The input CT's content digest, DICOM UIDs and acquisition in `provenance.input` (today: the
   series uuid only) - from haversack's `.input.json` once fields come through haversack.
3. Exporters writing 0.2 (`radar_export_modal.py`, `radar_encode_local.py`, the null tools still
   write 0.1 `.npz`); the 1,680 fields on R2 are 0.1 and stay readable.
4. The head as a packaged, digest-checked artifact with a fetch step (today: hand-placed files in
   `~/.cache/feldglas/radar/`; RADAR's encoder imported from a local upstream clone).
5. A client command: `feldglas describe FIELD --labels labels.seg.nrrd -o radar.json` through
   `Session.table()`.
6. duckn core: `linear` along an axis (per-channel slope and intercept), for int8. A duckn change of
   its own - duckn has its own sessions.
7. Delivery through haversack: a `radar` deliverable (a file in the result's generation, not part
   of its key) returning the field beside the labels. Needs feldglas published (it has no remote).
8. Derived products, later: multiscale lesion-probability maps as seg-extension fractional label
   maps per lattice with `thickness`, `role: "unknown"` outside the gate (designed in the
   capabilities session, 14:45); deviation maps against a donor atlas (5.6: plain distance in the
   liver); score maps restored to the CT grid through rankfield's `regions`.

## Open questions

- Prior art: is there an existing convention for embeddings in chunked arrays (OME-NGFF feature
  maps, geospatial raster embeddings, ML dataset formats)? Survey before naming `embedding` 1.0.
- Should `sources` carry the full transform from the CT's grid to the model grid (as haversack's
  frame does), or is world placement plus the CT's grid enough? Every client operation so far needs
  only the latter - including the client test below.

## The client test (2026-09-22)

A field of IDC series `159dff32-...` (RADAR, fp16, encoded on the M2), its CT and haversack's
`ts.v2:total_fast` labels were handed to an agent with a client's directions and nothing else - no
feldglas, haversack, duckn or rankfield code, no other files. It verified placement three ways
(the liver's tokens average 0.9 mm from the segmentation's liver centroid; ridge-predicted box HU
R2 0.83 at the stated placement, 0.36-0.50 with an axis flipped; organ nearest-centroid 0.39 against
0.19-0.21), found paired organs pooling alike (7 of 8), a liver-likeness map (AUC 0.93-0.99) and a
coherent pair of fine liver tokens at cosine 0.15-0.17 over a faint 8-10 mm low-density focus. Its
usability findings, and what changed: (1) a third of each lattice past the scan, unmarked -
`data_box` and `extent`; (2) the provenance grid was RADAR's reoriented one - now the delivered
file's; (3) `crop` / `resample_target` in different axis orders, undocumented - `extra` is declared
internal, `data_box` is the interface; (4) axis order and spacing only in `space_direction` - the
guide says so; (5) reach width-or-radius and offset sign unstated - defined above; (6) edge tokens
less reliable, (7) raw cosines high everywhere (a shared mean must come off before comparing) - in
the guide; (8) `schema` pointed at a file the client lacks - the guide, `src/feldglas/field_readme.md`,
is packed into every field as `README.md` and `schema` names it first.
