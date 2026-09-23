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
7. **int8 goes through a value transform, never sibling scale arrays** (duckn §4.3 warns against
   metadata and bytes kept in agreement by hand; zarr's scale-offset codec is a numcodecs extension
   JavaScript readers lack). Built 2026-09-22 WITHOUT waiting for duckn (the user: try it): the
   transform `embedding.linear_along_axis` (`axis`, per-channel `slope` and `intercept`) sits in
   core `value_transforms`, namespaced by this record. duckn's own rule makes that safe - a reader
   meeting an unknown transform name treats the value mapping as unknown and offers only the stored
   integers - where scales kept only in an extension would let a plain duckn reader take int8 for
   the values. feldglas refuses any transform it does not know. Measured on a RADAR field against
   its fp16 tokens: token cosine >= 0.9998, pooled organ cosine (shared mean removed) >= 0.99992,
   33.8 -> 15.7 MB (zstd). One scale per lattice - duckn's standard scalar `linear`, no invention -
   was ten times worse (token cosine >= 0.9968, 12.9 MB); fp16 against fp32 is itself ~0.9997. If
   duckn core adds a linear along an axis, rename to it. Float (fp16) stays the default.
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

## Encoding moves into haversack (decided 2026-09-23)

*The user's reason: weights management in ONE program.* Settled the same day:

- **The split.** haversack owns encoding: the encoders, their weights (fetch, list, remove,
  coverage), inputs, jobs, caching and serving - `haversack encode` beside `haversack segment`.
  feldglas owns the field FORMAT (it writes and reads fields) and the receiving end: the client,
  references, scoring, the `feldglas` command, the guide every field carries. haversack depends on
  feldglas for the writer only; feldglas never imports haversack (today its null adapter does - that
  cycle goes). One author per thing: feldglas the format, haversack the encoders. No torch in feldglas.
- **Names: haversack's task grammar, `family[.version]:name[@revision]`,** adopted now, while the
  names in fields and references are few. `radar:generalist@<hf-commit>` (RADAR's released
  checkpoint; the name follows its Hugging Face repository, `radar-generalist/RADAR` - makers' names
  are not renamed); `ts.v2:total_fast` is the encoder of that network (stages 2-4), `ts.v2:total`
  the 1.5 mm variant. `segment X` and `encode X` name the same network; the verb decides labels or
  tokens. `haversack encoders` lists what can be encoded, apart from `haversack tasks`. Old names
  resolve through an alias table (`radar`, `null-totalsegmentator`, `null-totalsegmentator-1.5mm`),
  as haversack keeps legacy bare task names resolving; the name is part of every field's and every
  reference's comparability key.
- **Modularity: one generic pipeline, one small module per algorithm** (`haversack.encoders`). The
  pipeline does what every encoder shares - fetch and record the input, reorient and resample to the
  encoder's declared convention, crop, pad, the data box, placement, provenance (the input CT's grid,
  license, version), writing through feldglas. An algorithm supplies a SPEC (name, pinned weights and
  digest, license and citation, input conventions, lattices: layers, kernels, widths; measured reach
  and look offset when known), `load(weights, device, dtype)` and `encode(model, tensor)`. haversack's
  engine rule carries over: a new need is a new spec FIELD, not a branch. An encoder completeness test
  (pinned weights, attribution, a random-weights run that writes a valid field, a placement check),
  after `test_engine_completeness`. The null model's tiling and stage extraction are generic over
  nnU-Net networks: one "nnU-Net encoder" module makes every nnU-Net catalog haversack has (ts.v2,
  ts.v3, moose, cads, mrsegmentator) encodable by a registry row naming its stages.
- **Interface.** `haversack encode INPUT --encoder radar:generalist -o scan.zarr.zip [--int8]
  [--device] [--dtype]` (INPUT: any haversack input - a file, DICOM, `idc:`/`tcia:`); `haversack
  encoders`; `haversack weights fetch radar:generalist`; `haversack remote encode`. Server: `POST
  /v1/jobs` with `kind=encode`; `GET|HEAD /v1/<source>/<id>/<encoder>/field.zarr.zip` (ETag the
  content digest); `GET /v1/encoders`. `feldglas score --haversack ... --series ...` then takes the
  field and the labels from one server.
- **Phases.** 1: local `haversack encode` - the RADAR and nnU-Net encoders, the Hugging Face weights
  fetch, `weights list/remove/coverage` generalized beyond TotalSegmentator, `encode` and
  `encoders`, attribution (CC BY-NC-SA for RADAR's weights), the completeness test. No server change.
  2: the encode job on the server - an output KIND through the result cache, jobs, routes,
  `/v1/jobs/{id}/result` and the Modal publish (every one of them assumes `labels.seg.nrrd` today),
  a GPU worker, `/v1/encoders`, `remote encode`. Labels stay the default kind; result keys do not
  move. Not a DELIVERABLE of a segmentation: those are light renders from the input and labels in
  memory, and a field needs a GPU, 1.5 GB of weights and the input - gone on a cache hit.
- **The head.** RADAR's attention head and finding-text table are cut from the same checkpoint;
  `haversack weights fetch radar:generalist` produces them too, and feldglas (whose numpy head
  applies them, client-side) reads them from where haversack puts them.
- **Migration.** feldglas's encode halves (`adapters/radar.py`, `adapters/null.py` encode parts,
  `tools/*_encode_local.py`, the preprocessing borrowed from `tools/radar_export_modal.py`) move to
  haversack; the study tools call haversack until phase 2 makes the Modal exporters haversack jobs.
- **Prerequisite:** feldglas on GitHub with a tag, so haversack can pin it (its CI and Modal images
  install siblings from pinned Git URLs). Development may start against an editable install.

## What is built, and what is not

Built (feldglas main, 2026-09-22):
- **Store 0.2** (`4bd9efa`, aligned to this record the same evening, `src/feldglas/store.py`): the
  group, the C-order lattice arrays placed by duckn and checked by duckn's own `VolumeGeometry`, no
  mask, fp16, the model grid derived and cross-checked; `intent`, `thickness`, the `embedding`
  extension as above, and `provenance.input` (the input CT's identity and grid). A draft written
  with the first day's `feldglas` extension is refused by name. In the contract: `Embedding` on
  every `Field`, `Provenance.input`. RADAR states its layers, measured reach (20 / 40 / 80 mm,
  EXPLORATION section 2's point spread) and look offsets, and its input grid from the exporter's
  `image_affine_ras` (the LAS-reoriented image's grid: the CT's voxels, axes permuted and flipped);
  the null model states its layers (`encoder.stages.N`) and nothing it has not measured.
- **`examples/organ_vectors.py`**, the simplest client: one vector per organ from the field and a
  segmentation (zarr, numpy, SimpleITK; no feldglas), then organ-to-organ similarity (tokens whose
  center is in the organ, body mean removed) and, given RADAR's head, finding scores per organ
  (every token whose box TOUCHES the organ - the head's own gate; centers only read small organs
  wrongly: duodenum "diverticulum" 0.91 from 5 tokens against 0.02 from 20). Its reader is held to
  the store's by a test; on sample `159dff32` its scores agree with `gate.select(rule="any")` to ~0.04.
  0.8 s a scan on the M2. The head needs `findings_en.npz` (the text table with upstream's English
  names) beside `head.npz` - part of Not built 4.
- **A deviation from normal needs no head** (EXPLORATION 5.14, 2026-09-22). RADAR's fields pooled by
  the mean per lattice (`radar_atlas_modal.py` as `radar-mean`) detect tumor as well as or better
  than through the head (32 mm AUC 0.971 / 0.905 / 0.931 against 0.920 / 0.850 / 0.835). With only a
  few normals, test TOKENS, not boxes: `tools/roi_atlas_demo.py` - fine tokens against their 5
  nearest normal tokens, thresholded at the largest held-out normal - flagged every painted liver
  lesion from 10 mm / -40 HU, with two false sites; 32 mm boxes flagged none. The technique is in
  the client guide ("Is a region unlike normal tissue?"). A shipped per-organ normal (a few donors'
  tokens or mean vectors, per protocol) would let a client do this with the field alone - not built.
- Mixed-width lattices and `LatticeMeanHead` (`ab5f73b`); the atlas gates by haversack labels
  (`236ea87`); boxes centered correctly (`e54a23b`).

Not built:
1. The null model's reach and look offset (a phantom run like RADAR's), and its input grid: its
   exporters record `image_shape` but no affine.
2. The input CT's content digest, DICOM UIDs and acquisition in `provenance.input` (today: the
   series uuid only) - from haversack's `.input.json` once fields come through haversack.
3. Exporters writing 0.2 (`radar_export_modal.py`, `radar_encode_local.py`, the null tools still
   write 0.1 `.npz`); the 1,680 fields on R2 are 0.1 and stay readable.
4. The head as a packaged, digest-checked artifact with a fetch step (today: hand-placed files in
   `~/.cache/feldglas/radar/`; RADAR's encoder imported from a local upstream clone).
5. ~~A client command~~ - BUILT 2026-09-23 as the `feldglas` command (click; `feldglas[client]`):
   `info`, `vectors`, `reference build`, `score`, every file a path or an http(s) URL (obstore; ETag
   revalidation; a token only to its own origin, never over plain http elsewhere). Labels come by
   `--haversack SERVER --series SOURCE:ID --task TASK` from a haversack result path. Real run: labels
   from the public twin for a colorectal-metastasis scan scored against the ldct liver reference -
   69 tokens flagged in 13 sites: another collection and a contrast phase, the guide's "suspect the
   protocol first" in practice. RADAR's head findings (`Session.table()`) are not on the command yet. Then four more adversarial
   reviews (2026-09-23), an agent DRIVING the command among them. Fixed: a URL could send the token
   over plain http to another host (Python's and obstore's parsers disagreed on `\`; $HTTP_PROXY got
   it too) - one parse now decides host and token, local hosts bypass proxies; concurrent fetches
   could pin the wrong version - a lock, private temp files, a sha256 checked before use; haversack's
   202 "still computing" was cached as the labels - bodies must be the kind the caller expects; a
   stale copy served after a REFUSAL - only after a network failure now; a scan scored with ANOTHER
   scan's labels, silently - labels are checked against the field's recorded CT grid; references
   used on other structures, duplicate normals, tampered files, negative erosion - refused;
   tracebacks - one line, and one JSON object under --json. Added for agents: `warnings[]`, how a
   reference's threshold was set (per normal, and where), `info` on references, each site's depth in
   the organ (the review's real lesion was 35 mm deep, its false sites 19-22), typo suggestions,
   `health` importing its dependencies. Erosion runs on the region's box: build 40 -> 11 s, score
   13 -> 3.7 s. Mutation: 27 of 93 killed before; every survivor re-run killed after (one equivalent,
   the two size-cap checks cover each other).
6. duckn core: `linear` along an axis (per-channel slope and intercept). int8 fields work without it
   (decision 7); a core transform would let generic duckn readers decode them. A duckn change of its
   own - duckn has its own sessions.
7. Delivery through haversack: the field by URL beside the labels, so a client needs nothing but
   URLs and a reference - `<server>/v1/<source>/<id>/radar/field.zarr.zip` on the same path surface,
   the encoder in the task's place (haversack has no encoder task today). The client side is ready
   (fields open from URLs, 2026-09-23). Needs feldglas published (it has no remote).
8. Derived products, later: multiscale lesion-probability maps as seg-extension fractional label
   maps per lattice with `thickness`, `role: "unknown"` outside the gate (designed in the
   capabilities session, 14:45); deviation maps against a donor atlas (5.6: plain distance in the
   liver); score maps restored to the CT grid through rankfield's `regions`.
9. **A client library, one API in two languages** (proposed 2026-09-23, after the TypeScript client
   test below; the Python half STARTED the same day - `src/feldglas/client.py`: `open_field`, `Mask`
   on any grid, `occupancy` / `tokens_in` (center, touch, occupancy rules), `organ_vectors`, and
   `Reference.build` / `.score` for focal sites. It reproduces the painted-lesion demo from CT-grid
   masks: threshold 0.205, every lesion found, the same two false sites, 0.3 s a scan. Then four
   adversarial reviews (geometry, README-as-spec, mutation, robustness): the first `occupancy` sent
   mask voxels to tokens at a stride - touch-gating missed 3-33 % of an organ's touching tokens, a
   cropped ROI reported slivers as full, a mask coarser than the tokens lost them - so occupancy now
   samples each token's own BOX (0 outside the mask array); the client refuses every lie the store
   refuses (a reversed member list had made `fine` the coarsest lattice) and anything non-finite,
   empty or integer-without-a-transform; keys tell unnamed lattices apart; sites join by single
   linkage on a grid; the guide lost two definitions of "the body" and gained the transform's JSON,
   the optional keys, the NIfTI both-codes-zero case. Mutation: 34 of 69 killed before, every
   survivor re-run killed after. Real data again: threshold 0.195, same sites, 1.7 s a scan (exact
   box sampling costs more than the stride did). Masks from `.seg.nrrd` are the featured input: the one
   reader (`labels.read_seg_nrrd`) now takes 3D Slicer's LAYERED files (overlapping segments - a
   lesion over its organ - on a 4-D `list` axis, `SegmentN_Layer`) and RAS files as well as
   haversack's one-layer LPS ones; `structure_mask` / `seg_mask` union by name across layers, with
   `optional=` for lesion segments a scan may not have (haversack lists only what a scan has); the
   guide describes the file for any language. Still to do:
   depth strata and the diffuse score, the reference file, the fixtures, the TypeScript half). Python `feldglas.client` on numpy + zarr alone: open (decoded tokens, centers,
   extent, provenance, the input grid), tokens under a mask (center or touch rule), organ vectors,
   and `Reference.build(normals, regions, strata)` / `.score(field, region)` giving focal sites
   (tokens against their nearest normal tokens) and a diffuse score per depth stratum (surface
   included - cirrhosis lives there). A **reference file** in the same zarr zip form, carrying the
   comparability key and the protocol so a mismatched pair is refused. Then `@feldglas/client` in
   TypeScript on zarrita and `@duckn/spatial`. The two are held together by shared conformance
   fixtures (a tiny field, float and int8, a mask, a reference, and the expected outputs as JSON),
   the README being the specification. What the TS agent wrote by hand is that package's list: the
   value transform, an NRRD label reader, a 3 x 3 solve, RAS/LPS.

**A TypeScript client (2026-09-23).** A fourth agent, Node and npm only - no Python, no feldglas or
duckn package - on an int8 field: zarrita with `@zarrita/storage`'s `ZipFileStore` opened the zip and
decoded zstd with no code of its own; open, read and decode of all three lattices 172 ms; the value
transform took 8 lines from the README. Placement held (nearest-prototype 0.43 / 0.53 against
0.25-0.28 mirrored), the input grid matched `ct.nii.gz` to 5e-10 mm, and its within-scan "unlike the
rest of the liver" found the same vessel-and-fissure spot as every other agent - while saying what
it cannot do without normals. Its findings, in the guide: the README was Python-centric (zarr's
`ZipStore`, numpy calls, `imshow`) - readers for both languages are named and the recipes are math;
"nibabel's i, j, k" named a library where a RULE was needed - a NIfTI's grid is its sform when
`sform_code > 0`, else its qform (nifti-reader-js prefers the qform when its code is higher, so JS
and Python readers can disagree on a file where they differ); the shared-mean recipe now says what is
and is not re-normalized; "less reliable" edge tokens got a rule (index `lo` or `hi - 1`); the
"unlike normal" section says up front that it needs normal scans; `provenance.extra`'s keys are named
as informational.

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

A second agent, same rules, given only the rebuilt field (whose packed `README.md` was its sole
documentation), the CT and the labels: the input grid matched `ct.nii.gz` to < 1e-9 mm, placement
held (ridge R2 0.64 as stated against 0.16-0.30 flipped, peak within half a step), the liver
prototype scored held-out liver at AUC 0.92, and centering took pooled-organ cosines from a median
0.85 to 0.39 with sensible neighbors (liver-spleen 0.91, heart-aorta 0.92). Its findings, fixed in
the guide: `extent` does not cover what RADAR cropped away before encoding (the converse is now
stated); the model grid was undefined (a formula now); when to use `support.offset`; `directions`
rows carry the spacing; `version` versus `format_version`. A third agent, on an int8 field, decoded it from the README alone
(placement best at the stated position, 0.552 balanced accuracy against 0.28-0.32 mirrored); its
findings - a worked, world-coordinate recipe for drawing a slice (its first overlay came out
mirrored), `data_box`'s axis order, reach against the drawn box, the model grid against the CT,
`duckn` undefined, the reoriented affine in `extra`, the body mean's definition - are in the guide
or fixed. Left open: the identity is a bare series
id with no archive or digest (Not built 2), and **`thickness` may understate the reach** - its
strongest liver outlier sat at the liver dome against the heart with no lesion in the CT, as if the
fine tokens saw more than 20 mm.
