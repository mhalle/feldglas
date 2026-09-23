# An embedding field: how to read and use this file

*Guide of 2026-09-23, for `feldglas-field` 0.2 with the `embedding` extension 0.1. A field carries
the guide current when it was written.*

This zip is an **embedding field**: what an image encoder computed from one CT, placed in the
patient's world. It is packed into every field as `README.md`; the design record behind it is
feldglas `docs/embedding-field.md`. Everything below is plain arithmetic on a CPU, in any
language; recipes are written as math, not as one library's calls.

## What it holds

One or more **lattices of tokens**. A token is a vector describing a box of the patient; the
lattices are different layers of the encoder, coarse to fine, not a pyramid of one image. Tokens are
compared by the `metric` each lattice states (cosine so far). They are `raw` unless `stage` says
otherwise: raw tokens are the encoder's own features, in no shared text space, with no labels.

There is **no mask** in the field. Which tokens belong to which organ is for you to decide, with
any mask on any grid (a segmentation of the CT, a drawn region).

## Reading it

`duckn` is a metadata convention for n-dimensional arrays (NRRD's geometry, in JSON): each array's
attributes carry a `duckn` object saying where its samples are in the world. Nothing here needs a
duckn library; the fields used are explained below.

A **zarr v3 group inside a zip** with stored entries (chunks are zstd). Readers that open it
directly, zstd included:
- Python: `zarr` (3.x), `zarr.open_group(store=zarr.storage.ZipStore(path, mode="r"), mode="r")`.
- JavaScript / TypeScript: `zarrita` with `ZipFileStore` from `@zarrita/storage`
  (`ZipFileStore.fromBlob(blob)`; zarrita decodes zstd itself). Top-level `await` in Node needs an
  ES module (`"type": "module"`).
Or read the JSON and chunks from the zip yourself: entries are stored, so each is a byte range.
In Python, `feldglas.client` implements this guide (`open_field`, masks on any grid, organ vectors,
`Reference` for normal tissue); a TypeScript client is planned.

- The root `zarr.json`'s attributes hold `duckn` metadata. Its `extensions.embedding`:
  - `group`: `id` (the field's) and `members`, the lattice arrays' names, in order. Each lattice's
    own `group` repeats `id` and gives its place: `member` (its index in that list) and `members`
    (the list's LENGTH). A lattice that disagrees with the root is a broken file.
  - `version` is this extension's version; `format_version` is the file layout's. A reader should
    refuse versions it does not know: a 0.x extension may change incompatibly.
  - Optional, each: `data_box` (and each lattice's `extent`), `support`, `thickness`, `projects_to`
    (present when `stage` is not `raw`: the shared space the vectors were projected into), and
    `provenance.input.grid`. A file without an extent says nothing about padding: treat every
    token as having seen image, and know that it may not have.
  - `data_box`: the half-open box of MODEL-grid voxels (`lo`, `hi`) that held image data, in the
    lattices' axis order (as `kernel` and `extent` are). See "The model grid" and "Which tokens saw
    the scan".
  - `provenance`: `encoder`, `weights`, `license`, and `input` - the input CT's identity and its
    `grid` **as the CT file was delivered** (`shape`; `directions`, one LPS row per array axis,
    each row a full step - its length is the spacing, not a unit vector; `origin`). For a NIfTI,
    the array axes are the file's own voxel axes i, j, k, and the grid is the one nibabel reports:
    the **sform** when `sform_code > 0`, else the **qform** when `qform_code > 0`, else nibabel's
    fallback (the voxel spacing on the diagonal, x negated, centered on the volume - a file with
    both codes 0 states no orientation, and no grid comparison with it means much). Readers
    differ (nifti-reader-js takes the qform when its code is higher), so apply the rule yourself. Compare this grid with your CT's to know a mask on it belongs to this image. `identity` is what the encoder was told about the series (it may be
    only an id); the grid is the check you can make yourself.
  - `provenance.extra` is the encoder's internal record, informational and encoder-specific
    (RADAR: `crop`, `resample_target`, `encode_s`, `prep_max_abs_vs_upstream`). No use of the field
    needs it; do not depend on its keys.
- Each lattice is an array of shape `(A0, A1, A2, C)`, C order: `arr[i0, i1, i2, :]` is one token.

## Where a token is

Coordinates are **LPS millimeters**: x increases to the patient's Left, y to Posterior, z to
Superior (DICOM's convention). A NIfTI's sform/qform is RAS: negate the x and y COMPONENTS - the
first two rows of its 4 x 4 affine, i.e. the first two entries of every direction and of the origin
(not the first two direction rows of `input.grid`, which are array axes).

For a lattice, with `o = space_origin` and `d_a = axes[a].space_direction`:

    center of token (i0, i1, i2) = o + i0 * d_0 + i1 * d_1 + i2 * d_2

Do not assume which anatomical direction an index axis runs: read `space_direction` (the RADAR and
TotalSegmentator encoders store (Z, Y, X), and their Y may run anterior). The exact spacing is
`|d_a|`. `centering: "cell"`: a token is DRAWN as the box half a step either side of its center.
What it SEES is wider - see `thickness` below (RADAR: 20 / 40 / 80 mm, about 2.5 steps in-plane and
2 along the slice axis) - so a token drawn inside an organ
still sees past the organ's edge. The three `space` axes are followed by a `list` axis: the channels.

**To draw a lattice over a CT slice, go through world coordinates**, never through array axes - an
index axis may run anterior or right-to-left, and drawing the lattice's array beside the CT's then
comes out mirrored. For each CT pixel of the slice, take its world point `p` (from your CT's own
grid), and find the token there: with `D` the 3 x 3 matrix whose COLUMNS are `d_0, d_1, d_2`,
solve `D i = p - o` for `i` (a 3 x 3 inverse, computed once) and round each component to the
nearest integer; check `i` is inside the extent. Then paint the pixel with that token's value. The picture is in the CT's own orientation by
construction.

## The model grid

The encoder read the CT resampled onto its own MODEL grid; each token is a box of `kernel` model
voxels (in the lattice's axis order). For any lattice, the model grid has step `d_a / kernel[a]`
along each axis, and its first voxel's center is at

    o - sum_a ((kernel[a] - 1) / 2) * d_a / kernel[a]

Every lattice gives the same model grid. `data_box` is in its voxel indices. It is NOT the CT's
grid: it is whatever the encoder resampled the CT onto (RADAR: about 5 x 1 x 1 mm in its own
(Z, Y, X) axis order, its outer edges on the CT's). Work in world coordinates and nothing depends on it.

## Stored values: float, or int8 with a transform

A lattice is stored either as floats (the tokens as they are) or as **int8**. An int8 lattice's
`duckn.value_transforms` holds one entry,
`{"name": "embedding.linear_along_axis", "parameters": {"axis": 3, "slope": [...], "intercept": [...]}}`
- `axis` 3 is the channel axis, and `slope` and `intercept` have one entry per channel. A float
lattice has no `value_transforms` at all. Integer values with no transform are not tokens: refuse
them. Decode each value in 32-bit float:

    token[..., c] = stored[..., c] * slope[c] + intercept[c]        (c: the channel index)

This transform is defined here, not by duckn itself (yet). A duckn reader that does not know it
treats the values' meaning as unknown - **never use the raw int8 values as tokens**: each channel
has its own scale. Each channel's range is spread over all 256 codes (-128 to 127), so decoded
tokens are within half a step of the originals (cosine to them about
0.9998 or better).

## Which tokens saw the scan

Encoders pad their input, so a lattice usually extends past the scan - often by a third or more.
Each lattice's `extensions.embedding.extent` (`lo`, `hi`, half-open token indices) holds the tokens
whose box contains any voxel of `data_box` - image the encoder was given. **Drop tokens outside
it**: they saw padding only, and can look unlike anything real (a padded deep token's norm can
exceed the body's). A token on the extent's boundary - index `lo` or `hi - 1` along any axis - saw
partly padding; leave boundary tokens out of scoring and references unless the region you study
reaches the scan's edge.

The converse does not hold: an encoder may also CROP the CT before it starts (RADAR trims air
margins), so parts of your CT near its edges may lie under no token at all. Compare the extent's
world box with your CT's to see what the field does not cover.

## What each lattice says about itself

On `extensions.embedding` of each lattice:
- `space`: `model`, `weights`, `layer`, `stage`. Two vectors are comparable only if all four agree.
- `metric`, `normalized` (false: normalize yourself), `kernel` (model voxels per token), `group`.
- `support.offset` (mm along the lattice's three axes, optional): where a token's evidence is
  centered MINUS where its box is drawn; positive toward increasing index. The grid is exact - the
  offset says where the encoder actually looks. To gate by where a token looks rather than where
  it is drawn, add the offset to the token's center (along each axis's unit direction) before
  sampling a mask. It is a fraction of a step; plain centers are fine for most uses.

And in core duckn, per space axis:
- `thickness` (mm, optional): the full width of a token's measured reach along that axis. A map
  read off the lattice is no sharper than this, whatever its spacing.

## Masks from a `.seg.nrrd`

The usual source of a mask is a segmentation of the same CT in a `.seg.nrrd` - what haversack
returns, and what 3D Slicer saves. Nothing but a text parser and gzip is needed to read one:

- **Layout:** a text header (`NRRD0004`, then `key: value` lines, and `key:=value` lines for the
  segment table), a blank line, then the voxels - `gzip` or `raw` per `encoding`, `endian` for
  multi-byte types - in Fortran order: the FIRST axis in `sizes` varies fastest.
- **Place:** `space directions` gives one vector per array axis (a full step) and `space origin` the
  first voxel's center; world = origin + i*d_0 + j*d_1 + k*d_2. `space` is `left-posterior-superior`
  (use as is) or `right-anterior-superior` (negate the x and y components of every direction and of
  the origin to reach LPS).
- **Segments:** for each `N`, `SegmentN_Name`, `SegmentN_LabelValue`, and in layered files
  `SegmentN_Layer`. A structure is the voxels of ITS layer that equal ITS label value. Select by
  NAME - label values differ between tasks, versions and tools.
- **Two styles.** LABELMAP style (`dimension: 3`, what haversack writes): one volume, each voxel the
  label value of its segment - so segments cannot overlap, and a structure is the voxels equal to
  its value. LAYERED style (`dimension: 4`, what Slicer writes when segments overlap - a lesion drawn
  over its organ): a first axis of kind `list` (its space direction `none`), each layer a labelmap of
  its own. A voxel may then belong to several structures, so there is no single label per voxel:
  build each structure's mask from its own layer, and union across layers.
- **A plain labelmap `.nrrd`** (no `Segment` lines; its names in a separate color table) is labelmap
  style with the names supplied from elsewhere - `read_seg_nrrd(path, names={"liver": 5, ...})`.
- A name must name one segment, and one layer's label value one name; refuse a file that breaks
  either - a lookup by name would otherwise be a silent choice.

In Python: `feldglas.client.read_seg_nrrd(path)`, then `structure_mask(labels, *names)` (or
`seg_mask(path, *names)` in one step). In JavaScript, the header and gzip are about 30 lines.

## Using a mask

**A region may be several segments: union them.** A segmentation often labels a lesion as its
own segment rather than as part of the organ - TotalSegmentator's `kidney_cyst_left` beside
`kidney_left`; the tumor classes of liver and kidney tumor models. A token outside your region is
never scored, so testing the organ label alone misses such a lesion silently. The region you test
is the union of the organ and its lesion segments - on whatever layers they sit. A lesion segment
is often ABSENT: haversack's files list only the structures a scan has, so a scan without a cyst has
no `kidney_cyst_left`. Treat lesion names as optional - joined when present, skipped when not -
and the organ's as required (`feldglas.client.seg_mask("labels.seg.nrrd", "kidney_left",
optional=["kidney_cyst_left"])`). A normal-tissue reference, the other way round, takes the organ
alone.

Two more ways a lesion falls outside the region, and what to do about each:
- **A hole in the organ label.** An organ model may carve out tissue that does not look like organ
  (RADAR's own kidney mask held a median 0.79 of the expert-segmented tumor volume, and under half
  in 33 of 148 kidney-tumor scans). Fill the region's holes before testing: a lesion the organ
  surrounds but excludes is a hole in it.
- **A lesion past the contour.** One that bulges out of the organ lies partly outside every organ
  label. Grow the test region by a few mm, up to about one fine-token step (RADAR: 8-10 mm).

And an ERODED test region (as in "Is a region unlike normal tissue?" below) cannot see anything
within the erosion of the surface: test the whole unioned organ as well, reading its surface
tokens with care - they see the neighbors, and normal surfaces vary more between people.

1. For each in-extent token, find the label at its center (sample your label map at that world
   point), or the fraction of its box inside the structure.
2. Keep tokens whose center (or most of whose box) is in the structure.
3. Pool: normalize each token to unit length, average, normalize the average.

With the encoder's head (which ships separately), gate as the head was trained: RADAR's pools every
token whose box TOUCHES the structure, on all lattices. Fed only tokens whose centers are inside,
small structures read wrongly (one normal scan: duodenum "diverticulum" 0.91 from 5 tokens, 0.02
from its 20 touching ones).

**Raw cosines are high everywhere** (any two body tokens are often 0.6-0.8 alike, pooled organs
0.8-0.9), so a table of pooled organs compared directly is nearly uniform. Subtract a shared
reference first. Exactly, per lattice:
- a structure's tokens = the in-extent tokens whose CENTER is in the structure (at least a few -
  `feldglas.client` asks 5);
- `r` = the mean of the unit-length in-extent tokens whose centers lie in ANY structure of your
  label map (lungs included; not an HU rule - the two select different tokens and change which
  organ is nearest which), NOT re-normalized;
- a structure's vector = the mean of its unit-length tokens, NOT re-normalized, minus `r`, and
  THEN set to unit length (a structure that is the whole label map has no direction);
- compare two such vectors by their dot product (cosine).
Compare within one lattice; different lattices rank structures differently.

## Is a region unlike normal tissue? (no head needed)

The raw tokens carry abnormality on their own: no head, no vocabulary, no labels of disease. What
they need is a **reference of normal tissue** - the same organ in a few normal scans. **With one
scan and no normals this does not work**: ranking a scan's tokens against the rest of its own
organ always flags something (the threshold is the scan's own tail), is blind to diffuse disease
(it raises the whole organ), and cannot tell a protocol's effect from disease. Measured
on RADAR's fields, 2026-09-22 (feldglas `docs/embedding-field.md`; the study's EXPLORATION 5.14).

**The reference must match the scans you test**: same scanner, protocol and reconstruction.
Acquisition moves the vectors a lot (noise alone put a low-dose scan's pooled liver vector 1.7x
further from its own full-dose twin than from another patient's - measured with RADAR's head on
simulated low-dose scans), so an unknown from another
protocol looks abnormal everywhere. If every region of a scan is flagged, suspect the protocol
first.

**With a few normal scans (three is enough to start): token by token.**
1. In each normal scan, mark normal tissue: an organ from a segmentation, eroded by about 15 mm
   (the setting measured; RADAR's fine `thickness` is 20 mm) so that every token compared is
   interior, or an ROI you draw.
2. Take the FINE lattice's in-extent tokens, off the extent's boundary, at least half inside that
   region - half of the token's BOX VOLUME, which `feldglas.client` estimates at a grid of points
   inside the box - unit length. These are the reference.
3. Score a token by its distance to normal: `1 - mean cosine to its 5 most similar reference
   tokens`. Nearest neighbors, not a mean: normal tissue is many kinds of token (parenchyma,
   vessels, fissures), and their average is none of them.
4. Set the threshold with the normals themselves: score each normal's tokens against the OTHER
   normals, and take the largest distance any held-out normal token reached.
5. In the scan you test, score the region's tokens - the organ unioned with its lesion segments
   (see "Using a mask"), eroded as the reference was; tokens beyond the threshold are flagged. Join flagged tokens
   into sites by single linkage - two within 15 mm are one site, so a site can chain further - and
   report each site's place (the mean of its tokens' centers) and its largest distance, strongest
   first.

On a public full-dose CT with spheres painted deep in the liver (the scan's own texture kept,
three normals of the same study as reference), every painted lesion was flagged, down to 10 mm
at -40 HU (lesion tokens 0.30-0.74 against a held-out normal maximum of 0.20). The price was two
false sites in the unpainted scan too - one at a vessel and fissure, one 28 mm from the lesion. Treat a site as
a place to look, not a finding.

**Pooling a box or an organ into one vector dilutes a small lesion.** With three normals, 32 mm
boxes found none of the painted lesions: a box's mean barely moved, and the reference's own
person-to-person differences were larger than the lesion's effect. Test tokens, not boxes, until
there are many normals.

**With many normal scans (tens): boxes against the normal's mean.** Sweep the organ with 32 mm
boxes at half-box steps; pool each box as the mean per lattice (each lattice's tokens in the box,
averaged, at unit length; the lattices side by side; the whole at unit length); fit the normals'
mean box vector per organ; score a box by its plain (Euclidean) distance from that mean. On 80
normal donors against 395 patients with expert-segmented tumors, this separated tumor boxes from
clean ones at AUC 0.97 (colorectal metastases), 0.91 (hepatocellular carcinoma) and 0.93 (kidney
tumors), and at one false alarm per scan found 75 %, 82 % and 86 % of lesions. By AUC that beats the
same atlas pooled through RADAR's head in all three; per lesion it is ahead on the liver cancers
and kidney tumors and behind on metastases (the head's 81 %, RADAR's named finding 90 %). With the head, three donors did about as well as eighty (not yet re-measured without it); a few
normals from YOUR site matter more than many from elsewhere. Prefer the plain distance: weighting
by the normals' covariance ("whitening") raises AUC but makes each site's own quirks into false
alarms.

What neither method says is WHAT is there: a deviation from normal is a place to look. Naming a
finding takes the encoder's head.

## License

The embeddings inherit the license of the weights that made them: `provenance.license`.
