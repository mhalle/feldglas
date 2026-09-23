# An embedding field: how to read and use this file

This zip is an **embedding field**: what an image encoder computed from one CT, placed in the
patient's world. It is packed into every field as `README.md`; the design record behind it is
feldglas `docs/embedding-field.md`. Everything below can be done on a CPU with numpy and zarr.

## What it holds

One or more **lattices of tokens**. A token is a vector describing a box of the patient; the
lattices are different layers of the encoder, coarse to fine, not a pyramid of one image. Tokens are
compared by the `metric` each lattice states (cosine so far). They are `raw` unless `stage` says
otherwise: raw tokens are the encoder's own features, in no shared text space, with no labels.

There is **no mask** in the field. Which tokens belong to which organ is for you to decide, with
any mask on any grid (a segmentation of the CT, a drawn region).

## Reading it

A **zarr v3 group inside a zip** with stored entries (chunks are zstd). Open it with zarr's
`ZipStore(path, mode="r")`, or read the JSON and chunks from the zip yourself.

- The root `zarr.json`'s attributes hold `duckn` metadata. Its `extensions.embedding`:
  - `group.members`: the lattice arrays, in order.
  - `version` is this extension's version; `format_version` is the file layout's.
  - `data_box`: the half-open box of MODEL-grid voxels (`lo`, `hi`) that held image data (see
    "The model grid" and "Which tokens saw the scan").
  - `provenance`: `encoder`, `weights`, `license`, and `input` - the input CT's identity and its
    `grid` **as the CT file was delivered** (`shape`; `directions`, one LPS row per array axis,
    each row a full step - its length is the spacing, not a unit vector; `origin`; for a NIfTI,
    array axes are nibabel's i, j, k). Compare this grid with your CT's to know a mask on it
    belongs to this image. `identity` is what the encoder was told about the series (it may be
    only an id); the grid is the check you can make yourself.
  - `provenance.extra` is the encoder's internal record (its crop, resample, timings) - not an
    interface; do not depend on its keys.
- Each lattice is an array of shape `(A0, A1, A2, C)`, C order: `arr[i0, i1, i2, :]` is one token.

## Where a token is

Coordinates are **LPS millimeters**: x increases to the patient's Left, y to Posterior, z to
Superior (DICOM's convention). NIfTI affines as nibabel reports them are RAS: negate x and y.

For a lattice, with `o = space_origin` and `d_a = axes[a].space_direction`:

    center of token (i0, i1, i2) = o + i0 * d_0 + i1 * d_1 + i2 * d_2

Do not assume which anatomical direction an index axis runs: read `space_direction` (the RADAR and
TotalSegmentator encoders store (Z, Y, X), and their Y may run anterior). The exact spacing is
`|d_a|`. `centering: "cell"`: a token covers half a step either side of its center. The three
`space` axes are followed by a `list` axis: the channels.

## The model grid

The encoder read the CT resampled onto its own MODEL grid; each token is a box of `kernel` model
voxels (in the lattice's axis order). For any lattice, the model grid has step `d_a / kernel[a]`
along each axis, and its first voxel's center is at

    o - sum_a ((kernel[a] - 1) / 2) * d_a / kernel[a]

Every lattice gives the same model grid. `data_box` is in its voxel indices.

## Stored values: float, or int8 with a transform

A lattice is stored either as floats (the tokens as they are) or as **int8**. An int8 lattice's
`duckn.value_transforms` holds one entry, `embedding.linear_along_axis`, with `axis` (3: the
channel axis), and `slope` and `intercept`, one per channel. Decode it in float32:

    tokens = int8_values.astype(float32) * slope[channel] + intercept[channel]

This transform is defined here, not by duckn itself (yet). A duckn reader that does not know it
treats the values' meaning as unknown - **never use the raw int8 values as tokens**: each channel
has its own scale. Decoded tokens are within half a step of the originals (cosine to them about
0.9998 or better).

## Which tokens saw the scan

Encoders pad their input, so a lattice usually extends past the scan - often by a third or more.
Each lattice's `extensions.embedding.extent` (`lo`, `hi`, half-open token indices) holds the tokens
whose box contains any voxel of `data_box` - image the encoder was given. **Drop tokens outside
it**: they saw padding only, and can look unlike anything real (a padded deep token's norm can
exceed the body's). Tokens in the first or last row of the extent saw partly padding and are less
reliable than interior ones.

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

## Using a mask

1. For each in-extent token, find the label at its center (sample your label map at that world
   point), or the fraction of its box inside the structure.
2. Keep tokens whose center (or most of whose box) is in the structure.
3. Pool: normalize each token to unit length, average, normalize the average.

**Raw cosines are high everywhere** (any two body tokens are often 0.6-0.8 alike, pooled organs
0.8-0.9), so a table of pooled organs compared directly is nearly uniform. Subtract a shared reference first - the mean of
all in-extent tokens of that lattice, or of the body's - then compare by cosine. Compare within one
lattice; different lattices rank structures differently.

## License

The embeddings inherit the license of the weights that made them: `provenance.license`.
