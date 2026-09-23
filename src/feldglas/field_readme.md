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
  - `data_box`: the half-open box of MODEL-grid voxels (`lo`, `hi`) that held the scanned image.
  - `provenance`: `encoder`, `weights`, `license`, and `input` - the input CT's identity and its
    `grid` **as the CT file was delivered** (`shape`, `directions` one LPS row per array axis,
    `origin`; for a NIfTI, array axes are nibabel's i, j, k). Compare this grid with your CT's to
    know a mask on it belongs to this image.
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

## Which tokens saw the scan

Encoders pad their input, so a lattice usually extends past the scan - often by a third or more.
Each lattice's `extensions.embedding.extent` (`lo`, `hi`, half-open token indices) holds the tokens
whose box contains any scanned voxel. **Drop tokens outside it**: they saw padding only, and can
look unlike anything real (a padded deep token's norm can exceed the body's). Tokens in the first
or last row of the extent saw partly padding and are less reliable than interior ones.

## What each lattice says about itself

On `extensions.embedding` of each lattice:
- `space`: `model`, `weights`, `layer`, `stage`. Two vectors are comparable only if all four agree.
- `metric`, `normalized` (false: normalize yourself), `kernel` (model voxels per token), `group`.
- `support.offset` (mm along the lattice's three axes, optional): where a token's evidence is
  centered MINUS where its box is drawn; positive toward increasing index. The grid is exact - the
  offset says where the encoder actually looks.

And in core duckn, per space axis:
- `thickness` (mm, optional): the full width of a token's measured reach along that axis. A map
  read off the lattice is no sharper than this, whatever its spacing.

## Using a mask

1. For each in-extent token, find the label at its center (sample your label map at that world
   point), or the fraction of its box inside the structure.
2. Keep tokens whose center (or most of whose box) is in the structure.
3. Pool: normalize each token to unit length, average, normalize the average.

**Raw cosines are high everywhere** (any two body tokens are often 0.6-0.8 alike), so a table of
pooled organs compared directly is nearly uniform. Subtract a shared reference first - the mean of
all in-extent tokens of that lattice, or of the body's - then compare by cosine. Compare within one
lattice; different lattices rank structures differently.

## License

The embeddings inherit the license of the weights that made them: `provenance.license`.
