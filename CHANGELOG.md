# Changelog

## [0.1.2] - 2026-09-23

- **rankfield is pinned at v0.3.5, as haversack pins it**, for the reason 0.1.1 gave: uv
  resolves a git dependency's own `[tool.uv.sources]`, so a feldglas pin that differs from
  haversack's makes `haversack[encode]` unresolvable. rankfield 0.3.5 sizes its encoder slabs
  from a memory budget and reads fields straight from a store; feldglas uses only its
  `Geometry`, which is unchanged, so nothing feldglas computes changes.
- The design record notes haversack's phase 2 (encode jobs on the local server) as landed.
- This entry was committed after the `v0.1.2` tag, which does not carry it.

## [0.1.1] - 2026-09-23

- **rankfield is pinned at v0.3.3, as haversack pins it.** uv resolves a git dependency's own
  `[tool.uv.sources]`, so feldglas 0.1.0's v0.3.2 and haversack's v0.3.3 were two URLs for one
  package and `haversack[encode]` could not be installed at all. v0.3.3 is v0.3.2's code with
  duckn pinned at v0.5.1, which feldglas already uses; nothing feldglas computes changes.
- The design record names RADAR's encoder `radar:pretrain` (after its checkpoint), not
  `radar:generalist`.

## [0.1.0] - 2026-09-23

The first release, so that haversack can pin feldglas for the field writer when encoding moves
into haversack (docs/embedding-field.md, "Encoding moves into haversack").

- **The embedding field format** (`feldglas.store`): `feldglas-field` 0.2, a zarr v3 group in a
  zip with stored entries, one duckn-placed array per lattice, the `embedding` extension 0.1
  (comparability key, kernel, extent, support offset, group), int8 through
  `embedding.linear_along_axis`, the client guide packed in as `README.md`. 0.1 (`.npz`) still reads.
- **The receiving end** (`feldglas.client`): open a field (a path or an http(s) URL), masks on any
  grid from `.seg.nrrd` files (haversack's labelmap style, 3D Slicer's layered style, LPS or RAS),
  organ vectors, and normal-tissue references (`Reference.build/score/save/load`) that flag what is
  unlike normal tissue token by token.
- **The `feldglas` command** (extra `client`): `health`, `info`, `vectors`, `reference build`,
  `score`; `--json` everywhere, one line or one JSON object on failure, warnings for what a result
  cannot vouch for, labels checked against the field's recorded CT grid.
- **Files by URL** (`feldglas.fetch`): obstore, an ETag-revalidated cache that cannot pin the wrong
  version, a token only to its own origin.
- Four rounds of adversarial review, with an agent driving the command in the last.
