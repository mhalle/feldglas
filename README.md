# feldglas

An instrument for looking at the embedding fields of medical images.

A *Feldglas* is the old Galilean field glass: two lenses, no prisms, carried in a haversack and
used for scouting - to say where something deserves a closer look before a heavier instrument
comes out. This package is that for learned encoders of medical images. It keeps an encoder's
**token field** (its per-region vectors on a lattice over the scan) and decides everything else
afterwards: the region, the question, the reference. And it treats the encoder as what it is, a
measuring instrument, to be characterized - dynamic range, point spread, response, noise, what
it reads on normal tissue - before anyone asks it to find something.

> *Tell me what is unremarkable, and I will show you the remarkable.*

**Status: a scaffold (0.0.1), research use.** The contract, gates, the RADAR head, the observers
and one suite probe work and are tested; the second encoder is a stub; most of the suite is a
map of pilot scripts still living in the design record. The design record is
`../medseg/docs/radar-idc-validation/EXPLORATION.md`.

## What is here

| Module | What it is |
| --- | --- |
| `feldglas.contract` | what any encoder must hand over: a `Field` (token lattices over a model grid, placed in the world in rankfield's `Geometry`), a `Head` (tokens -> one vector), and `Provenance` with **the license the arrays inherit** |
| `feldglas.store` | a field on disk (`feldglas-field 0.1`), with a version gate |
| `feldglas.gate` | which tokens a region owns and *how much of each*: the raster rule, interior-only, a threshold, soft gates, boxes in millimetres |
| `feldglas.adapters.radar` | RADAR behind the contract: its head in numpy (one attention layer and a Linear per organ; matched the GPU to 3e-8, 0.2 ms per gate), finding scores, the pilot's fields |
| `feldglas.adapters.null` | the null model - an anatomy-only encoder through the same suite. A stub, with its plan |
| `feldglas.heads` | `MeanPoolHead`, for encoders that bring no head |
| `feldglas.observe` | detection theory on the channels: a `NormalModel` (the unremarkable), Hotelling templates, a bank for an unknown parameter |
| `feldglas.remote` | fields on a shared object store: the blobs are [provender](https://github.com/mhalle/provender)'s, the manifest that names them is ours |
| `feldglas.suite` | the characterization suite: a registry of probes and where each stands; `gateability` runs here |
| `tools/radar_export_modal.py` | the one GPU step: CTs -> RADAR token fields with exact geometry, on Modal |

## Quick start

```bash
uv sync --extra test
uv run pytest -q
```

With a RADAR head and fields exported (they are **not distributed**: see below):

```python
from feldglas.adapters import radar
from feldglas.gate import select
from feldglas.store import read_field
from feldglas.paths import encoder_dir

head = radar.RadarHead.load()                                   # ~/.cache/feldglas/radar/head.npz
field = read_field(next((encoder_dir("radar") / "fields").glob("*.npz")))
prepared = head.prepare(field.all_tokens())                     # once per scan, ~0.1 s

liver = field.native_mask == radar.ORGANS.index("liver") + 1
organ = head.pool(prepared, select(field, liver).index, "liver")            # the published gate
clean = select(field, liver, "interior")                                    # tokens entirely inside
soft = select(field, liver)
vector = head.pool(prepared, soft.index, "liver", bias=soft.soft_bias())    # occupancy-weighted

# a sweep: precompute the organ once, then a box costs ~0.3 ms to select, pool and score
from feldglas.gate import box_gate, occupancies
within = occupancies(field, liver)
region = head.pool(prepared, box_gate(field, (40, 160, 120), 32.0, within=within).index, "liver")
```

`tools/check_geometry.py` checks an exported field's placement in the patient against an
independent source (haversack's statistics); on the first two fields it agreed to 1-3 mm.

## What stays out of this repository

The code is Apache-2.0. **Anything derived from an encoder's weights is not**: RADAR's weights
are CC BY-NC-SA 4.0, so its pooling head, every token field and every vector pooled from one are
research-use only and share-alike. They live under `~/.cache/feldglas` (or `$FELDGLAS_CACHE`),
never in git, and every field file carries its license in its own metadata.

## Where things belong in the family

- **haversack** segments, across ecosystems and engines, and keeps what each model said as
  fields. feldglas consumes its masks, its distance fields and (soon) its stores; gates resolve
  against them.
- **rankfield** owns the stored form of a class field and the `Geometry` used here. Codecs for
  read-out and token fields belong there, not here.
- **feldglas** owns the encoder contract, pooling, the model of the unremarkable, the suite and
  the tools built on them.
