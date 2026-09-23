"""The encoder contract: what any encoder must hand over, and nothing about which one it is.

Everything downstream of an encoder - gates, pooling, the model of the unremarkable, templates,
retrieval, maps, the characterization suite - sees only what is declared here. RADAR is the
first adapter; its successors and anything trained here are further ones. The contract was
written down on 2026-09-20 from what a day of experiments actually used (medseg's
docs/radar-idc-validation/EXPLORATION.md, sections 2, 10 and 11), not from what an encoder might
someday offer, and it should grow the same way: from a second adapter that needs something, not
from foresight.

An encoder supplies

  a FIELD      one or more token lattices over a MODEL GRID, each token a fixed box of model
               voxels, with the geometry that ties the grid to the patient - in the family's
               one form, rankfield's ``Geometry`` (duckn's, which is NRRD's), so a scalar read
               off a lattice is an array the family's restore already knows how to place.
  a HEAD       how a set of tokens becomes one vector (``pool``). RADAR's is one attention
               layer and a Linear; an encoder with no learned head gets ``heads.MeanPoolHead``.
  PROVENANCE   what made the arrays, and the LICENSE THEY INHERIT. The code here is Apache-2.0;
               a token field derived from CC BY-NC-SA weights is not, and says so on itself.

The encode step itself (image -> Field) needs the encoder's weights and usually a GPU, so it
lives with each adapter's tooling (``tools/``), not behind a method here: nothing in this
package imports torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Protocol, runtime_checkable

import numpy as np
from rankfield.geometry import Geometry

Kernel = tuple[int, int, int]


@dataclass(frozen=True)
class Provenance:
    """What produced a field's arrays. ``license`` is the license of the WEIGHTS the arrays
    derive from - it travels with every file written, because it is the one fact about a token
    field that decides where the file may go."""

    encoder: str                    # the adapter's name: "radar"
    code: str = ""                  # upstream code revision
    weights: str = ""               # checkpoint name (and digest, when known)
    preprocessing: str = ""         # the adapter's own preprocessing version
    license: str = ""               # e.g. "CC-BY-NC-SA-4.0"
    source: str = ""                # the series' identity, e.g. an IDC crdc_series_uuid
    extra: dict = _field(default_factory=dict)
    #: The input CT, so a client holding a mask on the CT's own grid knows it is the same image
    #: (docs/embedding-field.md, "Provenance"; 2026-09-22). Keys, each optional: ``identity``
    #: (content digest, DICOM UIDs, collection), ``grid`` in duckn's form (``shape``,
    #: ``directions`` one LPS row per array axis, ``origin``), ``acquisition``, ``attribution``.
    #: Held here until duckn's provenance extension exists; it moves there when it does.
    input: dict = _field(default_factory=dict)


@dataclass(frozen=True)
class Embedding:
    """What the vectors ARE, as a client comparing them must know (docs/embedding-field.md,
    2026-09-22). Model and weights come from :class:`Provenance`; this adds the rest of the
    comparability key and each lattice's measured facts. Every per-lattice tuple is empty (not
    known) or one entry per lattice; an entry may be None.

    ``layers``       which encoder layer each lattice is (with model and weights: two vectors
                     compare only if all agree).
    ``stage``        ``raw`` tokens need a head; ``projected`` ones are in ``projects_to``'s space.
    ``receptive_mm`` a token's measured extent along the lattice's (Z, Y, X) axes - duckn's core
                     ``thickness``: a map read off the lattice is no sharper than this.
    ``support_offset_mm`` where a token's evidence is centered against its sample, same axes
                     (RADAR's look offset). ``space_origin`` stays the exact grid.
    """

    layers: tuple[str, ...] = ()
    stage: str = "raw"
    metric: str = "cosine"
    normalized: bool = False
    projects_to: dict | None = None
    receptive_mm: tuple[tuple[float, float, float] | None, ...] = ()
    support_offset_mm: tuple[tuple[float, float, float] | None, ...] = ()

    def lattice(self, name: str, j: int):
        v = getattr(self, name)
        return v[j] if v else None


@dataclass
class Field:
    """Token lattices over one model grid.

    ``tokens[j]`` is ``(N_j, C)``, row-major over lattice ``j``'s ``(Z, Y, X)``; ``kernels[j]``
    is the box of model voxels one of its tokens covers; ``grid`` places the MODEL grid in the
    world. A lattice's own geometry is derived (:meth:`lattice_geometry`), never stored - its
    sample (0, 0, 0) is the CENTER of the first token's box, which is what makes a per-token
    scalar an ordinary cell-centered array.

    ``exact_geometry`` is False for fields written before the exporter recorded its crop and
    resample (the 2026-09-20 pilot): their ``grid`` has the right shape and spacing and an
    arbitrary origin and orientation, which is enough to gate and pool and not enough to draw
    on the patient.
    """

    tokens: list[np.ndarray]
    kernels: list[Kernel]
    grid: Geometry
    provenance: Provenance
    native_mask: np.ndarray | None = None       # (Z, Y, X) on the model grid: the encoder's own labels
    native_labels: tuple[str, ...] = ()         # label value v is native_labels[v - 1]; 0 is background
    exact_geometry: bool = True
    embedding: Embedding = _field(default_factory=Embedding)

    def __post_init__(self):
        if len(self.tokens) != len(self.kernels) or not self.tokens:
            raise ValueError(f"{len(self.tokens)} token arrays for {len(self.kernels)} kernels")
        self.kernels = [tuple(int(v) for v in k) for k in self.kernels]
        for j, (t, k) in enumerate(zip(self.tokens, self.kernels)):
            if any(s % kk for s, kk in zip(self.grid.shape, k)):
                raise ValueError(f"lattice {j}: kernel {k} does not tile the model grid {self.grid.shape}")
            want = int(np.prod(self.lattice_shape(j)))
            if t.ndim != 2 or t.shape[0] != want:
                raise ValueError(f"lattice {j}: {t.shape} tokens for a lattice of {self.lattice_shape(j)} = {want}")
        for name in ("layers", "receptive_mm", "support_offset_mm"):
            n = len(getattr(self.embedding, name))
            if n not in (0, self.lattices):
                raise ValueError(f"embedding.{name}: {n} entries for {self.lattices} lattices")
        if self.native_mask is not None and tuple(self.native_mask.shape) != tuple(self.grid.shape):
            raise ValueError(f"native mask {self.native_mask.shape} is not on the model grid {self.grid.shape}")

    # -- shape --------------------------------------------------------------------------
    @property
    def lattices(self) -> int:
        return len(self.tokens)

    @property
    def widths(self) -> tuple[int, ...]:
        """Channels per lattice. RADAR projects every lattice to 256; an encoder with no learned
        projection hands over its skips as they are (the null model: 128 / 256 / 320)."""
        return tuple(int(t.shape[1]) for t in self.tokens)

    @property
    def uniform(self) -> bool:
        return len(set(self.widths)) == 1

    @property
    def channels(self) -> int:
        """The width of :meth:`all_tokens`: the lattices' common width, or their SUM when they
        differ (each lattice in its own block of channels)."""
        return self.widths[0] if self.uniform else sum(self.widths)

    def lattice_shape(self, j: int) -> tuple[int, int, int]:
        return tuple(s // k for s, k in zip(self.grid.shape, self.kernels[j]))

    @property
    def offsets(self) -> np.ndarray:
        """Where each lattice starts in the ONE index space gates and heads use: the lattices
        concatenated in the order given. ``offsets[-1]`` is the total."""
        return np.cumsum([0] + [int(np.prod(self.lattice_shape(j))) for j in range(self.lattices)])

    def all_tokens(self, dtype=np.float32) -> np.ndarray:
        """Every token, lattices concatenated - what a head's ``prepare`` takes.

        Lattices of different widths (2026-09-22, the null model - the contract's first growth
        from a second adapter) are laid side by side in CHANNELS as well as rows: lattice ``j``'s
        tokens fill columns ``block(j)`` and are zero elsewhere, so no two lattices' channels are
        ever added together - their channels mean different things. ``heads.LatticeMeanHead``
        pools such a field block by block."""
        if self.uniform:
            return np.concatenate([np.asarray(t, dtype) for t in self.tokens])
        out = np.zeros((int(self.offsets[-1]), self.channels), dtype)
        for j, t in enumerate(self.tokens):
            out[self.offsets[j]:self.offsets[j + 1], self.block(j)] = t
        return out

    def block(self, j: int) -> slice:
        """Lattice ``j``'s columns in :meth:`all_tokens` (every column when the widths agree)."""
        if self.uniform:
            return slice(0, self.channels)
        lo = sum(self.widths[:j])
        return slice(lo, lo + self.widths[j])

    # -- place --------------------------------------------------------------------------
    def lattice_geometry(self, j: int) -> Geometry:
        """Lattice ``j`` as a cell-centered grid in the world: each step is a kernel's worth of
        model voxels, and sample (0, 0, 0) sits at the center of the first token's box."""
        k = self.kernels[j]
        first = [(kk - 1) / 2.0 for kk in k]                      # model index of the first box's center
        return Geometry(shape=self.lattice_shape(j),
                        directions=tuple(tuple(float(v) * kk for v in row) for row, kk in zip(self.grid.directions, k)),
                        origin=tuple(float(v) for v in self.grid.world(first)))

    def token_centers(self, j: int) -> np.ndarray:
        """``(N_j, 3)`` world positions (LPS mm) of lattice ``j``'s tokens, in token order."""
        idx = np.stack(np.meshgrid(*[np.arange(s) for s in self.lattice_shape(j)], indexing="ij"), -1).reshape(-1, 3)
        return self.lattice_geometry(j).world(idx)


@runtime_checkable
class Head(Protocol):
    """How a set of tokens becomes one vector.

    ``prepare`` does once per scan whatever does not depend on the gate (RADAR: the attention's
    keys and values - 0.1 s); ``pool`` is then cheap enough to call thousands of times (RADAR:
    0.2 ms for a 32 mm box). ``index`` addresses the concatenated lattices (``Field.offsets``).
    ``query`` names WHOSE question is asked where the head has several (RADAR: one learned query
    and one projection per organ) and is ignored by a head that has one. ``bias`` is added to the
    pooling logits per token - the natural way to hand a soft gate (log occupancy) to a head -
    and a head with no logits treats it as log-weights.
    """

    queries: tuple[str, ...]

    def prepare(self, tokens: np.ndarray): ...

    def pool(self, prepared, index: np.ndarray, query: str | int | None = None,
             bias: np.ndarray | None = None) -> np.ndarray: ...
