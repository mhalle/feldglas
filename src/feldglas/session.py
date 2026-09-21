"""One scan, ready to be asked things: a field, a head, and gates taken BY NAME from a label map.

This is the surface a viewer and a server-side engine both call (2026-09-20). It holds what is
done ONCE per scan - the head's keys and values for every token (~0.1 s), the label map pulled
onto the model grid, each structure's token occupancy - so that every question afterwards is the
cheap part: a gate, a pooling, a dot product (0.2-0.4 ms for a 32 mm box).

Everything a question can be is ONE operation here, because the study found it to be one
(EXPLORATION 5.9, 5.10): a word (a finding's text pair), an example (a displacement between two
regions), a description (a painted template) and "unlike healthy people" (a normal model) are all
a direction or a distance applied to a gated vector. So `Session.score` takes a direction, and the
callers decide where it came from.

Gates come from a label map by structure NAME - haversack's segmentation, not the encoder's own
mask, which for RADAR misses a fifth of kidney tumor volume and cannot tell a left kidney from a
right one. RADAR pools under an organ QUERY, so a structure needs to know whose question it is
asked (`adapters.radar.query_for`); a head with one question ignores it.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Callable

import numpy as np

from .contract import Field
from .gate import Gate, _rule, occupancies
from .labels import GridLabels, LabelMap


@dataclass
class Session:
    field: Field
    head: object
    labels: GridLabels | None = None
    query_of: Callable[[str], str] | None = None      # structure name -> the head's query (None: the head has one)
    _prepared: object = None
    _occ: dict = _field(default_factory=dict)

    @classmethod
    def open(cls, field: Field, head, labels: LabelMap | GridLabels | None = None, query_of=None) -> "Session":
        if isinstance(labels, LabelMap):
            labels = labels.on_grid(field.grid)
        if labels is not None and tuple(labels.values.shape) != tuple(field.grid.shape):
            raise ValueError(f"labels {labels.values.shape} are not on the model grid {field.grid.shape}: pass the LabelMap")
        return cls(field, head, labels, query_of, head.prepare(field.all_tokens()))

    # -- gates ------------------------------------------------------------------------------
    def mask(self, *structures: str) -> np.ndarray:
        if self.labels is None:
            raise ValueError("this session has no label map: open it with labels=, or gate with a mask of your own")
        return self.labels.mask(*structures)

    def occupancy(self, *structures: str) -> list[np.ndarray]:
        """Token occupancy of a (union of) named structure(s), computed once and kept."""
        if structures not in self._occ:
            self._occ[structures] = occupancies(self.field, self.mask(*structures))
        return self._occ[structures]

    def gate(self, *structures: str, rule="any") -> Gate:
        """From the kept occupancies: a pass over the model grid costs ~85 ms on a real field, and
        a structure is gated many times (its vector, its findings, soft and hard)."""
        keep = _rule(rule)
        idx, occ, lat = [], [], []
        for j, (o, off) in enumerate(zip(self.occupancy(*structures), self.field.offsets[:-1])):
            hit = np.flatnonzero(keep(o))
            idx.append(hit + off); occ.append(o[hit]); lat.append(np.full(hit.size, j))
        return Gate(np.concatenate(idx).astype(np.int64), np.concatenate(occ).astype(np.float32), np.concatenate(lat).astype(np.int8))

    def query(self, *structures: str, query=None):
        if query is not None or self.query_of is None:
            return query
        qs = {self.query_of(s.rstrip("*") if s.endswith("*") else s) for s in structures}
        if len(qs) != 1:
            raise ValueError(f"{structures} are asked under different queries {sorted(qs)}: pass query= to choose one")
        return qs.pop()

    # -- vectors and scores -----------------------------------------------------------------
    def vector(self, *structures: str, query=None, rule="any", soft: bool = False) -> np.ndarray:
        """The pooled vector of the named structure(s). ``soft`` hands the head log-occupancy, so a
        token half inside counts half as much instead of exactly as much."""
        g = self.gate(*structures, rule=rule)
        if not len(g):
            raise ValueError(f"no token of this field touches {structures} under rule {rule!r}")
        return self.head.pool(self._prepared, g.index, self.query(*structures, query=query), g.soft_bias() if soft else None)

    def score(self, vector: np.ndarray, direction: np.ndarray, reference: np.ndarray | None = None) -> float:
        """``(vector - reference) . direction`` - a word, an example or a description, all alike."""
        v = np.asarray(vector, np.float64) - (0.0 if reference is None else np.asarray(reference, np.float64))
        return float(v @ np.asarray(direction, np.float64))

    def findings(self, *structures: str, text: dict, query=None, top: int | None = None) -> list[tuple[str, float]]:
        """RADAR's own vocabulary for the structure: every finding of its organ, as probabilities,
        highest first. ``text`` is ``{"Organ_Finding": (negative, positive)}`` (English keys)."""
        q = self.query(*structures, query=query)
        v = self.vector(*structures, query=q)
        rows = [(name, self.head.score(v, pair)) for name, pair in text.items() if name.lower().startswith(str(q).lower() + "_")]
        rows.sort(key=lambda r: -r[1])
        return rows[:top] if top else rows

    def table(self, text: dict | None = None, min_voxels: int = 400, top: int = 3) -> dict:
        """Every named structure the head has a question for: its query, size, vector and top findings."""
        out = {}
        vox_ml = float(np.prod(self.field.grid.spacing)) * 1e-3
        for name in self.labels.present(min_voxels):
            try:
                q = self.query(name)
            except KeyError:
                continue
            row = {"query": q, "ml": round(float(self.mask(name).sum()) * vox_ml, 1), "vector": self.vector(name, query=q)}
            if text is not None:
                row["findings"] = [(n, round(p, 3)) for n, p in self.findings(name, text=text, query=q, top=top)]
            out[name] = row
        return out

    # -- maps -------------------------------------------------------------------------------
    def sweep(self, *structures: str, size_mm: float = 32.0, query=None, rule="any"):
        """Boxes over the structure(s) and a vector for each: what every map is drawn from."""
        from .suite.normal_atlas import pool_sweep, sweep
        m = self.mask(*structures)
        sw = sweep(self.field, m, size_mm)
        return sw, pool_sweep(self.field, self.head, self._prepared, sw, m, self.query(*structures, query=query), rule=rule)

    def volume(self, sw, scores: np.ndarray, within: np.ndarray | None = None, reduce: str = "max") -> np.ndarray:
        """Box scores painted back onto the model grid (NaN where no box reaches): each voxel takes
        the ``max`` (a detection map) or ``mean`` of the boxes covering it, optionally only
        ``within`` a mask. The grid is the field's, so rankfield's geometry places it on the patient."""
        out = np.full(self.field.grid.shape, -np.inf if reduce == "max" else 0.0, np.float32)
        n = np.zeros(self.field.grid.shape, np.uint16)
        for lo, hi, s in zip(sw.lo, sw.hi, scores):
            if not np.isfinite(s):
                continue
            sl = tuple(slice(a, b) for a, b in zip(lo, hi))
            out[sl] = np.maximum(out[sl], s) if reduce == "max" else out[sl] + s
            n[sl] += 1
        out = np.where(n > 0, out if reduce == "max" else out / np.maximum(n, 1), np.nan).astype(np.float32)
        if within is not None:
            out[~within] = np.nan
        return out
