"""The characterization suite: what any encoder behind the contract is measured by.

The premise of the whole project is that a frozen encoder is a measuring INSTRUMENT, to be
characterized the way imaging physics characterizes one - dynamic range, point spread, response
to a known input, noise, what it reads on normal material - before anyone asks it to find
something. The same probes are then the acceptance test for a successor and the objective for an
encoder trained here (EXPLORATION sections 3.1 and 11).

This is the MAP of those probes, honest about where each one stands:

  "here"     implemented in this package, runs on stored fields
  "medseg"   ran once as a RADAR-specific pilot script in medseg's
             docs/radar-idc-validation/scripts/ (2026-09-20); porting it behind the contract is
             the work, and ``where`` names the script and the result it produced
  "planned"  designed in EXPLORATION, not yet run anywhere

``needs`` says what a probe costs: "fields" (stored token fields, CPU only), "intervention" (edit
the image and re-encode: a GPU, ~0.3 s per edit for RADAR), "pairs" (two acquisitions of one
patient), "labels" (expert masks or clinical tables), "stores" (haversack's segmentation stores).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    name: str
    question: str
    needs: str
    status: str
    where: str = ""


PROBES = (
    # -- the instrument ------------------------------------------------------------------
    Probe("exactness", "does the CPU head reproduce the encoder's own pooled vectors?", "fields", "medseg",
          "radar_instrument_modal.py -> instrument.json (3e-8); tests/test_radar_head.py re-checks it when the pilot's files are present"),
    Probe("latency", "what does one gate cost once a scan is prepared?", "fields", "medseg",
          "radar_instrument_modal.py -> instrument.json (0.2-0.35 ms per box, 0.6-1.3 ms per organ)"),
    Probe("dynamic_range", "which intensities does the input window forbid outright?", "intervention", "planned",
          "EXPLORATION WP1: ground glass, calcium and bone against RADAR's [-300, 400] HU clip"),
    Probe("point_spread", "how far does a local change reach, per lattice and per gate?", "intervention", "medseg",
          "radar_instrument_modal.py -> instrument.json (fine ~20 mm, mid ~40, deep ~80-120)"),
    Probe("pooling", "how far is the head from a uniform mean, and which lattice does it lean on?", "fields", "medseg",
          "radar_pooling_linearity.py -> pooling_linearity.json (cosine 0.46-0.49; 45 % of weight on the deep lattice)"),
    Probe("gateability", "how deep is each structure, against how many tokens lie entirely inside it?", "fields", "here",
          "feldglas.suite.gateability"),
    Probe("gate_occupancy", "how much of a vector is built from tokens that are mostly something else?", "fields", "medseg",
          "radar_gate_occupancy.py -> gate_occupancy.json (32 / 62 / 57 % for liver / spleen / kidney)"),
    Probe("lattice_restricted", "is the lattice a usable scale knob - sharper maps from fine tokens alone?", "labels", "planned",
          "EXPLORATION WP1"),
    Probe("jitter", "how much does a vector move when the lattice falls elsewhere on the same scan?", "intervention", "planned",
          "EXPLORATION WP1 and F3 (with lattice supersampling)"),
    # -- response -------------------------------------------------------------------------
    Probe("response_surface", "what can it see? detectability against size and contrast of a painted lesion", "intervention", "medseg",
          "radar_response_modal.py::cd -> contrast_detail.json (a switch: nothing at +-10 HU, saturated by +-40)"),
    Probe("response_naming", "what does a painted lesion read as, in the encoder's own vocabulary?", "intervention", "medseg",
          "radar_response_naming.py -> response_naming.json"),
    Probe("unknown_parameter", "is a parameter left unknown handled by a bank of templates?", "labels", "medseg",
          "radar_observer.py part B -> observer.json"),
    # -- noise and nuisance ---------------------------------------------------------------
    Probe("variance_ladder", "how far apart are vectors that differ in exactly one way?", "pairs", "medseg",
          "radar_variance_ladder.py -> variance_ladder.json (thickness 0.06, phase 0.10, patient 0.40, site 0.60)"),
    Probe("noise_pairs", "what does dose alone do, with the crop and the gate held fixed?", "pairs", "medseg",
          "radar_dose_pairs.py, radar_response_modal.py::dose -> dose_pairs.json, dose_fixed_gate.json"),
    Probe("noise_curve", "at what injected noise does a region move as far as another patient's?", "intervention", "planned",
          "EXPLORATION WP1, with and without an anti-aliased resample"),
    # -- the unremarkable -----------------------------------------------------------------
    Probe("normal_atlas", "does a normal model from healthy people find the remarkable elsewhere?", "labels", "medseg",
          "radar_donor_atlas.py, radar_observer.py part A -> donor_atlas.json (0.909 pooled vs 0.816), observer.json"),
    # -- variants (EXPLORATION section 12): an axis of the suite, for every encoder --------
    Probe("mirror", "does any layer notice an exact situs inversus (voxels flipped, header kept)?", "intervention", "planned", "V1"),
    Probe("accessory_spleen", "a blob of the patient's own spleen painted into fat: ranks, identity, self-match", "intervention", "planned", "V2"),
    Probe("identity_in_context", "what a token reads as, against what is expected at its place in the frame", "stores", "planned", "V4, after WP5's probes"),
    Probe("correspondence", "can a region be matched between subjects, by anatomy, by embedding, by both?", "stores", "planned", "WP5, V5"),
)


def by_status(status: str) -> tuple[Probe, ...]:
    return tuple(p for p in PROBES if p.status == status)


def probe(name: str) -> Probe:
    for p in PROBES:
        if p.name == name:
            return p
    raise KeyError(f"no probe named {name!r}; known: {[p.name for p in PROBES]}")
