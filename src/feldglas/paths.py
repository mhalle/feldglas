"""Where derived data lives: outside the repository, always.

A pooling head, a token field and every vector pooled from one derive from an encoder's weights
and inherit their license (RADAR's: CC BY-NC-SA 4.0), and a field is tens of megabytes. Neither
belongs in git or in a synced folder. ``$FELDGLAS_CACHE`` overrides the default.
"""
from __future__ import annotations

import os
import pathlib


def cache_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("FELDGLAS_CACHE") or pathlib.Path.home() / ".cache" / "feldglas")


def encoder_dir(encoder: str) -> pathlib.Path:
    """``<cache>/<encoder>/`` - its ``head.npz`` and its ``fields/<source>.npz``."""
    return cache_dir() / encoder


def pilot_dir() -> pathlib.Path:
    """The 2026-09-20 pilot's files (medseg's radar_instrument_modal.py): six token fields and
    the RADAR head, written before this package existed. ``adapters.radar.read_pilot_field`` reads them."""
    return pathlib.Path.home() / ".cache" / "radar-idc-validation"
