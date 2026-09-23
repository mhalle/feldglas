"""feldglas: an instrument for looking at the embedding fields of medical images.

A Feldglas is the old Galilean field glass: two lenses, no prisms, carried in a haversack and
used for scouting - to say where something deserves a closer look before a heavier instrument
comes out. This package is that for learned encoders of medical images: keep the encoder's token
FIELD, decide the region, the question and the reference afterwards, and characterize the
encoder as the measuring instrument it is. The design record is medseg's
docs/radar-idc-validation/EXPLORATION.md.

Importing this package pulls in numpy and rankfield's geometry and nothing heavier: no torch,
no scipy, no modal (tests/test_light_import.py).
"""
from .contract import Embedding, Field, Head, Kernel, Provenance

__version__ = "0.1.0"
__all__ = ["Embedding", "Field", "Head", "Kernel", "Provenance", "__version__"]
