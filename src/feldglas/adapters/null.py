"""The null model: an encoder that was never shown a report. NOT BUILT YET.

Why it exists (EXPLORATION section 11). RADAR's representation cares about pathology because
400,000 radiology reports taught it to. Whether that can be had another way decides how seriously
to take an encoder of our own that has no such reports - and the cheapest way to find out is to
put a network that has only ever learned ANATOMY through the same suite: the encoder activations
of the TotalSegmentator network haversack already runs. If a donor atlas on those features finds
tumors nearly as well as RADAR's (0.909 pooled, EXPLORATION 5.1), report supervision matters less
than it looks; if it lands near chance, the reports are what made the features pathology-aware
and a substitute has to be found. It is also the second adapter, which is what keeps the contract
honest - a contract with one implementation is that implementation's API.

Plan: a forward hook on the nnU-Net encoder inside haversack's ``network`` module, the skips at
two or three depths as the lattices (kernels = the cumulative strides), the model grid's geometry
from haversack's own frame (already rankfield's form), pooling by ``heads.MeanPoolHead`` since
there is no learned query, provenance carrying the weights' own license from haversack's
attribution table. Sliding-window inference tiles the volume, so tokens near a tile's edge see
less context than tokens at its centre - measure that with the suite's point-spread probe before
trusting a map.
"""
NAME = "null-totalsegmentator"


def export_field(*args, **kwargs):
    raise NotImplementedError("the null-model adapter is a stub: see this module's docstring for the plan")
