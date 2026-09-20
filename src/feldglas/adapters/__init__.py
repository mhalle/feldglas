"""One module per encoder behind the contract. An adapter supplies a ``Head``, the constants a
``Field`` of that encoder needs (kernels, native labels, provenance), and whatever reads its
side files; the encode step that PRODUCES fields lives in ``tools/`` with the encoder's
environment, because it needs the weights and usually a GPU."""
