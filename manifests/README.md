# manifests

One JSON file per export set: the ONLY map from a series to its blob on the object store. Blobs
are named by the SHA-256 of their bytes, so without this file a bucket of fields is a bucket of
hashes. Tracked in git on purpose - it names public series ids and digests, and carries no
derived data. Written by `tools/radar_export_modal.py --store ...`, read by `feldglas.remote`.

The shape is the one provender's 0.2 pointer body will have (`files` for garbage collection,
`body` ours), so moving to pointers is one upload. One manifest holds ONE provenance (encoder,
code, weights, preprocessing, license); a different one gets its own file.
