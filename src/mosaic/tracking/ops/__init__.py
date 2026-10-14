"""Built-in tracking ops.

Each op module registers its op when imported, and this file imports none of
them. :func:`~mosaic.tracking.registration.register_ops` imports every one. A
submodule such as ``prepared_datasets`` therefore imports without loading the
trainers and the trackers.
"""
