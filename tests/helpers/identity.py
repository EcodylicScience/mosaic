"""The digest a feature run is named by, computed from its parameters alone."""

from __future__ import annotations

from mosaic.core.params import Params
from mosaic.core.pipeline._utils import hash_params


def run_id_digest(params: Params) -> str:
    """Return the digest naming a run of *params* that reads no inputs.

    The hashable has the shape ``compute_run_id`` builds over every frame, so a
    field that moves this digest moves a real run's identifier too.
    """
    return hash_params(
        {"_params": params.identity_dump(), "_inputs": {}, "_frame_range": [None, None]}
    )
