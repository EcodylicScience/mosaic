"""What a dataset holds: every computed artifact, its identity and its coverage.

Nothing in the stack answered "what has been computed in this dataset, with what
params, over which sequences". ``mosaic sequences`` is a tracks listing wearing a
sequences name, ``mosaic runs``/``status`` are job-log surfaces reporting
*attempts*, and ``mosaic features list`` is the registry -- what the installation
knows how to compute, not what this dataset has.

This package is the answer, and it is deliberately a **read**. Truth for
artifacts lives on disk: the ``index.csv`` files plus the files themselves, on
any filesystem. Every view built here is a cache, is never authoritative, and is
never written anywhere it could be mistaken for truth.

This module is the only public import path; the submodules are its
implementation.
"""

from __future__ import annotations

from .params import (
    LABELS_EDGE,
    TRACKS_EDGE,
    ParamsState,
    ResolvedRef,
    RunParams,
    RunParamsRead,
    ScopeBlock,
    read_run_params,
)

__all__ = [
    "LABELS_EDGE",
    "ParamsState",
    "ResolvedRef",
    "RunParams",
    "RunParamsRead",
    "ScopeBlock",
    "TRACKS_EDGE",
    "read_run_params",
]
