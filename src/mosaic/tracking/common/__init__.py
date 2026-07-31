"""Machinery every integrated tracker shares.

TREx, SLEAP and Lightning Pose differ in what they run and what they emit. They
do not differ in how a run is *driven*: locate the tool in its own environment,
resolve the model to a content digest, mint a run identifier, claim each entry's
working directory, decide per phase whether the recorded marker still proves the
work, run what is stale, record what completed, bridge the output into
``tracks/``, and write a row. That algorithm lives here once.

This package imports ``core`` and nothing from ``behavior``; ``core`` imports
nothing from here. A tracker's own module keeps what is genuinely its own -- the
argv it builds, the settings that define its identity, the converter it bridges
through -- and reaches this package for everything else.
"""

from __future__ import annotations

from mosaic.tracking.common.toolenv import (
    BinMode,
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
    conda_invocation,
    subprocess_env,
    tool_invocation,
)

__all__ = [
    "BinMode",
    "ToolEnv",
    "ToolExitError",
    "ToolNotFoundError",
    "conda_invocation",
    "subprocess_env",
    "tool_invocation",
]
