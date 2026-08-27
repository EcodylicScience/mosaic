from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from mosaic.core.params import Declared
from mosaic.core.strict_model import StrictModel

__all__ = [
    "InterpolationConfig",
    "PoolConfig",
    "SamplingConfig",
]

_LINEAR_INTERP_LIMIT_DESCRIPTION = (
    "Maximum run of consecutive missing values filled by linear interpolation."
)

_EDGE_FILL_LIMIT_DESCRIPTION = (
    "Maximum run of consecutive missing values forward-filled, then "
    "back-filled, after interpolation leaves them unresolved."
)

_MAX_MISSING_FRACTION_DESCRIPTION = (
    "Fraction of missing columns above which a row is dropped entirely."
)

_FPS_DEFAULT_DESCRIPTION = (
    "Frame rate used when the data's fps column is absent or does not "
    "resolve to exactly one value."
)

_SMOOTH_WIN_DESCRIPTION = (
    "Length of the moving-average window applied to smooth position and "
    "angle values before feature computation. A value of 1 or less "
    "disables smoothing."
)

_POOL_SIZE_DESCRIPTION = (
    "Number of candidates collected before selecting the final templates. "
    "Unset sets the pool to the target template count, so selection makes "
    "no reduction."
)

_ALLOCATION_DESCRIPTION = (
    "How the per-entry quota for the pool is computed. reservoir performs "
    "weighted reservoir sampling in one pass. exact counts rows first, "
    "then samples a second pass with proportional quotas."
)

_MAX_ENTRY_FRACTION_DESCRIPTION = (
    "Cap on one entry's contribution to the pool, as a fraction of the "
    "pool size. Unset applies no cap, so each entry's share is "
    "proportional to its row count. The effective cap never drops below "
    "one divided by the number of entries seen so far, so the pool can "
    "still fill completely."
)


class InterpolationConfig(StrictModel):
    """Interpolation parameters for missing pose/position data.

    Attributes:
        linear_interp_limit: Maximum run of consecutive missing values filled
            by linear interpolation.
        edge_fill_limit: Maximum run of consecutive missing values
            forward-filled, then back-filled, after interpolation leaves them
            unresolved.
        max_missing_fraction: Fraction of missing columns above which a row
            is dropped entirely.
    """

    linear_interp_limit: Annotated[
        int, Declared(_LINEAR_INTERP_LIMIT_DESCRIPTION, unit="frames")
    ] = Field(default=10, ge=1)
    edge_fill_limit: Annotated[
        int, Declared(_EDGE_FILL_LIMIT_DESCRIPTION, unit="frames")
    ] = Field(default=3, ge=0)
    max_missing_fraction: Annotated[
        float, Declared(_MAX_MISSING_FRACTION_DESCRIPTION)
    ] = Field(default=0.10, ge=0.0, le=1.0)


class SamplingConfig(StrictModel):
    """Frame rate and temporal smoothing parameters.

    Attributes:
        fps_default: Frame rate used when the data's fps column is absent
            or does not resolve to exactly one value.
        smooth_win: Length of the moving-average window applied to smooth
            position and angle values before feature computation. A value
            of 1 or less disables smoothing.
    """

    fps_default: Annotated[float, Declared(_FPS_DEFAULT_DESCRIPTION, unit="fps")] = (
        Field(default=30.0, gt=0)
    )
    smooth_win: Annotated[int, Declared(_SMOOTH_WIN_DESCRIPTION, unit="frames")] = (
        Field(default=0, ge=0)
    )


class PoolConfig(StrictModel):
    """Candidate pool configuration for template extraction.

    Controls how per-entry contributions to the candidate pool are
    allocated before the final template selection step.

    Attributes:
        size: Number of candidates collected before selecting the final
            templates. Unset sets the pool to the target template count,
            so selection makes no reduction.
        allocation: How the per-entry quota for the pool is computed.
            reservoir performs weighted reservoir sampling in one pass.
            exact counts rows first, then samples a second pass with
            proportional quotas.
        max_entry_fraction: Cap on one entry's contribution to the pool, as
            a fraction of the pool size. Unset applies no cap, so each
            entry's share is proportional to its row count. The effective
            cap never drops below one divided by the number of entries
            seen so far, so the pool can still fill completely.
    """

    size: Annotated[int | None, Declared(_POOL_SIZE_DESCRIPTION)] = None
    allocation: Annotated[
        Literal["reservoir", "exact"], Declared(_ALLOCATION_DESCRIPTION)
    ] = "reservoir"
    max_entry_fraction: Annotated[
        float | None, Declared(_MAX_ENTRY_FRACTION_DESCRIPTION)
    ] = Field(default=None, ge=0.0, le=1.0)
