from __future__ import annotations

from typing import Literal

from pydantic import Field

from mosaic.core.pipeline._loaders import StrictModel

__all__ = [
    "InterpolationConfig",
    "PoolConfig",
    "SamplingConfig",
]


class InterpolationConfig(StrictModel):
    """Interpolation parameters for missing pose/position data.

    Attributes:
        linear_interp_limit: Max consecutive NaN frames to fill via linear
            interpolation. Default 10, must be >= 1.
        edge_fill_limit: Max frames to forward/backward fill at sequence edges.
            Default 3, must be >= 0.
        max_missing_fraction: Rows with a higher fraction of NaN columns are
            dropped entirely. Default 0.10, range [0, 1].
    """

    linear_interp_limit: int = Field(default=10, ge=1)
    edge_fill_limit: int = Field(default=3, ge=0)
    max_missing_fraction: float = Field(default=0.10, ge=0.0, le=1.0)


class SamplingConfig(StrictModel):
    """Frame rate and temporal smoothing parameters.

    Attributes:
        fps_default: Fallback frames-per-second when the data does not carry an
            fps column. Default 30.0, must be > 0.
        smooth_win: Moving-average window size applied to pose coordinates
            before feature computation. 0 disables smoothing. Default 0.
    """

    fps_default: float = Field(default=30.0, gt=0)
    smooth_win: int = Field(default=0, ge=0)


class PoolConfig(StrictModel):
    """Candidate pool configuration for template extraction.

    Controls how per-entry contributions to the candidate pool are
    allocated before the final template selection step.

    Attributes:
        size: Candidate pool size. For "random" strategy, defaults to
            n_templates (pool == output). For "farthest_first", should
            be larger (e.g. n_templates * 3).
        allocation: How per-entry quotas are computed.
            "reservoir": weighted reservoir sampling, single pass.
            "exact": two-pass -- first counts rows, second samples
            with exact proportional quotas.
            Default "reservoir".
        max_entry_fraction: Cap per entry as fraction of pool size.
            None means no cap (purely proportional). At runtime,
            effective cap is max(max_entry_fraction, 1 / n_entries)
            so the pool can always be filled completely. Default None.
    """

    size: int | None = None
    allocation: Literal["reservoir", "exact"] = "reservoir"
    max_entry_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
