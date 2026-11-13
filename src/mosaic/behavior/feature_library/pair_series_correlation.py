"""
PairSeriesCorrelation -- one row per pair of individuals, correlating a column.

For every pair of individuals of an entry the feature correlates one column of
the two across the frames on which both have a finite value, such as the heading
that ``heading`` or the tracks carry. ``method="pearson"`` correlates the column
as written. ``method="circular"`` is the Fisher-Lee correlation of two angles in
radians, which does not depend on where the angles wrap. With ``max_lag_s`` set,
the feature also searches the lags within that many seconds for the largest
correlation and reports the lag at which it falls, so a pair whose members follow
one another with a delay is found at that delay.

The two statistics read the samples of a pair as they are: the frames both
individuals have, one pair at a time. A table that dropped frames is read on its
frame numbers, so a lag is a whole number of frame steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from mosaic.core.individuals import id_arenas, individual_pairs, refuse_repeated_rows
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    ORDER_COLUMN,
    Column,
    ParamColumn,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    DependencyLookup,
    EmitsLevel,
    Inputs,
    InputStream,
    Result,
    TrackInput,
    resolve_order_col,
)
from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.schema import ARENA_COLUMN, TRACK_SPACES, TrackSpace
from mosaic.core.timebase import frame_step, row_frames, uniform_grid_rate

from .helpers import ensure_columns, entry_identity
from .registry import register_feature

type FloatArray = npt.NDArray[np.float64]
type Identity = int | str

_DEGENERATE_ANGLES: Final = 1e-12
"""The share of ``n ** 2`` below which a Fisher-Lee denominator term is zero.

An angle series whose doubled angles all agree leaves ``n ** 2 - E ** 2 - F ** 2``
as rounding error rather than as exactly 0.
"""

_LAG_ROUNDING: Final = 1e-9
"""Added to the count of lag steps before it is floored, against a rate that is a
quotient and measures a rounding error short of a whole number of steps."""

_VALUE_COLUMN_DESCRIPTION = (
    "The column to correlate between the two individuals of a pair. The default "
    "is the body orientation column of the tracks, in radians."
)

_METHOD_DESCRIPTION = (
    "How to correlate the two series. pearson is Pearson's r of the column as "
    "written. circular is the Fisher-Lee correlation of two angles in radians, "
    "which treats an angle and the same angle plus a turn as one."
)

_MAX_LAG_DESCRIPTION = (
    "The longest lag to search for the largest correlation, in seconds. At a lag, "
    "the first individual's value at a frame is paired with the second "
    "individual's value that long after it. Unset, only the correlation without "
    "a lag is computed and peak_correlation and peak_lag_s are NaN."
)

_MIN_SAMPLES_DESCRIPTION = (
    "The fewest frames, with a finite value for both individuals, that a "
    "correlation needs. A pair with fewer has a NaN correlation, and a lag with "
    "fewer is not searched."
)


@final
@register_feature
class PairSeriesCorrelation:
    """
    Per-sequence feature correlating a column between the individuals of a pair.

    Output columns (one row per ordered pair, so two per pair of individuals,
    keyed by ``(id1, id2, perspective)``):
      - group, sequence: the entry, copied from the frame
      - id1, id2, perspective: the pair, perspective 0 for the lower id first
        and 1 for the mirrored row
      - arena: when the input has it
      - correlation: the correlation of ``value_column`` over the frames on
        which both individuals have a value, at no lag
      - n_samples: the number of those frames
      - peak_correlation: the largest correlation over the lags within
        ``max_lag_s``, NaN without ``max_lag_s``
      - peak_lag_s: the lag of that peak in seconds, positive when the second
        individual follows the first, NaN without ``max_lag_s``

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.pair_series_correlation.PairSeriesCorrelation.Params`.

    Notes
    -----
    Only individuals of one arena form a pair. The correlation, ``n_samples`` and
    ``peak_correlation`` are the same on the two rows of a pair, and
    ``peak_lag_s`` changes sign, because the second row reads the pair the other
    way round. A pair with fewer than ``min_samples`` frames, a series that does
    not vary, and a circular series whose angles all agree have a NaN
    correlation.

    At each lag the first individual's frame ``f`` is paired with the second
    individual's frame ``f + lag``, and the lags are the whole frame steps of the
    entry from ``-max_lag_s`` to ``max_lag_s``. The peak is the largest signed
    correlation among the lags with ``min_samples`` samples. Two equal peaks take
    the lag nearest 0, and the negative one of two at the same distance. The
    rate that turns frames into seconds is measured from ``time``, and a table
    whose steps are uneven logs a warning that names ``resample-tracks``.

    An entry with fewer than two individuals, or none that share an arena, has
    no rows.
    """

    category = "summary"
    name = "pair-series-correlation"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "arena-aware"
    # One row per pair of the entry, which a neighbor's rows would change.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    follows_ids: bool = True
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(TRACK_SPACES)
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        Column(C.time_col),
        Column(C.group_col),
        Column(C.seq_col),
        ORDER_COLUMN,
        ParamColumn("value_column"),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.group_col),
            Column(C.seq_col),
            Column("id1"),
            Column("id2"),
            Column("perspective"),
            Column("correlation"),
            Column("n_samples"),
            Column("peak_correlation"),
            Column("peak_lag_s"),
        ),
        conditional=kept(ARENA_COLUMN),
        symmetric=(
            Column("correlation"),
            Column("n_samples"),
            Column("peak_correlation"),
        ),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        value_column: Annotated[str, Declared(_VALUE_COLUMN_DESCRIPTION)] = (
            C.orientation_col
        )
        method: Annotated[
            Literal["pearson", "circular"], Declared(_METHOD_DESCRIPTION)
        ] = "pearson"
        max_lag_s: Annotated[float | None, Declared(_MAX_LAG_DESCRIPTION, unit="s")] = (
            Field(default=None, gt=0.0)
        )
        min_samples: Annotated[int, Declared(_MIN_SAMPLES_DESCRIPTION)] = Field(
            default=10, ge=3
        )

    def __init__(
        self,
        inputs: PairSeriesCorrelation.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- State protocol (stateless) ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        params = self.params
        ensure_columns(df, [C.id_col, C.time_col, params.value_column])
        group, sequence = entry_identity(df)
        order_col = resolve_order_col(df)
        refuse_repeated_rows(df, self.name, order_col)
        ordered = df.sort_values(order_col, kind="stable")
        series = _series_of_individuals(ordered, params.value_column)
        arenas = id_arenas(df)
        pairs = individual_pairs(sorted(series), arenas)
        if not pairs:
            return pd.DataFrame()
        lags, rate = self._lag_grid(ordered)
        rows: list[dict[str, object]] = []
        for first, second in pairs:
            measured = _measure_pair(series[first], series[second], lags, rate, params)
            for perspective, (id1, id2, sign) in enumerate(
                ((first, second, 1.0), (second, first, -1.0))
            ):
                rows.append(
                    {
                        C.group_col: group,
                        C.seq_col: sequence,
                        "id1": id1,
                        "id2": id2,
                        "perspective": perspective,
                        **({ARENA_COLUMN: arenas[first]} if arenas is not None else {}),
                        "correlation": measured.correlation,
                        "n_samples": measured.n_samples,
                        "peak_correlation": measured.peak_correlation,
                        # Adding 0.0 turns the -0.0 of a negated lag 0 into 0.0.
                        "peak_lag_s": sign * measured.peak_lag_s + 0.0,
                    }
                )
        out = pd.DataFrame(rows)
        return out.sort_values(["id1", "id2"], kind="stable").reset_index(drop=True)

    def _lag_grid(self, df: pd.DataFrame) -> tuple[list[float], float]:
        """The lags to search, in frames, and the frame rate that converts them.

        The lags run from ``-L`` to ``L`` whole frame steps, ordered by distance
        from 0 and the negative one first, so the first of equal peaks is the one
        the tie rule takes. Without ``max_lag_s`` there is no lag to search, and
        the rate is NaN.
        """
        max_lag_s = self.params.max_lag_s
        if max_lag_s is None:
            return [], math.nan
        rate = uniform_grid_rate(df, self.name)
        step = frame_step(row_frames(df))
        reach = math.floor(max_lag_s * rate / step + _LAG_ROUNDING)
        lags = [0.0]
        for k in range(1, reach + 1):
            lags.extend((-k * step, k * step))
        return lags, rate


@dataclass(frozen=True, slots=True)
class _Series:
    """One individual's finite values on the frames that carry them."""

    frames: FloatArray
    values: FloatArray


@dataclass(frozen=True, slots=True)
class _Measured:
    """What one pair measures: its correlation and its peak over the lags."""

    correlation: float
    n_samples: int
    peak_correlation: float
    peak_lag_s: float


def _series_of_individuals(
    df: pd.DataFrame, value_column: str
) -> dict[Identity, _Series]:
    """Each individual's frames and values, leaving out a row that is not finite.

    An individual with no finite row has an empty series.
    """
    frames = row_frames(df)
    values = np.asarray(df[value_column], dtype=np.float64)
    named = df[C.id_col].notna().to_numpy()
    usable = np.isfinite(frames) & np.isfinite(values)
    identities, codes = np.unique(df[C.id_col].to_numpy()[named], return_inverse=True)
    rows = np.flatnonzero(named)
    series: dict[Identity, _Series] = {}
    for code, identity in enumerate(identities.tolist()):
        own = rows[codes == code]
        keep = own[usable[own]]
        series[identity] = _Series(frames[keep], values[keep])
    return series


def _aligned(
    first: _Series, second: _Series, lag: float
) -> tuple[FloatArray, FloatArray]:
    """The values of *first* and of *second* *lag* frames after it, on shared frames."""
    _, at_first, at_second = np.intersect1d(
        first.frames + lag, second.frames, assume_unique=True, return_indices=True
    )
    return first.values[at_first], second.values[at_second]


def _measure_pair(
    first: _Series,
    second: _Series,
    lags: list[float],
    rate: float,
    params: PairSeriesCorrelation.Params,
) -> _Measured:
    """The correlation of a pair without a lag, and its peak over *lags*."""
    method = _pearson if params.method == "pearson" else _fisher_lee
    shared, others = _aligned(first, second, 0.0)
    correlation = method(shared, others, params.min_samples)
    peak_correlation = math.nan
    peak_lag_s = math.nan
    for lag in lags:
        shared_at_lag, others_at_lag = _aligned(first, second, lag)
        candidate = method(shared_at_lag, others_at_lag, params.min_samples)
        # A strict comparison keeps the first of equal peaks, which the order of
        # *lags* makes the one nearest 0 and the negative one of two.
        if math.isfinite(candidate) and (
            math.isnan(peak_correlation) or candidate > peak_correlation
        ):
            peak_correlation = candidate
            peak_lag_s = lag / rate
    return _Measured(correlation, shared.size, peak_correlation, peak_lag_s)


def _pearson(first: FloatArray, second: FloatArray, min_samples: int) -> float:
    """Pearson's r of two series, NaN for too few samples or a series that is constant."""
    if first.size < min_samples or np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return math.nan
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    denominator = math.sqrt(
        float(centered_first @ centered_first)
        * float(centered_second @ centered_second)
    )
    if denominator == 0.0:
        return math.nan
    return float(centered_first @ centered_second) / denominator


def _fisher_lee(first: FloatArray, second: FloatArray, min_samples: int) -> float:
    """The Fisher-Lee correlation of two series of angles in radians.

    Computed in the form that sums over samples once: with ``A = sum(cos a cos b)``,
    ``B = sum(sin a sin b)``, ``C = sum(cos a sin b)``, ``D = sum(sin a cos b)``,
    and ``E, F`` and ``G, H`` the sums of the cosine and the sine of the doubled
    angles of each series, the correlation is
    ``4 (A B - C D) / sqrt((n^2 - E^2 - F^2) (n^2 - G^2 - H^2))``. NaN for too few
    samples, and where a series has no spread so a term of the denominator is 0.
    """
    count = first.size
    if count < min_samples:
        return math.nan
    cos_a, sin_a = np.cos(first), np.sin(first)
    cos_b, sin_b = np.cos(second), np.sin(second)
    first_spread = count**2 - float(np.cos(2 * first).sum()) ** 2
    first_spread -= float(np.sin(2 * first).sum()) ** 2
    second_spread = count**2 - float(np.cos(2 * second).sum()) ** 2
    second_spread -= float(np.sin(2 * second).sum()) ** 2
    floor = _DEGENERATE_ANGLES * count**2
    if first_spread <= floor or second_spread <= floor:
        return math.nan
    numerator = 4.0 * (
        float(cos_a @ cos_b) * float(sin_a @ sin_b)
        - float(cos_a @ sin_b) * float(sin_a @ cos_b)
    )
    return numerator / math.sqrt(first_spread * second_spread)
