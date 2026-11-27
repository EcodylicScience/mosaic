"""TrackCoverage -- how much of a recording a presence rule finds animals in.

One summary row per entry, and per arena when the input has an ``arena`` column.
A presence rule names columns and a mode: a row passes when every column
(``all``) or at least one column (``any``) is finite, a missing value read as
NaN. The span is every frame from the table's first to its last, so a frame
without a row finds no id present. For each rule the row counts the frames
where every id passes, the frames where no id passes, and the gaps: the runs of
consecutive span frames where no id passes.

The pieces that read a table are :func:`~mosaic.core.track_quality.rule_presence`
and :func:`~mosaic.core.track_quality.gap_lengths`. The feature reads finiteness
alone, so it runs over a table in any track space.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, final

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from mosaic.core.params import Declared, DeclaredModel, Params, has_fields
from mosaic.core.pipeline.column_contract import (
    Column,
    Conditional,
    ParamColumn,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel, RelatesLevel
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.schema import ARENA_COLUMN
from mosaic.core.spaces import TRACK_SPACES, TrackSpace
from mosaic.core.track_quality import gap_lengths, rule_presence

from .aggregate_common import (
    QUANTILE_METHOD_NAMES,
    QuantileMethod,
    distinct_quantiles,
    quantile_text,
    quantiles_or_nan,
)
from .registry import register_feature

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["PresenceRule", "TrackCoverage"]

_RULE_NAME_PATTERN = r"^[A-Za-z0-9_]+$"

TIME_STEP_MEDIAN = "time_step_median"
"""The column of the median time between consecutive frames that have rows."""

TIME_STEP_MAX = "time_step_max"
"""The column of the largest time between consecutive frames that have rows."""

_NAME_DESCRIPTION = (
    "The rule's name, letters, digits and underscores. It prefixes the rule's "
    "columns, as `<name>_frames_all`, and no two rules share one."
)

_RULE_COLUMNS_DESCRIPTION = (
    "The columns the rule reads, at least one. Each is read as float64, a "
    "missing value as NaN, and a row passes when its value is finite."
)

_MODE_DESCRIPTION = (
    "all: a row passes when every column is finite. any: a row passes when at "
    "least one column is finite."
)

_RULES_DESCRIPTION = (
    "The presence rules, each written as its own set of columns. The default is "
    "one rule named xy over X and Y with mode all. Each rule is counted "
    "separately over every id of the entry, and an id is present in a frame "
    "when one of its rows there passes. A frame without a row finds every id "
    "absent."
)

_GAP_QUANTILES_DESCRIPTION = (
    "Quantiles of the gap lengths of each rule, each a number from 0 to 1 with "
    "no two giving the same column name. Each writes "
    "`<rule>_gap_quantile_<text>`, the numpy.quantile of the lengths in frames "
    "under quantile_method, or NaN without gaps. The text is q as a percent "
    "without trailing zeros and with the point written _: 0.95 gives 95 and "
    "0.995 gives 99_5."
)

_QUANTILE_METHOD_DESCRIPTION = (
    f"The method of numpy.quantile for gap_quantiles: {QUANTILE_METHOD_NAMES}. "
    "Needs gap_quantiles."
)

_GAP_THRESHOLDS_DESCRIPTION = (
    "Thresholds in frames, each at least 0 and none listed twice. Each writes "
    "`<rule>_gaps_over_<t>`, the number of gaps strictly longer than t frames."
)


class PresenceRule(DeclaredModel):
    """One presence rule: its name, the columns it reads and how it combines them."""

    name: Annotated[str, Field(pattern=_RULE_NAME_PATTERN), Declared(_NAME_DESCRIPTION)]
    columns: Annotated[
        list[str], Field(min_length=1), Declared(_RULE_COLUMNS_DESCRIPTION)
    ]
    mode: Annotated[Literal["all", "any"], Declared(_MODE_DESCRIPTION)]


def _default_rules() -> list[PresenceRule]:
    """The rule of the default params: X and Y, all."""
    return [PresenceRule(name="xy", columns=["X", "Y"], mode="all")]


@final
@register_feature
class TrackCoverage:
    """Summarize, per entry, the frames where animals are present and the gaps.

    Output columns (one row per entry, and per arena when the input has it):
      - group, sequence, arena: copied from the input when present
      - n_ids: the distinct ids that are not null
      - n_frames: the length of the span, the frames from the first to the last
      - n_frames_with_rows: the distinct frames that have a row
      - per rule, prefixed ``<rule>_``: frames_all (frames where every id
        passes), frames_none (frames where no id passes), n_gaps, gap_mean (NaN
        without gaps), gap_max (0 without gaps), gap_quantile_<text> for each of
        ``gap_quantiles`` and gaps_over_<t> for each of ``gap_thresholds``
      - time_step_median, time_step_max: when the input has ``time``, the
        differences of each frame's first time between consecutive frames that
        have rows, NaN with fewer than two such frames

    The span is the whole input's, so the arenas of an entry share it, and an
    arena without a row in a frame finds its ids absent there. A row without an
    id belongs to no id. An id present at a frame has at least one passing row
    there, and two rows of one id count once. ``frames_all`` is 0 for an arena
    that has no id. A row without an arena, in an input that has the column, is
    refused.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.track_coverage.TrackCoverage.Params`.
    """

    category = "summary"
    name = "track-coverage"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    # One row per entry, which a neighbor's rows would change.
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "unidentified"
    # Frames where every id is present read ids that follow animals.
    follows_ids: bool = True
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(TRACK_SPACES)
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        ParamColumn("rules", key="columns"),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column("n_ids"),
            Column("n_frames"),
            Column("n_frames_with_rows"),
            ParamColumn("rules", key="name", suffix="_frames_all"),
            ParamColumn("rules", key="name", suffix="_frames_none"),
            ParamColumn("rules", key="name", suffix="_n_gaps"),
            ParamColumn("rules", key="name", suffix="_gap_mean"),
            ParamColumn("rules", key="name", suffix="_gap_max"),
        ),
        conditional=(
            *kept(C.group_col, C.seq_col, ARENA_COLUMN),
            Conditional(
                ParamColumn(
                    "rules",
                    key="name",
                    suffix="_gap_quantile_",
                    open_families=("_gap_quantile_",),
                ),
                when_param="gap_quantiles",
            ),
            Conditional(
                ParamColumn(
                    "rules",
                    key="name",
                    suffix="_gaps_over_",
                    open_families=("_gaps_over_",),
                ),
                when_param="gap_thresholds",
            ),
            Conditional(Column(TIME_STEP_MEDIAN), when_present=Column(C.time_col)),
            Conditional(Column(TIME_STEP_MAX), when_present=Column(C.time_col)),
        ),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        rules: Annotated[
            list[PresenceRule], Field(min_length=1), Declared(_RULES_DESCRIPTION)
        ] = Field(default_factory=_default_rules)
        gap_quantiles: Annotated[
            list[Annotated[float, Field(ge=0, le=1)]],
            Declared(_GAP_QUANTILES_DESCRIPTION),
        ] = Field(default_factory=list)
        quantile_method: Annotated[
            QuantileMethod, Declared(_QUANTILE_METHOD_DESCRIPTION)
        ] = "linear"
        gap_thresholds: Annotated[
            list[Annotated[int, Field(ge=0)]], Declared(_GAP_THRESHOLDS_DESCRIPTION)
        ] = Field(default_factory=list)

        @field_validator("rules", check_fields=False)
        @classmethod
        def _rule_names_are_distinct(
            cls, value: list[PresenceRule]
        ) -> list[PresenceRule]:
            """Refuse a rule name listed twice, which would name its columns twice."""
            seen: set[str] = set()
            for rule in value:
                if rule.name in seen:
                    raise ValueError(
                        f"the rule {rule.name!r} is listed twice. Each rule needs "
                        "its own name."
                    )
                seen.add(rule.name)
            return value

        @field_validator("gap_quantiles", check_fields=False)
        @classmethod
        def _quantiles_name_distinct_columns(cls, value: list[float]) -> list[float]:
            """Refuse two quantiles that give one column name suffix."""
            return distinct_quantiles(value, "gap_quantiles")

        @field_validator("gap_thresholds", check_fields=False)
        @classmethod
        def _thresholds_are_distinct(cls, value: list[int]) -> list[int]:
            """Refuse a threshold listed twice, which would name a column twice."""
            repeated = sorted({t for t in value if value.count(t) > 1})
            if repeated:
                raise ValueError(f"gap_thresholds lists {repeated[0]} more than once")
            return value

        @model_validator(mode="after")
        def _a_method_needs_quantiles(self) -> Self:
            """Refuse a ``quantile_method`` other than ``linear`` without quantiles."""
            if not has_fields(self, "gap_quantiles", "quantile_method"):
                return self
            if self.quantile_method != "linear" and not self.gap_quantiles:
                raise ValueError(
                    f"quantile_method is {self.quantile_method!r} but gap_quantiles "
                    "is empty"
                )
            return self

    def __init__(
        self,
        inputs: TrackCoverage.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

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

    def _refuse_missing_columns(self, df: pd.DataFrame) -> None:
        """Refuse an input that lacks ``frame``, ``id`` or a column a rule reads."""
        present = {str(name) for name in df.columns}
        wanted = [C.frame_col, C.id_col]
        for rule in self.params.rules:
            wanted.extend(rule.columns)
        missing = [name for name in dict.fromkeys(wanted) if name not in present]
        if missing:
            raise ValueError(
                f"{self.name}: the input lacks the column {missing[0]!r}. Name "
                "columns the input has in rules."
            )

    def _dtypes(self, has_time: bool) -> dict[str, type]:
        """The dtype of each column that is not copied from the input."""
        dtypes: dict[str, type] = {
            "n_ids": np.int64,
            "n_frames": np.int64,
            "n_frames_with_rows": np.int64,
        }
        for rule in self.params.rules:
            dtypes[f"{rule.name}_frames_all"] = np.int64
            dtypes[f"{rule.name}_frames_none"] = np.int64
            dtypes[f"{rule.name}_n_gaps"] = np.int64
            dtypes[f"{rule.name}_gap_mean"] = np.float64
            dtypes[f"{rule.name}_gap_max"] = np.int64
            for q in self.params.gap_quantiles:
                dtypes[f"{rule.name}_gap_quantile_{quantile_text(q)}"] = np.float64
            for t in self.params.gap_thresholds:
                dtypes[f"{rule.name}_gaps_over_{t}"] = np.int64
        if has_time:
            dtypes[TIME_STEP_MEDIAN] = np.float64
            dtypes[TIME_STEP_MAX] = np.float64
        return dtypes

    def _rule_values(
        self,
        rule: PresenceRule,
        part: pd.DataFrame,
        span: npt.NDArray[np.int64],
        present: _Presence,
    ) -> dict[str, int | float]:
        """The columns of *rule* over the rows of one arena."""
        passes = rule_presence(part, rule.columns, rule.mode)
        counts = present.counts(passes, span.size)
        none_present = counts == 0
        gaps = gap_lengths(span, none_present)
        values: dict[str, int | float] = {
            f"{rule.name}_frames_all": int(
                (counts == present.n_ids).sum() if present.n_ids else 0
            ),
            f"{rule.name}_frames_none": int(none_present.sum()),
            f"{rule.name}_n_gaps": int(gaps.size),
            f"{rule.name}_gap_mean": float(np.mean(gaps)) if gaps.size else np.nan,
            f"{rule.name}_gap_max": int(gaps.max()) if gaps.size else 0,
        }
        quantiles = self.params.gap_quantiles
        if quantiles:
            found = quantiles_or_nan(
                gaps.astype(np.float64), quantiles, self.params.quantile_method
            )
            for q, value in zip(quantiles, found, strict=True):
                values[f"{rule.name}_gap_quantile_{quantile_text(q)}"] = float(value)
        for t in self.params.gap_thresholds:
            values[f"{rule.name}_gaps_over_{t}"] = int((gaps > t).sum())
        return values

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        self._refuse_missing_columns(df)
        if ARENA_COLUMN in df.columns and df[ARENA_COLUMN].isna().any():
            raise ValueError(
                f"{self.name}: {int(df[ARENA_COLUMN].isna().sum())} of {len(df)} "
                f"rows have no {ARENA_COLUMN}. Every row of an input with an "
                f"{ARENA_COLUMN} column needs one."
            )

        frames = df[C.frame_col].to_numpy(dtype=np.int64)
        first = int(frames.min())
        span = np.arange(first, int(frames.max()) + 1, dtype=np.int64)
        has_time = C.time_col in df.columns
        times = (
            df[C.time_col].to_numpy(dtype=np.float64, na_value=np.nan)
            if has_time
            else None
        )

        arena_codes: npt.NDArray[np.intp] | None = None
        arenas: list[object] = []
        if ARENA_COLUMN in df.columns:
            arena_codes, uniques = pd.factorize(df[ARENA_COLUMN], sort=True)
            arenas = [*uniques]
        groups = range(len(arenas)) if arena_codes is not None else range(1)

        rows: list[dict[str, object]] = []
        for index in groups:
            member = (
                arena_codes == index
                if arena_codes is not None
                else np.ones(len(df), dtype=bool)
            )
            part = df[member]
            part_frames = frames[member]
            present = _Presence(part_frames - first, part[C.id_col])
            row: dict[str, object] = {}
            for name in (C.group_col, C.seq_col):
                if name in df.columns:
                    row[name] = df[name].iloc[0]
            if arena_codes is not None:
                row[ARENA_COLUMN] = arenas[index]
            row["n_ids"] = present.n_ids
            row["n_frames"] = int(span.size)
            row["n_frames_with_rows"] = int(np.unique(part_frames).size)
            for rule in self.params.rules:
                row.update(self._rule_values(rule, part, span, present))
            if times is not None:
                median, largest = _time_steps(part_frames, times[member])
                row[TIME_STEP_MEDIAN] = median
                row[TIME_STEP_MAX] = largest
            rows.append(row)

        out = pd.DataFrame(rows)
        return out.astype(self._dtypes(has_time))


class _Presence:
    """The ids of one arena and where each row's id sits among them.

    Attributes:
        n_ids: The distinct ids that are not null.
    """

    def __init__(self, frame_offsets: npt.NDArray[np.int64], ids: pd.Series) -> None:
        codes, uniques = pd.factorize(ids)
        self._frame_offsets = frame_offsets
        self._codes = np.asarray(codes, dtype=np.int64)
        self.n_ids = int(len(uniques))

    def counts(
        self, passes: npt.NDArray[np.bool_], n_frames: int
    ) -> npt.NDArray[np.int64]:
        """Return, for each span frame, the distinct ids with a passing row there.

        Args:
            passes: Whether each row of the arena passes the rule.
            n_frames: The length of the span.

        Returns:
            One int64 count per span frame. A row without an id counts for none.
        """
        keep = passes & (self._codes >= 0)
        if not keep.any():
            return np.zeros(n_frames, dtype=np.int64)
        pairs = np.unique(self._frame_offsets[keep] * self.n_ids + self._codes[keep])
        return np.bincount(pairs // self.n_ids, minlength=n_frames).astype(np.int64)


def _time_steps(
    frames: npt.NDArray[np.int64], times: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Return the median and the largest time between consecutive frames with rows.

    Each frame's time is that of its first row. NaN comes back for both with
    fewer than two frames.
    """
    distinct, first_rows = np.unique(frames, return_index=True)
    if distinct.size < 2:
        return np.nan, np.nan
    steps = np.diff(times[first_rows])
    return float(np.median(steps)), float(np.max(steps))
