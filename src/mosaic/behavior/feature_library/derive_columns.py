"""DeriveColumns -- new columns from row arithmetic over a table's columns.

Each entry of ``columns`` names a column and the expression that computes it,
such as ``speed * 30 / 100`` or ``nan if isnull(x) else x ** 2``. The
expression language is :mod:`mosaic.core.expressions`: numpy arithmetic,
comparisons, ``&``, ``|``, ``~``, a conditional and a short list of functions,
evaluated over whole columns. Entries run in list order, and an entry may read
an earlier entry's column. The output is the input table with each derived
column appended after every input column.

A derived column does not overwrite an input column: a name the input already
has is refused when the feature runs. Its dtype is what numpy gives the
operations, ``float64``, ``int64`` or ``bool``, and a literal is ``float64``.
The computation is in the units of the table, so the feature runs over a table
in any track space.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Self, final

import pandas as pd
from pydantic import Field, field_validator, model_validator

from mosaic.core.expressions import (
    CONSTANTS,
    ExpressionError,
    evaluate,
    names_read,
    parse_expression,
)
from mosaic.core.params import Declared, DeclaredModel, Params, has_fields
from mosaic.core.pipeline.column_contract import (
    Expressions,
    ParamColumn,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel, RelatesLevel
from mosaic.core.pipeline.types import (
    META_COLS,
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.schema import ARENA_COLUMN
from mosaic.core.spaces import TRACK_SPACES, TrackSpace

from .registry import register_feature

if TYPE_CHECKING:
    from mosaic.core.expressions import ColumnValues

__all__ = ["DerivedColumn", "DeriveColumns"]

_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"

_METADATA_NAMES: frozenset[str] = META_COLS | {ARENA_COLUMN}
"""The identity and pair columns, and ``arena``, which no derived column takes."""

_NAME_DESCRIPTION = (
    "The name of the new column: letters, digits and underscores, not starting "
    "with a digit. It cannot repeat, be pi, nan or inf, be a metadata column "
    "(group, sequence, frame, time, id, id1, id2, perspective, arena), or name a "
    "column the input has."
)

_EXPR_DESCRIPTION = (
    "The expression that computes the column from the input's columns and from "
    "the columns of earlier entries, as in 'speed * 30 / 100'. Checked when the "
    "step is declared."
)

_COLUMNS_DESCRIPTION = (
    "The columns to add, in the order they are computed. A later entry may read "
    "an earlier one, and no entry may read a column that a later entry writes."
)


class DerivedColumn(DeclaredModel):
    """One column to add: its name and the expression that computes it."""

    name: Annotated[
        str, Field(pattern=_NAME_PATTERN, min_length=1), Declared(_NAME_DESCRIPTION)
    ]
    expr: Annotated[str, Declared(_EXPR_DESCRIPTION)]

    @field_validator("expr")
    @classmethod
    def _expression_parses(cls, expr: str) -> str:
        """Refuse an expression that uses a node outside the accepted syntax."""
        try:
            parse_expression(expr)
        except ExpressionError as error:
            raise ValueError(str(error)) from error
        return expr


@final
@register_feature
class DeriveColumns:
    """Add columns computed from row arithmetic over the table's columns.

    Outputs the input's rows in their order with every input column, then each
    derived column in list order. Field documentation is on
    :class:`~mosaic.behavior.feature_library.derive_columns.DeriveColumns.Params`.
    """

    category = "per-frame"
    name = "derive-columns"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    temporal_reach = 0  # each output row reads its own row
    relates: RelatesLevel = "none"
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    follows_ids: bool = False
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(TRACK_SPACES)
    requires: ClassVar[tuple[Requirement, ...]] = (
        Expressions("columns", "name", "expr"),
    )
    provides: ClassVar[Provides] = Provides(
        passthrough=True, adds=(ParamColumn("columns", key="name"),)
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        columns: Annotated[
            list[DerivedColumn], Field(min_length=1), Declared(_COLUMNS_DESCRIPTION)
        ]

        @model_validator(mode="after")
        def _names_are_distinct_and_free(self) -> Self:
            """Refuse a name listed twice and a name among the metadata columns."""
            if not has_fields(self, "columns"):
                return self
            seen: set[str] = set()
            for column in self.columns:
                if column.name in seen:
                    raise ValueError(
                        f"derive-columns: the column {column.name!r} is listed "
                        "twice. Each derived column has one expression."
                    )
                seen.add(column.name)
                if column.name in CONSTANTS:
                    raise ValueError(
                        f"derive-columns: {column.name!r} is a constant of the "
                        "expression language, so an expression could not read a "
                        "column of that name."
                    )
                if column.name in _METADATA_NAMES:
                    raise ValueError(
                        f"derive-columns: {column.name!r} is a metadata column and "
                        "cannot be derived."
                    )
            return self

        @model_validator(mode="after")
        def _no_expression_reads_a_later_entry(self) -> Self:
            """Refuse an expression that reads a column that a later entry writes."""
            if not has_fields(self, "columns"):
                return self
            for index, column in enumerate(self.columns):
                later = {other.name for other in self.columns[index + 1 :]}
                for read in names_read(parse_expression(column.expr)):
                    if read in later:
                        raise ValueError(
                            f"derive-columns: the expression {column.expr!r} of "
                            f"{column.name!r} reads {read!r}, which a later entry "
                            "writes. Move that entry above this one."
                        )
            return self

    def __init__(
        self,
        inputs: DeriveColumns.Inputs = Inputs(("tracks",)),
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

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        names = [str(name) for name in df.columns]
        taken = sorted(
            column.name for column in self.params.columns if column.name in names
        )
        if taken:
            raise ValueError(
                f"{self.name}: the input already has the column {taken[0]!r}, and a "
                "derived column does not overwrite an input column. Choose another "
                "name."
            )

        readable: dict[str, ColumnValues] = {str(name): df[name] for name in df.columns}
        derived: dict[str, ColumnValues] = {}
        for column in self.params.columns:
            try:
                values = evaluate(parse_expression(column.expr), readable, len(df))
            except ExpressionError as error:
                raise ValueError(
                    f"{self.name}: cannot derive the column {column.name!r}: {error}"
                ) from error
            readable[column.name] = values
            derived[column.name] = values
        return pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
