from __future__ import annotations

from typing import Annotated, Generic, Self, override

from pydantic import Field
from typing_extensions import TypeVar

from mosaic.core.params import Declared, DeclaredModel
from mosaic.core.strict_model import StrictModel

F = TypeVar("F", bound=str, default=str)

_FEATURE_DESCRIPTION = "Feature name whose output to consume."

_RUN_ID_DESCRIPTION = "Specific run ID, or None for latest finished run."

_EXECUTION_ID_DESCRIPTION = (
    "ULID of the attempt that produced this result (attempt identity, not "
    "content identity)."
)

_CACHE_HIT_DESCRIPTION = "Whether the producing run was fully served from cache."

_FAILED_ENTRIES_DESCRIPTION = (
    "Entry keys whose apply raised while the run carried on. Empty for a clean "
    "run; non-empty means a partial one, and the producing run's outputs are "
    "missing exactly these entities."
)

_COLUMN_DESCRIPTION = "Column name to extract from the parquet output."

_ENTRIES_WRITTEN_DESCRIPTION = (
    "How many entries the producing run left holding a valid output row, cache "
    "hits included -- so a resumed run and a fresh one report the same number "
    "over the same scope."
)


class Result(DeclaredModel, Generic[F]):
    """Reference to a prior feature's output as pipeline input.

    ``execution_id``, ``cache_hit``, ``failed_entries`` and ``entries_written``
    are ``exclude=True`` so they never enter ``model_dump()``. This is
    load-bearing: a ``Result`` doubles as a pipeline *input reference* whose
    ``model_dump()`` feeds the ``run_id`` hash of every downstream feature --
    excluding these attempt-level fields keeps that hash (and thus
    caching/determinism) unperturbed.
    """

    feature: Annotated[F, Declared(_FEATURE_DESCRIPTION)]
    run_id: Annotated[str | None, Declared(_RUN_ID_DESCRIPTION)] = None
    execution_id: Annotated[str | None, Declared(_EXECUTION_ID_DESCRIPTION)] = Field(
        default=None, exclude=True
    )
    cache_hit: Annotated[bool, Declared(_CACHE_HIT_DESCRIPTION)] = Field(
        default=False, exclude=True
    )
    failed_entries: Annotated[
        tuple[str, ...], Declared(_FAILED_ENTRIES_DESCRIPTION)
    ] = Field(default=(), exclude=True)
    # Declared last so the generated per-feature params tables in
    # ``docs/reference/features.md`` gain one appended row each rather than
    # shifting every existing one.
    entries_written: Annotated[int, Declared(_ENTRIES_WRITTEN_DESCRIPTION)] = Field(
        default=0, exclude=True
    )

    def use_latest(self) -> Self:
        """Return a copy with run_id=None (resolves to latest run)."""
        return self.model_copy(update={"run_id": None})

    @override
    def __str__(self) -> str:
        return repr(self)


class NNResult(Result[str]):
    """Result for a nearest-neighbor-family feature.

    Accepts any feature name (default ``"nearest-neighbor"``) so that
    auto-derived names like ``nearest-neighbor__from__tracks`` or variants
    computed from different upstream data (e.g. smoothed tracks) can be
    referenced.  Use ``from_result()`` to copy feature+run_id from an
    existing run.
    """

    feature: str = "nearest-neighbor"

    def from_result(self, result: Result[str]) -> Self:
        """Return a copy with feature and run_id set from another Result."""
        return self.model_copy(
            update={"feature": result.feature, "run_id": result.run_id}
        )


class BodyScaleResult(Result[str]):
    """Result for a body-scale-family feature.

    Accepts any feature name (default ``"body-scale"``) so that auto-derived
    names or upstream variants can be referenced.  Use ``from_result()`` to
    copy feature+run_id from an existing run.
    """

    feature: str = "body-scale"

    def from_result(self, result: Result[str]) -> Self:
        """Return a copy with feature and run_id set from another Result."""
        return self.model_copy(
            update={"feature": result.feature, "run_id": result.run_id}
        )


class TracksColumn(StrictModel):
    """Reference to a column in the tracks data.

    Attributes:
        column: Column name to extract from tracks.
    """

    column: str


class ResultColumn(Result[str]):
    """Reference to a column in a feature's standard parquet output."""

    feature: str = ""
    column: Annotated[str, Declared(_COLUMN_DESCRIPTION)]

    def from_result(self, result: Result[str]) -> Self:
        """Return a copy with feature and run_id set from another Result."""
        return self.model_copy(
            update={"feature": result.feature, "run_id": result.run_id}
        )
