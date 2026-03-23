from __future__ import annotations

from typing import Generic, Literal, Self, override

from typing_extensions import TypeVar

from mosaic.core.pipeline._loaders import StrictModel

F = TypeVar("F", bound=str, default=str)


class Result(StrictModel, Generic[F]):
    """Reference to a prior feature's output as pipeline input.

    Attributes:
        feature: Feature name whose output to consume.
        run_id: Specific run ID, or None for latest finished run.
    """

    feature: F
    run_id: str | None = None

    def use_latest(self) -> Self:
        """Return a copy with run_id=None (resolves to latest run)."""
        return self.model_copy(update={"run_id": None})

    @override
    def __str__(self) -> str:
        return repr(self)


class NNResult(Result[Literal["nearest-neighbor"]]):
    """Result narrowed to the nearest-neighbor feature."""

    feature: Literal["nearest-neighbor"] = "nearest-neighbor"


class BodyScaleResult(Result[Literal["body-scale"]]):
    """Result narrowed to the body-scale feature."""

    feature: Literal["body-scale"] = "body-scale"


class TracksColumn(StrictModel):
    """Reference to a column in the tracks data.

    Attributes:
        column: Column name to extract from tracks.
    """

    column: str


class ResultColumn(Result[str]):
    """Reference to a column in a feature's standard parquet output.

    Attributes:
        feature: Source feature name.
        column: Column name to extract from the parquet output.
        run_id: Specific run ID, or None for latest.
    """

    feature: str = ""
    column: str

    def from_result(self, result: Result[str]) -> Self:
        """Return a copy with feature and run_id set from another Result."""
        return self.model_copy(
            update={"feature": result.feature, "run_id": result.run_id}
        )
