from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import (
    ClassVar,
    Generic,
    Literal,
    Protocol,
    Self,
    override,
)

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, RootModel, model_validator
from typing_extensions import TypeVar

from mosaic.core.pipeline._loaders import (
    DictModel,
    JoblibLoadSpec,
    LoadSpec,
    NpzLoadSpec,
    ParquetLoadSpec,
    load_from_spec,
)

OutputType = Literal["per_frame", "global", "summary", "viz"] | None
InputRequire = Literal["nonempty", "empty", "any"]


class Params(DictModel):
    """Base for all feature parameter models.

    Provides from_overrides() constructor for user-config dicts.
    Subclasses declare feature-specific fields.
    """

    @classmethod
    def from_overrides(cls, overrides: dict[str, object] | None = None) -> Self:
        """Construct from user dict; missing keys get field defaults.

        For BaseModel-typed fields with default_factory, partial dict overrides
        are merged on top of the default before validation (1-level deep merge).
        This replaces the per-feature deep-merge hacks in global_ward.
        """
        if not overrides:
            return cls()
        merged = dict(overrides)
        for key, value in list(merged.items()):
            if not isinstance(value, dict):
                continue
            field_info = cls.model_fields.get(key)
            if field_info is None or field_info.default_factory is None:
                continue
            default_obj: object = field_info.get_default(call_default_factory=True)  # pyright: ignore[reportAny]
            if isinstance(default_obj, BaseModel):
                merged[key] = {**default_obj.model_dump(), **value}
        return cls.model_validate(merged)


TrackInput = Literal["tracks"]

F = TypeVar("F", bound=str, default=str)
K = TypeVar("K", bound=str, default=str)
L = TypeVar(
    "L",
    bound=NpzLoadSpec | ParquetLoadSpec | JoblibLoadSpec,
    default=NpzLoadSpec | ParquetLoadSpec | JoblibLoadSpec,
    covariant=True,
)
R = TypeVar("R", default=object, covariant=True)


class Result(DictModel, Generic[F]):
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


class ArtifactSpec(Result[str], Generic[L, R]):
    """Reference to a feature artifact with load specification.

    Type parameters:
        L: Load spec type (NpzLoadSpec, ParquetLoadSpec, JoblibLoadSpec).
        R: Return type of from_path(). Defaults to object.

    Attributes:
        load: How to load the matched files.
        pattern: Glob pattern. Auto-derived from load.kind when empty.
    """

    load: L
    pattern: str = ""

    @model_validator(mode="after")
    def _derive_pattern(self) -> Self:
        kind_ext = f".{self.load.kind}"
        if not self.pattern:
            self.pattern = f"*{kind_ext}"
        elif not self.pattern.endswith(kind_ext):
            raise ValueError(
                f"pattern {self.pattern!r} extension does not match load kind {self.load.kind!r}"
            )
        return self

    @classmethod
    def from_result(cls, result: Result[str]) -> Self:
        """Create from a Result, validating feature match.

        Typed artifact subclasses (with a default feature) validate
        that result.feature matches. Base ArtifactSpec passes through.
        """
        from pydantic_core import PydanticUndefined

        expected = cls.model_fields["feature"].default
        if expected is not PydanticUndefined and isinstance(expected, str):
            if not (
                result.feature == expected
                or result.feature.startswith(f"{expected}__from__")
            ):
                raise ValueError(
                    f"{cls.__name__} expects feature={expected!r} (or {expected}__from__...), got {result.feature!r}"
                )
        return cls.model_validate({"feature": result.feature, "run_id": result.run_id})

    def from_path(self, path: Path) -> R:
        """Load artifact from a resolved file path.

        Dispatches on load-spec type via load_from_spec().
        Return type is determined by the R type parameter.
        """
        return load_from_spec(path, self.load)  # pyright: ignore[reportReturnType]


NpzArtifact = ArtifactSpec[NpzLoadSpec, np.ndarray]
ParquetArtifact = ArtifactSpec[ParquetLoadSpec, pd.DataFrame]
JoblibArtifact = ArtifactSpec[JoblibLoadSpec, R]


class FeatureLabelsSource(ArtifactSpec[NpzLoadSpec, np.ndarray]):
    """Labels loaded from a feature's output files."""

    source: Literal["feature"] = "feature"
    load: NpzLoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="labels"))


class LabelsSource(DictModel, Generic[K]):
    """Base class for dataset label dependencies.

    Resolved by _resolve_dependency_paths (Task 9) to
    <dataset_root>/labels/<kind>/.
    """

    kind: K


class GroundTruthLabelsSource(LabelsSource[Literal["behavior"]]):
    """Labels loaded from labels/<kind>/index.csv."""

    source: Literal["labels"] = "labels"
    kind: Literal["behavior"] = "behavior"
    load: LoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="labels"))
    pattern: str | None = None


LabelsSourceSpec = ResultColumn | GroundTruthLabelsSource


InputItem = TypeVar("InputItem", bound=TrackInput | Result, default=TrackInput | Result)


class InputsLike(Protocol):
    """Read-only interface satisfied by any Inputs[InputItem]."""

    @property
    def root(self) -> tuple[TrackInput | Result, ...]: ...
    @property
    def has_tracks(self) -> bool: ...
    @property
    def is_single_tracks(self) -> bool: ...
    @property
    def is_single_feature(self) -> bool: ...
    @property
    def is_multi(self) -> bool: ...
    @property
    def is_empty(self) -> bool: ...
    @property
    def feature_inputs(self) -> tuple[Result, ...]: ...
    def storage_suffix(self) -> str | None: ...
    def model_dump(self) -> dict[str, object]: ...


class Inputs(RootModel[tuple[InputItem, ...]], Generic[InputItem]):
    """Base class for feature input collections. Mirrors Params.

    Each Feature subclasses to narrow allowed input types,
    paralleling class Params(Params):.

    Examples:
        Inputs(("tracks",))
        Inputs((Result(feature="speed-angvel"),))
        Inputs(("tracks", Result(feature="nn", run_id="0.1-abc")))

    Per-feature narrowing:
        class Inputs(Inputs[TrackInput]):
            pass

    Features that take no pipeline inputs:
        class Inputs(Inputs[Result]):
            _require: ClassVar[InputRequire] = "empty"

    Self-loading features that optionally accept inputs (e.g. fit + assign):
        class Inputs(Inputs[Result]):
            _require: ClassVar[InputRequire] = "any"
    """

    # "nonempty" (default): at least one input required
    # "empty": must be empty (no pipeline inputs)
    # "any": both empty and non-empty are valid
    _require: ClassVar[InputRequire] = "nonempty"

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self._require == "empty":
            if self.root:
                raise ValueError("This feature takes no pipeline inputs")
            return self
        if self._require == "nonempty" and not self.root:
            raise ValueError("Inputs must have at least one item")
        keys = [i if isinstance(i, str) else (i.feature, i.run_id) for i in self.root]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate inputs: {keys}")
        return self

    @property
    def has_tracks(self) -> bool:
        return any(i == "tracks" for i in self.root)

    @property
    def feature_inputs(self) -> tuple[Result, ...]:  # type: ignore[type-arg]
        return tuple(i for i in self.root if isinstance(i, Result))

    @property
    def is_single_tracks(self) -> bool:
        return len(self.root) == 1 and self.root[0] == "tracks"

    @property
    def is_single_feature(self) -> bool:
        return len(self.root) == 1 and isinstance(self.root[0], Result)

    @property
    def is_multi(self) -> bool:
        return len(self.root) > 1

    @property
    def is_empty(self) -> bool:
        return len(self.root) == 0

    def storage_suffix(self) -> str | None:
        feats = self.feature_inputs
        if not feats:
            return None
        if len(feats) == 1:
            return feats[0].feature
        return "+".join(f.feature for f in feats)


class Feature(Protocol):
    """Feature protocol -- 4 attributes, 4 methods."""

    name: str
    version: str
    parallelizable: bool
    scope_dependent: bool

    @property
    def inputs(self) -> InputsLike: ...

    @property
    def params(self) -> Params: ...

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool: ...

    def fit(self, inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None: ...

    def save_state(self, run_root: Path) -> None: ...

    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...
