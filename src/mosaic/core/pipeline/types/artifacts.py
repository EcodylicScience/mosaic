from __future__ import annotations

from pathlib import Path
from typing import Final, Generic, Literal, Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from typing_extensions import TypeVar

from mosaic.core.pipeline._loaders import (
    JoblibLoadSpec,
    NpzLoadSpec,
    ParquetLoadSpec,
    load_from_spec,
)
from mosaic.core.pipeline.types.results import Result

L = TypeVar(
    "L",
    bound=NpzLoadSpec | ParquetLoadSpec | JoblibLoadSpec,
    default=NpzLoadSpec | ParquetLoadSpec | JoblibLoadSpec,
    covariant=True,
)
R = TypeVar("R", default=object, covariant=True)


class ArtifactSpec(Result[str], Generic[L, R]):
    """Reference to a feature artifact with load specification.

    `L` is the load spec type -- `NpzLoadSpec`, `ParquetLoadSpec` or
    `JoblibLoadSpec` -- and `R` is the return type of `from_path()`, which
    defaults to `object`. Both are declared through `Generic[L, R]` rather than
    the PEP 695 form, because each carries a `default=` that PEP 696 makes
    available only from Python 3.13 and this package targets 3.12.

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


TEMPLATES_ARTIFACT_NAME: Final = "templates.parquet"
"""What every producer of a templates matrix calls it, written once.

Both ``extract-templates`` and ``extract-labeled-templates`` write this name, and
``GlobalModelParams.templates`` resolves it. The consumer lives in ``core`` and the
producers in ``behavior``, so the name cannot be owned by either producer without
``core`` importing upward -- and a name spelled once on each side is a name that can
drift into resolving the wrong file.
"""


class TemplatesRef(ArtifactSpec[ParquetLoadSpec, pd.DataFrame]):
    """The templates matrix a global feature fits on.

    Pinned rather than generic. A run root is not a directory of named state files:
    ``run_feature`` writes one per-entry output parquet per sequence beside whatever
    ``save_state`` wrote, so the derived ``*.parquet`` glob a bare ``ParquetArtifact``
    falls back to matches those too, and they sort first.

    ``feature`` defaults to empty, which means *no reference* rather than a reference
    to nothing -- ``_resolve_dependencies`` skips it and it never reaches an index.
    """

    feature: str = ""
    pattern: str = TEMPLATES_ARTIFACT_NAME
    load: ParquetLoadSpec = Field(default_factory=ParquetLoadSpec)


class LabeledTemplatesRef(TemplatesRef):
    """A templates matrix carrying ``label`` and ``split`` beside its features.

    ``numeric_only=False`` is the whole difference and it is load-bearing: ``split``
    holds strings, so the default numeric filter drops it, and both consumers of a
    labeled matrix require it by name.
    """

    load: ParquetLoadSpec = Field(
        default_factory=lambda: ParquetLoadSpec(numeric_only=False)
    )


class FeatureLabelsSource(ArtifactSpec[NpzLoadSpec, np.ndarray]):
    """Labels loaded from a feature's output files."""

    source: Literal["feature"] = "feature"
    load: NpzLoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="labels"))
