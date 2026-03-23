"""Tests for GlobalModelParams exclusive-source validation."""

from __future__ import annotations

import pytest
from pydantic import Field

from mosaic.core.pipeline.types import (
    GlobalModelParams,
    JoblibArtifact,
    JoblibLoadSpec,
)


class _StubModelArtifact(JoblibArtifact[object]):
    feature: str = "stub"
    pattern: str = "stub.joblib"
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


class _StubParams(GlobalModelParams[_StubModelArtifact]):
    model: _StubModelArtifact | None = Field(default_factory=_StubModelArtifact)


class TestGlobalModelParamsValidation:
    def test_requires_exactly_one_source(self) -> None:
        # Neither provided
        with pytest.raises(ValueError, match="Exactly one"):
            _StubParams.from_overrides({})

        # Both provided
        with pytest.raises(ValueError, match="Exactly one"):
            _StubParams.from_overrides(
                {
                    "templates": {
                        "feature": "x",
                        "pattern": "x.parquet",
                        "load": {},
                    },
                    "model": {
                        "feature": "x",
                        "pattern": "x.joblib",
                        "load": {},
                    },
                }
            )

    def test_templates_only_valid(self) -> None:
        params = _StubParams.from_overrides(
            {
                "templates": {
                    "feature": "x",
                    "pattern": "x.parquet",
                },
            }
        )
        assert params.templates is not None
        assert params.model is None

    def test_model_only_valid(self) -> None:
        params = _StubParams.from_overrides(
            {
                "model": {
                    "feature": "x",
                    "pattern": "x.joblib",
                },
            }
        )
        assert params.model is not None
        assert params.templates is None
