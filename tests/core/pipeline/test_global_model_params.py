"""Tests for GlobalModelParams exclusive-source validation."""

from __future__ import annotations

import pytest
from pydantic import Field, ValidationError

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


# --- A feature's params, across a round trip ----------------------------------


@pytest.mark.parametrize("source", ["templates", "model"])
def test_a_global_model_params_file_reads_back(source: str) -> None:
    """What ``run_feature`` writes, ``reconcile`` has to be able to rebuild from.

    ``model_dump`` emits both ``templates`` and ``model``, the unused one as an
    explicit null, so a validator counting key *presence* rejected every params
    file any global model feature ever wrote. ``reconcile`` rebuilds a run's
    feature from exactly that file, so it could confirm none of them.
    """
    from mosaic.behavior.feature_library import GlobalTSNE

    params = GlobalTSNE.Params.from_overrides({source: {"feature": "upstream"}})
    dumped = params.model_dump()
    assert "templates" in dumped and "model" in dumped, "both keys are written"

    restored = GlobalTSNE.Params.from_overrides(dumped)

    assert restored.identity_dump() == params.identity_dump()


def test_a_global_model_params_still_needs_exactly_one_source() -> None:
    """The rule itself is unchanged: neither and both are still refused."""
    from mosaic.behavior.feature_library import GlobalTSNE

    with pytest.raises(ValidationError):
        _ = GlobalTSNE.Params.from_overrides({})
    with pytest.raises(ValidationError):
        _ = GlobalTSNE.Params.from_overrides(
            {"templates": {"feature": "t"}, "model": {"feature": "m"}}
        )
