"""Tests for typed LoadSpec dispatch and data loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline.types import (
    ArtifactSpec,
    JoblibLoadSpec,
    NpzLoadSpec,
    ParquetLoadSpec,
    Result,
)


@pytest.fixture()
def parquet_path(tmp_path: Path) -> Path:
    p = tmp_path / "data.parquet"
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
        }
    )
    df.to_parquet(p, index=False)
    return p


@pytest.fixture()
def npz_path(tmp_path: Path) -> Path:
    p = tmp_path / "data.npz"
    np.savez(p, features=np.arange(12, dtype=np.float32).reshape(3, 4))
    return p


def test_load_joblib_artifact_rejects_non_joblib() -> None:
    from mosaic.core.pipeline.loading import load_joblib_artifact

    artifact = ArtifactSpec(feature="test", load=ParquetLoadSpec())
    with pytest.raises(ValueError, match="requires JoblibLoadSpec"):
        load_joblib_artifact(None, artifact)


def test_load_joblib_artifact_refuses_an_ambiguous_pattern(tmp_path: Path) -> None:
    """The public loader resolves through the same funnel as a feature run.

    Its run root holds every joblib a global fitter saved -- a model beside its
    class names and its training history -- so a derived ``*.joblib`` names no
    one of them.
    """
    import joblib

    from mosaic.core.pipeline.loading import load_joblib_artifact
    from tests.helpers import make_dataset

    dataset = make_dataset(tmp_path / "ds")
    run_root = dataset.get_root("features") / "up" / "0.1-abc"
    run_root.mkdir(parents=True)
    for name in ("identity_names.joblib", "model.joblib"):
        joblib.dump({"value": 1}, run_root / name)

    artifact = ArtifactSpec(feature="up", run_id="0.1-abc", load=JoblibLoadSpec())

    with pytest.raises(ValueError, match="holds 2 files matching"):
        _ = load_joblib_artifact(dataset, artifact)


def test_load_joblib_artifact_still_raises_when_nothing_matches(
    tmp_path: Path,
) -> None:
    """No match keeps its own error: nothing to load is not an ambiguity."""
    from mosaic.core.pipeline.loading import load_joblib_artifact
    from tests.helpers import make_dataset

    dataset = make_dataset(tmp_path / "ds")
    run_root = dataset.get_root("features") / "up" / "0.1-abc"
    run_root.mkdir(parents=True)

    artifact = ArtifactSpec(feature="up", run_id="0.1-abc", load=JoblibLoadSpec())

    with pytest.raises(FileNotFoundError, match="No files matching"):
        _ = load_joblib_artifact(dataset, artifact)


def test_temporal_stacking_stateless_protocol() -> None:
    from mosaic.behavior.feature_library.temporal_stacking import (
        TemporalStackingFeature,
    )

    inputs = TemporalStackingFeature.Inputs(
        (Result(feature="speed-angvel"), Result(feature="pair-wavelet"))
    )
    ts = TemporalStackingFeature(inputs=inputs)

    assert ts.name == "temporal-stack"
    assert ts.parallelizable is True
    assert ts.scope_dependent is False
    assert ts.load_state(Path("/tmp"), {}, {}) is True

    # apply on simple data
    df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3, 4],
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    result = ts.apply(df)
    assert isinstance(result, pd.DataFrame)
    assert "frame" in result.columns
    assert len(result) == 5


class TestLoadParquetDataFrame:
    """Tests for load_parquet_dataframe."""

    def test_basic_load(self, parquet_path: Path) -> None:
        from mosaic.core.pipeline.loading import load_parquet_dataframe

        spec = ParquetLoadSpec()
        df = load_parquet_dataframe(parquet_path, spec)
        assert df is not None
        assert "feat_a" in df.columns
        assert "feat_b" in df.columns
        assert "frame" in df.columns

    def test_with_df_filter(self, parquet_path: Path) -> None:
        from mosaic.core.pipeline.loading import load_parquet_dataframe

        spec = ParquetLoadSpec()
        df = load_parquet_dataframe(
            parquet_path, spec, df_filter=lambda d: d[d["frame"] >= 1]
        )
        assert df is not None
        assert len(df) == 2

    def test_empty_after_filter_returns_none(self, parquet_path: Path) -> None:
        from mosaic.core.pipeline.loading import load_parquet_dataframe

        spec = ParquetLoadSpec()
        df = load_parquet_dataframe(
            parquet_path, spec, df_filter=lambda d: d[d["frame"] > 100]
        )
        assert df is None

    def test_non_parquet_returns_none(self, npz_path: Path) -> None:
        from mosaic.core.pipeline.loading import load_parquet_dataframe

        spec = NpzLoadSpec(key="features")
        df = load_parquet_dataframe(npz_path, spec)
        assert df is None
