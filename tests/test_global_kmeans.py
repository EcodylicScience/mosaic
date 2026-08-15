"""Tests for GlobalKMeansClustering feature."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.global_kmeans import GlobalKMeansClustering
from tests.helpers import make_sequence_df, make_templates, write_templates


def _make_feature(
    params: dict[str, object] | None = None,
) -> GlobalKMeansClustering:
    inputs = GlobalKMeansClustering.Inputs(())
    return GlobalKMeansClustering(inputs=inputs, params=params)


def _fit_feature(
    tmp_path: Path, n_templates: int = 200, k: int = 5
) -> GlobalKMeansClustering:
    """Create, load, and fit a GlobalKMeansClustering feature."""
    templates = make_templates(n_templates, 5)
    template_path = write_templates(tmp_path, templates)

    feat = _make_feature(
        params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            },
            "k": k,
            "n_init": 1,
            "max_iter": 10,
        }
    )
    feat.load_state(
        tmp_path / "run",
        {"templates": template_path},
        {},
    )
    feat.fit(lambda: iter([]))
    return feat


class TestFitAndSave:
    def test_fit_produces_model_and_artifacts(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path)
        feat.save_state(tmp_path / "run")

        assert (tmp_path / "run" / "model.joblib").exists()
        assert (tmp_path / "run" / "cluster_centers.npz").exists()
        assert (tmp_path / "run" / "cluster_sizes.parquet").exists()

    def test_model_bundle_contents(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path, k=5)
        feat.save_state(tmp_path / "run")

        bundle = joblib.load(tmp_path / "run" / "model.joblib")
        assert "kmeans" in bundle
        assert "feature_columns" in bundle
        assert "version" in bundle
        assert bundle["feature_columns"] == [f"feat_{i}" for i in range(5)]
        assert bundle["version"] == "0.4"

    def test_cluster_centers_shape(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path, k=5)
        feat.save_state(tmp_path / "run")

        data = np.load(tmp_path / "run" / "cluster_centers.npz")
        assert data["centers"].shape == (5, 5)

    def test_cluster_sizes_sum(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path, n_templates=200, k=5)
        feat.save_state(tmp_path / "run")

        sizes = pd.read_parquet(tmp_path / "run" / "cluster_sizes.parquet")
        assert sizes["count"].sum() == 200

    def test_fit_raises_without_templates(self) -> None:
        feat = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        with pytest.raises(RuntimeError, match="No templates"):
            feat.fit(lambda: iter([]))


class TestApply:
    def test_apply_assigns_cluster_column(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path, k=5)

        df = make_sequence_df(30, 5)
        result = feat.apply(df)

        assert "cluster" in result.columns
        assert result.shape[0] == 30
        assert result["cluster"].between(0, 4).all()

    def test_apply_preserves_metadata(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path, k=5)

        df = make_sequence_df(20, 5)
        result = feat.apply(df)

        pd.testing.assert_series_equal(result["frame"], df["frame"], check_names=False)
        pd.testing.assert_series_equal(
            result["sequence"], df["sequence"], check_names=False
        )
        pd.testing.assert_series_equal(result["group"], df["group"], check_names=False)

    def test_apply_removes_feature_columns(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path, k=5)

        df = make_sequence_df(20, 5)
        result = feat.apply(df)

        for i in range(5):
            assert f"feat_{i}" not in result.columns

    def test_apply_handles_nan(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path, k=5)

        df = make_sequence_df(20, 5)
        df.iloc[5, df.columns.get_loc("feat_0")] = np.nan
        result = feat.apply(df)

        # NaN row should be dropped
        assert result.shape[0] == 19

    def test_apply_raises_before_fit(self) -> None:
        feat = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        df = make_sequence_df(10, 5)
        with pytest.raises(RuntimeError, match="not fitted"):
            feat.apply(df)


class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        feat1 = _fit_feature(tmp_path, k=5)
        feat1.save_state(tmp_path / "run")

        feat2 = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
                "k": 5,
            }
        )
        loaded = feat2.load_state(tmp_path / "run", {}, {})
        assert loaded is True

        df = make_sequence_df(30, 5)
        result1 = feat1.apply(df)
        result2 = feat2.apply(df)
        pd.testing.assert_frame_equal(result1, result2)

    def test_load_state_returns_false_when_missing(self, tmp_path: Path) -> None:
        feat = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        assert feat.load_state(tmp_path, {}, {}) is False
