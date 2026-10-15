"""Tests for GlobalScaler feature."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.global_scaler import GlobalScaler
from mosaic.core.pipeline.types import InputStream
from tests.helpers import make_sequence_df, make_templates, write_templates

_TEMPLATES_PARAMS: dict[str, object] = {
    "templates": {"feature": "extract-templates", "pattern": "templates.parquet"},
}
"""A reference to the templates that :func:`write_templates` writes."""


def _make_feature(
    params: dict[str, object] | None = None,
) -> GlobalScaler:
    inputs = GlobalScaler.Inputs(())
    return GlobalScaler(inputs=inputs, params=params)


def _fitted(tmp_path: Path, templates: pd.DataFrame) -> GlobalScaler:
    """A scaler fitted on *templates*, with its state not yet saved."""
    feat = _make_feature(params=_TEMPLATES_PARAMS)
    template_path = write_templates(tmp_path, templates)
    _ = feat.load_state(tmp_path / "run", {"templates": template_path}, {})
    feat.fit(InputStream(lambda: iter([]), n_entries=0))
    return feat


class TestFitAndSave:
    def test_fit_produces_scaler_and_scaled_templates(self, tmp_path: Path) -> None:
        templates = make_templates(100, 3, offset=5.0, scale=2.0)
        _fitted(tmp_path, templates).save_state(tmp_path / "run")

        # Read back artifacts
        assert (tmp_path / "run" / "scaler.joblib").exists()
        assert (tmp_path / "run" / "scaled_templates.parquet").exists()

        scaled = pd.read_parquet(tmp_path / "run" / "scaled_templates.parquet")
        assert scaled.shape == templates.shape
        assert list(scaled.columns) == list(templates.columns)

    def test_scaled_templates_are_standardized(self, tmp_path: Path) -> None:
        templates = make_templates(200, 4, offset=10.0, scale=3.0)
        _fitted(tmp_path, templates).save_state(tmp_path / "run")

        scaled = pd.read_parquet(tmp_path / "run" / "scaled_templates.parquet")
        # Mean should be ~0, std should be ~1
        means = scaled.mean()
        stds = scaled.std()
        for col in scaled.columns:
            assert abs(means[col]) < 0.1
            assert abs(stds[col] - 1.0) < 0.1

    def test_fit_reads_nan_as_missing_per_column(self, tmp_path: Path) -> None:
        """A template set with NaN rows fits, and the scaled rows keep their NaN.

        The scaler fits each column on its finite values, and its apply keeps a
        NaN where its input holds one, so its fit drops no row.
        """
        templates = make_templates(100, 3, offset=5.0, scale=2.0, nan_rows=(0, 7))
        _fitted(tmp_path, templates).save_state(tmp_path / "run")

        scaled = pd.read_parquet(tmp_path / "run" / "scaled_templates.parquet")
        assert scaled.shape == templates.shape
        assert scaled.loc[[0, 7], "feat_2"].isna().all()
        assert scaled.drop(index=[0, 7]).notna().all().all()

    def test_model_bundle_contents(self, tmp_path: Path) -> None:
        templates = make_templates(50, 2, offset=5.0, scale=2.0)
        _fitted(tmp_path, templates).save_state(tmp_path / "run")

        bundle = joblib.load(tmp_path / "run" / "scaler.joblib")
        assert "scaler" in bundle
        assert "feature_columns" in bundle
        assert "version" in bundle
        assert bundle["feature_columns"] == ["feat_0", "feat_1"]
        assert bundle["version"] == "0.2"


class TestApply:
    def test_apply_scales_feature_columns(self, tmp_path: Path) -> None:
        feat = _fitted(tmp_path, make_templates(100, 3, offset=5.0, scale=2.0))

        df = make_sequence_df(50, 3, offset=5.0, scale=2.0)
        result = feat.apply(df)

        # Metadata columns preserved
        assert "frame" in result.columns
        assert "sequence" in result.columns
        assert "group" in result.columns

        # Feature columns scaled (mean shifted toward 0)
        for col in ["feat_0", "feat_1", "feat_2"]:
            assert abs(result[col].mean()) < abs(df[col].mean())

    def test_apply_preserves_metadata(self, tmp_path: Path) -> None:
        feat = _fitted(tmp_path, make_templates(100, 3, offset=5.0, scale=2.0))

        df = make_sequence_df(30, 3, offset=5.0, scale=2.0)
        result = feat.apply(df)

        pd.testing.assert_series_equal(result["frame"], df["frame"])
        pd.testing.assert_series_equal(result["sequence"], df["sequence"])
        pd.testing.assert_series_equal(result["group"], df["group"])

    def test_apply_writes_only_the_fitted_payload_columns(self, tmp_path: Path) -> None:
        """The output drops each unfitted payload column, which a later fit would read.

        The identity and bookkeeping columns are kept unchanged, ``nn_id``
        among them.
        """
        feat = _fitted(tmp_path, make_templates(100, 2, offset=5.0, scale=2.0))
        df = make_sequence_df(30, 4, offset=5.0, scale=2.0)
        df["fps"] = 30.0
        df["nn_id"] = np.arange(30) % 3

        result = feat.apply(df)

        assert list(result.columns) == [
            "frame",
            "time",
            "id",
            "group",
            "sequence",
            "feat_0",
            "feat_1",
            "fps",
            "nn_id",
        ]
        kept = ["frame", "time", "id", "group", "sequence", "fps", "nn_id"]
        pd.testing.assert_frame_equal(result[kept], df[kept])

    def test_apply_raises_before_fit(self) -> None:
        feat = _make_feature(params=_TEMPLATES_PARAMS)
        df = make_sequence_df(10, 3, offset=5.0, scale=2.0)
        with pytest.raises(RuntimeError, match="Not fitted"):
            feat.apply(df)


class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        feat1 = _fitted(tmp_path, make_templates(100, 3, offset=5.0, scale=2.0))
        feat1.save_state(tmp_path / "run")

        # Load into fresh instance
        feat2 = _make_feature(params=_TEMPLATES_PARAMS)
        loaded = feat2.load_state(tmp_path / "run", {}, {})
        assert loaded is True

        # Apply should produce same results
        df = make_sequence_df(50, 3, offset=5.0, scale=2.0)
        result1 = feat1.apply(df)
        result2 = feat2.apply(df)
        pd.testing.assert_frame_equal(result1, result2)

    def test_load_state_returns_false_when_missing(self, tmp_path: Path) -> None:
        feat = _make_feature(params=_TEMPLATES_PARAMS)
        assert feat.load_state(tmp_path, {}, {}) is False
