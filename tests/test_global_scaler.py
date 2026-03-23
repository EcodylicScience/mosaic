"""Tests for GlobalScaler feature."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.global_scaler import GlobalScaler


def _make_feature(
    params: dict[str, object] | None = None,
) -> GlobalScaler:
    inputs = GlobalScaler.Inputs(())
    return GlobalScaler(inputs=inputs, params=params)


def _make_templates(
    n_rows: int = 100,
    n_features: int = 3,
    offset: float = 5.0,
    scale: float = 2.0,
) -> pd.DataFrame:
    """Build a templates DataFrame with known mean/std."""
    rng = np.random.default_rng(42)
    data: dict[str, object] = {}
    for i in range(n_features):
        data[f"feat_{i}"] = rng.standard_normal(n_rows) * scale + offset
    return pd.DataFrame(data)


def _make_sequence_df(
    n_rows: int = 50,
    n_features: int = 3,
    offset: float = 5.0,
    scale: float = 2.0,
    sequence: str = "seq_a",
    group: str = "grp_a",
) -> pd.DataFrame:
    """Build a per-sequence DataFrame with metadata + feature columns."""
    rng = np.random.default_rng(99)
    data: dict[str, object] = {
        "frame": np.arange(n_rows),
        "time": np.arange(n_rows, dtype=float) / 30.0,
        "id": np.zeros(n_rows, dtype=int),
        "group": [group] * n_rows,
        "sequence": [sequence] * n_rows,
    }
    for i in range(n_features):
        data[f"feat_{i}"] = rng.standard_normal(n_rows) * scale + offset
    return pd.DataFrame(data)


def _setup_templates_dir(
    tmp_path: Path,
    templates: pd.DataFrame,
) -> Path:
    """Write templates.parquet and return the directory."""
    template_dir = tmp_path / "templates_run"
    template_dir.mkdir()
    templates.to_parquet(template_dir / "templates.parquet", index=False)
    return template_dir


class TestFitAndSave:
    def test_fit_produces_scaler_and_scaled_templates(self, tmp_path: Path) -> None:
        templates = _make_templates(100, 3)
        template_dir = _setup_templates_dir(tmp_path, templates)

        feat = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        feat.load_state(
            tmp_path / "run",
            {"templates": template_dir / "templates.parquet"},
            {},
        )
        feat.fit(lambda: iter([]))
        feat.save_state(tmp_path / "run")

        # Read back artifacts
        assert (tmp_path / "run" / "scaler.joblib").exists()
        assert (tmp_path / "run" / "scaled_templates.parquet").exists()

        scaled = pd.read_parquet(tmp_path / "run" / "scaled_templates.parquet")
        assert scaled.shape == templates.shape
        assert list(scaled.columns) == list(templates.columns)

    def test_scaled_templates_are_standardized(self, tmp_path: Path) -> None:
        templates = _make_templates(200, 4, offset=10.0, scale=3.0)
        template_dir = _setup_templates_dir(tmp_path, templates)

        feat = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        feat.load_state(
            tmp_path / "run",
            {"templates": template_dir / "templates.parquet"},
            {},
        )
        feat.fit(lambda: iter([]))
        feat.save_state(tmp_path / "run")

        scaled = pd.read_parquet(tmp_path / "run" / "scaled_templates.parquet")
        # Mean should be ~0, std should be ~1
        means = scaled.mean()
        stds = scaled.std()
        for col in scaled.columns:
            assert abs(means[col]) < 0.1
            assert abs(stds[col] - 1.0) < 0.1

    def test_model_bundle_contents(self, tmp_path: Path) -> None:
        templates = _make_templates(50, 2)
        template_dir = _setup_templates_dir(tmp_path, templates)

        feat = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        feat.load_state(
            tmp_path / "run",
            {"templates": template_dir / "templates.parquet"},
            {},
        )
        feat.fit(lambda: iter([]))
        feat.save_state(tmp_path / "run")

        bundle = joblib.load(tmp_path / "run" / "scaler.joblib")
        assert "scaler" in bundle
        assert "feature_columns" in bundle
        assert "version" in bundle
        assert bundle["feature_columns"] == ["feat_0", "feat_1"]
        assert bundle["version"] == "0.1"


class TestApply:
    def test_apply_scales_feature_columns(self, tmp_path: Path) -> None:
        templates = _make_templates(100, 3, offset=5.0, scale=2.0)
        template_dir = _setup_templates_dir(tmp_path, templates)

        feat = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        feat.load_state(
            tmp_path / "run",
            {"templates": template_dir / "templates.parquet"},
            {},
        )
        feat.fit(lambda: iter([]))

        df = _make_sequence_df(50, 3, offset=5.0, scale=2.0)
        result = feat.apply(df)

        # Metadata columns preserved
        assert "frame" in result.columns
        assert "sequence" in result.columns
        assert "group" in result.columns

        # Feature columns scaled (mean shifted toward 0)
        for col in ["feat_0", "feat_1", "feat_2"]:
            assert abs(result[col].mean()) < abs(df[col].mean())

    def test_apply_preserves_metadata(self, tmp_path: Path) -> None:
        templates = _make_templates(100, 3)
        template_dir = _setup_templates_dir(tmp_path, templates)

        feat = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        feat.load_state(
            tmp_path / "run",
            {"templates": template_dir / "templates.parquet"},
            {},
        )
        feat.fit(lambda: iter([]))

        df = _make_sequence_df(30, 3)
        result = feat.apply(df)

        pd.testing.assert_series_equal(result["frame"], df["frame"])
        pd.testing.assert_series_equal(result["sequence"], df["sequence"])
        pd.testing.assert_series_equal(result["group"], df["group"])

    def test_apply_raises_before_fit(self) -> None:
        feat = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        df = _make_sequence_df(10, 3)
        with pytest.raises(RuntimeError, match="Not fitted"):
            feat.apply(df)


class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        templates = _make_templates(100, 3)
        template_dir = _setup_templates_dir(tmp_path, templates)

        # Fit and save
        feat1 = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        feat1.load_state(
            tmp_path / "run",
            {"templates": template_dir / "templates.parquet"},
            {},
        )
        feat1.fit(lambda: iter([]))
        feat1.save_state(tmp_path / "run")

        # Load into fresh instance
        feat2 = _make_feature(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                },
            }
        )
        loaded = feat2.load_state(tmp_path / "run", {}, {})
        assert loaded is True

        # Apply should produce same results
        df = _make_sequence_df(50, 3)
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
