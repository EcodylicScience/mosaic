"""Tests for GlobalTSNE feature."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.global_tsne import GlobalTSNE


def _make_feature(
    params: dict[str, object] | None = None,
) -> GlobalTSNE:
    inputs = GlobalTSNE.Inputs(())
    return GlobalTSNE(inputs=inputs, params=params)


def _make_templates(
    n_rows: int = 100,
    n_features: int = 5,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data: dict[str, object] = {}
    for i in range(n_features):
        data[f"feat_{i}"] = rng.standard_normal(n_rows)
    return pd.DataFrame(data)


def _make_sequence_df(
    n_rows: int = 50,
    n_features: int = 5,
    sequence: str = "seq_a",
    group: str = "grp_a",
) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    data: dict[str, object] = {
        "frame": np.arange(n_rows),
        "time": np.arange(n_rows, dtype=float) / 30.0,
        "id": np.zeros(n_rows, dtype=int),
        "group": [group] * n_rows,
        "sequence": [sequence] * n_rows,
    }
    for i in range(n_features):
        data[f"feat_{i}"] = rng.standard_normal(n_rows)
    return pd.DataFrame(data)


def _setup_templates(tmp_path: Path, templates: pd.DataFrame) -> Path:
    template_dir = tmp_path / "templates_run"
    template_dir.mkdir()
    path = template_dir / "templates.parquet"
    templates.to_parquet(path, index=False)
    return path


def _fit_feature(tmp_path: Path, n_templates: int = 100) -> GlobalTSNE:
    """Create, load, and fit a GlobalTSNE feature."""
    templates = _make_templates(n_templates)
    template_path = _setup_templates(tmp_path, templates)

    feat = _make_feature(params={
        "templates": {
            "feature": "extract-templates",
            "pattern": "templates.parquet",
        },
        "perplexity": 10,
        "n_jobs": 1,
        "fit": {"exaggeration_iters": 1, "iters": 1},
        "mapping": {"iters": 1},
    })
    feat.load_state(
        tmp_path / "run",
        {"templates": template_path},
        {},
    )
    feat.fit(lambda: iter([]))
    return feat


class TestFitAndSave:
    def test_fit_produces_embedding(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path)
        feat.save_state(tmp_path / "run")

        assert (tmp_path / "run" / "embedding.joblib").exists()
        assert (tmp_path / "run" / "global_tsne_templates.npz").exists()

    def test_model_bundle_contents(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path)
        feat.save_state(tmp_path / "run")

        bundle = joblib.load(tmp_path / "run" / "embedding.joblib")
        assert "embedding" in bundle
        assert "feature_columns" in bundle
        assert "version" in bundle
        assert bundle["feature_columns"] == [f"feat_{i}" for i in range(5)]
        assert bundle["version"] == "0.4"

    def test_template_coords_shape(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path, n_templates=80)
        feat.save_state(tmp_path / "run")

        data = np.load(tmp_path / "run" / "global_tsne_templates.npz")
        assert data["Y"].shape == (80, 2)

    def test_fit_raises_without_templates(self) -> None:
        feat = _make_feature(params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            },
            "perplexity": 10,
        })
        with pytest.raises(RuntimeError, match="No templates"):
            feat.fit(lambda: iter([]))


class TestApply:
    def test_apply_produces_tsne_columns(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path)

        df = _make_sequence_df(30, 5)
        result = feat.apply(df)

        assert "tsne_x" in result.columns
        assert "tsne_y" in result.columns
        assert result.shape[0] == 30

    def test_apply_preserves_metadata(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path)

        df = _make_sequence_df(20, 5)
        result = feat.apply(df)

        pd.testing.assert_series_equal(result["frame"], df["frame"])
        pd.testing.assert_series_equal(result["sequence"], df["sequence"])
        pd.testing.assert_series_equal(result["group"], df["group"])

    def test_apply_removes_feature_columns(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path)

        df = _make_sequence_df(20, 5)
        result = feat.apply(df)

        for i in range(5):
            assert f"feat_{i}" not in result.columns

    def test_apply_handles_nan(self, tmp_path: Path) -> None:
        feat = _fit_feature(tmp_path)

        df = _make_sequence_df(20, 5)
        df.iloc[5, df.columns.get_loc("feat_0")] = np.nan
        result = feat.apply(df)

        assert np.isnan(result["tsne_x"].iloc[5])
        assert np.isnan(result["tsne_y"].iloc[5])
        assert not np.isnan(result["tsne_x"].iloc[0])

    def test_apply_raises_before_fit(self) -> None:
        feat = _make_feature(params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            },
        })
        df = _make_sequence_df(10, 5)
        with pytest.raises(RuntimeError, match="Not fitted"):
            feat.apply(df)


class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        feat1 = _fit_feature(tmp_path)
        feat1.save_state(tmp_path / "run")

        feat2 = _make_feature(params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            },
            "perplexity": 10,
            "n_jobs": 1,
            "mapping": {"iters": 1},
        })
        loaded = feat2.load_state(tmp_path / "run", {}, {})
        assert loaded is True

        df = _make_sequence_df(30, 5)
        result1 = feat1.apply(df)
        result2 = feat2.apply(df)
        pd.testing.assert_frame_equal(result1, result2)

    def test_load_state_returns_false_when_missing(self, tmp_path: Path) -> None:
        feat = _make_feature(params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            },
        })
        assert feat.load_state(tmp_path, {}, {}) is False
