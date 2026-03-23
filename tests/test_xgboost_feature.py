"""Tests for XgboostFeature."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.xgboost_feature import XgboostFeature
from mosaic.core.pipeline.types import InputStream, Result


class TestXgboostParams:
    def test_default_strategy(self) -> None:
        params = XgboostFeature.Params.from_overrides({
            "templates": {"feature": "extract-labeled-templates"},
            "default_class": 0,
        })
        assert params.strategy == "multiclass"
        assert params.class_weight == "balanced"

    def test_one_vs_rest_strategy(self) -> None:
        params = XgboostFeature.Params.from_overrides({
            "templates": {"feature": "extract-labeled-templates"},
            "default_class": 0,
            "strategy": "one_vs_rest",
        })
        assert params.strategy == "one_vs_rest"

    def test_decision_threshold_float(self) -> None:
        params = XgboostFeature.Params.from_overrides({
            "templates": {"feature": "extract-labeled-templates"},
            "default_class": 0,
            "decision_threshold": 0.7,
        })
        assert params.decision_threshold == 0.7

    def test_decision_threshold_mapping(self) -> None:
        params = XgboostFeature.Params.from_overrides({
            "templates": {"feature": "extract-labeled-templates"},
            "default_class": 0,
            "decision_threshold": {0: 0.5, 1: 0.8},
        })
        assert params.decision_threshold == {0: 0.5, 1: 0.8}


from xgboost import XGBClassifier


def _make_templates_parquet(
    tmp_path: Path,
    n_per_class: int = 50,
    n_features: int = 5,
    n_classes: int = 3,
    include_test: bool = True,
) -> Path:
    """Create a templates.parquet matching ExtractLabeledTemplates output."""
    rng = np.random.default_rng(42)
    blocks = []
    for cls in range(n_classes):
        features = rng.standard_normal((n_per_class, n_features)) + cls * 2
        df = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(n_features)])
        df["label"] = cls
        if include_test:
            split = ["train"] * (n_per_class - 10) + ["test"] * 10
        else:
            split = ["train"] * n_per_class
        df["split"] = split
        blocks.append(df)
    result = pd.concat(blocks, ignore_index=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "templates.parquet"
    result.to_parquet(path, index=False)
    return path


class TestXgboostFit:
    def test_multiclass_fit(self, tmp_path: Path) -> None:
        templates_path = _make_templates_parquet(tmp_path / "templates")
        feat = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={
                "templates": {"feature": "extract-labeled-templates"},
                "default_class": 0,
                "n_estimators": 10,
                "max_depth": 3,
            },
        )
        feat._templates = pd.read_parquet(templates_path)
        feat._feature_columns = [f"feat_{i}" for i in range(5)]

        feat.fit(InputStream(lambda: iter([]), n_entries=0))

        assert feat._model is not None
        assert isinstance(feat._model, XGBClassifier)
        assert feat._classes == [0, 1, 2]

    def test_one_vs_rest_fit(self, tmp_path: Path) -> None:
        templates_path = _make_templates_parquet(tmp_path / "templates")
        feat = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={
                "templates": {"feature": "extract-labeled-templates"},
                "default_class": 0,
                "strategy": "one_vs_rest",
                "n_estimators": 10,
                "max_depth": 3,
            },
        )
        feat._templates = pd.read_parquet(templates_path)
        feat._feature_columns = [f"feat_{i}" for i in range(5)]

        feat.fit(InputStream(lambda: iter([]), n_entries=0))

        assert feat._model is not None
        assert isinstance(feat._model, list)
        assert len(feat._model) == 3

    def test_fit_with_evaluation(self, tmp_path: Path) -> None:
        """When test split is present, metrics should be computed."""
        templates_path = _make_templates_parquet(
            tmp_path / "templates", include_test=True
        )
        feat = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={
                "templates": {"feature": "extract-labeled-templates"},
                "default_class": 0,
                "n_estimators": 10,
                "max_depth": 3,
            },
        )
        feat._templates = pd.read_parquet(templates_path)
        feat._feature_columns = [f"feat_{i}" for i in range(5)]

        feat.fit(InputStream(lambda: iter([]), n_entries=0))

        assert feat._metrics is not None

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        templates_path = _make_templates_parquet(tmp_path / "templates")
        feat = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={
                "templates": {"feature": "extract-labeled-templates"},
                "default_class": 0,
                "n_estimators": 10,
            },
        )
        feat._templates = pd.read_parquet(templates_path)
        feat._feature_columns = [f"feat_{i}" for i in range(5)]
        feat.fit(InputStream(lambda: iter([]), n_entries=0))

        run_root = tmp_path / "run"
        feat.save_state(run_root)

        feat2 = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={
                "templates": {"feature": "extract-labeled-templates"},
                "default_class": 0,
            },
        )
        loaded = feat2.load_state(run_root, {}, {})
        assert loaded is True
        assert feat2._classes == [0, 1, 2]


class TestXgboostApply:
    def _fit_feature(self, tmp_path: Path, **params_override: object) -> XgboostFeature:
        """Helper: create and fit a feature for apply testing."""
        templates_path = _make_templates_parquet(
            tmp_path, n_per_class=50, include_test=False
        )
        base_params: dict[str, object] = {
            "templates": {"feature": "extract-labeled-templates"},
            "default_class": 0,
            "n_estimators": 10,
            "max_depth": 3,
        }
        base_params.update(params_override)
        feat = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params=base_params,
        )
        feat._templates = pd.read_parquet(templates_path)
        feat._feature_columns = [f"feat_{i}" for i in range(5)]
        feat.fit(InputStream(lambda: iter([]), n_entries=0))
        return feat

    def test_apply_produces_predictions(self, tmp_path: Path) -> None:
        feat = self._fit_feature(tmp_path / "t")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "frame": np.arange(20),
            "time": np.arange(20, dtype=float) / 30.0,
            "id": np.zeros(20, dtype=int),
            **{f"feat_{i}": rng.standard_normal(20) for i in range(5)},
        })
        result = feat.apply(df)

        assert "predicted_label" in result.columns
        assert "frame" in result.columns
        for cls in [0, 1, 2]:
            assert f"prob_{cls}" in result.columns

    def test_apply_with_threshold(self, tmp_path: Path) -> None:
        feat = self._fit_feature(tmp_path / "t", decision_threshold=0.99)
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "frame": np.arange(20),
            "time": np.arange(20, dtype=float) / 30.0,
            "id": np.zeros(20, dtype=int),
            **{f"feat_{i}": rng.standard_normal(20) for i in range(5)},
        })
        result = feat.apply(df)

        # With very high threshold, most predictions should fall back to default_class
        assert (result["predicted_label"] == 0).sum() > 0

    def test_apply_missing_columns_raises(self, tmp_path: Path) -> None:
        feat = self._fit_feature(tmp_path / "t")
        df = pd.DataFrame({"frame": [0, 1], "wrong_col": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            feat.apply(df)

    def test_apply_one_vs_rest(self, tmp_path: Path) -> None:
        feat = self._fit_feature(tmp_path / "t", strategy="one_vs_rest")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "frame": np.arange(20),
            "time": np.arange(20, dtype=float) / 30.0,
            "id": np.zeros(20, dtype=int),
            **{f"feat_{i}": rng.standard_normal(20) for i in range(5)},
        })
        result = feat.apply(df)

        assert "predicted_label" in result.columns
        for cls in [0, 1, 2]:
            assert f"prob_{cls}" in result.columns
