"""Tests for XgboostFeature."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from mosaic.behavior.feature_library.xgboost_feature import XgboostFeature
from mosaic.core.pipeline.types import InputStream, Result


class TestXgboostParams:
    def test_default_strategy(self) -> None:
        params = XgboostFeature.Params.from_overrides(
            {
                "templates": {"feature": "extract-labeled-templates"},
                "default_class": 0,
            }
        )
        assert params.strategy == "multiclass"
        assert params.class_weight == "balanced"

    def test_one_vs_rest_strategy(self) -> None:
        params = XgboostFeature.Params.from_overrides(
            {
                "templates": {"feature": "extract-labeled-templates"},
                "default_class": 0,
                "strategy": "one_vs_rest",
            }
        )
        assert params.strategy == "one_vs_rest"

    def test_decision_threshold_float(self) -> None:
        params = XgboostFeature.Params.from_overrides(
            {
                "templates": {"feature": "extract-labeled-templates"},
                "default_class": 0,
                "decision_threshold": 0.7,
            }
        )
        assert params.decision_threshold == 0.7

    def test_decision_threshold_mapping(self) -> None:
        params = XgboostFeature.Params.from_overrides(
            {
                "templates": {"feature": "extract-labeled-templates"},
                "default_class": 0,
                "decision_threshold": {0: 0.5, 1: 0.8},
            }
        )
        assert params.decision_threshold == {0: 0.5, 1: 0.8}


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


def _make_sparse_class_templates(
    tmp_path: Path,
    train_classes: tuple[int, ...],
    test_classes: tuple[int, ...],
    n_per_class: int = 40,
    n_features: int = 5,
) -> Path:
    """Templates whose labels are behaviour ids, not a contiguous 0-based range.

    The real shape this guards: a four-class corpus subset to a few recordings,
    where a behaviour simply does not occur in the training ones. Every other
    fixture here builds ``range(n_classes)``, which is why a label set starting
    at 1 went unnoticed.
    """
    rng = np.random.default_rng(0)
    blocks = []
    for split, classes in (("train", train_classes), ("test", test_classes)):
        for cls in classes:
            features = rng.standard_normal((n_per_class, n_features)) + cls * 2
            df = pd.DataFrame(
                features, columns=[f"feat_{i}" for i in range(n_features)]
            )
            df["label"] = cls
            df["split"] = split
            blocks.append(df)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "templates.parquet"
    pd.concat(blocks, ignore_index=True).to_parquet(path, index=False)
    return path


class TestXgboostSparseClasses:
    """A class absent from the training split must not break the fit."""

    def _feature(self, templates_path: Path) -> XgboostFeature:
        feat = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={
                "templates": {"feature": "extract-labeled-templates"},
                "default_class": 3,
                "n_estimators": 10,
                "max_depth": 3,
            },
        )
        feat._templates = pd.read_parquet(templates_path)
        feat._feature_columns = [f"feat_{i}" for i in range(5)]
        return feat

    def test_fit_with_labels_not_starting_at_zero(self, tmp_path: Path) -> None:
        # Train holds 1, 2, 3 and test holds 0, 1, 3 -- what CalMS21 gives when
        # `attack` never occurs in the training recordings. Fitting on the raw
        # labels made XGBoost compare them against the 0..num_class-1 range it
        # derives from `num_class` and refuse: "Invalid classes inferred from
        # unique values of `y`. Expected: [0 1 2], got [1 2 3]".
        path = _make_sparse_class_templates(
            tmp_path / "templates", train_classes=(1, 2, 3), test_classes=(0, 1, 3)
        )
        feat = self._feature(path)

        feat.fit(InputStream(lambda: iter([]), n_entries=0))

        assert feat._classes == [1, 2, 3]
        assert isinstance(feat._model, XGBClassifier)
        # The evaluation report must be in label space, not index space.
        assert feat._metrics is not None
        reported = {k for k in feat._metrics if k.isdigit()}
        assert reported <= {"0", "1", "2", "3"}
        assert "1" in reported

    def test_apply_keeps_pair_identity(self, tmp_path: Path) -> None:
        # Pair-level input carries id1/id2 plus a `perspective` that separates A->B
        # from B->A -- both perspectives share the same pair ids. Dropping them left
        # two rows per frame with nothing to tell them apart, so the predictions
        # could not be joined back to the features they came from.
        path = _make_sparse_class_templates(
            tmp_path / "templates", train_classes=(1, 2, 3), test_classes=(1, 3)
        )
        feat = self._feature(path)
        feat.fit(InputStream(lambda: iter([]), n_entries=0))

        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            rng.standard_normal((12, 5)), columns=[f"feat_{i}" for i in range(5)]
        )
        frame["frame"] = np.repeat(np.arange(6), 2)
        frame["id1"] = 0
        frame["id2"] = 1
        frame["perspective"] = np.tile([0, 1], 6)

        out = feat.apply(frame)

        assert {"frame", "id1", "id2", "perspective"} <= set(out.columns)
        key = ["frame", "id1", "id2", "perspective"]
        assert not out.duplicated(subset=key).any()

    def test_predictions_are_labels_not_indices(self, tmp_path: Path) -> None:
        path = _make_sparse_class_templates(
            tmp_path / "templates", train_classes=(1, 2, 3), test_classes=(1, 3)
        )
        feat = self._feature(path)
        feat.fit(InputStream(lambda: iter([]), n_entries=0))

        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            rng.standard_normal((30, 5)) + 2.0,
            columns=[f"feat_{i}" for i in range(5)],
        )
        out = feat.apply(frame)

        predicted = set(out["predicted_label"].unique().tolist())
        # Index space would offer 0; label space cannot, since 0 was never trained.
        assert predicted <= {1, 2, 3}, predicted
        assert 0 not in predicted


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
        df = pd.DataFrame(
            {
                "frame": np.arange(20),
                "time": np.arange(20, dtype=float) / 30.0,
                "id": np.zeros(20, dtype=int),
                **{f"feat_{i}": rng.standard_normal(20) for i in range(5)},
            }
        )
        result = feat.apply(df)

        assert "predicted_label" in result.columns
        assert "frame" in result.columns
        for cls in [0, 1, 2]:
            assert f"prob_{cls}" in result.columns

    def test_apply_with_threshold(self, tmp_path: Path) -> None:
        feat = self._fit_feature(tmp_path / "t", decision_threshold=0.99)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "frame": np.arange(20),
                "time": np.arange(20, dtype=float) / 30.0,
                "id": np.zeros(20, dtype=int),
                **{f"feat_{i}": rng.standard_normal(20) for i in range(5)},
            }
        )
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
        df = pd.DataFrame(
            {
                "frame": np.arange(20),
                "time": np.arange(20, dtype=float) / 30.0,
                "id": np.zeros(20, dtype=int),
                **{f"feat_{i}": rng.standard_normal(20) for i in range(5)},
            }
        )
        result = feat.apply(df)

        assert "predicted_label" in result.columns
        for cls in [0, 1, 2]:
            assert f"prob_{cls}" in result.columns
