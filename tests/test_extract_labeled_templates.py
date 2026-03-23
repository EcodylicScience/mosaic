"""Tests for ExtractLabeledTemplates feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.extract_labeled_templates import (
    ExtractLabeledTemplates,
)
from mosaic.behavior.feature_library.helpers import feature_columns
from mosaic.core.pipeline.types import GroundTruthLabelsSource, InputStream, Result


class TestMetaColumnReservation:
    def test_feature_columns_excludes_label_and_split(self) -> None:
        df = pd.DataFrame({
            "frame": [0, 1, 2],
            "time": [0.0, 1.0, 2.0],
            "id": [0, 0, 0],
            "label": [1, 2, 0],
            "split": ["train", "test", "train"],
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
        })
        cols = feature_columns(df)
        assert "label" not in cols
        assert "split" not in cols
        assert cols == ["feat_a", "feat_b"]


class TestParams:
    def test_requires_exactly_one_sampling_spec(self) -> None:
        with pytest.raises(ValueError, match="Exactly one"):
            ExtractLabeledTemplates.Params.from_overrides({
                "labels": GroundTruthLabelsSource(),
            })

    def test_rejects_both_sampling_specs(self) -> None:
        with pytest.raises(ValueError, match="Exactly one"):
            ExtractLabeledTemplates.Params.from_overrides({
                "labels": GroundTruthLabelsSource(),
                "n_per_class": 100,
                "n_total": 500,
            })

    def test_n_per_class_int(self) -> None:
        params = ExtractLabeledTemplates.Params.from_overrides({
            "labels": GroundTruthLabelsSource(),
            "n_per_class": 100,
        })
        assert params.n_per_class == 100

    def test_n_total(self) -> None:
        params = ExtractLabeledTemplates.Params.from_overrides({
            "labels": GroundTruthLabelsSource(),
            "n_total": 500,
        })
        assert params.n_total == 500

    def test_test_fraction_validation(self) -> None:
        with pytest.raises(Exception):
            ExtractLabeledTemplates.Params.from_overrides({
                "labels": GroundTruthLabelsSource(),
                "n_total": 100,
                "test_fraction": 1.5,
            })

    def test_strategy_param(self) -> None:
        params = ExtractLabeledTemplates.Params.from_overrides({
            "labels": GroundTruthLabelsSource(),
            "n_total": 100,
            "strategy": "farthest_first",
        })
        assert params.strategy == "farthest_first"


from collections.abc import Callable, Iterator
from pathlib import Path

InputFactory = Callable[[], Iterator[tuple[str, pd.DataFrame]]]


def _make_labeled_df(
    n_rows: int,
    n_features: int = 3,
    sequence: str = "seq_a",
    group: str = "grp_a",
) -> pd.DataFrame:
    """Build a DataFrame with metadata + features + frame column for label alignment."""
    rng = np.random.default_rng(hash((sequence, n_rows)) % 2**32)
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


def _make_factory(
    entries: list[tuple[str, pd.DataFrame]],
) -> InputStream:
    def factory() -> Iterator[tuple[str, pd.DataFrame]]:
        yield from entries
    return InputStream(factory, n_entries=len(entries))


def _setup_labels_lookup(
    labels_dir: Path,
    entries: list[tuple[str, str, np.ndarray]],
) -> dict[tuple[str, str], Path]:
    """Write dense label NPZs and return a (group, sequence) -> Path lookup."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    lookup: dict[tuple[str, str], Path] = {}
    for group, sequence, labels_array in entries:
        path = labels_dir / f"{group}__{sequence}.npz"
        np.savez(path, labels=labels_array)
        lookup[(group, sequence)] = path
    return lookup


def _make_feature_with_labels(
    labels_lookup: dict[tuple[str, str], Path],
    params: dict[str, object] | None = None,
) -> ExtractLabeledTemplates:
    """Create feature and simulate load_state with labels lookup."""
    inputs = ExtractLabeledTemplates.Inputs((Result(feature="upstream"),))
    feat = ExtractLabeledTemplates(inputs=inputs, params=params)
    # Simulate _resolve_dependencies providing the labels lookup
    feat.load_state(
        Path("/nonexistent"),
        {},
        {"labels": labels_lookup},
    )
    return feat


class TestReservoirLabeled:
    def test_basic_two_classes(self, tmp_path: Path) -> None:
        """Two sequences, two classes, reservoir should produce templates with labels."""
        n = 100
        labels_a = np.array([0] * 50 + [1] * 50)
        labels_b = np.array([0] * 30 + [1] * 70)

        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", labels_a), ("g", "s2", labels_b)],
        )

        feat = _make_feature_with_labels(labels_lookup, params={
            "labels": {"kind": "behavior"},
            "n_per_class": 20,
            "test_fraction": 0.0,
            "random_state": 42,
        })

        df_a = _make_labeled_df(n, sequence="s1", group="g")
        df_b = _make_labeled_df(n, sequence="s2", group="g")
        feat.fit(_make_factory([("g__s1", df_a), ("g__s2", df_b)]))

        run_root = tmp_path / "run"
        feat.save_state(run_root)

        templates = pd.read_parquet(run_root / "templates.parquet")
        assert "label" in templates.columns
        assert "split" in templates.columns
        assert set(templates["label"].unique()) == {0, 1}
        # With n_per_class=20, expect ~20 per class (reservoir may have fewer if data is small)
        for label in [0, 1]:
            count = (templates["label"] == label).sum()
            assert count <= 20

    def test_split_assignment(self, tmp_path: Path) -> None:
        """Sequences should be assigned to train/test splits."""
        n = 50
        labels = np.array([0] * 25 + [1] * 25)

        # Use many sequences so split assignment is meaningful
        entries_labels = []
        entries_data = []
        for i in range(20):
            seq = f"s{i}"
            entries_labels.append(("g", seq, labels))
            entries_data.append((f"g__{seq}", _make_labeled_df(n, sequence=seq, group="g")))

        labels_lookup = _setup_labels_lookup(tmp_path / "labels" / "behavior", entries_labels)
        feat = _make_feature_with_labels(labels_lookup, params={
            "labels": {"kind": "behavior"},
            "n_total": 200,
            "test_fraction": 0.3,
            "random_state": 42,
        })
        feat.fit(_make_factory(entries_data))

        run_root = tmp_path / "run"
        feat.save_state(run_root)
        templates = pd.read_parquet(run_root / "templates.parquet")

        assert "train" in templates["split"].values
        assert "test" in templates["split"].values

    def test_n_total_balanced(self, tmp_path: Path) -> None:
        """n_total should divide evenly across classes (exact mode)."""
        n = 100
        labels = np.array([0] * 50 + [1] * 50)

        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", labels)],
        )
        feat = _make_feature_with_labels(labels_lookup, params={
            "labels": {"kind": "behavior"},
            "n_total": 40,
            "pool": {"allocation": "exact"},
            "test_fraction": 0.0,
        })
        feat.fit(_make_factory([("g__s1", _make_labeled_df(n, sequence="s1", group="g"))]))

        run_root = tmp_path / "run"
        feat.save_state(run_root)
        templates = pd.read_parquet(run_root / "templates.parquet")

        assert len(templates) == 40
        assert (templates["label"] == 0).sum() == 20
        assert (templates["label"] == 1).sum() == 20

    def test_deterministic_with_seed(self, tmp_path: Path) -> None:
        """Same seed should produce identical results."""
        n = 100
        labels = np.array([0] * 50 + [1] * 50)
        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", labels)],
        )

        results = []
        for run in range(2):
            feat = _make_feature_with_labels(labels_lookup, params={
                "labels": {"kind": "behavior"},
                "n_per_class": 15,
                "test_fraction": 0.0,
                "random_state": 123,
            })
            df = _make_labeled_df(n, sequence="s1", group="g")
            feat.fit(_make_factory([("g__s1", df)]))
            run_root = tmp_path / f"run_{run}"
            feat.save_state(run_root)
            results.append(pd.read_parquet(run_root / "templates.parquet"))

        pd.testing.assert_frame_equal(results[0], results[1])


class TestExactLabeled:
    def test_exact_per_class_mapping(self, tmp_path: Path) -> None:
        """Mapping[int, int] should allow per-class counts."""
        n = 200
        labels = np.array([0] * 100 + [1] * 100)
        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", labels)],
        )
        feat = _make_feature_with_labels(labels_lookup, params={
            "labels": {"kind": "behavior"},
            "n_per_class": {0: 10, 1: 30},
            "pool": {"allocation": "exact"},
            "test_fraction": 0.0,
        })
        df = _make_labeled_df(n, sequence="s1", group="g")
        feat.fit(_make_factory([("g__s1", df)]))

        run_root = tmp_path / "run"
        feat.save_state(run_root)
        templates = pd.read_parquet(run_root / "templates.parquet")

        assert (templates["label"] == 0).sum() == 10
        assert (templates["label"] == 1).sum() == 30


class TestFarthestFirstLabeled:
    def test_farthest_first_per_class(self, tmp_path: Path) -> None:
        """Farthest-first should select spread points within each class."""
        n = 200
        labels = np.array([0] * 100 + [1] * 100)
        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", labels)],
        )
        feat = _make_feature_with_labels(labels_lookup, params={
            "labels": {"kind": "behavior"},
            "n_per_class": 10,
            "strategy": "farthest_first",
            "pool": {"size": 50},
            "test_fraction": 0.0,
        })
        df = _make_labeled_df(n, sequence="s1", group="g")
        feat.fit(_make_factory([("g__s1", df)]))

        run_root = tmp_path / "run"
        feat.save_state(run_root)
        templates = pd.read_parquet(run_root / "templates.parquet")

        assert (templates["label"] == 0).sum() == 10
        assert (templates["label"] == 1).sum() == 10


class TestSaveLoadLabeled:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        n = 100
        labels = np.array([0] * 50 + [1] * 50)
        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", labels)],
        )
        feat = _make_feature_with_labels(labels_lookup, params={
            "labels": {"kind": "behavior"},
            "n_per_class": 15,
            "test_fraction": 0.0,
        })
        df = _make_labeled_df(n, sequence="s1", group="g")
        feat.fit(_make_factory([("g__s1", df)]))

        run_root = tmp_path / "run"
        feat.save_state(run_root)

        # Load into fresh instance
        feat2 = ExtractLabeledTemplates(
            ExtractLabeledTemplates.Inputs((Result(feature="upstream"),)),
            params={"labels": {"kind": "behavior"}, "n_per_class": 15},
        )
        loaded = feat2.load_state(run_root, {}, {})
        assert loaded is True

        feat2.save_state(tmp_path / "reload")
        original = pd.read_parquet(run_root / "templates.parquet")
        reloaded = pd.read_parquet(tmp_path / "reload" / "templates.parquet")
        pd.testing.assert_frame_equal(original, reloaded)

    def test_load_state_returns_false_when_missing(self, tmp_path: Path) -> None:
        feat = ExtractLabeledTemplates(
            ExtractLabeledTemplates.Inputs((Result(feature="upstream"),)),
            params={"labels": {"kind": "behavior"}, "n_per_class": 10},
        )
        assert feat.load_state(tmp_path, {}, {}) is False

    def test_apply_adds_label_and_split(self) -> None:
        df = _make_labeled_df(10)
        feat = ExtractLabeledTemplates(
            ExtractLabeledTemplates.Inputs((Result(feature="upstream"),)),
            params={"labels": {"kind": "behavior"}, "n_per_class": 5},
        )
        result = feat.apply(df)
        assert "label" in result.columns
        assert "split" in result.columns
        # Without labels_lookup, all labels default to 0
        assert (result["label"] == 0).all()
        # Without sequence_splits, split defaults to "train"
        assert (result["split"] == "train").all()
        # Original columns are preserved
        for col in df.columns:
            assert col in result.columns


class TestEdgeCasesLabeled:
    def test_empty_inputs_raises(self, tmp_path: Path) -> None:
        labels_lookup = _setup_labels_lookup(tmp_path / "labels" / "behavior", [])
        feat = _make_feature_with_labels(labels_lookup, params={
            "labels": {"kind": "behavior"},
            "n_per_class": 10,
        })
        with pytest.raises(RuntimeError, match="No data"):
            feat.fit(InputStream(lambda: iter([]), n_entries=0))

    def test_single_class(self, tmp_path: Path) -> None:
        """All rows have the same label."""
        n = 100
        labels = np.zeros(n, dtype=np.int64)
        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", labels)],
        )
        feat = _make_feature_with_labels(labels_lookup, params={
            "labels": {"kind": "behavior"},
            "n_per_class": 20,
            "test_fraction": 0.0,
        })
        df = _make_labeled_df(n, sequence="s1", group="g")
        feat.fit(_make_factory([("g__s1", df)]))

        run_root = tmp_path / "run"
        feat.save_state(run_root)
        templates = pd.read_parquet(run_root / "templates.parquet")
        assert set(templates["label"].unique()) == {0}
        assert len(templates) == 20
