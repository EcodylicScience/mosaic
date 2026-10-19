"""Tests for ExtractLabeledTemplates feature."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.attention_target import AttentionTarget
from mosaic.behavior.feature_library.extract_labeled_templates import (
    ExtractLabeledTemplates,
)
from mosaic.behavior.feature_library import helpers
from mosaic.behavior.feature_library.helpers import feature_columns
from mosaic.behavior.feature_library.nearestneighbor import NearestNeighbor
from mosaic.core.pipeline.loading import load_entry_data
from mosaic.core.pipeline.types import (
    GroundTruthLabelsSource,
    InputStream,
    LoadSpec,
    ParquetLoadSpec,
    Result,
)
from tests.helpers import run_id_digest, track_table, write_sparse_labels


class TestMetaColumnReservation:
    def test_feature_columns_excludes_label_and_split(self) -> None:
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "time": [0.0, 1.0, 2.0],
                "id": [0, 0, 0],
                "label": [1, 2, 0],
                "split": ["train", "test", "train"],
                "feat_a": [1.0, 2.0, 3.0],
                "feat_b": [4.0, 5.0, 6.0],
            }
        )
        cols = feature_columns(df)
        assert "label" not in cols
        assert "split" not in cols
        assert cols == ["feat_a", "feat_b"]


def _attention_target_output() -> pd.DataFrame:
    """``attention-target`` over two individuals, with one focal frame facing no one."""
    facing = pd.DataFrame(
        {
            "frame": [0, 0, 1, 1],
            "id1": [0, 1, 0, 1],
            "id2": [1, 0, 1, 0],
            "angle_diff_deg": [5.0, 10.0, 7.0, 40.0],
            "is_facing": [True, True, True, False],
        }
    )
    return AttentionTarget().apply(facing)


class TestNeighborIdReservation:
    """A column that records another individual's track id is no measurement."""

    def test_feature_columns_excludes_the_nearest_neighbor_id(self) -> None:
        df = NearestNeighbor().apply(track_table("g", "s", n_rows=4, n_ids=3))
        assert pd.api.types.is_numeric_dtype(df["nn_id"])

        cols = feature_columns(df)

        assert "nn_id" not in cols
        assert cols == [
            "nn_delta_x",
            "nn_delta_x_ego",
            "nn_delta_y",
            "nn_delta_y_ego",
            "nn_dist",
        ]

    def test_feature_columns_excludes_the_attention_target_id(self) -> None:
        df = _attention_target_output()
        assert pd.api.types.is_numeric_dtype(df["attention_target_id"])
        assert df["attention_target_id"].isna().any()

        assert feature_columns(df) == ["attention_target_angle_diff_deg"]

    def test_a_merge_of_two_neighbor_ids_numbers_no_copy(self, tmp_path: Path) -> None:
        """The later input's ``nn_id`` is dropped, as a collided identity column is."""
        tracks = track_table("g", "s", n_rows=4, n_ids=3)
        specs: list[tuple[Path, LoadSpec]] = []
        for position in range(2):
            path = tmp_path / f"{position}.parquet"
            NearestNeighbor().apply(tracks).to_parquet(path)
            specs.append((path, ParquetLoadSpec()))

        merged = load_entry_data(specs)

        assert merged is not None
        assert "nn_id" in merged.columns
        assert not [column for column in merged.columns if column.startswith("nn_id_")]
        cols = feature_columns(merged)
        assert "nn_dist__1" in cols
        assert not [column for column in cols if column.startswith("nn_id")]


class TestParams:
    def test_requires_exactly_one_sampling_spec(self) -> None:
        with pytest.raises(ValueError, match="Exactly one"):
            ExtractLabeledTemplates.Params.from_overrides(
                {
                    "labels": GroundTruthLabelsSource(),
                }
            )

    def test_rejects_both_sampling_specs(self) -> None:
        with pytest.raises(ValueError, match="Exactly one"):
            ExtractLabeledTemplates.Params.from_overrides(
                {
                    "labels": GroundTruthLabelsSource(),
                    "n_per_class": 100,
                    "n_total": 500,
                }
            )

    def test_n_per_class_int(self) -> None:
        params = ExtractLabeledTemplates.Params.from_overrides(
            {
                "labels": GroundTruthLabelsSource(),
                "n_per_class": 100,
            }
        )
        assert params.n_per_class == 100

    def test_a_mapping_value_of_null_or_all_takes_every_frame_of_the_class(
        self,
    ) -> None:
        def params(per_class: dict[int, object]) -> ExtractLabeledTemplates.Params:
            return ExtractLabeledTemplates.Params.from_overrides(
                {"labels": GroundTruthLabelsSource(), "n_per_class": per_class}
            )

        assert params({0: None, 1: 5}).n_per_class == {0: None, 1: 5}
        assert params({0: "all", 1: 5}).n_per_class == {0: None, 1: 5}
        # One spelling, one run id.
        assert run_id_digest(params({0: "all"})) == run_id_digest(params({0: None}))
        with pytest.raises(ValueError, match="n_per_class"):
            params({0: "some"})

    def test_n_total(self) -> None:
        params = ExtractLabeledTemplates.Params.from_overrides(
            {
                "labels": GroundTruthLabelsSource(),
                "n_total": 500,
            }
        )
        assert params.n_total == 500

    def test_test_fraction_validation(self) -> None:
        with pytest.raises(Exception):
            ExtractLabeledTemplates.Params.from_overrides(
                {
                    "labels": GroundTruthLabelsSource(),
                    "n_total": 100,
                    "test_fraction": 1.5,
                }
            )

    def test_strategy_param(self) -> None:
        params = ExtractLabeledTemplates.Params.from_overrides(
            {
                "labels": GroundTruthLabelsSource(),
                "n_total": 100,
                "strategy": "farthest_first",
            }
        )
        assert params.strategy == "farthest_first"


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

        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_per_class": 20,
                "test_fraction": 0.0,
                "random_state": 42,
            },
        )

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
            entries_data.append(
                (f"g__{seq}", _make_labeled_df(n, sequence=seq, group="g"))
            )

        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior", entries_labels
        )
        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_total": 200,
                "test_fraction": 0.3,
                "random_state": 42,
            },
        )
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
        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_total": 40,
                "pool": {"allocation": "exact"},
                "test_fraction": 0.0,
            },
        )
        feat.fit(
            _make_factory([("g__s1", _make_labeled_df(n, sequence="s1", group="g"))])
        )

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
            feat = _make_feature_with_labels(
                labels_lookup,
                params={
                    "labels": {"kind": "behavior"},
                    "n_per_class": 15,
                    "test_fraction": 0.0,
                    "random_state": 123,
                },
            )
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
        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_per_class": {0: 10, 1: 30},
                "pool": {"allocation": "exact"},
                "test_fraction": 0.0,
            },
        )
        df = _make_labeled_df(n, sequence="s1", group="g")
        feat.fit(_make_factory([("g__s1", df)]))

        run_root = tmp_path / "run"
        feat.save_state(run_root)
        templates = pd.read_parquet(run_root / "templates.parquet")

        assert (templates["label"] == 0).sum() == 10
        assert (templates["label"] == 1).sum() == 30


class TestExactAllocationIsProportional:
    """Each sequence draws its share of a class's quota, not the quota's first rows."""

    def _drawn(
        self,
        tmp_path: Path,
        sequences: list[tuple[str, int, int]],
        params: dict[str, object],
    ) -> pd.DataFrame:
        """Fit ``(sequence, class 0 rows, class 1 rows)`` entries and return the sample."""
        labels = {
            name: np.array([0] * n0 + [1] * n1, dtype=np.int64)
            for name, n0, n1 in sequences
        }
        lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", name, array) for name, array in labels.items()],
        )
        feat = _make_feature_with_labels(
            lookup,
            params={
                "labels": {"kind": "behavior"},
                "pool": {"allocation": "exact"},
                "test_fraction": 0.0,
                **params,
            },
        )
        feat.fit(
            _make_factory(
                [
                    (f"g__{name}", _coded_df(name, "g", len(labels[name]), 0.0))
                    for name in labels
                ]
            )
        )
        feat.save_state(tmp_path / "run")
        return pd.read_parquet(tmp_path / "run" / "templates.parquet")

    @staticmethod
    def _shares(templates: pd.DataFrame, label: int) -> dict[str, int]:
        rows = templates[templates["label"] == label]
        return {str(k): int(v) for k, v in rows["sequence"].value_counts().items()}

    def test_equal_sequences_draw_equal_shares(self, tmp_path: Path) -> None:
        drawn = self._drawn(
            tmp_path,
            [("a", 100, 100), ("b", 100, 100), ("c", 100, 100)],
            {"n_per_class": 30},
        )

        assert self._shares(drawn, 0) == {"a": 10, "b": 10, "c": 10}
        assert self._shares(drawn, 1) == {"a": 10, "b": 10, "c": 10}

    def test_shares_follow_the_rows_each_sequence_holds(self, tmp_path: Path) -> None:
        drawn = self._drawn(
            tmp_path,
            [("a", 100, 20), ("b", 50, 20), ("c", 50, 60)],
            {"n_per_class": {0: 40, 1: 50}},
        )

        assert self._shares(drawn, 0) == {"a": 20, "b": 10, "c": 10}
        assert self._shares(drawn, 1) == {"a": 10, "b": 10, "c": 30}

    def test_a_remainder_goes_to_the_larger_fractions_and_the_total_is_exact(
        self, tmp_path: Path
    ) -> None:
        drawn = self._drawn(
            tmp_path,
            [("a", 100, 100), ("b", 100, 100), ("c", 100, 100)],
            {"n_per_class": 10},
        )

        for label in (0, 1):
            shares = self._shares(drawn, label)
            assert sum(shares.values()) == 10
            assert sorted(shares.values()) == [3, 3, 4]

    def test_a_sequence_never_draws_more_than_it_holds(self, tmp_path: Path) -> None:
        drawn = self._drawn(
            tmp_path,
            [("a", 1, 50), ("b", 1, 50), ("c", 98, 50)],
            {"n_per_class": {0: 3, 1: 30}},
        )

        assert self._shares(drawn, 0) == {"c": 3}
        assert self._shares(drawn, 1) == {"a": 10, "b": 10, "c": 10}

    def test_a_quota_above_the_rows_takes_every_row(self, tmp_path: Path) -> None:
        drawn = self._drawn(
            tmp_path,
            [("a", 7, 5), ("b", 9, 3)],
            {"n_per_class": 100},
        )

        assert self._shares(drawn, 0) == {"a": 7, "b": 9}
        assert self._shares(drawn, 1) == {"a": 5, "b": 3}

    def test_the_draw_is_deterministic(self, tmp_path: Path) -> None:
        sequences = [("a", 100, 100), ("b", 60, 30)]
        first = self._drawn(tmp_path / "one", sequences, {"n_per_class": 25})
        second = self._drawn(tmp_path / "two", sequences, {"n_per_class": 25})

        pd.testing.assert_frame_equal(first, second)

    def test_the_train_and_the_test_split_each_draw_their_own_shares(
        self, tmp_path: Path
    ) -> None:
        drawn = self._drawn(
            tmp_path,
            [(name, 100, 100) for name in ("a", "b", "c", "d")],
            {"n_per_class": 20, "test_fraction": 0.5},
        )

        for split in ("train", "test"):
            rows = drawn[(drawn["split"] == split) & (drawn["label"] == 0)]
            assert len(rows) == 20
            assert set(rows["sequence"].value_counts()) == {10}
            assert rows["sequence"].nunique() == 2


@pytest.mark.parametrize("allocation", ["reservoir", "exact"])
class TestEveryFrameOfAClass:
    """A null count takes every frame of its class, and the others keep their cap."""

    def _drawn(
        self, tmp_path: Path, allocation: str, params: dict[str, object]
    ) -> pd.DataFrame:
        labels = np.array([0] * 70 + [1] * 70 + [2] * 10)
        lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "a", labels), ("g", "b", labels)],
        )
        feat = _make_feature_with_labels(
            lookup,
            params={
                "labels": {"kind": "behavior"},
                "test_fraction": 0.0,
                "pool": {"allocation": allocation},
                **params,
            },
        )
        feat.fit(
            _make_factory(
                [
                    ("g__a", _coded_df("a", "g", 150, 0.0)),
                    ("g__b", _coded_df("b", "g", 150, 1000.0)),
                ]
            )
        )
        feat.save_state(tmp_path / "run")
        return pd.read_parquet(tmp_path / "run" / "templates.parquet")

    def test_the_class_keeps_every_frame_and_the_others_their_cap(
        self, tmp_path: Path, allocation: str
    ) -> None:
        drawn = self._drawn(
            tmp_path, allocation, {"n_per_class": {0: 25, 1: None, 2: None}}
        )

        counts = drawn["label"].value_counts().to_dict()
        assert counts == {0: 25, 1: 140, 2: 20}
        every = drawn[drawn["label"] == 1]
        assert len(every.drop_duplicates(subset=["feat_0"])) == 140

    def test_a_class_the_mapping_leaves_out_contributes_nothing(
        self, tmp_path: Path, allocation: str
    ) -> None:
        drawn = self._drawn(tmp_path, allocation, {"n_per_class": {1: None}})

        assert set(drawn["label"]) == {1}
        assert len(drawn) == 140


def test_a_target_far_above_the_rows_a_class_holds_allocates_only_those_rows(
    tmp_path: Path,
) -> None:
    """The reservoir grows with the rows it receives, not to its target at once.

    A target of five million rows of 3 columns is 120 MB per class if allocated up
    front, though the class holds fifty rows.
    """
    import tracemalloc

    labels = np.array([0] * 50 + [1] * 50)
    lookup = _setup_labels_lookup(
        tmp_path / "labels" / "behavior", [("g", "a", labels)]
    )
    feat = _make_feature_with_labels(
        lookup,
        params={
            "labels": {"kind": "behavior"},
            "n_per_class": 5_000_000,
            "test_fraction": 0.0,
        },
    )
    entry = _make_factory([("g__a", _coded_df("a", "g", 100, 0.0))])

    tracemalloc.start()
    try:
        feat.fit(entry)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 10_000_000
    feat.save_state(tmp_path / "run")
    assert len(pd.read_parquet(tmp_path / "run" / "templates.parquet")) == 100


@pytest.mark.parametrize("allocation", ["reservoir", "exact"])
def test_a_draw_that_will_not_fit_in_memory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, allocation: str
) -> None:
    labels = np.array([0] * 50 + [1] * 50)
    lookup = _setup_labels_lookup(
        tmp_path / "labels" / "behavior", [("g", "a", labels)]
    )
    feat = _make_feature_with_labels(
        lookup,
        params={
            "labels": {"kind": "behavior"},
            "n_per_class": {0: None, 1: None},
            "test_fraction": 0.0,
            "pool": {"allocation": allocation},
        },
    )
    monkeypatch.setattr(helpers, "host_memory_bytes", lambda: (2000, 1000))

    with pytest.raises(
        MemoryError, match=r"\[extract-labeled-templates\] .*n_per_class"
    ):
        feat.fit(_make_factory([("g__a", _coded_df("a", "g", 100, 0.0))]))


class TestFarthestFirstLabeled:
    def test_farthest_first_per_class(self, tmp_path: Path) -> None:
        """Farthest-first should select spread points within each class."""
        n = 200
        labels = np.array([0] * 100 + [1] * 100)
        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", labels)],
        )
        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_per_class": 10,
                "strategy": "farthest_first",
                "pool": {"size": 50},
                "test_fraction": 0.0,
            },
        )
        df = _make_labeled_df(n, sequence="s1", group="g")
        feat.fit(_make_factory([("g__s1", df)]))

        run_root = tmp_path / "run"
        feat.save_state(run_root)
        templates = pd.read_parquet(run_root / "templates.parquet")

        assert (templates["label"] == 0).sum() == 10
        assert (templates["label"] == 1).sum() == 10

    def test_rows_holding_nan_leave_the_selection_distinct(
        self, tmp_path: Path
    ) -> None:
        """A NaN cell is compared as missing, and the selection does not repeat a row.

        Every tenth row holds one NaN. Each distance to such a row was NaN, and
        argmax then chose row 0 for every later pick.
        """
        n = 200
        labels = np.array([0] * 100 + [1] * 100)
        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", labels)],
        )
        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_per_class": 20,
                "strategy": "farthest_first",
                "pool": {"size": 80},
                "test_fraction": 0.0,
            },
        )
        df = _make_labeled_df(n, sequence="s1", group="g")
        df.loc[df.index % 10 == 0, "feat_1"] = np.nan
        feat.fit(_make_factory([("g__s1", df)]))

        run_root = tmp_path / "run"
        feat.save_state(run_root)
        templates = pd.read_parquet(run_root / "templates.parquet")

        assert len(templates) == 40
        assert not templates.duplicated().any()


class TestSaveLoadLabeled:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        n = 100
        labels = np.array([0] * 50 + [1] * 50)
        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", labels)],
        )
        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_per_class": 15,
                "test_fraction": 0.0,
            },
        )
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

    def test_apply_writes_an_empty_table_and_the_sample_is_saved(
        self, tmp_path: Path
    ) -> None:
        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels", [("g", "s1", np.array([0] * 10 + [1] * 10))]
        )
        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_per_class": 5,
                "test_fraction": 0.0,
            },
        )
        df = _make_labeled_df(20, sequence="s1", group="g")
        feat.fit(_make_factory([("g__s1", df)]))
        feat.save_state(tmp_path / "run")

        result = feat.apply(df)
        assert result.empty
        assert result.columns.empty
        templates = pd.read_parquet(tmp_path / "run" / "templates.parquet")
        assert sorted(templates["label"]) == [0] * 5 + [1] * 5
        assert (templates["split"] == "train").all()
        provenance = pd.read_parquet(tmp_path / "run" / "template_provenance.parquet")
        assert provenance[["group", "sequence", "split"]].values.tolist() == [
            ["g", "s1", "train"]
        ]


def _sparse_labels(
    path: Path, events: list[tuple[int, int, int, int]]
) -> dict[tuple[str, str], Path]:
    """A CalMS21-shaped file of ``(frame, id1, id2, label)`` events, no background."""
    return {("g", "s"): write_sparse_labels(path, events)}


@pytest.mark.parametrize("allocation", ["reservoir", "exact"])
def test_a_frame_without_a_label_event_yields_no_template(
    tmp_path: Path, allocation: str
) -> None:
    """Twenty frames, ten annotated: the other ten are unlabeled, not class 0."""
    lookup = _sparse_labels(
        tmp_path / "labels" / "s.npz", [(f, 0, -1, 1) for f in range(10)]
    )
    feat = _make_feature_with_labels(
        lookup,
        params={
            "labels": {"kind": "behavior"},
            "n_per_class": 100,
            "test_fraction": 0.0,
            "pool": {"allocation": allocation},
        },
    )

    feat.fit(_make_factory([("g__s", _make_labeled_df(20, sequence="s", group="g"))]))
    feat.save_state(tmp_path / "run")

    templates = pd.read_parquet(tmp_path / "run" / "templates.parquet")
    assert templates["label"].tolist() == [1] * 10


@pytest.mark.parametrize("allocation", ["reservoir", "exact"])
def test_only_the_labeled_perspective_is_sampled(
    tmp_path: Path, allocation: str
) -> None:
    """A resident -> intruder event labels the resident-focal pair row only."""
    n_frames = 8
    lookup = _sparse_labels(
        tmp_path / "labels" / "s.npz", [(f, 0, 1, 0) for f in range(n_frames)]
    )
    rng = np.random.default_rng(0)
    pairs = pd.DataFrame(
        {
            "frame": np.repeat(np.arange(n_frames), 2),
            "id1": [0, 1] * n_frames,
            "id2": [1, 0] * n_frames,
            "group": "g",
            "sequence": "s",
            "feat_0": rng.standard_normal(2 * n_frames),
        }
    )
    feat = _make_feature_with_labels(
        lookup,
        params={
            "labels": {"kind": "behavior"},
            "n_per_class": 100,
            "test_fraction": 0.0,
            "pool": {"allocation": allocation},
        },
    )

    feat.fit(_make_factory([("g__s", pairs)]))
    feat.save_state(tmp_path / "run")

    templates = pd.read_parquet(tmp_path / "run" / "templates.parquet")
    resident_focal = pairs.loc[pairs["id1"] == 0, "feat_0"]
    assert sorted(templates["feat_0"]) == sorted(resident_focal)
    assert templates["label"].tolist() == [0] * n_frames


def _coded_df(sequence: str, group: str, n_rows: int, code: float) -> pd.DataFrame:
    """A table whose ``feat_0`` says which entry a row came from: ``code`` plus its frame."""
    return pd.DataFrame(
        {
            "frame": np.arange(n_rows),
            "time": np.arange(n_rows, dtype=float) / 30.0,
            "id": np.zeros(n_rows, dtype=int),
            "group": [group] * n_rows,
            "sequence": [sequence] * n_rows,
            "feat_0": code + np.arange(n_rows, dtype=float),
            "feat_1": np.zeros(n_rows),
        }
    )


class TestIdentityColumns:
    """Every template row names the entry it was drawn from."""

    ENTRIES = (("", "10", 1000.0), ("g", "s2", 2000.0), ("g", "s3", 3000.0))

    def _templates(self, tmp_path: Path, params: dict[str, object]) -> pd.DataFrame:
        labels = np.array([0] * 20 + [1] * 20)
        lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [(group, sequence, labels) for group, sequence, _ in self.ENTRIES],
        )
        feat = _make_feature_with_labels(
            lookup,
            params={"labels": {"kind": "behavior"}, "test_fraction": 0.0, **params},
        )
        feat.fit(
            _make_factory(
                [
                    (f"{group}__{sequence}", _coded_df(sequence, group, 40, code))
                    for group, sequence, code in self.ENTRIES
                ]
            )
        )
        feat.save_state(tmp_path / "run")
        return pd.read_parquet(tmp_path / "run" / "templates.parquet")

    @pytest.mark.parametrize(
        "params",
        [
            {"n_per_class": 15},
            {"n_per_class": 60, "pool": {"allocation": "exact"}},
            {"n_per_class": 6, "strategy": "farthest_first", "pool": {"size": 15}},
        ],
        ids=["reservoir", "exact", "farthest_first"],
    )
    def test_each_row_names_its_own_entry(
        self, tmp_path: Path, params: dict[str, object]
    ) -> None:
        templates = self._templates(tmp_path, params)

        code_of = {(g, s): code for g, s, code in self.ENTRIES}
        assert set(zip(templates["group"], templates["sequence"])) == set(code_of)
        for (group, sequence), rows in templates.groupby(["group", "sequence"]):
            code = code_of[(str(group), str(sequence))]
            assert rows["feat_0"].between(code, code + 39).all()

    def test_the_identity_is_text_and_a_blank_group_stays_blank(
        self, tmp_path: Path
    ) -> None:
        templates = self._templates(tmp_path, {"n_per_class": 15})

        assert not pd.api.types.is_numeric_dtype(templates["group"])
        assert not pd.api.types.is_numeric_dtype(templates["sequence"])
        assert set(templates["group"]) == {"", "g"}
        assert set(templates["sequence"]) == {"10", "s2", "s3"}

    def test_the_identity_is_not_a_feature(self, tmp_path: Path) -> None:
        templates = self._templates(tmp_path, {"n_per_class": 15})

        assert feature_columns(templates) == ["feat_0", "feat_1"]

    def test_the_identity_survives_a_save_after_a_load(self, tmp_path: Path) -> None:
        _ = self._templates(tmp_path, {"n_per_class": 15})
        feat = ExtractLabeledTemplates(
            ExtractLabeledTemplates.Inputs((Result(feature="upstream"),)),
            params={"labels": {"kind": "behavior"}, "n_per_class": 15},
        )

        assert feat.load_state(tmp_path / "run", {}, {}) is True
        feat.save_state(tmp_path / "reload")

        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "run" / "templates.parquet"),
            pd.read_parquet(tmp_path / "reload" / "templates.parquet"),
        )


class TestColumns:
    """``columns`` names the payload columns that the templates sample."""

    def _templates(
        self, tmp_path: Path, run: str, params: dict[str, object]
    ) -> pd.DataFrame:
        """Fit on one sequence of two 25-row classes, keeping every row."""
        labels_lookup = _setup_labels_lookup(
            tmp_path / "labels" / "behavior",
            [("g", "s1", np.array([0] * 25 + [1] * 25))],
        )
        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_per_class": 25,
                "test_fraction": 0.0,
                **params,
            },
        )
        df = _make_labeled_df(50, sequence="s1", group="g")
        feat.fit(_make_factory([("g__s1", df)]))
        feat.save_state(tmp_path / run)
        return pd.read_parquet(tmp_path / run / "templates.parquet")

    def test_the_templates_hold_only_the_named_columns(self, tmp_path: Path) -> None:
        every = self._templates(tmp_path, "every", {})
        named = self._templates(tmp_path, "named", {"columns": ["feat_2", "feat_0"]})

        assert list(every.columns) == [
            "feat_0",
            "feat_1",
            "feat_2",
            "label",
            "split",
            "group",
            "sequence",
        ]
        kept = ["feat_0", "feat_2", "label", "split", "group", "sequence"]
        pd.testing.assert_frame_equal(named, every[kept])

    @pytest.mark.parametrize(
        ("named", "reason"),
        [
            (["feat_0", "split"], "Identity or bookkeeping columns: ['split']."),
            (["feat_0", "feat_9"], "Not a numeric column of the input: ['feat_9']."),
        ],
        ids=["bookkeeping", "absent"],
    )
    def test_a_name_outside_the_payload_is_refused(
        self, tmp_path: Path, named: list[str], reason: str
    ) -> None:
        feat = _make_feature_with_labels(
            {},
            params={"labels": {"kind": "behavior"}, "n_per_class": 5, "columns": named},
        )
        df = _make_labeled_df(10, sequence="s1", group="g")

        with pytest.raises(ValueError, match="columns may name only payload") as raised:
            feat.fit(_make_factory([("g__s1", df)]))

        assert reason in str(raised.value)
        payload = "The input's payload columns: ['feat_0', 'feat_1', 'feat_2']."
        assert payload in str(raised.value)

    def test_the_order_and_repeats_of_the_names_leave_the_run_id(self) -> None:
        def digest(columns: list[str]) -> str:
            params = ExtractLabeledTemplates.Params.from_overrides(
                {
                    "labels": GroundTruthLabelsSource(),
                    "n_total": 10,
                    "columns": columns,
                }
            )
            return run_id_digest(params)

        assert digest(["feat_2", "feat_0", "feat_2"]) == digest(["feat_0", "feat_2"])
        assert digest(["feat_0"]) != digest(["feat_0", "feat_2"])


class TestEdgeCasesLabeled:
    def test_empty_inputs_raises(self, tmp_path: Path) -> None:
        labels_lookup = _setup_labels_lookup(tmp_path / "labels" / "behavior", [])
        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_per_class": 10,
            },
        )
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
        feat = _make_feature_with_labels(
            labels_lookup,
            params={
                "labels": {"kind": "behavior"},
                "n_per_class": 20,
                "test_fraction": 0.0,
            },
        )
        df = _make_labeled_df(n, sequence="s1", group="g")
        feat.fit(_make_factory([("g__s1", df)]))

        run_root = tmp_path / "run"
        feat.save_state(run_root)
        templates = pd.read_parquet(run_root / "templates.parquet")
        assert set(templates["label"].unique()) == {0}
        assert len(templates) == 20
