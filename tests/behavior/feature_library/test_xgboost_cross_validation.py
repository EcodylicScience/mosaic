"""`xgboost` cross-validates by sequence: each training sequence is scored by a model that never saw it.

With `cv_folds` set, one fit holds K fold models beside the full model, and `apply`
scores a training sequence with the fold model whose training rows exclude it. These
tests build a small world of sequences, fit it, and check the property from every
side: which rows each model saw, which model scored which entry, what a rerun and a
kill leave on disk, and what is refused.

A world's template rows carry a `code` column that names their entry
(`1000 * entry + row`). XGBoost fits on it like any column, and a spy on `fit` reads
from it which sequences each model was fitted on.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score  # pyright: ignore[reportUnknownVariableType]
from sklearn.model_selection import LeaveOneGroupOut
from xgboost import XGBClassifier

from mosaic.behavior.feature_library import helpers
from mosaic.behavior.feature_library.classifier_common import (
    block_fold_table,
    deal_blocks,
    read_fold_table,
)
from mosaic.behavior.feature_library.helpers import cv_fold_assignment
from mosaic.behavior.feature_library.xgboost_feature import XgboostFeature
from mosaic.core.pipeline.job import Cancelled, CancelToken
from mosaic.core.pipeline.progress import NullProgressCallback
from mosaic.core.pipeline.run import process_apply_worker
from mosaic.core.pipeline.types import InputStream, Result
from tests.helpers import run_id_digest

Entry = tuple[str, str]
SIGNAL = 3.0
NO_INPUT = InputStream(lambda: iter([]), n_entries=0)


@dataclass
class World:
    """Sequences with labeled rows, the templates drawn from them, and their tables."""

    path: Path
    frames: dict[Entry, pd.DataFrame]
    labels: dict[Entry, np.ndarray]
    train: list[Entry]

    def rows(self, entries: Sequence[Entry]) -> pd.DataFrame:
        return pd.concat([self.frames[e] for e in entries], ignore_index=True)


def make_world(
    root: Path,
    train: Mapping[Entry, Mapping[int, int]],
    *,
    test_split: Mapping[Entry, Mapping[int, int]] | None = None,
    outside: Mapping[Entry, Mapping[int, int]] | None = None,
    empty: Mapping[Entry, Mapping[int, int]] | None = None,
    n_features: int = 3,
    seed: int = 0,
    constant_per_entry: bool = False,
    shuffle_provenance: bool = False,
    with_frames: bool = False,
    interleave: bool = False,
) -> World:
    """Write `templates.parquet` and `template_provenance.parquet` into *root*.

    Each mapping is `entry -> {class: rows}`. `train` and `test_split` entries
    contribute template rows to their split. `empty` entries are train entries of the
    provenance that drew no template row, and `outside` entries are in no templates
    run, as the entries of a test group are not. All four have a table to score.
    With `constant_per_entry`, every feature of an entry is one constant. With
    `with_frames`, the templates keep each row's `frame` and the provenance records
    each entry's labeled frames, as `extract-labeled-templates` 0.7 does. With
    `interleave`, the classes of an entry alternate in frame order and are not
    one stretch each.
    """
    rng = np.random.default_rng(seed)
    frames: dict[Entry, pd.DataFrame] = {}
    labels: dict[Entry, np.ndarray] = {}
    template_rows: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    groups = {
        "train": train,
        "test": test_split or {},
        "outside": outside or {},
        "empty": empty or {},
    }
    entry_index = 0
    for kind, entries in groups.items():
        for entry, per_class in entries.items():
            label = np.concatenate(
                [np.full(n, c, dtype=np.int64) for c, n in sorted(per_class.items())]
            )
            if interleave:
                label = rng.permutation(label)
            n = len(label)
            code = entry_index * 1000.0 + np.arange(n)
            if constant_per_entry:
                features = np.tile(rng.standard_normal(n_features), (n, 1))
            else:
                features = rng.standard_normal((n, n_features))
                features[:, 0] += label * SIGNAL
                features[:, 1] -= label * SIGNAL / 2
            frame = pd.DataFrame(
                {
                    "frame": np.arange(n),
                    "time": np.arange(n) / 30.0,
                    "id": np.zeros(n, dtype=int),
                    "group": [entry[0]] * n,
                    "sequence": [entry[1]] * n,
                    "code": code,
                    **{f"feat_{i}": features[:, i] for i in range(n_features)},
                }
            )
            frames[entry], labels[entry] = frame, label
            entry_index += 1
            if kind in ("train", "test"):
                rows = frame.copy()
                rows["label"] = label
                rows["split"] = kind
                template_rows.append(rows)
            if kind in ("train", "test", "empty"):
                provenance.append(
                    {
                        "group": entry[0],
                        "sequence": entry[1],
                        "split": "test" if kind == "test" else "train",
                        "count": 0 if kind == "empty" else n,
                        "proportion": 0.0,
                        **(
                            {"labeled_start": 0, "labeled_stop": n}
                            if with_frames
                            else {}
                        ),
                    }
                )
    templates = pd.concat(template_rows, ignore_index=True)
    templates = templates.drop(
        columns=["time", "id"] if with_frames else ["frame", "time", "id"]
    )
    root.mkdir(parents=True, exist_ok=True)
    templates.to_parquet(root / "templates.parquet", index=False)
    table = pd.DataFrame(provenance)
    if shuffle_provenance:
        table = table.sample(frac=1.0, random_state=7).reset_index(drop=True)
    table.to_parquet(root / "template_provenance.parquet", index=False)
    return World(root / "templates.parquet", frames, labels, list(train))


def sequences(
    n: int, per_class: Mapping[int, int], prefix: str = "s"
) -> dict[Entry, Mapping[int, int]]:
    return {("g", f"{prefix}{i:02d}"): per_class for i in range(n)}


PARAMS: dict[str, object] = {
    "templates": {"feature": "extract-labeled-templates"},
    "default_class": 0,
    "n_estimators": 8,
    "max_depth": 3,
    "n_jobs": 1,
}


def make_feature(world: World, run_root: Path, **params: object) -> XgboostFeature:
    feature = XgboostFeature(
        XgboostFeature.Inputs((Result(feature="upstream"),)),
        params={**PARAMS, **params},
    )
    assert feature.load_state(run_root, {"templates": world.path}, {}) is False
    return feature


def fitted(
    world: World, run_root: Path, **params: object
) -> tuple[XgboostFeature, dict[str, Any]]:
    feature = make_feature(world, run_root, **params)
    feature.fit(NO_INPUT)
    return feature, vars(feature)


def sidecar(run_root: Path) -> pd.DataFrame:
    return pd.read_parquet(run_root / "cv_folds.parquet")


@dataclass
class FitCall:
    model: XGBClassifier
    x: np.ndarray
    y: np.ndarray
    sample_weight: np.ndarray | None

    @property
    def entries(self) -> set[int]:
        """The entry numbers of the rows this fit saw, from their `code` column."""
        return {int(code // 1000) for code in self.x[:, 0]}


@pytest.fixture
def fits(monkeypatch: pytest.MonkeyPatch) -> list[FitCall]:
    """Every `XGBClassifier.fit` of the test, with the rows it was handed."""
    calls: list[FitCall] = []
    original = XGBClassifier.fit

    def spy(self: XGBClassifier, X: Any, y: Any, **kwargs: Any) -> XGBClassifier:
        weight = kwargs.get("sample_weight")
        calls.append(
            FitCall(
                self,
                np.array(X, copy=True),
                np.array(y, copy=True),
                None if weight is None else np.array(weight, copy=True),
            )
        )
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(XGBClassifier, "fit", spy)
    return calls


def prob_columns(out: pd.DataFrame) -> list[str]:
    return [c for c in out.columns if c.startswith("prob_")]


def test_params_default_to_no_cross_validation_and_the_library_threads() -> None:
    params = XgboostFeature.Params.from_overrides(
        {"templates": PARAMS["templates"], "default_class": 0}
    )

    assert params.cv_folds is None
    assert params.cv_seed == 0
    assert params.n_jobs is None
    assert params.device == "cpu"


class TestParams:
    def test_cv_folds_is_at_least_two(self) -> None:
        with pytest.raises(ValueError, match="cv_folds"):
            XgboostFeature.Params.from_overrides({**PARAMS, "cv_folds": 1})

    def test_cv_folds_is_refused_beside_a_model(self) -> None:
        with pytest.raises(ValueError, match="cv_folds.*templates.*model"):
            XgboostFeature.Params.from_overrides(
                {
                    "model": {"feature": "xgboost", "run_id": "0.2-abc"},
                    "default_class": 0,
                    "cv_folds": 3,
                }
            )

    def test_a_cv_seed_without_cv_folds_is_refused(self) -> None:
        """The same output would otherwise carry a second run id."""
        with pytest.raises(ValueError, match="cv_seed.*cv_folds"):
            XgboostFeature.Params.from_overrides({**PARAMS, "cv_seed": 3})

    def test_cv_folds_and_cv_seed_enter_the_run_id(self) -> None:
        def digest(**extra: object) -> str:
            return run_id_digest(
                XgboostFeature.Params.from_overrides({**PARAMS, **extra})
            )

        assert digest(cv_folds=3) != digest()
        assert digest(cv_folds=3) != digest(cv_folds=4)
        assert digest(cv_folds=3, cv_seed=1) != digest(cv_folds=3)

    def test_n_jobs_is_not_part_of_the_run_id(self) -> None:
        def digest(**extra: object) -> str:
            return run_id_digest(
                XgboostFeature.Params.from_overrides({**PARAMS, **extra})
            )

        assert digest(n_jobs=1) == digest(n_jobs=4) == digest(n_jobs=None)

    def test_device_enters_the_run_id(self) -> None:
        def digest(**extra: object) -> str:
            return run_id_digest(
                XgboostFeature.Params.from_overrides({**PARAMS, **extra})
            )

        assert digest(device="cuda") != digest(device="cpu") == digest()

    def test_device_is_cpu_or_cuda(self) -> None:
        with pytest.raises(ValueError, match="device"):
            XgboostFeature.Params.from_overrides({**PARAMS, "device": "gpu"})

    def test_a_class_weight_may_be_balanced_none_or_a_mapping(self) -> None:
        def weight(value: object) -> object:
            return XgboostFeature.Params.from_overrides(
                {**PARAMS, "class_weight": value}
            ).class_weight

        assert weight("balanced") == "balanced"
        assert weight(None) is None
        assert weight({0: 1.0, 2: 3.5}) == {0: 1.0, 2: 3.5}
        with pytest.raises(ValueError, match="class_weight"):
            weight({0: 0.0})
        with pytest.raises(ValueError, match="class_weight"):
            weight("sqrt")


class TestFolds:
    def _world(self, tmp_path: Path, **kwargs: Any) -> World:
        return make_world(
            tmp_path / "templates",
            sequences(10, {0: 12, 1: 12, 2: 12}),
            **kwargs,
        )

    def test_the_folds_partition_the_train_entries_of_the_provenance(
        self, tmp_path: Path
    ) -> None:
        world = self._world(
            tmp_path,
            test_split=sequences(2, {0: 6, 1: 6, 2: 6}, prefix="t"),
            empty={("g", "drew-nothing"): {0: 5}},
        )
        feature, _ = fitted(world, tmp_path / "run", cv_folds=4)
        feature.save_state(tmp_path / "run")

        folds = sidecar(tmp_path / "run")

        entries = [*world.train, ("g", "drew-nothing")]
        assert sorted(zip(folds["group"], folds["sequence"])) == sorted(entries)
        sizes = folds["fold"].value_counts()
        assert sizes.max() - sizes.min() <= 1
        assert set(folds["fold"]) == {0, 1, 2, 3}
        assert dict(zip(zip(folds["group"], folds["sequence"]), folds["fold"])) == (
            cv_fold_assignment(entries, 4, seed=0)
        )

    def test_cv_seed_reaches_the_deal_and_changes_the_folds_the_sidecar_reports(
        self, tmp_path: Path
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(12, {0: 8, 1: 8}))
        dealt: dict[int, dict[Entry, int]] = {}

        for seed in (0, 1, 7):
            feature, _ = fitted(
                world, tmp_path / f"run{seed}", cv_folds=3, cv_seed=seed
            )
            feature.save_state(tmp_path / f"run{seed}")
            folds = sidecar(tmp_path / f"run{seed}")
            dealt[seed] = dict(
                zip(zip(folds["group"], folds["sequence"]), folds["fold"])
            )
            assert dealt[seed] == cv_fold_assignment(world.train, 3, seed=seed)

        assert dealt[0] != dealt[1] != dealt[7]

    def test_the_sidecar_counts_the_rows_of_each_class_in_each_entry(
        self, tmp_path: Path
    ) -> None:
        world = make_world(
            tmp_path / "templates",
            {
                ("g", "a"): {0: 10, 1: 4},
                ("g", "b"): {0: 3, 1: 9},
                ("g", "c"): {0: 2, 1: 7},
            },
            empty={("g", "none"): {1: 2}},
        )
        feature, _ = fitted(world, tmp_path / "run", cv_folds=2)
        feature.save_state(tmp_path / "run")

        folds = sidecar(tmp_path / "run")

        assert list(folds.columns) == [
            "group",
            "sequence",
            "fold",
            "n_rows",
            "rows_0",
            "rows_1",
        ]
        counts = {
            str(row["sequence"]): [int(row[c]) for c in ("n_rows", "rows_0", "rows_1")]
            for row in folds.to_dict("records")
        }
        assert counts == {
            "a": [14, 10, 4],
            "b": [12, 3, 9],
            "c": [9, 2, 7],
            "none": [0, 0, 0],
        }

    def test_the_names_a_b_c_and_ab_c_are_two_entries(self, tmp_path: Path) -> None:
        world = make_world(
            tmp_path / "templates",
            {
                ("a", "bc"): {0: 8, 1: 8},
                ("ab", "c"): {0: 8, 1: 8},
                ("", "x"): {0: 8, 1: 8},
                ("x", ""): {0: 8, 1: 8},
            },
        )
        feature, _ = fitted(world, tmp_path / "run", cv_folds=2)
        feature.save_state(tmp_path / "run")

        folds = sidecar(tmp_path / "run")

        assert len(folds) == 4
        assert len(folds.drop_duplicates(["group", "sequence"])) == 4

    def test_the_order_of_the_provenance_changes_nothing(self, tmp_path: Path) -> None:
        def folds(root: str, shuffle: bool) -> pd.DataFrame:
            world = make_world(
                tmp_path / root,
                sequences(9, {0: 8, 1: 8}),
                shuffle_provenance=shuffle,
            )
            feature, _ = fitted(world, tmp_path / f"{root}-run", cv_folds=3, cv_seed=5)
            feature.save_state(tmp_path / f"{root}-run")
            return sidecar(tmp_path / f"{root}-run")

        pd.testing.assert_frame_equal(folds("ordered", False), folds("shuffled", True))

    def test_another_random_state_leaves_the_folds_where_they_are(
        self, tmp_path: Path
    ) -> None:
        """A new sample can leave a sequence without rows. The folds still deal it."""
        full = make_world(tmp_path / "one", sequences(8, {0: 8, 1: 8}))
        thinner = make_world(
            tmp_path / "two",
            {e: c for e, c in sequences(8, {0: 8, 1: 8}).items() if e[1] != "s03"},
            empty={("g", "s03"): {0: 8, 1: 8}},
        )
        first, _ = fitted(full, tmp_path / "run1", cv_folds=3, random_state=1)
        second, _ = fitted(thinner, tmp_path / "run2", cv_folds=3, random_state=2)
        first.save_state(tmp_path / "run1")
        second.save_state(tmp_path / "run2")

        key = ["group", "sequence", "fold"]
        pd.testing.assert_frame_equal(
            sidecar(tmp_path / "run1")[key], sidecar(tmp_path / "run2")[key]
        )
        assert (
            sidecar(tmp_path / "run2").set_index("sequence").loc["s03", "n_rows"] == 0
        )


class TestHeldOut:
    def test_no_model_is_fitted_on_a_row_of_the_sequence_it_scores(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = make_world(
            tmp_path / "templates",
            sequences(10, {0: 12, 1: 12, 2: 12}),
            test_split=sequences(2, {0: 6, 1: 6, 2: 6}, prefix="t"),
            empty={("g", "drew-nothing"): {0: 5}},
        )
        order = [*world.frames]
        number = {entry: i for i, entry in enumerate(order)}
        feature, _ = fitted(world, tmp_path / "run", cv_folds=4)
        feature.save_state(tmp_path / "run")
        folds = sidecar(tmp_path / "run")
        fold_of = dict(zip(zip(folds["group"], folds["sequence"]), folds["fold"]))

        assert len(fits) == 4 + 1
        train_numbers = {number[e] for e in world.train}
        for fold in range(4):
            held_out = {number[e] for e, f in fold_of.items() if f == fold}
            assert fits[fold].entries.isdisjoint(held_out)
            assert fits[fold].entries == train_numbers - held_out
        assert fits[4].entries == train_numbers
        test_numbers = {number[("g", "t00")], number[("g", "t01")]}
        assert all(call.entries.isdisjoint(test_numbers) for call in fits)

    def test_a_train_entry_is_scored_by_its_fold_and_never_by_the_full_model(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = make_world(
            tmp_path / "templates",
            sequences(8, {0: 12, 1: 12, 2: 12}),
            empty={("g", "drew-nothing"): {0: 5, 1: 5}},
            outside={("test", "z"): {0: 9, 1: 9, 2: 9}},
        )
        feature, state = fitted(world, tmp_path / "run", cv_folds=3)
        feature.save_state(tmp_path / "run")
        folds = sidecar(tmp_path / "run")
        full = fits[-1].model

        for _, row in folds.iterrows():
            entry = (str(row["group"]), str(row["sequence"]))
            frame = world.frames[entry]
            out = feature.apply(frame)
            columns: list[str] = state["_feature_columns"]
            x = frame[columns].to_numpy(dtype=np.float64)
            fold_model = fits[int(row["fold"])].model
            np.testing.assert_allclose(
                out[prob_columns(out)].to_numpy(), fold_model.predict_proba(x)
            )
            assert not np.allclose(
                out[prob_columns(out)].to_numpy(), full.predict_proba(x)
            )

    def test_an_entry_outside_the_templates_is_scored_by_the_full_model(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = make_world(
            tmp_path / "templates",
            sequences(8, {0: 12, 1: 12}),
            test_split={("g", "held-back"): {0: 6, 1: 6}},
            outside={("test", "z"): {0: 9, 1: 9}},
        )
        feature, state = fitted(world, tmp_path / "run", cv_folds=3)

        for entry in (("test", "z"), ("g", "held-back")):
            frame = world.frames[entry]
            out = feature.apply(frame)
            columns: list[str] = state["_feature_columns"]
            x = frame[columns].to_numpy(dtype=np.float64)
            np.testing.assert_allclose(
                out[prob_columns(out)].to_numpy(), fits[-1].model.predict_proba(x)
            )

    def test_the_full_model_is_the_model_a_run_without_folds_fits(
        self, tmp_path: Path
    ) -> None:
        world = make_world(
            tmp_path / "templates",
            sequences(8, {0: 12, 1: 12, 2: 12}),
            outside={("test", "z"): {0: 9, 1: 9, 2: 9}},
        )
        folded, _ = fitted(world, tmp_path / "cv", cv_folds=4)
        plain, _ = fitted(world, tmp_path / "plain")

        frame = world.frames[("test", "z")]
        pd.testing.assert_frame_equal(folded.apply(frame), plain.apply(frame))

    @pytest.mark.parametrize(
        "renamed", ["group", "sequence"], ids=["no group", "no sequence"]
    )
    def test_a_table_without_the_identity_columns_is_refused_and_not_scored(
        self, tmp_path: Path, renamed: str
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(6, {0: 10, 1: 10}))
        feature, _ = fitted(world, tmp_path / "run", cv_folds=3)
        frame = world.frames[world.train[0]].rename(columns={renamed: renamed.upper()})

        with pytest.raises(ValueError, match=rf"^\[xgboost\] .*\['{renamed}'\]"):
            feature.apply(frame)

    def test_the_fold_models_do_not_memorise_a_sequence(self, tmp_path: Path) -> None:
        """Features are a constant per sequence and the label a random class per sequence.

        In sample, a model recalls every sequence. Held out, it can only guess.
        """
        rng = np.random.default_rng(3)
        entries = {("g", f"s{i:02d}"): {int(rng.integers(0, 3)): 6} for i in range(60)}
        world = make_world(
            tmp_path / "templates", entries, constant_per_entry=True, n_features=5
        )
        feature, state = fitted(world, tmp_path / "run", cv_folds=5, n_estimators=20)
        in_sample, _ = fitted(world, tmp_path / "plain")

        truth = np.concatenate([world.labels[e] for e in entries])
        held_out = pd.concat(
            [feature.apply(world.frames[e]) for e in entries], ignore_index=True
        )
        memorised = in_sample.apply(world.rows(list(entries)))

        assert (memorised["predicted_label"].to_numpy() == truth).mean() > 0.95
        assert (held_out["predicted_label"].to_numpy() == truth).mean() < 0.55
        assert state["_cv"] is not None

    def test_a_fold_that_never_saw_a_class_gives_it_probability_zero(
        self, tmp_path: Path
    ) -> None:
        for strategy in ("multiclass", "one_vs_rest"):
            entries = {
                **sequences(5, {0: 10, 1: 10}),
                ("g", "rare"): {0: 5, 1: 5, 2: 20},
            }
            world = make_world(tmp_path / f"templates-{strategy}", entries, seed=2)
            feature, state = fitted(
                world, tmp_path / f"run-{strategy}", cv_folds=3, strategy=strategy
            )

            out = feature.apply(world.frames[("g", "rare")])
            other = feature.apply(world.frames[("g", "s00")])

            assert prob_columns(out) == ["prob_0", "prob_1", "prob_2"]
            assert (out["prob_2"] == 0.0).all()
            assert (out["predicted_label"] != 2).all()
            assert prob_columns(other) == ["prob_0", "prob_1", "prob_2"]
            assert state["_cv"]["classes"].count([0, 1]) == 1

    @pytest.mark.parametrize("strategy", ["multiclass", "one_vs_rest"])
    def test_a_fold_missing_a_middle_class_writes_the_others_to_their_own_columns(
        self, tmp_path: Path, strategy: str
    ) -> None:
        """Classes 0, 1 and 3 with class 1 in one sequence: that fold's model has 0 and 3.

        Written to the first two columns, its probabilities would stand for classes 0
        and 1. Only a class absent from the middle of the order shows it.
        """
        entries = {
            **sequences(5, {0: 10, 3: 10}),
            ("g", "rare"): {0: 5, 1: 20, 3: 5},
        }
        world = make_world(tmp_path / "templates", entries, seed=2)
        feature, state = fitted(world, tmp_path / "run", cv_folds=3, strategy=strategy)
        fold: int = state["_fold_of"][("g", "rare")]
        columns: list[str] = state["_feature_columns"]
        folded: list[XGBClassifier | list[XGBClassifier]] = state["_cv"]["models"]
        frame = world.frames[("g", "rare")]
        x = frame[columns].to_numpy(dtype=np.float64)
        model = folded[fold]
        expected: np.ndarray = (
            np.column_stack([m.predict_proba(x)[:, 1] for m in model])
            if isinstance(model, list)
            else model.predict_proba(x)
        )

        out = feature.apply(frame)

        assert state["_cv"]["classes"][fold] == [0, 3]
        assert prob_columns(out) == ["prob_0", "prob_1", "prob_3"]
        np.testing.assert_array_equal(out["prob_1"].to_numpy(), 0.0)
        np.testing.assert_allclose(out[["prob_0", "prob_3"]].to_numpy(), expected)
        assert set(out["predicted_label"]) <= {0, 3}


class TestRefusals:
    def test_more_folds_than_train_entries_is_refused(self, tmp_path: Path) -> None:
        world = make_world(tmp_path / "templates", sequences(4, {0: 8, 1: 8}))
        feature = make_feature(world, tmp_path / "run", cv_folds=5)

        with pytest.raises(ValueError, match=r"^\[xgboost\] .*4 train entries into 5"):
            feature.fit(NO_INPUT)

    def test_a_fold_whose_training_rows_hold_one_class_is_refused(
        self, tmp_path: Path
    ) -> None:
        world = make_world(
            tmp_path / "templates",
            {("g", "only-0"): {0: 20}, ("g", "only-1"): {1: 20}},
        )
        feature = make_feature(world, tmp_path / "run", cv_folds=2)

        with pytest.raises(
            ValueError, match=r"outside fold \d hold 1 of the 2 classes.*Lower cv_folds"
        ):
            feature.fit(NO_INPUT)
        assert feature._model is None  # pyright: ignore[reportPrivateUsage]

    def test_templates_without_the_identity_columns_are_refused(
        self, tmp_path: Path
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(4, {0: 8, 1: 8}))
        templates = pd.read_parquet(world.path).drop(columns=["group", "sequence"])
        templates.to_parquet(world.path, index=False)
        feature = make_feature(world, tmp_path / "run", cv_folds=2)

        with pytest.raises(ValueError, match=r"Rerun extract-labeled-templates 0.6"):
            feature.fit(NO_INPUT)

    def test_templates_without_a_provenance_are_refused(self, tmp_path: Path) -> None:
        world = make_world(tmp_path / "templates", sequences(4, {0: 8, 1: 8}))
        (world.path.parent / "template_provenance.parquet").unlink()
        feature = make_feature(world, tmp_path / "run", cv_folds=2)

        with pytest.raises(ValueError, match=r"template_provenance\.parquet"):
            feature.fit(NO_INPUT)

    def test_template_rows_of_an_entry_the_provenance_does_not_list_are_refused(
        self, tmp_path: Path
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(4, {0: 8, 1: 8}))
        provenance = world.path.parent / "template_provenance.parquet"
        table = pd.read_parquet(provenance)
        table.iloc[1:].to_parquet(provenance, index=False)
        feature = make_feature(world, tmp_path / "run", cv_folds=2)

        with pytest.raises(
            ValueError, match=r"\('g', 's00'\).*provenance does not list"
        ):
            feature.fit(NO_INPUT)

    def test_a_class_in_fewer_sequences_than_folds_is_reported(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        world = make_world(
            tmp_path / "templates",
            {**sequences(5, {0: 10, 1: 10}), ("g", "rare"): {0: 3, 1: 3, 2: 20}},
        )
        caplog.set_level(logging.WARNING)

        _ = fitted(world, tmp_path / "run", cv_folds=3)

        assert "class 2 has template rows in 1 sequence" in caplog.text
        assert "fewer than cv_folds=3" in caplog.text

    def test_a_fold_holding_most_of_a_class_is_reported(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        world = make_world(
            tmp_path / "templates",
            {
                **sequences(4, {0: 10, 1: 10}),
                ("g", "big"): {0: 10, 1: 10, 2: 90},
                ("g", "small-a"): {2: 5},
                ("g", "small-b"): {2: 5},
            },
        )
        caplog.set_level(logging.WARNING)

        _ = fitted(world, tmp_path / "run", cv_folds=3)

        assert "under half of" in caplog.text
        assert "class 2" in caplog.text

    def test_smote_in_a_fold_that_leaves_a_class_too_few_rows_is_refused_before_any_fit(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        """The full fit has six rows of class 2. The fold without them has two."""
        entries = {
            **sequences(5, {0: 30, 1: 30}),
            ("g", "big"): {0: 10, 1: 10, 2: 6},
            ("g", "small"): {0: 10, 1: 10, 2: 2},
        }
        world = make_world(tmp_path / "templates", entries, seed=2)
        feature = make_feature(world, tmp_path / "run", cv_folds=7, use_smote=True)

        with pytest.raises(
            ValueError, match=r"^\[xgboost\] .*outside fold \d hold class 2 \(2 rows\)"
        ):
            feature.fit(NO_INPUT)
        assert fits == []

        without = make_feature(world, tmp_path / "plain", cv_folds=7)
        without.fit(NO_INPUT)

    def test_smote_leaves_a_largest_class_alone_so_a_tie_of_small_ones_is_fitted(
        self, tmp_path: Path
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(3, {0: 2, 1: 2}))

        _ = fitted(world, tmp_path / "run", cv_folds=3, use_smote=True)

    def test_a_class_undersampled_to_a_tie_does_not_need_smote_neighbours(
        self, tmp_path: Path
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(4, {0: 30, 1: 1}))

        _ = fitted(
            world, tmp_path / "run", cv_folds=4, use_smote=True, undersample_ratio=1.0
        )

    def test_a_fit_that_will_not_fit_in_memory_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(6, {0: 10, 1: 10}))
        feature = make_feature(world, tmp_path / "run", cv_folds=3)
        monkeypatch.setattr(helpers, "host_memory_bytes", lambda: (2000, 1000))

        with pytest.raises(MemoryError, match=r"^\[xgboost\] .*n_per_class"):
            feature.fit(NO_INPUT)
        assert feature._model is None  # pyright: ignore[reportPrivateUsage]


class TestAgainstAnOracle:
    def _run(
        self, tmp_path: Path, strategy: str
    ) -> tuple[World, XgboostFeature, dict[Entry, int]]:
        world = make_world(
            tmp_path / "templates",
            {
                **sequences(7, {0: 15, 1: 15}),
                **sequences(5, {0: 5, 1: 15, 2: 15}, prefix="r"),
            },
            seed=4,
        )
        feature, _ = fitted(
            world,
            tmp_path / "run",
            cv_folds=4,
            cv_seed=2,
            class_weight=None,
            strategy=strategy,
        )
        feature.save_state(tmp_path / "run")
        folds = sidecar(tmp_path / "run")
        return (
            world,
            feature,
            dict(zip(zip(folds["group"], folds["sequence"]), folds["fold"])),
        )

    def test_pooled_predictions_match_leave_one_group_out_on_the_runs_folds(
        self, tmp_path: Path
    ) -> None:
        world, feature, fold_of = self._run(tmp_path, "multiclass")
        columns = [
            c
            for c in world.frames[world.train[0]].columns
            if c == "code" or c.startswith("feat_")
        ]
        frames = world.rows(world.train)
        x = frames[sorted(columns)].to_numpy(dtype=np.float64)
        y = np.concatenate([world.labels[e] for e in world.train])
        groups = np.array(
            [fold_of[(g, s)] for g, s in zip(frames["group"], frames["sequence"])]
        )

        oracle = np.zeros((len(y), 3))
        splits = LeaveOneGroupOut().split(x, y, groups)  # pyright: ignore[reportUnknownMemberType]
        for train, held in splits:
            classes = sorted(set(y[train].tolist()))
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=len(classes),
                n_estimators=8,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
                n_jobs=1,
            )
            model.fit(x[train], np.searchsorted(classes, y[train]))
            oracle[np.ix_(held, classes)] = model.predict_proba(x[held])

        run = pd.concat(
            [feature.apply(world.frames[e]) for e in world.train], ignore_index=True
        )

        np.testing.assert_allclose(run[prob_columns(run)].to_numpy(), oracle)
        assert f1_score(y, run["predicted_label"], average="macro") == pytest.approx(
            f1_score(y, oracle.argmax(axis=1), average="macro")
        )

    def test_the_one_vs_rest_strategy_agrees_with_its_own_folds(
        self, tmp_path: Path
    ) -> None:
        world, feature, fold_of = self._run(tmp_path, "one_vs_rest")

        out = pd.concat(
            [feature.apply(world.frames[e]) for e in world.train], ignore_index=True
        )

        assert prob_columns(out) == ["prob_0", "prob_1", "prob_2"]
        assert set(fold_of.values()) == {0, 1, 2, 3}


class TestBundle:
    def _fitted(self, tmp_path: Path, **params: object) -> tuple[World, XgboostFeature]:
        world = make_world(
            tmp_path / "templates",
            sequences(6, {0: 10, 1: 10}),
            outside={("test", "z"): {0: 9, 1: 9}},
        )
        feature, _ = fitted(world, tmp_path / "run", **params)
        return world, feature

    def test_a_cross_validated_run_reloads_from_its_bundle(
        self, tmp_path: Path
    ) -> None:
        world, feature = self._fitted(tmp_path, cv_folds=3)
        feature.save_state(tmp_path / "run")

        again = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={**PARAMS, "cv_folds": 3},
        )

        assert again.load_state(tmp_path / "run", {}, {}) is True
        for entry in (world.train[0], ("test", "z")):
            pd.testing.assert_frame_equal(
                again.apply(world.frames[entry]), feature.apply(world.frames[entry])
            )

    def test_a_process_worker_rebuilds_the_feature_and_scores_each_entry_alike(
        self, tmp_path: Path
    ) -> None:
        """A process worker builds the feature from the dumps of its inputs and params.

        The dump of a cross-validated run names `model` as `None` beside `templates`
        and `cv_folds`, and the worker must still take the folds from the bundle.
        """
        world, feature = self._fitted(tmp_path, cv_folds=3, cv_seed=2)
        feature.save_state(tmp_path / "run")

        for entry in (world.train[0], world.train[3], ("test", "z")):
            frame = world.frames[entry]
            worker = process_apply_worker(
                feature.__module__,
                type(feature).__name__,
                feature.inputs.model_dump(),
                feature.params.model_dump(),
                str(tmp_path / "run"),
                {},
                {},
                frame,
            )
            pd.testing.assert_frame_equal(worker, feature.apply(frame))

    def test_the_bundle_holds_the_folds_beside_the_full_model(
        self, tmp_path: Path
    ) -> None:
        _, feature = self._fitted(tmp_path, cv_folds=3, cv_seed=4)
        feature.save_state(tmp_path / "run")

        bundle = joblib.load(tmp_path / "run" / "xgboost_model.joblib")

        assert bundle["cv"]["folds"] == 3
        assert bundle["cv"]["seed"] == 4
        assert len(bundle["cv"]["models"]) == 3
        assert bundle["cv"]["classes"] == [[0, 1]] * 3
        assert sorted(row[2] for row in bundle["cv"]["fold_of"]) == [0, 0, 1, 1, 2, 2]
        assert isinstance(bundle["model"], XGBClassifier)

    def test_a_run_without_folds_writes_no_cv_entry_and_no_sidecar(
        self, tmp_path: Path
    ) -> None:
        _, feature = self._fitted(tmp_path)
        feature.save_state(tmp_path / "run")

        bundle = joblib.load(tmp_path / "run" / "xgboost_model.joblib")

        assert "cv" not in bundle
        assert not (tmp_path / "run" / "cv_folds.parquet").exists()

    def test_a_bundle_without_a_cv_entry_loads_through_model(
        self, tmp_path: Path
    ) -> None:
        world, feature = self._fitted(tmp_path)
        feature.save_state(tmp_path / "run")
        applied = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={
                "model": {"feature": "xgboost", "run_id": "0.2-old"},
                "default_class": 0,
            },
        )

        assert applied.load_state(
            tmp_path / "other", {"model": tmp_path / "run" / "xgboost_model.joblib"}, {}
        )
        frame = world.frames[world.train[0]]
        pd.testing.assert_frame_equal(applied.apply(frame), feature.apply(frame))

    def test_a_cross_validated_bundle_loaded_through_model_scores_every_entry_in_full(
        self, tmp_path: Path
    ) -> None:
        world, feature = self._fitted(tmp_path, cv_folds=3)
        feature.save_state(tmp_path / "run")
        applied = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={
                "model": {"feature": "xgboost", "run_id": "0.3-cv"},
                "default_class": 0,
            },
        )
        assert applied.load_state(
            tmp_path / "other", {"model": tmp_path / "run" / "xgboost_model.joblib"}, {}
        )

        frame = world.frames[("test", "z")]
        pd.testing.assert_frame_equal(applied.apply(frame), feature.apply(frame))
        train_entry = world.frames[world.train[0]]
        assert not np.allclose(
            applied.apply(train_entry)["prob_1"], feature.apply(train_entry)["prob_1"]
        )

    def test_a_cached_model_without_folds_is_not_taken_for_a_cross_validated_run(
        self, tmp_path: Path
    ) -> None:
        _, feature = self._fitted(tmp_path)
        feature.save_state(tmp_path / "run")
        folded = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={**PARAMS, "cv_folds": 3},
        )

        with pytest.raises(RuntimeError, match=r"cv_folds is set.*no folds"):
            folded.load_state(tmp_path / "run", {}, {})

    def test_a_kill_during_the_dump_leaves_no_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, feature = self._fitted(tmp_path, cv_folds=3)
        run_root = tmp_path / "run"

        def killed(obj: object, path: Any, *args: Any, **kwargs: Any) -> None:
            Path(path).write_bytes(b"partial")
            raise KeyboardInterrupt

        monkeypatch.setattr(joblib, "dump", killed)
        with pytest.raises(KeyboardInterrupt):
            feature.save_state(run_root)

        assert not (run_root / "xgboost_model.joblib").exists()
        assert [p for p in run_root.iterdir() if p.suffix == ".tmp"] == []
        reader = make_feature(
            make_world(tmp_path / "again", sequences(6, {0: 10, 1: 10})),
            run_root,
            cv_folds=3,
        )
        assert reader._model is None  # pyright: ignore[reportPrivateUsage]

    def test_a_failed_dump_leaves_a_finished_bundle_as_it_was(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, feature = self._fitted(tmp_path)
        feature.save_state(tmp_path / "run")
        before = (tmp_path / "run" / "xgboost_model.joblib").read_bytes()

        def failing(obj: object, path: Any, *args: Any, **kwargs: Any) -> None:
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")

        monkeypatch.setattr(joblib, "dump", failing)
        with pytest.raises(OSError, match="disk full"):
            feature.save_state(tmp_path / "run")

        assert (tmp_path / "run" / "xgboost_model.joblib").read_bytes() == before

    def test_two_invocations_of_one_run_never_leave_a_reader_a_partial_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, first = self._fitted(tmp_path, cv_folds=3)
        _, second = self._fitted(tmp_path / "second", cv_folds=3)
        run_root = tmp_path / "run"
        real_dump = joblib.dump

        def slow(obj: object, path: Any, *args: Any, **kwargs: Any) -> None:
            buffer = io.BytesIO()
            real_dump(obj, buffer)
            raw = buffer.getvalue()
            with open(path, "wb") as handle:
                handle.write(raw[: len(raw) // 2])
                handle.flush()
                time.sleep(0.15)
                handle.write(raw[len(raw) // 2 :])

        monkeypatch.setattr(joblib, "dump", slow)
        bundle = run_root / "xgboost_model.joblib"
        writers = [
            threading.Thread(target=first.save_state, args=(run_root,)),
            threading.Thread(target=second.save_state, args=(run_root,)),
        ]
        seen: list[str] = []
        for writer in writers:
            writer.start()
        while any(writer.is_alive() for writer in writers):
            if bundle.exists():
                try:
                    joblib.load(bundle)
                    seen.append("whole")
                except Exception as exc:  # noqa: BLE001 - any failure is a partial read
                    seen.append(f"partial: {exc}")
            time.sleep(0.01)
        for writer in writers:
            writer.join()

        assert [s for s in seen if s.startswith("partial")] == []
        assert joblib.load(bundle)["cv"]["folds"] == 3


class TestRunningState:
    def test_the_engine_can_wire_progress_and_cancel(self) -> None:
        feature = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)), params=PARAMS
        )

        assert hasattr(feature, "progress_callback")
        assert hasattr(feature, "cancel_token")

    def test_progress_counts_each_fit_of_the_folds_and_the_full_model(
        self, tmp_path: Path
    ) -> None:
        class Recorder(NullProgressCallback):
            def __init__(self) -> None:
                self.epochs: list[tuple[int, int]] = []

            def on_epoch_end(
                self, epoch: int, total_epochs: int, metrics: dict[str, float]
            ) -> None:
                self.epochs.append((epoch, total_epochs))

        world = make_world(tmp_path / "templates", sequences(6, {0: 10, 1: 10}))
        feature = make_feature(world, tmp_path / "run", cv_folds=3)
        recorder = Recorder()
        feature.progress_callback = recorder

        feature.fit(NO_INPUT)

        assert recorder.epochs == [(0, 4), (1, 4), (2, 4), (3, 4)]

    @pytest.mark.parametrize(
        "cancelled_during",
        [1, 2, 3],
        ids=["first fold", "second fold", "last fold, before the full model"],
    )
    def test_a_cancel_stops_before_the_next_fit_and_writes_no_bundle(
        self,
        tmp_path: Path,
        fits: list[FitCall],
        monkeypatch: pytest.MonkeyPatch,
        cancelled_during: int,
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(6, {0: 10, 1: 10}))
        feature = make_feature(world, tmp_path / "run", cv_folds=3)
        token = CancelToken()
        feature.cancel_token = token
        original = XGBClassifier.fit

        def cancelling(
            self: XGBClassifier, X: Any, y: Any, **kwargs: Any
        ) -> XGBClassifier:
            fitted = original(self, X, y, **kwargs)
            if len(fits) == cancelled_during:
                token.cancel()
            return fitted

        monkeypatch.setattr(XGBClassifier, "fit", cancelling)
        # `fits` wraps the class `fit` that the monkeypatch replaced, so count through it.
        with pytest.raises(Cancelled):
            feature.fit(NO_INPUT)
        feature.save_state(tmp_path / "run")

        assert len(fits) == cancelled_during
        assert not (tmp_path / "run" / "xgboost_model.joblib").exists()
        assert not (tmp_path / "run" / "cv_folds.parquet").exists()


class TestThroughputAndWeights:
    def test_n_jobs_reaches_every_model(self, tmp_path: Path) -> None:
        world = make_world(tmp_path / "templates", sequences(6, {0: 10, 1: 10}))
        for strategy, count in (("multiclass", 4), ("one_vs_rest", 8)):
            _, state = fitted(
                world, tmp_path / strategy, cv_folds=3, n_jobs=2, strategy=strategy
            )
            fold_models: list[XGBClassifier | list[XGBClassifier]] = state["_cv"][
                "models"
            ]
            full: XGBClassifier | list[XGBClassifier] = state["_model"]
            flat: list[XGBClassifier] = []
            for model in [*fold_models, full]:
                flat.extend(model if isinstance(model, list) else [model])
            assert len(flat) == count
            assert {m.get_params()["n_jobs"] for m in flat} == {2}

    def test_device_reaches_every_fit_and_a_fitted_model_predicts_on_the_cpu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The CI runners have no CUDA device, so the fit itself runs on the CPU."""
        at_fit: list[str] = []
        original = XGBClassifier.fit

        def on_cpu(self: XGBClassifier, X: Any, y: Any, **kwargs: Any) -> XGBClassifier:
            at_fit.append(self.get_params()["device"])
            self.set_params(device="cpu")
            return original(self, X, y, **kwargs)

        monkeypatch.setattr(XGBClassifier, "fit", on_cpu)
        world = make_world(tmp_path / "templates", sequences(6, {0: 10, 1: 10}))
        for strategy, count in (("multiclass", 4), ("one_vs_rest", 8)):
            at_fit.clear()
            _, state = fitted(
                world,
                tmp_path / strategy,
                cv_folds=3,
                device="cuda",
                strategy=strategy,
            )
            fold_models: list[XGBClassifier | list[XGBClassifier]] = state["_cv"][
                "models"
            ]
            full: XGBClassifier | list[XGBClassifier] = state["_model"]
            flat: list[XGBClassifier] = []
            for model in [*fold_models, full]:
                flat.extend(model if isinstance(model, list) else [model])
            assert at_fit == ["cuda"] * count
            assert {m.get_params()["device"] for m in flat} == {"cpu"}

    def test_the_default_device_fits_on_the_cpu(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(6, {0: 10, 1: 10}))
        fitted(world, tmp_path / "run")

        assert {call.model.get_params()["device"] for call in fits} == {"cpu"}

    def test_balanced_weights_each_row_by_the_inverse_of_its_class_frequency(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = make_world(
            tmp_path / "templates",
            {("g", "a"): {0: 30, 1: 10}, ("g", "b"): {0: 30, 1: 10}},
        )

        _ = fitted(world, tmp_path / "run")

        weights = fits[0].sample_weight
        assert weights is not None
        np.testing.assert_allclose(weights[fits[0].y == 0], 80 / (2 * 60))
        np.testing.assert_allclose(weights[fits[0].y == 1], 80 / (2 * 20))

    def test_no_class_weight_passes_no_weights(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(4, {0: 10, 1: 10}))

        _ = fitted(world, tmp_path / "run", class_weight=None)

        assert fits[0].sample_weight is None

    def test_a_mapping_weighs_each_row_by_its_class_and_an_unlisted_class_by_one(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(4, {0: 10, 1: 10, 2: 10}))

        _ = fitted(world, tmp_path / "run", class_weight={1: 4.0, 2: 0.5})

        weights = fits[0].sample_weight
        assert weights is not None
        np.testing.assert_allclose(weights[fits[0].y == 0], 1.0)
        np.testing.assert_allclose(weights[fits[0].y == 1], 4.0)
        np.testing.assert_allclose(weights[fits[0].y == 2], 0.5)

    def test_a_mapping_weighs_every_model_of_one_vs_rest_alike(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(4, {0: 10, 1: 10}))

        _ = fitted(
            world, tmp_path / "run", class_weight={1: 3.0}, strategy="one_vs_rest"
        )

        assert len(fits) == 2
        for call in fits:
            assert call.sample_weight is not None
            assert sorted(set(call.sample_weight.tolist())) == [1.0, 3.0]


class TestBlocks:
    """`cv_unit="block"` holds out blocks of frames, so one training sequence is still cross-validated."""

    ONLY: Entry = ("g", "only")

    def _world(self, tmp_path: Path, n_entries: int = 1, **kwargs: Any) -> World:
        per_class = {0: 30, 1: 30}
        train = (
            {self.ONLY: per_class}
            if n_entries == 1
            else sequences(n_entries, per_class)
        )
        return make_world(
            tmp_path / "templates",
            train,
            with_frames=True,
            interleave=True,
            **kwargs,
        )

    @staticmethod
    def _bounds(folds: pd.DataFrame, fold: int) -> tuple[int, int]:
        """The frames ``start`` up to ``stop`` of the one block that *fold* holds out."""
        (block,) = folds.loc[[fold]].to_dict("records")
        return int(block["start"]), int(block["stop"])

    @staticmethod
    def _frames(call: FitCall) -> set[int]:
        """The frames of the rows a fit saw: a row's `code` is its entry's thousands plus its frame."""
        return {int(code % 1000) for code in call.x[:, 0]}

    def test_one_training_sequence_is_cut_into_blocks_and_each_fold_holds_one_out(
        self, tmp_path: Path
    ) -> None:
        world = self._world(tmp_path)
        feature, _ = fitted(world, tmp_path / "run", cv_folds=3, cv_unit="block")
        feature.save_state(tmp_path / "run")

        folds = sidecar(tmp_path / "run")

        assert list(folds.columns) == [
            "group",
            "sequence",
            "block",
            "start",
            "stop",
            "fold",
            "n_rows",
            "rows_0",
            "rows_1",
        ]
        assert folds[["block", "start", "stop"]].values.tolist() == [
            [0, 0, 20],
            [1, 20, 40],
            [2, 40, 60],
        ]
        assert sorted(folds["fold"]) == [0, 1, 2]
        assert (folds["n_rows"] == 20).all()
        assert (folds["rows_0"] + folds["rows_1"] == folds["n_rows"]).all()

    def test_a_fold_trains_on_no_row_of_its_blocks_or_of_the_gap_around_them(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = self._world(tmp_path)
        feature, _ = fitted(
            world, tmp_path / "run", cv_folds=3, cv_unit="block", cv_gap_frames=2
        )
        feature.save_state(tmp_path / "run")
        folds = sidecar(tmp_path / "run").set_index("fold")

        assert len(fits) == 3 + 1
        for fold in range(3):
            start, stop = self._bounds(folds, fold)
            assert self._frames(fits[fold]) == {
                f for f in range(60) if not start - 2 <= f < stop + 2
            }
        assert self._frames(fits[3]) == set(range(60))

    def test_a_row_is_scored_by_the_model_of_the_block_that_holds_its_frame(
        self, tmp_path: Path
    ) -> None:
        world = self._world(tmp_path, outside={("test", "z"): {0: 9, 1: 9}})
        feature, state = fitted(
            world, tmp_path / "run", cv_folds=3, cv_unit="block", cv_gap_frames=2
        )
        feature.save_state(tmp_path / "run")
        folds = sidecar(tmp_path / "run").set_index("fold")
        columns: list[str] = state["_feature_columns"]
        models: list[XGBClassifier] = state["_cv"]["models"]
        frame = world.frames[self.ONLY]

        out = feature.apply(frame)

        expected = np.empty(len(frame))
        for fold in range(3):
            start, stop = self._bounds(folds, fold)
            rows = ((frame["frame"] >= start) & (frame["frame"] < stop)).to_numpy()
            expected[rows] = models[fold].predict_proba(
                frame.loc[rows, columns].to_numpy(dtype=np.float64)
            )[:, 1]
        np.testing.assert_allclose(out["prob_1"].to_numpy(), expected)
        assert len({round(float(p), 9) for p in expected}) > 3

    def test_a_frame_in_no_block_and_an_entry_outside_the_templates_are_scored_by_the_full_model(
        self, tmp_path: Path
    ) -> None:
        world = self._world(tmp_path, outside={("test", "z"): {0: 9, 1: 9}})
        feature, state = fitted(world, tmp_path / "run", cv_folds=3, cv_unit="block")
        columns: list[str] = state["_feature_columns"]
        full: XGBClassifier = state["_model"]
        beyond = world.frames[self.ONLY].iloc[:10].copy()
        beyond["frame"] += 500

        for frame in (beyond, world.frames[("test", "z")]):
            expected = full.predict_proba(frame[columns].to_numpy(dtype=np.float64))
            np.testing.assert_allclose(
                feature.apply(frame)["prob_1"].to_numpy(), expected[:, 1]
            )

    def test_every_fold_trains_on_every_sequence_but_its_blocks(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = self._world(tmp_path, n_entries=3)
        _ = fitted(world, tmp_path / "run", cv_folds=2, cv_unit="block")

        assert len(fits) == 2 + 1
        assert fits[0].entries == fits[1].entries == fits[2].entries == {0, 1, 2}
        assert len(fits[0].x) + len(fits[1].x) == len(fits[2].x)

    def test_more_blocks_than_folds_hold_several_blocks_out_of_each_fold(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        world = self._world(tmp_path)
        feature, _ = fitted(
            world,
            tmp_path / "run",
            cv_folds=2,
            cv_unit="block",
            cv_blocks_per_sequence=4,
        )
        feature.save_state(tmp_path / "run")

        folds = sidecar(tmp_path / "run")

        assert folds["start"].tolist() == [0, 15, 30, 45]
        assert sorted(folds["fold"]) == [0, 0, 1, 1]
        for fold in (0, 1):
            held = {
                f
                for start, stop in folds.loc[folds["fold"] == fold, ["start", "stop"]]
                .to_numpy()
                .tolist()
                for f in range(start, stop)
            }
            assert self._frames(fits[fold]) == set(range(60)) - held

    def test_the_blocks_are_dealt_as_sequence_classifier_deals_them(
        self, tmp_path: Path
    ) -> None:
        """One fold table serves both, so that a `probability-ensemble` can pool their runs."""
        world = self._world(tmp_path, n_entries=3)
        feature, _ = fitted(
            world,
            tmp_path / "run",
            cv_folds=3,
            cv_unit="block",
            cv_blocks_per_sequence=2,
            cv_seed=5,
        )
        feature.save_state(tmp_path / "run")
        blocks = deal_blocks(
            "sequence-classifier", {entry: (0, 60) for entry in world.train}, 2, 3, 5
        )
        # sequence-classifier counts labeled frames, and this step the rows it drew
        other = block_fold_table(
            blocks,
            pd.DataFrame(
                0,
                index=pd.MultiIndex.from_tuples(
                    [(b.group, b.sequence, b.block) for b in blocks]
                ),
                columns=[0, 1],
            ),
            [0, 1],
        )
        other.to_parquet(tmp_path / "other.parquet", index=False)

        pd.testing.assert_frame_equal(
            read_fold_table(tmp_path / "run" / "cv_folds.parquet"),
            read_fold_table(tmp_path / "other.parquet"),
        )

    def test_the_seed_changes_which_blocks_share_a_fold(self, tmp_path: Path) -> None:
        world = self._world(tmp_path)

        def dealt(seed: int) -> list[int]:
            feature, _ = fitted(
                world,
                tmp_path / f"run{seed}",
                cv_folds=3,
                cv_unit="block",
                cv_blocks_per_sequence=6,
                cv_seed=seed,
            )
            feature.save_state(tmp_path / f"run{seed}")
            return sidecar(tmp_path / f"run{seed}")["fold"].tolist()

        assert dealt(0) != dealt(1)

    def test_fewer_blocks_than_folds_is_refused(self, tmp_path: Path) -> None:
        world = self._world(tmp_path)
        feature = make_feature(
            world,
            tmp_path / "run",
            cv_folds=3,
            cv_unit="block",
            cv_blocks_per_sequence=2,
        )

        with pytest.raises(ValueError, match=r"^\[xgboost\] .*2 blocks into 3"):
            feature.fit(NO_INPUT)

    def test_a_gap_that_leaves_a_fold_one_class_is_refused_before_any_fit(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        """Stretches of one class: the gap eats the only block of the other."""
        world = make_world(
            tmp_path / "templates",
            {self.ONLY: {0: 20, 1: 20}},
            with_frames=True,
        )
        feature = make_feature(
            world,
            tmp_path / "run",
            cv_folds=4,
            cv_unit="block",
            cv_gap_frames=12,
        )

        with pytest.raises(ValueError, match=r"leaves training rows of \d class"):
            feature.fit(NO_INPUT)
        assert fits == []

    def test_blocks_that_leave_a_fold_one_class_are_refused(
        self, tmp_path: Path, fits: list[FitCall]
    ) -> None:
        """One stretch of each class, cut in two: the training half of a fold has one class."""
        world = make_world(
            tmp_path / "templates", {self.ONLY: {0: 20, 1: 20}}, with_frames=True
        )
        feature = make_feature(world, tmp_path / "run", cv_folds=2, cv_unit="block")

        with pytest.raises(
            ValueError,
            match=r"outside fold \d hold 1 of the 2 classes.*label each class",
        ):
            feature.fit(NO_INPUT)
        assert fits == []

    def test_templates_without_a_frame_column_are_refused_and_name_the_remedy(
        self, tmp_path: Path
    ) -> None:
        world = make_world(tmp_path / "templates", sequences(4, {0: 8, 1: 8}))
        feature = make_feature(world, tmp_path / "run", cv_folds=2, cv_unit="block")

        with pytest.raises(
            ValueError,
            match=r"cv_unit='block'.*record no frame.*extract-labeled-templates again",
        ):
            feature.fit(NO_INPUT)

    def test_a_provenance_without_labeled_frames_is_refused(
        self, tmp_path: Path
    ) -> None:
        world = self._world(tmp_path)
        provenance = world.path.parent / "template_provenance.parquet"
        pd.read_parquet(provenance).drop(
            columns=["labeled_start", "labeled_stop"]
        ).to_parquet(provenance, index=False)
        feature = make_feature(world, tmp_path / "run", cv_folds=2, cv_unit="block")

        with pytest.raises(
            ValueError, match=r"records no labeled frames.*extract-labeled-templates"
        ):
            feature.fit(NO_INPUT)

    def test_template_rows_outside_the_labeled_frames_are_refused(
        self, tmp_path: Path
    ) -> None:
        world = self._world(tmp_path)
        provenance = world.path.parent / "template_provenance.parquet"
        table = pd.read_parquet(provenance)
        table["labeled_stop"] = 40
        table.to_parquet(provenance, index=False)
        feature = make_feature(world, tmp_path / "run", cv_folds=2, cv_unit="block")

        with pytest.raises(ValueError, match="outside the labeled frames"):
            feature.fit(NO_INPUT)

    def test_old_templates_stay_readable_by_sequence(self, tmp_path: Path) -> None:
        """Templates with no frame column, and a provenance without ranges, still cross-validate by sequence."""
        world = make_world(tmp_path / "templates", sequences(6, {0: 10, 1: 10}))

        feature, _ = fitted(world, tmp_path / "run", cv_folds=3)
        feature.save_state(tmp_path / "run")

        assert "start" not in sidecar(tmp_path / "run").columns

    def test_templates_that_record_frames_cross_validate_by_sequence_as_before(
        self, tmp_path: Path
    ) -> None:
        plain = make_world(tmp_path / "plain", sequences(6, {0: 10, 1: 10}))
        framed = make_world(
            tmp_path / "framed", sequences(6, {0: 10, 1: 10}), with_frames=True
        )
        scored: list[pd.DataFrame] = []
        for name, world in (("plain", plain), ("framed", framed)):
            feature, _ = fitted(world, tmp_path / f"{name}-run", cv_folds=3, cv_seed=2)
            feature.save_state(tmp_path / f"{name}-run")
            scored.append(
                pd.concat([feature.apply(world.frames[entry]) for entry in world.train])
            )

        pd.testing.assert_frame_equal(
            sidecar(tmp_path / "plain-run"), sidecar(tmp_path / "framed-run")
        )
        pd.testing.assert_frame_equal(scored[0], scored[1])

    def test_the_block_settings_are_refused_where_they_do_not_apply(self) -> None:
        with pytest.raises(ValueError, match="cv_unit.*cv_folds"):
            XgboostFeature.Params.from_overrides({**PARAMS, "cv_unit": "block"})
        with pytest.raises(ValueError, match="only apply with cv_unit block"):
            XgboostFeature.Params.from_overrides(
                {**PARAMS, "cv_folds": 3, "cv_gap_frames": 5}
            )
        with pytest.raises(ValueError, match="cv_blocks_per_sequence"):
            XgboostFeature.Params.from_overrides(
                {**PARAMS, "cv_blocks_per_sequence": 1}
            )

    def test_the_block_settings_enter_the_run_id_only_under_the_block_unit(
        self,
    ) -> None:
        def digest(**extra: object) -> str:
            return run_id_digest(
                XgboostFeature.Params.from_overrides({**PARAMS, **extra})
            )

        assert digest(cv_folds=3, cv_unit="sequence") == digest(cv_folds=3)
        assert digest(cv_folds=3, cv_unit="block") != digest(cv_folds=3)
        assert digest(cv_folds=3, cv_unit="block") == digest(
            cv_folds=3, cv_unit="block", cv_blocks_per_sequence=3
        )
        blocks = digest(cv_folds=3, cv_unit="block")
        assert digest(cv_folds=3, cv_unit="block", cv_blocks_per_sequence=6) != blocks
        assert digest(cv_folds=3, cv_unit="block", cv_gap_frames=4) != blocks

    def test_a_block_run_reloads_from_its_bundle_and_a_worker_scores_alike(
        self, tmp_path: Path
    ) -> None:
        world = self._world(tmp_path, outside={("test", "z"): {0: 9, 1: 9}})
        params = {"cv_folds": 3, "cv_unit": "block", "cv_gap_frames": 1}
        feature, _ = fitted(world, tmp_path / "run", **params)
        feature.save_state(tmp_path / "run")
        bundle = joblib.load(tmp_path / "run" / "xgboost_model.joblib")
        again = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={**PARAMS, **params},
        )

        assert bundle["cv"]["unit"] == "block"
        assert bundle["cv"]["gap_frames"] == 1
        assert bundle["cv"]["blocks_per_sequence"] == 3
        assert len(bundle["cv"]["blocks"]) == 3
        assert again.load_state(tmp_path / "run", {}, {}) is True
        for entry in (self.ONLY, ("test", "z")):
            frame = world.frames[entry]
            pd.testing.assert_frame_equal(again.apply(frame), feature.apply(frame))
            worker = process_apply_worker(
                feature.__module__,
                type(feature).__name__,
                feature.inputs.model_dump(),
                feature.params.model_dump(),
                str(tmp_path / "run"),
                {},
                {},
                frame,
            )
            pd.testing.assert_frame_equal(worker, feature.apply(frame))

    def test_a_cached_model_of_the_other_unit_is_not_taken_for_this_one(
        self, tmp_path: Path
    ) -> None:
        world = self._world(tmp_path)
        feature, _ = fitted(world, tmp_path / "run", cv_folds=3, cv_unit="block")
        feature.save_state(tmp_path / "run")
        by_sequence = XgboostFeature(
            XgboostFeature.Inputs((Result(feature="upstream"),)),
            params={**PARAMS, "cv_folds": 3},
        )

        with pytest.raises(RuntimeError, match="fitted with the other unit"):
            by_sequence.load_state(tmp_path / "run", {}, {})

    def test_the_one_vs_rest_strategy_scores_each_block_by_its_fold(
        self, tmp_path: Path
    ) -> None:
        world = self._world(tmp_path)
        feature, state = fitted(
            world,
            tmp_path / "run",
            cv_folds=3,
            cv_unit="block",
            strategy="one_vs_rest",
        )
        folds = state["_blocks"].folds(
            self.ONLY, world.frames[self.ONLY]["frame"].to_numpy()
        )
        columns: list[str] = state["_feature_columns"]
        frame = world.frames[self.ONLY]
        row = int(np.flatnonzero(folds == 1)[0])
        model: list[XGBClassifier] = state["_cv"]["models"][1]

        out = feature.apply(frame)

        x = frame.loc[[row], columns].to_numpy(dtype=np.float64)
        expected = np.column_stack([m.predict_proba(x)[:, 1] for m in model])
        np.testing.assert_allclose(
            out.loc[row, ["prob_0", "prob_1"]].to_numpy(dtype=float), expected[0]
        )
