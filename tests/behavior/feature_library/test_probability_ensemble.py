"""``probability-ensemble``: averaging and blending the probabilities of classifier runs.

The step reads classifier runs through ``inputs``, so these tests build the merged
table that the engine would hand it, from real parquet files and the engine's own
merge. The torch-free parts of a recipe that fits it end to end are in
``test_probability_ensemble_pipeline.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from mosaic.behavior.feature_library.probability_ensemble import (
    MAX_GRID_POINTS,
    ProbabilityEnsembleFeature,
    member_probability_column,
    merged_classes,
    simplex_grid,
)
from mosaic.core.evaluation import classification_scores
from mosaic.core.pipeline.loading import load_entry_data
from mosaic.core.pipeline.types import (
    DependencyLookup,
    InputStream,
    ParquetLoadSpec,
    Result,
)
from tests.helpers import Event, write_sparse_labels

GROUP = "g"
CLASSES = [0, 1, 2]
Entry = tuple[str, str]


def _inputs(members: int) -> ProbabilityEnsembleFeature.Inputs:
    return ProbabilityEnsembleFeature.Inputs(
        tuple(
            Result(feature="classifier", run_id=f"0.1-{position:010d}")
            for position in range(members)
        )
    )


def _feature(members: int = 2, **params: object) -> ProbabilityEnsembleFeature:
    return ProbabilityEnsembleFeature(
        _inputs(members), params={"default_class": 0, **params}
    )


def _member(
    probs: npt.NDArray[np.float64],
    *,
    sequence: str = "s0",
    classes: Sequence[int] = CLASSES,
) -> pd.DataFrame:
    """One classifier's output for one entry: identity, ``prob_<class>``, ``predicted_label``."""
    table = pd.DataFrame(
        {
            "frame": np.arange(len(probs)),
            "id": 0,
            "group": GROUP,
            "sequence": sequence,
        }
    )
    for column, label in enumerate(classes):
        table[f"prob_{label}"] = probs[:, column]
    table["predicted_label"] = np.asarray(classes)[np.argmax(probs, axis=1)]
    return table


def _merged(members: Sequence[pd.DataFrame], directory: Path) -> pd.DataFrame:
    """The tables of *members* merged as the engine merges the inputs of a step."""
    directory.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[Path, ParquetLoadSpec]] = []
    for position, table in enumerate(members):
        path = directory / f"{position}.parquet"
        table.to_parquet(path)
        specs.append((path, ParquetLoadSpec()))
    merged = load_entry_data(list(specs))
    assert merged is not None
    return merged


def _unread() -> Iterator[tuple[str, pd.DataFrame]]:
    raise AssertionError("the stream was read")


_UNREAD = InputStream(_unread, n_entries=0)


def _probs(*rows: list[float]) -> npt.NDArray[np.float64]:
    return np.array(rows, dtype=np.float64)


def _fitted(
    feature: ProbabilityEnsembleFeature, tmp_path: Path
) -> ProbabilityEnsembleFeature:
    """Fit *feature* where it reads no stream, so that apply has its weights."""
    assert not feature.load_state(tmp_path / "run", {}, {})
    feature.fit(InputStream(lambda: iter(()), n_entries=0))
    return feature


# --- Column names ---------------------------------------------------------------


def test_the_columns_of_a_merge_are_the_ones_the_helper_names(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    members = [_member(rng.dirichlet(np.ones(3), size=6)) for _ in range(3)]

    merged = _merged(members, tmp_path)

    assert merged_classes([str(c) for c in merged.columns]) == CLASSES
    for position in range(3):
        for label in CLASSES:
            assert member_probability_column(label, position) in merged.columns
    # A later member also contributes a label column, which the step does not read.
    assert "predicted_label__2" in merged.columns


def test_a_table_without_probabilities_is_refused() -> None:
    with pytest.raises(ValueError, match="no prob_<class> columns"):
        merged_classes(["frame", "X", "prob_a"])


# --- Apply ----------------------------------------------------------------------


def test_uniform_weights_average_the_members_and_keep_the_identity(
    tmp_path: Path,
) -> None:
    first = _probs([0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.5, 0.4, 0.1])
    second = _probs([0.2, 0.7, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8])
    merged = _merged([_member(first), _member(second)], tmp_path / "in")
    feature = _fitted(_feature(2), tmp_path)

    result = feature.apply(merged)

    assert list(result.columns) == [
        "frame",
        "group",
        "id",
        "sequence",
        "prob_0",
        "prob_1",
        "prob_2",
        "predicted_label",
    ]
    mean = (first + second) / 2
    np.testing.assert_allclose(result[["prob_0", "prob_1", "prob_2"]].to_numpy(), mean)
    assert result["predicted_label"].tolist() == [1, 2, 2]


def test_listed_weights_are_scaled_to_sum_to_one(tmp_path: Path) -> None:
    first = _probs([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    second = _probs([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    merged = _merged([_member(first), _member(second)], tmp_path / "in")
    feature = _fitted(_feature(2, weights=[3.0, 1.0]), tmp_path)

    result = feature.apply(merged)

    np.testing.assert_allclose(result["prob_0"].to_numpy(), [0.75, 0.0])
    np.testing.assert_allclose(result["prob_2"].to_numpy(), [0.25, 0.25])


def test_a_member_without_weight_decides_no_row_whatever_it_holds(
    tmp_path: Path,
) -> None:
    first = _probs([0.7, 0.2, 0.1], [0.1, 0.8, 0.1])
    second = _probs([np.nan, np.nan, np.nan], [0.3, 0.3, 0.4])
    merged = _merged([_member(first), _member(second)], tmp_path / "in")
    feature = _fitted(_feature(2, weights=[1.0, 0.0]), tmp_path)

    result = feature.apply(merged)

    assert result["predicted_label"].tolist() == [0, 1]
    np.testing.assert_allclose(result[["prob_0", "prob_1", "prob_2"]].to_numpy(), first)


def test_a_row_a_member_has_no_probabilities_for_takes_the_default_class(
    tmp_path: Path,
) -> None:
    first = _probs([0.7, 0.2, 0.1], [0.1, 0.8, 0.1])
    second = _probs([0.6, 0.3, 0.1], [np.nan, np.nan, np.nan])
    merged = _merged([_member(first), _member(second)], tmp_path / "in")
    feature = _fitted(_feature(2, default_class=2), tmp_path)

    result = feature.apply(merged)

    assert result["predicted_label"].tolist() == [0, 2]
    assert result["prob_0"].isna().tolist() == [False, True]


def test_a_decision_threshold_leaves_a_row_no_class_clears_at_the_default_class(
    tmp_path: Path,
) -> None:
    first = _probs([0.4, 0.3, 0.3], [0.1, 0.8, 0.1])
    merged = _merged([_member(first), _member(first)], tmp_path / "in")
    feature = _fitted(_feature(2, default_class=2, decision_threshold=0.5), tmp_path)

    assert feature.apply(merged)["predicted_label"].tolist() == [2, 1]


def test_members_that_write_other_classes_are_refused(tmp_path: Path) -> None:
    first = _probs([0.6, 0.3, 0.1], [0.2, 0.2, 0.6])
    second = _probs([0.7, 0.3], [0.4, 0.6])
    merged = _merged([_member(first), _member(second, classes=[0, 1])], tmp_path / "in")
    feature = _fitted(_feature(2), tmp_path)

    with pytest.raises(ValueError, match=r"member 1 \(prob_2__1\)"):
        feature.apply(merged)


# --- Parameters -----------------------------------------------------------------


def test_listed_weights_name_every_input() -> None:
    with pytest.raises(ValueError, match="lists 3 weights and the step has 2 inputs"):
        _feature(2, weights=[1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([], "lists no weight"),
        ([1.0, -0.5], "finite and not negative"),
        ([1.0, math.inf], "finite and not negative"),
        ([0.0, 0.0], "not all be 0"),
    ],
)
def test_listed_weights_are_finite_non_negative_and_not_all_zero(
    weights: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _feature(max(len(weights), 1), weights=weights)


def test_only_fit_takes_labels_score_classes_and_a_grid_step() -> None:
    with pytest.raises(ValueError, match="labels is required with weights fit"):
        _feature(2, weights="fit")
    labels = {"kind": "behavior"}
    for name, value in (
        ("labels", labels),
        ("score_classes", [0, 1]),
        ("grid_step", 0.1),
    ):
        with pytest.raises(ValueError, match=f"{name} only apply with weights fit"):
            _feature(2, **{name: value})


def test_no_mode_hashes_the_scope_and_only_a_fit_fits() -> None:
    fit = _feature(2, weights="fit", labels={"kind": "behavior"})

    assert ProbabilityEnsembleFeature.scope_dependent is False
    assert ProbabilityEnsembleFeature.fit_set_pinned_by == "inputs"
    assert _feature(2).scope_dependent is False
    assert _feature(2, weights=[1.0, 2.0]).scope_dependent is False
    assert fit.scope_dependent is False
    assert fit.fits is True
    assert _feature(2).fits is False
    assert _feature(2, weights=[1.0, 2.0]).fits is False


# --- The weight grid ------------------------------------------------------------


def test_the_grid_splits_the_units_in_lexicographic_order() -> None:
    grid = list(simplex_grid(3, 4))

    assert len(grid) == math.comb(4 + 2, 2)
    assert all(sum(point) == 4 for point in grid)
    assert grid == sorted(grid)
    assert grid[0] == (0, 0, 4) and grid[-1] == (4, 0, 0)
    assert list(simplex_grid(1, 5)) == [(5,)]


def test_a_grid_over_the_limit_is_refused_before_the_stream_is_read() -> None:
    feature = _feature(6, weights="fit", labels={"kind": "behavior"}, grid_step=0.01)

    with pytest.raises(ValueError, match=rf"over the limit of {MAX_GRID_POINTS:,}"):
        feature.fit(_UNREAD)


def test_a_grid_step_must_divide_one() -> None:
    feature = _feature(2, weights="fit", labels={"kind": "behavior"}, grid_step=0.3)

    with pytest.raises(ValueError, match="grid_step 0.3 does not divide 1"):
        feature.fit(_UNREAD)


# --- Fold tables ----------------------------------------------------------------

TRAIN: list[Entry] = [(GROUP, f"s{i}") for i in range(4)]


def _fold_table(
    folds: dict[Entry, int], *, scale: int = 1, extra: dict[str, int] | None = None
) -> pd.DataFrame:
    """A ``cv_folds.parquet``, with the counts of a classifier that counts *scale* per row."""
    return pd.DataFrame(
        [
            {
                "group": group,
                "sequence": sequence,
                "fold": fold,
                "n_rows": 10 * scale,
                "rows_0": 4 * scale,
                "rows_1": 6 * scale,
                **(extra or {}),
            }
            for (group, sequence), fold in sorted(folds.items())
        ]
    )


def _run(
    root: Path, name: str, table: pd.DataFrame | None, entries: Sequence[Entry] = TRAIN
) -> DependencyLookup:
    """A classifier's run root: one output per entry, and its fold table if it has one."""
    run = root / name
    run.mkdir(parents=True)
    lookup: DependencyLookup = {}
    for group, sequence in entries:
        output = run / f"{group}__{sequence}.parquet"
        output.write_bytes(b"")
        lookup[(group, sequence)] = output
    if table is not None:
        table.to_parquet(run / "cv_folds.parquet")
    return lookup


FOLDS: dict[Entry, int] = {entry: index % 2 for index, entry in enumerate(TRAIN)}


def _ensemble_run(
    root: Path, name: str, lookups: Sequence[DependencyLookup], **params: object
) -> tuple[Path, DependencyLookup]:
    """Run the ensemble over *lookups* and return its run root and output lookup."""
    feature = _feature(len(lookups), **params)
    run = root / name
    dependency_lookups = {f"_input_{i}": lookup for i, lookup in enumerate(lookups)}
    assert not feature.load_state(run, {}, dependency_lookups)
    feature.fit(InputStream(lambda: iter(()), n_entries=0))
    feature.save_state(run)
    output = run / f"{GROUP}__s0.parquet"
    output.write_bytes(b"")
    return run, {(GROUP, "s0"): output}


def test_members_without_a_fold_table_leave_the_blend_without_one(
    tmp_path: Path,
) -> None:
    lookups = [_run(tmp_path, "a", None), _run(tmp_path, "b", None)]

    run, _ = _ensemble_run(tmp_path, "blend", lookups)

    assert not (run / "cv_folds.parquet").exists()
    assert (run / "ensemble.json").exists()


def test_members_that_deal_one_set_of_folds_pass_the_fold_table_on(
    tmp_path: Path,
) -> None:
    # One member counts sampled rows and the other labeled frames, so the count
    # columns differ and only the folds are compared.
    lookups = [
        _run(tmp_path, "a", _fold_table(FOLDS)),
        _run(tmp_path, "b", _fold_table(FOLDS, scale=250, extra={"rows_2": 3})),
    ]

    run, _ = _ensemble_run(tmp_path, "blend", lookups)

    table = pd.read_parquet(run / "cv_folds.parquet")
    assert list(table.columns) == ["group", "sequence", "fold"]
    assert dict(zip(zip(table["group"], table["sequence"]), table["fold"])) == FOLDS


def test_fitted_weights_refuse_members_of_which_some_have_a_fold_table(
    tmp_path: Path,
) -> None:
    lookups = [_run(tmp_path, "a", _fold_table(FOLDS)), _run(tmp_path, "b", None)]
    feature = _feature(2, weights="fit", labels={"kind": "behavior"})
    dependency_lookups = {f"_input_{i}": lookup for i, lookup in enumerate(lookups)}

    with pytest.raises(
        ValueError, match="inputs 1 write no cv_folds.parquet and the others do"
    ):
        feature.load_state(tmp_path / "blend", {}, dependency_lookups)


def test_fitted_weights_refuse_members_that_deal_different_folds(
    tmp_path: Path,
) -> None:
    other = {entry: 1 - fold for entry, fold in FOLDS.items()} | {TRAIN[0]: 5}
    lookups = [
        _run(tmp_path, "a", _fold_table(FOLDS)),
        _run(tmp_path, "b", _fold_table(other)),
    ]
    feature = _feature(2, weights="fit", labels={"kind": "behavior"})
    dependency_lookups = {f"_input_{i}": lookup for i, lookup in enumerate(lookups)}

    with pytest.raises(ValueError, match="input 1 deals its cross-validation folds"):
        feature.load_state(tmp_path / "blend", {}, dependency_lookups)


BLOCKS: dict[tuple[Entry, int], tuple[int, int, int]] = {
    (TRAIN[0], 0): (0, 50, 1),
    (TRAIN[0], 1): (50, 100, 0),
    (TRAIN[1], 0): (10, 60, 0),
    (TRAIN[1], 1): (60, 110, 1),
}
"""The blocks of two entries as ``(start, stop, fold)``, which a block run's fold table lists."""


def _block_table(
    blocks: dict[tuple[Entry, int], tuple[int, int, int]],
    *,
    scale: int = 1,
) -> pd.DataFrame:
    """A ``cv_folds.parquet`` by block, with the counts of a classifier that counts *scale* per row."""
    return pd.DataFrame(
        [
            {
                "group": group,
                "sequence": sequence,
                "block": block,
                "start": start,
                "stop": stop,
                "fold": fold,
                "n_rows": 10 * scale,
                "rows_0": 4 * scale,
                "rows_1": 6 * scale,
            }
            for ((group, sequence), block), (start, stop, fold) in sorted(
                blocks.items()
            )
        ]
    )


def test_members_that_cut_and_deal_the_same_blocks_pass_the_fold_table_on(
    tmp_path: Path,
) -> None:
    """The count columns differ, as an ``xgboost`` run's sampled rows and a network's labeled frames do."""
    lookups = [
        _run(tmp_path, "a", _block_table(BLOCKS), entries=TRAIN[:2]),
        _run(tmp_path, "b", _block_table(BLOCKS, scale=250), entries=TRAIN[:2]),
    ]

    run, _ = _ensemble_run(tmp_path, "blend", lookups)

    table = pd.read_parquet(run / "cv_folds.parquet")
    assert list(table.columns) == ["group", "sequence", "fold", "start", "stop"]
    assert table[["start", "stop", "fold"]].values.tolist() == [
        [0, 50, 1],
        [50, 100, 0],
        [10, 60, 0],
        [60, 110, 1],
    ]


@pytest.mark.parametrize(
    "other",
    [
        {**BLOCKS, (TRAIN[0], 1): (50, 99, 0)},
        {**BLOCKS, (TRAIN[1], 0): (11, 60, 0)},
        {**BLOCKS, (TRAIN[0], 0): (0, 50, 0)},
    ],
    ids=["stop", "start", "fold"],
)
def test_fitted_weights_refuse_members_that_cut_or_deal_their_blocks_differently(
    tmp_path: Path, other: dict[tuple[Entry, int], tuple[int, int, int]]
) -> None:
    lookups = [
        _run(tmp_path, "a", _block_table(BLOCKS), entries=TRAIN[:2]),
        _run(tmp_path, "b", _block_table(other), entries=TRAIN[:2]),
    ]
    feature = _feature(2, weights="fit", labels={"kind": "behavior"})
    dependency_lookups = {f"_input_{i}": lookup for i, lookup in enumerate(lookups)}

    with pytest.raises(ValueError, match="input 1 deals its cross-validation folds"):
        feature.load_state(tmp_path / "blend", {}, dependency_lookups)


def test_a_member_by_block_and_a_member_by_sequence_are_not_pooled(
    tmp_path: Path,
) -> None:
    by_sequence = {TRAIN[0]: 0, TRAIN[1]: 1}
    lookups = [
        _run(tmp_path, "a", _block_table(BLOCKS), entries=TRAIN[:2]),
        _run(tmp_path, "b", _fold_table(by_sequence), entries=TRAIN[:2]),
    ]
    feature = _feature(2, weights="fit", labels={"kind": "behavior"})
    dependency_lookups = {f"_input_{i}": lookup for i, lookup in enumerate(lookups)}

    with pytest.raises(ValueError, match="input 1 deals its cross-validation folds"):
        feature.load_state(tmp_path / "blend", {}, dependency_lookups)

    run, _ = _ensemble_run(tmp_path, "average", lookups)
    assert not (run / "cv_folds.parquet").exists()


def test_fitted_weights_say_so_when_an_input_resolved_no_output(
    tmp_path: Path,
) -> None:
    """An empty member lookup is not a member that writes no fold table."""
    lookups = [_run(tmp_path, "a", _fold_table(FOLDS)), {}]
    feature = _feature(2, weights="fit", labels={"kind": "behavior"})
    dependency_lookups = {f"_input_{i}": lookup for i, lookup in enumerate(lookups)}

    with pytest.raises(ValueError, match="inputs 1 resolved no output entry"):
        feature.load_state(tmp_path / "blend", {}, dependency_lookups)


@pytest.mark.parametrize("weights", ["uniform", [3.0, 1.0]])
def test_fixed_weights_average_members_of_which_some_have_a_fold_table(
    tmp_path: Path, weights: str | list[float]
) -> None:
    """Fixed weights need no fold table, so mixed members are averaged without one."""
    lookups = [_run(tmp_path, "a", _fold_table(FOLDS)), _run(tmp_path, "b", None)]

    run, _ = _ensemble_run(tmp_path, "blend", lookups, weights=weights)

    assert (run / "ensemble.json").exists()
    assert not (run / "cv_folds.parquet").exists()


def test_fixed_weights_average_members_that_deal_different_folds(
    tmp_path: Path,
) -> None:
    other = {entry: 1 - fold for entry, fold in FOLDS.items()} | {TRAIN[0]: 5}
    lookups = [
        _run(tmp_path, "a", _fold_table(FOLDS)),
        _run(tmp_path, "b", _fold_table(other)),
    ]

    run, _ = _ensemble_run(tmp_path, "blend", lookups)

    assert (run / "ensemble.json").exists()
    assert not (run / "cv_folds.parquet").exists()


def test_fixed_weights_over_an_input_that_resolved_no_output_fit_nothing(
    tmp_path: Path,
) -> None:
    lookups = [_run(tmp_path, "a", _fold_table(FOLDS)), {}]

    run, _ = _ensemble_run(tmp_path, "blend", lookups)

    assert (run / "ensemble.json").exists()
    assert not (run / "cv_folds.parquet").exists()


def test_an_ensemble_of_ensembles_finds_the_fold_table_where_a_classifier_has_it(
    tmp_path: Path,
) -> None:
    first = [
        _run(tmp_path, "a", _fold_table(FOLDS)),
        _run(tmp_path, "b", _fold_table(FOLDS)),
    ]
    second = [
        _run(tmp_path, "c", _fold_table(FOLDS)),
        _run(tmp_path, "d", _fold_table(FOLDS)),
    ]
    _, average_one = _ensemble_run(tmp_path, "average-one", first)
    _, average_two = _ensemble_run(tmp_path, "average-two", second)

    run, _ = _ensemble_run(tmp_path, "blend", [average_one, average_two])

    table = pd.read_parquet(run / "cv_folds.parquet")
    assert dict(zip(zip(table["group"], table["sequence"]), table["fold"])) == FOLDS


def test_a_cached_run_gives_its_weights_back_without_a_fit(tmp_path: Path) -> None:
    lookups = [_run(tmp_path, "a", None), _run(tmp_path, "b", None)]
    run, _ = _ensemble_run(tmp_path, "blend", lookups, weights=[3.0, 1.0])
    first = _probs([1.0, 0.0, 0.0])
    second = _probs([0.0, 0.0, 1.0])
    merged = _merged([_member(first), _member(second)], tmp_path / "in")

    again = _feature(2, weights=[3.0, 1.0])
    assert again.load_state(run, {}, {})

    np.testing.assert_allclose(again.apply(merged)["prob_0"].to_numpy(), [0.75])
    assert json.loads((run / "ensemble_report.json").read_text())["weights_used"] == [
        0.75,
        0.25,
    ]


# --- Fitted weights -------------------------------------------------------------

FRAMES = 40
EPSILON = 1e-12


def _truth(sequence: int) -> npt.NDArray[np.int64]:
    """The labels of an entry: the classes in turn, in runs of three frames."""
    return np.asarray(
        [CLASSES[(frame // 3 + sequence) % 3] for frame in range(FRAMES)],
        dtype=np.int64,
    )


def _members_for(
    sequence: int, *, accuracy: tuple[float, float], seed: int
) -> list[pd.DataFrame]:
    """Two members that name the true class with the given accuracies, else another."""
    rng = np.random.default_rng(seed + sequence)
    truth = _truth(sequence)
    members: list[pd.DataFrame] = []
    for hit in accuracy:
        probs = rng.dirichlet(np.ones(3), size=FRAMES) * 0.5
        wrong = rng.random(FRAMES) >= hit
        named = np.where(wrong, (truth + rng.integers(1, 3, FRAMES)) % 3, truth)
        probs[np.arange(FRAMES), named] += 0.5
        members.append(_member(probs, sequence=f"s{sequence}"))
    return members


class _Study:
    """Train entries with labels and members, a fold table, and the lookups of a fit."""

    def __init__(self, root: Path, *, entries: int = 4, seed: int = 3) -> None:
        self.root = root
        self.tables: dict[Entry, pd.DataFrame] = {}
        self.files: DependencyLookup = {}
        for sequence in range(entries):
            entry = (GROUP, f"s{sequence}")
            members = _members_for(sequence, accuracy=(0.75, 0.6), seed=seed)
            self.tables[entry] = _merged(members, root / "merged" / entry[1])
            events: list[Event] = [
                (frame, 0, -1, int(label))
                for frame, label in enumerate(_truth(sequence))
            ]
            self.files[entry] = write_sparse_labels(
                root / "labels" / f"{entry[1]}.npz", events
            )
        self.folds = {entry: index % 2 for index, entry in enumerate(self.tables)}

    def lookups(self, fold_table: bool = True) -> dict[str, DependencyLookup]:
        table = _fold_table(self.folds) if fold_table else None
        entries = list(self.tables)
        return {
            "_input_0": _run(self.root, "member-0", table, entries),
            "_input_1": _run(self.root, "member-1", table, entries),
            "labels": self.files,
        }

    def stream(self, order: Sequence[Entry] | None = None) -> InputStream:
        keys = list(self.tables) if order is None else list(order)
        items = [(f"{g}__{s}", self.tables[(g, s)]) for g, s in keys]
        return InputStream(lambda: iter(items), n_entries=len(items))


def _fit_feature(
    study: _Study, **params: object
) -> tuple[ProbabilityEnsembleFeature, Path]:
    feature = _feature(
        2, weights="fit", labels={"kind": "behavior"}, score_classes=[0, 1, 2], **params
    )
    run = study.root / "fit-run"
    assert not feature.load_state(run, {}, study.lookups())
    return feature, run


def _oracle(study: _Study, grid_step: float = 0.05) -> tuple[list[float], float]:
    """The weights that a brute-force search over the grid scores highest."""
    truth = np.concatenate([_truth(i) for i in range(len(study.tables))])
    stacked = np.stack(
        [
            np.concatenate(
                [
                    study.tables[(GROUP, f"s{i}")][
                        [member_probability_column(c, position) for c in CLASSES]
                    ].to_numpy()
                    for i in range(len(study.tables))
                ]
            )
            for position in range(2)
        ]
    )
    parts = round(1 / grid_step)
    best, best_score = [0.0, 1.0], -1.0
    for units in simplex_grid(2, parts):
        weights = np.asarray(units) / parts
        predicted = np.asarray(CLASSES)[
            np.argmax(np.tensordot(weights, stacked, 1), axis=1)
        ]
        value = classification_scores(
            truth, predicted, classes=CLASSES, average_over=CLASSES
        )["macro_f1"]
        if value > best_score + EPSILON:
            best, best_score = weights.tolist(), value
    return best, best_score


def test_the_fit_chooses_the_weights_a_brute_force_search_scores_highest(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path)
    feature, run = _fit_feature(study)

    feature.fit(study.stream())
    feature.save_state(run)

    expected, expected_score = _oracle(study)
    report = json.loads((run / "ensemble_report.json").read_text())
    np.testing.assert_allclose(report["weights_used"], expected)
    assert report["objective"] == pytest.approx(expected_score)
    assert report["objective"] >= max(report["member_objectives"])
    assert report["n_entries"] == 4 and report["n_frames"] == 4 * FRAMES
    assert 0.0 < report["weights_used"][0] < 1.0


def test_a_tie_between_weights_goes_to_the_first_in_the_grid(tmp_path: Path) -> None:
    study = _Study(tmp_path)
    for entry, table in study.tables.items():
        copy = table.copy()
        for label in CLASSES:
            copy[member_probability_column(label, 1)] = copy[
                member_probability_column(label, 0)
            ]
        study.tables[entry] = copy
    feature, _ = _fit_feature(study)

    feature.fit(study.stream())

    # Every weight vector blends two equal members into the same predictions.
    run = tmp_path / "tie"
    feature.save_state(run)
    report = json.loads((run / "ensemble_report.json").read_text())
    assert report["weights_used"] == [0.0, 1.0]


def test_the_weights_do_not_depend_on_the_order_entries_arrive_in(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path)
    forward, forward_run = _fit_feature(study)
    forward.fit(study.stream())
    forward.save_state(forward_run)
    study.root = tmp_path / "second"
    study.root.mkdir()
    backward, backward_run = _fit_feature(study)
    backward.fit(study.stream(list(reversed(study.tables))))
    backward.save_state(backward_run)

    first = json.loads((forward_run / "ensemble_report.json").read_text())
    second = json.loads((backward_run / "ensemble_report.json").read_text())
    assert first["weights_used"] == second["weights_used"]


def test_entries_outside_the_fold_table_do_not_choose_the_weights(
    tmp_path: Path,
) -> None:
    study = _Study(tmp_path)
    plain, plain_run = _fit_feature(study)
    plain.fit(study.stream())
    plain.save_state(plain_run)
    # A test entry whose members are wrong about everything, labeled all the same.
    test_entry = ("test", "t0")
    truth = np.zeros(FRAMES, dtype=np.int64)
    adversarial = [
        _member(np.tile([0.0, 1.0, 0.0], (FRAMES, 1)), sequence="t0"),
        _member(np.tile([0.0, 0.0, 1.0], (FRAMES, 1)), sequence="t0"),
    ]
    adversarial[0]["group"] = adversarial[1]["group"] = "test"
    study.tables[test_entry] = _merged(adversarial, tmp_path / "merged" / "t0")
    study.files[test_entry] = write_sparse_labels(
        tmp_path / "labels" / "t0.npz",
        [(frame, 0, -1, int(label)) for frame, label in enumerate(truth)],
    )
    study.root = tmp_path / "with-test"
    study.root.mkdir()
    wider, wider_run = _fit_feature(study)

    wider.fit(study.stream())
    wider.save_state(wider_run)

    first = json.loads((plain_run / "ensemble_report.json").read_text())
    second = json.loads((wider_run / "ensemble_report.json").read_text())
    assert first["weights_used"] == second["weights_used"]


def test_a_fit_over_members_without_fold_tables_is_refused(tmp_path: Path) -> None:
    study = _Study(tmp_path)
    feature = _feature(2, weights="fit", labels={"kind": "behavior"})
    assert not feature.load_state(tmp_path / "run", {}, study.lookups(fold_table=False))

    with pytest.raises(ValueError, match="no input writes cv_folds.parquet"):
        feature.fit(study.stream())


def test_a_fold_entry_that_is_not_in_the_scope_is_refused(tmp_path: Path) -> None:
    study = _Study(tmp_path)
    feature, _ = _fit_feature(study)
    kept = [entry for entry in study.tables if entry != (GROUP, "s3")]

    with pytest.raises(ValueError, match=r"1 entries of the members' folds.*g/s3"):
        feature.fit(study.stream(kept))


def _fit_over_blocks(
    study: _Study, name: str, blocks: dict[tuple[Entry, int], tuple[int, int, int]]
) -> dict[str, object]:
    """Fit the weights over members whose fold table is *blocks*, and read the report."""
    table = _block_table(blocks)
    entries = list(study.tables)
    lookups = {
        "_input_0": _run(study.root, f"{name}-member-0", table, entries),
        "_input_1": _run(study.root, f"{name}-member-1", table, entries),
        "labels": study.files,
    }
    feature = _feature(
        2, weights="fit", labels={"kind": "behavior"}, score_classes=[0, 1, 2]
    )
    run = study.root / name
    assert not feature.load_state(run, {}, lookups)
    feature.fit(study.stream())
    feature.save_state(run)
    return json.loads((run / "ensemble_report.json").read_text())


def test_fitted_weights_pool_the_rows_inside_the_blocks_and_no_others(
    tmp_path: Path,
) -> None:
    half = FRAMES // 2
    blocks = {
        (entry, 0): (0, half, index % 2)
        for index, entry in enumerate(_Study(tmp_path / "probe").tables)
    }
    clean = _Study(tmp_path / "clean")
    poisoned = _Study(tmp_path / "poisoned")
    for entry, table in poisoned.tables.items():
        copy = table.copy()
        # Past the block, the members are certain of the wrong class.
        for position in range(2):
            for column, label in enumerate(CLASSES):
                copy.loc[
                    copy["frame"] >= half, member_probability_column(label, position)
                ] = 1.0 if column == 0 else 0.0
        poisoned.tables[entry] = copy

    wanted = _fit_over_blocks(clean, "fit", blocks)
    got = _fit_over_blocks(poisoned, "fit", blocks)

    assert got["weights_used"] == wanted["weights_used"]
    assert got["objective"] == wanted["objective"]
    assert got["n_frames"] == 4 * half
    whole = _fit_over_blocks(
        _Study(tmp_path / "whole"),
        "fit",
        {(entry, 0): (0, FRAMES, 0) for entry in TRAIN},
    )
    assert whole["n_frames"] == 4 * FRAMES
