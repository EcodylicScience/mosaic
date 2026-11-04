"""Predictions made outside mosaic, converted and scored by recipe steps.

Each test writes a predictions table under ``labels_raw/predictions/<set>/``,
scans it as a ``predictions_table`` labels source, and runs a recipe whose
``convert-labels`` step of kind ``predictions`` feeds ``label-predictions``,
which the scoring steps read as they read a classifier. An ``xgboost`` run
written out as such a table scores, tunes and blends as the run itself does.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score  # pyright: ignore[reportUnknownVariableType]

from mosaic.behavior.feature_library.classifier_common import read_fold_table
from mosaic.behavior.feature_library.helpers import cv_block_assignment
from mosaic.core.dataset import Dataset
from mosaic.core.evaluation import EvaluationReport
from mosaic.core.helpers import make_entry_key
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import PipelineRun, Recipe, plan_pipeline, run_pipeline
from mosaic.core.scope import Scope
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    calms21_dataset,
    scan_predictions_tables,
)

TRAIN = Scope(groups=["train"])
BOTH = Scope(groups=["train", "test"])
TRAIN_ENTRIES = [("train", f"s{i}") for i in range(1, 7)]
TEST_ENTRIES = [("test", "t1"), ("test", "t2")]
CLASSES = [0, 1, 2]
PROBABILITIES = [f"prob_{c}" for c in CLASSES]

Step = dict[str, JsonValue]


def _classifier_steps(**classifier: JsonValue) -> list[Step]:
    """The CalMS21 conversions, the pair features and a cross-validated ``xgboost``."""
    return [
        CONVERT_TRACKS_STEP,
        CONVERT_LABELS_STEP,
        {
            "id": "heading",
            "type": "feature",
            "feature": "heading",
            "inputs": ["tracks"],
            "tracks": {"step": "tracks"},
            "params": {"method": "two_point", "front_idx": 0, "rear_idx": 3},
        },
        {
            "id": "pairs",
            "type": "feature",
            "feature": "pair-position",
            "inputs": ["tracks", {"step": "heading"}],
            "tracks": {"step": "tracks"},
        },
        {
            "id": "labeled",
            "type": "feature",
            "feature": "extract-labeled-templates",
            "inputs": [{"step": "pairs"}],
            "params": {
                "labels": {"step": "labels"},
                "n_total": 300,
                "test_fraction": 0.0,
            },
        },
        {
            "id": "clf",
            "type": "feature",
            "feature": "xgboost",
            "inputs": [{"step": "pairs"}],
            "params": {
                "templates": {"step": "labeled"},
                "default_class": 0,
                "n_estimators": 5,
                "n_jobs": 1,
                "random_state": 1,
                "cv_folds": 3,
                **classifier,
            },
        },
    ]


def _converted(step: str, prediction_set: str) -> Step:
    """A ``convert-labels`` step of kind ``predictions`` over one set's files."""
    return {
        "id": step,
        "type": "op",
        "kind": "convert-labels",
        "params": {
            "kind": "predictions",
            "source_format": "predictions_table",
            "params": {"files": f"labels_raw/predictions/{prediction_set}/*"},
        },
    }


def _bridge(step: str, converted: str, rows: JsonValue) -> Step:
    """``label-predictions`` reading the variant of *converted* onto the rows of *rows*."""
    bridge: Step = {
        "id": step,
        "type": "feature",
        "feature": "label-predictions",
        "inputs": [rows],
        "params": {"predictions": {"step": converted}},
    }
    if rows == "tracks":
        bridge["tracks"] = {"step": "tracks"}
    return bridge


def _evaluate(step: str, scored: str, **params: JsonValue) -> Step:
    return {
        "id": step,
        "type": "feature",
        "feature": "evaluate-labels",
        "inputs": [{"step": scored}],
        "params": {"labels": {"step": "labels"}, "average_over": [*CLASSES], **params},
    }


def _tune(step: str, tuned: str) -> Step:
    return {
        "id": step,
        "type": "feature",
        "feature": "tune-threshold",
        "inputs": [{"step": tuned}],
        "params": {
            "labels": {"step": "labels"},
            "default_class": 0,
            "average_over": [*CLASSES],
        },
    }


def _blend(step: str, members: tuple[str, str]) -> Step:
    return {
        "id": step,
        "type": "feature",
        "feature": "probability-ensemble",
        "inputs": [{"step": member} for member in members],
        "params": {
            "default_class": 0,
            "weights": "fit",
            "labels": {"step": "labels"},
            "score_classes": [*CLASSES],
        },
    }


def _recipe(steps: list[Step]) -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": "external", "steps": steps}
    )


def _root(
    ds: Dataset,
    recipe: Recipe,
    run: PipelineRun,
    step: str,
    scopes: Mapping[str, Scope],
) -> Path:
    outcome = next(o for o in run.outcomes if o.step_id == step)
    planned = plan_pipeline(ds, recipe, step_scopes=scopes).step(step)
    return ds.get_root("features") / planned.storage_name / str(outcome.run_id)


def _outputs(root: Path, entries: list[tuple[str, str]]) -> pd.DataFrame:
    """The rows a run wrote for *entries*, in entry order."""
    return pd.concat(
        [
            pd.read_parquet(root / f"{make_entry_key(group, sequence)}.parquet")
            for group, sequence in entries
        ],
        ignore_index=True,
    )


def _as_predictions(rows: pd.DataFrame, folds: npt.NDArray[np.int64]) -> pd.DataFrame:
    """A classifier's pair rows as a predictions table, with each row's fold."""
    table = rows[
        ["group", "sequence", "frame", "id1", "id2", "predicted_label", *PROBABILITIES]
    ].copy()
    table["fold"] = folds
    return table


def _report(root: Path) -> EvaluationReport:
    return EvaluationReport.model_validate_json((root / "metrics.json").read_text())


def _annotations(ds: Dataset, stem: str) -> dict[str, npt.NDArray[np.int64]]:
    """The CalMS21 annotation of every frame of each sequence of one raw file."""
    payload = np.load(ds.base_dir / "raw" / f"{stem}.npy", allow_pickle=True).item()
    return {
        sequence: np.asarray(arrays["annotations"], dtype=np.int64)
        for sequences in payload.values()
        for sequence, arrays in sequences.items()
    }


def _pair_table(
    group: str, annotations: Mapping[str, npt.NDArray[np.int64]]
) -> pd.DataFrame:
    """The resident-to-intruder predictions of each frame, as CalMS21 pairs them."""
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "group": group,
                    "sequence": sequence,
                    "frame": np.arange(len(labels)),
                    "id1": 0,
                    "id2": 1,
                    "predicted_label": labels,
                }
            )
            for sequence, labels in sorted(annotations.items())
        ],
        ignore_index=True,
    )


def _noisy_probabilities(
    labels: npt.NDArray[np.int64], seed: int
) -> npt.NDArray[np.float64]:
    """Probabilities that favor each row's label, from a fixed seed."""
    rng = np.random.default_rng(seed)
    noise = rng.dirichlet(np.ones(len(CLASSES)), size=len(labels))
    return 0.5 * np.eye(len(CLASSES))[labels] + 0.5 * noise


# --- the same metrics as mosaic ------------------------------------------------


def test_an_xgboost_run_written_as_predictions_scores_tunes_and_blends_as_itself(
    tmp_path: Path,
) -> None:
    ds = calms21_dataset(
        tmp_path / "ds",
        {
            "train": {"a": {name: 240 for _, name in TRAIN_ENTRIES}},
            "test": {"a": {name: 240 for _, name in TEST_ENTRIES}},
        },
        seed=11,
        n_classes=3,
    )
    classifier = _recipe(_classifier_steps())
    scopes = {"labeled": TRAIN, "clf": BOTH}
    first = run_pipeline(ds, classifier, step_scopes=scopes)
    clf = _root(ds, classifier, first, "clf", scopes)
    fold_of = {
        (str(g), str(s)): int(f)
        for g, s, f in read_fold_table(clf / "cv_folds.parquet").itertuples(index=False)
    }
    assert sorted(fold_of) == sorted(TRAIN_ENTRIES)
    rows = _outputs(clf, TRAIN_ENTRIES + TEST_ENTRIES)
    folds = np.asarray(
        [
            fold_of.get((g, s), -1)
            for g, s in zip(rows["group"], rows["sequence"], strict=True)
        ],
        dtype=np.int64,
    )
    scan_predictions_tables(ds, {"xgb": _as_predictions(rows, folds)})

    recipe = _recipe(
        [
            *_classifier_steps(),
            _converted("predictions", "xgb"),
            _bridge("bridge", "predictions", {"step": "pairs"}),
            _evaluate("evaluate_clf", "clf", perspective=0),
            _evaluate("evaluate_bridge", "bridge", perspective=0),
            _tune("tune_clf", "clf"),
            _tune("tune_bridge", "bridge"),
            _blend("blend", ("clf", "bridge")),
        ]
    )
    scopes = {
        **scopes,
        **{
            step: TRAIN
            for step in (
                "evaluate_clf",
                "evaluate_bridge",
                "tune_clf",
                "tune_bridge",
                "blend",
            )
        },
    }
    run = run_pipeline(ds, recipe, step_scopes=scopes)

    states = {o.step_id: o.state for o in run.outcomes}
    assert states["clf"] == "cached"
    assert {states[step] for step in scopes if step not in ("labeled", "clf")} == {
        "ran"
    }

    def root(step: str) -> Path:
        return _root(ds, recipe, run, step, scopes)

    assert (root("evaluate_bridge") / "metrics.json").read_text() == (
        root("evaluate_clf") / "metrics.json"
    ).read_text()
    assert _report(root("evaluate_bridge")).n_entries == len(TRAIN_ENTRIES)
    pd.testing.assert_frame_equal(
        pd.read_parquet(root("tune_bridge") / "thresholds.parquet"),
        pd.read_parquet(root("tune_clf") / "thresholds.parquet"),
    )
    tuned_bridge = json.loads((root("tune_bridge") / "tune_report.json").read_text())
    tuned_clf = json.loads((root("tune_clf") / "tune_report.json").read_text())
    assert tuned_bridge == tuned_clf
    assert tuned_bridge["nested"] is not None
    pd.testing.assert_frame_equal(
        read_fold_table(root("bridge") / "cv_folds.parquet"),
        read_fold_table(clf / "cv_folds.parquet"),
    )
    blended = json.loads((root("blend") / "ensemble_report.json").read_text())
    assert len(blended["weights_used"]) == 2
    assert blended["n_entries"] == len(TRAIN_ENTRIES)


# --- block folds ---------------------------------------------------------------

BLOCKS = 6
CV_SEED = 0
BY_BLOCK: dict[str, JsonValue] = {
    "cv_unit": "block",
    "cv_blocks_per_sequence": BLOCKS,
    "cv_gap_frames": 5,
    "cv_seed": CV_SEED,
}


def _block_of(rows: pd.DataFrame, table: pd.DataFrame) -> npt.NDArray[np.int64]:
    """The block of each row's frame in a fold table of blocks, ``-1`` outside every block."""
    blocks = np.full(len(rows), -1, dtype=np.int64)
    frames = rows["frame"].to_numpy()
    for block, start, stop in zip(
        table["block"], table["start"], table["stop"], strict=True
    ):
        blocks[(frames >= start) & (frames < stop)] = int(block)
    return blocks


def test_block_folds_match_with_the_block_column_and_merge_without_it(
    tmp_path: Path,
) -> None:
    dealt = cv_block_assignment(
        [("train", "s1", block) for block in range(BLOCKS)], 3, CV_SEED
    )
    neighbors = [
        block
        for block in range(BLOCKS - 1)
        if dealt[("train", "s1", block)] == dealt[("train", "s1", block + 1)]
    ]
    assert neighbors, "the seed must give two neighboring blocks one fold"

    ds = calms21_dataset(
        tmp_path / "ds", {"train": {"a": {"s1": 600}}}, seed=11, n_classes=3
    )
    classifier = _recipe(_classifier_steps(**BY_BLOCK))
    scopes = {"labeled": TRAIN, "clf": TRAIN}
    first = run_pipeline(ds, classifier, step_scopes=scopes)
    clf = _root(ds, classifier, first, "clf", scopes)
    blocks = pd.read_parquet(clf / "cv_folds.parquet")
    assert len(blocks) == BLOCKS
    rows = _outputs(clf, [("train", "s1")])
    block = _block_of(rows, blocks)
    fold_of = dict(zip(blocks["block"].tolist(), blocks["fold"].tolist(), strict=True))
    folds = np.asarray([fold_of.get(int(b), -1) for b in block], dtype=np.int64)
    assert (block >= 0).any()
    with_block = _as_predictions(rows, folds)
    with_block["block"] = pd.array(
        [int(b) if b >= 0 else None for b in block], dtype="Int64"
    )
    scan_predictions_tables(
        ds,
        {"with_block": with_block, "without_block": _as_predictions(rows, folds)},
    )

    named = _recipe(
        [
            *_classifier_steps(**BY_BLOCK),
            _converted("with_block", "with_block"),
            _bridge("bridge", "with_block", {"step": "pairs"}),
        ]
    )
    run = run_pipeline(ds, named, step_scopes=scopes)
    bridged = _root(ds, named, run, "bridge", scopes)
    pd.testing.assert_frame_equal(
        read_fold_table(bridged / "cv_folds.parquet"),
        read_fold_table(clf / "cv_folds.parquet"),
    )

    merged = _recipe(
        [
            *_classifier_steps(**BY_BLOCK),
            _converted("without_block", "without_block"),
            _bridge("bridge", "without_block", {"step": "pairs"}),
            _blend("blend", ("clf", "bridge")),
        ]
    )
    scopes = {**scopes, "blend": TRAIN}
    with pytest.raises(ValueError, match="input 1 deals its cross-validation folds"):
        _ = run_pipeline(ds, merged, step_scopes=scopes)
    without = plan_pipeline(ds, merged, step_scopes=scopes).step("bridge")
    unnamed = (
        ds.get_root("features") / without.storage_name / str(without.run_id)
    ) / "cv_folds.parquet"
    assert len(pd.read_parquet(unnamed)) == BLOCKS - len(neighbors)


# --- identical kinds -------------------------------------------------------------


def test_an_annotators_labels_scored_against_themselves_agree_fully(
    tmp_path: Path,
) -> None:
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 240, "s2": 240}}},
        seed=11,
        n_classes=3,
    )
    annotations = _annotations(ds, "train")
    same = _pair_table("train", annotations)
    changed = same.copy()
    tenth = np.arange(len(changed)) % 10 == 0
    changed.loc[tenth, "predicted_label"] = (
        changed.loc[tenth, "predicted_label"] + 1
    ) % len(CLASSES)
    scan_predictions_tables(ds, {"same": same, "changed": changed})
    recipe = _recipe(
        [
            CONVERT_TRACKS_STEP,
            CONVERT_LABELS_STEP,
            _converted("same", "same"),
            _converted("changed", "changed"),
            _bridge("bridge_same", "same", "tracks"),
            _bridge("bridge_changed", "changed", "tracks"),
            _evaluate("evaluate_same", "bridge_same"),
            _evaluate("evaluate_changed", "bridge_changed"),
        ]
    )

    run = run_pipeline(ds, recipe)

    assert {o.state for o in run.outcomes} == {"ran"}
    identical = _report(_root(ds, recipe, run, "evaluate_same", {}))
    assert identical.accuracy == 1.0
    assert identical.kappa == 1.0
    assert identical.n_scored == len(same)
    different = _report(_root(ds, recipe, run, "evaluate_changed", {}))
    assert different.n_scored == len(same)
    assert different.kappa == pytest.approx(
        cohen_kappa_score(same["predicted_label"], changed["predicted_label"]),
        rel=1e-12,
    )


# --- nested on external folds ----------------------------------------------------


def test_folds_by_entry_give_a_nested_estimate(tmp_path: Path) -> None:
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {name: 240 for _, name in TRAIN_ENTRIES}}},
        seed=11,
        n_classes=3,
    )
    table = _pair_table("train", _annotations(ds, "train"))
    labels = table.pop("predicted_label").to_numpy(dtype=np.int64)
    table[PROBABILITIES] = _noisy_probabilities(labels, seed=3)
    position = {sequence: i for i, (_, sequence) in enumerate(TRAIN_ENTRIES)}
    table["fold"] = [position[s] % 3 for s in table["sequence"]]
    scan_predictions_tables(ds, {"external": table})
    recipe = _recipe(
        [
            CONVERT_TRACKS_STEP,
            CONVERT_LABELS_STEP,
            _converted("predictions", "external"),
            _bridge("bridge", "predictions", "tracks"),
            _tune("tune", "bridge"),
        ]
    )

    run = run_pipeline(ds, recipe)

    folds = read_fold_table(_root(ds, recipe, run, "bridge", {}) / "cv_folds.parquet")
    assert folds.to_dict("list") == {
        "group": ["train"] * len(TRAIN_ENTRIES),
        "sequence": [s for _, s in TRAIN_ENTRIES],
        "fold": [i % 3 for i in range(len(TRAIN_ENTRIES))],
    }
    report = json.loads(
        (_root(ds, recipe, run, "tune", {}) / "tune_report.json").read_text()
    )
    assert report["n_entries"] == len(TRAIN_ENTRIES)
    assert report["nested"] is not None


def test_folds_by_block_within_one_entry_give_a_nested_estimate(
    tmp_path: Path,
) -> None:
    ds = calms21_dataset(
        tmp_path / "ds", {"train": {"a": {"s1": 600}}}, seed=11, n_classes=3
    )
    table = _pair_table("train", _annotations(ds, "train"))
    labels = table.pop("predicted_label").to_numpy(dtype=np.int64)
    table[PROBABILITIES] = _noisy_probabilities(labels, seed=5)
    table["fold"] = (table["frame"] // 100) % 3
    scan_predictions_tables(ds, {"external": table})
    recipe = _recipe(
        [
            CONVERT_TRACKS_STEP,
            CONVERT_LABELS_STEP,
            _converted("predictions", "external"),
            _bridge("bridge", "predictions", "tracks"),
            _tune("tune", "bridge"),
        ]
    )

    run = run_pipeline(ds, recipe)

    blocks = pd.read_parquet(_root(ds, recipe, run, "bridge", {}) / "cv_folds.parquet")
    assert blocks[["block", "start", "stop", "fold"]].to_dict("list") == {
        "block": list(range(6)),
        "start": [100 * b for b in range(6)],
        "stop": [100 * (b + 1) for b in range(6)],
        "fold": [b % 3 for b in range(6)],
    }
    report = json.loads(
        (_root(ds, recipe, run, "tune", {}) / "tune_report.json").read_text()
    )
    assert report["n_entries"] == 1
    assert report["nested"] is not None
