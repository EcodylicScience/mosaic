"""`lightning-action` in a pair-level chain: trained on the entries its templates run kept, applied to all.

The CalMS21 chain scopes `extract-labeled-templates` to the training group, and the
classifier to both groups, so that one step fits and then predicts every entry. The
entries that train are those of the templates run's provenance, so a held-out group
never trains however the step is scoped. The package is stubbed, as in
`test_lightning_action_feature.py`, so that the chain runs in every job.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import PipelineRun, Recipe, run_pipeline
from mosaic.core.scope import Scope
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    LightningActionRecorder,
    calms21_dataset,
    install_lightning_action_stub,
)

TRAIN, TEST = Scope(groups=["train"]), Scope(groups=["test"])
BOTH = Scope(groups=["train", "test"])
WINDOW = 10
N_CLASSES = 3
DEFAULT_CLASS = N_CLASSES  # a class that no frame is labeled with
TRAIN_FRAMES = {"s1": 60, "s2": 70, "s3": 80, "s4": 90, "s5": 100, "s6": 110}
TEST_FRAMES = {"t1": 45, "t2": 73}


@pytest.fixture(autouse=True)
def package(monkeypatch: pytest.MonkeyPatch) -> LightningActionRecorder:
    return install_lightning_action_stub(monkeypatch)


def dataset(base: Path) -> Dataset:
    """Entries of distinct lengths, so that a training file names its entry."""
    return calms21_dataset(
        base,
        {
            "train": {"a": TRAIN_FRAMES},
            "test": {"a": TEST_FRAMES},
        },
        seed=11,
        n_classes=N_CLASSES,
    )


def recipe(*, test_fraction: float, apply_only: bool = False) -> Recipe:
    steps: list[Mapping[str, object]] = [
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
                "test_fraction": test_fraction,
            },
        },
        {
            "id": "classifier",
            "type": "feature",
            "feature": "lightning-action",
            "inputs": [{"step": "pairs"}],
            "params": {
                "templates": {"step": "labeled"},
                "labels": {"step": "labels"},
                "default_class": DEFAULT_CLASS,
                "device": "cpu",
                "num_epochs": 1,
                "sequence_length": WINDOW,
                "num_layers": 1,
                "num_lags": 1,
            },
        },
        {
            "id": "evaluate",
            "type": "feature",
            "feature": "evaluate-labels",
            "inputs": [{"step": "classifier"}],
            "params": {
                "labels": {"step": "labels"},
                "perspective": 0,
                "average_over": [0, 1, 2],
            },
        },
    ]
    if apply_only:
        steps.append(
            {
                "id": "applied",
                "type": "feature",
                "feature": "lightning-action",
                "inputs": [{"step": "pairs"}],
                "params": {
                    "model": {"step": "classifier"},
                    "default_class": DEFAULT_CLASS,
                    "device": "cpu",
                },
            }
        )
    return Recipe.model_validate(
        {"schema_version": 1, "name": "sequence-classifier", "steps": steps}
    )


def _run_ids(run: PipelineRun) -> dict[str, str]:
    return {outcome.step_id: str(outcome.run_id) for outcome in run.outcomes}


def _states(run: PipelineRun) -> dict[str, str]:
    return {outcome.step_id: outcome.state for outcome in run.outcomes}


def _trained_lengths(package: LightningActionRecorder) -> list[int]:
    """The rows of each training file, which are the frames of one resident-focal series."""
    return sorted(len(features) for features, _ in package.trained.values())


def _outputs(ds: Dataset, step_run_id: str) -> dict[str, pd.DataFrame]:
    """Each entry's output table of a feature run, by `group/sequence`."""
    tables: dict[str, pd.DataFrame] = {}
    for path in ds.get_root("features").glob(
        f"lightning-action__*/{step_run_id}/*.parquet"
    ):
        table = pd.read_parquet(path)
        tables[f"{table['group'].iloc[0]}/{table['sequence'].iloc[0]}"] = table
    return tables


def test_the_scoped_chain_trains_on_the_train_group_and_predicts_every_entry(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    ds = dataset(tmp_path / "ds")

    run = run_pipeline(
        ds,
        recipe(test_fraction=0.0),
        step_scopes={"labeled": TRAIN, "classifier": BOTH, "evaluate": TEST},
    )

    assert set(_states(run).values()) == {"ran"}
    # The resident-focal series of each training entry is one run of every frame, cut
    # in whole windows. The intruder-focal series is unlabeled and writes no file.
    assert _trained_lengths(package) == sorted(TRAIN_FRAMES.values())
    outputs = _outputs(ds, _run_ids(run)["classifier"])
    expected = {f"train/{s}": n for s, n in TRAIN_FRAMES.items()} | {
        f"test/{s}": n for s, n in TEST_FRAMES.items()
    }
    assert {key: len(table) // 2 for key, table in outputs.items()} == expected
    for table in outputs.values():
        probabilities = table.filter(like="prob_")
        assert not probabilities.isna().any(axis=None)
        assert table["predicted_label"].isin(range(N_CLASSES)).all()
        assert set(table["perspective"]) == {0, 1}


def test_a_held_out_entry_of_the_templates_run_never_trains_when_one_scope_holds_it(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    ds = dataset(tmp_path / "ds")

    run = run_pipeline(
        ds,
        recipe(test_fraction=0.25),
        step_scopes={"labeled": BOTH, "classifier": BOTH, "evaluate": TEST},
    )

    assert set(_states(run).values()) == {"ran"}
    [path] = ds.get_root("features").glob(
        "extract-labeled-templates__*/*/template_provenance.parquet"
    )
    provenance = pd.read_parquet(path)
    lengths = TRAIN_FRAMES | TEST_FRAMES
    kept = sorted(
        lengths[sequence]
        for sequence, split in zip(provenance["sequence"], provenance["split"])
        if split == "train"
    )
    held_out = provenance[provenance["split"] == "test"]
    assert len(held_out) == 2
    assert _trained_lengths(package) == [n - n % WINDOW for n in kept]
    # The held-out entries are still predicted.
    assert len(_outputs(ds, _run_ids(run)["classifier"])) == len(lengths)


def test_a_step_given_the_fitted_model_predicts_new_entries_without_training_again(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    ds = dataset(tmp_path / "ds")

    run = run_pipeline(
        ds,
        recipe(test_fraction=0.0, apply_only=True),
        step_scopes={
            "labeled": TRAIN,
            "classifier": TRAIN,
            "applied": TEST,
            "evaluate": TRAIN,
        },
    )

    assert set(_states(run).values()) == {"ran"}
    assert package.trainings == 1
    applied = _outputs(ds, _run_ids(run)["applied"])
    assert sorted(applied) == ["test/t1", "test/t2"]

    # A run that fits nothing is scope-free: more entries are the same run.
    wider = run_pipeline(
        ds,
        recipe(test_fraction=0.0, apply_only=True),
        step_scopes={
            "labeled": TRAIN,
            "classifier": TRAIN,
            "applied": BOTH,
            "evaluate": TRAIN,
        },
    )
    assert _run_ids(wider)["applied"] == _run_ids(run)["applied"]
    assert package.trainings == 1


def test_widening_the_scope_of_the_fitting_step_is_a_new_run_that_trains_again(
    tmp_path: Path, package: LightningActionRecorder
) -> None:
    """A fit reads its own stream, so its scope is part of its identity."""
    ds = dataset(tmp_path / "ds")
    narrow = run_pipeline(
        ds,
        recipe(test_fraction=0.0),
        step_scopes={"labeled": TRAIN, "classifier": TRAIN, "evaluate": TRAIN},
    )
    wide = run_pipeline(
        ds,
        recipe(test_fraction=0.0),
        step_scopes={"labeled": TRAIN, "classifier": BOTH, "evaluate": TEST},
    )

    assert _run_ids(narrow)["classifier"] != _run_ids(wide)["classifier"]
    assert package.trainings == 2
