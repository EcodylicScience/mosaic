"""The state of every fit a row read, as ``consumed_model_state`` records it.

A row records one part per fit reference of its run, keyed by its params
field, and a part ``_state`` for the run's own state when the root it loaded
from holds ``fit_scope.json``. One part is a bare digest and several are
sorted ``field=digest`` pairs.

The k-means chain is the built-in clustering: ``speed-angvel`` over the
tracks, ``extract-templates`` over it, ``global-kmeans`` fitted on those
templates, and a second ``global-kmeans`` step that applies the fit through
``model``. The fit is scope-free and seeded: fitting the same templates again
saves the same state.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import new_execution_id
from mosaic.core.pipeline.fit_scope import (
    FIT_SCOPE_NAME,
    FitStateUnreadable,
    list_run_root,
    read_fit_scope,
    read_fit_state_digest,
    saved_state_digest,
    write_fit_scope,
)
from mosaic.core.pipeline.index import (
    feature_index_path,
    feature_run_root,
    fit_state_cell,
)
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.run_log import read_run, run_log_dir
from mosaic.core.pipeline.types import Result
from mosaic.core.scope import Scope
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    FITTED_AGAIN,
    KMeansChain,
    KMeansFit,
    add_tracks_variant,
    calms21_dataset,
    drifted,
    kmeans_apply,
    kmeans_apply_feature,
    kmeans_fit,
    kmeans_fit_and_apply,
    make_dataset,
    model_cells,
    model_reference,
    move_tracks,
    refit_kmeans,
    refusals,
    result_run_root,
    run_step,
    tree_snapshot,
)

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue
    from mosaic.core.pipeline.graph import Recipe

VARIANT: Final = "convert-trex.0.2-1111111111"
SEQUENCES: Final = ("seq_a", "seq_b")
KMEANS_NAME: Final = "global-kmeans"
TEMPLATES_NAME: Final = "extract-templates"

KMEANS_STATE: Final = frozenset(
    {
        "model.joblib",
        "cluster_centers.npz",
        "artifact_labels.npz",
        "cluster_sizes.parquet",
    }
)
"""The files ``global-kmeans``'s ``save_state`` writes."""

WORKERS: Final = 2
MODE: Final = "thread"

NOTHING_RECORDED: Final = "nothing records which fits it read"
NO_LONGER_RECORDED: Final = "a fit it reads no longer records a fit-state digest"


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two track tables of one variant, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def _digest(ds: Dataset, result: Result[str]) -> str:
    """The fit-state digest the run of *result* records, which must be one."""
    digest = read_fit_state_digest(result_run_root(ds, result))
    assert digest
    return digest


def _set_model_cells(ds: Dataset, result: Result[str], cell: str) -> None:
    """Set the ``consumed_model_state`` of every row of *result* to *cell*."""
    path = feature_index_path(ds, result.feature)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows.loc[rows["run_id"] == result.run_id, "consumed_model_state"] = cell
    rows.to_csv(path, index=False)


def _outputs(ds: Dataset, result: Result[str]) -> dict[str, tuple[int, int, int]]:
    """The inode, size and ``mtime_ns`` of each entry output of *result*."""
    return {
        path: stamp
        for path, stamp in list_run_root(result_run_root(ds, result)).items()
        if path.endswith(".parquet")
    }


def _rewrite_fit_scope(root: Path, edit: Callable[[dict[str, object]], str]) -> None:
    """Replace the ``fit_scope.json`` of *root* by *edit* of its record."""
    path = root / FIT_SCOPE_NAME
    record: dict[str, object] = json.loads(path.read_text())
    _ = path.write_text(edit(record))


def _without_digest(record: dict[str, object]) -> str:
    """*record* without its ``state_digest``, as written before the digest."""
    del record["state_digest"]
    return json.dumps(record)


def _blank_states_read(ds: Dataset, result: Result[str]) -> None:
    """Blank the digest of each fit that *result*'s fit read, in its record.

    A record written before those digests holds none. Its fit reads as
    unchanged whatever the fits it read record now.
    """
    root = result_run_root(ds, result)
    record = read_fit_scope(root, strict=True)
    assert record is not None
    via = tuple(replace(read, state_digest="") for read in record.via)
    write_fit_scope(root, replace(record, via=via))


def _apply_capturing(
    ds: Dataset,
    chain: KMeansFit,
    capsys: pytest.CaptureFixture[str],
    *,
    mode: str = MODE,
) -> str:
    """Apply the fit of *chain* again, and what that run printed to stderr."""
    _ = capsys.readouterr()
    applied = kmeans_apply(ds, chain, workers=WORKERS, mode=mode)
    if isinstance(chain, KMeansChain):
        assert applied.run_id == chain.apply.run_id
    return capsys.readouterr().err


def _templates_capturing(
    ds: Dataset, runs: KMeansFit, capsys: pytest.CaptureFixture[str]
) -> str:
    """Run the ``extract-templates`` step of *runs* again, and its stderr."""
    _ = capsys.readouterr()
    again = run_step(
        ds,
        TEMPLATES_NAME,
        [model_reference(runs.speed)],
        {"n_templates": 20},
        workers=WORKERS,
        mode=MODE,
    )
    assert again.run_id == runs.templates.run_id
    return capsys.readouterr().err


def _fit_capturing(ds: Dataset, capsys: pytest.CaptureFixture[str]) -> str:
    """Run the k-means fit chain again, and what it printed to stderr."""
    _ = capsys.readouterr()
    _ = kmeans_fit(ds, workers=WORKERS, mode=MODE)
    return capsys.readouterr().err


# --- a fit records the digest of its state ---------------------------------


def test_a_fit_records_the_digest_of_the_state_it_saved(tracked: Dataset) -> None:
    """The digest covers the k-means state files, and no entry output."""
    chain = kmeans_fit_and_apply(tracked, workers=2, mode="thread")

    root = result_run_root(tracked, chain.fit)
    listing = list_run_root(root)
    saved = {path: stamp for path, stamp in listing.items() if path in KMEANS_STATE}
    assert set(saved) == KMEANS_STATE
    assert (root / FIT_SCOPE_NAME).is_file()
    recorded = read_fit_state_digest(root)
    assert recorded
    assert recorded == saved_state_digest(root, {}, saved)


def test_a_fits_own_rows_record_its_digest_and_a_second_run_serves_them(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fit that names no other fit records its own state as a bare cell."""
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    own = _digest(tracked, runs.templates)
    assert fit_state_cell({"_state": own}) == own
    cells = model_cells(tracked, runs.templates)
    assert cells == {"seq_a": own, "seq_b": own}
    outputs = _outputs(tracked, runs.templates)

    err = _templates_capturing(tracked, runs, capsys)

    assert refusals(err, TEMPLATES_NAME) == {}
    assert _outputs(tracked, runs.templates) == outputs
    assert model_cells(tracked, runs.templates) == cells


def test_a_run_that_reads_no_fit_records_blank_cells(tracked: Dataset) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)

    cells = model_cells(tracked, runs.speed)

    assert cells
    assert set(cells.values()) == {""}


# --- an apply through model alone ------------------------------------------


def test_an_apply_records_the_digest_of_its_fit_and_serves_after(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    """One part, the ``model``: the cell is the bare digest of the fit."""
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    digest = _digest(tracked, chain.fit)
    assert model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    outputs = _outputs(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys)

    assert refusals(err, KMEANS_NAME) == {}
    assert _outputs(tracked, chain.apply) == outputs
    assert model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}


def test_an_apply_recomputes_after_its_fit_is_written_again(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    move_tracks(tracked, VARIANT, SEQUENCES)
    speed = run_step(tracked, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    assert speed.run_id == chain.speed.run_id
    # The apply first recomputes for its moved input, which is compared before
    # the model, and records the digest of the fit not yet written again.
    _ = _apply_capturing(tracked, chain, capsys)
    digest = refit_kmeans(tracked, chain, workers=WORKERS, mode=MODE)
    outputs = _outputs(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys)

    assert refusals(err, KMEANS_NAME) == {"seq_a": FITTED_AGAIN, "seq_b": FITTED_AGAIN}
    recomputed = _outputs(tracked, chain.apply)
    assert all(recomputed[path] != stamp for path, stamp in outputs.items())
    assert model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    assert refusals(_apply_capturing(tracked, chain, capsys), KMEANS_NAME) == {}
    assert _outputs(tracked, chain.apply) == recomputed


# --- a fit through templates ------------------------------------------------


def test_a_fit_on_templates_records_their_digest_and_recomputes_once_they_are_written_again(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    templates = _digest(tracked, runs.templates)
    cell = fit_state_cell(
        {"_state": _digest(tracked, runs.fit), "templates": templates}
    )
    assert cell == f"_state={_digest(tracked, runs.fit)},templates={templates}"
    assert model_cells(tracked, runs.fit) == {"seq_a": cell, "seq_b": cell}
    move_tracks(tracked, VARIANT, SEQUENCES)

    # The templates are fitted again over their moved entries, and the fit
    # again over the new templates. A fit run again computes every entry.
    err = _fit_capturing(tracked, capsys)

    again = _digest(tracked, runs.templates)
    assert again != templates
    assert refusals(err, KMEANS_NAME) == {}
    now = fit_state_cell({"_state": _digest(tracked, runs.fit), "templates": again})
    assert model_cells(tracked, runs.fit) == {"seq_a": now, "seq_b": now}
    assert refusals(_fit_capturing(tracked, capsys), KMEANS_NAME) == {}


# --- evaluate-labels through label_map ---------------------------------------


EVALUATE_SCOPES: Final = {
    "templates": Scope(groups=["train"]),
    "map": Scope(groups=["train"]),
    "score": Scope(groups=["test"]),
}


def _evaluate_recipe() -> Recipe:
    """Clusters mapped to labels on the train group, scored on the test."""
    from mosaic.core.pipeline.graph import Recipe

    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "held-out-map",
            "steps": [
                CONVERT_TRACKS_STEP,
                CONVERT_LABELS_STEP,
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "tracks": {"step": "tracks"},
                },
                {
                    "id": "templates",
                    "type": "feature",
                    "feature": "extract-templates",
                    "inputs": [{"step": "speed"}],
                    "params": {"n_templates": 40},
                },
                {
                    "id": "kmeans",
                    "type": "feature",
                    "feature": "global-kmeans",
                    "inputs": [{"step": "speed"}],
                    "params": {"templates": {"step": "templates"}, "k": 3},
                },
                {
                    "id": "map",
                    "type": "feature",
                    "feature": "fit-label-map",
                    "inputs": [{"step": "kmeans"}],
                    "params": {"labels": {"step": "labels"}},
                },
                {
                    "id": "score",
                    "type": "feature",
                    "feature": "evaluate-labels",
                    "inputs": [{"step": "kmeans"}],
                    "params": {
                        "labels": {"step": "labels"},
                        "mode": "clusters",
                        "prediction_column": "cluster",
                        "label_map": {"step": "map"},
                    },
                },
            ],
        }
    )


def _step_result(
    ds: Dataset, recipe: Recipe, scopes: Mapping[str, Scope], step: str
) -> Result[str]:
    """The run that *step* of *recipe* plans under *scopes*."""
    from mosaic.core.pipeline.graph import plan_pipeline

    planned = plan_pipeline(ds, recipe, step_scopes=dict(scopes)).step(step)
    assert planned.run_id is not None
    return Result(feature=planned.storage_name, run_id=planned.run_id)


def _pipeline_capturing(
    ds: Dataset,
    recipe: Recipe,
    scopes: Mapping[str, Scope],
    capsys: pytest.CaptureFixture[str],
) -> str:
    """Run *recipe* over *ds*, and what the run printed to stderr."""
    from mosaic.core.pipeline.graph import run_pipeline

    _ = capsys.readouterr()
    _ = run_pipeline(ds, recipe, step_scopes=dict(scopes))
    return capsys.readouterr().err


def _shift_labels(ds: Dataset, group: str) -> None:
    """Write each converted label file of *group* again, its classes moved.

    The files are the ones this test wrote through its own conversion.
    """
    for path in sorted(ds.get_root("labels").rglob(f"{group}__*.npz")):
        with np.load(path, allow_pickle=True) as stored:
            arrays = {name: stored[name] for name in stored.files}
        arrays["labels"] = (arrays["labels"] + 1) % 3
        with path.open("wb") as handle:
            np.savez(handle, **arrays)


def test_an_evaluation_records_its_label_map_and_its_own_state_and_recomputes_after_the_map_is_fitted_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 200, "s2": 200}}, "test": {"a": {"s3": 200}}},
        seed=3,
        n_classes=3,
    )
    recipe = _evaluate_recipe()
    _ = _pipeline_capturing(ds, recipe, EVALUATE_SCOPES, capsys)
    label_map = _step_result(ds, recipe, EVALUATE_SCOPES, "map")
    score = _step_result(ds, recipe, EVALUATE_SCOPES, "score")
    mapped = _digest(ds, label_map)
    own = _digest(ds, score)
    assert model_cells(ds, score) == {"s3": f"_state={own},label_map={mapped}"}

    shutil.rmtree(result_run_root(ds, label_map))
    _shift_labels(ds, "train")
    err = _pipeline_capturing(ds, recipe, EVALUATE_SCOPES, capsys)

    again = _digest(ds, label_map)
    assert again != mapped
    # The evaluation reads another map, scores again and computes s3.
    assert refusals(err, "evaluate-labels") == {}
    scored = _digest(ds, score)
    assert model_cells(ds, score) == {"s3": f"_state={scored},label_map={again}"}
    err = _pipeline_capturing(ds, recipe, EVALUATE_SCOPES, capsys)
    assert refusals(err, "evaluate-labels") == {}


# --- sequence-classifier through init_from -----------------------------------


TRAIN: Final = Scope(groups=["train"])
TEST: Final = Scope(groups=["test"])
WARM_SCOPES: Final = {
    "labeled": TRAIN,
    "classifier": TRAIN,
    "tune_labeled": TEST,
    "finetune": TEST,
}
NETWORK: Final[dict[str, JsonValue]] = {
    "labels": {"step": "labels"},
    "default_class": 0,
    "device": "cpu",
    "head": "tcn",
    "head_config": {"hidden": 8, "blocks": 2},
    "batch_size": 4,
    "crop_length": 60,
    "learning_rate": 0.01,
}


def _warm_recipe() -> Recipe:
    """A classifier on the train group, and one fine-tuned from it."""
    from mosaic.core.pipeline.graph import Recipe

    def labeled(step: str, n_total: int) -> dict[str, JsonValue]:
        return {
            "id": step,
            "type": "feature",
            "feature": "extract-labeled-templates",
            "inputs": [{"step": "pairs"}],
            "params": {
                "labels": {"step": "labels"},
                "n_total": n_total,
                "test_fraction": 0.0,
            },
        }

    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "warm-start",
            "steps": [
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
                labeled("labeled", 300),
                {
                    "id": "classifier",
                    "type": "feature",
                    "feature": "sequence-classifier",
                    "inputs": [{"step": "pairs"}],
                    "params": {
                        **NETWORK,
                        "templates": {"step": "labeled"},
                        "steps": 20,
                    },
                },
                labeled("tune_labeled", 100),
                {
                    "id": "finetune",
                    "type": "feature",
                    "feature": "sequence-classifier",
                    "inputs": [{"step": "pairs"}],
                    "params": {
                        **NETWORK,
                        "templates": {"step": "tune_labeled"},
                        "init_from": {"step": "classifier"},
                        "steps": 20,
                    },
                },
            ],
        }
    )


def _new_digest(record: dict[str, object]) -> str:
    """*record* with another ``state_digest``, as a fit written again holds."""
    record["state_digest"] = "0" * 16
    return json.dumps(record)


@pytest.mark.identity
def test_a_warm_started_fit_records_the_fit_it_started_from_and_recomputes_after_it_is_written_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _ = pytest.importorskip("torch")
    ds = calms21_dataset(
        tmp_path / "ds",
        {
            "train": {"a": {"s1": 240, "s2": 240}},
            "test": {"a": {"t1": 240, "t2": 240}},
        },
        seed=11,
        n_classes=3,
    )
    recipe = _warm_recipe()
    _ = _pipeline_capturing(ds, recipe, WARM_SCOPES, capsys)
    pretrained = _step_result(ds, recipe, WARM_SCOPES, "classifier")
    templates = _step_result(ds, recipe, WARM_SCOPES, "tune_labeled")
    finetune = _step_result(ds, recipe, WARM_SCOPES, "finetune")
    own = _digest(ds, finetune)
    cell = fit_state_cell(
        {
            "_state": own,
            "init_from": _digest(ds, pretrained),
            "templates": _digest(ds, templates),
        }
    )
    assert f"init_from={_digest(ds, pretrained)}" in cell.split(",")
    assert model_cells(ds, finetune) == {"t1": cell, "t2": cell}

    _rewrite_fit_scope(result_run_root(ds, pretrained), _new_digest)
    err = _pipeline_capturing(ds, recipe, WARM_SCOPES, capsys)

    # The pretrained fit's own rows read its state as their own part. The
    # fine-tuning started from it is fitted again and computes its entries.
    assert refusals(err, "sequence-classifier") == {
        "s1": FITTED_AGAIN,
        "s2": FITTED_AGAIN,
    }
    now = fit_state_cell(
        {
            "_state": _digest(ds, finetune),
            "init_from": "0" * 16,
            "templates": _digest(ds, templates),
        }
    )
    assert model_cells(ds, finetune) == {"t1": now, "t2": now}


# --- smooth-probabilities through decision_threshold -------------------------


SMOOTH_ENTRIES: Final = ("s1", "s2", "s3", "s4")
SMOOTH_SCOPED: Final = ("labeled", "clf", "smooth", "tune", "apply")


def _smooth_recipe() -> Recipe:
    """A classifier smoothed, its thresholds tuned, and then applied."""
    from mosaic.core.pipeline.graph import Recipe

    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "thresholds",
            "steps": [
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
                        "cv_folds": 2,
                    },
                },
                {
                    "id": "smooth",
                    "type": "feature",
                    "feature": "smooth-probabilities",
                    "inputs": [{"step": "clf"}],
                    "params": {"default_class": 0},
                },
                {
                    "id": "tune",
                    "type": "feature",
                    "feature": "tune-threshold",
                    "inputs": [{"step": "smooth"}],
                    "params": {
                        "labels": {"step": "labels"},
                        "default_class": 0,
                        "average_over": [0, 1, 2],
                    },
                },
                {
                    "id": "apply",
                    "type": "feature",
                    "feature": "smooth-probabilities",
                    "inputs": [{"step": "smooth"}],
                    "params": {
                        "method": "none",
                        "default_class": 0,
                        "decision_threshold": {"step": "tune"},
                    },
                },
            ],
        }
    )


def test_an_apply_of_tuned_thresholds_records_field_pairs_and_a_bare_row_recomputes_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": dict.fromkeys(SMOOTH_ENTRIES, 240)}},
        seed=11,
        n_classes=3,
    )
    recipe = _smooth_recipe()
    scopes: dict[str, Scope] = dict.fromkeys(SMOOTH_SCOPED, TRAIN)
    _ = _pipeline_capturing(ds, recipe, scopes, capsys)
    tune = _step_result(ds, recipe, scopes, "tune")
    applied = _step_result(ds, recipe, scopes, "apply")
    thresholds = _digest(ds, tune)
    cell = f"_state={_digest(ds, applied)},decision_threshold={thresholds}"
    assert cell == fit_state_cell(
        {"decision_threshold": thresholds, "_state": _digest(ds, applied)}
    )
    assert model_cells(ds, applied) == dict.fromkeys(SMOOTH_ENTRIES, cell)

    # A row holding one bare digest, the form a cell of one part takes.
    _set_model_cells(ds, applied, thresholds)
    err = _pipeline_capturing(ds, recipe, scopes, capsys)

    assert refusals(err, "smooth-probabilities") == dict.fromkeys(
        SMOOTH_ENTRIES, FITTED_AGAIN
    )
    assert model_cells(ds, applied) == dict.fromkeys(SMOOTH_ENTRIES, cell)
    err = _pipeline_capturing(ds, recipe, scopes, capsys)
    assert refusals(err, "smooth-probabilities") == {}


# --- rows and fits written before the cell -----------------------------------


def test_a_fits_row_written_before_its_own_part_recomputes_once(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fit that records a digest, and rows that record none, as before."""
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    own = _digest(tracked, runs.templates)
    _set_model_cells(tracked, runs.templates, "")

    err = _templates_capturing(tracked, runs, capsys)

    assert refusals(err, TEMPLATES_NAME) == {
        "seq_a": NOTHING_RECORDED,
        "seq_b": NOTHING_RECORDED,
    }
    assert model_cells(tracked, runs.templates) == {"seq_a": own, "seq_b": own}
    assert refusals(_templates_capturing(tracked, runs, capsys), TEMPLATES_NAME) == {}


def test_a_fit_and_rows_that_both_predate_the_digest_are_served(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    _rewrite_fit_scope(result_run_root(tracked, runs.templates), _without_digest)
    _set_model_cells(tracked, runs.templates, "")
    outputs = _outputs(tracked, runs.templates)

    err = _templates_capturing(tracked, runs, capsys)

    assert refusals(err, TEMPLATES_NAME) == {}
    assert _outputs(tracked, runs.templates) == outputs
    assert model_cells(tracked, runs.templates) == {"seq_a": "", "seq_b": ""}


def test_an_apply_row_without_a_digest_recomputes_once(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    digest = _digest(tracked, chain.fit)
    _set_model_cells(tracked, chain.apply, "")

    err = _apply_capturing(tracked, chain, capsys)

    assert refusals(err, KMEANS_NAME) == {
        "seq_a": NOTHING_RECORDED,
        "seq_b": NOTHING_RECORDED,
    }
    assert model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    assert refusals(_apply_capturing(tracked, chain, capsys), KMEANS_NAME) == {}


def test_an_apply_of_a_fit_that_no_longer_records_a_digest_recomputes(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    _rewrite_fit_scope(result_run_root(tracked, chain.fit), _without_digest)

    err = _apply_capturing(tracked, chain, capsys)

    assert refusals(err, KMEANS_NAME) == {
        "seq_a": NO_LONGER_RECORDED,
        "seq_b": NO_LONGER_RECORDED,
    }
    assert model_cells(tracked, chain.apply) == {"seq_a": "", "seq_b": ""}


def test_a_fit_without_a_digest_is_served_with_one_warning_naming_it(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    _rewrite_fit_scope(result_run_root(tracked, chain.fit), _without_digest)
    _set_model_cells(tracked, chain.apply, "")
    outputs = _outputs(tracked, chain.apply)

    err = _apply_capturing(tracked, chain, capsys)

    assert refusals(err, KMEANS_NAME) == {}
    assert _outputs(tracked, chain.apply) == outputs
    assert model_cells(tracked, chain.apply) == {"seq_a": "", "seq_b": ""}
    warning = (
        "[feature:global-kmeans] 2 entries read fits that record no fit-state "
        f"digest (model ({chain.fit.feature} {chain.fit.run_id})): a fit "
        "written again under its run id goes undetected. Remove each run and "
        "fit it again to record one"
    )
    assert err.splitlines().count(warning) == 1


def test_the_warning_names_each_blank_fit_and_no_fit_that_records_a_digest(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fit's own blank part is not named: it reads no other run."""
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    _blank_states_read(tracked, runs.fit)
    for result in (runs.templates, runs.fit):
        _rewrite_fit_scope(result_run_root(tracked, result), _without_digest)
        _set_model_cells(tracked, result, "")

    err = _fit_capturing(tracked, capsys)

    assert refusals(err, KMEANS_NAME) == {}
    named = f"(templates ({runs.templates.feature} {runs.templates.run_id}))"
    warnings = [line for line in err.splitlines() if "record no fit-state" in line]
    assert len(warnings) == 1
    assert warnings[0].startswith("[feature:global-kmeans] 2 entries read fits")
    assert named in warnings[0]


# --- an unreadable record ----------------------------------------------------


def _unparsable(_record: dict[str, object]) -> str:
    """A ``fit_scope.json`` that is not JSON."""
    return "{not json"


def _numeric_digest(record: dict[str, object]) -> str:
    """*record* with a ``state_digest`` that is not a string."""
    record["state_digest"] = 3
    return json.dumps(record)


@pytest.mark.parametrize("edit", [_unparsable, _numeric_digest])
def test_an_unreadable_fit_digest_fails_the_apply_before_any_entry(
    tracked: Dataset, edit: Callable[[dict[str, object]], str]
) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    fit = runs.fit
    assert fit.run_id is not None
    _rewrite_fit_scope(result_run_root(tracked, fit), edit)
    features_root = tracked.get_root("features")
    before = tree_snapshot(features_root)
    feature = kmeans_apply_feature(runs)
    execution_id = new_execution_id()

    with pytest.raises(FitStateUnreadable) as caught:
        _ = run_feature(
            tracked,
            feature,
            parallel_workers=WORKERS,
            parallel_mode=MODE,
            execution_id=execution_id,
        )

    assert caught.value.error_detail() == {"feature": fit.feature, "run_id": fit.run_id}
    assert tree_snapshot(features_root) == before
    logged = read_run(run_log_dir(tracked.base_dir), execution_id)
    assert logged is not None
    assert logged["status"] == "failed"


@pytest.mark.parametrize("edit", [_unparsable, _numeric_digest])
def test_an_unreadable_record_of_the_runs_own_state_fails_the_run(
    tracked: Dataset, edit: Callable[[dict[str, object]], str]
) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    root = result_run_root(tracked, runs.templates)
    _rewrite_fit_scope(root, edit)

    with pytest.raises(FitStateUnreadable) as caught:
        _ = run_step(
            tracked,
            TEMPLATES_NAME,
            [model_reference(runs.speed)],
            {"n_templates": 20},
            workers=WORKERS,
            mode=MODE,
        )

    assert caught.value.error_detail() == {
        "feature": runs.templates.feature,
        "run_id": runs.templates.run_id,
    }
    assert root == feature_run_root(
        tracked, runs.templates.feature, str(runs.templates.run_id)
    )


# --- the plan ---------------------------------------------------------------


def test_a_plan_names_every_apply_entry_after_the_refit_and_none_after_the_apply(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    move_tracks(tracked, VARIANT, SEQUENCES)
    speed = run_step(tracked, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    assert speed.run_id == chain.speed.run_id
    _ = _apply_capturing(tracked, chain, capsys)
    assert drifted(tracked, chain.apply) == ()
    _ = refit_kmeans(tracked, chain, workers=WORKERS, mode=MODE)

    assert drifted(tracked, chain.apply) == (("", "seq_a"), ("", "seq_b"))

    _ = _apply_capturing(tracked, chain, capsys)

    assert drifted(tracked, chain.apply) == ()


def test_a_plan_names_a_fits_entries_after_the_templates_are_written_again(
    tracked: Dataset,
) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    assert drifted(tracked, runs.templates) == ()
    assert drifted(tracked, runs.fit) == ()
    _rewrite_fit_scope(result_run_root(tracked, runs.templates), _new_digest)

    assert drifted(tracked, runs.templates) == (("", "seq_a"), ("", "seq_b"))
    assert drifted(tracked, runs.fit) == (("", "seq_a"), ("", "seq_b"))
