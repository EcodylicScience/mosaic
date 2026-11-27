"""A fit's inputs, and the fit state a feature input carries down a chain.

The chain is the k-means fit, its apply through ``model``, a per-entry feature
over the apply's output and a second one over that. A fit written again moves
the apply's ``consumed_model_state``, which the feature reading the apply
records in its inputs cell, and the one below it in its own. Each recomputes
in turn.

A fit records the cells of each entry it read, the attempt that fitted and the
state of each fit it read, in its ``fit_scope.json``. Its drift compares that
record with what the dataset holds now.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from mosaic.core.dataset import Dataset, LinkUnreachableError
from mosaic.core.entry import Entry
from mosaic.core.pipeline.composition import input_part
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.fit_scope import (
    FIT_SCOPE_NAME,
    CannotFitAgain,
    FitEntryInputs,
    FitScope,
    FitSourceRef,
    FitStateUnreadable,
    fit_inputs_digest,
    read_fit_scope,
    read_fit_state_digest,
    write_fit_scope,
)
from mosaic.core.pipeline.index import (
    OWN_STATE_PART,
    DriftReads,
    EntryCells,
    FitDrift,
    current_entry_cells,
    feature_index,
    feature_index_path,
    fit_drift,
    fit_state_cell,
    refit_summary,
)
from mosaic.core.pipeline.inventory.params import (
    read_linked_run_params,
    read_run_params,
)
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Feature,
    Inputs,
    InputStream,
    Result,
)
from mosaic.core.scope import Scope
from mosaic.runlog import read_run_progress, run_log_dir
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    KMEANS,
    FitControl,
    KMeansChain,
    KMeansFit,
    PlainFeature,
    ScopedFittingFeature,
    add_tracks_table,
    add_tracks_variant,
    assert_tree_unchanged,
    calms21_dataset,
    copy_of,
    drifted,
    index_cells,
    kmeans_apply,
    kmeans_apply_feature,
    kmeans_fit,
    kmeans_fit_and_apply,
    make_dataset,
    model_cells,
    model_reference,
    move_tracks,
    reconvert_entry,
    refit_kmeans,
    refusals,
    result_run_root,
    run_step,
    scored_classifier_recipe,
    take_link_offline,
    track_table,
    tree_snapshot,
)

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue
    from mosaic.core.pipeline.graph import Recipe
    from mosaic.core.pipeline.graph.resolve import StepSpec

VARIANT: Final = "convert-trex.0.2-1111111111"
SEQUENCES: Final = ("seq_a", "seq_b")
BOTH: Final = (("", "seq_a"), ("", "seq_b"))
WORKERS: Final = 2
MODE: Final = "thread"


class _BelowApply(PlainFeature):
    """A per-entry feature over the output of the apply."""

    name = "fit-below-apply"
    accepts_overlap = False
    emits: EmitsLevel = "as-input"


class _BelowThat(_BelowApply):
    """A per-entry feature over the output of :class:`_BelowApply`."""

    name = "fit-below-that"


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two track tables of one variant, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, *SEQUENCES, std_format="mosaic_v1")
    return dataset


def _read(upstream: Result[str]) -> Inputs:
    """The inputs of a feature reading the run of *upstream*."""
    return Inputs((Result(feature=upstream.feature, run_id=upstream.run_id),))


def _below(
    ds: Dataset, chain: KMeansChain, capsys: pytest.CaptureFixture[str]
) -> tuple[Result[str], Result[str], str]:
    """Run both features below the apply, and what they printed to stderr."""
    _ = capsys.readouterr()
    first = run_feature(ds, _BelowApply(_read(chain.apply)), track=False)
    second = run_feature(ds, _BelowThat(_read(first)), track=False)
    return first, second, capsys.readouterr().err


def _settled(
    ds: Dataset, capsys: pytest.CaptureFixture[str]
) -> tuple[KMeansChain, Result[str], Result[str]]:
    """The chain over tracks moved once, every run recomputed and served again."""
    chain = kmeans_fit_and_apply(ds, workers=WORKERS, mode=MODE)
    move_tracks(ds, VARIANT, SEQUENCES)
    speed = run_step(ds, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    assert speed.run_id == chain.speed.run_id
    _ = kmeans_apply(ds, chain, workers=WORKERS, mode=MODE)
    first, second, _ = _below(ds, chain, capsys)
    assert drifted(ds, chain.apply) == ()
    assert drifted(ds, first) == ()
    assert drifted(ds, second) == ()
    return chain, first, second


def test_a_feature_input_records_the_state_of_the_fit_its_upstream_read(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    digest = model_cells(tracked, chain.apply)["seq_a"]
    assert digest

    first, second, _ = _below(tracked, chain, capsys)

    part = input_part("", "", digest, "")
    assert part == f"{digest}|/"
    assert index_cells(tracked, first, "consumed_input_composition") == {
        "seq_a": part,
        "seq_b": part,
    }
    deeper = input_part("", part, "", "")
    assert index_cells(tracked, second, "consumed_input_composition") == {
        "seq_a": deeper,
        "seq_b": deeper,
    }
    assert deeper == f"/{digest}|/"


def test_a_fit_written_again_recomputes_every_row_below_it_in_turn(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain, first, second = _settled(tracked, capsys)

    digest = refit_kmeans(tracked, chain, workers=WORKERS, mode=MODE)

    assert drifted(tracked, chain.apply) == BOTH
    assert drifted(tracked, first) == ()
    assert drifted(tracked, second) == ()

    _ = kmeans_apply(tracked, chain, workers=WORKERS, mode=MODE)
    assert model_cells(tracked, chain.apply) == {"seq_a": digest, "seq_b": digest}
    assert drifted(tracked, chain.apply) == ()
    assert drifted(tracked, first) == BOTH
    assert drifted(tracked, second) == ()

    _ = capsys.readouterr()
    first_again = run_feature(tracked, _BelowApply(_read(chain.apply)), track=False)
    err = capsys.readouterr().err
    assert refusals(err, _BelowApply.name).keys() == set(SEQUENCES)
    assert first_again.run_id == first.run_id
    assert drifted(tracked, first) == ()
    assert drifted(tracked, second) == BOTH

    _ = capsys.readouterr()
    _ = run_feature(tracked, _BelowThat(_read(first)), track=False)
    err = capsys.readouterr().err
    assert refusals(err, _BelowThat.name).keys() == set(SEQUENCES)
    assert drifted(tracked, second) == ()
    part = input_part("", "", digest, "")
    assert index_cells(tracked, first, "consumed_input_composition") == {
        "seq_a": part,
        "seq_b": part,
    }


def test_a_chain_whose_upstream_rows_record_no_state_keeps_every_cells_form(
    tracked: Dataset,
) -> None:
    speed = run_step(tracked, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    first = run_feature(tracked, _BelowApply(_read(speed)), track=False)
    second = run_feature(tracked, _BelowThat(_read(first)), track=False)

    assert set(model_cells(tracked, speed).values()) == {""}
    for result in (first, second):
        cells = index_cells(tracked, result, "consumed_input_composition")
        assert set(cells) == set(SEQUENCES)
        assert all("|" not in cell for cell in cells.values())
        assert set(cells.values()) == {""}
        assert drifted(tracked, result) == ()


# --- the record a fit writes ------------------------------------------------


EXECUTION: Final = "01JD8M3Q6W2F5Y0RKX7T4N9BCA"
TRAIN: Final = Scope(groups=["train"])
TEST: Final = Scope(groups=["test"])
WIDER: Final = Scope(groups=["train", "test"])
TRAIN_ENTRIES: Final = (("train", "s1"), ("train", "s2"))


def _record(ds: Dataset, result: Result[str]) -> FitScope:
    """The fit record of *result*'s run, which must hold one."""
    fitted = read_fit_scope(result_run_root(ds, result), strict=True)
    assert fitted is not None
    return fitted


def _cells_now(
    ds: Dataset, result: Result[str], entries: Iterable[Entry]
) -> tuple[FitEntryInputs, ...]:
    """The cells each of *entries* holds now for the run of *result*."""
    params = read_run_params(result_run_root(ds, result)).params
    assert params is not None
    wanted = sorted(entries)
    cells, held = current_entry_cells(DriftReads(ds), params, wanted)
    assert held.issuperset(wanted)
    return tuple(
        FitEntryInputs(group=group, sequence=sequence, cells=cells[(group, sequence)])
        for group, sequence in wanted
    )


def _recorded_entries(fitted: FitScope) -> list[Entry]:
    """The entries whose cells *fitted* records."""
    return [(item.group, item.sequence) for item in fitted.inputs or ()]


def _state_read(ref: FitSourceRef) -> tuple[str, str, str, str]:
    """The field, the run and the state digest a ``via`` item names."""
    return (ref.field, ref.feature, ref.run_id, ref.state_digest)


def _fit_state(ds: Dataset, result: Result[str]) -> tuple[str, str, str]:
    """The feature, the run and the recorded state digest of *result*'s fit."""
    digest = read_fit_state_digest(result_run_root(ds, result))
    assert digest
    assert result.run_id is not None
    return (result.feature, result.run_id, digest)


def _templates(ds: Dataset, speed: Result[str], scope: Scope | None) -> Result[str]:
    """Fit ``extract-templates`` over the ``speed-angvel`` run *speed*."""
    from mosaic.core.pipeline.graph import build_feature

    feature = build_feature(
        "extract-templates", [model_reference(speed)], {"n_templates": 20}
    )
    return run_feature(ds, feature, scope=scope, execution_id=EXECUTION, track=False)


def test_a_templates_fit_records_its_entries_their_cells_and_its_attempt(
    tracked: Dataset,
) -> None:
    move_tracks(tracked, VARIANT, SEQUENCES)
    speed = run_step(tracked, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)

    templates = _templates(tracked, speed, None)

    fitted = _record(tracked, templates)
    assert fitted.inputs == _cells_now(tracked, templates, BOTH)
    assert all(item.cells.tracks for item in fitted.inputs or ())
    assert fitted.execution_id == EXECUTION
    assert fitted.via == ()


def test_a_kmeans_fit_records_no_entry_and_the_state_of_its_templates(
    tracked: Dataset,
) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)

    fitted = _record(tracked, runs.fit)
    assert fitted.inputs == ()
    assert [_state_read(ref) for ref in fitted.via] == [
        ("templates", *_fit_state(tracked, runs.templates))
    ]


def test_the_records_bytes_do_not_depend_on_the_manifests_order(
    tmp_path: Path,
) -> None:
    texts: list[str] = []
    for name, order in (("ab", SEQUENCES), ("ba", SEQUENCES[::-1])):
        ds = make_dataset(tmp_path / name)
        add_tracks_variant(ds, VARIANT, *order, std_format="mosaic_v1")
        speed = run_step(ds, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
        entries = [("", sequence) for sequence in order]
        templates = _templates(ds, speed, Scope(entries=entries))
        record = result_run_root(ds, templates) / FIT_SCOPE_NAME
        fitted = _record(ds, templates)
        assert len(fitted.inputs or ()) == len(SEQUENCES)
        texts.append(record.read_text().replace(fitted.fitted_at, "<fitted_at>"))

    assert texts[0] == texts[1]


def _labeled_steps() -> list[dict[str, JsonValue]]:
    """The conversions, pair positions and labeled templates of the train group."""
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
    ]


def _recipe(name: str, steps: list[dict[str, JsonValue]]) -> Recipe:
    """A recipe of *steps*."""
    from mosaic.core.pipeline.graph import Recipe

    return Recipe.model_validate({"schema_version": 1, "name": name, "steps": steps})


def _labeled_dataset(base: Path) -> Dataset:
    """Two labeled train sequences and one test sequence."""
    return calms21_dataset(
        base,
        {"train": {"a": {"s1": 240, "s2": 240}}, "test": {"a": {"t1": 240}}},
        seed=11,
        n_classes=3,
    )


def _run_recipe(ds: Dataset, recipe: Recipe, scopes: Mapping[str, Scope]) -> None:
    """Run *recipe* over *ds* under *scopes*."""
    from mosaic.core.pipeline.graph import run_pipeline

    _ = run_pipeline(ds, recipe, step_scopes=dict(scopes))


def _step_result(
    ds: Dataset, recipe: Recipe, scopes: Mapping[str, Scope], step: str
) -> Result[str]:
    """The run that *step* of *recipe* plans under *scopes*."""
    from mosaic.core.pipeline.graph import plan_pipeline

    planned = plan_pipeline(ds, recipe, step_scopes=dict(scopes)).step(step)
    assert planned.run_id is not None
    return Result(feature=planned.storage_name, run_id=planned.run_id)


def test_a_classifier_asked_for_a_wider_scope_records_its_training_entries(
    tmp_path: Path,
) -> None:
    _ = pytest.importorskip("torch")
    ds = _labeled_dataset(tmp_path / "ds")
    recipe = _recipe(
        "classifier",
        [
            *_labeled_steps(),
            {
                "id": "classifier",
                "type": "feature",
                "feature": "sequence-classifier",
                "inputs": [{"step": "pairs"}],
                "params": {
                    "templates": {"step": "labeled"},
                    "labels": {"step": "labels"},
                    "default_class": 0,
                    "device": "cpu",
                    "head": "tcn",
                    "head_config": {"hidden": 8, "blocks": 2},
                    "batch_size": 4,
                    "crop_length": 60,
                    "learning_rate": 0.01,
                    "steps": 20,
                },
            },
        ],
    )
    scopes = {"labeled": TRAIN, "classifier": WIDER}

    _run_recipe(ds, recipe, scopes)

    classifier = _step_result(ds, recipe, scopes, "classifier")
    fitted = _record(ds, classifier)
    assert _recorded_entries(fitted) == list(TRAIN_ENTRIES)
    assert fitted.inputs == _cells_now(ds, classifier, TRAIN_ENTRIES)
    labeled = _step_result(ds, recipe, scopes, "labeled")
    assert [_state_read(ref) for ref in fitted.via] == [
        ("templates", *_fit_state(ds, labeled))
    ]


def test_a_tuning_asked_for_a_wider_scope_records_its_fold_tables_entries(
    tmp_path: Path,
) -> None:
    ds = _labeled_dataset(tmp_path / "ds")
    recipe = _recipe(
        "tuned",
        [
            *_labeled_steps(),
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
                "id": "tune",
                "type": "feature",
                "feature": "tune-threshold",
                "inputs": [{"step": "clf"}],
                "params": {
                    "labels": {"step": "labels"},
                    "default_class": 0,
                    "average_over": [0, 1, 2],
                },
            },
        ],
    )
    scopes = {"labeled": TRAIN, "clf": WIDER, "tune": WIDER}

    _run_recipe(ds, recipe, scopes)

    tune = _step_result(ds, recipe, scopes, "tune")
    fitted = _record(ds, tune)
    assert _recorded_entries(fitted) == list(TRAIN_ENTRIES)
    assert fitted.inputs == _cells_now(ds, tune, TRAIN_ENTRIES)
    assert fitted.via == ()


def test_an_evaluation_records_its_entries_and_the_state_of_its_label_map(
    tmp_path: Path,
) -> None:
    ds = _labeled_dataset(tmp_path / "ds")
    recipe = _recipe(
        "held-out-map",
        [
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
    )
    scopes = {"templates": TRAIN, "map": TRAIN, "score": TEST}

    _run_recipe(ds, recipe, scopes)

    score = _step_result(ds, recipe, scopes, "score")
    fitted = _record(ds, score)
    assert fitted.inputs == _cells_now(ds, score, [("test", "t1")])
    label_map = _step_result(ds, recipe, scopes, "map")
    assert [_state_read(ref) for ref in fitted.via] == [
        ("label_map", *_fit_state(ds, label_map))
    ]


# --- the drift of a fit ------------------------------------------------------


GONE: Final = ("", "seq_b")


def _drift(
    ds: Dataset, result: Result[str], reads: DriftReads | None = None
) -> FitDrift | None:
    """What :func:`fit_drift` reports for *result*'s run, read through *ds*."""
    assert result.run_id is not None
    params = read_linked_run_params(ds, result.feature, result.run_id).params
    return fit_drift(ds, result.feature, result.run_id, params, reads)


def _remove_entry(ds: Dataset, entry: Entry, *upstreams: Result[str]) -> None:
    """Remove *entry* from the tracks index and the rows of *upstreams*."""
    scope = Scope(entries=[entry])
    assert ds.drop_entries(scope=scope) > 0
    for upstream in upstreams:
        index = feature_index(feature_index_path(ds, upstream.feature))
        dropped = index.drop_entries(scope=scope, run_id=upstream.run_id)
        assert not dropped.empty


def _speed_again(ds: Dataset) -> Result[str]:
    """Run ``speed-angvel`` over the tracks again, under its run id."""
    return run_step(ds, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)


def test_a_templates_fit_of_which_one_entry_was_converted_again_moved(
    tracked: Dataset,
) -> None:
    speed = _speed_again(tracked)
    templates = _templates(tracked, speed, None)
    reconvert_entry(tracked, ("", "seq_a"), composition="converted-again")
    assert _speed_again(tracked).run_id == speed.run_id

    drift = _drift(tracked, templates)

    assert drift is not None
    assert drift == FitDrift(
        link_id="",
        recorded=True,
        moved=(("", "seq_a"),),
        dropped=(),
        refitted=(),
    )
    assert drift.drifted


def test_a_fit_whose_inputs_did_not_change_has_not_drifted(tracked: Dataset) -> None:
    speed = _speed_again(tracked)
    templates = _templates(tracked, speed, None)

    drift = _drift(tracked, templates)

    assert drift is not None
    assert drift == FitDrift(
        link_id="", recorded=True, moved=(), dropped=(), refitted=()
    )
    assert not drift.drifted


def test_an_entry_gone_from_a_templates_fit_is_neither_moved_nor_dropped(
    tracked: Dataset,
) -> None:
    speed = _speed_again(tracked)
    templates = _templates(tracked, speed, None)
    _remove_entry(tracked, GONE, speed)

    drift = _drift(tracked, templates)

    assert drift is not None
    assert (drift.moved, drift.dropped, drift.drifted) == ((), (), False)


def _classifier_steps() -> list[dict[str, JsonValue]]:
    """The labeled steps and a sequence classifier trained on the templates."""
    return [
        *_labeled_steps(),
        {
            "id": "classifier",
            "type": "feature",
            "feature": "sequence-classifier",
            "inputs": [{"step": "pairs"}],
            "params": {
                "templates": {"step": "labeled"},
                "labels": {"step": "labels"},
                "default_class": 0,
                "device": "cpu",
                "head": "tcn",
                "head_config": {"hidden": 8, "blocks": 2},
                "batch_size": 4,
                "crop_length": 60,
                "learning_rate": 0.01,
                "steps": 20,
            },
        },
    ]


def _fitted_classifier(base: Path) -> tuple[Dataset, Recipe, Result[str]]:
    """A classifier trained on the train group and applied to both groups."""
    _ = pytest.importorskip("torch")
    ds = _labeled_dataset(base)
    recipe = _recipe("classifier", _classifier_steps())
    scopes = {"labeled": TRAIN, "classifier": WIDER}
    _run_recipe(ds, recipe, scopes)
    return ds, recipe, _step_result(ds, recipe, scopes, "classifier")


def test_a_training_entry_gone_from_a_classifiers_inputs_is_dropped(
    tmp_path: Path,
) -> None:
    ds, recipe, classifier = _fitted_classifier(tmp_path / "ds")
    pairs = _step_result(ds, recipe, {}, "pairs")
    _remove_entry(ds, ("train", "s2"), pairs)

    drift = _drift(ds, classifier)

    assert drift is not None
    assert drift == FitDrift(
        link_id="",
        recorded=True,
        moved=(),
        dropped=(("train", "s2"),),
        refitted=(),
    )
    assert drift.drifted


def test_an_entry_outside_a_classifiers_training_set_converted_again_is_not_drift(
    tmp_path: Path,
) -> None:
    ds, recipe, classifier = _fitted_classifier(tmp_path / "ds")
    pairs = _step_result(ds, recipe, {}, "pairs")
    params = read_linked_run_params(ds, classifier.feature, str(classifier.run_id))
    assert params.params is not None
    before, _ = current_entry_cells(DriftReads(ds), params.params, [("test", "t1")])
    reconvert_entry(ds, ("test", "t1"), composition="converted-again")
    upstream = _recipe("upstream", _labeled_steps()[:4])
    _run_recipe(ds, upstream, {})
    assert _step_result(ds, upstream, {}, "pairs") == pairs
    after, _ = current_entry_cells(DriftReads(ds), params.params, [("test", "t1")])
    assert after[("test", "t1")].tracks != before[("test", "t1")].tracks

    drift = _drift(ds, classifier)

    assert drift is not None
    assert not drift.drifted


def test_templates_written_again_under_their_run_id_refit_a_kmeans_fit(
    tracked: Dataset,
) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    move_tracks(tracked, VARIANT, SEQUENCES)
    _ = _speed_again(tracked)
    shutil.rmtree(result_run_root(tracked, runs.templates))
    again = _templates(tracked, runs.speed, None)
    assert again.run_id == runs.templates.run_id

    drift = _drift(tracked, runs.fit)

    assert drift is not None
    assert drift == FitDrift(
        link_id="", recorded=True, moved=(), dropped=(), refitted=("templates",)
    )
    assert drift.drifted


def test_an_entry_converted_again_does_not_refit_a_kmeans_fit(
    tracked: Dataset,
) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    reconvert_entry(tracked, ("", "seq_a"), composition="converted-again")
    _ = _speed_again(tracked)

    drift = _drift(tracked, runs.fit)

    assert drift is not None
    assert not drift.drifted


def test_a_record_without_its_inputs_is_not_recorded(tracked: Dataset) -> None:
    speed = _speed_again(tracked)
    templates = _templates(tracked, speed, None)
    record = result_run_root(tracked, templates) / FIT_SCOPE_NAME
    document = json.loads(record.read_text(encoding="utf-8"))
    del document["fit_inputs"]
    _ = record.write_text(json.dumps(document), encoding="utf-8")
    reconvert_entry(tracked, ("", "seq_a"), composition="converted-again")
    _ = _speed_again(tracked)

    drift = _drift(tracked, templates)

    assert drift is not None
    assert not drift.recorded
    assert not drift.drifted


def test_a_run_without_a_record_has_no_drift(tracked: Dataset) -> None:
    speed = _speed_again(tracked)

    assert _drift(tracked, speed) is None


def test_a_record_a_link_holds_names_the_link(tracked: Dataset, tmp_path: Path) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    copy = copy_of(tracked, tmp_path / "copy")

    drift = _drift(copy, runs.fit)

    assert drift == FitDrift(
        link_id=f"source-{tracked.uuid}",
        recorded=True,
        moved=(),
        dropped=(),
        refitted=(),
    )


def test_a_fit_read_under_an_unreachable_link_reads_as_blank(
    tracked: Dataset, tmp_path: Path
) -> None:
    speed = _speed_again(tracked)
    templates = _templates(tracked, speed, None)
    copy = copy_of(tracked, tmp_path / "copy")
    fit = run_step(
        copy,
        "global-kmeans",
        [model_reference(speed)],
        {"templates": model_reference(templates), **KMEANS},
        workers=WORKERS,
        mode=MODE,
    )
    assert (result_run_root(copy, fit) / FIT_SCOPE_NAME).is_file()
    offline = take_link_offline(copy)

    drift = _drift(offline, fit)

    assert drift == FitDrift(
        link_id="", recorded=True, moved=(), dropped=(), refitted=("templates",)
    )


def test_an_unreadable_record_raises_only_under_strict_reads(
    tracked: Dataset,
) -> None:
    speed = _speed_again(tracked)
    templates = _templates(tracked, speed, None)
    record = result_run_root(tracked, templates) / FIT_SCOPE_NAME
    _ = record.write_text("{not json", encoding="utf-8")

    assert _drift(tracked, templates) is None
    with pytest.raises(FitStateUnreadable):
        _ = _drift(tracked, templates, DriftReads(tracked, strict=True))


@pytest.mark.parametrize(
    ("drift", "summary"),
    [
        (
            FitDrift(
                link_id="",
                recorded=True,
                moved=(("g", "b"), ("g", "a")),
                dropped=(("g", "c"),),
                refitted=("templates", "labels"),
            ),
            "2 moved, 1 gone, fitted again: labels, templates (g:a, g:b, g:c)",
        ),
        (
            FitDrift(
                link_id="",
                recorded=True,
                moved=(),
                dropped=(),
                refitted=("templates",),
            ),
            "fitted again: templates",
        ),
        (
            FitDrift(
                link_id="",
                recorded=True,
                moved=tuple(("", f"s{n}") for n in range(7)),
                dropped=(),
                refitted=(),
            ),
            "7 moved (s0, s1, s2, s3, s4 and 2 more)",
        ),
    ],
    ids=["moved-gone-fitted", "fitted-alone", "sampled"],
)
def test_a_refit_summary_counts_names_the_fits_and_samples_the_entries(
    drift: FitDrift, summary: str
) -> None:
    assert refit_summary(drift) == summary


def test_a_refit_summary_samples_as_many_entries_as_it_is_asked() -> None:
    drift = FitDrift(
        link_id="",
        recorded=True,
        moved=(("g", "a"),),
        dropped=(("g", "b"),),
        refitted=(),
    )

    assert refit_summary(drift, sample=1) == "1 moved, 1 gone (g:a and 1 more)"
    assert refit_summary(drift, sample=0) == "1 moved, 1 gone"


# --- fitting again -------------------------------------------------------------


AGAIN: Final = "again, since the inputs its fit read changed"
"""The words of the line a run prints when it fits again."""


@dataclass(frozen=True, slots=True)
class _Offer:
    """One ``load_state`` call: the root offered, what it held, and the answer."""

    root: Path
    held: tuple[str, ...]
    loaded: bool


def _spy_loads(monkeypatch: pytest.MonkeyPatch, cls: type[Feature]) -> list[_Offer]:
    """Record each ``load_state`` call of *cls* made from now on."""
    offers: list[_Offer] = []
    real = cls.load_state

    def load_state(
        self: Feature,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        held = (
            tuple(sorted(path.name for path in run_root.iterdir()))
            if run_root.is_dir()
            else ()
        )
        loaded = real(self, run_root, artifact_paths, dependency_lookups)
        offers.append(_Offer(root=run_root, held=held, loaded=loaded))
        return loaded

    monkeypatch.setattr(cls, "load_state", load_state)
    return offers


def _spy_kmeans_fits(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record each ``global-kmeans`` fit from now on, ``True`` for one given a model."""
    from mosaic.behavior.feature_library.global_kmeans import GlobalKMeansClustering

    fits: list[bool] = []
    real = GlobalKMeansClustering.fit

    def fit(self: GlobalKMeansClustering, inputs: InputStream) -> None:
        fits.append(self.params.model is not None)
        real(self, inputs)

    monkeypatch.setattr(GlobalKMeansClustering, "fit", fit)
    return fits


def _record_bytes(ds: Dataset, result: Result[str]) -> bytes:
    """The bytes of the fit record of *result*'s run."""
    return (result_run_root(ds, result) / FIT_SCOPE_NAME).read_bytes()


def _fit_phases(ds: Dataset, result: Result[str]) -> list[str]:
    """The ``fit`` phase messages the attempt of *result* recorded."""
    assert result.execution_id is not None
    return [
        event["message"]
        for event in read_run_progress(run_log_dir(ds.base_dir), result.execution_id)
        if event["step_type"] == "phase" and event["message"].startswith("fit")
    ]


def _again_line(feature: str, run_id: str | None, summary: str) -> str:
    """The line a run of *feature* prints when it fits *run_id* again."""
    return (
        f"[feature:{feature}] fitting run {run_id} again, since the inputs its fit "
        f"read changed: {summary}"
    )


def test_a_fit_whose_entry_was_converted_again_is_fitted_again(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    templates_was = _record(tracked, chain.templates)
    fit_was = _record(tracked, chain.fit)
    fit_rows_was = index_cells(tracked, chain.fit, "started_at")
    move_tracks(tracked, VARIANT, ["seq_a"])

    again = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)

    assert again.run_ids() == chain.run_ids()
    assert again.apply.run_id == chain.apply.run_id
    templates_now = _record(tracked, again.templates)
    assert templates_now.execution_id == again.templates.execution_id
    assert templates_now.state_digest != templates_was.state_digest
    assert templates_now.inputs != templates_was.inputs
    assert templates_now.inputs == _cells_now(tracked, again.templates, BOTH)
    fit_now = _record(tracked, again.fit)
    assert fit_now.execution_id == again.fit.execution_id
    assert fit_now.state_digest != fit_was.state_digest
    fit_rows_now = index_cells(tracked, again.fit, "started_at")
    assert fit_rows_now.keys() == set(SEQUENCES)
    assert all(fit_rows_now[seq] != fit_rows_was[seq] for seq in SEQUENCES)
    digest = fit_now.state_digest
    assert model_cells(tracked, again.apply) == {"seq_a": digest, "seq_b": digest}
    err = capsys.readouterr().err
    assert (
        _again_line("extract-templates", chain.templates.run_id, "1 moved (seq_a)")
        in err
    )
    assert (
        _again_line("global-kmeans", chain.fit.run_id, "fitted again: templates") in err
    )


def test_a_fit_whose_inputs_did_not_change_is_served_and_keeps_its_record(
    tracked: Dataset,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    records = [_record_bytes(tracked, run) for run in (chain.templates, chain.fit)]
    fits = _spy_kmeans_fits(monkeypatch)
    _ = capsys.readouterr()

    again = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)

    assert fits == []
    assert (again.fit.cache_hit, again.apply.cache_hit) == (True, True)
    assert [
        _record_bytes(tracked, run) for run in (again.templates, again.fit)
    ] == records
    assert AGAIN not in capsys.readouterr().err


def test_a_kmeans_apply_fits_nothing_and_announces_no_refit(
    tracked: Dataset,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    move_tracks(tracked, VARIANT, ["seq_a"])
    _ = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    fits = _spy_kmeans_fits(monkeypatch)
    _ = capsys.readouterr()

    applied = kmeans_apply(tracked, chain, workers=WORKERS, mode=MODE)

    assert fits == []
    assert applied.failed_entries == ()
    assert not (result_run_root(tracked, applied) / FIT_SCOPE_NAME).exists()
    assert AGAIN not in capsys.readouterr().err


def test_a_kmeans_fit_is_fitted_again_when_its_templates_are_not_on_a_conversion(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    was = _record(tracked, runs.fit)
    move_tracks(tracked, VARIANT, ["seq_a"])
    speed = _speed_again(tracked)
    fits = _spy_kmeans_fits(monkeypatch)

    served = run_step(
        tracked,
        "global-kmeans",
        [model_reference(speed)],
        {"templates": model_reference(runs.templates), **KMEANS},
        workers=WORKERS,
        mode=MODE,
    )

    assert fits == []
    assert _record(tracked, served).execution_id == was.execution_id

    _ = _templates(tracked, speed, None)
    fitted = run_step(
        tracked,
        "global-kmeans",
        [model_reference(speed)],
        {"templates": model_reference(runs.templates), **KMEANS},
        workers=WORKERS,
        mode=MODE,
    )

    assert fits == [False]
    assert fitted.run_id == runs.fit.run_id
    now = _record(tracked, fitted)
    assert now.execution_id == fitted.execution_id
    assert now.state_digest != was.state_digest


def test_an_entry_gone_from_a_templates_scope_gives_the_step_another_run(
    tracked: Dataset,
) -> None:
    speed = _speed_again(tracked)
    templates = _templates(tracked, speed, None)
    _remove_entry(tracked, GONE, speed)

    again = _templates(tracked, speed, None)

    assert again.run_id != templates.run_id
    drift = _drift(tracked, templates)
    assert drift is not None
    assert (drift.moved, drift.dropped, drift.drifted) == ((), (), False)


def test_the_line_and_the_fit_phase_of_a_refit_carry_its_summary(
    tracked: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    speed = _speed_again(tracked)
    templates = _templates(tracked, speed, None)
    move_tracks(tracked, VARIANT, ["seq_a"])
    _ = _speed_again(tracked)
    summary = refit_summary(
        FitDrift(
            link_id="", recorded=True, moved=(("", "seq_a"),), dropped=(), refitted=()
        )
    )
    _ = capsys.readouterr()

    from mosaic.core.pipeline.graph import build_feature

    feature = build_feature(
        "extract-templates", [model_reference(speed)], {"n_templates": 20}
    )
    again = run_feature(tracked, feature)

    assert again.run_id == templates.run_id
    err = capsys.readouterr().err
    assert err.count(AGAIN) == 1
    assert _again_line("extract-templates", templates.run_id, summary) in err
    (phase,) = _fit_phases(tracked, again)
    assert phase.startswith("fit: ")
    assert summary in phase


def _refit_kmeans_after_a_move(tracked: Dataset) -> tuple[KMeansFit, Result[str]]:
    """Fit the k-means chain, move ``seq_a`` and fit its templates again."""
    runs = kmeans_fit(tracked, workers=WORKERS, mode=MODE)
    move_tracks(tracked, VARIANT, ["seq_a"])
    speed = _speed_again(tracked)
    _ = _templates(tracked, speed, None)
    return runs, speed


def test_a_kmeans_refit_loads_from_an_empty_directory_that_is_gone_after(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.behavior.feature_library.global_kmeans import GlobalKMeansClustering

    runs, speed = _refit_kmeans_after_a_move(tracked)
    own_root = result_run_root(tracked, runs.fit)
    assert (own_root / "model.joblib").is_file()
    offers = _spy_loads(monkeypatch, GlobalKMeansClustering)

    fitted = run_step(
        tracked,
        "global-kmeans",
        [model_reference(speed)],
        {"templates": model_reference(runs.templates), **KMEANS},
        workers=1,
        mode=MODE,
    )

    assert fitted.failed_entries == ()
    first = offers[0]
    assert first.root != own_root
    assert (first.held, first.loaded) == ((), False)
    assert not first.root.exists()
    assert all(offer.root == own_root for offer in offers[1:])
    assert _record(tracked, fitted).execution_id == fitted.execution_id


def _evaluation_recipe() -> Recipe:
    """The ``evaluate-labels`` recipe scoring k-means clusters through a label map."""
    return _recipe(
        "held-out-map",
        [
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
    )


def _mark_fitted_again(ds: Dataset, result: Result[str]) -> None:
    """Record another state digest in the fit record of *result*'s run.

    Stands for a fit written again with other state, which every fit that read
    it then compares.
    """
    root = result_run_root(ds, result)
    record = _record(ds, result)
    write_fit_scope(root, replace(record, state_digest="f" * len(record.state_digest)))


def test_a_label_map_fitted_again_scores_the_evaluation_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.behavior.feature_library.evaluate_labels import (
        METRICS_FILE,
        EvaluateLabels,
    )

    ds = _labeled_dataset(tmp_path / "ds")
    recipe = _evaluation_recipe()
    scopes = {"templates": TRAIN, "map": TRAIN, "score": TEST}
    _run_recipe(ds, recipe, scopes)
    score = _step_result(ds, recipe, scopes, "score")
    was = _record(ds, score)
    own_root = result_run_root(ds, score)
    written = (own_root / METRICS_FILE).stat().st_mtime_ns
    _mark_fitted_again(ds, _step_result(ds, recipe, scopes, "map"))
    assert _drift(ds, score) == FitDrift(
        link_id="", recorded=True, moved=(), dropped=(), refitted=("label_map",)
    )
    offers = _spy_loads(monkeypatch, EvaluateLabels)

    _run_recipe(ds, recipe, scopes)

    now = _record(ds, score)
    assert now.execution_id != was.execution_id
    assert [ref.state_digest for ref in now.via] == ["f" * len(was.state_digest)]
    assert (own_root / METRICS_FILE).stat().st_mtime_ns != written
    first = offers[0]
    assert first.root != own_root
    assert (first.held, first.loaded) == ((), False)
    assert not first.root.exists()
    drift = _drift(ds, score)
    assert drift is not None
    assert not drift.drifted


class _LoadsElsewhere(ScopedFittingFeature):
    """Finds a state in an empty root, as a load from an apply's model does."""

    name = "fit-loads-elsewhere"

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        held = super().load_state(run_root, artifact_paths, dependency_lookups)
        return held or not any(run_root.iterdir())


class _WritesOffered(ScopedFittingFeature):
    """Writes into the root its last ``load_state`` was offered, while it fits."""

    name = "fit-writes-offered"
    written = "working.txt"

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self.offered = run_root
        return super().load_state(run_root, artifact_paths, dependency_lookups)

    def fit(self, inputs: InputStream) -> None:
        super().fit(inputs)
        _ = (self.offered / self.written).write_text("fitting")


class _WritesBound(ScopedFittingFeature):
    """Writes a working file into the run root it is bound to, while it fits."""

    name = "fit-writes-bound"
    written = "working.txt"

    def bind_run_root(self, run_root: Path) -> None:
        self.bound = run_root

    def fit(self, inputs: InputStream) -> None:
        super().fit(inputs)
        _ = (self.bound / self.written).write_text(f"fit {self.control.fits}")


class _ResumesFits(ScopedFittingFeature):
    """Records each fit-inputs digest the engine hands it."""

    name = "fit-resumes"

    def __init__(self, control: FitControl) -> None:
        super().__init__(control)
        self.digests: list[str] = []

    def bind_fit_inputs(self, digest: str) -> None:
        self.digests.append(digest)


def test_each_fit_is_handed_the_digest_of_the_inputs_it_reads(
    tracked: Dataset,
) -> None:
    control = FitControl()
    first = _ResumesFits(control)
    fitted = run_feature(tracked, first, track=False)
    was = _record(tracked, fitted)
    move_tracks(tracked, VARIANT, ["seq_a"])
    again = _ResumesFits(control)

    refitted = run_feature(tracked, again, track=False)

    assert control.fits == 2
    now = _record(tracked, refitted)
    assert first.digests == [fit_inputs_digest(was.inputs or (), {})]
    assert again.digests == [fit_inputs_digest(now.inputs or (), {})]
    assert first.digests != again.digests


def _entry_inputs(sequence: str, tracks: str) -> FitEntryInputs:
    """The cells of one entry of group ``g``."""
    return FitEntryInputs(
        group="g",
        sequence=sequence,
        cells=EntryCells(roots="", tracks=tracks, inputs=""),
    )


def test_the_fit_inputs_digest_ignores_order_and_follows_each_state() -> None:
    entries = (_entry_inputs("a", "t1"), _entry_inputs("b", "t2"))
    states = {"labels": "11", "templates": "22"}

    digest = fit_inputs_digest(entries, states)

    assert digest == fit_inputs_digest(entries[::-1], dict(reversed(states.items())))
    assert digest != fit_inputs_digest(entries, {**states, "templates": "33"})
    assert digest != fit_inputs_digest(
        (_entry_inputs("a", "t1"), _entry_inputs("b", "moved")), states
    )


def _state_files(root: Path) -> dict[str, bytes]:
    """The state and the fit record in *root*."""
    return {name: (root / name).read_bytes() for name in ("state.json", FIT_SCOPE_NAME)}


def test_a_refit_whose_load_finds_a_state_in_an_empty_directory_is_refused(
    tracked: Dataset,
) -> None:
    control = FitControl()
    first = run_feature(tracked, _LoadsElsewhere(control), track=False)
    root = result_run_root(tracked, first)
    kept = _state_files(root)
    move_tracks(tracked, VARIANT, ["seq_a"])

    with pytest.raises(CannotFitAgain) as refused:
        _ = run_feature(tracked, _LoadsElsewhere(control), track=False)

    assert refused.value.error_kind == "conflict"
    assert refused.value.error_detail() == {
        "feature": _LoadsElsewhere.name,
        "run_id": first.run_id,
        "cause": "loads_elsewhere",
        "files": [],
    }
    assert control.fits == 1
    assert _state_files(root) == kept


def test_a_refit_that_writes_into_the_offered_directory_is_refused(
    tracked: Dataset,
) -> None:
    control = FitControl()
    first = run_feature(tracked, _WritesOffered(control), track=False)
    root = result_run_root(tracked, first)
    kept = _state_files(root)
    move_tracks(tracked, VARIANT, ["seq_a"])

    with pytest.raises(CannotFitAgain) as refused:
        _ = run_feature(tracked, _WritesOffered(control), track=False)

    assert refused.value.error_detail() == {
        "feature": _WritesOffered.name,
        "run_id": first.run_id,
        "cause": "wrote_outside_run_root",
        "files": [_WritesOffered.written],
    }
    assert _WritesOffered.written in str(refused.value)
    assert control.fits == 2
    assert _state_files(root) == kept


def test_a_refit_writes_its_working_files_into_the_bound_run_root(
    tracked: Dataset,
) -> None:
    control = FitControl()
    first = run_feature(tracked, _WritesBound(control), track=False)
    root = result_run_root(tracked, first)
    was = _record(tracked, first)
    move_tracks(tracked, VARIANT, ["seq_a"])
    feature = _WritesBound(control)

    again = run_feature(tracked, feature, track=False)

    assert again.run_id == first.run_id
    assert control.fits == 2
    assert feature.bound == root
    assert (root / _WritesBound.written).read_text() == "fit 2"
    assert (root / "state.json").is_file()
    assert _record(tracked, again).fitted_at != was.fitted_at


def test_an_unreadable_fit_inputs_is_refused_before_the_run_root_is_written(
    tracked: Dataset,
) -> None:
    speed = _speed_again(tracked)
    templates = _templates(tracked, speed, None)
    root = result_run_root(tracked, templates)
    record = root / FIT_SCOPE_NAME
    document = json.loads(record.read_text(encoding="utf-8"))
    document["fit_inputs"] = ["not", "an", "object"]
    _ = record.write_text(json.dumps(document), encoding="utf-8")
    before = tree_snapshot(root)

    with pytest.raises(FitStateUnreadable):
        _ = _templates(tracked, speed, None)

    assert_tree_unchanged(root, before)


def test_a_drifted_fit_reading_a_fit_under_an_unreachable_link_is_refused(
    tracked: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    speed = _speed_again(tracked)
    templates = _templates(tracked, speed, None)
    copy = copy_of(tracked, tmp_path / "copy")
    for sequence in SEQUENCES:
        add_tracks_table(
            copy,
            VARIANT,
            track_table("", sequence, n_rows=40, n_ids=2),
            consumed_composition="converted-here",
            std_format="mosaic_v1",
        )
    assert _speed_again(copy).run_id == speed.run_id
    fit = run_step(
        copy,
        "global-kmeans",
        [model_reference(speed)],
        {"templates": model_reference(templates), **KMEANS},
        workers=WORKERS,
        mode=MODE,
    )
    offline = take_link_offline(copy)
    root = result_run_root(offline, fit)
    before = tree_snapshot(root)
    fits = _spy_kmeans_fits(monkeypatch)

    with pytest.raises(LinkUnreachableError):
        _ = run_step(
            offline,
            "global-kmeans",
            [model_reference(speed)],
            {"templates": model_reference(templates), **KMEANS},
            workers=WORKERS,
            mode=MODE,
        )

    assert fits == []
    assert_tree_unchanged(root, before)


def test_a_classifier_fitted_again_scores_its_evaluation_again(
    tmp_path: Path,
) -> None:
    ds = _labeled_dataset(tmp_path / "ds")
    recipe = scored_classifier_recipe()
    _run_recipe(ds, recipe, {})
    clf = _step_result(ds, recipe, {}, "clf")
    score = _step_result(ds, recipe, {}, "score")
    clf_was = _record(ds, clf)
    cells_was = model_cells(ds, clf)
    inputs_was = index_cells(ds, score, "consumed_input_composition")
    score_was = _record(ds, score)
    reconvert_entry(ds, ("train", "s1"), composition="converted-again", scale=2.0)

    _run_recipe(ds, recipe, {})

    assert _step_result(ds, recipe, {}, "clf") == clf
    clf_now = _record(ds, clf)
    assert clf_now.execution_id != clf_was.execution_id
    assert clf_now.state_digest != clf_was.state_digest
    cells_now = model_cells(ds, clf)
    assert cells_now.keys() == cells_was.keys()
    assert all(cells_now[seq] != cells_was[seq] for seq in cells_was)
    inputs_now = index_cells(ds, score, "consumed_input_composition")
    assert all(inputs_now[seq] != inputs_was[seq] for seq in inputs_was)
    assert _record(ds, score).execution_id != score_was.execution_id


# --- what a pinned fit streams when fitted again -------------------------------


CONVERTED_AGAIN: Final = ("train", "s1")
"""The training entry converted again from changed sources."""

GONE_TRAINING: Final = ("train", "s2")
"""The training entry removed from the tracks index."""


def _spy_classifier_fits(monkeypatch: pytest.MonkeyPatch) -> list[list[Entry]]:
    """Record the entries each ``sequence-classifier`` fit streams from now on."""
    from mosaic.behavior.feature_library.sequence_classifier import (
        SequenceClassifier,
    )

    fits: list[list[Entry]] = []
    real = SequenceClassifier.fit

    def fit(self: SequenceClassifier, inputs: InputStream) -> None:
        fits.append(sorted(self.fitted_entries or ()))
        real(self, inputs)

    monkeypatch.setattr(SequenceClassifier, "fit", fit)
    return fits


def _spy_streamed(monkeypatch: pytest.MonkeyPatch, cls: type[Feature]) -> list[int]:
    """Record the entry count of each stream a fit of *cls* is offered."""
    counts: list[int] = []
    real = cls.fit

    def fit(self: Feature, inputs: InputStream) -> None:
        counts.append(inputs.n_entries)
        real(self, inputs)

    monkeypatch.setattr(cls, "fit", fit)
    return counts


def _classifier_scopes(classifier: Scope) -> dict[str, Scope]:
    """The step scopes that ask the classifier for *classifier*."""
    return {"labeled": TRAIN, "classifier": classifier}


def _trained_classifier(base: Path) -> tuple[Dataset, Recipe, Result[str]]:
    """A classifier fitted and applied over the train group alone."""
    _ = pytest.importorskip("torch")
    ds = _labeled_dataset(base)
    recipe = _recipe("classifier", _classifier_steps())
    _run_recipe(ds, recipe, _classifier_scopes(TRAIN))
    return ds, recipe, _step_result(ds, recipe, _classifier_scopes(TRAIN), "classifier")


def test_an_apply_over_a_wider_scope_or_one_without_the_training_entries_fits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, recipe, classifier = _trained_classifier(tmp_path / "ds")
    was = _record(ds, classifier)
    fits = _spy_classifier_fits(monkeypatch)

    _run_recipe(ds, recipe, _classifier_scopes(WIDER))
    _run_recipe(ds, recipe, _classifier_scopes(TEST))

    assert fits == []
    assert _record(ds, classifier) == was
    assert model_cells(ds, classifier).keys() == {"s1", "s2", "t1"}


def test_a_classifier_asked_for_the_test_entries_fits_again_over_its_training_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.behavior.feature_library.sequence_classifier import (
        SequenceClassifier,
    )

    ds, recipe, classifier = _trained_classifier(tmp_path / "ds")
    was = _record(ds, classifier)
    own_root = result_run_root(ds, classifier)
    fits = _spy_classifier_fits(monkeypatch)
    streamed = _spy_streamed(monkeypatch, SequenceClassifier)
    offers = _spy_loads(monkeypatch, SequenceClassifier)
    reconvert_entry(ds, CONVERTED_AGAIN, composition="converted-again", scale=2.0)

    _run_recipe(ds, recipe, _classifier_scopes(TEST))

    assert fits == [list(TRAIN_ENTRIES)]
    assert streamed == [len(TRAIN_ENTRIES)]
    first = offers[0]
    assert first.root != own_root
    assert (first.held, first.loaded) == ((), False)
    now = _record(ds, classifier)
    assert now.execution_id != was.execution_id
    assert now.state_digest != was.state_digest
    assert _recorded_entries(now) == list(TRAIN_ENTRIES)
    assert now.inputs == _cells_now(ds, classifier, TRAIN_ENTRIES)
    assert "t1" in model_cells(ds, classifier)


def _labels_of(ds: Dataset, entry: Entry) -> Path:
    """The label file of *entry*."""
    from mosaic.core.pipeline.labels_index import labels_rows

    rows = labels_rows(ds, "behavior")
    group, sequence = entry
    (path,) = rows.loc[
        (rows["group"].astype(str) == group)
        & (rows["sequence"].astype(str) == sequence),
        "abs_path",
    ]
    return ds.resolve_path(str(path))


def _run_over(
    ds: Dataset, spec: StepSpec, scope: Scope, *, refit: bool = False
) -> Result[str]:
    """Run the feature step *spec* over *scope*, fitting it again when *refit*."""
    from mosaic.core.pipeline.graph.resolve import build_step_feature

    return run_feature(
        ds,
        build_step_feature(spec),
        scope=scope,
        tracks_run_id=spec.tracks_run_id,
        labels_run_id=spec.labels_run_id,
        track=False,
        refit=refit,
    )


def _without_a_training_entry(base: Path) -> tuple[Dataset, Result[str], StepSpec]:
    """A trained classifier, the step it runs as, and a training entry removed.

    The entry leaves the tracks index and the rows of the step's input, while
    the templates the classifier pins still name it.
    """
    from mosaic.core.pipeline.graph import plan_pipeline

    ds, recipe, classifier = _trained_classifier(base)
    spec = (
        plan_pipeline(ds, recipe, step_scopes=_classifier_scopes(TEST))
        .step("classifier")
        .spec
    )
    _remove_entry(ds, GONE_TRAINING, _step_result(ds, recipe, {}, "pairs"))
    return ds, classifier, spec


def test_a_refit_streams_the_training_entries_its_inputs_still_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.behavior.feature_library.sequence_classifier import (
        SequenceClassifier,
    )

    ds, classifier, spec = _without_a_training_entry(tmp_path / "ds")
    _labels_of(ds, GONE_TRAINING).unlink()
    was = _record(ds, classifier)
    streamed = _spy_streamed(monkeypatch, SequenceClassifier)

    applied = _run_over(ds, spec, TEST)

    assert applied.run_id == classifier.run_id
    assert streamed == [1]
    now = _record(ds, classifier)
    assert now.execution_id != was.execution_id
    assert _recorded_entries(now) == [CONVERTED_AGAIN]


def test_a_refit_missing_a_training_entry_the_templates_label_is_refused(
    tmp_path: Path,
) -> None:
    ds, classifier, spec = _without_a_training_entry(tmp_path / "ds")
    was = _record(ds, classifier)

    with pytest.raises(
        ValueError,
        match=(
            r"\[sequence-classifier\] 1 train entries of the templates run have "
            r"labels and are not in this step's scope: train/s2\."
        ),
    ):
        _ = _run_over(ds, spec, TEST)

    assert _record(ds, classifier) == was


def test_the_other_scope_of_a_classifier_fitted_again_drifts_through_its_own_state(
    tmp_path: Path,
) -> None:
    ds, recipe, classifier = _trained_classifier(tmp_path / "ds")
    _run_recipe(ds, recipe, _classifier_scopes(TEST))
    was = _record(ds, classifier)
    held_out_was = model_cells(ds, classifier)["t1"]
    reconvert_entry(ds, CONVERTED_AGAIN, composition="converted-again", scale=2.0)

    _run_recipe(ds, recipe, _classifier_scopes(TRAIN))

    now = _record(ds, classifier)
    assert now.state_digest != was.state_digest
    assert drifted(ds, classifier) == (("test", "t1"),)
    assert model_cells(ds, classifier)["t1"] == held_out_was

    _run_recipe(ds, recipe, _classifier_scopes(TEST))

    labeled = _step_result(ds, recipe, _classifier_scopes(TRAIN), "labeled")
    assert model_cells(ds, classifier)["t1"] == fit_state_cell(
        {"templates": _fit_state(ds, labeled)[2], OWN_STATE_PART: now.state_digest}
    )
    assert drifted(ds, classifier) == ()


def _tuning_recipe() -> Recipe:
    """The labeled steps, a classifier with folds and the tuning of its thresholds."""
    return _recipe(
        "tuned",
        [
            *_labeled_steps(),
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
                "id": "tune",
                "type": "feature",
                "feature": "tune-threshold",
                "inputs": [{"step": "clf"}],
                "params": {
                    "labels": {"step": "labels"},
                    "default_class": 0,
                    "average_over": [0, 1, 2],
                },
            },
        ],
    )


def test_a_tuning_asked_for_the_test_entries_fits_again_over_its_fold_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.behavior.feature_library.tune_threshold import TuneThresholdFeature

    ds = _labeled_dataset(tmp_path / "ds")
    recipe = _tuning_recipe()
    trained = {"labeled": TRAIN, "clf": WIDER, "tune": TRAIN}
    _run_recipe(ds, recipe, trained)
    tune = _step_result(ds, recipe, trained, "tune")
    was = _record(ds, tune)
    own_root = result_run_root(ds, tune)
    streamed = _spy_streamed(monkeypatch, TuneThresholdFeature)
    offers = _spy_loads(monkeypatch, TuneThresholdFeature)
    reconvert_entry(ds, CONVERTED_AGAIN, composition="converted-again", scale=2.0)

    _run_recipe(ds, recipe, {**trained, "tune": TEST})

    assert streamed == [len(TRAIN_ENTRIES)]
    first = offers[0]
    assert first.root != own_root
    assert (first.held, first.loaded) == ((), False)
    now = _record(ds, tune)
    assert now.execution_id != was.execution_id
    assert _recorded_entries(now) == list(TRAIN_ENTRIES)
    assert "t1" in model_cells(ds, tune)


# --- fitting again on request --------------------------------------------------


UNRECORDED: Final = "records no inputs of its fit"
"""The words of the line a run prints when its fit records no inputs."""


def _unrecorded_line(feature: str, run_id: str | None) -> str:
    """The line a run of *feature* prints when the fit of *run_id* records no inputs."""
    return (
        f"[feature:{feature}] run {run_id} records no inputs of its fit, so a change "
        "to them goes undetected: run the step with --refit to fit it again and "
        "record them"
    )


def _forget_inputs(root: Path, *, whole: bool) -> None:
    """Leave the fit in *root* as one saved before its record: no record, or no inputs."""
    record = root / FIT_SCOPE_NAME
    if whole:
        record.unlink()
        return
    document = json.loads(record.read_text(encoding="utf-8"))
    del document["fit_inputs"]
    _ = record.write_text(json.dumps(document), encoding="utf-8")


def _scoped_fit(
    ds: Dataset,
    control: FitControl,
    *,
    overwrite: bool = False,
    refit: bool = False,
    execution_id: str | None = None,
) -> Result[str]:
    """Run :class:`ScopedFittingFeature` over *ds*."""
    return run_feature(
        ds,
        ScopedFittingFeature(control),
        overwrite=overwrite,
        refit=refit,
        execution_id=execution_id,
        track=False,
    )


@pytest.mark.parametrize("whole", [False, True], ids=["no_inputs", "no_record"])
def test_a_fit_saved_before_its_record_is_served_with_a_warning_until_it_is_refit(
    tracked: Dataset, capsys: pytest.CaptureFixture[str], whole: bool
) -> None:
    control = FitControl()
    first = _scoped_fit(tracked, control)
    _forget_inputs(result_run_root(tracked, first), whole=whole)
    line = _unrecorded_line(ScopedFittingFeature.name, first.run_id)
    _ = capsys.readouterr()

    served = _scoped_fit(tracked, control)

    assert control.fits == 1
    assert served.run_id == first.run_id
    err = capsys.readouterr().err
    assert line in err
    assert err.count(UNRECORDED) == 1

    overwritten = _scoped_fit(tracked, control, overwrite=True)

    assert control.fits == 1
    assert overwritten.run_id == first.run_id
    assert capsys.readouterr().err.count(UNRECORDED) == 1

    refitted = _scoped_fit(tracked, control, refit=True)

    assert control.fits == 2
    assert refitted.run_id == first.run_id
    assert UNRECORDED not in capsys.readouterr().err
    now = _record(tracked, refitted)
    assert now.execution_id == refitted.execution_id
    assert now.inputs == _cells_now(tracked, refitted, BOTH)

    _ = _scoped_fit(tracked, control)

    assert control.fits == 2
    assert UNRECORDED not in capsys.readouterr().err


def test_a_refit_asked_again_under_the_execution_id_that_refit_loads_its_fit(
    tracked: Dataset,
) -> None:
    control = FitControl()
    first = _scoped_fit(tracked, control)
    refitted = _scoped_fit(tracked, control, refit=True, execution_id=EXECUTION)
    assert control.fits == 2
    record = _record_bytes(tracked, refitted)

    again = _scoped_fit(tracked, control, refit=True, execution_id=EXECUTION)

    assert control.fits == 2
    assert again.run_id == first.run_id
    assert again.cache_hit
    assert _record_bytes(tracked, again) == record

    _ = _scoped_fit(tracked, control, refit=True)

    assert control.fits == 3


def test_a_refit_of_a_kmeans_apply_loads_and_fits_nothing_and_warns_nothing(
    tracked: Dataset,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = kmeans_fit_and_apply(tracked, workers=WORKERS, mode=MODE)
    fits = _spy_kmeans_fits(monkeypatch)
    _ = capsys.readouterr()

    applied = run_feature(
        tracked,
        kmeans_apply_feature(chain),
        parallel_workers=WORKERS,
        parallel_mode=MODE,
        track=False,
        refit=True,
    )

    assert fits == []
    assert applied.run_id == chain.apply.run_id
    assert applied.cache_hit
    err = capsys.readouterr().err
    assert UNRECORDED not in err
    assert AGAIN not in err


def test_a_refit_asked_over_the_test_entries_fits_over_the_recorded_training_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.core.pipeline.graph import plan_pipeline

    ds, recipe, classifier = _trained_classifier(tmp_path / "ds")
    was = _record(ds, classifier)
    spec = (
        plan_pipeline(ds, recipe, step_scopes=_classifier_scopes(TEST))
        .step("classifier")
        .spec
    )
    fits = _spy_classifier_fits(monkeypatch)

    refitted = _run_over(ds, spec, TEST, refit=True)

    assert refitted.run_id == classifier.run_id
    assert fits == [list(TRAIN_ENTRIES)]
    now = _record(ds, classifier)
    assert now.execution_id == refitted.execution_id
    assert now.execution_id != was.execution_id
    assert _recorded_entries(now) == list(TRAIN_ENTRIES)
    assert now.inputs == _cells_now(ds, classifier, TRAIN_ENTRIES)
