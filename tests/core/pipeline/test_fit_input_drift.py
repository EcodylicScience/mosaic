"""A fit's inputs, and the fit state a feature input carries down a chain.

The chain is the k-means fit, its apply through ``model``, a per-entry feature
over the apply's output and a second one over that. A fit written again moves
the apply's ``consumed_model_state``, which the feature reading the apply
records in its inputs cell, and the one below it in its own, so each recomputes
in turn.

A fit records the cells of each entry it read, the attempt that fitted and the
state of each fit it read, in its ``fit_scope.json``. Its drift compares that
record with what the dataset holds now.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.composition import input_part
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.fit_scope import (
    FIT_SCOPE_NAME,
    FitEntryInputs,
    FitScope,
    FitSourceRef,
    FitStateUnreadable,
    read_fit_scope,
    read_fit_state_digest,
)
from mosaic.core.pipeline.index import (
    DriftReads,
    FitDrift,
    current_entry_cells,
    feature_index,
    feature_index_path,
    fit_drift,
    refit_summary,
)
from mosaic.core.pipeline.inventory.params import (
    read_linked_run_params,
    read_run_params,
)
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import Inputs, Result
from mosaic.core.scope import Scope
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    KMEANS,
    KMeansChain,
    PlainFeature,
    add_tracks_variant,
    calms21_dataset,
    copy_of,
    drifted,
    kmeans_apply,
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
    take_link_offline,
)

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue
    from mosaic.core.pipeline.graph import Recipe

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


def _inputs_cells(ds: Dataset, result: Result[str]) -> dict[str, str]:
    """The ``consumed_input_composition`` cell of each own row of *result*'s run."""
    rows = pd.read_csv(
        feature_index_path(ds, result.feature), dtype=str, keep_default_na=False
    )
    rows = rows[rows["run_id"] == result.run_id]
    return dict(
        zip(
            rows["sequence"].astype(str),
            rows["consumed_input_composition"].astype(str),
            strict=True,
        )
    )


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

    part = input_part("", "", digest)
    assert part == f"{digest}|/"
    assert _inputs_cells(tracked, first) == {"seq_a": part, "seq_b": part}
    deeper = input_part("", part, "")
    assert _inputs_cells(tracked, second) == {"seq_a": deeper, "seq_b": deeper}
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
    part = input_part("", "", digest)
    assert _inputs_cells(tracked, first) == {"seq_a": part, "seq_b": part}


def test_a_chain_whose_upstream_rows_record_no_state_keeps_every_cells_form(
    tracked: Dataset,
) -> None:
    speed = run_step(tracked, "speed-angvel", None, {}, workers=WORKERS, mode=MODE)
    first = run_feature(tracked, _BelowApply(_read(speed)), track=False)
    second = run_feature(tracked, _BelowThat(_read(first)), track=False)

    assert set(model_cells(tracked, speed).values()) == {""}
    for result in (first, second):
        cells = _inputs_cells(tracked, result)
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
