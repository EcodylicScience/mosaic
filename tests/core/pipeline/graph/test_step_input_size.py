"""What a step reads and which of its entries a plain attempt would compute.

The report is the planner's answer to "how big is this step" before it runs: the
footer rows of its inputs, and its entries to compute largest first, so a caller
can name an attempt's entries or cut an array balanced by rows. It writes nothing.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry, entry_token, parse_entry_tokens
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    BoundRef,
    Recipe,
    Request,
    build_step_feature,
    execute_step,
    plan_pipeline,
    submit_request,
)
from mosaic.core.pipeline.graph.claims import FileFailureStore
from mosaic.core.pipeline.graph.resolve import StepBuildError
from mosaic.core.pipeline.graph import input_size as input_size_module
from mosaic.core.pipeline.graph.input_size import (
    EntryRowsWire,
    StepInputSizeWire,
    step_input_size,
)
from mosaic.core.pipeline.params_facts import params_identity_digest
from mosaic.core.pipeline.input_size import (
    NO_COVARIATES,
    manifest_input_sizes,
    tracks_covariates,
)
from mosaic.core.pipeline.manifest import build_manifest
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import Inputs
from mosaic.core.scope import Scope
from tests.helpers import (
    SPEED_OF_CONVERTED_STEP,
    add_tracks_variant,
    calms21_dataset,
    chain_recipe,
    chain_with_sibling_recipe,
    converting_recipe,
    invoke_ok,
    make_dataset,
    quarantine_entry,
    run_speed,
)

VARIANT = "convert-trex.0.2-1111111111"

SIZES = {"seq_a": 10, "seq_b": 30, "seq_c": 20, "seq_d": 20}
"""Each sequence's table length. ``seq_c`` and ``seq_d`` tie, to read the tie-break."""

ARHMM_STEP: dict[str, JsonValue] = {
    "id": "arhmm",
    "type": "feature",
    "feature": "arhmm",
    "inputs": [{"step": "speed"}],
}

TEMPLATES_STEP: dict[str, JsonValue] = {
    "id": "templates",
    "type": "feature",
    "feature": "extract-templates",
    "inputs": [{"step": "speed"}],
    "params": {"n_templates": 4},
}


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Four tables of different lengths, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    for sequence, n_rows in SIZES.items():
        add_tracks_variant(
            dataset, VARIANT, sequence, n_rows=n_rows, std_format="mosaic_v1"
        )
    return dataset


def _rows_of(dataset: Dataset, sequence: str) -> int:
    """The footer rows of one sequence's tracks, read the way the manifest reads them."""
    manifest, _ = build_manifest(
        dataset, Inputs(("tracks",)), Scope(entries=[("", sequence)])
    )
    [size] = manifest_input_sizes(manifest).values()
    return size.rows


def _entries(report: StepInputSizeWire) -> list[Entry]:
    return parse_entry_tokens(row.token for row in report.entries_to_compute_by_rows)


def test_a_fresh_step_reports_its_size_and_every_entry_largest_first(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    request = submit_request(tracked, recipe).request

    report = step_input_size(tracked, request, "speed")

    rows = {sequence: _rows_of(tracked, sequence) for sequence in SIZES}
    assert rows["seq_b"] > rows["seq_c"] == rows["seq_d"] > rows["seq_a"]
    assert report.request_id == request.request_id
    assert (report.step, report.kind) == ("speed", "feature")
    assert report.inputs_ready is True
    assert report.reason == ""
    assert report.scope_dependent is False
    assert report.fits is False
    assert report.entries == report.entries_to_compute == 4
    assert report.rows == report.rows_to_compute == sum(rows.values())
    assert report.largest_entry_rows == rows["seq_b"]
    assert report.columns_max > 0
    assert _entries(report) == [
        ("", "seq_b"),
        ("", "seq_c"),
        ("", "seq_d"),
        ("", "seq_a"),
    ]
    assert [row.rows for row in report.entries_to_compute_by_rows] == [
        rows["seq_b"],
        rows["seq_c"],
        rows["seq_d"],
        rows["seq_a"],
    ]
    for row in report.entries_to_compute_by_rows:
        assert isinstance(row, EntryRowsWire)
        assert row.token == entry_token(row.group, row.sequence)
    assert report.run_id == plan_pipeline(tracked, recipe).step("speed").run_id


def test_a_step_after_a_partial_run_lists_only_the_entries_left(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    request = submit_request(tracked, recipe).request
    feature = build_step_feature(plan_pipeline(tracked, recipe).step("speed").spec)
    _ = run_feature(
        tracked, feature, scope=Scope(entries=[("", "seq_b"), ("", "seq_a")])
    )

    report = step_input_size(tracked, request, "speed")

    assert report.entries == 4
    assert report.entries_to_compute == 2
    assert _entries(report) == [("", "seq_c"), ("", "seq_d")]
    assert report.rows_to_compute == _rows_of(tracked, "seq_c") + _rows_of(
        tracked, "seq_d"
    )
    assert report.rows == sum(_rows_of(tracked, sequence) for sequence in SIZES)


def test_a_quarantined_entry_is_left_out_and_one_backing_off_is_listed(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    request = submit_request(tracked, recipe).request
    planned = plan_pipeline(tracked, recipe).step("speed")
    store = FileFailureStore(tracked.base_dir)
    store.exclude(request.request_id, "speed", [("", "seq_b")])
    _ = store.note_entry_failure(
        (planned.storage_name, planned.run_id or "", "", "seq_c"),
        error="boom",
        execution_id="attempt",
    )

    report = step_input_size(tracked, request, "speed")

    assert _entries(report) == [("", "seq_c"), ("", "seq_d"), ("", "seq_a")]
    assert report.entries == 3


def test_a_complete_step_has_no_entries_to_compute(tracked: Dataset) -> None:
    request = submit_request(tracked, chain_recipe()).request
    _ = execute_step(tracked, request, "speed")

    report = step_input_size(tracked, request, "speed")

    assert report.entries == 4
    assert report.entries_to_compute == 0
    assert report.entries_to_compute_by_rows == ()
    assert report.rows_to_compute == 0
    assert report.rows == sum(_rows_of(tracked, sequence) for sequence in SIZES)


def test_a_bound_step_has_no_entries_to_compute(tracked: Dataset) -> None:
    older = run_speed(tracked, 3)
    bind = {"speed": BoundRef(feature="speed-angvel__from__tracks", run_id=older)}
    request = submit_request(tracked, chain_recipe(), bind=bind).request

    report = step_input_size(tracked, request, "speed")

    assert report.run_id == older
    assert report.entries_to_compute == 0
    assert report.entries_to_compute_by_rows == ()
    assert report.rows_to_compute == 0


def test_a_quarantined_entry_not_yet_excluded_is_left_out_and_blocks_the_step(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    request = submit_request(tracked, recipe).request
    quarantine_entry(tracked, recipe, "speed", "seq_b")

    report = step_input_size(tracked, request, "speed")

    assert _entries(report) == [("", "seq_c"), ("", "seq_d"), ("", "seq_a")]
    assert report.entries_to_compute == 3
    assert report.inputs_ready is False
    assert "held back after repeated failures" in report.reason
    assert report.run_id == plan_pipeline(tracked, recipe).step("speed").run_id


def test_allow_partial_leaves_a_quarantined_entry_out_and_records_nothing(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    request = submit_request(tracked, recipe).request
    partial = request.model_copy(update={"allow_partial": True})
    quarantine_entry(tracked, recipe, "speed", "seq_b")

    report = step_input_size(tracked, partial, "speed")

    assert _entries(report) == [("", "seq_c"), ("", "seq_d"), ("", "seq_a")]
    assert report.entries == 3
    assert report.inputs_ready is True
    store = FileFailureStore(tracked.base_dir)
    assert set(store.exclusions(request.request_id).entries) == set()


def test_a_scope_dependent_step_reports_the_run_it_names_without_the_quarantined(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    request = submit_request(tracked, recipe).request
    _ = execute_step(tracked, request, "speed")
    whole = step_input_size(tracked, request, "templates")
    quarantine_entry(tracked, recipe, "templates", "seq_b")

    blocked = step_input_size(tracked, request, "templates")
    partial = step_input_size(
        tracked, request.model_copy(update={"allow_partial": True}), "templates"
    )

    narrowed = plan_pipeline(tracked, recipe, quarantined=[("", "seq_b")])
    assert blocked.run_id == whole.run_id
    assert blocked.entries == 4
    assert blocked.entries_to_compute == 3
    assert blocked.inputs_ready is False
    assert partial.entries == partial.entries_to_compute == 3
    assert partial.run_id == narrowed.step("templates").run_id != whole.run_id
    assert partial.inputs_ready is True
    assert ("", "seq_b") not in _entries(partial)


def test_a_step_below_an_unrun_parent_is_not_ready_and_then_ready(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    request = submit_request(tracked, recipe).request

    before = step_input_size(tracked, request, "templates")

    assert before.inputs_ready is False
    assert "speed" in before.reason
    assert before.scope_dependent is True
    assert before.fits is True
    assert before.entries == before.entries_to_compute == 4
    assert before.rows == before.rows_to_compute == 0
    assert [row.rows for row in before.entries_to_compute_by_rows] == [0, 0, 0, 0]
    assert _entries(before) == [
        ("", "seq_a"),
        ("", "seq_b"),
        ("", "seq_c"),
        ("", "seq_d"),
    ]

    _ = execute_step(tracked, request, "speed")
    after = step_input_size(tracked, request, "templates")

    assert after.inputs_ready is True
    assert after.reason == ""
    assert after.entries_to_compute == 4
    assert after.rows == after.rows_to_compute > 0
    assert len(after.entries_to_compute_by_rows) == 4
    assert after.run_id == plan_pipeline(tracked, recipe).step("templates").run_id


def test_a_step_that_reads_what_an_op_has_not_converted_is_not_ready(
    tmp_path: Path,
) -> None:
    dataset = calms21_dataset(
        tmp_path / "ds", {"train": {"a": {"s1": 40, "s2": 30}}}, seed=3, n_classes=2
    )
    recipe = converting_recipe(SPEED_OF_CONVERTED_STEP, TEMPLATES_STEP)
    request = submit_request(dataset, recipe).request
    plan = plan_pipeline(dataset, recipe)

    speed = step_input_size(dataset, request, "speed")
    templates = step_input_size(dataset, request, "templates")

    assert speed.inputs_ready is False
    assert speed.reason != ""
    assert speed.run_id == plan.step("speed").run_id
    assert templates.inputs_ready is False
    assert templates.run_id is None
    assert plan.step("templates").run_id is None


def test_an_op_step_reports_that_it_has_no_input_size(tmp_path: Path) -> None:
    dataset = calms21_dataset(
        tmp_path / "ds", {"train": {"a": {"s1": 40}}}, seed=3, n_classes=2
    )
    request = submit_request(dataset, converting_recipe()).request

    report = step_input_size(dataset, request, "tracks")

    assert report.kind == "op"
    assert report.inputs_ready is False
    assert "op" in report.reason
    assert report.rows == report.rows_to_compute == report.largest_entry_rows == 0
    assert report.entries_to_compute_by_rows == ()


def test_a_feature_step_reports_its_version_emits_and_params(tracked: Dataset) -> None:
    recipe = chain_recipe()
    request = submit_request(tracked, recipe).request
    feature = build_step_feature(plan_pipeline(tracked, recipe).step("speed").spec)

    report = step_input_size(tracked, request, "speed")

    assert report.feature_version == feature.version == "0.3"
    assert report.emits == feature.emits == "individual"
    assert report.params_digest == params_identity_digest(feature.params)
    assert report.params_default is True


def test_params_default_is_true_at_defaults_and_false_with_an_override(
    tracked: Dataset,
) -> None:
    recipe = chain_with_sibling_recipe()
    request = submit_request(tracked, recipe).request

    at_defaults = step_input_size(tracked, request, "speed")
    overridden = step_input_size(tracked, request, "sibling")

    assert at_defaults.params_default is True
    assert overridden.params_default is False
    assert at_defaults.params_digest != overridden.params_digest


def test_a_hash_excluded_field_leaves_the_params_digest(tracked: Dataset) -> None:
    excluded: dict[str, JsonValue] = {
        **ARHMM_STEP,
        "id": "other",
        "params": {"backend": "numpy"},
    }
    changed: dict[str, JsonValue] = {
        **ARHMM_STEP,
        "id": "changed",
        "params": {"n_iter": 10},
    }
    recipe = chain_recipe().model_dump()
    recipe["steps"] = [*recipe["steps"], ARHMM_STEP, excluded, changed]
    request = submit_request(tracked, Recipe.model_validate(recipe)).request

    plain = step_input_size(tracked, request, "arhmm")
    with_backend = step_input_size(tracked, request, "other")
    with_iterations = step_input_size(tracked, request, "changed")

    assert plain.params_digest is not None
    assert with_backend.params_digest == plain.params_digest
    assert with_backend.params_default is True
    assert with_iterations.params_digest != plain.params_digest
    assert with_iterations.params_default is False


def test_an_op_step_reports_no_version_or_params(tmp_path: Path) -> None:
    dataset = calms21_dataset(
        tmp_path / "ds", {"train": {"a": {"s1": 40}}}, seed=3, n_classes=2
    )
    request = submit_request(dataset, converting_recipe()).request

    report = step_input_size(dataset, request, "tracks")

    assert report.feature_version is None
    assert report.emits is None
    assert report.params_digest is None
    assert report.params_default is None


def test_each_entry_to_compute_carries_its_covariates(tracked: Dataset) -> None:
    request = submit_request(tracked, chain_recipe()).request

    report = step_input_size(tracked, request, "speed")

    expected = tracks_covariates(
        tracked, _entries(report), tracks_run_id=None, variants=(VARIANT,)
    )
    assert len(report.entries_to_compute_by_rows) == len(SIZES)
    for row in report.entries_to_compute_by_rows:
        covariates = expected[(row.group, row.sequence)]
        assert covariates != NO_COVARIATES
        assert covariates.keypoints == 7
        assert row.model_dump(include=set(dataclasses.asdict(covariates))) == (
            dataclasses.asdict(covariates)
        )


def test_a_step_whose_inputs_are_not_ready_still_reports_its_params(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    request = submit_request(tracked, recipe).request
    feature = build_step_feature(plan_pipeline(tracked, recipe).step("templates").spec)

    report = step_input_size(tracked, request, "templates")

    assert report.inputs_ready is False
    assert report.feature_version == feature.version
    assert report.emits == feature.emits
    assert report.params_digest == params_identity_digest(feature.params)
    assert report.params_default is False
    for row in report.entries_to_compute_by_rows:
        assert row.model_dump(include=set(dataclasses.asdict(NO_COVARIATES))) == (
            dataclasses.asdict(NO_COVARIATES)
        )


def _losing_its_tracklets(dataset: Dataset) -> Request:
    """Submit the chain recipe, then record tracks that carry no tracklets.

    The plan then refuses ``speed``, which follows ids, at its inputs.
    """
    request = submit_request(dataset, chain_recipe()).request
    for sequence in SIZES:
        add_tracks_variant(
            dataset,
            VARIANT,
            sequence,
            n_rows=SIZES[sequence],
            std_format="mosaic_v1",
            carries_tracklets=False,
        )
    return request


def test_a_step_the_plan_refuses_still_reports_its_facts(tracked: Dataset) -> None:
    request = _losing_its_tracklets(tracked)
    plan = plan_pipeline(tracked, chain_recipe(), refuse=False)
    assert [problem.step for problem in plan.problems] == ["speed"]
    feature = build_step_feature(plan.step("speed").spec)

    report = step_input_size(tracked, request, "speed")

    assert report.inputs_ready is False
    assert plan.problems[0].message in report.reason
    assert report.feature_version == feature.version
    assert report.emits == feature.emits
    assert report.params_digest == params_identity_digest(feature.params)
    assert report.params_default is True


def test_a_step_the_plan_refuses_whose_feature_cannot_be_built_reports_no_facts(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _losing_its_tracklets(tracked)

    def unbuildable(_spec: object) -> object:
        raise StepBuildError("params", "speed-angvel", "cannot be built")

    monkeypatch.setattr(input_size_module, "build_step_feature", unbuildable)

    report = step_input_size(tracked, request, "speed")

    assert report.inputs_ready is False
    assert report.feature_version is None
    assert report.emits is None
    assert report.params_digest is None
    assert report.params_default is None


def test_an_unknown_step_is_a_key_error(tracked: Dataset) -> None:
    request = submit_request(tracked, chain_recipe()).request

    with pytest.raises(KeyError):
        _ = step_input_size(tracked, request, "nowhere")


def test_the_verb_prints_the_wire_model_as_json(tracked: Dataset) -> None:
    request = submit_request(tracked, chain_recipe()).request
    arguments = [
        "pipeline",
        "input-size",
        "--manifest",
        str(tracked.manifest_path),
        "--request",
        request.request_id,
        "--step",
        "speed",
        "--json",
    ]

    result = invoke_ok(arguments)

    expected = step_input_size(tracked, request, "speed")
    assert json.loads(result.stdout) == expected.model_dump(mode="json")


def test_the_verb_refuses_an_unknown_step_and_an_unknown_request(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    base = ["pipeline", "input-size", "--manifest", str(tracked.manifest_path)]

    unknown_step = CliRunner().invoke(
        app, [*base, "--request", request.request_id, "--step", "nowhere"]
    )
    unknown_request = CliRunner().invoke(
        app, [*base, "--request", "01ARZ3NDEKTSV4RRFFQ69G5FAV", "--step", "speed"]
    )

    assert unknown_step.exit_code != 0
    assert "nowhere" in unknown_step.stderr
    assert unknown_request.exit_code != 0
