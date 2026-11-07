"""The problems a dataset finds with a recipe that validates.

Validation decides from declarations alone. A time filter is valid on any
dataset there, and a column that the track schemas allow is only a caution. A plan reads
the dataset. A time filter without a frame rate, and a required column that every
tracks table lacks, are problems there, each at the place in the recipe that would
change. ``plan_pipeline`` raises them together as ``RecipeRefused``, and a step
that finds one of its own when it re-plans itself is refused and recorded.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import mosaic.core.pipeline.graph.step as step_module
from mosaic.behavior.feature_library.pairposedistancepca import PairPoseDistancePCA
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.helpers import TIME_FILTER_NEEDS_FPS
from mosaic.core.json_value import JsonValue
from mosaic.core.label_converter import LABEL_CONVERTERS
from mosaic.core.pipeline.graph import (
    MISSING_SAMPLE,
    BoundRef,
    FeatureStepSpec,
    Recipe,
    RecipeInvalid,
    RecipeRefused,
    RecipeRefusedAtStep,
    Request,
    declaration_catalog,
    editor_catalog,
    execute_step,
    plan_pipeline,
    plan_wire,
    read_recipe,
    request_rollup,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.graph.columns import measured_columns
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE
from mosaic.core.scope import Scope
from mosaic.core.track_library.calms21 import Calms21Converter
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import (
    ListedEntries,
    add_tracks_variant,
    calms21_dataset,
    guppies_recipe,
    make_dataset,
    write_listed_entries,
)

runner = CliRunner()

VARIANT = "convert-trex.0.2-1111111111"

FILTER_PROBLEM = "a time filter needs the dataset's frame rate"


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two ``mosaic_v1`` tables with keypoints and no ``ANGLE``, and no frame rate."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def _delta(*, with_heading: bool) -> Recipe:
    """``nn-delta-response`` over smoothed tracks, which require ``ANGLE``.

    ``heading`` writes ``ANGLE`` and the identity columns. It joins the merge as
    one more input when *with_heading* is set.
    """
    heading: list[dict[str, JsonValue]] = (
        [{"id": "heading", "type": "feature", "feature": "heading"}]
        if with_heading
        else []
    )
    extra: list[JsonValue] = [{"step": "heading"}] if with_heading else []
    return read_recipe(
        {
            "steps": [
                *heading,
                {"id": "smooth", "type": "feature", "feature": "trajectory-smooth"},
                {
                    "id": "nn",
                    "type": "feature",
                    "feature": "nearest-neighbor",
                    "inputs": [{"step": "smooth"}],
                },
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": [{"step": "smooth"}],
                },
                {
                    "id": "delta",
                    "type": "feature",
                    "feature": "nn-delta-response",
                    "inputs": [
                        {"step": "smooth"},
                        {"step": "nn"},
                        {"step": "speed"},
                        *extra,
                    ],
                },
            ]
        }
    )


def _refused(dataset: Dataset, recipe: Recipe) -> RecipeRefused:
    with pytest.raises(RecipeRefused) as caught:
        _ = plan_pipeline(dataset, recipe)
    return caught.value


# --- a time filter without a frame rate --------------------------------------------


def test_a_time_filter_without_a_frame_rate_is_a_problem_at_the_filter(
    tracked: Dataset,
) -> None:
    recipe = guppies_recipe()
    filtered = {
        step.id
        for step in recipe.steps
        if isinstance(step, FeatureStepSpec) and step.run.filter_start_time is not None
    }

    refusal = _refused(tracked, recipe)

    at_filter = [p for p in refusal.problems if p.where == "run.filter_start_time"]
    assert {problem.step for problem in at_filter} == filtered
    assert all(FILTER_PROBLEM in problem.message for problem in at_filter)
    assert all(problem.message == TIME_FILTER_NEEDS_FPS for problem in at_filter)
    # Each filtered step is reported, and none of them has an identity.
    assert {
        step for step, run_id in refusal.plan.run_ids.items() if run_id is None
    } >= (filtered)


def test_a_refusal_is_a_recipe_refusal_and_lists_every_problem(
    tracked: Dataset,
) -> None:
    refusal = _refused(tracked, guppies_recipe())

    assert isinstance(refusal, RecipeInvalid)
    assert str(refusal).startswith("this recipe cannot be run on this dataset: ")
    assert f"{len(refusal.problems)} problems" in str(refusal)


def test_a_frame_rate_clears_the_filter_problems(tracked: Dataset) -> None:
    tracked.meta["fps_default"] = 30.0

    refusal = _refused(tracked, guppies_recipe())

    assert {problem.where for problem in refusal.problems} == {"inputs"}


def test_a_frame_and_a_time_on_one_boundary_are_refused_at_parse() -> None:
    document: dict[str, JsonValue] = {
        "steps": [
            {
                "id": "speed",
                "type": "feature",
                "feature": "speed-angvel",
                "run": {"filter_start_frame": 10, "filter_start_time": 1.0},
            }
        ]
    }

    with pytest.raises(RecipeInvalid) as caught:
        _ = read_recipe(document)

    (problem,) = caught.value.problems
    assert (problem.step, problem.where) == ("speed", "run")
    assert "one of filter_start_frame and filter_start_time" in problem.message


# --- a column that the tracks tables lack ------------------------------------------


def test_a_column_every_table_lacks_is_a_problem_naming_its_provider(
    tracked: Dataset,
) -> None:
    refusal = _refused(tracked, _delta(with_heading=False))

    (problem,) = refusal.problems
    assert (problem.step, problem.where) == ("delta", "inputs")
    assert "needs 'ANGLE'" in problem.message
    assert "heading provides it" in problem.message


def test_the_column_s_provider_as_an_input_plans(tracked: Dataset) -> None:
    plan = plan_pipeline(tracked, _delta(with_heading=True))

    assert plan.problems == ()
    assert all(run_id is not None for run_id in plan.run_ids.values())


def test_the_tables_decide_where_the_schemas_allow(tracked: Dataset) -> None:
    """Validation knows only what a schema allows, so ``ANGLE`` is a caution there."""
    from mosaic.core.pipeline.graph import validation_report

    report = validation_report(_delta(with_heading=False).model_dump(mode="json"))

    assert report.valid
    assert [(caution.step, caution.where) for caution in report.cautions] == [
        ("delta", "inputs")
    ]


def test_measured_columns_guarantee_what_every_table_holds() -> None:
    columns = measured_columns([("frame", "X", "ANGLE"), ("frame", "X")])

    assert columns is not None
    assert columns.guaranteed == {"frame", "X"}
    assert columns.possible == {"frame", "X", "ANGLE"}
    assert not columns.admits("Y")
    assert measured_columns([]) is None


# --- the plan, the wire and the CLI ------------------------------------------------


def test_an_unrefused_plan_carries_the_problems_onto_the_wire(
    tracked: Dataset,
) -> None:
    plan = plan_pipeline(tracked, _delta(with_heading=False), refuse=False)

    wire = plan_wire(plan)

    assert len(plan.problems) == 1
    assert wire.problems == plan.problems
    assert plan_wire(plan_pipeline(tracked, _delta(with_heading=True))).problems == ()


def test_the_cli_plan_prints_the_refused_plan_and_exits_one(
    tracked: Dataset, tmp_path: Path
) -> None:
    recipe_path = tmp_path / "recipe.json"
    _ = recipe_path.write_text(
        json.dumps(_delta(with_heading=False).model_dump(mode="json"))
    )

    result = runner.invoke(
        app,
        [
            "pipeline",
            "plan",
            "--recipe",
            f"@{recipe_path}",
            "--manifest",
            str(tracked.manifest_path),
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert [(found["step"], found["where"]) for found in payload["problems"]] == [
        ("delta", "inputs")
    ]


# --- a step that runs ----------------------------------------------------------------


def _losing_its_frame_rate(dataset: Dataset) -> Request:
    """A request submitted with a frame rate, on a dataset that then loses it.

    ``speed`` has no time filter and ``late`` has one.
    """
    dataset.meta["fps_default"] = 30.0
    recipe = read_recipe(
        {
            "steps": [
                {"id": "speed", "type": "feature", "feature": "speed-angvel"},
                {
                    "id": "late",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "run": {"filter_start_time": 0.5},
                },
            ]
        }
    )
    request = submit_request(dataset, recipe).request
    del dataset.meta["fps_default"]
    return request


def test_a_running_step_refuses_only_its_own_problem(tracked: Dataset) -> None:
    """The dataset can change between a submission and a step's attempt."""
    request = _losing_its_frame_rate(tracked)

    ran = execute_step(tracked, request, "speed")
    with pytest.raises(RecipeRefusedAtStep) as caught:
        _ = execute_step(tracked, request, "late")

    assert ran.run_id is not None
    assert caught.value.reason == "recipe_refused"
    assert [(p.step, p.where) for p in caught.value.problems] == [
        ("late", "run.filter_start_time")
    ]


def test_a_refused_running_step_records_its_attempt(tracked: Dataset) -> None:
    """A refusal with no run-log is a step, and a request, that reads as running."""
    request = _losing_its_frame_rate(tracked)
    _ = execute_step(tracked, request, "speed")

    with pytest.raises(RecipeRefusedAtStep):
        _ = execute_step(tracked, request, "late")

    logged = read_run(run_log_dir(tracked.base_dir), request.execution_of("late"))
    assert logged is not None
    assert logged["status"] == "failed"
    recorded = json.loads(logged["error_json"])
    assert recorded["reason"] == "recipe_refused"
    assert recorded["step"] == "late"
    assert [(p["step"], p["where"]) for p in recorded["problems"]] == [
        ("late", "run.filter_start_time")
    ]
    assert request_rollup(tracked.base_dir, request).status == "failed"


def test_the_cli_exits_with_the_refused_code_for_a_refused_step(
    tracked: Dataset,
) -> None:
    request = _losing_its_frame_rate(tracked)

    result = runner.invoke(
        app,
        [
            "run",
            "--json",
            "--manifest",
            str(tracked.manifest_path),
            "--graph-request",
            request.request_id,
            "--step",
            "late",
            "--execution-id",
            request.execution_of("late"),
        ],
    )

    assert result.exit_code == REFUSED_EXIT_CODE, result.output
    payload = json.loads(result.stdout)
    assert (payload["status"], payload["reason"]) == ("refused", "recipe_refused")


@pytest.mark.parametrize("kind", ["convert-tracks", "convert-labels"])
def test_a_convert_step_with_no_raw_rows_of_its_format_is_a_problem(
    tmp_path: Path, kind: str
) -> None:
    """A dataset whose sources were never scanned for the format is refused."""
    dataset = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4}}}, scan=False)
    recipe = read_recipe(
        {
            "schema_version": 1,
            "name": "convert",
            "steps": [
                {
                    "id": "convert",
                    "type": "op",
                    "kind": kind,
                    "params": {"source_format": "calms21_npy"},
                }
            ],
        }
    )

    with pytest.raises(RecipeRefused) as refused:
        _ = plan_pipeline(dataset, recipe)

    (problem,) = refused.value.problems
    assert (problem.step, problem.where) == ("convert", "params.source_format")
    assert "scan the dataset's sources first" in problem.message

    dataset.scan_tracks()
    dataset.scan_labels()
    plan = plan_pipeline(dataset, recipe)
    assert plan.step("convert").run_id


def _listed_conversion(params: dict[str, JsonValue]) -> Recipe:
    """A ``convert-labels`` step of :class:`ListedEntries` without a ``kind``."""
    return read_recipe(
        {
            "schema_version": 1,
            "name": "convert",
            "steps": [
                {
                    "id": "convert",
                    "type": "op",
                    "kind": "convert-labels",
                    "params": {
                        "source_format": ListedEntries.src_format,
                        "params": params,
                    },
                }
            ],
        }
    )


def test_a_convert_labels_step_whose_selection_takes_no_row_is_a_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step names no kind, and its converter is registered for ``behavior``."""
    monkeypatch.setitem(
        LABEL_CONVERTERS, (ListedEntries.src_format, "behavior"), ListedEntries
    )
    dataset = make_dataset(tmp_path / "ds")
    uploads = write_listed_entries(tmp_path / "uploads", {"f1": [("g", "s1")]})
    dataset.index_labels_raw(
        [uploads], patterns=["*.npy"], src_format=ListedEntries.src_format
    )

    with pytest.raises(RecipeRefused) as refused:
        _ = plan_pipeline(dataset, _listed_conversion({"name": "none*"}))

    (problem,) = refused.value.problems
    assert (problem.step, problem.where) == ("convert", "params.params")
    assert "'name': 'none*'" in problem.message
    assert (
        plan_pipeline(dataset, _listed_conversion({"name": "f1*"}))
        .step("convert")
        .run_id
    )


def test_a_convert_labels_step_whose_converter_params_do_not_build_is_a_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        LABEL_CONVERTERS, (ListedEntries.src_format, "behavior"), ListedEntries
    )
    dataset = make_dataset(tmp_path / "ds")
    uploads = write_listed_entries(tmp_path / "uploads", {"f1": [("g", "s1")]})
    dataset.index_labels_raw(
        [uploads], patterns=["*.npy"], src_format=ListedEntries.src_format
    )

    with pytest.raises(RecipeRefused) as refused:
        _ = plan_pipeline(dataset, _listed_conversion({"name": ["f1*"]}))

    (problem,) = refused.value.problems
    assert (problem.step, problem.where) == ("convert", "params.params")
    assert "name" in problem.message


def _predictions_conversion(files: str) -> Recipe:
    """A ``convert-labels`` step of ``predictions_table`` selecting *files*."""
    return read_recipe(
        {
            "schema_version": 1,
            "name": "convert",
            "steps": [
                {
                    "id": "predictions",
                    "type": "op",
                    "kind": "convert-labels",
                    "params": {
                        "kind": "predictions",
                        "source_format": "predictions_table",
                        "params": {"files": files},
                    },
                }
            ],
        }
    )


def test_a_predictions_table_step_whose_files_match_no_row_is_a_problem(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    root = dataset.get_root("labels_raw") / "predictions" / "own"
    root.mkdir(parents=True)
    pd.DataFrame({"frame": [0], "predicted_label": [1]}).to_csv(
        root / "s1.csv", index=False
    )
    dataset.index_labels_raw([root], patterns=["*.csv"], src_format="predictions_table")

    with pytest.raises(RecipeRefused) as refused:
        _ = plan_pipeline(dataset, _predictions_conversion("labels_raw/other/*"))

    (problem,) = refused.value.problems
    assert (problem.step, problem.where) == ("predictions", "params.params")
    assert "labels_raw/other/*" in problem.message
    plan = plan_pipeline(dataset, _predictions_conversion("labels_raw/predictions/*"))
    assert plan.step("predictions").run_id


# --- a step that follows ids over tracks without tracklets --------------------------


def _following_dataset(root: Path, **flags: bool | None) -> Dataset:
    """A dataset with one variant whose *flags* give each sequence's tracklets cell.

    A sequence maps to ``carries_tracklets``; ``None`` leaves the cell blank.
    """
    dataset = make_dataset(root)
    for sequence, flag in flags.items():
        add_tracks_variant(
            dataset, VARIANT, sequence, std_format="mosaic_v1", carries_tracklets=flag
        )
    return dataset


def _single(feature: str) -> Recipe:
    return read_recipe(
        {"steps": [{"id": "step", "type": "feature", "feature": feature}]}
    )


def test_a_step_that_follows_ids_is_refused_over_tracks_without_tracklets(
    tmp_path: Path,
) -> None:
    dataset = _following_dataset(tmp_path / "ds", seq_a=False)

    refusal = _refused(dataset, _single("speed-angvel"))

    (problem,) = refusal.problems
    assert (problem.step, problem.where) == ("step", "inputs")
    assert "speed-angvel" in problem.message
    assert "ids that name one animal over time" in problem.message
    assert "seq_a" in problem.message


def test_heading_is_not_refused_over_tracks_without_tracklets(tmp_path: Path) -> None:
    dataset = _following_dataset(tmp_path / "ds", seq_a=False)

    plan = plan_pipeline(dataset, _single("heading"))

    assert plan.problems == ()


def test_a_blank_tracklets_cell_is_not_refused(tmp_path: Path) -> None:
    dataset = _following_dataset(tmp_path / "ds", seq_a=None)

    plan = plan_pipeline(dataset, _single("speed-angvel"))

    assert plan.problems == ()


def test_a_step_reading_another_steps_output_is_not_checked(tmp_path: Path) -> None:
    dataset = _following_dataset(tmp_path / "ds", seq_a=False)
    recipe = read_recipe(
        {
            "steps": [
                {"id": "scaled", "type": "feature", "feature": "scale-to-cm"},
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": [{"step": "scaled"}],
                },
            ]
        }
    )

    plan = plan_pipeline(dataset, recipe, refuse=False)

    assert [problem for problem in plan.problems if problem.step == "speed"] == []


def test_one_entry_without_tracklets_refuses_the_step(tmp_path: Path) -> None:
    dataset = _following_dataset(tmp_path / "ds", seq_a=True, seq_b=False)

    refusal = _refused(dataset, _single("speed-angvel"))

    (problem,) = refusal.problems
    assert "seq_b" in problem.message
    assert "seq_a" not in problem.message


def test_a_scope_that_leaves_out_the_entry_without_tracklets_plans(
    tmp_path: Path,
) -> None:
    dataset = _following_dataset(tmp_path / "ds", seq_a=True, seq_b=False)

    plan = plan_pipeline(
        dataset, _single("speed-angvel"), scope=Scope(entries=[("", "seq_a")])
    )

    assert plan.problems == ()


def test_the_message_samples_the_entries(tmp_path: Path) -> None:
    sequences = {f"seq_{index}": False for index in range(MISSING_SAMPLE + 2)}
    dataset = _following_dataset(tmp_path / "ds", **sequences)

    refusal = _refused(dataset, _single("speed-angvel"))

    (problem,) = refusal.problems
    assert "seq_0" in problem.message
    assert f"seq_{MISSING_SAMPLE}" not in problem.message
    assert "2 more" in problem.message


def test_the_message_names_the_variant_of_the_rows_it_read(tmp_path: Path) -> None:
    dataset = _following_dataset(tmp_path / "ds", seq_a=False)

    refusal = _refused(dataset, _single("speed-angvel"))

    (problem,) = refusal.problems
    assert f"tracks variant '{VARIANT}'" in problem.message


def test_the_message_names_each_variant_in_order_and_counts_in_the_singular(
    tmp_path: Path,
) -> None:
    other = "convert-trex.0.2-0000000000"
    dataset = make_dataset(tmp_path / "ds")
    add_tracks_variant(
        dataset, VARIANT, "seq_a", std_format="mosaic_v1", carries_tracklets=False
    )
    add_tracks_variant(
        dataset, other, "seq_b", std_format="mosaic_v1", carries_tracklets=False
    )

    refusal = _refused(dataset, _single("speed-angvel"))

    (problem,) = refusal.problems
    assert f"tracks variants '{other}', '{VARIANT}'" in problem.message
    assert "for 2 entries" in problem.message
    single = _refused(
        _following_dataset(tmp_path / "one", seq_a=False), _single("speed-angvel")
    )
    assert "for 1 entry:" in single.problems[0].message


# --- a step that is bound, pinned or reads a variant an op mints ----------------------


def test_a_bound_step_is_not_refused_over_tracks_without_tracklets(
    tmp_path: Path,
) -> None:
    dataset = _following_dataset(tmp_path / "ds", seq_a=False)
    bound = BoundRef(feature="speed-angvel__from__tracks", run_id="0.1-aaaaaaaaaa")

    plan = plan_pipeline(
        dataset,
        _single("speed-angvel"),
        request=Request(request_id="R", recipe_digest="0" * 16, bind={"step": bound}),
        refuse=False,
    )

    assert [p for p in plan.problems if "one animal over time" in p.message] == []


@pytest.fixture
def untracked_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Dataset]:
    """CalMS21 tracks converted by a converter that records no tracklets."""
    with monkeypatch.context() as patch:
        patch.setattr(Calms21Converter, "carries_tracklets", False)
        dataset = calms21_dataset(
            tmp_path / "ds", {"train": {"a": {"s1": 40, "s2": 30}}}, seed=5
        )
        _ = dataset.convert_all_tracks()
    yield dataset


def test_a_pinned_step_is_not_refused_over_tracks_without_tracklets(
    untracked_split: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fitted step is pinned to a finished run, which reads none of the tracks."""
    recipe = read_recipe(
        {
            "steps": [
                {
                    "id": "fit",
                    "type": "feature",
                    "feature": "pair-posedistance-pca",
                    "inputs": ["tracks"],
                    "params": {"n_components": 2, "pose": {"pose_n": 3}},
                }
            ]
        }
    )
    run = run_pipeline(untracked_split, recipe)
    pin = run.outcomes[0].run_id
    assert pin
    with monkeypatch.context() as patch:
        patch.setattr(PairPoseDistancePCA, "follows_ids", True)
        declaration_catalog.cache_clear()
        editor_catalog.cache_clear()
        try:
            unpinned = plan_pipeline(untracked_split, recipe, refuse=False)
            pinned = plan_pipeline(
                untracked_split, recipe, pins={"fit": pin}, refuse=False
            )
        finally:
            declaration_catalog.cache_clear()
            editor_catalog.cache_clear()

    assert [p.where for p in unpinned.problems] == ["inputs"]
    assert pinned.problems == ()


def test_a_variant_an_op_mints_is_refused_when_the_reader_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No row of the variant exists at plan time, so only the step's own plan sees it."""
    dataset = _following_dataset(tmp_path / "ds", seq_a=True)
    recipe = read_recipe(
        {
            "steps": [
                {
                    "id": "regrid",
                    "type": "op",
                    "kind": "resample-tracks",
                    "params": {"target_fps": 15.0},
                },
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "tracks": {"step": "regrid"},
                },
            ]
        }
    )
    submitted = submit_request(dataset, recipe)  # the plan holds no problem
    regrid = submitted.plan.step("regrid")
    minted = regrid.tracks_variant
    assert minted

    def _resample(*args: object, **kwargs: object) -> str:
        add_tracks_variant(
            dataset,
            minted,
            "seq_a",
            std_format="mosaic_v1",
            producer_run_id=regrid.run_id or "",
            carries_tracklets=False,
        )
        return regrid.run_id or ""

    monkeypatch.setattr(step_module, "run_op", _resample)
    _ = execute_step(dataset, submitted.request, "regrid")
    with pytest.raises(RecipeRefusedAtStep) as caught:
        _ = execute_step(dataset, submitted.request, "speed")

    assert caught.value.reason == "recipe_refused"
    [problem] = caught.value.problems
    assert (problem.step, problem.where) == ("speed", "inputs")
    assert minted in problem.message
