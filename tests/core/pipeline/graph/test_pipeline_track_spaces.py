"""A recipe whose planar step would read world-frame tracks is refused before it runs.

``run_feature`` refuses a planar feature handed a world-frame table, but a
refusal at run time alone lets a recipe pass submission and fail at its step.
The plan therefore checks each feature step's spaces too: the space of the
tracks tables it selects, and ``world_m`` for a track-shaped input of a stored
run whose footer lists ``Z``. A submission raises ``schema_family_mismatch``, a
preview records the refusal as a problem at the step's ``inputs``, and a step
whose input did not exist at submission is refused with the same reason when it
starts.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from mosaic.behavior.feature_library.roi_occupancy import RoiOccupancy
from mosaic.behavior.feature_library.trajectory_smooth import TrajectorySmooth
import mosaic.core.pipeline.track_space as track_space_module
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    Plan,
    Recipe,
    StepRefused,
    declaration_catalog,
    execute_step,
    plan_fingerprint,
    plan_pipeline,
    plan_wire,
    read_recipe,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.graph.step_preflight import track_space_refusals
from mosaic.core.pipeline.graph.store import request_ids
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.track_space import TrackSpaceRefused
from mosaic.core.pipeline.types import Result
from mosaic.core.scope import Scope
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import (
    WORLD_POINTS_FORMAT,
    add_tracks_table,
    make_dataset,
    raw_world_dataset,
    registered_world_points,
    track_table,
    world_track_table,
)

WORLD_VARIANT = "convert-world.0.1-aaaaaaaaaa"
OTHER_WORLD_VARIANT = "convert-world.0.1-bbbbbbbbbb"
PIXEL_VARIANT = "convert-pixel.0.1-cccccccccc"
CM_VARIANT = "convert-cm.0.1-dddddddddd"

PLANAR_DETAIL: dict[str, JsonValue] = {
    "feature": "pair-egocentric",
    "accepts": ["image_px", "plane_cm"],
    "found": ["world_m"],
}


PAIR_PARAMS: dict[str, JsonValue] = {
    "pose": {"pose_n": 3},
    "neck_idx": 1,
    "tail_base_idx": 2,
}
"""``pair-egocentric`` over the three keypoints of :func:`world_track_table`."""


def _single(feature: str) -> Recipe:
    params = PAIR_PARAMS if feature == "pair-egocentric" else {}
    return read_recipe(
        {
            "steps": [
                {
                    "id": "step",
                    "type": "feature",
                    "feature": feature,
                    "inputs": ["tracks"],
                    "params": params,
                }
            ]
        }
    )


def _dataset(root: Path, schema: str, *variants: str) -> Dataset:
    """Entry ``seq`` with two animals of three keypoints, in each of *variants*.

    One variant by default. A schema other than ``mosaic_world_v1`` gets the
    world table without its ``Z`` columns.
    """
    dataset = make_dataset(root)
    for variant in variants or (WORLD_VARIANT,):
        table = world_track_table("", "seq", n_rows=10, n_ids=2)
        if schema != "mosaic_world_v1":
            table = table.drop(columns=["Z", "poseZ0", "poseZ1", "poseZ2"])
        add_tracks_table(dataset, variant, table, std_format=schema)
    return dataset


@pytest.fixture
def world(tmp_path: Path) -> Dataset:
    return _dataset(tmp_path / "ds", "mosaic_world_v1")


def _space_problems(plan: Plan) -> list[tuple[str, str]]:
    return [
        (problem.step, problem.where)
        for problem in plan.problems
        if "computes in" in problem.message
    ]


# --- submission and preview ---------------------------------------------------


def test_a_submission_refuses_a_planar_step_over_world_tracks(world: Dataset) -> None:
    with pytest.raises(StepRefused) as raised:
        _ = submit_request(world, _single("pair-egocentric"))

    assert raised.value.reason == "schema_family_mismatch"
    assert raised.value.step_id == "step"
    assert raised.value.detail == PLANAR_DETAIL
    assert request_ids(world.base_dir) == []


def test_a_preview_records_the_refusal_as_a_problem_at_inputs(world: Dataset) -> None:
    plan = plan_pipeline(world, _single("pair-egocentric"), refuse=False)

    assert [(problem.step, problem.where) for problem in plan.problems] == [
        ("step", "inputs")
    ]
    assert "world_m" in plan.problems[0].message
    refusal = plan.track_space_refusals["step"]
    assert isinstance(refusal, TrackSpaceRefused)
    assert refusal.detail == PLANAR_DETAIL
    assert plan.problems[0].message == str(refusal)


def test_the_wire_and_the_fingerprint_leave_the_refusals_out(world: Dataset) -> None:
    plan = plan_pipeline(world, _single("pair-egocentric"), refuse=False)
    without = replace(plan, track_space_refusals={})

    assert plan.track_space_refusals
    assert plan_wire(plan) == plan_wire(without)
    assert plan_fingerprint(plan) == plan_fingerprint(without)
    assert plan == without


def test_a_feature_declaring_the_world_space_plans_over_world_tracks(
    world: Dataset,
) -> None:
    plan = plan_pipeline(world, _single("keypoint-kinematics"), refuse=False)

    assert plan.problems == ()
    assert plan.track_space_refusals == {}


def test_a_planar_feature_plans_over_pixel_tracks(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "ds", "mosaic_v1", PIXEL_VARIANT)

    plan = plan_pipeline(dataset, _single("pair-egocentric"), refuse=False)

    assert plan.problems == ()
    assert plan.track_space_refusals == {}


def test_a_schema_this_process_does_not_register_is_skipped(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "ds", "registered_elsewhere_v1", PIXEL_VARIANT)

    plan = plan_pipeline(dataset, _single("pair-egocentric"), refuse=False)

    assert plan.track_space_refusals == {}
    assert _space_problems(plan) == []


def test_an_ambiguous_entry_adds_no_space_problem(tmp_path: Path) -> None:
    """The identity reports the ambiguity, and the tables have no single answer."""
    dataset = _dataset(
        tmp_path / "ds", "mosaic_world_v1", WORLD_VARIANT, OTHER_WORLD_VARIANT
    )

    plan = plan_pipeline(dataset, _single("pair-egocentric"), refuse=False)

    assert plan.track_space_refusals == {}
    assert _space_problems(plan) == []
    assert plan.problems


# --- a stored run read through ``inputs`` ----------------------------------------

SMOOTHED_SPEED = read_recipe(
    {
        "steps": [
            {
                "id": "smooth",
                "type": "feature",
                "feature": "trajectory-smooth",
                "inputs": ["tracks"],
            },
            {
                "id": "speed",
                "type": "feature",
                "feature": "speed-angvel",
                "inputs": [{"step": "smooth"}],
            },
        ]
    }
)


@pytest.fixture
def pixel(tmp_path: Path) -> Dataset:
    return _dataset(tmp_path / "ds", "mosaic_v1", PIXEL_VARIANT)


def test_a_parent_with_no_run_on_disk_plans(pixel: Dataset) -> None:
    """Its run is checked when the reader starts."""
    plan = plan_pipeline(pixel, SMOOTHED_SPEED)

    assert plan.track_space_refusals == {}


def _refusals_over_stored_parent(
    dataset: Dataset, table: pd.DataFrame
) -> dict[str, TrackSpaceRefused]:
    """Write *table* as the planned run of ``smooth``, then check the plan's spaces."""
    plan = plan_pipeline(dataset, SMOOTHED_SPEED)
    smooth = plan.step("smooth")
    assert smooth.run_id is not None
    root = feature_run_root(dataset, smooth.storage_name, smooth.run_id)
    root.mkdir(parents=True)
    table.to_parquet(root / "seq.parquet")
    return track_space_refusals(
        dataset,
        SMOOTHED_SPEED,
        {step.step_id: step for step in plan.steps},
        catalog=declaration_catalog(),
        targets={step.step_id: plan.scope for step in plan.steps},
        exempt=frozenset(),
    )


def test_a_stored_parent_carrying_z_refuses_a_planar_reader(pixel: Dataset) -> None:
    refusals = _refusals_over_stored_parent(
        pixel, world_track_table("", "seq", n_rows=4)
    )

    assert list(refusals) == ["speed"]
    assert refusals["speed"].detail["found"] == ["world_m"]


def test_a_stored_parent_without_z_refuses_nothing(pixel: Dataset) -> None:
    refusals = _refusals_over_stored_parent(pixel, track_table("", "seq", n_rows=4))

    assert refusals == {}


# --- a graph that converts its own world tracks --------------------------------


CONVERTING_RECIPE: dict[str, JsonValue] = {
    "schema_version": 1,
    "name": "world",
    "steps": [
        {
            "id": "tracks",
            "type": "op",
            "kind": "convert-tracks",
            "params": {"source_format": WORLD_POINTS_FORMAT},
        },
        {
            "id": "pair",
            "type": "feature",
            "feature": "pair-egocentric",
            "inputs": ["tracks"],
            "tracks": {"step": "tracks"},
            "params": PAIR_PARAMS,
        },
    ],
}


def test_a_step_reading_tracks_its_graph_converts_is_refused_when_it_starts(
    tmp_path: Path,
) -> None:
    """The variant has no rows at submission, so only the step's own plan sees it."""
    with registered_world_points():
        dataset = raw_world_dataset(tmp_path / "ds")
        request = submit_request(dataset, read_recipe(CONVERTING_RECIPE)).request

        _ = execute_step(dataset, request, "tracks")
        with pytest.raises(StepRefused) as raised:
            _ = execute_step(dataset, request, "pair")

    assert raised.value.reason == "schema_family_mismatch"
    assert raised.value.detail == PLANAR_DETAIL
    logged = read_run(run_log_dir(dataset.base_dir), request.execution_of("pair"))
    assert logged is not None
    assert logged["status"] == "failed"
    recorded = json.loads(logged["error_json"])
    assert recorded["reason"] == "schema_family_mismatch"
    assert {key: recorded[key] for key in PLANAR_DETAIL} == PLANAR_DETAIL


# --- the tracks behind a stored track-shaped run ---------------------------------

ZONE_SHAPE: dict[str, JsonValue] = {"kind": "circle", "cx": 5.0, "cy": 5.0, "r": 3.0}
ZONE: dict[str, JsonValue] = {"name": "zone", "shape": ZONE_SHAPE}
"""An absolute disc, so ``roi-occupancy`` reads no arena layout."""


def _smoothed_occupancy(*radii: float) -> Recipe:
    """``trajectory-smooth`` over the tracks, then one ``roi-occupancy`` per radius."""
    readers: list[JsonValue] = [
        {
            "id": f"roi_{index}",
            "type": "feature",
            "feature": "roi-occupancy",
            "inputs": [{"step": "smooth"}],
            "params": {
                "rois": [{"name": "zone", "shape": {**ZONE_SHAPE, "r": radius}}]
            },
        }
        for index, radius in enumerate(radii)
    ]
    return read_recipe(
        {
            "steps": [
                {
                    "id": "smooth",
                    "type": "feature",
                    "feature": "trajectory-smooth",
                    "inputs": ["tracks"],
                },
                *readers,
            ]
        }
    )


def _with_stored_smoothing(dataset: Dataset, recipe: Recipe) -> Result:
    """Run ``trajectory-smooth`` as the recipe plans it, and return its result."""
    produced = run_feature(dataset, TrajectorySmooth())
    assert plan_pipeline(dataset, recipe, refuse=False).step("smooth").run_id == (
        produced.run_id
    )
    return Result(feature=produced.feature, run_id=produced.run_id)


def test_a_plan_refuses_a_pixel_reader_of_smoothed_centimeters_as_a_run_does(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "ds", "mosaic_cm_v1", CM_VARIANT)
    recipe = _smoothed_occupancy(3.0)
    smoothed = _with_stored_smoothing(dataset, recipe)

    plan = plan_pipeline(dataset, recipe, refuse=False)
    with pytest.raises(TrackSpaceRefused) as raised:
        _ = run_feature(
            dataset,
            RoiOccupancy(
                inputs=RoiOccupancy.Inputs((smoothed,)), params={"rois": [ZONE]}
            ),
        )

    assert list(plan.track_space_refusals) == ["roi_0"]
    refusal = plan.track_space_refusals["roi_0"]
    assert refusal.reason == "schema_family_mismatch"
    assert str(refusal) == str(raised.value)


def test_a_plan_reads_the_tracks_once_per_variant_and_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path / "ds", "mosaic_cm_v1", CM_VARIANT)
    recipe = _smoothed_occupancy(3.0, 4.0)
    _ = _with_stored_smoothing(dataset, recipe)
    calls: list[tuple[str | None, frozenset[Entry]]] = []
    read = track_space_module.tracks_table_schemas

    def counted(
        ds: Dataset, tracks_run_id: str | None, scope: Scope
    ) -> dict[tuple[str, str], str]:
        calls.append((tracks_run_id, frozenset(scope.entry_pairs or ())))
        return read(ds, tracks_run_id, scope)

    monkeypatch.setattr(track_space_module, "tracks_table_schemas", counted)

    plan = plan_pipeline(dataset, recipe, refuse=False)

    assert sorted(plan.track_space_refusals) == ["roi_0", "roi_1"]
    assert len(calls) == len(set(calls))
    assert [call for call in calls if call[0] == CM_VARIANT] == [
        (CM_VARIANT, frozenset({("", "seq")}))
    ]


POSE_CLUSTERS = read_recipe(
    {
        "steps": [
            {
                "id": "pose",
                "type": "feature",
                "feature": "pose-egocentric",
                "inputs": ["tracks"],
                "params": {"origin": [0], "heading": [1, 2]},
            },
            {
                "id": "templates",
                "type": "feature",
                "feature": "extract-templates",
                "inputs": [{"step": "pose"}],
                "params": {"n_templates": 4},
            },
            {
                "id": "kmeans",
                "type": "feature",
                "feature": "global-kmeans",
                "inputs": [{"step": "pose"}],
                "params": {"templates": {"step": "templates"}, "k": 2},
            },
        ]
    }
)


def test_a_derived_result_shows_its_reader_no_space(world: Dataset) -> None:
    """``global-kmeans`` reads embeddings, not tracks, and is not shown ``world_m``."""
    plan = plan_pipeline(world, POSE_CLUSTERS)

    assert plan.track_space_refusals == {}
    _ = run_pipeline(world, POSE_CLUSTERS)
    after = plan_pipeline(world, POSE_CLUSTERS)
    assert after.is_complete
    assert after.track_space_refusals == {}
