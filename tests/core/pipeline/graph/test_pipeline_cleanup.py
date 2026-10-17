"""Which superseded runs of a plan's storages a cleanup may remove.

``plan_cleanup`` reports every run of a planned feature storage that the plan
does not expect, removable or kept with the first rule that keeps it. It reads
and never writes.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import socket
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    SUPERSEDED_SAMPLE,
    BoundRef,
    CleanupPlan,
    CleanupPolicy,
    DeclarationCatalog,
    PlanWire,
    Recipe,
    Request,
    SupersededRun,
    declaration_catalog,
    plan_cleanup,
    plan_pipeline,
    plan_wire,
    plan_with_cleanup,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.cancel import cancel_request
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory import InventoryCache
from mosaic.core.scope import Scope
from mosaic.runlog import JsonlRunLog, new_execution_id, run_log_path
from tests.helpers import (
    add_tracks_variant,
    assert_tree_unchanged,
    make_dataset,
    register_trained_model,
    run_speed,
    speed_recipe_stepping,
    tree_snapshot,
    write_driver_lease,
)

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"

TEMPLATES = "extract-templates__from__speed-angvel__from__tracks"

LOOSE = CleanupPolicy(keep_newest=0, min_age_hours=0)
"""A policy that keeps no run for its age or its recency."""

LATER = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)
"""An instant after every request's hold window and every run's age floor."""


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def _chain(step_size: int, *, scaler: bool = False) -> Recipe:
    """``speed-angvel`` at *step_size*, ``extract-templates`` over it, and a scaler.

    The scaler reads the templates through its params and the speed through
    its inputs.
    """
    steps: list[dict[str, object]] = [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
            "params": {"step_size": step_size},
        },
        {
            "id": "templates",
            "type": "feature",
            "feature": "extract-templates",
            "inputs": [{"step": "speed"}],
            "params": {"n_templates": 4},
        },
    ]
    if scaler:
        steps.append(
            {
                "id": "scaler",
                "type": "feature",
                "feature": "global-scaler",
                "inputs": [{"step": "speed"}],
                "params": {
                    "templates": {"step": "templates", "pattern": "templates.parquet"}
                },
            }
        )
    return Recipe.model_validate({"schema_version": 1, "name": "chain", "steps": steps})


def _cleanup(
    ds: Dataset,
    recipe: Recipe,
    *,
    policy: CleanupPolicy = LOOSE,
    now: datetime.datetime | None = None,
    **kwargs: object,
) -> CleanupPlan:
    plan = plan_with_cleanup(ds, recipe, cleanup=policy, now=now, **kwargs)  # pyright: ignore[reportArgumentType]
    assert plan.cleanup is not None
    return plan.cleanup


def _reasons(cleanup: CleanupPlan) -> dict[str, str]:
    return {run.run_id: run.reason for run in cleanup.runs}


def _run_ids(ds: Dataset, recipe: Recipe) -> dict[str, str]:
    """What each step of *recipe* is called on *ds*."""
    return {
        step_id: run_id
        for step_id, run_id in plan_pipeline(ds, recipe).run_ids.items()
        if run_id is not None
    }


def _open_attempt(ds: Dataset, run_id: str) -> str:
    """Record an attempt of this process that has named *run_id* and not ended."""
    execution_id = new_execution_id()
    with JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id) as log:
        log.started(
            kind="feature", target="x", host=socket.gethostname(), pid=os.getpid()
        )
        log.set_run_id(run_id)
    return execution_id


def _write_reader(
    ds: Dataset, storage: str, run_id: str, reads: tuple[str, str]
) -> None:
    """Write the ``params.json`` of a run of *storage* that reads one run."""
    run_root = feature_run_root(ds, storage, run_id)
    run_root.mkdir(parents=True)
    document: dict[str, object] = {
        "_params": {},
        "_inputs": [{"feature": reads[0], "run_id": reads[1]}],
        "_resolved": [{"where": "inputs[0]", "feature": reads[0], "run_id": reads[1]}],
    }
    _ = (run_root / "params.json").write_text(json.dumps(document))


# --- what is a candidate ------------------------------------------------------------


def test_a_superseded_run_of_a_planned_storage_is_removable(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    kept = run_speed(dataset, 2)

    cleanup = _cleanup(dataset, speed_recipe_stepping(2))

    [run] = cleanup.runs
    assert (run.run_id, run.removable, run.reason) == (old, True, "superseded")
    assert (run.step_id, run.storage, run.superseded_by) == ("speed", STORAGE, kept)
    assert run.entries == 2
    assert run.bytes > 0
    assert cleanup.removable_bytes == run.bytes


def test_runs_the_plan_expects_are_kept_including_shared_storage_siblings(
    dataset: Dataset,
) -> None:
    first, second, third = (run_speed(dataset, size) for size in (1, 2, 3))
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "two speeds",
            "steps": [
                {
                    "id": f"speed{size}",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "params": {"step_size": size},
                }
                for size in (1, 2)
            ],
        }
    )

    cleanup = _cleanup(dataset, recipe)

    assert _reasons(cleanup) == {third: "superseded"}
    assert cleanup.runs[0].superseded_by == ", ".join(sorted((first, second)))


def test_a_step_without_a_run_id_keeps_its_storage(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    recipe = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "params": {"step_size": 2},
                    "run": {"filter_start_time": 1.0},
                }
            ],
        }
    )

    cleanup = _cleanup(dataset, recipe, refuse=False)

    assert _reasons(cleanup) == {old: "keeper_unresolved"}


def test_models_and_tracks_roots_are_never_candidates(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    _ = run_pipeline(dataset, _chain(1))
    add_tracks_variant(
        dataset, "convert-trex.0.1-2222222222", "seq_c", std_format="mosaic_v1"
    )
    model = "train-pose.0.2-abcdef0123"
    weights = dataset.get_root("models") / "train-pose" / model / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"weights")
    register_trained_model(dataset, "train-pose", model, weights)

    cleanup = _cleanup(dataset, speed_recipe_stepping(2), now=LATER)

    assert {run.storage for run in cleanup.runs} == {STORAGE}
    assert old in _reasons(cleanup)


# --- the rules that keep a run --------------------------------------------------------


@pytest.mark.parametrize("site", ["inputs", "params"])
def test_a_run_a_retained_run_reads_through_inputs_or_params_is_kept(
    dataset: Dataset, site: str
) -> None:
    _ = run_pipeline(dataset, _chain(1, scaler=True))
    first = _run_ids(dataset, _chain(1, scaler=True))

    if site == "inputs":
        cleanup = _cleanup(dataset, speed_recipe_stepping(2), now=LATER)
        read, reader = first["speed"], first["templates"]
    else:
        cleanup = _cleanup(dataset, _chain(2), now=LATER)
        read, reader = first["templates"], first["scaler"]

    [run] = [run for run in cleanup.runs if run.run_id == read]
    assert run.reason == "read_by"
    assert reader in run.detail


def test_reads_are_followed_transitively_to_a_fixpoint(dataset: Dataset) -> None:
    _ = run_pipeline(dataset, _chain(1))
    first = _run_ids(dataset, _chain(1))
    _write_reader(
        dataset, "frame-aggregate", "0.1-reader0000", (TEMPLATES, first["templates"])
    )
    _ = _open_attempt(dataset, "0.1-reader0000")

    cleanup = _cleanup(dataset, _chain(2), now=LATER)

    reasons = _reasons(cleanup)
    assert (reasons[first["templates"]], reasons[first["speed"]]) == (
        "read_by",
        "read_by",
    )
    details = {run.run_id: run.detail for run in cleanup.runs}
    assert "0.1-reader0000" in details[first["templates"]]
    assert first["templates"] in details[first["speed"]]


def test_a_run_an_open_attempt_reads_is_kept(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    _write_reader(dataset, "frame-aggregate", "0.1-reader0000", (STORAGE, old))
    _ = _open_attempt(dataset, "0.1-reader0000")

    cleanup = _cleanup(dataset, speed_recipe_stepping(2))

    assert _reasons(cleanup) == {old: "read_by"}


def test_a_reader_with_unreadable_params_keeps_its_chain(dataset: Dataset) -> None:
    _ = run_pipeline(dataset, _chain(1))
    first = _run_ids(dataset, _chain(1))
    (feature_run_root(dataset, TEMPLATES, first["templates"]) / "params.json").unlink()

    cleanup = _cleanup(dataset, speed_recipe_stepping(2), now=LATER)

    assert _reasons(cleanup) == {first["speed"]: "read_by"}


@pytest.mark.parametrize("damage", ["absent", "unreadable"])
def test_a_candidate_with_unreadable_params_is_kept(
    dataset: Dataset, damage: str
) -> None:
    old = run_speed(dataset, 1)
    params = feature_run_root(dataset, STORAGE, old) / "params.json"
    if damage == "absent":
        params.unlink()
    else:
        _ = params.write_text("[")

    cleanup = _cleanup(dataset, speed_recipe_stepping(2))

    assert _reasons(cleanup) == {old: "params_unknown"}
    assert damage in cleanup.runs[0].detail


def _costly(catalog: DeclarationCatalog, name: str) -> DeclarationCatalog:
    entries = dict(catalog.entries)
    entries[name] = dataclasses.replace(entries[name], costly_fit=True)
    return DeclarationCatalog(entries=entries)


def test_a_costly_fit_is_kept(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    catalog = _costly(declaration_catalog(), "speed-angvel")

    cleanup = _cleanup(dataset, speed_recipe_stepping(2), catalog=catalog)

    assert _reasons(cleanup) == {old: "costly_fit"}


def test_the_newest_superseded_runs_are_kept(dataset: Dataset) -> None:
    oldest = run_speed(dataset, 1)
    newer = run_speed(dataset, 2)
    _ = run_speed(dataset, 3)

    cleanup = _cleanup(
        dataset,
        speed_recipe_stepping(3),
        policy=CleanupPolicy(keep_newest=1, min_age_hours=0),
    )

    assert _reasons(cleanup) == {oldest: "superseded", newer: "newest"}
    assert [run.run_id for run in cleanup.runs] == [oldest, newer]


def test_a_run_inside_the_age_window_is_kept(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    policy = CleanupPolicy(keep_newest=0, min_age_hours=24)

    young = _cleanup(dataset, speed_recipe_stepping(2), policy=policy)
    aged = _cleanup(dataset, speed_recipe_stepping(2), policy=policy, now=LATER)

    assert _reasons(young) == {old: "young"}
    assert _reasons(aged) == {old: "superseded"}


@pytest.mark.parametrize("how", ["recorded", "bound"])
def test_a_run_a_recent_request_recorded_or_bound_is_kept(
    dataset: Dataset, how: str
) -> None:
    if how == "recorded":
        _ = run_pipeline(dataset, speed_recipe_stepping(1))
        old = _run_ids(dataset, speed_recipe_stepping(1))["speed"]
    else:
        old = run_speed(dataset, 1)
        request = submit_request(
            dataset,
            speed_recipe_stepping(5),
            bind={"speed": BoundRef(feature=STORAGE, run_id=old)},
        ).request
        _ = cancel_request(dataset, request.request_id)

    cleanup = _cleanup(dataset, speed_recipe_stepping(2))

    assert _reasons(cleanup) == {old: "held_by_request"}


@pytest.mark.parametrize(("minutes", "held"), [(30, True), (90, False)])
def test_the_request_window_has_a_floor(
    dataset: Dataset, minutes: int, held: bool
) -> None:
    """``min_age_hours: 0`` still holds a request's runs for an hour."""
    _ = run_pipeline(dataset, speed_recipe_stepping(1))
    old = _run_ids(dataset, speed_recipe_stepping(1))["speed"]
    moment = datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=minutes)

    cleanup = _cleanup(dataset, speed_recipe_stepping(2), now=moment)

    assert _reasons(cleanup) == {old: "held_by_request" if held else "superseded"}


def test_a_request_older_than_the_window_holds_nothing(dataset: Dataset) -> None:
    _ = run_pipeline(dataset, speed_recipe_stepping(1))
    old = _run_ids(dataset, speed_recipe_stepping(1))["speed"]

    cleanup = _cleanup(dataset, speed_recipe_stepping(2), now=LATER)

    assert _reasons(cleanup) == {old: "superseded"}


def test_an_orphaned_request_holds_its_runs_at_any_age(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    request = submit_request(dataset, speed_recipe_stepping(1)).request
    execution_id = request.execution_of("speed")
    with JsonlRunLog(run_log_path(dataset.base_dir, execution_id), execution_id) as log:
        log.started(kind="feature", target=STORAGE)
        log.set_run_id(old)
    _ = write_driver_lease(dataset.base_dir, request.request_id, stale=True, pid=1)

    cleanup = _cleanup(dataset, speed_recipe_stepping(2), now=LATER)

    [run] = cleanup.runs
    assert (run.reason, run.run_id) == ("held_by_request", old)
    assert "orphaned" in run.detail


def test_a_running_request_holds_every_run_of_its_slugs(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    _ = submit_request(dataset, speed_recipe_stepping(9))

    cleanup = _cleanup(dataset, speed_recipe_stepping(2), now=LATER)

    assert _reasons(cleanup) == {old: "held_by_request"}
    assert "running" in cleanup.runs[0].detail


def test_the_own_requests_binds_are_kept(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    other = run_speed(dataset, 3)
    recipe = speed_recipe_stepping(2)
    plan = plan_pipeline(dataset, recipe)
    request = Request(
        request_id="R1",
        recipe_digest=plan.recipe_digest,
        bind={"elsewhere": BoundRef(feature=STORAGE, run_id=old)},
    )

    cleanup = plan_cleanup(
        dataset,
        plan,
        policy=LOOSE,
        inventory=InventoryCache(dataset),
        catalog=declaration_catalog(),
        own_request=request,
    )

    assert _reasons(cleanup) == {old: "bound", other: "superseded"}


def test_a_run_an_unended_attempt_names_is_kept(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    execution_id = _open_attempt(dataset, old)

    cleanup = _cleanup(dataset, speed_recipe_stepping(2))

    assert _reasons(cleanup) == {old: "attempt_open"}
    assert execution_id in cleanup.runs[0].detail


def test_a_run_holding_entries_its_replacement_lacks_is_kept(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)

    cleanup = _cleanup(
        dataset, speed_recipe_stepping(2), scope=Scope(entries=[("", "seq_a")])
    )

    assert _reasons(cleanup) == {old: "wider_coverage"}
    assert "1 entries" in cleanup.runs[0].detail


def test_a_scope_dependent_run_fitted_over_other_entries_is_kept(
    dataset: Dataset,
) -> None:
    _ = run_pipeline(dataset, _chain(1))
    first = _run_ids(dataset, _chain(1))

    cleanup = _cleanup(
        dataset, _chain(1), scope=Scope(entries=[("", "seq_a")]), now=LATER
    )

    assert _reasons(cleanup) == {first["templates"]: "other_fit_scope"}


def test_a_managed_dataset_removes_nothing(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    with dataset.manifest_path.open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")
    managed = Dataset(manifest_path=dataset.manifest_path).load(ensure_roots=False)

    cleanup = _cleanup(managed, speed_recipe_stepping(2))

    assert _reasons(cleanup) == {old: "managed_dataset"}


def test_an_unreadable_request_file_removes_nothing(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    requests = dataset.base_dir / ".mosaic" / "pipelines" / "requests"
    requests.mkdir(parents=True)
    _ = (requests / "broken.json").write_text("{")

    cleanup = _cleanup(dataset, speed_recipe_stepping(2))

    assert _reasons(cleanup) == {old: "request_unreadable"}
    assert "broken" in cleanup.runs[0].detail


# --- a read ---------------------------------------------------------------------------


def test_planning_a_cleanup_writes_nothing(dataset: Dataset) -> None:
    _ = run_pipeline(dataset, _chain(1))
    _ = run_speed(dataset, 3)
    before = tree_snapshot(dataset.base_dir)

    cleanup = _cleanup(dataset, _chain(2), now=LATER)

    assert cleanup.runs
    assert_tree_unchanged(dataset.base_dir, before)


def test_a_plan_without_cleanup_has_none(dataset: Dataset) -> None:
    _ = run_speed(dataset, 1)

    plan = plan_with_cleanup(dataset, speed_recipe_stepping(2), cleanup=None)

    assert plan.cleanup is None


# --- the wire -------------------------------------------------------------------------


def test_plan_wire_round_trips_with_and_without_cleanup(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    recipe = speed_recipe_stepping(2)

    asked = plan_wire(plan_with_cleanup(dataset, recipe, cleanup=LOOSE))
    unasked = plan_wire(plan_with_cleanup(dataset, recipe, cleanup=None))

    for wire in (asked, unasked):
        assert PlanWire.model_validate_json(wire.model_dump_json()) == wire
    assert unasked.cleanup is None
    assert unasked.steps[0].superseded == ()
    assert asked.cleanup is not None
    assert (asked.cleanup.removable_runs, asked.cleanup.kept_runs) == (1, 0)
    assert [run.run_id for run in asked.steps[0].superseded] == [old]


def test_a_step_lists_at_most_the_sample_of_superseded_runs(dataset: Dataset) -> None:
    plan = plan_with_cleanup(dataset, speed_recipe_stepping(2), cleanup=None)
    runs = tuple(
        SupersededRun(
            step_id="speed",
            storage=STORAGE,
            run_id=f"0.2-{index:010d}",
            removable=True,
            reason="superseded",
            detail="",
            bytes=1,
            entries=2,
            finished_at="",
            superseded_by="",
        )
        for index in range(SUPERSEDED_SAMPLE + 3)
    )

    wire = plan_wire(
        dataclasses.replace(plan, cleanup=CleanupPlan(policy=LOOSE, runs=runs))
    )

    assert len(wire.steps[0].superseded) == SUPERSEDED_SAMPLE
    assert wire.steps[0].superseded_omitted == 3
    assert wire.cleanup is not None
    assert wire.cleanup.removable_runs == SUPERSEDED_SAMPLE + 3
