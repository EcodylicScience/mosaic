"""Which superseded runs of a plan's storages a cleanup may remove.

``plan_cleanup`` reports every run of a planned feature storage that the plan
does not expect, removable or kept with the first rule that keeps it. It reads
and never writes.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
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
    RetireNeedsCleanup,
    RetireOwnRecipe,
    RetireUnknownRecipe,
    SupersededRun,
    declaration_catalog,
    plan_cleanup,
    plan_pipeline,
    plan_wire,
    plan_with_cleanup,
    recipe_path,
    run_pipeline,
    save_request,
    storage_name_of,
    submit_request,
)
from mosaic.core.pipeline.storage_names import SHORTENED_MARKER
from mosaic.core.pipeline.cancel import cancel_request
from mosaic.core.pipeline.graph.digest import document_digest, recipe_digest
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.inventory import InventoryCache
from mosaic.core.scope import Scope
from mosaic.runlog import JsonlRunLog, run_log_path
from tests.helpers import (
    add_tracks_variant,
    assert_tree_unchanged,
    make_dataset,
    record_open_attempt,
    register_trained_model,
    run_speed,
    speed_heading_recipe,
    speed_recipe_stepping,
    tree_snapshot,
    write_driver_lease,
    write_reader_params,
)

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"

TEMPLATES = "extract-templates__from__speed-angvel__from__tracks"

HEADING = "heading__from__tracks"

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


def _named(recipe: Recipe, name: str) -> Recipe:
    """*recipe* under *name*."""
    return Recipe.model_validate({**recipe.model_dump(mode="json"), "name": name})


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
        cleanup = _cleanup(
            dataset, _named(speed_recipe_stepping(2), "chain"), now=LATER
        )
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
    write_reader_params(
        dataset, "frame-aggregate", "0.1-reader0000", (TEMPLATES, first["templates"])
    )
    _ = record_open_attempt(dataset, "0.1-reader0000")

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
    write_reader_params(dataset, "frame-aggregate", "0.1-reader0000", (STORAGE, old))
    _ = record_open_attempt(dataset, "0.1-reader0000")

    cleanup = _cleanup(dataset, speed_recipe_stepping(2))

    assert _reasons(cleanup) == {old: "read_by"}


def test_a_reader_with_unreadable_params_keeps_its_chain(dataset: Dataset) -> None:
    _ = run_pipeline(dataset, _chain(1))
    first = _run_ids(dataset, _chain(1))
    (feature_run_root(dataset, TEMPLATES, first["templates"]) / "params.json").unlink()

    cleanup = _cleanup(dataset, _named(speed_recipe_stepping(2), "chain"), now=LATER)

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


@pytest.mark.parametrize(
    ("recorded_by", "planned", "reason"),
    [
        ("other", "speed", "recipe_uses"),
        ("speed", "speed", "superseded"),
        ("", "", "recipe_uses"),
    ],
)
def test_a_run_another_recipe_recorded_is_kept_after_its_hold(
    dataset: Dataset, recorded_by: str, planned: str, reason: str
) -> None:
    """Recipes of one non-empty name are versions of each other, unnamed ones not."""
    ran = run_pipeline(dataset, _named(speed_recipe_stepping(1), recorded_by))
    old = _run_ids(dataset, speed_recipe_stepping(1))["speed"]

    cleanup = _cleanup(dataset, _named(speed_recipe_stepping(2), planned), now=LATER)

    [run] = cleanup.runs
    assert (run.run_id, run.reason) == (old, reason)
    if reason == "recipe_uses":
        assert ran.request_id in run.detail
        assert (recorded_by or ran.recipe_digest) in run.detail


def test_a_run_another_recipe_binds_is_kept_after_its_hold(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    request = submit_request(
        dataset,
        _named(speed_recipe_stepping(5), "other"),
        bind={"speed": BoundRef(feature=STORAGE, run_id=old)},
    ).request
    _ = cancel_request(dataset, request.request_id)

    cleanup = _cleanup(dataset, speed_recipe_stepping(2), now=LATER)

    assert _reasons(cleanup) == {old: "recipe_uses"}


def test_a_request_saved_under_an_earlier_canonical_form_is_the_same_recipe(
    dataset: Dataset,
) -> None:
    """The legacy digest of the planned recipe, unchanged, names no other recipe."""
    old = run_speed(dataset, 1)
    unnamed = _named(speed_recipe_stepping(2), "")
    request = submit_request(
        dataset, unnamed, bind={"speed": BoundRef(feature=STORAGE, run_id=old)}
    ).request
    _ = cancel_request(dataset, request.request_id)
    legacy = unnamed.model_dump(mode="json")
    legacy["steps"][0]["params"]["step_size"] = 2.0
    legacy_digest = document_digest(legacy)
    _ = recipe_path(dataset.base_dir, legacy_digest).write_text(json.dumps(legacy))
    _ = save_request(
        dataset.base_dir, request.model_copy(update={"recipe_digest": legacy_digest})
    )

    cleanup = _cleanup(dataset, unnamed, now=LATER)

    assert legacy_digest != request.recipe_digest
    assert _reasons(cleanup) == {old: "superseded"}


@pytest.mark.parametrize("fault", ["missing", "a directory"])
def test_an_ended_request_whose_recipe_does_not_read_keeps_its_runs(
    dataset: Dataset, fault: str
) -> None:
    ran = run_pipeline(dataset, speed_recipe_stepping(1))
    old = _run_ids(dataset, speed_recipe_stepping(1))["speed"]
    saved = recipe_path(dataset.base_dir, ran.recipe_digest)
    saved.unlink()
    if fault == "a directory":
        saved.mkdir()

    cleanup = _cleanup(dataset, speed_recipe_stepping(2), now=LATER)

    [run] = cleanup.runs
    assert (run.run_id, run.reason) == (old, "recipe_uses")
    assert "does not read" in run.detail


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
        recipe_name=recipe.name,
        own_request=request,
    )

    assert _reasons(cleanup) == {old: "bound", other: "superseded"}


def test_a_run_an_unended_attempt_names_is_kept(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    execution_id = record_open_attempt(dataset, old)

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


# --- what the plan itself will read ----------------------------------------------------


def _pinning_scaler(templates_run: str) -> Recipe:
    """``_chain(2)`` with a scaler whose params pin *templates_run* by name."""
    recipe = _chain(2)
    document = recipe.model_dump(mode="json")
    document["steps"].append(
        {
            "id": "scaler",
            "type": "feature",
            "feature": "global-scaler",
            "inputs": [{"step": "speed"}],
            "params": {
                "templates": {
                    "feature": TEMPLATES,
                    "run_id": templates_run,
                    "pattern": "templates.parquet",
                }
            },
        }
    )
    return Recipe.model_validate(document)


def test_a_run_a_planned_step_pins_by_name_is_kept(dataset: Dataset) -> None:
    _ = run_pipeline(dataset, _chain(1))
    first = _run_ids(dataset, _chain(1))

    cleanup = _cleanup(dataset, _pinning_scaler(first["templates"]), now=LATER)

    reasons = _reasons(cleanup)
    assert (reasons[first["templates"]], reasons[first["speed"]]) == (
        "read_by",
        "read_by",
    )
    details = {run.run_id: run.detail for run in cleanup.runs}
    assert "scaler" in details[first["templates"]]


def test_a_run_a_running_requests_step_pins_by_name_is_held(dataset: Dataset) -> None:
    _ = run_pipeline(dataset, _chain(1))
    first = _run_ids(dataset, _chain(1))
    pinning = _pinning_scaler(first["templates"]).model_dump(mode="json")
    pinning["steps"] = [
        step for step in pinning["steps"] if step["feature"] != "extract-templates"
    ]
    _ = submit_request(dataset, Recipe.model_validate(pinning))

    cleanup = _cleanup(dataset, _chain(3), now=LATER)

    assert _reasons(cleanup)[first["templates"]] == "held_by_request"


def test_a_feature_the_catalog_does_not_know_is_kept(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    catalog = declaration_catalog()
    entries = {
        name: entry for name, entry in catalog.entries.items() if name != "speed-angvel"
    }
    plan = plan_pipeline(dataset, speed_recipe_stepping(2))

    cleanup = plan_cleanup(
        dataset,
        plan,
        policy=LOOSE,
        inventory=InventoryCache(dataset),
        catalog=DeclarationCatalog(entries=entries),
        recipe_name="speed",
    )

    assert _reasons(cleanup) == {old: "costly_fit"}


def test_a_run_an_attempt_that_has_not_named_its_run_may_read_is_kept(
    dataset: Dataset,
) -> None:
    """An attempt resolving its inputs reads its input storages, run unknown."""
    old = run_speed(dataset, 1)
    execution_id = record_open_attempt(
        dataset, "", target=f"frame-aggregate__from__{STORAGE}"
    )

    cleanup = _cleanup(dataset, speed_recipe_stepping(2))

    assert _reasons(cleanup) == {old: "read_by"}
    assert execution_id in cleanup.runs[0].detail


@pytest.mark.parametrize("unknown", ["unnamed", "unreadable"])
def test_an_abbreviated_reader_of_unknown_params_keeps_its_upstream_runs(
    dataset: Dataset, unknown: str
) -> None:
    """An abbreviated storage name does not list its inputs and reads every run.

    The reader's name is cut before ``speed-angvel__from__tracks``. The storage
    names that its name contains omit the run it reads. Its feature takes a run
    through none of its params fields, because such a field reads every run
    by itself.
    """
    old = run_speed(dataset, 1)
    stacked = storage_name_of("temporal-stack", ["tracks"] * 30)
    reader = storage_name_of("speed-angvel", [stacked, STORAGE])
    assert SHORTENED_MARKER in reader
    assert STORAGE not in reader
    if unknown == "unnamed":
        _ = record_open_attempt(dataset, "", target=reader)
    else:
        write_reader_params(dataset, reader, "0.1-reader0000", (STORAGE, old))
        _ = (
            feature_run_root(dataset, reader, "0.1-reader0000") / "params.json"
        ).write_text("[")
        _ = record_open_attempt(dataset, "0.1-reader0000")

    cleanup = _cleanup(dataset, speed_recipe_stepping(2))

    assert _reasons(cleanup) == {old: "read_by"}


def test_a_running_request_whose_recipe_does_not_read_removes_nothing(
    dataset: Dataset,
) -> None:
    old = run_speed(dataset, 1)
    request = submit_request(dataset, speed_recipe_stepping(9)).request
    recipe_path(dataset.base_dir, request.recipe_digest).unlink()

    cleanup = _cleanup(dataset, speed_recipe_stepping(2), now=LATER)

    assert _reasons(cleanup) == {old: "request_unreadable"}
    assert request.request_id in cleanup.runs[0].detail


# --- retiring a recipe ----------------------------------------------------------------
#
# The retired recipe runs ``speed-angvel`` (the storage the planned recipe names) and
# ``heading`` (a storage it does not name), under the name "old". The planned recipe,
# "new", runs ``speed-angvel`` at another step size.


def _old(name: str = "old") -> Recipe:
    return speed_heading_recipe(name)


def _new(name: str = "new") -> Recipe:
    return _named(speed_recipe_stepping(2), name)


def _run_heading(dataset: Dataset, **params: object) -> str:
    """Run ``heading`` directly, outside any request, and return its run id."""
    from mosaic.behavior.feature_library.heading import HeadingFeature
    from mosaic.core.pipeline.run import run_feature

    run_id = run_feature(dataset, HeadingFeature(params=dict(params))).run_id
    assert run_id is not None
    return run_id


def _by_run(cleanup: CleanupPlan) -> dict[str, SupersededRun]:
    return {run.run_id: run for run in cleanup.runs}


def _heading_of(ds: Dataset, recipe: Recipe) -> str:
    return _run_ids(ds, recipe)["heading"]


def test_a_run_in_a_storage_the_recipe_no_longer_names_is_not_a_candidate(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())
    speed = _run_ids(dataset, _old())["speed"]

    cleanup = _cleanup(dataset, _new(), now=LATER)

    assert {run.storage for run in cleanup.runs} == {STORAGE}
    assert _reasons(cleanup) == {speed: "recipe_uses"}
    assert cleanup.retired == ()
    assert ran.recipe_digest


def test_retiring_the_recipe_that_wrote_a_run_makes_it_removable(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[ran.recipe_digest])

    runs = _by_run(cleanup)
    heading = runs[ids["heading"]]
    assert (heading.storage, heading.step_id, heading.superseded_by) == (
        HEADING,
        "",
        "",
    )
    assert (heading.removable, heading.reason) == (True, "superseded")
    assert heading.retired_by == ran.recipe_digest
    assert heading.bytes > 0
    speed = runs[ids["speed"]]
    assert (speed.removable, speed.retired_by) == (True, ran.recipe_digest)
    assert speed.step_id == "speed"
    assert cleanup.retired == (ran.recipe_digest,)


def test_a_retired_run_is_kept_inside_its_requests_hold(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())

    cleanup = _cleanup(dataset, _new(), retire=[ran.recipe_digest])

    runs = _by_run(cleanup)
    assert runs[ids["heading"]].reason == "held_by_request"
    assert ran.request_id in runs[ids["heading"]].detail
    assert runs[ids["heading"]].retired_by == ran.recipe_digest
    assert cleanup.removable == ()


def test_a_retired_run_a_kept_run_reads_is_kept(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    heading = _heading_of(dataset, _old())
    write_reader_params(
        dataset, "frame-aggregate", "0.1-reader0000", (HEADING, heading)
    )
    _ = record_open_attempt(dataset, "0.1-reader0000")

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[ran.recipe_digest])

    run = _by_run(cleanup)[heading]
    assert run.reason == "read_by"
    assert "0.1-reader0000" in run.detail


def test_a_retired_run_of_a_costly_feature_is_kept(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    catalog = _costly(declaration_catalog(), "heading")

    cleanup = _cleanup(
        dataset, _new(), now=LATER, retire=[ran.recipe_digest], catalog=catalog
    )

    assert _by_run(cleanup)[_heading_of(dataset, _old())].reason == "costly_fit"


def test_a_retired_run_with_unreadable_params_is_kept(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    heading = _heading_of(dataset, _old())
    _ = (feature_run_root(dataset, HEADING, heading) / "params.json").write_text("[")

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[ran.recipe_digest])

    assert _by_run(cleanup)[heading].reason == "params_unknown"


def test_a_retired_run_in_an_abbreviated_storage_is_read_by_a_reader_of_unknown_params(
    dataset: Dataset,
) -> None:
    """The reader's name is cut before the retired run's storage, so it reads all."""
    ran = run_pipeline(dataset, _old())
    stacked = storage_name_of("temporal-stack", ["tracks"] * 30)
    reader = storage_name_of("speed-angvel", [stacked, HEADING])
    assert SHORTENED_MARKER in reader
    assert HEADING not in reader
    _ = record_open_attempt(dataset, "", target=reader)

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[ran.recipe_digest])

    assert _by_run(cleanup)[_heading_of(dataset, _old())].reason == "read_by"


def test_the_newest_retired_run_of_a_storage_follows_the_policy(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())
    heading = _heading_of(dataset, _old())

    cleanup = _cleanup(
        dataset,
        _new(),
        policy=CleanupPolicy(keep_newest=1, min_age_hours=0),
        now=LATER,
        retire=[ran.recipe_digest],
    )

    assert _by_run(cleanup)[heading].reason == "newest"


def test_a_run_no_request_wrote_is_never_a_retired_candidate(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    outside = _run_heading(dataset, output_col="elsewhere")

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[ran.recipe_digest])

    assert outside not in _by_run(cleanup)
    assert _heading_of(dataset, _old()) in _by_run(cleanup)


def test_a_run_the_retired_request_only_found_done_is_not_a_candidate(
    dataset: Dataset,
) -> None:
    found = _run_heading(dataset)
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[ran.recipe_digest])

    assert found == ids["heading"]
    assert found not in _by_run(cleanup)
    assert ids["speed"] in _by_run(cleanup)


def test_a_run_a_notebook_used_after_the_retired_request_wrote_it_is_not_a_candidate(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())
    used = _run_heading(dataset)
    assert used == _heading_of(dataset, _old())

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[ran.recipe_digest])

    assert used not in _by_run(cleanup)


def test_a_run_the_retired_request_found_in_a_named_storage_stays_recipe_uses(
    dataset: Dataset,
) -> None:
    found = run_speed(dataset, 1)
    ran = run_pipeline(dataset, _old())
    assert _run_ids(dataset, _old())["speed"] == found

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[ran.recipe_digest])

    run = _by_run(cleanup)[found]
    assert (run.reason, run.retired_by) == ("recipe_uses", "")


def test_a_run_the_retired_recipe_bound_stays_recipe_uses(dataset: Dataset) -> None:
    pinned = run_speed(dataset, 1)
    unwritten = submit_request(
        dataset,
        _old(),
        bind={"speed": BoundRef(feature=STORAGE, run_id=pinned)},
    ).request
    _ = cancel_request(dataset, unwritten.request_id)

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[unwritten.recipe_digest])

    assert _reasons(cleanup) == {pinned: "recipe_uses"}


def test_a_retired_run_another_recipe_also_recorded_stays_recipe_uses(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())
    other = run_pipeline(dataset, _old("other"))
    heading = _heading_of(dataset, _old())

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[ran.recipe_digest])

    run = _by_run(cleanup)[heading]
    assert (run.reason, run.retired_by) == ("recipe_uses", ran.recipe_digest)
    assert other.request_id in run.detail


def test_retiring_a_same_named_earlier_version_reaches_its_other_storages(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old("evolving"))
    heading = _heading_of(dataset, _old())

    plain = _cleanup(dataset, _new("evolving"), now=LATER)
    retired = _cleanup(dataset, _new("evolving"), now=LATER, retire=[ran.recipe_digest])

    assert heading not in _by_run(plain)
    assert _by_run(retired)[heading].removable


def test_retiring_with_no_candidate_in_a_named_storage_still_finds_the_runs(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())

    cleanup = _cleanup(
        dataset,
        _named(speed_recipe_stepping(7), "new"),
        now=LATER,
        retire=[ran.recipe_digest],
    )

    assert _heading_of(dataset, _old()) in _by_run(cleanup)


def test_a_recipe_run_again_after_the_cleaning_request_keeps_its_runs(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())
    heading = _heading_of(dataset, _old())
    cleaning = Request(
        request_id="R1",
        recipe_digest=recipe_digest(_new()),
        created_at="2000-01-01T00:00:00Z",
    )

    earlier = _cleanup(
        dataset, _new(), now=LATER, retire=[ran.recipe_digest], request=cleaning
    )
    later = _cleanup(
        dataset,
        _new(),
        now=LATER,
        retire=[ran.recipe_digest],
        request=cleaning.model_copy(update={"created_at": "2999-01-01T00:00:00Z"}),
    )

    assert heading not in _by_run(earlier)
    assert _by_run(later)[heading].removable


def test_retiring_the_calms21_shape_reclaims_only_the_abandoned_run(
    dataset: Dataset,
) -> None:
    """Round 2 stacked into a storage round 2b no longer reads, and round 2b reads
    what round 2 also read: only the stack is reclaimed."""
    round2 = run_pipeline(dataset, _named(_chain(1, scaler=True), "round2"))
    ids = _run_ids(dataset, _chain(1, scaler=True))
    round2b = _named(speed_recipe_stepping(1), "round2b")
    ids_b = _run_ids(dataset, round2b)

    cleanup = _cleanup(dataset, round2b, now=LATER, retire=[round2.recipe_digest])

    assert ids_b["speed"] == ids["speed"]
    removable = {run.storage for run in cleanup.removable}
    assert removable == {TEMPLATES, "global-scaler__from__speed-angvel__from__tracks"}
    assert ids["speed"] not in _by_run(cleanup)


@pytest.mark.parametrize("kind", ["unknown", "own", "without a cleanup"])
def test_a_retirement_that_cannot_stand_is_refused_while_planning(
    dataset: Dataset, kind: str
) -> None:
    ran = run_pipeline(dataset, _old())
    if kind == "unknown":
        with pytest.raises(RetireUnknownRecipe):
            _ = _cleanup(dataset, _new(), retire=["0123456789abcdef"])
    elif kind == "own":
        with pytest.raises(RetireOwnRecipe):
            _ = _cleanup(dataset, _old(), retire=[ran.recipe_digest])
    else:
        with pytest.raises(RetireNeedsCleanup):
            _ = plan_with_cleanup(
                dataset, _new(), cleanup=None, retire=[ran.recipe_digest]
            )


def test_planning_a_retirement_writes_nothing(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    before = tree_snapshot(dataset.base_dir)

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[ran.recipe_digest])

    assert cleanup.removable
    assert_tree_unchanged(dataset.base_dir, before)


@pytest.mark.parametrize(("pid", "candidate"), [(0, False), (4242, True)])
def test_only_an_attempt_that_ran_in_a_process_wrote_the_run_it_names(
    dataset: Dataset, pid: int, candidate: bool
) -> None:
    """A stalled or cached step records the run it found, with no process."""
    from mosaic.behavior.feature_library.heading import HeadingFeature
    from mosaic.core.pipeline.run import run_feature

    outside = run_feature(dataset, HeadingFeature(), track=False).run_id
    assert outside is not None
    request = submit_request(dataset, _old()).request
    for step_id, target, run_id in (
        ("speed", STORAGE, _run_ids(dataset, _old())["speed"]),
        ("heading", HEADING, outside),
    ):
        execution_id = request.execution_of(step_id)
        with JsonlRunLog(
            run_log_path(dataset.base_dir, execution_id), execution_id
        ) as log:
            log.started(kind="feature", target=target, host="somewhere", pid=pid)
            log.set_run_id(run_id)
            log.entries_written(2)
            log.finished()

    cleanup = _cleanup(dataset, _new(), now=LATER, retire=[request.recipe_digest])

    assert (outside in _by_run(cleanup)) is candidate
