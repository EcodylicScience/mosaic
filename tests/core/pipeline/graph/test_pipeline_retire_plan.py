"""Planning a retirement without a submission.

``plan_retirement`` reads the digests to retire, sets aside the ones it cannot
retire, plans each pipeline that stays, and judges the runs the retired recipes'
requests wrote. It refuses nothing, takes no lock and writes nothing.

The retired recipe ``old`` runs ``speed-angvel`` at step size 1 and ``heading``.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.cancel import cancel_request
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.graph import (
    CleanupPolicy,
    CleanupTarget,
    Recipe,
    RetirementKeeper,
    RetirementPlan,
    RetirementPlanWire,
    RetirementSpec,
    SupersededRun,
    plan_pipeline,
    plan_retirement,
    recipe_digest,
    recipe_path,
    retirement_fingerprint,
    retirement_wire,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.graph.digest import document_digest
from mosaic.core.scope import Scope
from tests.helpers import (
    CANARY_NAME,
    CANARY_SOURCE,
    CANARY_VARIABLE,
    add_tracks_variant,
    assert_tree_unchanged,
    chain_recipe,
    forget_imported_definitions,
    imported_definitions,
    make_dataset,
    named_recipe,
    record_open_attempt,
    run_speed,
    speed_heading_recipe,
    speed_recipe_stepping,
    tree_snapshot,
    write_reader_params,
)

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"

HEADING = "heading__from__tracks"

TEMPLATES = "extract-templates__from__speed-angvel__from__tracks"

LOOSE = CleanupPolicy(keep_newest=0, min_age_hours=0)
"""The policy mosaic-api plans a cleanup under."""

LATER = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=30)
"""An instant after every request's hold window and every run's age floor."""


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def _old(name: str = "old") -> Recipe:
    return speed_heading_recipe(name)


def _keeper(recipe: Recipe, scope: Scope | None = None) -> RetirementKeeper:
    return RetirementKeeper(recipe=recipe, scope=scope or Scope())


def _run_ids(ds: Dataset, recipe: Recipe) -> dict[str, str]:
    return {
        step_id: run_id
        for step_id, run_id in plan_pipeline(ds, recipe).run_ids.items()
        if run_id is not None
    }


def _plan(
    ds: Dataset,
    retire: list[str],
    keepers: list[RetirementKeeper] | None = None,
    *,
    policy: CleanupPolicy = LOOSE,
    manager: str = "",
    now: datetime.datetime = LATER,
) -> RetirementPlan:
    return plan_retirement(ds, retire, keepers or [], policy, manager, now=now)


def _by_run(plan: RetirementPlan) -> dict[str, SupersededRun]:
    return {run.run_id: run for run in plan.cleanup.runs}


def _reasons(plan: RetirementPlan) -> dict[str, str]:
    return {run.run_id: run.reason for run in plan.cleanup.runs}


# --- the digests ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("malformed", "lowercase hex"),
        ("missing", "no saved recipe"),
        ("unreadable", "does not read"),
    ],
)
def test_a_digest_that_does_not_read_is_listed_and_the_others_are_judged(
    dataset: Dataset, kind: str, message: str
) -> None:
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())
    other = run_pipeline(dataset, named_recipe(speed_recipe_stepping(3), "other"))
    bad = {"malformed": "../escape", "missing": "f" * 16}.get(kind, other.recipe_digest)
    if kind == "unreadable":
        _ = recipe_path(dataset.base_dir, other.recipe_digest).write_text("{not json")

    plan = _plan(dataset, [bad, ran.recipe_digest])

    [refused] = plan.not_retired
    assert (refused.digest, refused.reason) == (bad, "recipe_unreadable")
    assert message in refused.message
    assert plan.retired == (ran.recipe_digest,)
    assert set(_reasons(plan)) == {ids["speed"], ids["heading"]}
    assert all(run.removable for run in plan.cleanup.runs)


@pytest.mark.parametrize("form", ["current", "earlier"])
def test_the_recipe_a_keeper_runs_is_not_retired_and_its_runs_stay(
    dataset: Dataset, form: str
) -> None:
    """Two pipelines share ``old``: retiring it for one leaves the other's runs."""
    shared = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())
    other = run_pipeline(dataset, named_recipe(speed_recipe_stepping(3), "other"))
    given = shared.recipe_digest
    if form == "earlier":
        legacy = _old().model_dump(mode="json")
        legacy["steps"][0]["params"]["step_size"] = 1.0
        given = document_digest(legacy)
        _ = recipe_path(dataset.base_dir, given).write_text(json.dumps(legacy))

    plan = _plan(dataset, [given, other.recipe_digest], [_keeper(_old())])

    [refused] = plan.not_retired
    assert (refused.digest, refused.reason) == (given, "keeper_recipe")
    assert plan.retired == (other.recipe_digest,)
    runs = _by_run(plan)
    assert ids["speed"] not in runs
    assert ids["heading"] not in runs
    [retired] = runs.values()
    assert (retired.reason, retired.retired_by) == ("superseded", other.recipe_digest)


def test_a_digest_in_an_earlier_canonical_form_retires_the_current_recipe(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())
    legacy = _old().model_dump(mode="json")
    legacy["steps"][0]["params"]["step_size"] = 1.0
    given = document_digest(legacy)
    _ = recipe_path(dataset.base_dir, given).write_text(json.dumps(legacy))

    plan = _plan(dataset, [given])

    assert given != ran.recipe_digest
    assert plan.retired == (ran.recipe_digest,)
    assert _reasons(plan) == {ids["speed"]: "superseded", ids["heading"]: "superseded"}


# --- what is removable ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("after", "reason"),
    [(datetime.timedelta(minutes=30), "held_by_request"), (None, "superseded")],
)
def test_the_runs_a_retired_recipe_wrote_are_removable_after_the_hold_floor(
    dataset: Dataset, after: datetime.timedelta | None, reason: str
) -> None:
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())
    moment = LATER if after is None else datetime.datetime.now(datetime.UTC) + after

    plan = _plan(dataset, [ran.recipe_digest], now=moment)

    assert _reasons(plan) == {ids["speed"]: reason, ids["heading"]: reason}
    assert {run.retired_by for run in plan.cleanup.runs} == {ran.recipe_digest}
    assert plan.removable == plan.cleanup.removable
    assert plan.kept == plan.cleanup.kept


def test_a_run_a_keeper_expects_is_no_candidate(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())
    outside = run_speed(dataset, 3)
    keeper = named_recipe(speed_recipe_stepping(1), "keeper")

    plan = _plan(dataset, [ran.recipe_digest], [_keeper(keeper)])

    runs = _by_run(plan)
    assert ids["speed"] not in runs
    assert outside not in runs
    heading = runs[ids["heading"]]
    assert (heading.storage, heading.step_id, heading.superseded_by) == (
        HEADING,
        "",
        "",
    )
    assert heading.removable


def test_a_retired_run_with_entries_a_keeper_lacks_is_kept(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())
    keeper = named_recipe(speed_recipe_stepping(2), "keeper")

    plan = _plan(
        dataset,
        [ran.recipe_digest],
        [_keeper(keeper, Scope(entries=[("", "seq_a")]))],
    )

    speed = _by_run(plan)[ids["speed"]]
    assert (speed.storage, speed.step_id, speed.reason) == (
        STORAGE,
        "speed",
        "wider_coverage",
    )


def test_a_run_a_recipe_not_retired_recorded_is_kept(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    user = run_pipeline(dataset, _old("user"))
    ids = _run_ids(dataset, _old())

    plan = _plan(dataset, [ran.recipe_digest])

    heading = _by_run(plan)[ids["heading"]]
    assert (heading.reason, heading.retired_by) == ("recipe_uses", ran.recipe_digest)
    assert user.request_id in heading.detail


def test_a_run_a_kept_run_reads_is_kept(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    heading = _run_ids(dataset, _old())["heading"]
    write_reader_params(
        dataset, "frame-aggregate", "0.1-reader0000", (HEADING, heading)
    )
    _ = record_open_attempt(dataset, "0.1-reader0000", target="frame-aggregate")

    plan = _plan(dataset, [ran.recipe_digest])

    run = _by_run(plan)[heading]
    assert run.reason == "read_by"
    assert "0.1-reader0000" in run.detail


def _pinning_keeper(templates_run: str) -> Recipe:
    """``speed-angvel`` at step size 2 and a scaler whose params name *templates_run*."""
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "pinning keeper",
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "params": {"step_size": 2},
                },
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
                },
            ],
        }
    )


def test_a_run_a_keepers_step_reads_names_that_keeper(dataset: Dataset) -> None:
    chain = named_recipe(chain_recipe(), "old chain")
    ran = run_pipeline(dataset, chain)
    templates = _run_ids(dataset, chain)["templates"]
    pinning = _pinning_keeper(templates)
    other = named_recipe(speed_recipe_stepping(5), "other keeper")

    alone = _plan(dataset, [ran.recipe_digest], [_keeper(pinning)])
    beside = _plan(dataset, [ran.recipe_digest], [_keeper(other), _keeper(pinning)])

    assert _by_run(alone)[templates].detail == "step 'scaler' of the plan reads it"
    run = _by_run(beside)[templates]
    assert run.reason == "read_by"
    assert run.detail == (
        f"step 'scaler' of the plan of recipe {recipe_digest(pinning)} reads it"
    )


def test_a_retired_recipe_with_a_running_request_is_planned_and_its_runs_kept(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())
    running = submit_request(dataset, _old()).request

    plan = _plan(dataset, [ran.recipe_digest])

    assert plan.retired == (ran.recipe_digest,)
    assert _reasons(plan) == {
        ids["speed"]: "held_by_request",
        ids["heading"]: "held_by_request",
    }
    assert all(running.request_id in run.detail for run in plan.cleanup.runs)


@pytest.mark.parametrize(
    ("manager", "reason"),
    [
        ("mosaic-api", "superseded"),
        ("someone-else", "managed_dataset"),
        ("", "managed_dataset"),
    ],
)
def test_a_managed_dataset_keeps_every_run_unless_its_manager_plans(
    dataset: Dataset, manager: str, reason: str
) -> None:
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())
    managed = _managed(dataset)

    plan = _plan(managed, [ran.recipe_digest], manager=manager)

    assert _reasons(plan) == {ids["speed"]: reason, ids["heading"]: reason}


def _managed(dataset: Dataset) -> Dataset:
    with dataset.manifest_path.open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")
    return Dataset(manifest_path=dataset.manifest_path).load(ensure_roots=False)


# --- the fingerprint ------------------------------------------------------------------


def test_the_fingerprint_is_stable_and_ignores_the_instant(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    keeper = _keeper(named_recipe(speed_recipe_stepping(2), "keeper"))

    first = _plan(dataset, [ran.recipe_digest], [keeper])
    second = _plan(
        dataset, [ran.recipe_digest], [keeper], now=LATER + datetime.timedelta(days=1)
    )

    assert first.planned_at != second.planned_at
    assert first.fingerprint == second.fingerprint
    assert first.planned_at == LATER.isoformat()


def test_the_fingerprint_moves_with_a_new_retired_run(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    later = named_recipe(speed_recipe_stepping(3), "later")
    waiting = submit_request(dataset, later).request
    _ = cancel_request(dataset, waiting.request_id)
    retire = [ran.recipe_digest, waiting.recipe_digest]

    before = _plan(dataset, retire)
    _ = run_pipeline(dataset, later)
    after = _plan(dataset, retire)

    assert len(after.removable) == len(before.removable) + 1
    assert after.fingerprint != before.fingerprint


def test_the_fingerprint_moves_with_a_digest_not_retired(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())

    alone = _plan(dataset, [ran.recipe_digest])
    beside = _plan(dataset, [ran.recipe_digest, "f" * 16])

    assert alone.removable == beside.removable
    assert alone.fingerprint != beside.fingerprint


def test_the_fingerprint_moves_with_the_policy(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())

    loose = _plan(dataset, [ran.recipe_digest])
    newest = _plan(
        dataset,
        [ran.recipe_digest],
        policy=CleanupPolicy(keep_newest=5, min_age_hours=0),
    )

    assert loose.fingerprint != newest.fingerprint


def test_the_fingerprint_moves_with_a_policy_that_removes_the_same_runs(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())

    loose = _plan(dataset, [ran.recipe_digest])
    aged = _plan(
        dataset,
        [ran.recipe_digest],
        policy=CleanupPolicy(keep_newest=0, min_age_hours=1),
    )

    assert loose.removable == aged.removable
    assert loose.fingerprint != aged.fingerprint


def test_the_fingerprint_moves_with_the_dataset(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    plan = _plan(dataset, [ran.recipe_digest])

    def of(dataset_uuid: str) -> str:
        return retirement_fingerprint(
            dataset_uuid=dataset_uuid,
            policy=plan.policy,
            retired=plan.retired,
            not_retired=plan.not_retired,
            plans=plan.plans,
            cleanup=plan.cleanup,
        )

    own, other = of(plan.dataset_uuid), of("another-dataset")

    assert plan.dataset_uuid
    assert own == plan.fingerprint
    assert other != own


def test_the_fingerprint_moves_with_a_keepers_plan(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    keeper = _keeper(named_recipe(speed_recipe_stepping(2), "keeper"))

    before = _plan(dataset, [ran.recipe_digest], [keeper])
    _ = run_speed(dataset, 2)
    after = _plan(dataset, [ran.recipe_digest], [keeper])

    assert before.removable == after.removable
    assert before.fingerprint != after.fingerprint


def test_the_fingerprint_moves_when_a_run_ages_past_the_policy(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())
    policy = CleanupPolicy(keep_newest=0, min_age_hours=2)
    start = datetime.datetime.now(datetime.UTC)

    young = _plan(
        dataset,
        [ran.recipe_digest],
        policy=policy,
        now=start + datetime.timedelta(hours=1),
    )
    aged = _plan(
        dataset,
        [ran.recipe_digest],
        policy=policy,
        now=start + datetime.timedelta(hours=3),
    )

    assert not young.removable
    assert aged.removable
    assert young.fingerprint != aged.fingerprint


def test_the_fingerprint_moves_when_the_manager_lifts_the_hold(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, _old())
    managed = _managed(dataset)

    held = _plan(managed, [ran.recipe_digest])
    lifted = _plan(managed, [ran.recipe_digest], manager="mosaic-api")

    assert held.fingerprint != lifted.fingerprint


# --- the approved form and the wire ---------------------------------------------------


def test_the_spec_holds_what_was_approved_and_round_trips(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    ids = _run_ids(dataset, _old())
    write_reader_params(
        dataset, "frame-aggregate", "0.1-reader0000", (HEADING, ids["heading"])
    )
    _ = record_open_attempt(dataset, "0.1-reader0000", target="frame-aggregate")
    keeper = _keeper(named_recipe(speed_recipe_stepping(2), "keeper"))

    plan = _plan(dataset, [ran.recipe_digest, "f" * 16], [keeper])
    spec = plan.spec

    assert spec.retire == [ran.recipe_digest]
    assert spec.keepers == [keeper]
    assert spec.policy == LOOSE
    assert [run.run_id for run in plan.kept] == [ids["heading"]]
    assert spec.approved == [CleanupTarget(storage=STORAGE, run_id=ids["speed"])]
    assert (spec.planned_at, spec.fingerprint) == (plan.planned_at, plan.fingerprint)
    assert RetirementSpec.model_validate(spec.model_dump(mode="json")) == spec


def test_the_wire_lists_every_run_and_round_trips(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    keeper = named_recipe(speed_recipe_stepping(2), "keeper")

    plan = _plan(
        dataset,
        [ran.recipe_digest, "f" * 16, "f" * 16],
        [_keeper(keeper)],
        now=datetime.datetime.now(datetime.UTC),
    )
    wire = retirement_wire(plan)

    assert RetirementPlanWire.model_validate(wire.model_dump(mode="json")) == wire
    assert (wire.fingerprint, wire.planned_at, wire.policy) == (
        plan.fingerprint,
        plan.planned_at,
        LOOSE,
    )
    assert wire.retired == (ran.recipe_digest,)
    assert wire.keepers == (recipe_digest(keeper),)
    [refused] = wire.not_retired
    assert (refused.digest, refused.reason) == ("f" * 16, "recipe_unreadable")
    assert {(run.storage, run.run_id, run.reason) for run in wire.runs} == {
        (run.storage, run.run_id, run.reason) for run in plan.cleanup.runs
    }
    assert (wire.removable_runs, wire.kept_runs) == (
        len(plan.removable),
        len(plan.kept),
    )
    assert (wire.removable_bytes, wire.kept_bytes) == (
        plan.cleanup.removable_bytes,
        plan.cleanup.kept_bytes,
    )


# --- a read ---------------------------------------------------------------------------


def test_planning_a_retirement_writes_nothing(dataset: Dataset) -> None:
    ran = run_pipeline(dataset, _old())
    keeper = _keeper(named_recipe(speed_recipe_stepping(2), "keeper"))
    before = tree_snapshot(dataset.base_dir)

    plan = _plan(dataset, [ran.recipe_digest, "f" * 16], [keeper])

    assert plan.removable
    assert_tree_unchanged(dataset.base_dir, before)


def test_planning_a_retirement_imports_no_definition(
    dataset: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = define_feature(dataset, CANARY_NAME, CANARY_SOURCE)
    defined = Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "canary",
            "steps": [
                {
                    "id": "canary",
                    "type": "feature",
                    "feature": f"def:{CANARY_NAME}",
                    "inputs": ["tracks"],
                }
            ],
        }
    )
    ran = run_pipeline(dataset, defined)
    keepers = [
        _keeper(named_recipe(defined, "kept canary")),
        _keeper(named_recipe(speed_recipe_stepping(2), "keeper")),
    ]
    forget_imported_definitions()
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))

    plan = _plan(dataset, [ran.recipe_digest], keepers)

    assert plan.retired == (ran.recipe_digest,)
    assert not imported_definitions()
    assert not canary.exists()
