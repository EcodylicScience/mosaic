"""A submission's ``bind``: a bound step is the run it names, and is never executed.

The step's output is the pinned run, so its run id is the pinned run id and its
storage is the bind's. The steps below it read that run, and their run ids derive
from it as from any upstream run, and nothing above it reaches them. The plan
refuses a bind naming no step or an op step, one whose storage is not the step's
own, and one naming a run that does not exist, and a step that runs refuses such
a bind in a request submitted before the plan checked it.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.artifact_kinds import ARTIFACT_STATUSES
from mosaic.core.pipeline.graph import (
    BoundRef,
    CoverageShortfall,
    Recipe,
    RecipeRefused,
    RecipeRefusedAtStep,
    Request,
    execute_step,
    plan_pipeline,
    record_step_ended,
    save_request,
    request_rollup,
    request_status_wire,
    requests_root,
    run_pipeline,
    run_request,
    submit_request,
    wait_for_requests,
)
from mosaic.core.pipeline.graph.model import BOUND_STATUS, StepStatus
from mosaic.core.pipeline.job import CancelToken
from mosaic.runlog import read_run, run_log_dir
from mosaic.core.pipeline.run import run_feature
from mosaic.core.scope import Scope
from tests.helpers import (
    CHAIN_RECIPE,
    CONVERT_TRACKS_STEP,
    add_tracks_variant,
    chain_recipe,
    invoke_json,
    make_dataset,
    run_speed,
    speed_recipe,
    write_driver_lease,
)

SPEED_STORAGE = "speed-angvel__from__tracks"
VARIANT = "convert-trex.0.2-1111111111"


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables before any feature run."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return dataset


def _runs(dataset: Dataset, storage: str) -> list[str]:
    """The run directories of one feature storage."""
    root = dataset.get_root("features") / storage
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _bound_speed(older: str) -> dict[str, BoundRef]:
    return {"speed": BoundRef(feature=SPEED_STORAGE, run_id=older)}


def _chain_stepping(step_size: int) -> Recipe:
    """The chain recipe with its speed step at *step_size*."""
    document = Recipe.model_validate(CHAIN_RECIPE).model_dump(mode="json")
    document["steps"][0]["params"] = {"step_size": step_size}
    return Recipe.model_validate(document)


class _Clock:
    """A clock that only the sleep advances."""

    def __init__(self) -> None:
        self.now: float = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_a_bound_step_is_not_executed_in_process(tracked: Dataset) -> None:
    older = run_speed(tracked, 3)
    before = _runs(tracked, SPEED_STORAGE)
    submitted = submit_request(tracked, chain_recipe(), bind=_bound_speed(older))

    done = run_pipeline(tracked, chain_recipe(), request=submitted.request)

    assert [(o.step_id, o.state) for o in done.outcomes] == [
        ("speed", "bound"),
        ("templates", "ran"),
    ]
    assert _runs(tracked, SPEED_STORAGE) == before
    logs = run_log_dir(tracked.base_dir)
    record = read_run(logs, submitted.request.execution_of("speed"))
    assert record is not None
    assert (record["status"], record["run_id"], record["cache_hit"]) == (
        "finished",
        older,
        True,
    )


def test_a_driver_starts_no_process_for_a_bound_step(tracked: Dataset) -> None:
    older = run_speed(tracked, 3)
    before = _runs(tracked, SPEED_STORAGE)
    submitted = submit_request(tracked, chain_recipe(), bind=_bound_speed(older))

    done = run_request(tracked, submitted.request)

    assert (done.status, done.ran, done.skipped) == (
        "finished",
        ("templates",),
        ("speed",),
    )
    assert "step speed" not in done.log_path.read_text()
    assert _runs(tracked, SPEED_STORAGE) == before


def test_a_step_below_a_bound_step_reads_the_pinned_run(tracked: Dataset) -> None:
    older = run_speed(tracked, 3)
    # The chain whose own speed step computes that run, planned once it exists.
    expected = plan_pipeline(tracked, _chain_stepping(3)).step("templates").run_id
    assert expected is not None

    submitted = submit_request(tracked, chain_recipe(), bind=_bound_speed(older))
    speed = submitted.plan.step("speed")
    templates = submitted.plan.step("templates")
    done = run_pipeline(tracked, chain_recipe(), request=submitted.request)

    assert (speed.status, speed.run_id, speed.storage_name) == (
        "bound",
        older,
        SPEED_STORAGE,
    )
    assert speed.is_complete
    assert templates.spec.inputs == ({"feature": SPEED_STORAGE, "run_id": older},)
    assert templates.run_id == expected
    assert [(o.step_id, o.run_id) for o in done.outcomes] == [
        ("speed", older),
        ("templates", expected),
    ]


@pytest.mark.parametrize(
    ("step", "storage"),
    [("templates", SPEED_STORAGE), ("speed", "speed-angvel")],
    ids=["another-steps-storage", "the-feature-slug"],
)
def test_a_bind_naming_another_storage_is_refused_at_plan_time(
    tracked: Dataset, step: str, storage: str
) -> None:
    older = run_speed(tracked, 3)
    bind = {step: BoundRef(feature=storage, run_id=older)}

    with pytest.raises(RecipeRefused) as refused:
        _ = submit_request(tracked, chain_recipe(), bind=bind)

    [problem] = refused.value.plan.problems
    assert (problem.step, problem.where) == (step, f"bind.{step}")
    assert repr(storage) in problem.message
    assert not requests_root(tracked.base_dir).exists()


def test_a_bind_naming_a_missing_run_is_refused_at_plan_time(
    tracked: Dataset,
) -> None:
    bind = _bound_speed("0.2-0000000000")

    with pytest.raises(RecipeRefused) as refused:
        _ = submit_request(tracked, chain_recipe(), bind=bind)

    [problem] = refused.value.plan.problems
    assert (problem.step, problem.where) == ("speed", "bind.speed")
    assert "0.2-0000000000" in problem.message
    assert not requests_root(tracked.base_dir).exists()


def test_a_bound_step_reads_as_bound_and_done_in_the_request_status(
    tracked: Dataset,
) -> None:
    older = run_speed(tracked, 3)
    submitted = submit_request(tracked, chain_recipe(), bind=_bound_speed(older))
    _ = run_pipeline(tracked, chain_recipe(), request=submitted.request)

    status = request_status_wire(tracked.base_dir, submitted.request)

    speed = next(step for step in status.steps if step.step == "speed")
    assert (speed.outcome, speed.run_id, speed.liveness) == ("bound", older, None)
    assert (status.status, status.verdict) == ("finished", "finished")


def test_a_request_whose_every_step_is_bound_has_ended_and_a_wait_ends(
    tracked: Dataset,
) -> None:
    older = run_speed(tracked, 1)
    request = submit_request(tracked, speed_recipe(), bind=_bound_speed(older)).request
    clock = _Clock()

    waited = wait_for_requests(
        tracked.base_dir,
        [request],
        until="all",
        timeout=60,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert request_rollup(tracked.base_dir, request).status == "finished"
    assert waited.outcome == "finished"
    assert clock.now == 0


def test_a_step_status_is_an_artifact_status_or_bound() -> None:
    assert get_args(StepStatus.__value__) == (*ARTIFACT_STATUSES, BOUND_STATUS)


def _guarded_chain() -> Recipe:
    """The chain with a ``pre`` step that ``speed`` waits on through ``after``."""
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "pre, speed, templates",
            "steps": [
                {
                    "id": "pre",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "params": {"step_size": 2},
                },
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": ["tracks"],
                    "after": ["pre"],
                },
                {
                    "id": "templates",
                    "type": "feature",
                    "feature": "extract-templates",
                    "inputs": [{"step": "speed"}],
                    "params": {"n_templates": 4},
                },
            ],
        }
    )


def _fail_without_a_run(dataset: Dataset, execution_id: str, step_id: str) -> None:
    """Record a step ``failed`` with no run id, as a refusal leaves it."""
    record_step_ended(
        dataset,
        execution_id,
        step_id=step_id,
        owner="",
        status="failed",
        error_json='{"type": "ValueError", "message": "bad"}',
    )


def test_a_failure_above_a_bound_step_does_not_refuse_the_step_below(
    tracked: Dataset,
) -> None:
    older = run_speed(tracked, 3)
    request = submit_request(
        tracked, _guarded_chain(), bind=_bound_speed(older)
    ).request
    _fail_without_a_run(tracked, request.execution_of("pre"), "pre")

    outcome = execute_step(tracked, request, "templates")

    assert outcome.state == "ran"


def test_a_resumed_driver_runs_the_step_below_a_bound_step_whose_ancestor_failed(
    tracked: Dataset,
) -> None:
    """``pre`` failed and its driver was killed, leaving its lease. On resume the
    failure ends no branch through the bound ``speed``, and ``templates`` runs."""
    older = run_speed(tracked, 3)
    request = submit_request(
        tracked, _guarded_chain(), bind=_bound_speed(older)
    ).request
    _fail_without_a_run(tracked, request.execution_of("pre"), "pre")
    _ = write_driver_lease(tracked.base_dir, request.request_id, stale=True)

    done = run_request(tracked, request)

    assert (done.ran, done.skipped, done.blocked) == (("templates",), ("speed",), ())
    record = read_run(run_log_dir(tracked.base_dir), request.execution_of("templates"))
    assert record is not None
    assert record["status"] == "finished"


def test_a_bound_step_run_as_a_queue_runs_it_records_the_pinned_run(
    tracked: Dataset,
) -> None:
    older = run_speed(tracked, 3)
    expected = plan_pipeline(tracked, _chain_stepping(3)).step("templates").run_id
    before = _runs(tracked, SPEED_STORAGE)
    request = submit_request(tracked, chain_recipe(), bind=_bound_speed(older)).request

    speed = execute_step(tracked, request, "speed")
    templates = execute_step(tracked, request, "templates")

    assert (speed.state, speed.run_id) == ("bound", older)
    record = read_run(run_log_dir(tracked.base_dir), request.execution_of("speed"))
    assert record is not None
    assert (record["status"], record["run_id"], record["cache_hit"]) == (
        "finished",
        older,
        True,
    )
    assert _runs(tracked, SPEED_STORAGE) == before
    assert templates.run_id == expected


def test_the_run_command_reports_a_bound_step_as_a_cache_hit(tracked: Dataset) -> None:
    older = run_speed(tracked, 3)
    request = submit_request(tracked, chain_recipe(), bind=_bound_speed(older)).request

    payload = invoke_json(
        [
            "run",
            "-m",
            str(tracked.manifest_path),
            "--graph-request",
            request.request_id,
            "--step",
            "speed",
            "--execution-id",
            request.execution_of("speed"),
            "--json",
        ]
    )

    assert (payload["state"], payload["cache_hit"]) == ("bound", True)
    assert payload["run_id"] == older


def test_a_cancel_does_not_record_a_bound_step_cancelled(tracked: Dataset) -> None:
    older = run_speed(tracked, 3)
    request = submit_request(tracked, chain_recipe(), bind=_bound_speed(older)).request
    token = CancelToken()
    token.cancel()

    done = run_request(tracked, request, cancel_token=token)

    assert done.cancelled == ("templates",)
    record = read_run(run_log_dir(tracked.base_dir), request.execution_of("speed"))
    assert record is None or record["status"] != "cancelled"


def test_a_reader_of_a_bound_run_that_lacks_entries_is_refused_its_shortfall(
    tracked: Dataset,
) -> None:
    from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel

    partial = run_feature(
        tracked,
        SpeedAngvel(params={"step_size": 3}),
        scope=Scope(entries=[("", "seq_a")]),
    ).run_id
    assert partial is not None
    request = submit_request(
        tracked, chain_recipe(), bind=_bound_speed(partial)
    ).request

    with pytest.raises(CoverageShortfall):
        _ = execute_step(tracked, request, "templates")


def test_a_bind_of_an_op_step_is_refused_at_plan_time(tracked: Dataset) -> None:
    recipe = Recipe.model_validate(
        {"schema_version": 1, "steps": [CONVERT_TRACKS_STEP]}
    )
    bind = {"tracks": BoundRef(feature=SPEED_STORAGE, run_id=run_speed(tracked, 3))}

    plan = plan_pipeline(tracked, recipe, bind=bind, refuse=False)

    [problem] = [p for p in plan.problems if p.where == "bind.tracks"]
    assert problem.step == "tracks"
    assert "op step" in problem.message


def test_a_bind_naming_no_step_is_refused_at_plan_time(tracked: Dataset) -> None:
    bind = {"sped": BoundRef(feature=SPEED_STORAGE, run_id=run_speed(tracked, 3))}

    with pytest.raises(RecipeRefused) as refused:
        _ = submit_request(tracked, chain_recipe(), bind=bind)

    [problem] = refused.value.plan.problems
    assert (problem.step, problem.where) == ("", "bind")
    assert "'sped'" in problem.message
    assert "names no step of the recipe" in problem.message


def _stale(dataset: Dataset, older: str) -> Request:
    """A request whose bind pins the step below the one it meant to pin.

    It binds templates to a speed run, as a request submitted before binds were
    checked could, and is saved as such a request was.
    """
    submitted = submit_request(dataset, chain_recipe()).request
    stale = submitted.model_copy(
        update={"bind": {"templates": BoundRef(feature=SPEED_STORAGE, run_id=older)}}
    )
    _ = save_request(dataset.base_dir, stale)
    return stale


def test_running_a_stale_bind_in_process_refuses_its_step(tracked: Dataset) -> None:
    older = run_speed(tracked, 3)
    stale = _stale(tracked, older)

    with pytest.raises(RecipeRefusedAtStep):
        _ = run_pipeline(tracked, chain_recipe(), request=stale)

    record = read_run(run_log_dir(tracked.base_dir), stale.execution_of("templates"))
    assert record is not None
    assert record["status"] == "failed"
    assert "bind.templates" in record["error_json"]
    assert request_rollup(tracked.base_dir, stale).status == "failed"
    assert not (
        tracked.get_root("features")
        / "extract-templates__from__speed-angvel__from__tracks"
    ).exists()


def test_a_driver_refuses_a_stale_bind_without_a_process(tracked: Dataset) -> None:
    older = run_speed(tracked, 3)
    stale = _stale(tracked, older)

    done = run_request(tracked, stale)

    assert (done.status, done.ran) == ("failed", ("speed",))
    assert "step templates" not in done.log_path.read_text()
    record = read_run(run_log_dir(tracked.base_dir), stale.execution_of("templates"))
    assert record is not None
    assert "bind.templates" in record["error_json"]


def test_a_driver_that_refuses_a_stale_bind_blocks_the_step_below_it(
    tracked: Dataset,
) -> None:
    """Speed is bound to a run of the templates storage, which it did not write."""
    older = run_speed(tracked, 3)
    submitted = submit_request(tracked, chain_recipe()).request
    stale = submitted.model_copy(
        update={
            "bind": {
                "speed": BoundRef(
                    feature="extract-templates__from__speed-angvel__from__tracks",
                    run_id=older,
                )
            }
        }
    )
    _ = save_request(tracked.base_dir, stale)

    done = run_request(tracked, stale)

    assert done.status == "failed"
    assert done.blocked == ("templates",)
    assert "speed" not in done.skipped
    record = read_run(run_log_dir(tracked.base_dir), stale.execution_of("speed"))
    assert record is not None
    assert record["status"] == "failed"
    assert "bind.speed" in record["error_json"]
