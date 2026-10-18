"""Running a request's cleanup after its steps, and reporting it.

A request submitted with a cleanup removes the runs it approved once every step
has finished, in the driver (``run_request``) or in process (``run_pipeline``).
It removes no other run, keeps an approved run that has become reachable, and
holds the request ``running`` until its run-log ends.
"""

from __future__ import annotations

import importlib
import os
import socket
from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import feature_run_removal
from mosaic.core.pipeline.cancel import cancel_request
from mosaic.core.pipeline.graph import (
    BoundRef,
    CleanupPolicy,
    Recipe,
    Request,
    execute_step,
    plan_fingerprint,
    plan_with_cleanup,
    request_rollup,
    request_status_wire,
    request_verdict_of,
    run_cleanup,
    run_pipeline,
    run_request,
    submit_request,
)
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.job import CancelToken
from mosaic.runlog import (
    JsonlRunLog,
    new_execution_id,
    read_removal_records,
    run_log_dir,
    run_log_path,
)
from tests.helpers import (
    add_tracks_variant,
    exited_pid,
    make_dataset,
    record_open_attempt,
    run_speed,
    speed_heading_recipe,
    speed_recipe_stepping,
    write_driver_lease,
    write_reader_params,
)

VARIANT = "convert-trex.0.2-1111111111"

STORAGE = "speed-angvel__from__tracks"

HEADING = "heading__from__tracks"

LOOSE = CleanupPolicy(keep_newest=0, min_age_hours=0)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def _submit(ds: Dataset, recipe: Recipe, *, retire: tuple[str, ...] = ()) -> Request:
    """Submit *recipe* with a cleanup, under the fingerprint of its plan now."""
    fingerprint = plan_fingerprint(
        plan_with_cleanup(ds, recipe, cleanup=LOOSE, retire=retire)
    )
    return submit_request(
        ds, recipe, cleanup=LOOSE, expect_plan=fingerprint, retire=retire
    ).request


def _run_steps(ds: Dataset, request: Request) -> None:
    """Run every step of *request* here, without its cleanup."""
    for step_id in request.step_executions:
        _ = execute_step(ds, request, step_id)


def _exists(ds: Dataset, run_id: str) -> bool:
    return feature_run_root(ds, STORAGE, run_id).is_dir()


class _Killed(BaseException):
    """What stops a cleanup the way a killed process stops it."""


def test_the_driver_removes_the_approved_runs_after_the_steps_finish(
    dataset: Dataset,
) -> None:
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))

    done = run_request(dataset, request)

    assert (done.status, done.cleanup_status) == ("finished", "finished")
    assert not _exists(dataset, old)
    status = request_status_wire(dataset.base_dir, request)
    assert status.verdict == "finished"
    assert status.cleanup is not None
    assert (status.cleanup.status, status.cleanup.removed_runs) == ("finished", 1)
    assert status.cleanup.removed[0].run_id == old
    assert status.cleanup.removed_bytes > 0


def test_cleanup_never_extends_beyond_the_approved_set(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))
    later = run_speed(dataset, 3)

    run = run_pipeline(dataset, speed_recipe_stepping(2), request=request)

    assert run.cleanup_status == "finished"
    assert not _exists(dataset, old)
    assert _exists(dataset, later)


def test_an_approved_run_that_became_reachable_is_kept_and_reported(
    dataset: Dataset,
) -> None:
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))
    write_reader_params(dataset, "frame-aggregate", "0.1-reader0000", (STORAGE, old))
    _ = record_open_attempt(dataset, "0.1-reader0000")

    run = run_pipeline(dataset, speed_recipe_stepping(2), request=request)

    assert run.cleanup_status == "finished"
    assert _exists(dataset, old)
    status = request_status_wire(dataset.base_dir, request)
    assert status.cleanup is not None
    [kept] = status.cleanup.kept
    assert (kept.run_id, kept.reason) == (old, "read_by")
    assert "0.1-reader0000" in kept.detail


def test_cleanup_is_skipped_when_a_step_did_not_finish(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))
    _ = cancel_request(dataset, request.request_id)

    done = run_request(dataset, request)

    assert (done.status, done.cleanup_status) == ("cancelled", "")
    assert _exists(dataset, old)
    status = request_status_wire(dataset.base_dir, request)
    assert status.cleanup is not None
    assert status.cleanup.status == "skipped"


def test_a_cancel_during_cleanup_stops_between_runs(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = run_speed(dataset, 1), run_speed(dataset, 3)
    request = _submit(dataset, speed_recipe_stepping(2))
    _run_steps(dataset, request)
    token = CancelToken()
    remove = feature_run_removal.remove_feature_run

    def remove_then_cancel(*args: object, **kwargs: object) -> object:
        removed = remove(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        token.cancel()
        return removed

    monkeypatch.setattr(feature_run_removal, "remove_feature_run", remove_then_cancel)

    outcome = run_cleanup(dataset, request, token=token)

    assert outcome.status == "cancelled"
    assert [_exists(dataset, run_id) for run_id in (first, second)].count(True) == 1
    rollup = request_rollup(dataset.base_dir, request)
    assert rollup.status == "finished"
    status = request_status_wire(dataset.base_dir, request)
    assert status.cleanup is not None
    assert (status.cleanup.status, status.cleanup.removed_runs) == ("cancelled", 1)


def test_a_failed_removal_is_reported_and_the_cleanup_goes_on(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = run_speed(dataset, 1)
    _ = run_speed(dataset, 3)
    request = _submit(dataset, speed_recipe_stepping(2))
    _run_steps(dataset, request)
    calls: list[Path] = []

    def refuse_first(path: Path) -> None:
        calls.append(path)
        if len(calls) == 1:
            raise OSError(f"busy: {path}")
        raise_free(path)

    raise_free = feature_run_removal.shutil.rmtree
    monkeypatch.setattr(feature_run_removal.shutil, "rmtree", refuse_first)

    outcome = run_cleanup(dataset, request, token=CancelToken())

    assert (outcome.status, len(outcome.removed), len(outcome.failed)) == (
        "finished",
        1,
        1,
    )
    status = request_status_wire(dataset.base_dir, request)
    assert status.cleanup is not None
    assert (status.cleanup.failed_runs, status.cleanup.removed_runs) == (1, 1)
    assert status.verdict == "finished"


def _interrupted(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> tuple[Request, str]:
    """A request whose steps finished and whose cleanup stopped mid-removal."""
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))
    _run_steps(dataset, request)

    def killed(path: Path) -> None:
        raise _Killed(str(path))

    with monkeypatch.context() as patched:
        patched.setattr(feature_run_removal.shutil, "rmtree", killed)
        with pytest.raises(_Killed):
            _ = run_cleanup(dataset, request, token=CancelToken())
    return request, old


def test_a_request_with_cleanup_is_running_until_cleanup_ends(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, old = _interrupted(dataset, monkeypatch)

    status = request_status_wire(dataset.base_dir, request)

    assert status.status == "running"
    assert status.cleanup is not None
    assert (status.cleanup.status, status.cleanup.interrupted_runs) == ("running", 1)
    assert _exists(dataset, old)


def test_resuming_a_request_finishes_its_cleanup(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, old = _interrupted(dataset, monkeypatch)

    done = run_request(dataset, request)

    assert (done.status, done.ran, done.cleanup_status) == ("finished", (), "finished")
    assert not _exists(dataset, old)
    status = request_status_wire(dataset.base_dir, request)
    assert status.cleanup is not None
    assert (status.cleanup.removed_runs, status.cleanup.interrupted_runs) == (1, 0)


def test_a_killed_driver_mid_cleanup_reads_orphaned(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, _ = _interrupted(dataset, monkeypatch)
    _ = write_driver_lease(
        dataset.base_dir, request.request_id, stale=True, pid=exited_pid()
    )

    rollup = request_rollup(dataset.base_dir, request)

    assert request_verdict_of(dataset.base_dir, rollup) == "orphaned"


def test_a_pending_cleanup_without_a_driver_is_cancelled(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))
    _run_steps(dataset, request)
    assert request_rollup(dataset.base_dir, request).status == "running"

    cancelled = cancel_request(dataset, request.request_id)

    assert cancelled.status == "finished"
    assert _exists(dataset, old)
    status = request_status_wire(dataset.base_dir, request)
    assert status.cleanup is not None
    assert status.cleanup.status == "cancelled"


def test_run_pipeline_runs_the_cleanup_last(dataset: Dataset) -> None:
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))

    run = run_pipeline(dataset, speed_recipe_stepping(2), request=request)

    assert (run.cleanup_status, [outcome.step_id for outcome in run.outcomes]) == (
        "finished",
        ["speed"],
    )
    assert not _exists(dataset, old)
    assert request_rollup(dataset.base_dir, request).status == "finished"


def test_a_run_an_earlier_version_recorded_is_removed_by_name(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recipe's name reaches the judging at submission and at removal."""
    module = importlib.import_module("mosaic.core.pipeline.graph.cleanup")
    monkeypatch.setattr(module, "REQUEST_HOLD_FLOOR_HOURS", 0)
    first = run_pipeline(dataset, speed_recipe_stepping(1))
    [old] = [outcome.run_id for outcome in first.outcomes]
    request = _submit(dataset, speed_recipe_stepping(2))

    run = run_pipeline(dataset, speed_recipe_stepping(2), request=request)

    assert request.cleanup is not None
    assert [target.run_id for target in request.cleanup.approved] == [old]
    assert run.cleanup_status == "finished"
    assert not _exists(dataset, old)


def test_a_request_without_cleanup_rolls_up_as_before(dataset: Dataset) -> None:
    request = submit_request(dataset, speed_recipe_stepping(2)).request
    _run_steps(dataset, request)

    rollup = request_rollup(dataset.base_dir, request)
    status = request_status_wire(dataset.base_dir, request)

    assert (rollup.status, rollup.cleanup, status.cleanup) == ("finished", None, None)


# --- what appears while the cleanup runs ---------------------------------------------


def _during_cleanup(monkeypatch: pytest.MonkeyPatch, act: Callable[[], None]) -> None:
    """Call *act* once the cleanup has judged its runs, before it removes any."""
    module = importlib.import_module("mosaic.core.pipeline.graph.cleanup")
    judge = module.plan_cleanup

    def judge_then_act(*args: object, **kwargs: object) -> object:
        judged = judge(*args, **kwargs)
        act()
        return judged

    monkeypatch.setattr(module, "plan_cleanup", judge_then_act)


def test_a_run_read_by_an_attempt_that_ended_during_the_cleanup_is_kept(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attempt that starts and ends between the judging and the removal."""
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))
    _run_steps(dataset, request)

    def read_old_and_finish() -> None:
        write_reader_params(
            dataset, "frame-aggregate", "0.1-reader0000", (STORAGE, old)
        )
        execution_id = new_execution_id()
        path = run_log_path(dataset.base_dir, execution_id)
        with JsonlRunLog(path, execution_id) as log:
            log.started(
                kind="feature",
                target="frame-aggregate",
                host=socket.gethostname(),
                pid=os.getpid(),
            )
            log.set_run_id("0.1-reader0000")
            log.finished()

    _during_cleanup(monkeypatch, read_old_and_finish)

    outcome = run_cleanup(dataset, request, token=CancelToken())

    assert outcome.status == "finished"
    assert _exists(dataset, old)
    assert [kept[1] for kept in outcome.kept] == [old]


def test_a_run_a_request_submitted_during_the_cleanup_binds_is_kept(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))
    _run_steps(dataset, request)

    def bind_old() -> None:
        _ = submit_request(
            dataset,
            speed_recipe_stepping(5),
            bind={"speed": BoundRef(feature=STORAGE, run_id=old)},
        )

    _during_cleanup(monkeypatch, bind_old)

    outcome = run_cleanup(dataset, request, token=CancelToken())

    assert outcome.status == "finished"
    assert _exists(dataset, old)
    assert [(kept[1], kept[2]) for kept in outcome.kept] == [(old, "held_by_request")]


def test_an_interrupted_removal_is_not_finished_under_an_open_attempt(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows dropped and files left, then an attempt writing that run again."""
    request, old = _interrupted(dataset, monkeypatch)
    _ = record_open_attempt(dataset, old)

    outcome = run_cleanup(dataset, request, token=CancelToken())

    assert _exists(dataset, old)
    assert [(kept[1], kept[2]) for kept in outcome.kept] == [(old, "attempt_open")]


def test_the_cleanup_records_its_process_as_a_step_does(dataset: Dataset) -> None:
    """A cancel judges the cleanup's pid by its boot and namespace, as a step's."""
    from mosaic.core.pipeline.markers import current_boot_id, current_pid_namespace
    from mosaic.runlog import read_run, run_log_dir

    _ = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))
    _run_steps(dataset, request)
    assert request.cleanup is not None

    _ = run_cleanup(dataset, request, token=CancelToken())

    snapshot = read_run(run_log_dir(dataset.base_dir), request.cleanup.execution_id)
    assert snapshot is not None
    assert (snapshot["pid"], snapshot["boot_id"], snapshot["pid_ns"]) == (
        os.getpid(),
        current_boot_id(),
        current_pid_namespace(),
    )


def test_a_run_an_attempt_resolving_its_inputs_may_read_is_kept(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))
    _run_steps(dataset, request)

    def start_reader() -> None:
        _ = record_open_attempt(dataset, "", target=f"frame-aggregate__from__{STORAGE}")

    _during_cleanup(monkeypatch, start_reader)

    outcome = run_cleanup(dataset, request, token=CancelToken())

    assert _exists(dataset, old)
    assert [(kept[1], kept[2]) for kept in outcome.kept] == [(old, "attempt_open")]


# --- a retired recipe's runs -------------------------------------------------------


def test_a_retired_recipes_run_is_removed_and_the_removal_names_the_recipe(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retirement reaches the judging at submission and again at removal."""
    module = importlib.import_module("mosaic.core.pipeline.graph.cleanup")
    monkeypatch.setattr(module, "REQUEST_HOLD_FLOOR_HOURS", 0)
    old = run_pipeline(dataset, speed_heading_recipe())
    heading = next(o.run_id for o in old.outcomes if o.step_id == "heading")
    speed = next(o.run_id for o in old.outcomes if o.step_id == "speed")
    recipe = speed_recipe_stepping(2)

    request = _submit(dataset, recipe, retire=(old.recipe_digest,))
    run = run_pipeline(dataset, recipe, request=request)

    assert request.cleanup is not None
    assert request.cleanup.retired == [old.recipe_digest]
    assert {t.run_id for t in request.cleanup.approved} == {heading, speed}
    assert run.cleanup_status == "finished"
    assert not feature_run_root(dataset, HEADING, heading).exists()
    assert not _exists(dataset, speed)
    records = read_removal_records(
        run_log_dir(dataset.base_dir), request.cleanup.execution_id
    )
    by_run = {record["run_id"]: record for record in records}
    assert {(r["state"], r["reason"], r["retired"]) for r in records} == {
        ("removed", "retired", old.recipe_digest)
    }
    assert by_run[heading]["params"]
    assert by_run[heading]["superseded_by"] == ""
    status = request_status_wire(dataset.base_dir, request)
    assert status.cleanup is not None
    assert {run.retired for run in status.cleanup.removed} == {old.recipe_digest}
    assert status.cleanup.kept == ()


def test_a_run_removed_without_a_retirement_records_none(
    dataset: Dataset,
) -> None:
    old = run_speed(dataset, 1)
    request = _submit(dataset, speed_recipe_stepping(2))

    _ = run_pipeline(dataset, speed_recipe_stepping(2), request=request)

    assert request.cleanup is not None
    [record] = read_removal_records(
        run_log_dir(dataset.base_dir), request.cleanup.execution_id
    )
    assert (record["run_id"], record["reason"], record["retired"]) == (
        old,
        "superseded",
        "",
    )


def test_a_retired_recipe_that_opens_after_the_submission_keeps_its_runs(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("mosaic.core.pipeline.graph.cleanup")
    monkeypatch.setattr(module, "REQUEST_HOLD_FLOOR_HOURS", 0)
    old = run_pipeline(dataset, speed_heading_recipe())
    heading = next(o.run_id for o in old.outcomes if o.step_id == "heading")
    recipe = speed_recipe_stepping(2)
    request = _submit(dataset, recipe, retire=(old.recipe_digest,))
    _ = submit_request(dataset, speed_heading_recipe())

    _ = run_pipeline(dataset, recipe, request=request)

    assert feature_run_root(dataset, HEADING, heading).is_dir()
    status = request_status_wire(dataset.base_dir, request)
    assert status.cleanup is not None
    assert {kept.run_id: kept.reason for kept in status.cleanup.kept}[heading] == (
        "held_by_request"
    )
