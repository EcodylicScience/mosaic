"""Running an approved retirement as a job attempt.

``run_retirement`` plans every keeper again and judges the retired runs again,
then removes each approved run still removable. It records each run in the
attempt's run-log, reports a removal that raised as a lost entry, and stops on
a cancel between runs.

The retired recipe ``old`` runs ``speed-angvel`` at step size 1 and ``heading``.
"""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline import feature_run_removal
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.graph import (
    BoundRef,
    CleanupOutcome,
    CleanupPolicy,
    Recipe,
    RetirementKeeper,
    RetirementSpec,
    plan_retirement,
    run_pipeline,
    run_retirement,
    submit_request,
)
from mosaic.core.pipeline.graph.store import requests_root
from mosaic.core.pipeline.index import (
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.job import Cancelled, CancelToken, job_context
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.scope import Scope
from mosaic.runlog import (
    JsonlRunLog,
    RemovalRecord,
    RunLogSnapshot,
    new_execution_id,
    read_removal_records,
    reduce_run_log,
    run_log_dir,
    run_log_path,
)
from tests.helpers import (
    CANARY_NAME,
    CANARY_SOURCE,
    CANARY_VARIABLE,
    HEADING_STORAGE,
    LOOSE,
    OldRuns,
    SPEED_STORAGE,
    act_after_judging,
    add_tracks_variant,
    end_attempt,
    forget_imported_definitions,
    imported_definitions,
    kept_reasons,
    make_dataset,
    managed_dataset,
    named_recipe,
    observe_removals,
    record_open_attempt,
    run_id_of,
    run_speed,
    speed_heading_recipe,
    speed_recipe_stepping,
    write_reader_params,
)

VARIANT = "convert-trex.0.2-1111111111"


@pytest.fixture
def dataset(tmp_path: Path, lapsed_request_holds: None) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


def _keeper(recipe: Recipe) -> RetirementKeeper:
    return RetirementKeeper(recipe=recipe, scope=Scope())


def _retire(
    ds: Dataset,
    spec: RetirementSpec,
    manager: str = "",
    *,
    execution_id: str = "",
    token: CancelToken | None = None,
) -> tuple[CleanupOutcome, str]:
    """Run *spec* in a job attempt and return what it did and the attempt's id."""
    with job_context(
        ds,
        kind="retire-recipes",
        target="retire-recipes",
        execution_id=execution_id or None,
        cancel_token=token,
    ) as ctx:
        outcome = run_retirement(ds, spec, manager, ctx=ctx)
    return outcome, ctx.execution_id


def _snapshot(ds: Dataset, execution_id: str) -> RunLogSnapshot:
    snapshot = reduce_run_log(run_log_path(ds.base_dir, execution_id))
    assert snapshot is not None
    return snapshot


def _records(ds: Dataset, execution_id: str) -> dict[str, RemovalRecord]:
    return {
        record["run_id"]: record
        for record in read_removal_records(run_log_dir(ds.base_dir), execution_id)
    }


def _removed(outcome: CleanupOutcome) -> list[str]:
    return [run.run_id for run in outcome.removed]


def _speed_exists(ds: Dataset, run_id: str) -> bool:
    return feature_run_root(ds, SPEED_STORAGE, run_id).is_dir()


def _heading_exists(ds: Dataset, run_id: str) -> bool:
    return feature_run_root(ds, HEADING_STORAGE, run_id).is_dir()


class _Killed(BaseException):
    """What stops a retirement the way a killed process stops it."""


# --- removing ---------------------------------------------------------------------


def test_each_approved_run_still_removable_is_removed_as_retired(
    dataset: Dataset,
) -> None:
    old = OldRuns(dataset)

    outcome, execution_id = _retire(dataset, old.spec)

    assert {target.run_id for target in old.spec.approved} == {old.speed, old.heading}
    assert (outcome.status, outcome.execution_id) == ("finished", execution_id)
    assert sorted(_removed(outcome)) == sorted([old.speed, old.heading])
    assert (outcome.kept, outcome.failed) == ((), ())
    assert not _speed_exists(dataset, old.speed)
    assert not _heading_exists(dataset, old.heading)
    records = _records(dataset, execution_id)
    assert {
        (record["state"], record["reason"], record["retired"])
        for record in records.values()
    } == {("removed", "retired", old.digest)}
    assert records[old.speed]["params"]
    snapshot = _snapshot(dataset, execution_id)
    assert (snapshot["status"], snapshot["entries_failed"]) == ("finished", 0)
    assert (snapshot["progress_done"], snapshot["progress_total"]) == (2, 2)


def test_a_run_the_plan_did_not_approve_is_not_removed_though_now_removable(
    dataset: Dataset,
) -> None:
    ran = run_pipeline(dataset, speed_heading_recipe())
    heading = run_id_of(ran, "heading")
    write_reader_params(
        dataset, "frame-aggregate", "0.1-reader0000", (HEADING_STORAGE, heading)
    )
    reader = record_open_attempt(dataset, "0.1-reader0000", target="frame-aggregate")
    spec = plan_retirement(dataset, [ran.recipe_digest], [], LOOSE, "").spec
    end_attempt(dataset, reader)
    again = plan_retirement(dataset, [ran.recipe_digest], [], LOOSE, "")

    outcome, _ = _retire(dataset, spec)

    assert heading not in {target.run_id for target in spec.approved}
    assert heading in {run.run_id for run in again.removable}
    assert _removed(outcome) == [run_id_of(ran, "speed")]
    assert _heading_exists(dataset, heading)


def test_a_retirement_without_a_run_log_is_refused(dataset: Dataset) -> None:
    old = OldRuns(dataset)

    with job_context(dataset, kind="retire-recipes", target="x", track=False) as ctx:
        with pytest.raises(ValueError, match="run-log"):
            _ = run_retirement(dataset, old.spec, "", ctx=ctx)

    assert _speed_exists(dataset, old.speed)


# --- what keeps an approved run ---------------------------------------------------


def test_an_approved_run_a_running_request_submitted_after_the_plan_runs_is_kept(
    dataset: Dataset,
) -> None:
    """The request has not ended, and its recipe runs both features."""
    old = OldRuns(dataset)
    running = submit_request(dataset, speed_heading_recipe("running")).request

    outcome, execution_id = _retire(dataset, old.spec)

    assert kept_reasons(outcome) == {
        old.speed: "held_by_request",
        old.heading: "held_by_request",
    }
    assert outcome.removed == ()
    record = _records(dataset, execution_id)[old.speed]
    assert (record["state"], record["reason"]) == ("kept", "held_by_request")
    assert running.request_id in record["detail"]


def test_an_approved_run_a_request_submitted_during_the_retirement_pins_is_kept(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The request appears after the judging, so only the watch can see its pin."""
    old = OldRuns(dataset)
    pinned: list[str] = []

    def pin_speed() -> None:
        request = submit_request(
            dataset,
            speed_recipe_stepping(5),
            bind={"speed": BoundRef(feature=SPEED_STORAGE, run_id=old.speed)},
        ).request
        pinned.append(request.request_id)

    act_after_judging(monkeypatch, "judge_runs", pin_speed)

    outcome, execution_id = _retire(dataset, old.spec)

    assert kept_reasons(outcome) == {old.speed: "held_by_request"}
    assert _removed(outcome) == [old.heading]
    assert _speed_exists(dataset, old.speed)
    record = _records(dataset, execution_id)[old.speed]
    assert record["reason"] == "held_by_request"
    assert pinned[0] in record["detail"]
    assert "pins it" in record["detail"]


def test_no_request_is_written_between_the_replan_and_the_judging(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A submission waits for the retirement's judging to end."""
    old = OldRuns(dataset)
    requests = requests_root(dataset.base_dir)
    before = set(requests.glob("*.json"))
    errors: list[BaseException] = []
    seen: list[set[Path]] = []

    def submit() -> None:
        try:
            _ = submit_request(dataset, speed_recipe_stepping(5))
        except BaseException as error:
            errors.append(error)

    submitter = threading.Thread(target=submit)

    def submit_and_wait() -> None:
        submitter.start()
        submitter.join(timeout=3)
        seen.append(set(requests.glob("*.json")) - before)

    act_after_judging(monkeypatch, "judge_runs", submit_and_wait)

    _ = _retire(dataset, old.spec)
    submitter.join()

    assert seen == [set()]
    assert errors == []
    assert len(set(requests.glob("*.json")) - before) == 1


def test_an_approved_run_a_keeper_now_expects_is_kept(dataset: Dataset) -> None:
    old = OldRuns(dataset)
    keeper = _keeper(named_recipe(speed_recipe_stepping(1), "keeper"))
    spec = old.spec.model_copy(update={"keepers": [keeper]})

    outcome, execution_id = _retire(dataset, spec)

    assert kept_reasons(outcome) == {old.speed: "keeper"}
    assert _removed(outcome) == [old.heading]
    assert _speed_exists(dataset, old.speed)
    record = _records(dataset, execution_id)[old.speed]
    assert (record["state"], record["reason"]) == ("kept", "keeper")


def test_an_approved_run_an_attempt_of_a_later_request_writes_is_kept_open(
    dataset: Dataset,
) -> None:
    old = OldRuns(dataset)
    request = submit_request(dataset, speed_heading_recipe("user")).request
    attempt = request.step_executions["speed"]
    with JsonlRunLog(run_log_path(dataset.base_dir, attempt), attempt) as log:
        log.started(
            kind="feature",
            target=SPEED_STORAGE,
            host=socket.gethostname(),
            pid=os.getpid(),
        )
        log.set_run_id(old.speed)

    outcome, _ = _retire(dataset, old.spec)

    assert kept_reasons(outcome) == {
        old.speed: "attempt_open",
        old.heading: "held_by_request",
    }
    assert _speed_exists(dataset, old.speed)
    assert _heading_exists(dataset, old.heading)


@pytest.mark.parametrize(
    ("ends", "reason"),
    [
        ("never", "attempt_open"),
        ("during", "attempt_open"),
        ("before", "no_longer_candidate"),
    ],
)
def test_an_approved_run_an_outside_attempt_names_after_the_plan_is_kept(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch, ends: str, reason: str
) -> None:
    old = OldRuns(dataset)
    attempt = record_open_attempt(dataset, old.heading, target=HEADING_STORAGE)
    if ends == "before":
        end_attempt(dataset, attempt)
    if ends == "during":
        act_after_judging(
            monkeypatch, "judge_runs", lambda: end_attempt(dataset, attempt)
        )

    outcome, execution_id = _retire(dataset, old.spec)

    assert kept_reasons(outcome) == {old.heading: reason}
    assert _removed(outcome) == [old.speed]
    assert _heading_exists(dataset, old.heading)
    assert _records(dataset, execution_id)[old.heading]["reason"] == reason


def test_an_approved_run_already_gone_is_absent(dataset: Dataset) -> None:
    old = OldRuns(dataset)
    _ = feature_index(feature_index_path(dataset, HEADING_STORAGE)).drop_runs(
        {old.heading}
    )

    outcome, _ = _retire(dataset, old.spec)

    assert kept_reasons(outcome) == {old.heading: "absent"}
    assert _removed(outcome) == [old.speed]


def test_a_request_of_a_retired_recipe_made_after_the_plan_keeps_its_runs(
    dataset: Dataset,
) -> None:
    old = OldRuns(dataset)
    again = run_pipeline(dataset, speed_heading_recipe())

    outcome, _ = _retire(dataset, old.spec)

    assert again.recipe_digest == old.digest
    assert kept_reasons(outcome) == {
        old.speed: "recipe_uses",
        old.heading: "recipe_uses",
    }
    assert _speed_exists(dataset, old.speed)
    assert _heading_exists(dataset, old.heading)


def test_the_judging_ranks_only_the_retired_runs(dataset: Dataset) -> None:
    """A newer run of the keeper's storage that no retired request wrote is no
    candidate, so the retired run stays the newest of its storage.

    The keeper's own run is computed, so the retired run covers no entry the
    keeper lacks.
    """
    old = OldRuns(dataset)
    _ = run_speed(dataset, 2)
    newer = run_pipeline(dataset, named_recipe(speed_recipe_stepping(3), "other"))
    keeper = _keeper(named_recipe(speed_recipe_stepping(2), "keeper"))
    spec = old.spec.model_copy(
        update={
            "keepers": [keeper],
            "policy": CleanupPolicy(keep_newest=1, min_age_hours=0),
        }
    )

    outcome, _ = _retire(dataset, spec)

    assert run_id_of(newer, "speed") != old.speed
    assert kept_reasons(outcome) == {old.speed: "newest", old.heading: "newest"}
    assert _speed_exists(dataset, old.speed)


@pytest.mark.parametrize(
    ("manager", "removed"),
    [("mosaic-api", True), ("someone-else", False), ("", False)],
)
def test_a_managed_dataset_keeps_every_run_unless_its_manager_runs_it(
    dataset: Dataset, manager: str, removed: bool
) -> None:
    ran = run_pipeline(dataset, speed_heading_recipe())
    managed = managed_dataset(dataset)
    spec = plan_retirement(managed, [ran.recipe_digest], [], LOOSE, "mosaic-api").spec

    outcome, _ = _retire(managed, spec, manager)

    assert len(spec.approved) == 2
    if removed:
        assert (len(outcome.removed), outcome.kept) == (2, ())
    else:
        assert outcome.removed == ()
        assert set(kept_reasons(outcome).values()) == {"managed_dataset"}
        assert _speed_exists(managed, run_id_of(ran, "speed"))


# --- losses, cancels and reruns ---------------------------------------------------


def test_a_removal_that_raises_is_a_lost_entry_and_the_others_go_on(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = OldRuns(dataset)
    rmtree = feature_run_removal.shutil.rmtree
    calls: list[Path] = []

    def refuse_first(path: Path) -> None:
        calls.append(path)
        if len(calls) == 1:
            raise OSError(f"busy: {path}")
        rmtree(path)

    monkeypatch.setattr(feature_run_removal.shutil, "rmtree", refuse_first)

    outcome, execution_id = _retire(dataset, old.spec)

    assert (outcome.status, len(outcome.removed), len(outcome.failed)) == (
        "finished",
        1,
        1,
    )
    snapshot = _snapshot(dataset, execution_id)
    assert (snapshot["status"], snapshot["entries_failed"]) == ("finished", 1)
    assert {record["state"] for record in _records(dataset, execution_id).values()} == {
        "removed",
        "failed",
    }


def test_every_removal_raising_fails_the_retirement(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = OldRuns(dataset)

    def refuse(path: Path) -> None:
        raise OSError(f"busy: {path}")

    monkeypatch.setattr(feature_run_removal.shutil, "rmtree", refuse)
    execution_id = new_execution_id()

    with pytest.raises(AllEntriesFailed, match="2 approved runs"):
        _ = _retire(dataset, old.spec, execution_id=execution_id)

    snapshot = _snapshot(dataset, execution_id)
    assert (snapshot["status"], snapshot["entries_failed"]) == ("failed", 2)


def test_a_kept_run_and_a_removal_that_raises_fail_the_retirement(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kept run is no loss: the one removal tried raised."""
    old = OldRuns(dataset)
    _ = record_open_attempt(dataset, old.heading, target=HEADING_STORAGE)

    def refuse(path: Path) -> None:
        raise OSError(f"busy: {path}")

    monkeypatch.setattr(feature_run_removal.shutil, "rmtree", refuse)
    execution_id = new_execution_id()

    with pytest.raises(AllEntriesFailed, match="1 approved runs"):
        _ = _retire(dataset, old.spec, execution_id=execution_id)

    snapshot = _snapshot(dataset, execution_id)
    assert (snapshot["status"], snapshot["entries_failed"]) == ("failed", 1)
    assert _records(dataset, execution_id)[old.heading]["reason"] == "attempt_open"


def test_a_rerun_after_an_earlier_removal_reports_a_raise_as_a_lost_entry(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = OldRuns(dataset)
    first, second = (target.run_id for target in old.spec.approved)
    token = CancelToken()
    execution_id = new_execution_id()
    with monkeypatch.context() as patched:
        observe_removals(patched, after=token.cancel)
        with pytest.raises(Cancelled):
            _ = _retire(dataset, old.spec, execution_id=execution_id, token=token)

    def refuse(path: Path) -> None:
        raise OSError(f"busy: {path}")

    monkeypatch.setattr(feature_run_removal.shutil, "rmtree", refuse)

    outcome, _ = _retire(dataset, old.spec, execution_id=execution_id)

    assert (outcome.status, outcome.removed) == ("finished", ())
    assert [entry.rpartition("/")[2] for entry in outcome.failed] == [second]
    snapshot = _snapshot(dataset, execution_id)
    assert snapshot["status"] == "finished"
    assert snapshot["entries_failed"] > 0
    assert _records(dataset, execution_id)[first]["state"] == "removed"


def test_a_cancel_stops_between_runs(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = OldRuns(dataset)
    token = CancelToken()
    observe_removals(monkeypatch, after=token.cancel)
    execution_id = new_execution_id()

    with pytest.raises(Cancelled):
        _ = _retire(dataset, old.spec, execution_id=execution_id, token=token)

    assert [
        _speed_exists(dataset, old.speed),
        _heading_exists(dataset, old.heading),
    ].count(True) == 1
    assert _snapshot(dataset, execution_id)["status"] == "cancelled"
    assert len(_records(dataset, execution_id)) == 1


def test_a_rerun_removes_nothing_twice_and_finishes_an_interrupted_removal(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = OldRuns(dataset)
    first, second = (target.run_id for target in old.spec.approved)
    rmtree = feature_run_removal.shutil.rmtree
    calls: list[Path] = []

    def killed_at_second(path: Path) -> None:
        calls.append(path)
        if len(calls) == 2:
            raise _Killed(str(path))
        rmtree(path)

    execution_id = new_execution_id()
    with monkeypatch.context() as patched:
        patched.setattr(feature_run_removal.shutil, "rmtree", killed_at_second)
        with pytest.raises(_Killed):
            _ = _retire(dataset, old.spec, execution_id=execution_id)
    states = {
        run_id: record["state"]
        for run_id, record in _records(dataset, execution_id).items()
    }
    assert states == {first: "removed", second: "interrupted"}
    removals: list[None] = []
    observe_removals(monkeypatch, before=lambda: removals.append(None))

    outcome, _ = _retire(dataset, old.spec, execution_id=execution_id)

    assert removals == []
    assert (_removed(outcome), outcome.kept) == ([second], ())
    assert {record["state"] for record in _records(dataset, execution_id).values()} == {
        "removed"
    }
    assert not _speed_exists(dataset, old.speed)
    assert not _heading_exists(dataset, old.heading)


# --- a definition -----------------------------------------------------------------


def test_a_retirement_of_a_definitions_runs_loads_no_definition(
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
    keepers = [_keeper(named_recipe(speed_recipe_stepping(2), "keeper"))]
    spec = plan_retirement(dataset, [ran.recipe_digest], keepers, LOOSE, "").spec
    forget_imported_definitions()
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))

    outcome, _ = _retire(dataset, spec)

    assert _removed(outcome) == [run_id_of(ran, "canary")]
    assert not imported_definitions()
    assert not canary.exists()
