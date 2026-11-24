"""A settle-only attempt records a step that needs no work, and writes nothing else.

``execute_step(settle_only=True)`` answers a bound step or one whose plan reads
complete with the record an ordinary attempt writes, and declines every other
step before any file is written: it never reaches the preflight, the quarantine
narrowing or a run. ``settle_step`` measures a step as ``step_input_size`` does and
settles it only when nothing is left to compute.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    BoundRef,
    NamedSubset,
    Request,
    execute_step,
    load_recipe_for_request,
    request_rollup,
    request_status_wire,
    run_pipeline,
    submit_request,
)
from mosaic.core.pipeline.graph import step as step_module
from mosaic.core.pipeline.graph.claims import FileFailureStore
from mosaic.core.pipeline.graph.input_size import step_input_size
from mosaic.core.pipeline.graph.parents import pinned_parents
from mosaic.core.pipeline.graph.settle import (
    StepSettleOutcomeWire,
    StepSettleWire,
    settle_step,
)
from mosaic.core.pipeline.graph.step import SettleDeclined, StepOutcome
from mosaic.core.pipeline.graph.step_preflight import StepRefused
from mosaic.core.pipeline.job import Cancelled, CancelToken
from mosaic.core.pipeline.refusal import Refusal, ToolUnavailable
from mosaic.runlog import read_run, read_run_records, run_log_dir, run_log_path
from tests.helpers import (
    SPEED_OF_CONVERTED_STEP,
    add_tracks_variant,
    assert_tree_unchanged,
    calms21_dataset,
    chain_recipe,
    converting_recipe,
    make_dataset,
    quarantine_entry,
    run_speed,
    tree_snapshot,
)

VARIANT = "convert-trex.0.2-1111111111"

SEQUENCES = ("seq_a", "seq_b", "seq_c", "seq_d")


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Four tables of tracks, and nothing computed."""
    dataset = make_dataset(tmp_path / "tracked")
    for sequence in SEQUENCES:
        add_tracks_variant(
            dataset, VARIANT, sequence, n_rows=12, std_format="mosaic_v1"
        )
    return dataset


def _events(dataset: Dataset, execution_id: str) -> list[object]:
    return [
        record["ev"]
        for record in read_run_records(run_log_dir(dataset.base_dir), execution_id)
    ]


def _settled_fields(dataset: Dataset, execution_id: str) -> Mapping[str, object]:
    """The fields of an attempt's run-log that an answer from the plan records."""
    snapshot = read_run(run_log_dir(dataset.base_dir), execution_id)
    assert snapshot is not None
    return {
        "status": snapshot["status"],
        "cache_hit": snapshot["cache_hit"],
        "run_id": snapshot["run_id"],
        "entries_written": snapshot["entries_written"],
        "target": snapshot["target"],
        "kind": snapshot["kind"],
    }


def _declined(
    dataset: Dataset, request: Request, step_id: str = "speed"
) -> SettleDeclined:
    """Settle *step_id* and return its decline, with the tree left as it was."""
    before = tree_snapshot(dataset.base_dir)
    with pytest.raises(SettleDeclined) as raised:
        _ = execute_step(dataset, request, step_id, settle_only=True)
    assert_tree_unchanged(dataset.base_dir, before)
    return raised.value


def _raise_tool_unavailable(*_args: object, **_kwargs: object) -> NoReturn:
    raise ToolUnavailable("stub-tool", "environment", "the stub tool is missing")


def test_a_settle_decline_is_not_a_refusal() -> None:
    assert not issubclass(SettleDeclined, Refusal)
    declined = SettleDeclined("speed", "refused", refusal="version_moved")
    assert (declined.step_id, declined.reason, declined.refusal) == (
        "speed",
        "refused",
        "version_moved",
    )
    assert SettleDeclined("speed", "would_compute").refusal == ""


def test_a_complete_step_settles_with_the_record_of_a_cached_attempt(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    computed = submit_request(tracked, recipe).request
    _ = execute_step(tracked, computed, "speed")
    settling = submit_request(tracked, recipe).request
    ordinary = submit_request(tracked, recipe).request
    assert len({computed.request_id, settling.request_id, ordinary.request_id}) == 3

    settled = execute_step(tracked, settling, "speed", settle_only=True)
    cached = execute_step(tracked, ordinary, "speed")

    assert isinstance(settled, StepOutcome)
    assert settled.state == "cached"
    assert settled.execution_id == settling.execution_of("speed")
    assert settled.run_id == cached.run_id
    attempt = settling.execution_of("speed")
    fields = _settled_fields(tracked, attempt)
    assert fields == _settled_fields(tracked, ordinary.execution_of("speed"))
    assert fields["status"] == "finished"
    assert fields["cache_hit"] is True
    assert fields["run_id"] == settled.run_id
    assert fields["entries_written"] == len(SEQUENCES)
    assert _events(tracked, attempt) == [
        "started",
        "run_id",
        "cache_hit",
        "entries_written",
        "finished",
    ]


def test_a_bound_step_settles_with_the_record_of_a_bound_attempt(
    tracked: Dataset,
) -> None:
    older = run_speed(tracked, 3)
    bind = {"speed": BoundRef(feature="speed-angvel__from__tracks", run_id=older)}
    recipe = chain_recipe()
    settling = submit_request(tracked, recipe, bind=bind).request
    ordinary = submit_request(tracked, recipe, bind=bind).request

    settled = execute_step(tracked, settling, "speed", settle_only=True)
    bound = execute_step(tracked, ordinary, "speed")

    assert settled.state == bound.state == "bound"
    assert settled.run_id == older
    attempt = settling.execution_of("speed")
    fields = _settled_fields(tracked, attempt)
    assert fields == _settled_fields(tracked, ordinary.execution_of("speed"))
    assert fields["run_id"] == older
    assert fields["cache_hit"] is True
    assert _events(tracked, attempt) == _events(tracked, ordinary.execution_of("speed"))


def test_a_step_with_entries_to_compute_declines_and_writes_nothing(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request

    declined = _declined(tracked, request)

    assert (declined.step_id, declined.reason, declined.refusal) == (
        "speed",
        "would_compute",
        "",
    )


def test_a_step_that_only_quarantine_narrowing_completes_declines(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    first = submit_request(tracked, recipe).request
    _ = execute_step(
        tracked,
        first,
        "speed",
        subset=NamedSubset(entries=(("", "seq_a"), ("", "seq_c"), ("", "seq_d"))),
        execution_id="named-subset",
    )
    request = submit_request(tracked, recipe).request.model_copy(
        update={"allow_partial": True}
    )
    quarantine_entry(tracked, recipe, "speed", "seq_b")

    declined = _declined(tracked, request)

    assert declined.reason == "would_compute"
    store = FileFailureStore(tracked.base_dir)
    assert set(store.exclusions(request.request_id).entries) == set()
    # An ordinary attempt leaves the step complete by narrowing it, so the
    # decline is what keeps the settle from doing that.
    ordinary = execute_step(tracked, request, "speed")
    assert ordinary.state == "cached"


def test_a_moved_version_declines_as_refused_where_an_attempt_records_it(
    tracked: Dataset,
) -> None:
    submitted = submit_request(tracked, chain_recipe()).request
    moved = submitted.model_copy(
        update={"step_versions": {"speed": "0.0-not-the-installed-one"}}
    )

    declined = _declined(tracked, moved)

    assert declined.reason == "refused"
    assert declined.refusal == "version_moved"
    assert isinstance(declined.__cause__, StepRefused)
    with pytest.raises(StepRefused):
        _ = execute_step(tracked, moved, "speed")
    snapshot = read_run(run_log_dir(tracked.base_dir), moved.execution_of("speed"))
    assert snapshot is not None
    assert snapshot["status"] == "failed"


def test_a_tool_unavailable_from_a_check_declines_as_refused(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    monkeypatch.setattr(step_module, "refuse_mixed_schemas", _raise_tool_unavailable)

    declined = _declined(tracked, request)

    assert declined.reason == "refused"
    assert declined.refusal == "tool_unavailable"
    assert isinstance(declined.__cause__, ToolUnavailable)


def test_a_complete_step_settles_when_the_catalog_lacks_its_tool(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = chain_recipe()
    _ = execute_step(tracked, submit_request(tracked, recipe).request, "speed")
    settling = submit_request(tracked, recipe).request
    other = submit_request(tracked, recipe).request
    missing = ToolUnavailable("stub-tool", "environment", "the stub tool is missing")

    def lacks_the_tool(*_args: object, **_kwargs: object) -> ToolUnavailable:
        return missing

    monkeypatch.setattr(step_module, "unavailable_tool", lacks_the_tool)

    settled = execute_step(tracked, settling, "speed", settle_only=True)

    assert settled.state == "cached"
    with pytest.raises(ToolUnavailable):
        _ = execute_step(tracked, other, "speed")


def test_a_cancel_before_the_record_writes_nothing(tracked: Dataset) -> None:
    recipe = chain_recipe()
    _ = execute_step(tracked, submit_request(tracked, recipe).request, "speed")
    request = submit_request(tracked, recipe).request
    token = CancelToken()
    token.cancel()
    before = tree_snapshot(tracked.base_dir)

    with pytest.raises(Cancelled):
        _ = execute_step(
            tracked, request, "speed", settle_only=True, cancel_token=token
        )

    assert_tree_unchanged(tracked.base_dir, before)


def test_a_cancel_before_a_bound_record_writes_nothing(tracked: Dataset) -> None:
    older = run_speed(tracked, 3)
    bind = {"speed": BoundRef(feature="speed-angvel__from__tracks", run_id=older)}
    request = submit_request(tracked, chain_recipe(), bind=bind).request
    token = CancelToken()
    token.cancel()
    before = tree_snapshot(tracked.base_dir)

    with pytest.raises(Cancelled):
        _ = execute_step(
            tracked, request, "speed", settle_only=True, cancel_token=token
        )

    assert_tree_unchanged(tracked.base_dir, before)


def test_another_error_propagates_and_writes_nothing(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = submit_request(tracked, chain_recipe()).request

    def fail(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("the plan broke")

    monkeypatch.setattr(step_module, "_plan", fail)
    before = tree_snapshot(tracked.base_dir)

    with pytest.raises(RuntimeError, match="the plan broke"):
        _ = execute_step(tracked, request, "speed", settle_only=True)

    assert_tree_unchanged(tracked.base_dir, before)


def test_settle_only_with_a_subset_raises_before_anything_is_written(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    before = tree_snapshot(tracked.base_dir)

    with pytest.raises(ValueError, match="subset"):
        _ = execute_step(
            tracked,
            request,
            "speed",
            settle_only=True,
            subset=NamedSubset(entries=(("", "seq_a"),)),
            execution_id="named-subset",
        )

    assert_tree_unchanged(tracked.base_dir, before)


def test_settle_only_with_overwrite_raises_before_anything_is_written(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    before = tree_snapshot(tracked.base_dir)

    with pytest.raises(ValueError, match="overwrite"):
        _ = execute_step(tracked, request, "speed", settle_only=True, overwrite=True)

    assert_tree_unchanged(tracked.base_dir, before)


def test_settle_only_with_refit_raises_before_anything_is_written(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    before = tree_snapshot(tracked.base_dir)

    with pytest.raises(ValueError, match="refit"):
        _ = execute_step(tracked, request, "speed", settle_only=True, refit=True)

    assert_tree_unchanged(tracked.base_dir, before)


def test_a_settle_of_a_step_its_request_fits_again_declines_as_would_compute(
    tracked: Dataset,
) -> None:
    _ = run_pipeline(tracked, chain_recipe())
    request = submit_request(tracked, chain_recipe(), refit=["templates"]).request
    _ = execute_step(tracked, request, "speed")

    declined = _declined(tracked, request, "templates")

    assert declined.reason == "would_compute"


def test_settle_only_under_another_execution_id_raises_before_anything_is_written(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    before = tree_snapshot(tracked.base_dir)

    with pytest.raises(ValueError, match="execution id"):
        _ = execute_step(
            tracked, request, "speed", settle_only=True, execution_id="another"
        )

    assert_tree_unchanged(tracked.base_dir, before)


# --- settle_step -------------------------------------------------------------

OWNER = "gate"


def _size_fields(wire: StepSettleWire) -> dict[str, object]:
    """Every field of a settle report that an ``input-size`` report has."""
    return wire.model_dump(exclude={"settle"})


def _computed_by_another_request(dataset: Dataset) -> str:
    """Compute ``speed`` under a request of its own and return its run id."""
    other = submit_request(dataset, chain_recipe()).request
    return execute_step(dataset, other, "speed").run_id


def _settle_without_writing(
    dataset: Dataset, request: Request, step_id: str
) -> StepSettleOutcomeWire:
    """Settle *step_id*, and return its outcome with the tree left as it was."""
    before = tree_snapshot(dataset.base_dir)
    outcome = settle_step(dataset, request, step_id, owner=OWNER).settle
    assert_tree_unchanged(dataset.base_dir, before)
    return outcome


def test_a_complete_step_settles_as_cached_with_the_size_report_unchanged(
    tracked: Dataset,
) -> None:
    computed_run = _computed_by_another_request(tracked)
    request = submit_request(tracked, chain_recipe()).request
    sized = step_input_size(tracked, request, "speed")

    wire = settle_step(tracked, request, "speed", owner=OWNER)

    assert wire.settle == StepSettleOutcomeWire(
        outcome="settled",
        reason="",
        refusal="",
        execution_id=request.execution_of("speed"),
        state="cached",
        run_id=sized.run_id,
    )
    assert sized.run_id == computed_run
    assert _size_fields(wire) == sized.model_dump()
    snapshot = read_run(run_log_dir(tracked.base_dir), request.execution_of("speed"))
    assert snapshot is not None
    assert snapshot["status"] == "finished"
    assert snapshot["run_id"] == sized.run_id


def test_a_settled_run_id_is_the_one_an_ordinary_attempt_records(
    tracked: Dataset,
) -> None:
    _ = _computed_by_another_request(tracked)
    settling = submit_request(tracked, chain_recipe()).request
    ordinary = submit_request(tracked, chain_recipe()).request

    wire = settle_step(tracked, settling, "speed", owner=OWNER)
    outcome = execute_step(tracked, ordinary, "speed")

    assert wire.settle.run_id == outcome.run_id


def test_a_bound_step_settles_as_bound(tracked: Dataset) -> None:
    older = run_speed(tracked, 3)
    bind = {"speed": BoundRef(feature="speed-angvel__from__tracks", run_id=older)}
    request = submit_request(tracked, chain_recipe(), bind=bind).request

    wire = settle_step(tracked, request, "speed", owner=OWNER)

    assert (wire.settle.outcome, wire.settle.state) == ("settled", "bound")
    assert wire.settle.run_id == older
    assert wire.settle.execution_id == request.execution_of("speed")


def test_an_op_step_is_not_attempted(tmp_path: Path) -> None:
    dataset = calms21_dataset(
        tmp_path / "ds", {"train": {"a": {"s1": 40}}}, seed=3, n_classes=2
    )
    request = submit_request(dataset, converting_recipe()).request

    outcome = _settle_without_writing(dataset, request, "tracks")

    assert (outcome.outcome, outcome.reason) == ("not_attempted", "op_step")
    assert (outcome.state, outcome.run_id) == (None, None)
    assert outcome.execution_id == request.execution_of("tracks")


def test_a_step_below_an_unrun_op_is_not_attempted_for_its_inputs(
    tmp_path: Path,
) -> None:
    dataset = calms21_dataset(
        tmp_path / "ds", {"train": {"a": {"s1": 40, "s2": 30}}}, seed=3, n_classes=2
    )
    request = submit_request(
        dataset, converting_recipe(SPEED_OF_CONVERTED_STEP)
    ).request

    outcome = _settle_without_writing(dataset, request, "speed")

    assert (outcome.outcome, outcome.reason) == ("not_attempted", "inputs_not_ready")


def test_a_fresh_step_is_not_attempted_for_its_entries_to_compute(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request

    outcome = _settle_without_writing(tracked, request, "speed")

    assert (outcome.outcome, outcome.reason) == (
        "not_attempted",
        "entries_to_compute",
    )
    assert outcome.refusal == ""


def test_a_step_with_a_run_log_is_declined_and_its_run_log_is_left_alone(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    _ = execute_step(tracked, request, "speed")
    assert run_log_path(tracked.base_dir, request.execution_of("speed")).exists()

    outcome = _settle_without_writing(tracked, request, "speed")

    assert (outcome.outcome, outcome.reason) == ("declined", "attempt_recorded")
    assert (outcome.state, outcome.run_id) == (None, None)


def test_a_step_that_only_narrowing_completes_is_declined_as_would_compute(
    tracked: Dataset,
) -> None:
    recipe = chain_recipe()
    first = submit_request(tracked, recipe).request
    _ = execute_step(
        tracked,
        first,
        "speed",
        subset=NamedSubset(entries=(("", "seq_a"), ("", "seq_c"), ("", "seq_d"))),
        execution_id="named-subset",
    )
    request = submit_request(tracked, recipe).request.model_copy(
        update={"allow_partial": True}
    )
    quarantine_entry(tracked, recipe, "speed", "seq_b")

    outcome = _settle_without_writing(tracked, request, "speed")

    assert (outcome.outcome, outcome.reason) == ("declined", "would_compute")


def test_a_refusal_of_the_settle_is_declined_with_its_reason(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _computed_by_another_request(tracked)
    request = submit_request(tracked, chain_recipe()).request
    monkeypatch.setattr(step_module, "refuse_mixed_schemas", _raise_tool_unavailable)

    outcome = _settle_without_writing(tracked, request, "speed")

    assert (outcome.outcome, outcome.reason, outcome.refusal) == (
        "declined",
        "refused",
        "tool_unavailable",
    )
    assert (outcome.state, outcome.run_id) == (None, None)


def test_a_refusal_of_the_size_read_propagates_and_writes_nothing(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    log_path = run_log_path(tracked.base_dir, request.execution_of("speed"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _ = log_path.write_text(
        json.dumps({"t": "2026-01-01T00:00:00+00:00", "ev": "started"})
        + "\n"
        + json.dumps({"t": "2026-01-01T00:00:01+00:00", "ev": "finished"})
        + "\n"
    )
    before = tree_snapshot(tracked.base_dir)

    with pytest.raises(StepRefused) as raised:
        _ = settle_step(tracked, request, "templates", owner=OWNER)

    assert raised.value.reason == "parent_unrecorded"
    assert_tree_unchanged(tracked.base_dir, before)


def test_readers_see_a_settled_step_as_a_cached_attempt(tracked: Dataset) -> None:
    run_id = _computed_by_another_request(tracked)
    request = submit_request(tracked, chain_recipe()).request

    wire = settle_step(tracked, request, "speed", owner=OWNER)

    assert wire.settle.outcome == "settled"
    recipe = load_recipe_for_request(tracked.base_dir, request)
    assert pinned_parents(tracked, recipe, request, "templates") == {"speed": run_id}
    [rolled] = [
        attempt
        for attempt in request_rollup(tracked.base_dir, request).steps
        if attempt.step_id == "speed"
    ]
    assert (rolled.status, rolled.run_id, rolled.cache_hit) == (
        "finished",
        run_id,
        True,
    )
    [reported] = [
        step
        for step in request_status_wire(tracked.base_dir, request).steps
        if step.step == "speed"
    ]
    assert (reported.outcome, reported.run_id, reported.cache_hit) == (
        "finished",
        run_id,
        True,
    )


def test_a_cancelled_token_raises_before_the_settle_writes(tracked: Dataset) -> None:
    _ = _computed_by_another_request(tracked)
    request = submit_request(tracked, chain_recipe()).request
    token = CancelToken()
    token.cancel()
    before = tree_snapshot(tracked.base_dir)

    with pytest.raises(Cancelled):
        _ = settle_step(tracked, request, "speed", owner=OWNER, cancel_token=token)

    assert_tree_unchanged(tracked.base_dir, before)


def test_a_cancelled_token_raises_for_a_step_that_would_not_be_attempted(
    tracked: Dataset,
) -> None:
    request = submit_request(tracked, chain_recipe()).request
    token = CancelToken()
    token.cancel()
    before = tree_snapshot(tracked.base_dir)

    with pytest.raises(Cancelled):
        _ = settle_step(tracked, request, "speed", owner=OWNER, cancel_token=token)

    assert_tree_unchanged(tracked.base_dir, before)
