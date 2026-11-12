"""A step run over some of its entries, under an attempt other than its own.

A probe computes the largest few entries of a step's attempt to measure it, and a
named subset computes the entries a caller chose. Neither is the step's attempt:
the request's status reads only that one, and the step's own run afterward computes
what is left and serves the subset's entries from the cache.
"""

from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

import mosaic.core.pipeline.parallelism as parallelism
import mosaic.core.pipeline.graph.step as step_module
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.cancel import cancel_request
from mosaic.core.pipeline.graph import (
    BoundRef,
    NamedSubset,
    ProbeSubset,
    Recipe,
    Request,
    StepRefused,
    execute_step,
    plan_pipeline,
    request_status_wire,
    run_pipeline,
    run_status_wire,
    submit_request,
)
from mosaic.core.pipeline.graph.claims import FileFailureStore
from mosaic.core.pipeline.parallelism import ParallelSettings
from mosaic.run_cost import read_run_cost
from mosaic.runlog import (
    JsonlRunLog,
    list_runs,
    read_run,
    run_log_dir,
    run_log_path,
)
from tests.helpers import add_tracks_variant, make_dataset

VARIANT = "convert-trex.0.2-1111111111"
SPEED_STORAGE = "speed-angvel__from__tracks"

ROWS = {"seq_a": 40, "seq_b": 80, "seq_c": 120, "seq_d": 160, "seq_e": 200}
"""Five entries of distinct sizes, so the largest three are known."""

SPEED: dict[str, object] = {
    "id": "speed",
    "type": "feature",
    "feature": "speed-angvel",
    "inputs": ["tracks"],
}
TEMPLATES: dict[str, object] = {
    "id": "templates",
    "type": "feature",
    "feature": "extract-templates",
    "inputs": [{"step": "speed"}],
    "params": {"n_templates": 4},
}
REGRID: dict[str, object] = {
    "id": "regrid",
    "type": "op",
    "kind": "resample-tracks",
    "params": {"target_fps": 15.0},
}


def _recipe(*steps: dict[str, object]) -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": "subset", "steps": list(steps)}
    )


@pytest.fixture
def five(tmp_path: Path) -> Dataset:
    """Five track tables of distinct sizes, and nothing computed."""
    dataset = make_dataset(tmp_path / "five")
    for sequence, rows in ROWS.items():
        add_tracks_variant(
            dataset, VARIANT, sequence, n_rows=rows, std_format="mosaic_v1"
        )
    return dataset


def _events(ds: Dataset, execution_id: str, kind: str) -> list[dict[str, object]]:
    lines = run_log_path(ds.base_dir, execution_id).read_text().splitlines()
    return [record for record in map(json.loads, lines) if record["ev"] == kind]


def _computed(ds: Dataset, execution_id: str) -> list[str]:
    """The entries an attempt computed, by key, in the order they ended."""
    return [
        str(event["key"])
        for event in _events(ds, execution_id, "entry_end")
        if "wall_seconds" in event
    ]


def _step_status(ds: Dataset, request: Request, step_id: str) -> str:
    wire = request_status_wire(ds.base_dir, request)
    return next(step.status for step in wire.steps if step.step == step_id)


# --- a probe -------------------------------------------------------------------


def test_a_probe_computes_the_largest_entries_alone_then_in_a_pair(
    five: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallelism, "host_cpus", lambda: 4)
    request = submit_request(five, _recipe(SPEED)).request

    outcome = execute_step(
        five,
        request,
        "speed",
        execution_id="PROBEATTEMPT",
        subset=ProbeSubset(count=3),
    )

    assert outcome.subset == "probe"
    assert outcome.execution_id == "PROBEATTEMPT"
    assert outcome.entries_asked == (("", "seq_e"), ("", "seq_d"), ("", "seq_c"))
    assert sorted(_computed(five, "PROBEATTEMPT")) == ["seq_c", "seq_d", "seq_e"]
    (probe,) = _events(five, "PROBEATTEMPT", "probe")
    assert probe["kind"] == "probe"
    assert probe["step"] == "speed"
    assert probe["request_id"] == request.request_id
    assert probe["entries"] == ["seq_e", "seq_d", "seq_c"]
    in_flight = [
        event["in_flight"]
        for event in _events(five, "PROBEATTEMPT", "entry_end")
        if "in_flight" in event
    ]
    assert in_flight == [1, 2, 2]
    cost = read_run_cost(run_log_dir(five.base_dir), "PROBEATTEMPT")
    assert cost is not None
    assert cost.probe
    assert cost.thread_scaling is not None
    assert cost.alone_cpu_per_wall_second is not None
    assert cost.worker_base_rss_bytes is not None


def test_the_request_reads_the_steps_own_attempt_and_not_the_probe(
    five: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallelism, "host_cpus", lambda: 4)
    request = submit_request(five, _recipe(SPEED)).request

    _ = execute_step(
        five, request, "speed", execution_id="PROBEATTEMPT", subset=ProbeSubset(count=3)
    )

    assert _step_status(five, request, "speed") == ""
    assert request_status_wire(five.base_dir, request).status == "running"
    probe_run = run_status_wire(five.base_dir, "PROBEATTEMPT")
    assert probe_run is not None
    assert probe_run.probe
    assert probe_run.target == SPEED_STORAGE
    listed = list_runs(run_log_dir(five.base_dir))["items"]
    assert any(run["execution_id"] == "PROBEATTEMPT" and run["probe"] for run in listed)

    own = execute_step(five, request, "speed")

    assert own.subset == ""
    assert sorted(_computed(five, request.execution_of("speed"))) == [
        "seq_a",
        "seq_b",
    ]
    assert _step_status(five, request, "speed") == "finished"
    assert request_status_wire(five.base_dir, request).status == "finished"


def test_a_probe_takes_at_most_three_entries() -> None:
    """Two workers give one entry alone and then one pair; a fourth would make three."""
    assert ProbeSubset(count=3).count == 3
    with pytest.raises(ValidationError):
        _ = ProbeSubset(count=4)


def test_a_probe_leaves_out_an_entry_inside_its_backoff(
    five: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallelism, "host_cpus", lambda: 4)
    recipe = _recipe(SPEED)
    request = submit_request(five, recipe).request
    run_id = plan_pipeline(five, recipe).step("speed").run_id or ""
    _ = FileFailureStore(five.base_dir).note_entry_failure(
        (SPEED_STORAGE, run_id, "", "seq_e"), error="the entry failed"
    )

    _ = execute_step(
        five, request, "speed", execution_id="PROBEATTEMPT", subset=ProbeSubset(count=3)
    )

    assert sorted(_computed(five, "PROBEATTEMPT")) == ["seq_b", "seq_c", "seq_d"]


def test_a_probe_of_a_complete_step_is_answered_from_the_cache_and_marked(
    five: Dataset,
) -> None:
    request = submit_request(five, _recipe(SPEED)).request
    _ = execute_step(five, request, "speed")

    outcome = execute_step(
        five, request, "speed", execution_id="PROBECACHED", subset=ProbeSubset(count=3)
    )

    assert outcome.state == "cached"
    assert outcome.subset == "probe"
    snapshot = read_run(run_log_dir(five.base_dir), "PROBECACHED")
    assert snapshot is not None
    assert snapshot["probe"]
    assert snapshot["cache_hit"]


def test_a_probes_entry_failure_counts_toward_that_entrys_backoff(
    five: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel

    monkeypatch.setattr(parallelism, "host_cpus", lambda: 4)
    real_apply = SpeedAngvel.apply

    def apply(self: SpeedAngvel, df: pd.DataFrame) -> pd.DataFrame:
        if str(df["sequence"].iloc[0]) == "seq_d":
            msg = "apply failed on purpose"
            raise RuntimeError(msg)
        return real_apply(self, df)

    monkeypatch.setattr(SpeedAngvel, "apply", apply)
    recipe = _recipe(SPEED)
    request = submit_request(five, recipe).request
    run_id = plan_pipeline(five, recipe).step("speed").run_id or ""

    outcome = execute_step(
        five, request, "speed", execution_id="PROBEATTEMPT", subset=ProbeSubset(count=3)
    )

    assert outcome.failed_entries == ("seq_d",)
    store = FileFailureStore(five.base_dir)
    assert store.waiting_entries(SPEED_STORAGE, run_id, [("", s) for s in ROWS]) == {
        ("", "seq_d")
    }
    assert store.entry_record((SPEED_STORAGE, run_id, "", "seq_d")).attempts == 1
    assert store.entry_record((SPEED_STORAGE, run_id, "", "seq_e")).attempts == 0


def _asked(ds: Dataset, run_id: str) -> tuple[list[list[str]], bool]:
    """The ``_scope`` block's asked entries and ``asked_all`` of one run."""
    path = Path(ds.get_root("features")) / SPEED_STORAGE / run_id / "params.json"
    block = json.loads(path.read_text())["_scope"]
    return block["asked"], block["asked_all"]


def test_the_asked_scope_grows_by_the_subset_then_by_every_entry(
    five: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parallelism, "host_cpus", lambda: 4)
    request = submit_request(five, _recipe(SPEED)).request

    probe = execute_step(
        five, request, "speed", execution_id="PROBEATTEMPT", subset=ProbeSubset(count=2)
    )

    assert _asked(five, probe.run_id) == ([["", "seq_d"], ["", "seq_e"]], False)

    own = execute_step(five, request, "speed")

    assert own.run_id == probe.run_id
    assert _asked(five, own.run_id) == (
        [["", sequence] for sequence in sorted(ROWS)],
        True,
    )


def test_cancelling_the_request_does_not_reach_a_subset_attempt(
    five: Dataset,
) -> None:
    request = submit_request(five, _recipe(SPEED)).request
    gone = subprocess.Popen(["true"])
    _ = gone.wait()
    log = JsonlRunLog(run_log_path(five.base_dir, "PROBEATTEMPT"), "PROBEATTEMPT")
    try:
        # An unended subset attempt whose process has exited: a cancel that read
        # it would record it ``cancelled``.
        log.started(
            kind="feature",
            target=SPEED_STORAGE,
            host=socket.gethostname(),
            pid=gone.pid,
        )
        log.probe(
            request_id=request.request_id,
            step="speed",
            entries=["seq_e"],
            kind="probe",
        )
    finally:
        log.close()

    cancelled = cancel_request(five, request.request_id)

    assert "PROBEATTEMPT" not in {step.execution_id for step in cancelled.steps}
    snapshot = read_run(run_log_dir(five.base_dir), "PROBEATTEMPT")
    assert snapshot is not None
    assert snapshot["status"] == "running"


# --- a named subset ------------------------------------------------------------


def test_a_named_subset_runs_its_entries_with_the_callers_workers(
    five: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = submit_request(five, _recipe(SPEED)).request
    seen: dict[str, object] = {}
    real = step_module.run_feature

    def run_feature(*args: Any, **kwargs: Any) -> object:
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(step_module, "run_feature", run_feature)
    monkeypatch.setattr(parallelism, "host_cpus", lambda: 8)

    outcome = execute_step(
        five,
        request,
        "speed",
        execution_id="NAMEDATTEMPT",
        subset=NamedSubset(entries=(("", "seq_a"), ("", "seq_c"))),
        parallel=ParallelSettings(workers=3),
    )

    assert outcome.subset == "named"
    assert sorted(_computed(five, "NAMEDATTEMPT")) == ["seq_a", "seq_c"]
    assert seen["parallel_workers"] == 3
    assert seen["measure_worker_base"] is False
    (probe,) = _events(five, "NAMEDATTEMPT", "probe")
    assert probe["kind"] == "named"
    snapshot = read_run(run_log_dir(five.base_dir), "NAMEDATTEMPT")
    assert snapshot is not None
    assert not snapshot["probe"]


def test_two_named_subsets_and_the_steps_own_attempt_compute_each_entry_once(
    five: Dataset,
) -> None:
    request = submit_request(five, _recipe(SPEED)).request

    _ = execute_step(
        five,
        request,
        "speed",
        execution_id="ARRAYINDEX0",
        subset=NamedSubset(entries=(("", "seq_a"), ("", "seq_b"), ("", "seq_c"))),
    )
    _ = execute_step(
        five,
        request,
        "speed",
        execution_id="ARRAYINDEX1",
        subset=NamedSubset(entries=(("", "seq_d"), ("", "seq_e"))),
    )
    own = execute_step(five, request, "speed")

    computed = (
        _computed(five, "ARRAYINDEX0")
        + _computed(five, "ARRAYINDEX1")
        + _computed(five, request.execution_of("speed"))
    )
    assert sorted(computed) == sorted(ROWS)
    assert own.covered == own.target == len(ROWS)
    assert request_status_wire(five.base_dir, request).status == "finished"


# --- refusals ------------------------------------------------------------------


def _refused_cause(
    ds: Dataset, request: Request, step_id: str, subset: ProbeSubset | NamedSubset
) -> str:
    with pytest.raises(StepRefused) as caught:
        _ = execute_step(ds, request, step_id, execution_id="REFUSED", subset=subset)
    assert caught.value.reason == "subset_refused"
    snapshot = read_run(run_log_dir(ds.base_dir), "REFUSED")
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert not snapshot["cache_hit"]
    assert not snapshot["run_id"]
    cause = caught.value.detail["cause"]
    assert isinstance(cause, str)
    return cause


def test_a_scope_dependent_step_is_refused(five: Dataset) -> None:
    request = submit_request(five, _recipe(SPEED, TEMPLATES)).request
    _ = execute_step(five, request, "speed")

    assert (
        _refused_cause(five, request, "templates", ProbeSubset(count=2))
        == "scope_dependent"
    )


def test_an_op_step_is_refused(five: Dataset) -> None:
    request = submit_request(five, _recipe(REGRID)).request

    assert _refused_cause(five, request, "regrid", ProbeSubset(count=2)) == "op_step"


def test_a_bound_step_is_refused_and_not_recorded_as_bound(five: Dataset) -> None:
    done = run_pipeline(five, _recipe(SPEED))
    ran = done.outcomes[0].run_id
    bind = {"speed": BoundRef(feature=SPEED_STORAGE, run_id=ran)}
    request = submit_request(five, _recipe(SPEED), bind=bind).request

    assert _refused_cause(five, request, "speed", ProbeSubset(count=2)) == "stand_in"


def test_a_pinned_complete_step_is_refused_and_not_recorded_as_cached(
    five: Dataset,
) -> None:
    done = run_pipeline(five, _recipe(SPEED, TEMPLATES))
    fitted = next(o.run_id for o in done.outcomes if o.step_id == "templates")
    request = submit_request(
        five, _recipe(SPEED, TEMPLATES), pins={"templates": fitted}
    ).request

    assert (
        _refused_cause(five, request, "templates", ProbeSubset(count=2)) == "stand_in"
    )


def test_an_entry_outside_the_target_is_refused(five: Dataset) -> None:
    request = submit_request(five, _recipe(SPEED)).request
    subset = NamedSubset(entries=(("", "seq_a"), ("", "seq_z")))

    assert _refused_cause(five, request, "speed", subset) == "outside_target"


def test_the_steps_own_execution_id_is_refused_before_anything_is_recorded(
    five: Dataset,
) -> None:
    request = submit_request(five, _recipe(SPEED)).request
    own = request.execution_of("speed")

    for execution_id in (own, ""):
        with pytest.raises(ValueError, match="execution id"):
            _ = execute_step(
                five,
                request,
                "speed",
                execution_id=execution_id,
                subset=ProbeSubset(count=2),
            )

    assert read_run(run_log_dir(five.base_dir), own) is None
