"""Tests for the ``mosaic`` CLI (Layer 1 over the Job Contract).

Drives the Typer app with ``CliRunner`` against a real ``Dataset`` (built from a
manifest, with synthetic tracks) using only the lightweight ``speed-angvel``
feature -- so the suite runs under the default ``-m 'not slow'`` gate with no
torch/ultralytics. Asserts the ``--json`` stream-separation contract (one JSON
value on stdout; breadcrumbs on stderr).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from tests.helpers import invoke_json

runner = CliRunner()


# --- run -> status roundtrip ----------------------------------------------


def test_run_then_status_roundtrip(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    payload = invoke_json(
        ["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"]
    )

    assert (
        isinstance(payload["execution_id"], str)
        and len(str(payload["execution_id"])) == 26
    )
    # Read from the feature rather than written here as a literal: what this
    # asserts is that a run identifier carries its feature's declared version as
    # a visible segment, not that speed-angvel happens to be at a given one.
    assert str(payload["run_id"]).startswith(f"{SpeedAngvel.version}-")
    assert payload["cache_hit"] is False
    assert payload["status"] == "finished"

    status = invoke_json(
        [
            "status",
            "-m",
            str(manifest),
            "--execution-id",
            str(payload["execution_id"]),
            "--json",
        ]
    )
    assert status["status"] == "finished"
    assert status["run_id"] == payload["run_id"]
    assert status["kind"] == "feature"


def test_second_identical_run_is_cache_hit(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    first = invoke_json(
        ["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"]
    )
    second = invoke_json(
        ["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"]
    )
    assert second["cache_hit"] is True
    assert second["run_id"] == first["run_id"]
    assert second["execution_id"] != first["execution_id"]
    # The count is of what the scope *holds*, not of work done, so a cache hit
    # reports the same number as the run that did the computing. A version that
    # counted work would report 0 here, and a coverage bar built on it would show
    # a fully-computed dataset as empty the moment nothing was left to do.
    assert second["entries_written"] == first["entries_written"] == 2


def test_json_stream_separation(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    result = runner.invoke(
        app, ["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"]
    )
    assert result.exit_code == 0
    # stdout is exactly one JSON object (no stray prints).
    obj = json.loads(result.stdout)
    assert set(obj) == {
        "execution_id",
        "feature",
        "run_id",
        "status",
        "cache_hit",
        "failed_entries",
        "entries_written",
    }
    # ``failed_entries`` is always present rather than only when non-empty: this
    # payload is a machine contract, so a consumer should not have to tell an
    # absent key from an empty one to know whether a run lost anything.
    assert obj["status"] == "finished"
    assert obj["failed_entries"] == []
    # ``entries_written`` is present for the same reason, and carries the count
    # rather than a flag: "how much of this scope holds output now" is the
    # question, and a boolean cannot answer it for a partial run.
    assert obj["entries_written"] == 2
    # the execution_id breadcrumb went to stderr.
    assert "execution_id=" in result.stderr


def test_entries_scopes_to_one_sequence(dataset: tuple[Path, Dataset]) -> None:
    manifest, ds = dataset
    payload = invoke_json(
        [
            "run",
            "-m",
            str(manifest),
            "--feature",
            "speed-angvel",
            "--entries",
            "g:s1",
            "--json",
        ]
    )
    storage = str(payload["feature"])
    run_dir = ds.get_root("features") / storage / str(payload["run_id"])
    assert (run_dir / "g__s1.parquet").exists()
    assert not (run_dir / "g__s2.parquet").exists()


# --- observe ---------------------------------------------------------------


def test_runs_lists_the_attempt(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    run = invoke_json(
        ["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"]
    )
    rows = json.loads(
        runner.invoke(
            app, ["runs", "-m", str(manifest), "--kind", "feature", "--json"]
        ).stdout
    )
    assert any(r["execution_id"] == run["execution_id"] for r in rows)
    assert all(r["kind"] == "feature" for r in rows)


def test_cancel_on_finished_run_is_noop(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    run = invoke_json(
        ["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"]
    )
    res = invoke_json(
        [
            "cancel",
            "-m",
            str(manifest),
            "--execution-id",
            str(run["execution_id"]),
            "--json",
        ]
    )
    assert res["signalled"] is False
    assert res["status"] == "finished"


def _running_attempt(ds: Dataset, pid: int) -> str:
    """A run-log for an attempt that started on this host and never ended."""
    import socket

    from mosaic.runlog import JsonlRunLog, new_execution_id, run_log_path

    execution_id = new_execution_id()
    with JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id) as run_log:
        run_log.started(
            kind="op", target="train-sleap", host=socket.gethostname(), pid=pid
        )
    return execution_id


def test_cancel_on_a_dead_process_records_the_terminal_event(
    dataset: tuple[Path, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one state that was neither live nor reclaimable.

    ``inflight_state`` reads a holder as orphaned only once its run-log has
    gone terminal, so an attempt whose process died without writing one held
    its run root until the marker expired, with no gesture to release it.
    """
    import os

    from mosaic.runlog import read_run, run_log_dir

    manifest, ds = dataset
    execution_id = _running_attempt(ds, pid=424242)

    def gone(pid: int, sig: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr(os, "kill", gone)

    res = invoke_json(
        ["cancel", "-m", str(manifest), "--execution-id", execution_id, "--json"]
    )
    assert res["status"] == "cancelled"
    assert res["signalled"] is False

    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"


def test_a_reaped_attempt_no_longer_holds_its_run_root(
    dataset: tuple[Path, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the terminal event is *for*: the next attempt can have the root."""
    import os

    from mosaic.core.pipeline.markers import inflight_state, new_inflight

    manifest, ds = dataset
    execution_id = _running_attempt(ds, pid=424242)
    marker = new_inflight(
        execution_id=execution_id,
        host="",
        pid=424242,
        phase=None,
        idle_seconds=1800,
    )
    before = inflight_state(marker, run_log_base=ds.base_dir, execution_id="other")
    assert before == "live", "a running attempt holds its root"

    def gone(pid: int, sig: int) -> None:
        raise ProcessLookupError(pid)

    monkeypatch.setattr(os, "kill", gone)
    _ = invoke_json(
        ["cancel", "-m", str(manifest), "--execution-id", execution_id, "--json"]
    )

    after = inflight_state(marker, run_log_base=ds.base_dir, execution_id="other")
    assert after == "orphaned"


def test_cancel_on_a_live_process_leaves_the_run_log_alone(
    dataset: tuple[Path, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One file, one writer. The signalled process writes its own ending.

    Recording one here would put a second writer on a file whose first writer
    is still running, which is the thing the run-log format rules out.
    """
    import os

    from mosaic.runlog import read_run, run_log_dir

    manifest, ds = dataset
    execution_id = _running_attempt(ds, pid=424242)
    signalled: list[tuple[int, int]] = []

    def record(pid: int, sig: int) -> None:
        signalled.append((pid, sig))

    monkeypatch.setattr(os, "kill", record)

    res = invoke_json(
        ["cancel", "-m", str(manifest), "--execution-id", execution_id, "--json"]
    )
    assert res["signalled"] is True
    assert signalled == [(424242, 15)]

    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["status"] == "running"


# --- release ---------------------------------------------------------------


def _claimed_root(ds: Dataset, execution_id: str, pid: int, host: str) -> Path:
    """A run root carrying an in-flight claim, as an op would have left one."""
    from mosaic.core.pipeline.markers import new_inflight, try_create_inflight

    root = ds.base_dir / "models" / "train-sleap" / "train-sleap.0.1-abcdef0123"
    root.mkdir(parents=True, exist_ok=True)
    marker = new_inflight(
        execution_id=execution_id, host=host, pid=pid, phase=None, idle_seconds=1800
    )
    assert try_create_inflight(root, marker)
    return root


def test_release_frees_a_root_whose_process_is_gone(
    dataset: tuple[Path, Dataset],
) -> None:
    """The residual case: a claim with no terminal record anywhere to find.

    An untracked run leaves no run-log, so ``inflight_state`` can only wait out
    the marker's own expiry and the next attempt is refused for half an hour.
    """
    import socket

    from mosaic.core.pipeline.markers import INFLIGHT_MARKER_NAME

    manifest, ds = dataset
    root = _claimed_root(ds, "01ABANDONED", pid=424242, host=socket.gethostname())

    res = invoke_json(
        ["release", "-m", str(manifest), "--execution-id", "01ABANDONED", "--json"]
    )
    assert res["released"] == [str(root)]
    assert not (root / INFLIGHT_MARKER_NAME).exists()


def test_release_refuses_a_claim_held_from_another_host(
    dataset: tuple[Path, Dataset],
) -> None:
    """Deciding it here means guessing about a machine this process cannot see."""
    from mosaic.core.pipeline.markers import INFLIGHT_MARKER_NAME

    manifest, ds = dataset
    root = _claimed_root(ds, "01ELSEWHERE", pid=1, host="some-other-box")

    result = runner.invoke(
        app,
        ["release", "-m", str(manifest), "--execution-id", "01ELSEWHERE", "--json"],
    )
    assert result.exit_code != 0
    assert (root / INFLIGHT_MARKER_NAME).exists(), "the claim must survive a refusal"


def test_release_refuses_a_claim_whose_process_is_still_running(
    dataset: tuple[Path, Dataset],
) -> None:
    """A live run keeps its root; ``--force`` is what says otherwise."""
    import os
    import socket

    from mosaic.core.pipeline.markers import INFLIGHT_MARKER_NAME

    manifest, ds = dataset
    root = _claimed_root(ds, "01ALIVE", pid=os.getpid(), host=socket.gethostname())

    result = runner.invoke(
        app, ["release", "-m", str(manifest), "--execution-id", "01ALIVE", "--json"]
    )
    assert result.exit_code != 0
    assert (root / INFLIGHT_MARKER_NAME).exists()

    forced = invoke_json(
        [
            "release",
            "-m",
            str(manifest),
            "--execution-id",
            "01ALIVE",
            "--force",
            "--json",
        ]
    )
    assert forced["released"] == [str(root)]
    assert not (root / INFLIGHT_MARKER_NAME).exists()


def test_release_says_so_when_nothing_is_claimed(
    dataset: tuple[Path, Dataset],
) -> None:
    """Not an error: having nothing to release is the state the user wanted."""
    manifest, _ = dataset
    res = invoke_json(
        ["release", "-m", str(manifest), "--execution-id", "01NOTHING", "--json"]
    )
    assert res["released"] == []


def test_sequences(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    payload = invoke_json(["sequences", "-m", str(manifest), "--json"])
    assert payload["sequences"] == ["s1", "s2"]


def test_sequences_on_an_unconverted_dataset_says_what_to_run(
    dataset: tuple[Path, Dataset],
) -> None:
    """The one place the library's "absent is empty" must not stay silent.

    The library answers absent and empty alike; the CLI is the human boundary
    that turns "no rows" back into an instruction.
    """
    manifest, ds = dataset
    (ds.get_root("tracks") / "index.csv").unlink()

    result = runner.invoke(app, ["sequences", "-m", str(manifest)])
    assert result.exit_code != 0
    assert "convert tracks first" in result.stderr


def test_sequences_on_a_header_only_index_says_the_same_thing(
    dataset: tuple[Path, Dataset],
) -> None:
    """A header-only index is the same dataset state as an absent one.

    IndexCSV.ensure() makes it a common one, so the two must not diverge here.
    """
    from mosaic.core.pipeline.tracks_index import tracks_index, tracks_index_path

    manifest, ds = dataset
    path = tracks_index_path(ds)
    path.unlink()
    tracks_index(path).ensure()

    result = runner.invoke(app, ["sequences", "-m", str(manifest)])
    assert result.exit_code != 0
    assert "convert tracks first" in result.stderr


def test_sequences_narrowed_to_an_empty_group_still_succeeds(
    dataset: tuple[Path, Dataset],
) -> None:
    """--group matching nothing is not the same as having no tracks."""
    manifest, _ = dataset
    payload = invoke_json(
        ["sequences", "-m", str(manifest), "--group", "no-such-group", "--json"]
    )
    assert payload["sequences"] == []


# --- discovery -------------------------------------------------------------


def test_features_list_and_describe() -> None:
    rows = json.loads(runner.invoke(app, ["features", "list", "--json"]).stdout)
    names = {r["name"] for r in rows}
    assert "speed-angvel" in names

    desc = json.loads(
        runner.invoke(app, ["features", "describe", "speed-angvel", "--json"]).stdout
    )
    assert desc["name"] == "speed-angvel"
    assert "step_size" in desc["params_schema"]["properties"]


# Every registered op kind, as one literal. Separated from the discovery test
# below because the two answer different questions and change for different
# reasons: discovery asks whether ``tracking list`` and ``tracking describe``
# work, and is true of any non-empty registry, while this asks what is
# registered, and is false the moment anything is added. Held together, one
# literal governed both, so registering an op turned a test named for discovery
# red -- and two branches adding an op each edited the same assertion inside a
# test neither of them meant to touch.
_REGISTERED_OP_KINDS: frozenset[str] = frozenset(
    {
        "extract-frames",
        "train-pose",
        "train-points",
        "train-localizer",
        "infer-pose",
        "infer-points",
        "infer-localizer",
        "trex",
        "sleap",
        "litpose",
        "ultralytics",
        "convert-points",
        "prepare-training-data",
        "resample-tracks",
        "train-sleap",
        "train-litpose",
    }
)


def test_registered_op_kinds_are_exactly() -> None:
    """The registry's contents, pinned so an addition is a deliberate edit.

    Exact rather than a subset: an op that silently stops registering is as much
    a defect as one that appears unannounced, and only equality catches the first.
    """
    ops = json.loads(runner.invoke(app, ["tracking", "list", "--json"]).stdout)
    assert {o["kind"] for o in ops} == set(_REGISTERED_OP_KINDS)


def test_tracking_list_and_describe() -> None:
    """Discovery works: listing names kinds, describing one carries its schema."""
    ops = json.loads(runner.invoke(app, ["tracking", "list", "--json"]).stdout)
    kinds = {o["kind"] for o in ops}
    assert {"trex", "sleap", "litpose", "extract-frames"} <= kinds

    desc = json.loads(
        runner.invoke(app, ["tracking", "describe", "infer-pose", "--json"]).stdout
    )
    assert desc["kind"] == "infer-pose"
    assert "params_schema" in desc

    bogus = runner.invoke(app, ["tracking", "describe", "not-a-real-op", "--json"])
    assert bogus.exit_code == 1


# --- error paths -----------------------------------------------------------


def test_unknown_feature_lists_available(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    result = runner.invoke(
        app, ["run", "-m", str(manifest), "--feature", "no-such-feature"]
    )
    assert result.exit_code == 1
    assert "speed-angvel" in result.stderr


def test_feature_and_kind_are_mutually_exclusive(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    result = runner.invoke(
        app,
        [
            "run",
            "-m",
            str(manifest),
            "--feature",
            "speed-angvel",
            "--kind",
            "infer-pose",
        ],
    )
    assert result.exit_code == 1


def test_inputs_rejected_with_kind(dataset: tuple[Path, Dataset]) -> None:
    """An op declares its inputs in Params, where a feature takes them as a flag.

    ``--entries`` is no longer beside this one. Both arms now take the same
    three scope flags, covered in ``tests/cli/test_cli_run_scope.py``.
    """
    manifest, _ = dataset
    result = runner.invoke(
        app, ["run", "-m", str(manifest), "--kind", "infer-pose", "--inputs", '["x"]']
    )
    assert result.exit_code == 1
    assert "inputs" in result.stderr.lower()


def test_bad_params_json(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    result = runner.invoke(
        app,
        [
            "run",
            "-m",
            str(manifest),
            "--feature",
            "speed-angvel",
            "--params",
            "{not json}",
        ],
    )
    assert result.exit_code == 1
    assert "JSON" in result.stderr
