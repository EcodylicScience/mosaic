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
from tests.helpers import SLEEPER_COMMAND, invoke_json, speed_recipe

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
    dataset: tuple[Path, Dataset],
) -> None:
    """One file, one writer. The signalled process writes its own ending.

    Recording one here would put a second writer on a file whose first writer
    is still running, which is the thing the run-log format rules out.
    """
    import signal
    import subprocess

    from mosaic.runlog import read_run, run_log_dir

    manifest, ds = dataset
    sleeper = subprocess.Popen([*SLEEPER_COMMAND, "600"], stdin=subprocess.DEVNULL)
    execution_id = _running_attempt(ds, pid=sleeper.pid)
    try:
        res = invoke_json(
            ["cancel", "-m", str(manifest), "--execution-id", execution_id, "--json"]
        )
        assert res["signalled"] is True
        assert sleeper.wait(timeout=10) == -signal.SIGTERM
    finally:
        sleeper.kill()
        _ = sleeper.wait()

    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["status"] == "running"


def test_cancel_refuses_an_attempt_on_another_host(
    dataset: tuple[Path, Dataset],
) -> None:
    from mosaic.runlog import JsonlRunLog, new_execution_id, run_log_path

    manifest, ds = dataset
    execution_id = new_execution_id()
    with JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id) as log:
        log.started(kind="op", target="train-sleap", host="elsewhere", pid=424242)

    result = runner.invoke(
        app, ["cancel", "-m", str(manifest), "--execution-id", execution_id]
    )

    assert result.exit_code == 1
    assert "host 'elsewhere'" in result.stderr


def test_cancel_assume_exited_records_an_attempt_on_another_host(
    dataset: tuple[Path, Dataset],
) -> None:
    from mosaic.runlog import JsonlRunLog, new_execution_id, read_run, run_log_path

    manifest, ds = dataset
    execution_id = new_execution_id()
    with JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id) as log:
        log.started(kind="op", target="train-sleap", host="elsewhere", pid=424242)
    argv = ["cancel", "-m", str(manifest), "--execution-id", execution_id]

    refused = runner.invoke(app, argv)
    res = invoke_json([*argv, "--assume-exited", "--json"])

    assert refused.exit_code == 1
    assert "--assume-exited" in refused.stderr
    assert (res["outcome"], res["status"]) == ("recorded", "cancelled")
    snapshot = read_run(ds.base_dir / ".mosaic" / "runs", execution_id)
    assert snapshot is not None
    assert snapshot["status"] == "cancelled"


def test_cancel_records_an_unstarted_request_cancelled(
    dataset: tuple[Path, Dataset],
) -> None:
    from mosaic.core.pipeline.graph import request_rollup, submit_request

    manifest, ds = dataset
    request = submit_request(ds, speed_recipe()).request

    res = invoke_json(
        ["cancel", "-m", str(manifest), "--request", request.request_id, "--json"]
    )

    assert (res["outcome"], res["status"], res["driver_pid"]) == (
        "recorded",
        "cancelled",
        0,
    )
    assert request_rollup(ds.base_dir, request).status == "cancelled"


@pytest.mark.parametrize(
    "flags",
    [[], ["--execution-id", "a", "--request", "b"]],
    ids=["neither", "both"],
)
def test_cancel_takes_exactly_one_of_an_attempt_and_a_request(
    dataset: tuple[Path, Dataset], flags: list[str]
) -> None:
    manifest, _ = dataset

    result = runner.invoke(app, ["cancel", "-m", str(manifest), *flags])

    assert result.exit_code == 1
    assert "exactly one of --execution-id or --request" in result.stderr


STOPPED_STEP_PROGRAM = (
    "import sys, time\n"
    "from mosaic.core.pipeline.graph import run\n"
    "from mosaic.core.pipeline.job import Cancelled\n"
    "def waits(ds, request, step_id, *, cancel_token, **_):\n"
    "    while not cancel_token.is_cancelled():\n"
    "        time.sleep(0.05)\n"
    "    raise Cancelled()\n"
    "run.execute_step = waits\n"
    "from mosaic.cli import app\n"
    "app(sys.argv[1:], prog_name='mosaic')\n"
)
"""``mosaic`` with every step of an in-process run waiting for its cancel token."""


def test_pipeline_run_recipe_exits_130_on_sigterm(
    dataset: tuple[Path, Dataset], tmp_path: Path
) -> None:
    import signal
    import subprocess
    import sys

    from mosaic.core.pipeline.graph import (
        driver_lease_path,
        drivers_root,
        list_requests,
        load_request,
        read_driver_lease,
        request_rollup,
    )
    from tests.helpers import wait_for

    manifest, ds = dataset
    recipe = tmp_path / "speed.json"
    _ = recipe.write_text(speed_recipe().model_dump_json())

    def step_lease_named() -> bool:
        leases = list(drivers_root(ds.base_dir).glob("*.lease.json"))
        if not leases:
            return False
        lease = read_driver_lease(ds.base_dir, leases[0].name.split(".")[0])
        return lease is not None and lease.step_id == "speed"

    with (tmp_path / "run.out").open("w") as output:
        driver = subprocess.Popen(
            [
                sys.executable,
                "-c",
                STOPPED_STEP_PROGRAM,
                "pipeline",
                "run",
                "-m",
                str(manifest),
                "-r",
                f"@{recipe}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
        )
    try:
        wait_for(step_lease_named, 180.0, "the run did not start its step")
        driver.send_signal(signal.SIGTERM)
        assert driver.wait(timeout=180) == 130
    finally:
        if driver.poll() is None:
            driver.kill()
            _ = driver.wait()

    (summary,) = list_requests(ds.base_dir).items
    request = load_request(ds.base_dir, summary.request_id)
    assert request_rollup(ds.base_dir, request).status == "cancelled"
    assert not driver_lease_path(ds.base_dir, request.request_id).exists()


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
    by_name = {r["name"]: r for r in rows}
    assert "speed-angvel" in by_name
    assert {r["type"] for r in rows} == {"feature"}
    assert by_name["speed-angvel"]["lane"] == "feature-compute"
    assert "params_schema" not in by_name["speed-angvel"]
    assert by_name["speed-angvel"]["license_note"] is None
    assert "non-commercial" in by_name["kpms"]["license_note"]

    desc = json.loads(
        runner.invoke(app, ["features", "describe", "speed-angvel", "--json"]).stdout
    )
    assert desc["name"] == "speed-angvel"
    assert "step_size" in [row["path"] for row in desc["params"]]
    assert desc["params_schema"] is None
    assert {"version", "category", "requires", "provides", "unavailable"} <= set(desc)
    assert desc["unavailable"] is None
    assert desc["provides"] and all(
        isinstance(label, str) for label in desc["provides"]
    )
    assert desc["lane"] == "feature-compute"

    schema = json.loads(
        runner.invoke(
            app,
            ["features", "describe", "speed-angvel", "--detail", "schema", "--json"],
        ).stdout
    )
    assert "step_size" in schema["params_schema"]["properties"]
    plain = runner.invoke(app, ["features", "describe", "speed-angvel"])
    assert plain.exit_code == 0
    assert any(line.startswith("step_size\t") for line in plain.stdout.splitlines())

    kpms = json.loads(
        runner.invoke(app, ["features", "describe", "kpms", "--json"]).stdout
    )
    assert kpms["license_note"] == by_name["kpms"]["license_note"]


def test_features_list_ranks_the_features_a_query_matches() -> None:
    every = json.loads(runner.invoke(app, ["features", "list", "--json"]).stdout)
    found = json.loads(
        runner.invoke(app, ["features", "list", "--query", "speed", "--json"]).stdout
    )
    plain = runner.invoke(app, ["features", "list", "--query", "Speed_Angvel"])

    names = [row["name"] for row in every]
    assert names == sorted(names)
    assert found[0]["name"] == "speed-angvel"
    assert 1 < len(found) < len(every)
    assert {row["type"] for row in found} == {"feature"}
    assert plain.exit_code == 0
    assert plain.stdout.splitlines()[1].split()[0] == "speed-angvel"


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
        "assign-arenas",
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
        "prepare-embedding-data",
        "prepare-training-data",
        "resample-tracks",
        "train-sleap",
        "train-litpose",
        "convert-tracks",
        "convert-labels",
        "predict-frames",
        "score-predictions",
        "select-frames",
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


def test_an_unknown_feature_under_json_is_an_invalid_envelope(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, _ = dataset
    result = runner.invoke(
        app, ["run", "-m", str(manifest), "--feature", "no-such-feature", "--json"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["kind"] == "invalid"
    assert payload["reason"] == "slug"
    assert "speed-angvel" in payload["message"]


def test_a_missing_optional_dependency_under_json_names_its_extra(
    dataset: tuple[Path, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error's install hint replaces the generic one, and the detail names the extra."""
    import mosaic.core.pipeline.run as run_module
    from mosaic.optional_dependency import MissingOptionalDependency

    missing = MissingOptionalDependency("torch", "deep-learning", "the identity model")

    def run_feature_without_torch(*_args: object, **_kwargs: object) -> object:
        raise missing

    monkeypatch.setattr(run_module, "run_feature", run_feature_without_torch)
    manifest, _ = dataset
    result = runner.invoke(
        app, ["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["kind"] == "unavailable"
    assert payload["message"] == str(missing)
    assert payload["detail"] == {"module": "torch", "extra": "deep-learning"}


def test_a_classified_failure_names_the_step_the_command_knows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import typer

    from mosaic.cli._io import fail_classified
    from mosaic.optional_dependency import MissingOptionalDependency

    with pytest.raises(typer.Exit):
        fail_classified(
            MissingOptionalDependency("torch", "deep-learning", "a test"),
            as_json=True,
            step="fit",
        )

    assert json.loads(capsys.readouterr().out)["step"] == "fit"


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
