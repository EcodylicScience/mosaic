"""Tests for the ``mosaic`` CLI (Layer 1 over the Job Contract).

Drives the Typer app with ``CliRunner`` against a real ``Dataset`` (built from a
manifest, with synthetic tracks) using only the lightweight ``speed-angvel``
feature -- so the suite runs under the default ``-m 'not slow'`` gate with no
torch/ultralytics. Asserts the ``--json`` stream-separation contract (one JSON
value on stdout; breadcrumbs on stderr).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.cli import app
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS
from mosaic.core.media.probe_row import probe_video_metadata


def _make_runner() -> CliRunner:
    # click <8.2 needs mix_stderr=False to split streams; >=8.2 splits by default.
    try:
        return CliRunner(mix_stderr=False)  # pyright: ignore[reportCallIssue]
    except TypeError:
        return CliRunner()


runner = _make_runner()


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Dataset]:
    """A real Dataset with two synthetic tracks (columns speed-angvel needs)."""
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load()
    tracks_root = ds.get_root("tracks")
    rows = []
    for group, sequence in [("g", "s1"), ("g", "s2")]:
        n = 12
        df = pd.DataFrame(
            {
                "frame": range(n),
                "time": [f / 30.0 for f in range(n)],
                "id": [0] * n,
                "X": np.linspace(0.0, 5.0, n),
                "Y": np.linspace(0.0, 2.0, n),
            }
        )
        path = tracks_root / f"{group}__{sequence}.parquet"
        df.to_parquet(path)
        rows.append({"group": group, "sequence": sequence, "abs_path": str(path)})
    pd.DataFrame(rows).to_csv(tracks_root / "index.csv", index=False)
    return manifest, ds


def _run_json(args: list[str]) -> dict[str, object]:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (
        f"exit={result.exit_code}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    return json.loads(result.stdout)


# --- run -> status roundtrip ----------------------------------------------


def test_run_then_status_roundtrip(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    payload = _run_json(
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

    status = _run_json(
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
    first = _run_json(
        ["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"]
    )
    second = _run_json(
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
    payload = _run_json(
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
    run = _run_json(["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"])
    rows = json.loads(
        runner.invoke(
            app, ["runs", "-m", str(manifest), "--kind", "feature", "--json"]
        ).stdout
    )
    assert any(r["execution_id"] == run["execution_id"] for r in rows)
    assert all(r["kind"] == "feature" for r in rows)


def test_cancel_on_finished_run_is_noop(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    run = _run_json(["run", "-m", str(manifest), "--feature", "speed-angvel", "--json"])
    res = _run_json(
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

    res = _run_json(
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
    _ = _run_json(
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

    res = _run_json(
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

    res = _run_json(
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

    forced = _run_json(
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
    res = _run_json(
        ["release", "-m", str(manifest), "--execution-id", "01NOTHING", "--json"]
    )
    assert res["released"] == []


def test_sequences(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    payload = _run_json(["sequences", "-m", str(manifest), "--json"])
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
    payload = _run_json(
        ["sequences", "-m", str(manifest), "--group", "no-such-group", "--json"]
    )
    assert payload["sequences"] == []


# --- the tracks-index query methods ----------------------------------------
#
# None of these had any test at all, so a regression in three of them was
# invisible.


def test_query_methods_on_an_unconverted_dataset_are_empty(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    (ds.get_root("tracks") / "index.csv").unlink()

    assert ds.list_groups() == []
    assert ds.list_sequences() == []
    assert ds.query_sequences(sequence_contains="s") == []
    assert len(ds.get_sequence_metadata()) == 0


def test_query_methods_on_a_populated_dataset(dataset: tuple[Path, Dataset]) -> None:
    _, ds = dataset

    assert ds.list_groups() == ["g"]
    assert ds.list_sequences() == ["s1", "s2"]
    assert ds.query_sequences(sequence_contains="s1") == [("g", "s1")]
    meta = ds.get_sequence_metadata()
    assert len(meta) == 2
    # The safe-name columns this method documents are re-derived, not stored.
    assert list(meta["sequence_safe"]) == ["s1", "s2"]


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
    three scope flags, covered in ``tests/test_cli_run_scope.py``.
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


# --- reprobe-media ---------------------------------------------------------


LEGACY_CLI_COLUMNS = [
    "name",
    "group",
    "sequence",
    "sequence_safe",
    "abs_path",
    "media_type",
    "video_order",
]


def _legacy_cli_row(name: str, sequence: str, path: Path, order: str) -> dict[str, str]:
    return {
        "name": name,
        "group": "",
        "sequence": sequence,
        "sequence_safe": sequence,
        "abs_path": str(path),
        "media_type": "video",
        "video_order": order,
    }


def _seed_legacy_media_index(
    ds: Dataset,
    write_video: Callable[..., None],
    *,
    extra: list[dict[str, str]],
    curated_column: str = "",
) -> Path:
    """One readable video plus a pre-identity header: the detached-dataset shape.

    *extra* appends further rows under the same legacy header, so a test can add
    an unreadable row without a second index writer. *curated_column* adds a
    column outside the media-index schema, which a rewrite drops.
    """
    media_root = ds.get_root("media_raw")
    write_video(media_root / "seq" / "a.mp4")
    index_path = media_root / "index.csv"
    columns = LEGACY_CLI_COLUMNS + ([curated_column] if curated_column else [])
    first = _legacy_cli_row("a.mp4", "seq", media_root / "seq" / "a.mp4", "0")
    if curated_column:
        first[curated_column] = "collected by MW, do not delete"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows([first, *extra])
    return index_path


def _seed_stale_facts_media_index(
    ds: Dataset, write_video: Callable[..., None]
) -> Path:
    """One current-schema row whose stored facts cell no longer reconstructs.

    Identity is the file's real measured identity, so the row classifies
    ``unchanged`` and the unreconstructable cell is the only thing the run has to
    rewrite it for -- the state whose rewrite no other report line explains.
    """
    media_root = ds.get_root("media_raw")
    video = media_root / "seq" / "a.mp4"
    write_video(video)
    probe = probe_video_metadata(video)
    row = {column: "" for column in MEDIA_INDEX_COLUMNS}
    row.update(
        {
            "name": "a.mp4",
            "sequence": "seq",
            "sequence_safe": "seq",
            "abs_path": str(video),
            "media_type": "video",
            "video_order": "0",
            "video_uuid": probe["video_uuid"],
            "content_digest": probe["content_digest"],
            # Parses as JSON, and reconstructing MediaFacts from it still fails.
            "media_facts": json.dumps({"video_uuid": probe["video_uuid"]}),
        }
    )
    index_path = media_root / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEDIA_INDEX_COLUMNS)
        writer.writeheader()
        _ = writer.writerows([row])
    return index_path


def test_reprobe_media_names_the_facts_cell_it_rebuilds(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    write_cfr_mp4: Callable[..., None],
) -> None:
    # Without this line the operator reads "1 row(s) rewritten" against a summary
    # that reports every row as already current, and nothing says what the
    # rewrite did.
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    _ = _seed_stale_facts_media_index(ds, write_cfr_mp4)

    result = runner.invoke(
        app, ["reprobe-media", "-m", str(ds.manifest_path), "--apply"]
    )

    assert result.exit_code == 0, result.stderr
    assert "facts cell rebuilt in the media_raw index: 1 row(s)" in result.stdout

    payload = _run_json(["reprobe-media", "-m", str(ds.manifest_path), "--json"])
    # The applied run healed the cell, so the second look reports no rebuild.
    assert payload["facts_rebuilt"] == 0


def _seed_origin_less_derivative_index(
    ds: Dataset, write_video: Callable[..., None]
) -> Path:
    """A derivative row recording no origin, and the path of the file it names.

    The shape nothing that mints a derivative row produces, so it can only be
    written here directly.
    """
    media_root = ds.get_root("media")
    derivative = media_root / "seq.analysis.mp4"
    write_video(derivative, frames=4)
    row = {column: "" for column in MEDIA_INDEX_COLUMNS}
    row.update(
        {
            "name": "seq.analysis.mp4",
            "sequence": "seq",
            "sequence_safe": "seq",
            "abs_path": str(derivative),
            "media_type": "video",
            "video_order": "0",
        }
    )
    index_path = media_root / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEDIA_INDEX_COLUMNS)
        writer.writeheader()
        _ = writer.writerows([row])
    return derivative


def test_reprobe_media_names_a_derivative_row_recording_no_origin(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    write_cfr_mp4: Callable[..., None],
) -> None:
    # The condition is an invariant violation, so it is reported loudly -- but it
    # is not this command's to repair, so it gates neither the write nor the
    # exit code.
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    _ = _seed_legacy_media_index(ds, write_cfr_mp4, extra=[])
    derivative = _seed_origin_less_derivative_index(ds, write_cfr_mp4)

    result = runner.invoke(
        app, ["reprobe-media", "-m", str(ds.manifest_path), "--apply"]
    )

    assert result.exit_code == 0, result.stderr
    assert "1 row(s) record no source_path" in result.stdout
    # The indented detail line the group exists to produce, not the basename
    # loose anywhere in the output.
    assert f"  media index row 0: {derivative}" in result.stdout


def test_reprobe_media_still_names_an_origin_less_row_once_nothing_changes(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    write_cfr_mp4: Callable[..., None],
) -> None:
    # The violation outlives the run reporting it, because nothing this command
    # does repairs it. So the steady state is the state an operator sees for as
    # long as the row survives, and a report that fires only on the run that
    # happens to change something else is silent exactly when it matters.
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    _ = _seed_legacy_media_index(ds, write_cfr_mp4, extra=[])
    _ = _seed_origin_less_derivative_index(ds, write_cfr_mp4)

    first = runner.invoke(
        app, ["reprobe-media", "-m", str(ds.manifest_path), "--apply"]
    )
    second = runner.invoke(
        app, ["reprobe-media", "-m", str(ds.manifest_path), "--apply"]
    )

    assert first.exit_code == 0, first.stderr
    assert second.exit_code == 0, second.stderr
    assert "already fully probed" in second.stdout
    assert "record no source_path" in second.stdout


def test_reprobe_media_dry_run_is_the_default_and_writes_nothing(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    write_cfr_mp4: Callable[..., None],
) -> None:
    # No --dry-run flag is passed: writing is opt-in.
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    index_path = _seed_legacy_media_index(ds, write_cfr_mp4, extra=[])
    before = index_path.read_bytes()

    payload = _run_json(["reprobe-media", "-m", str(ds.manifest_path), "--json"])

    assert payload["changed"] is True
    assert payload["applied"] is False
    assert payload["identity_minted"] == 1
    assert index_path.read_bytes() == before
    assert not list(index_path.parent.glob("*.backup"))


def test_reprobe_media_apply_writes_the_migrated_index(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    write_cfr_mp4: Callable[..., None],
    read_index_header: Callable[[Path], list[str]],
) -> None:
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    index_path = _seed_legacy_media_index(ds, write_cfr_mp4, extra=[])
    before = index_path.read_bytes()

    payload = _run_json(
        ["reprobe-media", "-m", str(ds.manifest_path), "--apply", "--json"]
    )

    assert payload["applied"] is True
    assert payload["identity_minted"] == 1
    assert index_path.read_bytes() != before
    assert read_index_header(index_path) == MEDIA_INDEX_COLUMNS
    assert len(list(index_path.parent.glob("*.backup"))) == 1


def test_reprobe_media_aborts_non_zero_on_unreadable_media(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    write_cfr_mp4: Callable[..., None],
) -> None:
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    index_path = _seed_legacy_media_index(ds, write_cfr_mp4, extra=[])
    # The file the index names goes away after it is indexed.
    (ds.get_root("media_raw") / "seq" / "a.mp4").unlink()
    before = index_path.read_bytes()

    result = runner.invoke(
        app, ["reprobe-media", "-m", str(ds.manifest_path), "--apply"]
    )

    assert result.exit_code != 0
    assert "not on disk" in result.stderr
    assert index_path.read_bytes() == before


def test_reprobe_media_report_lists_the_unreadable_groups_apart(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    write_cfr_mp4: Callable[..., None],
) -> None:
    # A missing file and a corrupt one are different signals to an operator, so
    # the human report counts and lists them under separate headers.
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    media_root = ds.get_root("media_raw")
    broken = media_root / "seq" / "broken.mp4"
    index_path = _seed_legacy_media_index(
        ds,
        write_cfr_mp4,
        extra=[
            _legacy_cli_row("gone.mp4", "dead", media_root / "seq" / "gone.mp4", "4"),
            _legacy_cli_row("broken.mp4", "corrupt", broken, "7"),
        ],
    )
    broken.write_bytes(b"not a video")

    result = runner.invoke(
        app, ["reprobe-media", "-m", str(ds.manifest_path), "--skip-unreadable"]
    )

    assert result.exit_code == 0, result.stderr
    missing_header = "1 row(s) left untouched -- media missing from disk:"
    unprobeable_header = "1 row(s) left untouched -- media present but unprobeable:"
    assert missing_header in result.stdout
    assert unprobeable_header in result.stdout
    assert "gone.mp4" in result.stdout
    assert "broken.mp4" in result.stdout
    assert index_path.exists()


def test_reprobe_media_names_the_column_it_drops(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    write_cfr_mp4: Callable[..., None],
    read_index_header: Callable[[Path], list[str]],
) -> None:
    # The only data this command destroys, so the operator's one warning has to
    # reach the human report and the JSON alike.
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    index_path = _seed_legacy_media_index(
        ds, write_cfr_mp4, extra=[], curated_column="operator_note"
    )

    result = runner.invoke(
        app, ["reprobe-media", "-m", str(ds.manifest_path), "--apply"]
    )

    assert result.exit_code == 0, result.stderr
    assert "operator_note" in result.stdout
    assert "dropped from the media_raw index" in result.stdout
    assert "operator_note" not in read_index_header(index_path)

    payload = _run_json(["reprobe-media", "-m", str(ds.manifest_path), "--json"])
    assert payload["unknown_columns_dropped"] == []


# --- measure-tracks --------------------------------------------------------
#
# The only way to ask an already-published table whether its frame axis is its
# media's. A run records the comparison as it publishes, but a table on disk
# cannot be re-bridged without re-tracking, so a session tracked before anyone
# was recording the second number can only be measured afterwards.


def test_measure_tracks_is_a_dry_run_by_default(dataset: tuple[Path, Dataset]) -> None:
    manifest, ds = dataset
    payload = _run_json(["measure-tracks", "-m", str(manifest), "--json"])

    assert payload["applied"] is False
    assert payload["frame_extents_measured"] == 2
    from mosaic.core.pipeline.tracks_index import read_frame_extents

    assert read_frame_extents(ds) == {}, "a dry run must not write"


def test_measure_tracks_apply_records_the_extents(
    dataset: tuple[Path, Dataset],
) -> None:
    manifest, ds = dataset
    payload = _run_json(["measure-tracks", "-m", str(manifest), "--apply", "--json"])

    assert payload["applied"] is True
    from mosaic.core.pipeline.tracks_index import read_frame_extents

    assert read_frame_extents(ds) == {("g", "s1"): (0, 11), ("g", "s2"): (0, 11)}


def test_measure_tracks_names_a_frame_axis_that_is_not_its_media(
    dataset: tuple[Path, Dataset],
) -> None:
    """Both numbers, because the gap is the content of the report."""
    from mosaic.core.pipeline.tracks_index import (
        read_tracks_index,
        tracks_index_path,
        write_tracks_row,
    )

    manifest, ds = dataset
    out = ds.get_root("tracks") / "g__s1.parquet"
    write_tracks_row(
        ds,
        run_id="v1",
        group="g",
        sequence="s1",
        out_path=out,
        producer="trex",
        std_format="trex_v2",
        n_rows=12,
        media_frames=20,
    )
    # The fixture's hand-written rows carry no run_id, so drop them: an
    # unlabelled row and a labelled one for the same entry is a resolution
    # question this test is not about.
    frame = read_tracks_index(ds)
    frame[frame["run_id"] == "v1"].to_csv(tracks_index_path(ds), index=False)

    payload = _run_json(["measure-tracks", "-m", str(manifest), "--apply", "--json"])

    assert payload["frame_axis_mismatch"] == [
        {
            "group": "g",
            "sequence": "s1",
            "tracked_frames": 12,
            "media_frames": 20,
        }
    ]
