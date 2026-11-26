"""The ``rebuild-lmt-events`` op: its identity, its planning refusals and its entries.

The runner runs under the suite's interpreter on the stand-in checkout that
:func:`~tests.helpers.write_fake_lmt_analysis` commits, so each run here spawns the
real runner program. Every test that runs the op checks that the raw file's bytes
and modification time are unchanged.
"""

from __future__ import annotations

import hashlib
import platform
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import mosaic.tracking.ops.rebuild_lmt_events as rebuild_module
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.identity_scheme import read_identity_scheme
from mosaic.core.pipeline.job import Cancelled, CancelToken
from mosaic.core.pipeline.markers import (
    INFLIGHT_MARKER_NAME,
    InflightMarker,
    read_phase_marker,
    write_inflight,
)
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME
from mosaic.core.pipeline.ops import OPS, run_op
from mosaic.core.pipeline.refusal import ToolUnavailable
from mosaic.core.pipeline.labels_index import read_labels_index
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.track_library.lmt_rebuild import (
    REBUILD_KIND,
    RebuildManifest,
    copy_unchanged,
    read_rebuild_manifest,
)
from mosaic.core.scope import Scope
from mosaic.tracking import register_ops
from mosaic.tracking.common.lmt_env import (
    LMT_ANALYSIS_DIR_VAR,
    LMT_PYTHON_VAR,
    LmtRunnerReport,
)
from mosaic.tracking.ops.rebuild_lmt_events import (
    RebuildLmtEventsParams,
    rebuild_index,
    rebuild_lmt_events_run_id,
)
from tests.helpers import (
    LmtAnalysisInstaller,
    LmtRecording,
    alter_lmt_sqlite,
    default_lmt_recording,
    entry_error_lines,
    fail_lmt_rebuilds,
    latest_events,
    latest_snapshot,
    lmt_dataset,
    lmt_source_state,
)

register_ops()

_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_BUILDERS = ("BuildEventDetection", "BuildEventFlickering")


def _second_recording() -> LmtRecording:
    """A recording whose file differs from the default one."""
    default = default_lmt_recording()
    return replace(default, events=default.events[:3])


def _two_entries(tmp_path: Path) -> Dataset:
    return lmt_dataset(
        tmp_path / "ds",
        recordings={"first": default_lmt_recording(), "second": _second_recording()},
    )


def _one_entry(tmp_path: Path) -> Dataset:
    return lmt_dataset(tmp_path / "ds", recordings={"only": default_lmt_recording()})


def _params(commit: str, **values: object) -> dict[str, object]:
    return {"lmt_analysis_commit": commit, **values}


def _identity(ds: Dataset, params: dict[str, object]) -> str:
    op = OPS[REBUILD_KIND]()
    validated = RebuildLmtEventsParams.model_validate(params)
    return op.plan_identity(ds, validated, ds.resolve_scope(None)).run_id


def _run_root(ds: Dataset, run_id: str) -> Path:
    return ds.get_root(REBUILD_KIND) / run_id


def _events_of(copy: Path) -> list[str]:
    connection = sqlite3.connect(copy)
    try:
        rows = connection.execute("SELECT NAME FROM EVENT ORDER BY ID").fetchall()
    finally:
        connection.close()
    return [str(name) for (name,) in rows]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


# --- identity ------------------------------------------------------------


def test_the_run_id_is_equal_for_equal_inputs_in_any_order() -> None:
    params = RebuildLmtEventsParams(lmt_analysis_commit=_COMMIT)

    forward = rebuild_lmt_events_run_id(params, ["b" * 32, "a" * 32])
    backward = rebuild_lmt_events_run_id(params, ["a" * 32, "b" * 32])

    assert forward == backward
    assert forward.startswith("rebuild-lmt-events.0.1-")


def test_the_run_id_moves_with_the_commit_the_builders_and_a_source() -> None:
    sources = ["a" * 32, "b" * 32]
    base = rebuild_lmt_events_run_id(
        RebuildLmtEventsParams(lmt_analysis_commit=_COMMIT), sources
    )

    other_commit = rebuild_lmt_events_run_id(
        RebuildLmtEventsParams(lmt_analysis_commit="f" * 40), sources
    )
    named = rebuild_lmt_events_run_id(
        RebuildLmtEventsParams(
            lmt_analysis_commit=_COMMIT, builders=["BuildEventStop"]
        ),
        sources,
    )
    other_source = rebuild_lmt_events_run_id(
        RebuildLmtEventsParams(lmt_analysis_commit=_COMMIT), ["a" * 32, "c" * 32]
    )

    assert len({base, other_commit, named, other_source}) == 4


def test_the_plan_does_not_read_the_checkout_or_an_entry_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = lmt_dataset(tmp_path / "a", recordings={"first": default_lmt_recording()})
    renamed = lmt_dataset(
        tmp_path / "b", recordings={"other-name": default_lmt_recording()}
    )
    params = _params(_COMMIT)
    monkeypatch.setenv(LMT_ANALYSIS_DIR_VAR, str(tmp_path / "one"))
    first = _identity(ds, params)
    monkeypatch.setenv(LMT_ANALYSIS_DIR_VAR, str(tmp_path / "another"))

    assert _identity(ds, params) == first
    assert _md5(ds.base_dir / "raw" / "first.sqlite") == _md5(
        renamed.base_dir / "raw" / "other-name.sqlite"
    )
    assert _identity(renamed, params) == first


def test_a_source_that_changes_before_the_scan_moves_the_run_id(
    tmp_path: Path,
) -> None:
    ds = lmt_dataset(tmp_path / "a", recordings={"first": default_lmt_recording()})
    changed = lmt_dataset(tmp_path / "b", recordings={"first": _second_recording()})

    assert _identity(ds, _params(_COMMIT)) != _identity(changed, _params(_COMMIT))


@pytest.mark.parametrize(
    "commit", ["0123456", "0123456789ABCDEF0123456789ABCDEF01234567"]
)
def test_the_commit_has_to_be_forty_lowercase_digits(commit: str) -> None:
    with pytest.raises(ValidationError, match="second run id"):
        _ = RebuildLmtEventsParams(lmt_analysis_commit=commit)


@pytest.mark.parametrize(
    ("builders", "message"),
    [
        ([], "empty"),
        (["BuildEventStop", "BuildEventStop"], "repeats BuildEventStop"),
        (["a.b"], "'a.b'"),
    ],
)
def test_the_builders_are_named_once_each_as_module_names(
    builders: list[str], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _ = RebuildLmtEventsParams(lmt_analysis_commit=_COMMIT, builders=builders)


# --- planning refusals ---------------------------------------------------


def test_an_entry_without_an_lmt_sqlite_row_is_refused_by_name(tmp_path: Path) -> None:
    ds = lmt_dataset(tmp_path / "ds")
    op = OPS[REBUILD_KIND]()
    params = RebuildLmtEventsParams(lmt_analysis_commit=_COMMIT)
    scope = ds.resolve_scope(Scope(entries=[("", "recording"), ("", "missing")]))

    with pytest.raises(ValueError, match="without one: missing"):
        _ = op.plan_identity(ds, params, scope)


def test_a_dataset_without_lmt_sqlite_files_is_refused(tmp_path: Path) -> None:
    ds = lmt_dataset(tmp_path / "ds", recordings={})

    with pytest.raises(ValueError, match="covers none"):
        _ = _identity(ds, _params(_COMMIT))


def test_an_entry_with_two_files_is_refused(tmp_path: Path) -> None:
    ds = lmt_dataset(tmp_path / "ds")
    index_path = ds.get_root("tracks_raw") / "index.csv"
    rows = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    pd.concat([rows, rows]).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match=r"more than one lmt_sqlite file.*: recording"):
        _ = _identity(ds, _params(_COMMIT))


def test_a_file_without_an_md5_is_refused_naming_the_scan(tmp_path: Path) -> None:
    ds = lmt_dataset(tmp_path / "ds", md5=False)

    with pytest.raises(ValueError, match=r"recording.*mosaic scan"):
        _ = _identity(ds, _params(_COMMIT))


# --- a run ---------------------------------------------------------------


def test_a_run_writes_the_layout_of_each_entry(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    checkout = install_lmt()
    ds = _two_entries(tmp_path)
    before = lmt_source_state(ds)

    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert lmt_source_state(ds) == before
    root = _run_root(ds, run_id)
    assert read_identity_scheme(root) == OP_IDENTITY_SCHEME
    for name in ("first", "second"):
        entry_dir = root / name
        copy = entry_dir / f"{name}.sqlite"
        assert set(_events_of(copy)) >= {"Fake Detection", "Fake Flickering"}
        manifest = read_rebuild_manifest(copy)
        source = ds.base_dir / "raw" / f"{name}.sqlite"
        assert manifest.run_id == run_id
        assert manifest.op_version == "0.1"
        assert manifest.lmt_analysis_commit == checkout.commit
        assert manifest.builders == _BUILDERS
        assert manifest.source_md5 == _md5(source)
        assert manifest.source_name == source.name
        assert manifest.source_size == source.stat().st_size
        assert manifest.copy_name == copy.name
        assert manifest.copy_size == copy.stat().st_size
        assert manifest.script_max_frame == 300
        assert manifest.max_frame == 500
        assert manifest.python == platform.python_version()
        assert manifest.packages
        assert copy_unchanged(copy, manifest)
        report = LmtRunnerReport.model_validate_json(
            (entry_dir / "rebuild-report.json").read_text()
        )
        assert report.builders == _BUILDERS
        assert (entry_dir / "rebuild.log").is_file()
        marker = read_phase_marker(entry_dir, "rebuild")
        assert marker is not None
        assert marker.run_id == run_id
        assert marker.source == f"raw/{name}.sqlite"
        assert marker.source_uid == _md5(source)
        assert not list(entry_dir.glob(".incoming-*"))
    index = rebuild_index(ds.get_root(REBUILD_KIND) / "index.csv").read()
    assert sorted(index["sequence"]) == ["first", "second"]
    assert set(index["run_id"]) == {run_id}
    assert sorted(Path(str(cell)).name for cell in index["abs_path"]) == [
        "first",
        "second",
    ]
    assert sorted(index["source_md5"]) == sorted(
        _md5(ds.base_dir / "raw" / f"{name}.sqlite") for name in ("first", "second")
    )


def test_a_run_records_its_phases_progress_and_entries(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    checkout = install_lmt()
    ds = _two_entries(tmp_path)
    before = lmt_source_state(ds)

    _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "finished"
    assert snapshot["progress_total"] == 2
    assert snapshot["progress_done"] == 2
    assert snapshot["entries_written"] == 2
    phases = [str(event["phase"]) for event in latest_events(ds, "phase")]
    assert phases == ["probe", "copy", "rebuild", "copy", "rebuild"]
    assert lmt_source_state(ds) == before


def test_the_runners_output_is_kept_in_the_log(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    release = tmp_path / "release"
    checkout = install_lmt(wait_for=release)
    release.write_text("go")
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)

    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    log = (_run_root(ds, run_id) / "only" / "rebuild.log").read_text()
    assert f"waiting for {release}" in log
    assert lmt_source_state(ds) == before


# --- an entry that fails -------------------------------------------------


def test_a_source_edited_after_the_scan_fails_its_entry(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    checkout = install_lmt()
    ds = _two_entries(tmp_path)
    alter_lmt_sqlite(
        ds.base_dir / "raw" / "first.sqlite", "UPDATE ANIMAL SET NAME = 'edited';"
    )
    before = lmt_source_state(ds)

    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert lmt_source_state(ds) == before
    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "finished"
    assert snapshot["entries_failed"] == 1
    assert snapshot["entries_written"] == 1
    [line] = entry_error_lines(ds, snapshot["execution_id"])
    assert "first" in line
    assert "changed after the scan" in line
    root = _run_root(ds, run_id)
    assert not (root / "first").exists()
    assert (root / "second" / "second.sqlite").is_file()


def test_every_entry_failing_raises_all_entries_failed(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    checkout = install_lmt()
    ds = _two_entries(tmp_path)
    for name in ("first", "second"):
        alter_lmt_sqlite(
            ds.base_dir / "raw" / f"{name}.sqlite", "UPDATE ANIMAL SET NAME = 'edited';"
        )
    before = lmt_source_state(ds)

    with pytest.raises(AllEntriesFailed, match=r"first, second"):
        _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert latest_snapshot(ds)["status"] == "failed"
    assert lmt_source_state(ds) == before


class _DiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def test_too_little_space_fails_the_entry_naming_both_numbers(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    size = (ds.base_dir / "raw" / "only.sqlite").stat().st_size

    def scarce(_path: Path) -> _DiskUsage:
        return _DiskUsage(10**12, 10**12 - size, size)

    monkeypatch.setattr(rebuild_module.shutil, "disk_usage", scarce)

    with pytest.raises(AllEntriesFailed):
        _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    [line] = entry_error_lines(ds, latest_snapshot(ds)["execution_id"])
    assert f"{size} free bytes" in line
    assert f"needs {2 * size}" in line
    [root] = list(ds.get_root(REBUILD_KIND).glob("rebuild-lmt-events.*"))
    assert not (root / "only").exists()
    assert lmt_source_state(ds) == before


def test_a_journal_beside_the_source_fails_the_entry(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    journal = ds.base_dir / "raw" / "only.sqlite-journal"
    _ = journal.write_bytes(b"pending")
    before = lmt_source_state(ds)

    with pytest.raises(AllEntriesFailed):
        _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert lmt_source_state(ds) == before
    [line] = entry_error_lines(ds, latest_snapshot(ds)["execution_id"])
    assert "only.sqlite-journal" in line
    [root] = list(ds.get_root(REBUILD_KIND).glob("rebuild-lmt-events.*"))
    assert not (root / "only").exists()


def _assert_no_index_row(ds: Dataset) -> None:
    """The rebuild index is absent or empty: no row names an unfinished entry."""
    index_path = ds.get_root(REBUILD_KIND) / "index.csv"
    assert not index_path.exists() or rebuild_index(index_path).read().empty


def _leftovers(entry_dir: Path) -> list[str]:
    """The files of *entry_dir* that a failed entry may not leave."""
    return sorted(
        path.name
        for path in entry_dir.iterdir()
        if path.name.startswith(".incoming-")
        or ".sqlite" in path.name
        or path.name == "rebuild.json"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="a symlink needs privileges")
def test_a_journal_beside_the_database_a_symlink_names_fails_the_entry(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    """An index cell written by hand or remapped through ``path_map`` may name a symlink.

    A scan stores the resolved path, and SQLite reads the file the symlink names.
    """
    checkout = install_lmt()
    ds = lmt_dataset(
        tmp_path / "ds", recordings={"only": default_lmt_recording()}, tracks=False
    )
    real = tmp_path / "real" / "only.sqlite"
    real.parent.mkdir()
    _ = (ds.base_dir / "raw" / "only.sqlite").rename(real)
    walked = tmp_path / "walked"
    walked.mkdir()
    (walked / "only.sqlite").symlink_to(real)
    journal = real.with_name("only.sqlite-journal")
    _ = journal.write_bytes(b"pending")
    _ = ds.index_tracks_raw(
        [walked], patterns=["*.sqlite"], src_format="lmt_sqlite", compute_md5=True
    )
    index_path = ds.get_root("tracks_raw") / "index.csv"
    rows = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    rows["abs_path"] = str(walked / "only.sqlite")
    rows.to_csv(index_path, index=False)
    kept = (real, journal)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in kept}

    with pytest.raises(AllEntriesFailed):
        _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in kept
    } == before
    [line] = entry_error_lines(ds, latest_snapshot(ds)["execution_id"])
    assert "only.sqlite-journal" in line
    [root] = list(ds.get_root(REBUILD_KIND).glob("rebuild-lmt-events.*"))
    assert not (root / "only").exists()
    _assert_no_index_row(ds)


def test_a_runner_failure_leaves_the_logs_and_no_copy(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    checkout = install_lmt(raises=True)
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)

    with pytest.raises(AllEntriesFailed):
        _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert lmt_source_state(ds) == before
    [line] = entry_error_lines(ds, latest_snapshot(ds)["execution_id"])
    assert "the stand-in rebuild was asked to fail" in line
    [root] = list(ds.get_root(REBUILD_KIND).glob("rebuild-lmt-events.*"))
    entry_dir = root / "only"
    assert _leftovers(entry_dir) == []
    assert (entry_dir / "rebuild.log").is_file()
    report = LmtRunnerReport.model_validate_json(
        (entry_dir / "rebuild-report.json").read_text()
    )
    assert report.error is not None
    assert read_phase_marker(entry_dir, "rebuild") is None
    _assert_no_index_row(ds)


def test_a_journal_left_beside_the_rebuilt_file_fails_the_entry(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    real = rebuild_module.run_lmt_rebuild

    def leaving_a_journal(
        database: Path,
        commit: str,
        builders: Sequence[str] | None,
        *,
        report_path: Path,
        log_path: Path,
        cancel_check: Callable[[], bool] | None,
        on_line: Callable[[str], None],
    ) -> LmtRunnerReport:
        report = real(
            database,
            commit,
            builders,
            report_path=report_path,
            log_path=log_path,
            cancel_check=cancel_check,
            on_line=on_line,
        )
        _ = database.with_name(database.name + "-journal").write_bytes(b"pending")
        return report

    monkeypatch.setattr(rebuild_module, "run_lmt_rebuild", leaving_a_journal)

    with pytest.raises(AllEntriesFailed):
        _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    [line] = entry_error_lines(ds, latest_snapshot(ds)["execution_id"])
    assert "-journal" in line
    [root] = list(ds.get_root(REBUILD_KIND).glob("rebuild-lmt-events.*"))
    assert _leftovers(root / "only") == []
    _assert_no_index_row(ds)
    assert lmt_source_state(ds) == before


def test_a_failure_after_the_rename_removes_the_published_copy(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)

    def failing(_self: RebuildManifest, _entry_dir: Path) -> None:
        raise OSError("the manifest could not be written")

    monkeypatch.setattr(RebuildManifest, "write", failing)

    with pytest.raises(AllEntriesFailed):
        _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    [root] = list(ds.get_root(REBUILD_KIND).glob("rebuild-lmt-events.*"))
    assert _leftovers(root / "only") == []
    assert read_phase_marker(root / "only", "rebuild") is None
    _assert_no_index_row(ds)
    assert lmt_source_state(ds) == before


# --- the probe's refusals ------------------------------------------------


def _no_entry_directory(ds: Dataset, params: dict[str, object]) -> None:
    run_id = _identity(ds, params)
    assert not _run_root(ds, run_id).exists()


def test_a_checkout_at_another_commit_is_refused_before_any_copy(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    _ = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    params = _params(_COMMIT)

    with pytest.raises(ToolUnavailable, match=_COMMIT):
        _ = run_op(ds, REBUILD_KIND, params)

    _no_entry_directory(ds, params)
    assert lmt_source_state(ds) == before


def test_an_unknown_builder_is_refused_before_any_copy(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    params = _params(checkout.commit, builders=["BuildEventNoSuchThing"])

    with pytest.raises(ValueError, match="BuildEventNoSuchThing"):
        _ = run_op(ds, REBUILD_KIND, params)

    _no_entry_directory(ds, params)
    assert lmt_source_state(ds) == before


def test_a_missing_checkout_is_refused_before_any_copy(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    monkeypatch.setenv(LMT_ANALYSIS_DIR_VAR, str(tmp_path / "nowhere"))
    params = _params(checkout.commit)

    with pytest.raises(ToolUnavailable, match="nowhere"):
        _ = run_op(ds, REBUILD_KIND, params)

    _no_entry_directory(ds, params)
    assert lmt_source_state(ds) == before


def test_a_missing_environment_is_refused_before_any_copy(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    monkeypatch.delenv(LMT_PYTHON_VAR)
    params = _params(checkout.commit)

    with pytest.raises(ToolUnavailable, match="README"):
        _ = run_op(ds, REBUILD_KIND, params)

    _no_entry_directory(ds, params)
    assert lmt_source_state(ds) == before


# --- the end to end ------------------------------------------------------


def test_conversions_with_the_run_id_read_the_copy(
    tmp_path: Path,
    install_lmt: LmtAnalysisInstaller,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = install_lmt()
    ds = lmt_dataset(
        tmp_path / "ds",
        recordings={"only": default_lmt_recording()},
        tracks=True,
        labels=True,
    )
    before = lmt_source_state(ds)
    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))
    plain = ds.convert_all_tracks()
    assert plain.failed == 0

    rebuilt = ds.convert_all_tracks({"rebuild_run_id": run_id})

    assert rebuilt.failed == 0
    assert rebuilt.variants != plain.variants
    assert f"from rebuild {run_id}" in capsys.readouterr().err
    labels = ds.convert_all_labels(
        kind="lmt_events",
        source_format="lmt_sqlite",
        params={
            "rebuild_run_id": run_id,
            "extra_event_names": ["Fake Detection", "Fake Flickering"],
        },
        tracks_run_id=rebuilt.variants[0],
    )
    assert labels.written == 1
    index = read_labels_index(ds, "lmt_events")
    (path,) = index["abs_path"].tolist()
    with np.load(ds.resolve_path(str(path)), allow_pickle=False) as archive:
        assert str(archive["rebuild_run_id"]) == run_id
        assert str(archive["lmt_analysis_commit"]) == checkout.commit
        process_names = [row[0] for row in archive["rebuild_log"].tolist()]
    assert process_names[-1] == "Stand-in rebuild"
    assert lmt_source_state(ds) == before


# --- resume, overwrite and cancel ----------------------------------------


_LONG_AGO = "2020-01-01T00:00:00+00:00"


def _counting_runner(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count the rebuilds the op starts, naming each by its copy's file name."""
    started: list[str] = []
    real = rebuild_module.run_lmt_rebuild

    def counting(
        database: Path,
        commit: str,
        builders: Sequence[str] | None,
        *,
        report_path: Path,
        log_path: Path,
        cancel_check: Callable[[], bool] | None,
        on_line: Callable[[str], None],
    ) -> LmtRunnerReport:
        started.append(database.name.rsplit("-", 1)[-1])
        return real(
            database,
            commit,
            builders,
            report_path=report_path,
            log_path=log_path,
            cancel_check=cancel_check,
            on_line=on_line,
        )

    monkeypatch.setattr(rebuild_module, "run_lmt_rebuild", counting)
    return started


def _copy_files(ds: Dataset, run_id: str) -> dict[Path, tuple[bytes, int]]:
    """Every copy of the run, with its bytes and modification time."""
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(_run_root(ds, run_id).glob("*/*.sqlite"))
    }


def _index_sequences(ds: Dataset) -> list[str]:
    index = rebuild_index(ds.get_root(REBUILD_KIND) / "index.csv").read()
    return sorted(str(sequence) for sequence in index["sequence"])


def test_a_second_run_rebuilds_nothing_and_records_a_cache_hit(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _two_entries(tmp_path)
    before = lmt_source_state(ds)
    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))
    first = latest_snapshot(ds)
    copies = _copy_files(ds, run_id)
    started = _counting_runner(monkeypatch)

    again = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert again == run_id
    assert started == []
    assert _copy_files(ds, run_id) == copies
    assert _index_sequences(ds) == ["first", "second"]
    assert first["cache_hit"] is False
    second = latest_snapshot(ds)
    assert second["execution_id"] != first["execution_id"]
    assert second["status"] == "finished"
    assert second["cache_hit"] is True
    assert second["entries_written"] == 2
    assert second["progress_done"] == 2
    assert lmt_source_state(ds) == before


def test_overwrite_rebuilds_every_entry(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _two_entries(tmp_path)
    before = lmt_source_state(ds)
    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))
    started = _counting_runner(monkeypatch)

    _ = run_op(ds, REBUILD_KIND, _params(checkout.commit), overwrite=True)

    assert sorted(started) == ["first.sqlite", "second.sqlite"]
    snapshot = latest_snapshot(ds)
    assert snapshot["cache_hit"] is False
    assert snapshot["entries_written"] == 2
    for name in ("first", "second"):
        entry_dir = _run_root(ds, run_id) / name
        manifest = read_rebuild_manifest(entry_dir / f"{name}.sqlite")
        assert copy_unchanged(entry_dir / f"{name}.sqlite", manifest)
        assert read_phase_marker(entry_dir, "rebuild") is not None
    assert _index_sequences(ds) == ["first", "second"]
    assert lmt_source_state(ds) == before


def test_a_copy_that_grew_is_rebuilt_and_the_other_is_left(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _two_entries(tmp_path)
    before = lmt_source_state(ds)
    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))
    grown = _run_root(ds, run_id) / "first" / "first.sqlite"
    with grown.open("ab") as appending:
        _ = appending.write(b"\0")
    other = _run_root(ds, run_id) / "second" / "second.sqlite"
    other_before = (other.read_bytes(), other.stat().st_mtime_ns)
    started = _counting_runner(monkeypatch)

    _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert started == ["first.sqlite"]
    assert (other.read_bytes(), other.stat().st_mtime_ns) == other_before
    assert copy_unchanged(grown, read_rebuild_manifest(grown))
    assert set(_events_of(grown)) >= {"Fake Detection", "Fake Flickering"}
    snapshot = latest_snapshot(ds)
    assert snapshot["cache_hit"] is False
    assert snapshot["entries_written"] == 2
    assert lmt_source_state(ds) == before


def test_a_finished_entry_without_an_index_row_is_rowed_and_not_rebuilt(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _two_entries(tmp_path)
    before = lmt_source_state(ds)
    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))
    copies = _copy_files(ds, run_id)
    _ = rebuild_index(ds.get_root(REBUILD_KIND) / "index.csv").drop_entries(
        scope=Scope(entries=[("", "first")])
    )
    assert _index_sequences(ds) == ["second"]
    started = _counting_runner(monkeypatch)

    _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert started == []
    assert _copy_files(ds, run_id) == copies
    index = rebuild_index(ds.get_root(REBUILD_KIND) / "index.csv").read()
    assert sorted(index["sequence"]) == ["first", "second"]
    assert set(index["run_id"]) == {run_id}
    first_row = index[index["sequence"] == "first"].iloc[0]
    assert Path(str(first_row["abs_path"])).name == "first"
    snapshot = latest_snapshot(ds)
    assert snapshot["cache_hit"] is True
    assert snapshot["entries_written"] == 2
    assert first_row["source_md5"] == _md5(ds.base_dir / "raw" / "first.sqlite")
    assert lmt_source_state(ds) == before


@pytest.mark.parametrize(
    "stale",
    [
        {"lmt_analysis_commit": "f" * 40},
        {"builders": ("BuildEventStop",)},
        {"source_md5": "0" * 32},
    ],
    ids=["commit", "builders", "source_md5"],
)
def test_a_manifest_that_names_another_input_is_rebuilt(
    tmp_path: Path,
    install_lmt: LmtAnalysisInstaller,
    monkeypatch: pytest.MonkeyPatch,
    stale: dict[str, object],
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))
    entry_dir = _run_root(ds, run_id) / "only"
    copy = entry_dir / "only.sqlite"
    manifest = read_rebuild_manifest(copy)
    manifest.model_copy(update=stale).write(entry_dir)
    started = _counting_runner(monkeypatch)

    _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert started == ["only.sqlite"]
    rebuilt = read_rebuild_manifest(copy)
    assert rebuilt.lmt_analysis_commit == checkout.commit
    assert rebuilt.builders == _BUILDERS
    assert rebuilt.source_md5 == _md5(ds.base_dir / "raw" / "only.sqlite")
    assert read_phase_marker(entry_dir, "rebuild") is not None
    assert lmt_source_state(ds) == before


def test_an_entry_without_its_marker_is_rebuilt(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))
    entry_dir = _run_root(ds, run_id) / "only"
    (entry_dir / ".mosaic-rebuild.json").unlink()
    started = _counting_runner(monkeypatch)

    _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert started == ["only.sqlite"]
    assert read_phase_marker(entry_dir, "rebuild") is not None
    assert lmt_source_state(ds) == before


def test_the_next_run_after_a_runner_failure_clears_the_logs_and_rebuilds(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    real = rebuild_module.run_lmt_rebuild

    fail_lmt_rebuilds(monkeypatch)
    with pytest.raises(AllEntriesFailed):
        _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))
    [root] = list(ds.get_root(REBUILD_KIND).glob("rebuild-lmt-events.*"))
    entry_dir = root / "only"
    assert "the failed attempt's output" in (entry_dir / "rebuild.log").read_text()
    assert (entry_dir / "rebuild-report.json").is_file()
    monkeypatch.setattr(rebuild_module, "run_lmt_rebuild", real)

    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    assert root.name == run_id
    log = (entry_dir / "rebuild.log").read_text()
    assert "the failed attempt's output" not in log
    report = LmtRunnerReport.model_validate_json(
        (entry_dir / "rebuild-report.json").read_text()
    )
    assert report.error is None
    copy = entry_dir / "only.sqlite"
    assert copy_unchanged(copy, read_rebuild_manifest(copy))
    assert read_phase_marker(entry_dir, "rebuild") is not None
    assert _index_sequences(ds) == ["only"]
    assert lmt_source_state(ds) == before


def test_a_failed_overwrite_leaves_no_index_row_for_a_finished_entry(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _two_entries(tmp_path)
    before = lmt_source_state(ds)
    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))
    assert _index_sequences(ds) == ["first", "second"]
    fail_lmt_rebuilds(monkeypatch)

    with pytest.raises(AllEntriesFailed):
        _ = run_op(ds, REBUILD_KIND, _params(checkout.commit), overwrite=True)

    for name in ("first", "second"):
        entry_dir = _run_root(ds, run_id) / name
        assert _leftovers(entry_dir) == []
        assert read_phase_marker(entry_dir, "rebuild") is None
    _assert_no_index_row(ds)
    assert lmt_source_state(ds) == before


def test_a_failed_redo_of_a_grown_copy_drops_only_its_index_row(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _two_entries(tmp_path)
    before = lmt_source_state(ds)
    run_id = run_op(ds, REBUILD_KIND, _params(checkout.commit))
    grown = _run_root(ds, run_id) / "first" / "first.sqlite"
    with grown.open("ab") as appending:
        _ = appending.write(b"\0")
    fail_lmt_rebuilds(monkeypatch)

    _ = run_op(ds, REBUILD_KIND, _params(checkout.commit))

    entry_dir = grown.parent
    assert _leftovers(entry_dir) == []
    assert read_phase_marker(entry_dir, "rebuild") is None
    assert _index_sequences(ds) == ["second"]
    second = _run_root(ds, run_id) / "second"
    assert read_phase_marker(second, "rebuild") is not None
    assert lmt_source_state(ds) == before


def test_a_cancel_while_the_runner_waits_ends_the_run_cancelled(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller
) -> None:
    never = tmp_path / "never"
    checkout = install_lmt(wait_for=never)
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    run_id = _identity(ds, _params(checkout.commit))
    entry_dir = _run_root(ds, run_id) / "only"
    token = CancelToken()
    stop = threading.Event()

    def cancel_once_waiting() -> None:
        log = entry_dir / "rebuild.log"
        deadline = time.monotonic() + 60
        while not stop.is_set() and time.monotonic() < deadline:
            if log.is_file() and "waiting for" in log.read_text():
                token.cancel()
                return
            _ = stop.wait(0.05)
        if not token.is_cancelled():
            _ = never.write_text("go")

    watcher = threading.Thread(target=cancel_once_waiting)
    watcher.start()
    try:
        with pytest.raises(Cancelled):
            _ = run_op(ds, REBUILD_KIND, _params(checkout.commit), cancel_token=token)
    finally:
        stop.set()
        watcher.join()

    assert token.is_cancelled()
    assert latest_snapshot(ds)["status"] == "cancelled"
    assert entry_error_lines(ds, latest_snapshot(ds)["execution_id"]) == []
    assert not (entry_dir / INFLIGHT_MARKER_NAME).exists()
    assert _leftovers(entry_dir) == []
    assert read_phase_marker(entry_dir, "rebuild") is None
    _assert_no_index_row(ds)
    assert lmt_source_state(ds) == before


def test_a_directory_left_by_a_killed_process_is_cleared_and_rebuilt(
    tmp_path: Path, install_lmt: LmtAnalysisInstaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = install_lmt()
    ds = _one_entry(tmp_path)
    before = lmt_source_state(ds)
    run_id = _identity(ds, _params(checkout.commit))
    entry_dir = _run_root(ds, run_id) / "only"
    entry_dir.mkdir(parents=True)
    write_inflight(
        entry_dir,
        InflightMarker(
            execution_id="killed",
            host="elsewhere",
            pid=1,
            phase=None,
            started_at=_LONG_AGO,
            expires_at=_LONG_AGO,
        ),
    )
    leftover = entry_dir / ".incoming-killed-only.sqlite"
    _ = leftover.write_bytes(b"half a copy")
    started = _counting_runner(monkeypatch)

    assert run_op(ds, REBUILD_KIND, _params(checkout.commit)) == run_id

    assert started == ["only.sqlite"]
    assert not leftover.exists()
    assert _leftovers(entry_dir) == ["only.sqlite", "rebuild.json"]
    copy = entry_dir / "only.sqlite"
    assert copy_unchanged(copy, read_rebuild_manifest(copy))
    assert read_phase_marker(entry_dir, "rebuild") is not None
    assert not (entry_dir / INFLIGHT_MARKER_NAME).exists()
    assert _index_sequences(ds) == ["only"]
    assert lmt_source_state(ds) == before
