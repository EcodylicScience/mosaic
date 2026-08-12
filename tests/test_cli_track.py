"""``mosaic track <kind>``, which replaced the hand-wired ``mosaic trex``.

The old command was 211 lines of per-flag wiring for one of the three trackers,
justified by a claim that had stopped being true: that ``run_trex`` had no
Pydantic ``Params`` and so could not ride the schema-driven runner. These pin
that the replacement covers every tracker, reads its flags from the params rather
than from a hand-written list, and refuses what it cannot honor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.cli.track import tracker_kinds
from mosaic.core.dataset import Dataset, new_dataset_manifest

runner = CliRunner()


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    """A dataset with an empty but present media index.

    Empty is what makes these tests cheap: every run below resolves an empty
    scope and returns before reaching a binary, so what is exercised is the flag
    plumbing rather than a tracker.
    """
    path = new_dataset_manifest("t", base_dir=tmp_path)
    ds = Dataset(manifest_path=path).load(ensure_roots=True)
    _ = ds.index_media([ds.get_root(ds.resolve_media_root())])
    return path


def test_every_integrated_tracker_is_runnable() -> None:
    """Read from the roots table, so a fourth tracker is covered when its row is."""
    assert tracker_kinds() == ["litpose", "sleap", "trex", "ultralytics"]


def test_the_help_names_the_trackers() -> None:
    result = runner.invoke(app, ["track", "--help"])

    assert result.exit_code == 0
    assert "trex" in result.output


def test_an_unknown_tracker_names_the_ones_that_exist(manifest: Path) -> None:
    result = runner.invoke(app, ["track", "nosuchtool", "-m", str(manifest)])

    assert result.exit_code == 1
    assert "sleap" in result.stderr


def test_an_unknown_parameter_is_refused_with_the_available_ones(
    manifest: Path,
) -> None:
    """The payoff of deriving flags from the schema: an honest error, for free."""
    result = runner.invoke(
        app, ["track", "trex", "-m", str(manifest), "--set", "detect_modle=x"]
    )

    assert result.exit_code == 1
    assert "detect_modle" in result.stderr
    assert "detect_model" in result.stderr


def test_a_malformed_set_token_is_refused(manifest: Path) -> None:
    result = runner.invoke(
        app, ["track", "trex", "-m", str(manifest), "--set", "detect_model"]
    )

    assert result.exit_code == 1
    assert "key=value" in result.stderr


def test_a_tracker_specific_parameter_reaches_the_op(manifest: Path) -> None:
    """An empty scope short-circuits before any binary, so this exercises the wiring.

    ``cm_per_pixel`` is TREx's and nothing in this file knows that -- it is
    accepted because ``TrexParams`` declares it.
    """
    result = runner.invoke(
        app,
        [
            "track",
            "trex",
            "-m",
            str(manifest),
            "--sequences",
            "nonexistent",
            "--set",
            "cm_per_pixel=0.05",
            "--set",
            "track_max_individuals=4",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "trex." in result.stdout


def test_a_json_value_is_read_as_json_not_as_a_string(manifest: Path) -> None:
    """``--set`` has to carry a mapping, because two trackers take one."""
    result = runner.invoke(
        app,
        [
            "track",
            "trex",
            "-m",
            str(manifest),
            "--sequences",
            "nonexistent",
            "--set",
            'track_extra_settings={"blob_size_range": [1, 100]}',
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output


def test_the_deleted_command_is_gone(manifest: Path) -> None:
    """One tracker having its own verb was the asymmetry this removed."""
    result = runner.invoke(app, ["trex", "-m", str(manifest)])

    assert result.exit_code != 0


def test_a_clean_run_reports_finished_with_no_lost_entries(manifest: Path) -> None:
    """The baseline the partial case is measured against.

    An empty scope loses nothing, so the status stays ``finished`` and the count
    is present and zero -- a reader should not have to tell "no failures" from
    "this command does not report failures" by the absence of a key.
    """
    result = runner.invoke(
        app, ["track", "trex", "-m", str(manifest), "--sequences", "none", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "finished"
    assert payload["entries_failed"] == 0


def test_a_run_that_lost_an_entry_reports_partial(
    manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status a run that published nothing for an entry must not claim.

    Derived from the attempt's own run-log rather than from anything the op
    returns, so it works for every op and needs no change to ``run_op``'s
    signature -- which mosaic-queue and mosaic-api also call. The exit code
    stays 0: ``partial`` is a reporting word, and mosaic-queue maps exit 0 to a
    ``finished`` ledger row, with ``entries_failed`` recorded beside it.
    """
    import mosaic.core.pipeline.ops as ops_module

    real_run_op = ops_module.run_op

    def run_op_losing_an_entry(ds, kind, params, **kwargs):  # type: ignore[no-untyped-def]
        from mosaic.core.pipeline.run_log import JsonlRunLog, run_log_path

        run_id = real_run_op(ds, kind, params, **kwargs)
        # Append the event a lost entry writes, to the attempt this call made.
        execution_id = kwargs["execution_id"]
        log = JsonlRunLog(run_log_path(ds.base_dir, execution_id), execution_id)
        log.entry_failed("some__entry", '{"type": "UnknownTrexUnitsError"}')
        log.close()
        return run_id

    monkeypatch.setattr(ops_module, "run_op", run_op_losing_an_entry)

    result = runner.invoke(
        app, ["track", "trex", "-m", str(manifest), "--sequences", "none", "--json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "partial"
    assert payload["entries_failed"] == 1
