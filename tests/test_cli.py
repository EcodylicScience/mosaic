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
    assert str(payload["run_id"]).startswith("0.1-")
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
    }
    # ``failed_entries`` is always present rather than only when non-empty: this
    # payload is a machine contract, so a consumer should not have to tell an
    # absent key from an empty one to know whether a run lost anything.
    assert obj["status"] == "finished"
    assert obj["failed_entries"] == []
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


def test_entries_rejected_with_kind(dataset: tuple[Path, Dataset]) -> None:
    manifest, _ = dataset
    result = runner.invoke(
        app, ["run", "-m", str(manifest), "--kind", "infer-pose", "--entries", "g:s1"]
    )
    assert result.exit_code == 1
    assert "entries" in result.stderr.lower()


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
