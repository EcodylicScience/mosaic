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
    assert set(obj) == {"execution_id", "feature", "run_id", "status", "cache_hit"}
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


def test_tracking_list_and_describe() -> None:
    ops = json.loads(runner.invoke(app, ["tracking", "list", "--json"]).stdout)
    kinds = {o["kind"] for o in ops}
    assert kinds == {
        "extract-frames",
        "train-pose",
        "train-points",
        "train-localizer",
        "infer-pose",
        "infer-points",
        "infer-localizer",
        "trex",
        "convert-points",
    }

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
