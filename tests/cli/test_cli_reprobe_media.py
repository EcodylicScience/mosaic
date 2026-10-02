"""``mosaic reprobe-media``: re-measure the media index, report, and apply on request.

A dry run is the default. The report names every row it would rewrite and why,
the media it could not read, and the columns a rewrite drops.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS
from mosaic.core.media.probe_row import probe_video_metadata
from tests.helpers import invoke_json

runner = CliRunner()


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

    payload = invoke_json(["reprobe-media", "-m", str(ds.manifest_path), "--json"])
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

    payload = invoke_json(["reprobe-media", "-m", str(ds.manifest_path), "--json"])

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

    payload = invoke_json(
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

    payload = invoke_json(["reprobe-media", "-m", str(ds.manifest_path), "--json"])
    assert payload["unknown_columns_dropped"] == []
