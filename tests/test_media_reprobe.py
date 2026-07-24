"""Tests for the in-place media-index re-probe.

The media indexes here are the shape the detached production datasets have:
curated ``(group, sequence)`` and ``video_order``, transcode links nothing else
re-keys, and in the legacy cases a header that lacks the identity columns
outright. So most assertions are about what survives a re-probe rather than
about what it adds.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import to_safe_name
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS, row_to_facts
from mosaic.core.media.probe_row import probe_video_metadata
from mosaic.core.media.reprobe import (
    UNMEASURED_LINK_COLUMNS,
    ProbedRow,
    ReprobeAbort,
    ReprobeReport,
    measured_cells,
)
from mosaic.core.pipeline.media_index import (
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)

# The exact cells a re-probe may overwrite on an existing row. Pinned as a
# literal rather than re-derived, because the risk this guards is the derivation
# quietly growing: measured_cells subtracts a denylist from whatever the probe
# carries, so a column added to facts_to_row later would widen the write set
# without a word. Widening it is a deliberate act -- update this set and say why.
WRITABLE_COLUMNS = frozenset(
    {
        "width",
        "height",
        "fps",
        "codec",
        "frame_count",
        "analysis_transcode",
        "stream_transcode",
        "video_uuid",
        "content_digest",
        "media_facts",
        "size_bytes",
        "mtime_iso",
    }
)

# The media-index header as it stood before the identity columns existed. The
# legacy fixtures write exactly this and nothing else, so a test exercises the
# absent-column path rather than a full-schema file with blanked cells.
LEGACY_COLUMNS = [
    "name",
    "group",
    "sequence",
    "group_safe",
    "sequence_safe",
    "abs_path",
    "size_bytes",
    "mtime_iso",
    "width",
    "height",
    "fps",
    "codec",
    "media_type",
    "video_order",
]

WriteVideo = Callable[..., None]
ReadHeader = Callable[[Path], list[str]]
MakeDataset = Callable[[Path], Dataset]
MakeImgstore = Callable[..., tuple[Path, list[np.ndarray]]]


@pytest.fixture
def dataset(tmp_path: Path, make_media_dataset: MakeDataset) -> Dataset:
    # tmp_path is resolved first so the macOS /var -> /private/var symlink cannot
    # desynchronize the dataset base directory from abs_path.resolve().
    return make_media_dataset((tmp_path / "dataset").resolve())


def _row(
    *,
    name: str,
    group: str,
    sequence: str,
    abs_path: str,
    video_order: str = "0",
    **overrides: str,
) -> dict[str, object]:
    """A current-schema index row carrying no identity."""
    row: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
    row.update(
        {
            "name": name,
            "group": group,
            "sequence": sequence,
            "group_safe": to_safe_name(group) if group else "",
            "sequence_safe": to_safe_name(sequence) if sequence else "",
            "abs_path": abs_path,
            "media_type": "video",
            "video_order": video_order,
        }
    )
    row.update(overrides)
    return row


def _legacy_row(
    *, name: str, group: str, sequence: str, abs_path: str, video_order: str = "0"
) -> dict[str, str]:
    """A row carrying only the pre-identity columns."""
    return {
        "name": name,
        "group": group,
        "sequence": sequence,
        "group_safe": to_safe_name(group) if group else "",
        "sequence_safe": to_safe_name(sequence) if sequence else "",
        "abs_path": abs_path,
        "size_bytes": "0",
        "mtime_iso": "",
        "width": "64",
        "height": "48",
        "fps": "30.0",
        "codec": "mpeg4",
        "media_type": "video",
        "video_order": video_order,
    }


def _identity(path: Path) -> dict[str, str]:
    """The identity cells a fresh probe of *path* measures.

    Seeding a row with these makes it classify ``unchanged``, so a test can put
    an index into the already-migrated state without an applied run whose
    backups would confuse a later backup assertion.
    """
    probe = probe_video_metadata(path)
    return {
        "video_uuid": probe["video_uuid"],
        "content_digest": probe["content_digest"],
    }


def _index_path(ds: Dataset) -> Path:
    return ds.get_root(ds.resolve_media_root()) / "index.csv"


def _derivative_index_path(ds: Dataset) -> Path:
    return ds.get_root("media") / "index.csv"


def _seed(ds: Dataset, rows: list[dict[str, object]]) -> Path:
    index_path = _index_path(ds)
    write_media_index_rows(index_path, frame_from_rows(rows))
    return index_path


def _write_withread_index_header(
    index_path: Path, columns: list[str], rows: list[dict[str, str]]
) -> Path:
    """Write an index whose header is exactly *columns*.

    The only way to seed a header that is not the current schema: the production
    writer projects onto ``MEDIA_INDEX_COLUMNS``, so it can express neither an
    absent column nor an extra one.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return index_path


def _seed_legacy(ds: Dataset, rows: list[dict[str, str]]) -> Path:
    """Write a media index whose header is only the pre-identity subset."""
    return _write_withread_index_header(_index_path(ds), LEGACY_COLUMNS, rows)


def _rows_by_name(index_path: Path) -> dict[str, dict[str, str]]:
    return {record["name"]: record for record in read_media_index(index_path)}


def _by_name(ds: Dataset) -> dict[str, dict[str, str]]:
    return _rows_by_name(_index_path(ds))


def _mutable_rows(index_path: Path) -> list[dict[str, object]]:
    return [
        {key: value for key, value in record.items()}
        for record in read_media_index(index_path)
    ]


def _backups(index_path: Path) -> list[Path]:
    return sorted(index_path.parent.glob("*.backup"))


def _probed_row(path: Path) -> ProbedRow:
    stat_result = path.stat()
    return ProbedRow(
        row_number=0,
        stored_path=str(path),
        resolved_path=path.resolve(),
        probe=probe_video_metadata(path),
        size_bytes=stat_result.st_size,
        modified_at="1970-01-01T00:00:00+00:00",
        change="minted",
        needs_patch=True,
        facts_rebuilt=False,
    )


# --- the write set ---------------------------------------------------------


def test_the_patch_writes_exactly_the_measured_columns(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)

    written = set(measured_cells(_probed_row(clip)))

    assert written == WRITABLE_COLUMNS, (
        "the re-probe write set changed; a column added to the probe row widens "
        "what a patch overwrites, which the media index is authoritative for"
    )


def test_the_patch_never_writes_a_transcode_link(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)

    written = set(measured_cells(_probed_row(clip)))

    assert not (written & UNMEASURED_LINK_COLUMNS), (
        "a fresh probe leaves the transcode link cells empty, so writing them "
        "would erase the index's link graph"
    )


def test_every_written_column_is_in_the_schema(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)

    written = set(measured_cells(_probed_row(clip)))

    assert written <= set(MEDIA_INDEX_COLUMNS)


# --- legacy schema migration and cell fidelity -----------------------------


def test_a_legacy_index_is_migrated_to_the_full_schema(
    dataset: Dataset,
    write_cfr_mp4: WriteVideo,
    read_index_header: ReadHeader,
) -> None:
    write_cfr_mp4(dataset.base_dir / "media_raw" / "seq" / "a.mp4")
    index_path = _seed_legacy(
        dataset,
        [
            _legacy_row(
                name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4"
            )
        ],
    )
    assert "video_uuid" not in read_index_header(index_path)

    report = dataset.reprobe_media(apply=True)

    assert "video_uuid" in report.schema_columns_added
    assert "content_digest" in report.schema_columns_added
    # Exact list, so both membership and order are pinned: the writer projects
    # onto the canonical schema, and a reader keyed by position would break if
    # the order drifted.
    assert read_index_header(index_path) == MEDIA_INDEX_COLUMNS
    row = _by_name(dataset)["a.mp4"]
    assert row["video_uuid"]
    assert row["content_digest"]
    assert row["media_facts"]


def test_na_shaped_cells_survive_the_rewrite(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # pandas' CSV reader turns each of these literals into NaN, which is the
    # data-loss defect the exact-string read path exists to prevent. The
    # assertions are on the literal text, not merely on the cell being non-empty.
    write_cfr_mp4(dataset.base_dir / "media_raw" / "na" / "a.mp4")
    _ = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="NA",
                sequence="None",
                abs_path="media_raw/na/a.mp4",
                camera="NULL",
                sync_uuid="None",
            )
        ],
    )

    _ = dataset.reprobe_media(apply=True)

    row = _by_name(dataset)["a.mp4"]
    assert row["group"] == "NA"
    assert row["sequence"] == "None"
    assert row["camera"] == "NULL"
    assert row["sync_uuid"] == "None"
    assert row["video_uuid"]


def test_a_legacy_index_reports_no_drift(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # Absence is checked before divergence and wins: a row that recorded no
    # identity has nothing for the fresh measurement to disagree with, so the
    # migrating run over a production index reports zero drift.
    base = dataset.base_dir
    write_cfr_mp4(base / "media_raw" / "a" / "a.mp4")
    write_cfr_mp4(base / "media_raw" / "b" / "b.mp4", frames=9)
    _ = _seed_legacy(
        dataset,
        [
            _legacy_row(
                name="a.mp4", group="", sequence="a", abs_path="media_raw/a/a.mp4"
            ),
            _legacy_row(
                name="b.mp4", group="", sequence="b", abs_path="media_raw/b/b.mp4"
            ),
        ],
    )

    report = dataset.reprobe_media(apply=True)

    assert report.minted == 2
    assert report.content_digest_changed == []
    assert report.video_uuid_changed == []


# --- what the media index asserts is never re-derived ----------------------


def test_curated_cells_and_transcode_links_are_untouched(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # The directory says "day1"; the index says colony/trial-7, and holds a link
    # graph a fresh probe leaves empty. Re-deriving either would destroy data
    # that exists nowhere else on a detached dataset.
    write_cfr_mp4(dataset.base_dir / "media_raw" / "day1" / "clip.mp4")
    curated = _row(
        name="clip.mp4",
        group="colony",
        sequence="trial-7",
        abs_path="media_raw/day1/clip.mp4",
        video_order="3",
        camera="cam-2",
        sync_uuid="recording-9",
        analysis_derivative_path="trial-7.analysis.mp4",
        playback_derivative_path="trial-7.playback.mp4",
        source_path="day1/clip.mp4",
        source_video_uuid="00000000-0000-8000-8000-00000000000a",
    )
    _ = _seed(dataset, [curated])

    _ = dataset.reprobe_media(apply=True)

    row = _by_name(dataset)["clip.mp4"]
    for column in (
        "name",
        "group",
        "sequence",
        "video_order",
        "abs_path",
        "camera",
        "sync_uuid",
        "analysis_derivative_path",
        "playback_derivative_path",
        "source_path",
        "source_video_uuid",
    ):
        assert row[column] == curated[column], column
    assert row["video_uuid"]


def test_video_order_survives_including_a_blank_cell(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # A curated order that contradicts the filename order, plus one row with no
    # order at all -- the shape that makes a re-parsing reader emit "0.0".
    base = dataset.base_dir
    for index, filename in enumerate(("a.mp4", "b.mp4", "c.mp4")):
        write_cfr_mp4(base / "media_raw" / "seq" / filename, frames=6 + index)
    _ = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/a.mp4",
                video_order="1",
            ),
            _row(
                name="b.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/b.mp4",
                video_order="",
            ),
            _row(
                name="c.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/c.mp4",
                video_order="0",
            ),
        ],
    )

    _ = dataset.reprobe_media(apply=True)

    rows = _by_name(dataset)
    assert rows["a.mp4"]["video_order"] == "1"
    assert rows["b.mp4"]["video_order"] == ""
    assert rows["c.mp4"]["video_order"] == "0"


def test_external_rows_are_probed_and_keep_their_path(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> None:
    resolved_tmp = tmp_path.resolve()
    ds = make_media_dataset(resolved_tmp / "dataset")
    external = resolved_tmp / "nas" / "clip.mp4"
    write_cfr_mp4(external)
    _ = _seed(
        ds,
        [_row(name="clip.mp4", group="", sequence="remote", abs_path=str(external))],
    )

    report = ds.reprobe_media(apply=True)

    assert report.minted == 1
    row = _by_name(ds)["clip.mp4"]
    assert row["video_uuid"]
    assert row["abs_path"] == str(external)


def test_safe_names_are_filled_when_absent_and_never_overwritten(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # group_safe and sequence_safe are pure functions of cells the row already
    # holds, so an absent one is filled from its own row; a stored one that
    # differs from the derivation is a curated choice this command does not
    # overrule.
    base = dataset.base_dir
    write_cfr_mp4(base / "media_raw" / "one" / "a.mp4")
    write_cfr_mp4(base / "media_raw" / "two" / "b.mp4", frames=9)
    _ = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="colony",
                sequence="trial 7",
                abs_path="media_raw/one/a.mp4",
                group_safe="",
                sequence_safe="",
            ),
            _row(
                name="b.mp4",
                group="colony",
                sequence="trial 8",
                abs_path="media_raw/two/b.mp4",
                sequence_safe="curated",
            ),
        ],
    )

    _ = dataset.reprobe_media(apply=True)

    rows = _by_name(dataset)
    assert rows["a.mp4"]["group_safe"] == to_safe_name("colony")
    assert rows["a.mp4"]["sequence_safe"] == to_safe_name("trial 7")
    assert rows["b.mp4"]["sequence_safe"] == "curated"


# --- derivative links ------------------------------------------------------


def _seed_with_derivative(dataset: Dataset, write_cfr_mp4: WriteVideo) -> Path:
    """An original with a forward link, and its derivative row, both unminted."""
    base = dataset.base_dir
    write_cfr_mp4(base / "media_raw" / "seq" / "a.mp4")
    write_cfr_mp4(base / "media" / "seq.analysis.mp4", frames=4)
    _ = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/a.mp4",
                analysis_derivative_path="seq.analysis.mp4",
            )
        ],
    )
    derivative_index = _derivative_index_path(dataset)
    write_media_index_rows(
        derivative_index,
        frame_from_rows(
            [
                _row(
                    name="seq.analysis.mp4",
                    group="",
                    sequence="seq",
                    abs_path="media/seq.analysis.mp4",
                    source_path="seq/a.mp4",
                )
            ]
        ),
    )
    return derivative_index


def test_derivative_links_survive_and_rekey_onto_source_video_uuid(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    derivative_index = _seed_with_derivative(dataset, write_cfr_mp4)

    report = dataset.reprobe_media(apply=True)

    original = _by_name(dataset)["a.mp4"]
    assert original["analysis_derivative_path"] == "seq.analysis.mp4"
    assert original["video_uuid"]
    assert report.derivative.relinked == 1
    derivative = _rows_by_name(derivative_index)["seq.analysis.mp4"]
    # source_path is the stored back-link by path and is never rewritten;
    # source_video_uuid is re-keyed from the source's fresh measurement.
    assert derivative["source_path"] == "seq/a.mp4"
    assert derivative["source_video_uuid"] == original["video_uuid"]
    assert derivative["video_uuid"]
    assert derivative["video_uuid"] != original["video_uuid"]


def test_a_skipped_derivative_row_keeps_its_back_link(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # A derivative whose own media will not probe is left verbatim, back-link
    # included: re-keying it would break the promise that a skipped row is
    # untouched, and nothing reads the uuid of a derivative that cannot be read.
    base = dataset.base_dir
    write_cfr_mp4(base / "media_raw" / "seq" / "a.mp4")
    (base / "media").mkdir(parents=True, exist_ok=True)
    _ = (base / "media" / "seq.analysis.mp4").write_bytes(b"not a video")
    _ = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/a.mp4",
            )
        ],
    )
    derivative_index = _derivative_index_path(dataset)
    stale_uuid = "00000000-0000-8000-8000-00000000000b"
    write_media_index_rows(
        derivative_index,
        frame_from_rows(
            [
                _row(
                    name="seq.analysis.mp4",
                    group="",
                    sequence="seq",
                    abs_path="media/seq.analysis.mp4",
                    source_path="seq/a.mp4",
                    source_video_uuid=stale_uuid,
                )
            ]
        ),
    )
    before = derivative_index.read_bytes()

    report = dataset.reprobe_media(apply=True, skip_unreadable=True)

    assert report.minted == 1
    assert len(report.derivative.unprobeable) == 1
    assert report.derivative.relinked == 0
    assert derivative_index.read_bytes() == before
    derivative = _rows_by_name(derivative_index)["seq.analysis.mp4"]
    assert derivative["source_video_uuid"] == stale_uuid
    assert derivative["source_path"] == "seq/a.mp4"


def test_a_reprobed_uuid_is_the_recorded_one_and_links_still_resolve(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # video_uuid is derived from probed content and timing, so an unchanged file
    # re-probes to the value already stored -- which is what keeps a back-link
    # keyed on it valid across a re-probe.
    derivative_index = _seed_with_derivative(dataset, write_cfr_mp4)
    _ = dataset.reprobe_media(apply=True)
    minted = _by_name(dataset)["a.mp4"]["video_uuid"]
    after_first = derivative_index.read_bytes()

    second = dataset.reprobe_media(apply=True)

    assert not second.changed
    assert second.backups == []
    assert derivative_index.read_bytes() == after_first
    assert _by_name(dataset)["a.mp4"]["video_uuid"] == minted
    derivative = _rows_by_name(derivative_index)["seq.analysis.mp4"]
    assert derivative["source_video_uuid"] == minted


# --- aborts and --skip-unreadable ------------------------------------------


def _seed_one_of_each_unreadable(dataset: Dataset, write_cfr_mp4: WriteVideo) -> Path:
    """A media index with one good row, one missing file, and one corrupt file."""
    base = dataset.base_dir
    write_cfr_mp4(base / "media_raw" / "seq" / "a.mp4")
    _ = (base / "media_raw" / "seq" / "broken.mp4").write_bytes(b"not a video")
    return _seed_legacy(
        dataset,
        [
            _legacy_row(
                name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4"
            ),
            _legacy_row(
                name="gone.mp4",
                group="g",
                sequence="dead",
                abs_path="media_raw/seq/gone.mp4",
                video_order="4",
            ),
            _legacy_row(
                name="broken.mp4",
                group="g",
                sequence="corrupt",
                abs_path="media_raw/seq/broken.mp4",
                video_order="7",
            ),
        ],
    )


def test_a_missing_media_file_aborts_and_writes_nothing(
    dataset: Dataset,
    write_cfr_mp4: WriteVideo,
    read_index_header: ReadHeader,
) -> None:
    write_cfr_mp4(dataset.base_dir / "media_raw" / "seq" / "a.mp4")
    index_path = _seed_legacy(
        dataset,
        [
            _legacy_row(
                name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4"
            ),
            _legacy_row(
                name="gone.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/gone.mp4",
            ),
        ],
    )
    before = index_path.read_bytes()

    with pytest.raises(ReprobeAbort) as caught:
        _ = dataset.reprobe_media(apply=True)

    assert any("gone.mp4" in detail for detail in caught.value.missing)
    assert caught.value.unprobeable == []
    assert "not on disk" in str(caught.value)
    assert "cannot be probed" not in str(caught.value)
    # Nothing written: not the values, and not the schema either.
    assert index_path.read_bytes() == before
    assert read_index_header(index_path) == LEGACY_COLUMNS
    assert _backups(index_path) == []


def test_an_unprobeable_file_aborts_and_writes_nothing(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    base = dataset.base_dir
    write_cfr_mp4(base / "media_raw" / "seq" / "a.mp4")
    _ = (base / "media_raw" / "seq" / "broken.mp4").write_bytes(b"not a video")
    index_path = _seed(
        dataset,
        [
            _row(
                name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4"
            ),
            _row(
                name="broken.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/broken.mp4",
            ),
        ],
    )
    before = index_path.read_bytes()

    with pytest.raises(ReprobeAbort) as caught:
        _ = dataset.reprobe_media(apply=True)

    assert any("broken.mp4" in detail for detail in caught.value.unprobeable)
    assert caught.value.missing == []
    assert "cannot be probed" in str(caught.value)
    assert "not on disk" not in str(caught.value)
    assert index_path.read_bytes() == before
    assert _backups(index_path) == []


def test_the_abort_names_both_conditions_and_keeps_them_apart(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # One flag governs both, but the operator is told which rows are archived
    # media and which are a corruption signal.
    _ = _seed_one_of_each_unreadable(dataset, write_cfr_mp4)

    with pytest.raises(ReprobeAbort) as caught:
        _ = dataset.reprobe_media(apply=True)

    message = str(caught.value)
    assert "1 row(s) point at media that is not on disk" in message
    assert "1 row(s) point at media that cannot be probed" in message
    assert len(caught.value.missing) == 1
    assert "gone.mp4" in caught.value.missing[0]
    assert len(caught.value.unprobeable) == 1
    assert "broken.mp4" in caught.value.unprobeable[0]


def test_skip_unreadable_reports_the_two_conditions_apart(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    _ = _seed_one_of_each_unreadable(dataset, write_cfr_mp4)

    report = dataset.reprobe_media(apply=True, skip_unreadable=True)

    assert len(report.missing) == 1
    assert "gone.mp4" in report.missing[0]
    assert len(report.unprobeable) == 1
    assert "broken.mp4" in report.unprobeable[0]
    payload = report.payload()
    assert payload["media_missing_count"] == 1
    assert payload["media_missing"] == report.missing
    assert payload["media_unprobeable_count"] == 1
    assert payload["media_unprobeable"] == report.unprobeable


def test_skip_unreadable_leaves_both_rows_verbatim_and_migrates_the_rest(
    dataset: Dataset,
    write_cfr_mp4: WriteVideo,
    read_index_header: ReadHeader,
) -> None:
    index_path = _seed_one_of_each_unreadable(dataset, write_cfr_mp4)

    report = dataset.reprobe_media(apply=True, skip_unreadable=True)

    assert report.minted == 1
    # The schema still migrates for every row, including the skipped ones.
    assert read_index_header(index_path) == MEDIA_INDEX_COLUMNS
    rows = _by_name(dataset)
    assert rows["a.mp4"]["video_uuid"]
    for name, group, sequence, order in (
        ("gone.mp4", "g", "dead", "4"),
        ("broken.mp4", "g", "corrupt", "7"),
    ):
        assert rows[name]["video_uuid"] == ""
        assert rows[name]["content_digest"] == ""
        assert rows[name]["media_facts"] == ""
        assert rows[name]["group"] == group
        assert rows[name]["sequence"] == sequence
        assert rows[name]["video_order"] == order
        # The unmeasured filesystem cells keep the values the legacy file held.
        assert rows[name]["size_bytes"] == "0"
        assert rows[name]["mtime_iso"] == ""


# --- dry-run, apply, backups, idempotence ----------------------------------


def test_dry_run_writes_nothing(dataset: Dataset, write_cfr_mp4: WriteVideo) -> None:
    write_cfr_mp4(dataset.base_dir / "media_raw" / "seq" / "a.mp4")
    index_path = _seed(
        dataset,
        [_row(name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4")],
    )
    before = index_path.read_bytes()

    report = dataset.reprobe_media(apply=False)

    assert report.changed
    assert not report.applied
    assert report.minted == 1
    assert report.backups == []
    assert index_path.read_bytes() == before
    assert _backups(index_path) == []


def test_apply_writes_and_leaves_one_backup(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    write_cfr_mp4(dataset.base_dir / "media_raw" / "seq" / "a.mp4")
    index_path = _seed(
        dataset,
        [_row(name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4")],
    )
    before = index_path.read_bytes()

    report = dataset.reprobe_media(apply=True)

    assert report.applied
    assert len(report.backups) == 1
    assert report.backups[0].read_bytes() == before
    assert index_path.read_bytes() != before
    assert _backups(index_path) == report.backups


def _assert_second_run_is_a_no_op(
    index_path: Path,
    second: ReprobeReport,
    after_first: bytes,
    backups_after_first: list[Path],
) -> None:
    """The invariant shared by every "a second applied run changes nothing" test.

    A prior run already settled the index, so the second run must report no
    change, apply nothing, and patch no row -- and the file and its backup set
    must come out exactly as the first run left them. The exact listing
    comparison is the real guard on the backup set: a bare count would still
    pass if a same-second run overwrote its backup (one-second stamp
    granularity) instead of adding one, so it is kept alongside rather than in
    place of it.
    """
    assert not second.changed
    assert not second.applied
    assert second.rows_patched == 0
    assert second.backups == []
    assert index_path.read_bytes() == after_first
    assert _backups(index_path) == backups_after_first


def test_a_second_run_changes_nothing_and_leaves_no_second_backup(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    base = dataset.base_dir
    write_cfr_mp4(base / "media_raw" / "seq" / "a.mp4")
    write_cfr_mp4(base / "media_raw" / "other" / "b.mp4", frames=9)
    index_path = _seed_legacy(
        dataset,
        [
            _legacy_row(
                name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4"
            ),
            _legacy_row(
                name="b.mp4",
                group="g",
                sequence="other",
                abs_path="media_raw/other/b.mp4",
            ),
        ],
    )

    first = dataset.reprobe_media(apply=True)
    after_first = index_path.read_bytes()
    backups_after_first = _backups(index_path)
    second = dataset.reprobe_media(apply=True)

    assert first.changed
    assert second.unchanged == 2
    _assert_second_run_is_a_no_op(index_path, second, after_first, backups_after_first)


def test_an_imgstore_row_does_not_force_a_rewrite_every_run(
    dataset: Dataset, make_imgstore: MakeImgstore
) -> None:
    # store_facts gives a store an empty identity permanently, so it is always
    # "unmintable". Its stored facts, not its identity, decide whether it needs a
    # write -- otherwise every applied run would rewrite it and drop another
    # backup, without bound.
    store_dir, _frames = make_imgstore(
        name="store", parent=dataset.base_dir / "media_raw"
    )
    index_path = _seed(
        dataset,
        [
            _row(
                name=store_dir.name,
                group="",
                sequence="store",
                abs_path=f"media_raw/{store_dir.name}",
                media_type="imgstore",
            )
        ],
    )

    first = dataset.reprobe_media(apply=True)
    after_first = index_path.read_bytes()
    backups_after_first = _backups(index_path)
    second = dataset.reprobe_media(apply=True)

    assert first.changed
    assert first.unmintable == 1
    assert _by_name(dataset)[store_dir.name]["media_facts"]
    assert len(backups_after_first) == 1
    assert second.unmintable == 1
    _assert_second_run_is_a_no_op(index_path, second, after_first, backups_after_first)


def _stale_facts_json(video: Path) -> str:
    """A facts cell as it would have been written before the two provenance
    fields existed: the current probe's payload, with ``identity_scheme`` and
    ``prober_version`` stripped back out.
    """
    payload = json.loads(probe_video_metadata(video)["media_facts"])
    del payload["identity_scheme"]
    del payload["prober_version"]
    return json.dumps(payload)


def test_a_present_but_unparsable_facts_cell_is_rewritten_and_then_reconstructs(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # The stale cell still mints the same uuid and digest from the same bytes,
    # so identity classifies this row "unchanged" -- the one state identity
    # comparison cannot see. Only an applied run that reconstructs the cell
    # itself can tell it is stale.
    video = dataset.base_dir / "media_raw" / "seq" / "a.mp4"
    write_cfr_mp4(video)
    _ = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/a.mp4",
                **_identity(video),
                media_facts=_stale_facts_json(video),
            )
        ],
    )

    report = dataset.reprobe_media(apply=True)

    assert report.changed
    assert report.rows_patched == 1
    # The rewrite's only reason, so it is the count that explains it.
    assert report.facts_rebuilt == 1
    row = _by_name(dataset)["a.mp4"]
    reconstructed = row_to_facts(row)
    assert reconstructed.video_uuid == row["video_uuid"]
    assert reconstructed.identity_scheme != ""
    assert reconstructed.prober_version != ""


def test_the_repaired_facts_cell_does_not_force_a_second_rewrite(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    video = dataset.base_dir / "media_raw" / "seq" / "a.mp4"
    write_cfr_mp4(video)
    index_path = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/a.mp4",
                **_identity(video),
                media_facts=_stale_facts_json(video),
            )
        ],
    )

    first = dataset.reprobe_media(apply=True)
    after_first = index_path.read_bytes()
    backups_after_first = _backups(index_path)
    second = dataset.reprobe_media(apply=True)

    assert first.changed
    _assert_second_run_is_a_no_op(index_path, second, after_first, backups_after_first)


def test_a_drifted_row_is_not_also_counted_as_a_facts_rebuild(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # This rewrite already has a reason of its own -- the content_digest line --
    # and the facts count exists only to explain a rewrite nothing else names.
    # Counting the row twice would read as two rewritten rows.
    video = dataset.base_dir / "media_raw" / "seq" / "a.mp4"
    write_cfr_mp4(video)
    _ = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/a.mp4",
                video_uuid=_identity(video)["video_uuid"],
                content_digest="0" * 64,
                media_facts=_stale_facts_json(video),
            )
        ],
    )

    report = dataset.reprobe_media(apply=False)

    assert report.content_digest_changed == ["media_raw/seq/a.mp4"]
    assert report.rows_patched == 1
    assert report.facts_rebuilt == 0
    assert report.payload()["facts_rebuilt"] == 0


def test_a_derivative_facts_rebuild_is_counted_on_the_derivative_plan(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # The derivative pass runs the same heal on rows classified exactly as the
    # originals are, so the count is reported per index rather than merged into
    # one number that names neither file.
    base = dataset.base_dir
    source = base / "media_raw" / "seq" / "a.mp4"
    derivative_video = base / "media" / "seq.analysis.mp4"
    write_cfr_mp4(source)
    write_cfr_mp4(derivative_video, frames=4)
    _ = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/a.mp4",
                analysis_derivative_path="seq.analysis.mp4",
                **_identity(source),
            )
        ],
    )
    write_media_index_rows(
        _derivative_index_path(dataset),
        frame_from_rows(
            [
                _row(
                    name="seq.analysis.mp4",
                    group="",
                    sequence="seq",
                    abs_path="media/seq.analysis.mp4",
                    source_path="seq/a.mp4",
                    # Already back-linked, so the stale facts cell is the only
                    # thing left for the run to rewrite this row for.
                    source_video_uuid=_identity(source)["video_uuid"],
                    **_identity(derivative_video),
                    media_facts=_stale_facts_json(derivative_video),
                )
            ]
        ),
    )

    report = dataset.reprobe_media(apply=False)

    assert report.rows_patched == 0
    assert report.facts_rebuilt == 0
    assert report.derivative.relinked == 0
    assert report.derivative.facts_rebuilt == 1
    assert report.payload()["derivative_facts_rebuilt"] == 1


def test_a_derivative_only_run_leaves_no_backup_beside_the_originals_index(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # Each backup sits inside the branch that rewrites its own file. Here the
    # originals index is already fully probed, so only the derivative back-link
    # is outstanding and only the derivative index is copied aside.
    base = dataset.base_dir
    source = base / "media_raw" / "seq" / "a.mp4"
    derivative_video = base / "media" / "seq.analysis.mp4"
    write_cfr_mp4(source)
    write_cfr_mp4(derivative_video, frames=4)
    index_path = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/a.mp4",
                analysis_derivative_path="seq.analysis.mp4",
                **_identity(source),
            )
        ],
    )
    derivative_index = _derivative_index_path(dataset)
    write_media_index_rows(
        derivative_index,
        frame_from_rows(
            [
                _row(
                    name="seq.analysis.mp4",
                    group="",
                    sequence="seq",
                    abs_path="media/seq.analysis.mp4",
                    source_path="seq/a.mp4",
                    **_identity(derivative_video),
                )
            ]
        ),
    )
    originals_before = index_path.read_bytes()

    report = dataset.reprobe_media(apply=True)

    assert report.changed
    assert report.rows_patched == 0
    assert report.derivative.relinked == 1
    assert len(report.backups) == 1
    assert report.backups[0].parent == derivative_index.parent
    assert index_path.read_bytes() == originals_before
    assert _backups(index_path) == []
    assert len(_backups(derivative_index)) == 1


# --- drift -----------------------------------------------------------------


def test_changed_bytes_are_reported_as_a_content_digest_divergence(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    video = dataset.base_dir / "media_raw" / "seq" / "a.mp4"
    write_cfr_mp4(video)
    _ = _seed(
        dataset,
        [_row(name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4")],
    )
    _ = dataset.reprobe_media(apply=True)
    recorded = _by_name(dataset)["a.mp4"]["content_digest"]

    # The same filename, different content.
    write_cfr_mp4(video, frames=12)
    report = dataset.reprobe_media(apply=True)

    assert report.content_digest_changed == ["media_raw/seq/a.mp4"]
    assert report.video_uuid_changed == []
    assert report.minted == 0
    assert report.applied
    # The fresh measurement wins.
    written = _by_name(dataset)["a.mp4"]["content_digest"]
    assert written
    assert written != recorded
    assert written == _identity(video)["content_digest"]


def test_a_disagreeing_recorded_uuid_is_reported_not_fatal(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # Keep the recorded content_digest, replace the recorded video_uuid: the
    # state a container rewrite at unchanged coded content leaves behind.
    video = dataset.base_dir / "media_raw" / "seq" / "a.mp4"
    write_cfr_mp4(video)
    index_path = _seed(
        dataset,
        [
            _row(
                name="a.mp4",
                group="",
                sequence="seq",
                abs_path="media_raw/seq/a.mp4",
                **_identity(video),
            )
        ],
    )
    probed_uuid = _identity(video)["video_uuid"]
    rows = _mutable_rows(index_path)
    rows[0]["video_uuid"] = "00000000-0000-8000-8000-000000000000"
    write_media_index_rows(index_path, frame_from_rows(rows))

    report = dataset.reprobe_media(apply=True)

    assert report.video_uuid_changed == ["media_raw/seq/a.mp4"]
    assert report.content_digest_changed == []
    assert report.minted == 0
    assert _by_name(dataset)["a.mp4"]["video_uuid"] == probed_uuid


def test_a_partially_probed_index_completes_only_what_is_missing(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    base = dataset.base_dir
    write_cfr_mp4(base / "media_raw" / "seq" / "a.mp4")
    index_path = _seed(
        dataset,
        [_row(name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4")],
    )
    _ = dataset.reprobe_media(apply=True)
    done = _by_name(dataset)["a.mp4"]

    write_cfr_mp4(base / "media_raw" / "other" / "b.mp4", frames=9)
    rows = _mutable_rows(index_path)
    rows.append(
        _row(
            name="b.mp4",
            group="g",
            sequence="other",
            abs_path="media_raw/other/b.mp4",
        )
    )
    write_media_index_rows(index_path, frame_from_rows(rows))

    report = dataset.reprobe_media(apply=True)

    assert report.minted == 1
    assert report.unchanged == 1
    assert report.rows_patched == 1
    after = _by_name(dataset)
    assert after["b.mp4"]["video_uuid"]
    assert after["a.mp4"]["video_uuid"] == done["video_uuid"]
    assert after["a.mp4"]["mtime_iso"] == done["mtime_iso"]


def test_the_report_serializes_to_json(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    write_cfr_mp4(dataset.base_dir / "media_raw" / "seq" / "a.mp4")
    _ = _seed(
        dataset,
        [_row(name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4")],
    )

    decoded = json.loads(json.dumps(dataset.reprobe_media(apply=False).payload()))

    assert decoded["changed"] is True
    assert decoded["applied"] is False
    assert decoded["identity_minted"] == 1
    assert decoded["media_missing"] == []
    assert decoded["media_unprobeable"] == []


# --- columns outside the schema --------------------------------------------

CURATED_COLUMN = "operator_note"


def test_a_column_outside_the_schema_is_dropped_reported_and_recoverable(
    dataset: Dataset,
    write_cfr_mp4: WriteVideo,
    read_index_header: ReadHeader,
) -> None:
    # The one thing this command destroys. Every other cell survives a rewrite
    # byte-identical; a column mosaic never defined cannot, because the writer
    # projects onto MEDIA_INDEX_COLUMNS. So the report has to name it and the
    # backup has to still hold it -- that pair is the whole mitigation.
    write_cfr_mp4(dataset.base_dir / "media_raw" / "seq" / "a.mp4")
    index_path = _write_withread_index_header(
        _index_path(dataset),
        [*LEGACY_COLUMNS, CURATED_COLUMN],
        [
            {
                **_legacy_row(
                    name="a.mp4",
                    group="",
                    sequence="seq",
                    abs_path="media_raw/seq/a.mp4",
                ),
                CURATED_COLUMN: "collected by MW, do not delete",
            }
        ],
    )

    report = dataset.reprobe_media(apply=True)

    assert report.unknown_columns == [CURATED_COLUMN]
    assert report.payload()["unknown_columns_dropped"] == [CURATED_COLUMN]
    # Gone from the index...
    assert CURATED_COLUMN not in read_index_header(index_path)
    assert read_index_header(index_path) == MEDIA_INDEX_COLUMNS
    # ...and still in the backup, which is the only way back.
    assert len(report.backups) == 1
    backup = report.backups[0]
    assert CURATED_COLUMN in read_index_header(backup)
    backup_row = _rows_by_name(backup)["a.mp4"]
    assert backup_row[CURATED_COLUMN] == "collected by MW, do not delete"


def test_a_dry_run_reports_the_column_it_would_drop_but_drops_nothing(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    write_cfr_mp4(dataset.base_dir / "media_raw" / "seq" / "a.mp4")
    index_path = _write_withread_index_header(
        _index_path(dataset),
        [*LEGACY_COLUMNS, CURATED_COLUMN],
        [
            {
                **_legacy_row(
                    name="a.mp4",
                    group="",
                    sequence="seq",
                    abs_path="media_raw/seq/a.mp4",
                ),
                CURATED_COLUMN: "keep",
            }
        ],
    )
    before = index_path.read_bytes()

    report = dataset.reprobe_media(apply=False)

    assert report.unknown_columns == [CURATED_COLUMN]
    assert index_path.read_bytes() == before


def test_an_unchanged_run_reports_no_dropped_column(
    dataset: Dataset,
    write_cfr_mp4: WriteVideo,
    read_index_header: ReadHeader,
) -> None:
    # A fully migrated index that happens to carry an extra column needs no
    # rewrite, so nothing is dropped and the report must not warn about a write
    # that will not happen. The column is still there afterwards.
    video = dataset.base_dir / "media_raw" / "seq" / "a.mp4"
    write_cfr_mp4(video)
    seeded = _row(
        name="a.mp4",
        group="",
        sequence="seq",
        abs_path="media_raw/seq/a.mp4",
        **_identity(video),
    )
    probe = probe_video_metadata(video)
    stat_result = video.stat()
    seeded.update(
        {
            "media_facts": probe["media_facts"],
            "size_bytes": str(stat_result.st_size),
        }
    )
    index_path = _write_withread_index_header(
        _index_path(dataset),
        [*MEDIA_INDEX_COLUMNS, CURATED_COLUMN],
        [{**{k: str(v) for k, v in seeded.items()}, CURATED_COLUMN: "keep"}],
    )
    before = index_path.read_bytes()

    report = dataset.reprobe_media(apply=True)

    assert not report.changed
    assert report.unknown_columns == []
    assert report.payload()["unknown_columns_dropped"] == []
    assert index_path.read_bytes() == before
    assert CURATED_COLUMN in read_index_header(index_path)


def test_a_column_dropped_from_the_derivative_index_is_reported(
    dataset: Dataset,
    write_cfr_mp4: WriteVideo,
    read_index_header: ReadHeader,
) -> None:
    # Spec 7.2 does not distinguish the two files: a curated column in the
    # media index is destroyed exactly as one in the media_raw index is.
    base = dataset.base_dir
    write_cfr_mp4(base / "media_raw" / "seq" / "a.mp4")
    write_cfr_mp4(base / "media" / "seq.analysis.mp4", frames=4)
    _ = _seed(
        dataset,
        [_row(name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4")],
    )
    derivative_index = _write_withread_index_header(
        _derivative_index_path(dataset),
        [*LEGACY_COLUMNS, "source_path", CURATED_COLUMN],
        [
            {
                **_legacy_row(
                    name="seq.analysis.mp4",
                    group="",
                    sequence="seq",
                    abs_path="media/seq.analysis.mp4",
                ),
                "source_path": "seq/a.mp4",
                CURATED_COLUMN: "encoded on the old box",
            }
        ],
    )

    report = dataset.reprobe_media(apply=True)

    assert report.derivative.unknown_columns == [CURATED_COLUMN]
    payload = report.payload()
    assert payload["derivative_unknown_columns_dropped"] == [CURATED_COLUMN]
    assert CURATED_COLUMN not in read_index_header(derivative_index)


# --- unresolved back-links and drift in the derivative index ---------------


def test_a_back_link_to_no_indexed_original_is_reported_unresolved(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # source_path names a file the media_raw index does not list, so there is no
    # measured identity to re-key onto. Reported, and the row left alone.
    base = dataset.base_dir
    write_cfr_mp4(base / "media_raw" / "seq" / "a.mp4")
    write_cfr_mp4(base / "media" / "seq.analysis.mp4", frames=4)
    _ = _seed(
        dataset,
        [_row(name="a.mp4", group="", sequence="seq", abs_path="media_raw/seq/a.mp4")],
    )
    derivative_index = _derivative_index_path(dataset)
    write_media_index_rows(
        derivative_index,
        frame_from_rows(
            [
                _row(
                    name="seq.analysis.mp4",
                    group="",
                    sequence="seq",
                    abs_path="media/seq.analysis.mp4",
                    source_path="seq/vanished.mp4",
                )
            ]
        ),
    )

    report = dataset.reprobe_media(apply=True)

    assert len(report.derivative.unresolved) == 1
    assert "seq/vanished.mp4" in report.derivative.unresolved[0]
    assert report.derivative.relinked == 0
    assert (
        report.payload()["derivative_links_unresolved"] == report.derivative.unresolved
    )
    derivative = _rows_by_name(derivative_index)["seq.analysis.mp4"]
    assert derivative["source_path"] == "seq/vanished.mp4"
    assert derivative["source_video_uuid"] == ""


def test_derivative_drift_is_counted_and_reported(
    dataset: Dataset, write_cfr_mp4: WriteVideo
) -> None:
    # A derivative is a file like any other: re-encoded in place, its recorded
    # identity no longer describes it, and the run must say so.
    derivative_index = _seed_with_derivative(dataset, write_cfr_mp4)
    _ = dataset.reprobe_media(apply=True)

    write_cfr_mp4(dataset.base_dir / "media" / "seq.analysis.mp4", frames=11)
    report = dataset.reprobe_media(apply=True)

    assert report.derivative.content_digest_changed == ["media/seq.analysis.mp4"]
    assert report.content_digest_changed == []
    payload = report.payload()
    assert payload["derivative_content_digest_changed"] == ["media/seq.analysis.mp4"]
    assert payload["derivative_video_uuid_changed"] == []
    # The fresh measurement wins.
    written = _rows_by_name(derivative_index)["seq.analysis.mp4"]
    assert (
        written["content_digest"]
        == _identity(dataset.base_dir / "media" / "seq.analysis.mp4")["content_digest"]
    )


# --- aborts before any measurement -----------------------------------------


def test_a_missing_index_file_aborts(dataset: Dataset) -> None:
    # There is nothing to re-probe. This command never builds an index, so an
    # absent one is an abort rather than an empty run.
    index_path = _index_path(dataset)
    assert not index_path.exists()

    with pytest.raises(ReprobeAbort) as caught:
        _ = dataset.reprobe_media(apply=True)

    assert "no media index to re-probe" in str(caught.value)
    assert str(index_path) in str(caught.value)
    assert caught.value.missing == []
    assert caught.value.unprobeable == []


def test_a_manifest_with_no_media_root_aborts(tmp_path: Path) -> None:
    # The abort the CLI's ReprobeAbort handler exists to turn into a clean exit
    # rather than a traceback out of get_root.
    base = (tmp_path / "dataset").resolve()
    base.mkdir(parents=True, exist_ok=True)
    ds = Dataset(manifest_path=base / "dataset.yaml", roots={})
    ds.save()

    with pytest.raises(ReprobeAbort) as caught:
        _ = ds.reprobe_media(apply=True)

    assert "no usable media root" in str(caught.value)


def test_an_imgstore_row_is_detected_without_a_media_type_cell(
    dataset: Dataset, make_imgstore: MakeImgstore
) -> None:
    # A pre-identity index may also predate media_type. Probing a store as a
    # plain video fails, and the row would land in `unprobeable` and abort the
    # whole migration -- so the filesystem answers when the cell does not.
    store_dir, _frames = make_imgstore(
        name="store", parent=dataset.base_dir / "media_raw"
    )
    _ = _seed(
        dataset,
        [
            _row(
                name=store_dir.name,
                group="",
                sequence="store",
                abs_path=f"media_raw/{store_dir.name}",
                media_type="",
            )
        ],
    )

    report = dataset.reprobe_media(apply=True)

    assert report.unprobeable == []
    assert report.unmintable == 1
    assert _by_name(dataset)[store_dir.name]["media_facts"]
