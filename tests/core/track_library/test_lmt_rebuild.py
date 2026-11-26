"""The rebuild manifest, the copy check and the root that holds a rebuilt copy."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.core.track_library.lmt_rebuild import (
    REBUILD_KIND,
    REBUILD_MANIFEST_NAME,
    RebuildManifest,
    copy_unchanged,
    mtime_seconds,
    read_rebuild_manifest,
    rebuilt_copy,
    refuse_foreign_rebuild_run_id,
)
from tests.helpers import (
    LMT_REBUILD_COMMIT,
    LMT_REBUILD_RUN_ID,
    LmtEvent,
    default_lmt_recording,
    lmt_dataset,
    place_rebuilt_copy,
)


def _manifest(copy: Path) -> RebuildManifest:
    """The manifest of *copy*, as written beside it after a rebuild."""
    stat = copy.stat()
    fields: dict[str, object] = {
        "manifest_format": 1,
        "run_id": "rebuild-lmt-events.0.1-abcd1234",
        "op_version": "0.1",
        "lmt_analysis_commit": "0123456789abcdef0123456789abcdef01234567",
        "builders": ("BuildEventNest3", "BuildEventStop"),
        "source_md5": "d41d8cd98f00b204e9800998ecf8427e",
        "source_name": copy.name,
        "source_size": stat.st_size,
        "copy_name": copy.name,
        "copy_size": stat.st_size,
        "copy_mtime_ns": stat.st_mtime_ns,
        "max_frame": 1000,
        "script_max_frame": 1000,
        "python": "3.12.3",
        "packages": {"numpy": "2.1.0"},
        "completed_at": "2026-11-26T12:00:00+00:00",
    }
    return RebuildManifest.model_validate(fields)


@pytest.fixture
def copy(tmp_path: Path) -> Path:
    path = tmp_path / "entry" / "recording.sqlite"
    path.parent.mkdir()
    path.write_bytes(b"x" * 4096)
    return path


def test_a_manifest_written_and_read_back_is_equal(copy: Path) -> None:
    manifest = _manifest(copy)

    manifest.write(copy.parent)

    assert (copy.parent / REBUILD_MANIFEST_NAME).is_file()
    assert read_rebuild_manifest(copy) == manifest


def test_an_unknown_key_names_the_manifest_file(copy: Path) -> None:
    _manifest(copy).write(copy.parent)
    path = copy.parent / REBUILD_MANIFEST_NAME
    document = json.loads(path.read_text())
    document["surprise"] = 1
    _ = path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=REBUILD_MANIFEST_NAME):
        _ = read_rebuild_manifest(copy)


def test_an_unknown_manifest_format_names_the_manifest_file(copy: Path) -> None:
    _manifest(copy).write(copy.parent)
    path = copy.parent / REBUILD_MANIFEST_NAME
    document = json.loads(path.read_text())
    document["manifest_format"] = 2
    _ = path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=REBUILD_MANIFEST_NAME):
        _ = read_rebuild_manifest(copy)


def test_a_missing_manifest_names_the_path(copy: Path) -> None:
    with pytest.raises(ValueError, match=REBUILD_MANIFEST_NAME) as raised:
        _ = read_rebuild_manifest(copy)

    assert str(copy.parent) in str(raised.value)


def test_an_unchanged_copy_is_unchanged(copy: Path) -> None:
    assert copy_unchanged(copy, _manifest(copy))


def test_a_copy_with_an_appended_byte_is_changed(copy: Path) -> None:
    manifest = _manifest(copy)
    stat = copy.stat()
    with copy.open("ab") as handle:
        _ = handle.write(b"y")
    os.utime(copy, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert not copy_unchanged(copy, manifest)


def test_a_copy_whose_time_moved_two_seconds_is_changed(copy: Path) -> None:
    manifest = _manifest(copy)
    stat = copy.stat()
    os.utime(copy, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2 * 10**9))

    assert not copy_unchanged(copy, manifest)


def test_a_copy_whose_time_moved_within_the_second_is_unchanged(
    tmp_path: Path,
) -> None:
    """A server that stores a coarser tick reads the same second back."""
    path = tmp_path / "recording.sqlite"
    path.write_bytes(b"x" * 4096)
    whole_second = 1_800_000_000 * 10**9
    os.utime(path, ns=(whole_second, whole_second + 100_000_000))
    manifest = _manifest(path)
    os.utime(path, ns=(whole_second, whole_second + 600_000_000))

    assert copy_unchanged(path, manifest)


def test_the_modification_time_is_read_in_whole_seconds(tmp_path: Path) -> None:
    path = tmp_path / "recording.sqlite"
    path.write_bytes(b"x")
    os.utime(path, ns=(0, 7 * 10**9 + 999_999_999))

    assert mtime_seconds(path) == 7


def test_the_rebuild_kind_is_a_root_that_reads_no_media() -> None:
    root = TRACKING_ROOTS[REBUILD_KIND]

    assert root.retention == "rebuild"
    assert root.reads == "none"
    assert root.phases == ("rebuild",)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _two_sources(tmp_path: Path) -> tuple[Dataset, Path, Path]:
    """A dataset with two databases that differ in one event."""
    recording = default_lmt_recording()
    other = replace(
        recording, events=(*recording.events, LmtEvent("Contact", 10, 20, a=1, b=2))
    )
    ds = lmt_dataset(tmp_path / "ds", recordings={"first": recording, "second": other})
    raw = ds.base_dir / "raw"
    return ds, raw / "first.sqlite", raw / "second.sqlite"


def test_a_run_id_of_the_rebuild_kind_or_none_is_accepted() -> None:
    assert refuse_foreign_rebuild_run_id(None) is None
    assert refuse_foreign_rebuild_run_id(LMT_REBUILD_RUN_ID) == LMT_REBUILD_RUN_ID


@pytest.mark.parametrize("value", ["convert-tracks.0.1-0123456789", "abc"])
def test_a_run_id_of_another_kind_is_refused_naming_the_value_and_the_kind(
    value: str,
) -> None:
    with pytest.raises(ValueError, match=REBUILD_KIND) as raised:
        _ = refuse_foreign_rebuild_run_id(value)

    assert value in str(raised.value)


def test_the_copy_is_found_by_the_md5_of_its_source(tmp_path: Path) -> None:
    ds, first, second = _two_sources(tmp_path)
    first_copy = place_rebuilt_copy(
        ds, LMT_REBUILD_RUN_ID, first, entry_key="a", commit=LMT_REBUILD_COMMIT
    )
    second_copy = place_rebuilt_copy(
        ds, LMT_REBUILD_RUN_ID, second, entry_key="b", commit=LMT_REBUILD_COMMIT
    )
    root = ds.get_root(REBUILD_KIND)

    assert rebuilt_copy(root, LMT_REBUILD_RUN_ID, _md5(first), str(first)) == first_copy
    assert (
        rebuilt_copy(root, LMT_REBUILD_RUN_ID, _md5(second), str(second)) == second_copy
    )


def test_two_copies_of_one_md5_give_the_first_by_entry_key(tmp_path: Path) -> None:
    ds, first, _ = _two_sources(tmp_path)
    _ = place_rebuilt_copy(
        ds, LMT_REBUILD_RUN_ID, first, entry_key="b", commit=LMT_REBUILD_COMMIT
    )
    earlier = place_rebuilt_copy(
        ds, LMT_REBUILD_RUN_ID, first, entry_key="a", commit=LMT_REBUILD_COMMIT
    )

    found = rebuilt_copy(
        ds.get_root(REBUILD_KIND), LMT_REBUILD_RUN_ID, _md5(first), str(first)
    )

    assert found == earlier


def _refusal(root: Path | None, run_id: str, md5: str, file: Path) -> str:
    with pytest.raises(ValueError) as raised:
        _ = rebuilt_copy(root, run_id, md5, str(file))
    text = str(raised.value)
    assert str(file) in text
    assert run_id in text
    return text


def test_an_empty_md5_is_refused_naming_scan(tmp_path: Path) -> None:
    ds, first, _ = _two_sources(tmp_path)

    text = _refusal(ds.get_root(REBUILD_KIND), LMT_REBUILD_RUN_ID, "", first)

    assert "mosaic scan" in text


def test_a_dataset_without_the_root_is_refused_naming_the_root_and_the_op(
    tmp_path: Path,
) -> None:
    _, first, _ = _two_sources(tmp_path)

    text = _refusal(None, LMT_REBUILD_RUN_ID, _md5(first), first)

    assert "rebuild-lmt-events root" in text
    assert REBUILD_KIND in text


def test_a_missing_run_directory_is_refused_naming_the_root_and_the_op(
    tmp_path: Path,
) -> None:
    ds, first, _ = _two_sources(tmp_path)

    text = _refusal(ds.get_root(REBUILD_KIND), LMT_REBUILD_RUN_ID, _md5(first), first)

    assert "rebuild-lmt-events root" in text
    assert REBUILD_KIND in text


def test_a_source_the_run_did_not_copy_is_refused_naming_the_md5_and_the_count(
    tmp_path: Path,
) -> None:
    ds, first, second = _two_sources(tmp_path)
    _ = place_rebuilt_copy(
        ds, LMT_REBUILD_RUN_ID, first, entry_key="a", commit=LMT_REBUILD_COMMIT
    )

    text = _refusal(ds.get_root(REBUILD_KIND), LMT_REBUILD_RUN_ID, _md5(second), second)

    assert _md5(second) in text
    assert "1 cop" in text
    assert "added or changed after the run" in text


def test_an_unfinished_copy_is_refused(tmp_path: Path) -> None:
    ds, first, _ = _two_sources(tmp_path)
    _ = place_rebuilt_copy(
        ds,
        LMT_REBUILD_RUN_ID,
        first,
        entry_key="a",
        commit=LMT_REBUILD_COMMIT,
        finished=False,
    )

    text = _refusal(ds.get_root(REBUILD_KIND), LMT_REBUILD_RUN_ID, _md5(first), first)

    assert "did not finish" in text


def test_a_copy_written_after_its_rebuild_is_refused(tmp_path: Path) -> None:
    ds, first, _ = _two_sources(tmp_path)
    copy = place_rebuilt_copy(
        ds, LMT_REBUILD_RUN_ID, first, entry_key="a", commit=LMT_REBUILD_COMMIT
    )
    with copy.open("ab") as handle:
        _ = handle.write(b"x")

    text = _refusal(ds.get_root(REBUILD_KIND), LMT_REBUILD_RUN_ID, _md5(first), first)

    assert "written after its rebuild" in text


def test_a_copy_deleted_after_its_rebuild_is_refused_as_missing(
    tmp_path: Path,
) -> None:
    ds, first, _ = _two_sources(tmp_path)
    copy = place_rebuilt_copy(
        ds, LMT_REBUILD_RUN_ID, first, entry_key="a", commit=LMT_REBUILD_COMMIT
    )
    copy.unlink()

    text = _refusal(ds.get_root(REBUILD_KIND), LMT_REBUILD_RUN_ID, _md5(first), first)

    assert "is missing" in text
