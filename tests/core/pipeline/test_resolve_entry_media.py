"""Test resolving every camera of each entry from one read of the media index.

An entry that cannot be resolved fails alone, with the error that explains it. A
nameless entry is labeled by its first file's stem, which the scope's read cannot key
back to the entry, so it is resolved by itself. Without an entry list, the whole
scope is resolved and each entry is keyed by its label.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mosaic_media import MediaProbeError

from mosaic.core.dataset import Dataset, ResolvedScopeEntry
from mosaic.core.pipeline.consumed_camera import resolve_entry_media

from tests.helpers import MediaClip, count_index_reads, make_dataset, write_media_index


def _cameras(media: list[ResolvedScopeEntry] | Exception) -> list[str]:
    """Return the cameras of one entry's resolution, which must have succeeded."""
    assert isinstance(media, list), f"expected cameras, got {media!r}"
    return [item.camera for item in media]


def test_every_camera_of_an_entry_is_returned(two_camera_dataset: Dataset) -> None:
    resolved = resolve_entry_media(two_camera_dataset, [("A", "one")])

    assert list(resolved) == [("A", "one")]
    assert _cameras(resolved[("A", "one")]) == ["cam0", "cam1"]


def test_the_entries_resolve_in_one_read_of_the_media_index(
    three_entry_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [("B", "one"), ("A", "one"), ("A", "two")]
    reads = count_index_reads(monkeypatch)

    resolved = resolve_entry_media(three_entry_dataset, entries)

    assert reads.media_scopes == 1
    assert list(resolved) == entries
    for group, sequence in entries:
        media = resolved[(group, sequence)]
        assert isinstance(media, list)
        assert [(item.group, item.sequence) for item in media] == [(group, sequence)]


def test_an_entry_awaiting_a_transcode_fails_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rotated original needs an analysis transcode, which has not run."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(filename="a.mp4", sequence="a", rotation=90),
            MediaClip(filename="b.mp4", sequence="b"),
        ],
    )
    reads = count_index_reads(monkeypatch)

    resolved = resolve_entry_media(ds, [("", "a"), ("", "b")])

    assert reads.media_scopes == 1
    assert isinstance(resolved[("", "a")], MediaProbeError)
    assert _cameras(resolved[("", "b")]) == [""]


def test_without_a_media_index_every_entry_fails_with_the_one_error(
    dataset_without_index: Dataset,
) -> None:
    resolved = resolve_entry_media(dataset_without_index, [("", "a"), ("", "b")])

    first, second = resolved[("", "a")], resolved[("", "b")]
    assert isinstance(first, FileNotFoundError)
    assert first is second


def test_an_entry_without_indexed_media_fails_naming_it(
    three_entry_dataset: Dataset,
) -> None:
    resolved = resolve_entry_media(three_entry_dataset, [("A", "one"), ("C", "gone")])

    missing = resolved[("C", "gone")]
    assert isinstance(missing, FileNotFoundError)
    assert str(missing) == "C__gone: no media is indexed for it"
    assert _cameras(resolved[("A", "one")]) == [""]


def test_a_nameless_entry_is_keyed_as_asked_and_labeled_by_its_files_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(filename="clip.mp4", sequence=""),
            MediaClip(filename="b.mp4", sequence="b"),
        ],
    )
    reads = count_index_reads(monkeypatch)

    resolved = resolve_entry_media(ds, [("", ""), ("", "b")])

    assert reads.media_scopes == 2
    media = resolved[("", "")]
    assert isinstance(media, list)
    assert [(item.group, item.sequence) for item in media] == [("", "clip")]
    assert _cameras(resolved[("", "b")]) == [""]


def test_the_whole_scope_is_keyed_by_each_entrys_label(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(filename="clip.mp4", sequence=""),
            MediaClip(filename="a.mp4", sequence="a", rotation=90),
            MediaClip(filename="left.mp4", sequence="b", camera="left"),
            MediaClip(filename="right.mp4", sequence="b", camera="right"),
        ],
    )

    resolved = resolve_entry_media(ds, None)

    assert list(resolved) == [("", "a"), ("", "b"), ("", "clip")]
    assert isinstance(resolved[("", "a")], MediaProbeError)
    assert _cameras(resolved[("", "b")]) == ["left", "right"]
    assert _cameras(resolved[("", "clip")]) == [""]


def test_the_whole_scope_without_a_media_index_is_empty(
    dataset_without_index: Dataset,
) -> None:
    assert resolve_entry_media(dataset_without_index, None) == {}
