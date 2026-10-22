"""What a tool is handed for an entry, and what stops it before it reads the entry.

A planner asks both before a run. Answering them does not decode media. The
answers come from the media index, the stores' metadata and the files on disk.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from mosaic_media import MediaProbeError

from mosaic.core.dataset import Dataset, ResolvedScopeEntry
from mosaic.core.pipeline.joined_export import EntryJoinMissingError
from mosaic.tracking.common.tool_input import (
    StoreExportMissingError,
    handed_codecs,
    refuse_unreadable_entry,
)
from tests.helpers import (
    MediaClip,
    export_entry_store,
    index_media_sequence,
    make_dataset,
    point_at_a_store,
    store_dataset,
    stub_join,
    video_store_maker,
    write_h264_mp4,
    write_media_index,
)

WriteVideo = Callable[..., None]
MakeMediaDataset = Callable[[Path], Dataset]


def _entry(ds: Dataset, sequence: str = "sess") -> ResolvedScopeEntry:
    """The one resolved camera of *sequence*."""
    (entry,) = [
        entry for entry in ds.resolve_media_scope(None) if entry.sequence == sequence
    ]
    return entry


def _directory_store(tmp_path: Path) -> tuple[Dataset, Path]:
    """A dataset whose entry ``sess`` is an image-directory store, and the store.

    The store's chunks are not video. Its index row keeps the H.264 facts of the
    clip it replaced.
    """
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip()])
    store = point_at_a_store(ds, "sess", ds.get_root("media_raw") / "sess.store")
    return ds, store


# --- the codecs a tool is handed ----------------------------------------------


@pytest.fixture
def clips(tmp_path: Path, write_cfr_mp4: WriteVideo) -> Dataset:
    """Entry ``sess`` in an AV1 and two H.264 clips, and ``one`` in one AV1 clip."""
    ds = make_dataset(tmp_path / "ds")
    session = ds.get_root("media_raw") / "sess"
    write_cfr_mp4(session / "c0.mp4")
    write_h264_mp4(session / "c1.mp4", shade=40)
    write_h264_mp4(session / "c2.mp4", shade=80)
    index_media_sequence(ds, "sess", ["c0.mp4", "c1.mp4", "c2.mp4"])
    write_cfr_mp4(ds.get_root("media_raw") / "one" / "clip.mp4", frames=7)
    index_media_sequence(ds, "one", ["clip.mp4"])
    return ds


@pytest.mark.parametrize(
    ("kind", "session", "one"),
    [
        ("trex", {"h264"}, {"av1"}),
        ("litpose", {"h264"}, {"av1"}),
        ("ultralytics", {"av1", "h264"}, {"av1"}),
        ("infer-localizer", set[str](), set[str]()),
    ],
)
def test_clips_are_handed_as_themselves_or_as_their_join(
    clips: Dataset, kind: str, session: set[str], one: set[str]
) -> None:
    """A one-file tool opens one clip, or the join of several in the codec most share.

    mosaic's runner opens every clip, and an in-process reader does not open a file.
    """
    assert handed_codecs(clips, _entry(clips, "sess"), kind=kind) == session
    assert handed_codecs(clips, _entry(clips, "one"), kind=kind) == one


def test_a_video_store_is_read_by_chunk_or_through_its_export(
    tmp_path: Path, make_media_dataset: MakeMediaDataset, write_cfr_mp4: WriteVideo
) -> None:
    """A one-file tool reads a video store's export, and mosaic's runner its chunks.

    The runner reads the chunks even beside an export. The export is the linked
    file while it exists, and otherwise the one an export run would write, which
    copies these H.264 chunks.
    """
    ds = store_dataset(tmp_path, make_media_dataset, video_store_maker([5, 5, 2]))
    (entry,) = ds.resolve_media_scope(None)

    assert handed_codecs(ds, entry, kind="ultralytics") == {"h264"}
    assert handed_codecs(ds, entry, kind="trex") == {"h264"}

    export = export_entry_store(ds, entry.group, entry.sequence)
    export.unlink()
    write_cfr_mp4(export)
    assert handed_codecs(ds, entry, kind="trex") == {"av1"}, "the linked file"
    assert handed_codecs(ds, entry, kind="ultralytics") == {"h264"}, "the chunks"

    export.unlink()
    assert handed_codecs(ds, entry, kind="trex") == {"h264"}, "the export to write"


@pytest.mark.parametrize(
    ("kind", "codecs"),
    [("trex", {"av1"}), ("ultralytics", {"av1"}), ("infer-localizer", set[str]())],
)
def test_a_store_without_video_chunks_is_handed_as_the_export_it_would_get(
    tmp_path: Path, kind: str, codecs: set[str]
) -> None:
    """An export of it is encoded in AV1.

    The H.264 that the store's index row records is not the codec of a file a tool
    opens, and is not read.
    """
    ds, _store = _directory_store(tmp_path)

    assert handed_codecs(ds, _entry(ds), kind=kind) == codecs


@pytest.mark.parametrize("kind", ["trex", "ultralytics"])
def test_a_store_whose_metadata_does_not_read_is_refused(
    tmp_path: Path, kind: str
) -> None:
    ds, store = _directory_store(tmp_path)
    _ = (store / "metadata.yaml").write_text(
        "__store: {class: DirectoryImgStore, format: npy, imgshape: [48]}\n"
    )

    with pytest.raises(MediaProbeError, match="imgshape"):
        _ = handed_codecs(ds, _entry(ds), kind=kind)


# --- what stops a tool before it reads an entry -------------------------------


def test_an_entry_is_refused_as_the_run_of_its_tool_refuses_it(tmp_path: Path) -> None:
    """Two clips without a join stop a one-file tool, and mosaic's runner reads them."""
    ds = make_dataset(tmp_path / "ds")
    write_media_index(
        ds,
        [
            MediaClip(
                filename=f"c{order}.mp4", video_order=order, video_uuid=f"uid-{order}"
            )
            for order in range(2)
        ],
    )
    entry = _entry(ds)

    with pytest.raises(EntryJoinMissingError) as refused:
        refuse_unreadable_entry(ds, entry, kind="trex")
    assert refused.value.cause == "no_join"
    refuse_unreadable_entry(ds, entry, kind="ultralytics")

    _ = stub_join(ds, ["uid-0", "uid-1"])
    refuse_unreadable_entry(ds, entry, kind="trex")


@pytest.mark.parametrize("kind", ["trex", "ultralytics"])
@pytest.mark.parametrize("query", [handed_codecs, refuse_unreadable_entry])
def test_the_store_of_a_nameless_entry_matches_no_index_row(
    tmp_path: Path, kind: str, query: Callable[..., object]
) -> None:
    """The export of the store is looked up by the entry's label.

    The label does not match an index row. The error differs from that of a
    missing export, which ``export-store`` builds.
    """
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(sequence="", filename="rec.mp4")])
    _ = point_at_a_store(ds, "", ds.get_root("media_raw") / "rec.store")
    (entry,) = ds.resolve_media_scope(None)

    with pytest.raises(FileNotFoundError, match="No media file found") as raised:
        _ = query(ds, entry, kind=kind)
    assert not isinstance(raised.value, StoreExportMissingError)
