"""What a tool is handed for an entry, and what stops it before it reads the entry.

A planner asks both before a run. Answering them does not decode media. The
answers come from the media index, the stores' metadata and the files on disk.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import MediaProbeError

from mosaic.core.dataset import Dataset, ResolvedScopeEntry
from mosaic.core.pipeline.joined_export import EntryJoinMissingError
from mosaic.core.pipeline.media_index import write_media_index_rows
from mosaic.core.pipeline.preprocess_layout import media_variant_path
from mosaic.core.pipeline.tracking_roots import ToolCodecError
from mosaic.tracking.common.scope import build_work_items
from mosaic.tracking.common.tool_input import (
    StoreExportMissingError,
    StoreExportStaleError,
    handed_codecs,
    refuse_unreadable_entry,
    resolve_tool_inputs,
)
from tests.helpers import (
    MediaClip,
    add_media_variant,
    add_transcode_derivative,
    count_index_reads,
    export_entry_store,
    index_media_sequence,
    make_dataset,
    nameless_store_dataset,
    point_at_a_store,
    printed_scope_files,
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


def test_an_export_at_another_rate_than_its_store_is_refused(
    tmp_path: Path, make_media_dataset: MakeMediaDataset
) -> None:
    """An export stamped at its chunks' 25 fps is not the 20 fps store it copies.

    The refusal names the run that exports the store again. mosaic's runner reads
    the store's chunks and is not refused.
    """
    ds = store_dataset(
        tmp_path,
        make_media_dataset,
        video_store_maker([5, 5, 2], fps=20.0, container_fps=25.0),
    )
    (entry,) = ds.resolve_media_scope(None)
    _ = export_entry_store(ds, entry.group, entry.sequence)
    refuse_unreadable_entry(ds, entry, kind="trex")

    index = ds.get_root("media") / "index.csv"
    derivatives = pd.read_csv(index, dtype=str, keep_default_na=False)
    derivatives["fps"] = "25.0"
    write_media_index_rows(index, derivatives)

    with pytest.raises(StoreExportStaleError, match="--overwrite") as refused:
        refuse_unreadable_entry(ds, entry, kind="trex")
    assert printed_scope_files(ds, str(refused.value)) == entry.resolved.paths
    refuse_unreadable_entry(ds, entry, kind="ultralytics")


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


def _nameless_store(tmp_path: Path) -> tuple[Dataset, ResolvedScopeEntry]:
    """Return a :func:`nameless_store_dataset` and its entry, as a scope labels it."""
    ds = nameless_store_dataset(tmp_path / "ds")
    (entry,) = ds.resolve_media_scope(None)
    assert (entry.group, entry.sequence) == ("", "rec")
    return ds, entry


@pytest.mark.parametrize("kind", ["trex", "ultralytics"])
def test_the_store_of_a_nameless_entry_is_handed_as_the_export_it_would_get(
    tmp_path: Path, kind: str
) -> None:
    """The store's row is found by the store's file, whatever the entry's label."""
    ds, entry = _nameless_store(tmp_path)

    assert handed_codecs(ds, entry, kind=kind) == {"av1"}


@pytest.mark.parametrize("kind", ["trex", "ultralytics"])
def test_the_export_command_of_a_nameless_entry_names_its_indexed_key(
    tmp_path: Path, kind: str
) -> None:
    ds, entry = _nameless_store(tmp_path)

    with pytest.raises(StoreExportMissingError) as refused:
        refuse_unreadable_entry(ds, entry, kind=kind)

    assert printed_scope_files(ds, str(refused.value)) == entry.resolved.paths


def test_the_export_refusal_reads_the_media_index_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store's row gives the export that it links and the entry's indexed key."""
    ds, entry = _nameless_store(tmp_path)
    reads = count_index_reads(monkeypatch)

    with pytest.raises(StoreExportMissingError):
        refuse_unreadable_entry(ds, entry, kind="trex")

    assert reads.media_rows == 1


def test_a_tool_reads_the_store_of_a_nameless_entry_through_its_export(
    tmp_path: Path,
) -> None:
    ds, entry = _nameless_store(tmp_path)
    export = add_transcode_derivative(ds, "", target="analysis")
    (item,) = build_work_items(ds, [entry], kind="trex").items

    refuse_unreadable_entry(ds, entry, kind="ultralytics")
    assert resolve_tool_inputs(ds, item, kind="trex") == (export,)


def test_a_codec_refusal_names_a_nameless_entry_by_its_indexed_key(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    """Lightning Pose does not decode AV1 in any environment, and the variant is AV1.

    The refusal names the run that makes the variant again in another codec.
    """
    ds = make_dataset(tmp_path / "ds")
    write_media_index(ds, [MediaClip(sequence="", filename="rec.mp4", frame_count=12)])
    variant = "preprocess.0.1-0123456789"
    file = media_variant_path(ds, variant, "", "rec", "")
    _ = add_media_variant(ds, variant, "rec")
    file.unlink()
    write_cfr_mp4(file)
    scope = ds.resolve_media_scope(None)
    (item,) = build_work_items(ds, scope, kind="litpose", media=variant).items

    with pytest.raises(ToolCodecError) as refused:
        _ = resolve_tool_inputs(ds, item, kind="litpose")

    (entry,) = scope
    assert printed_scope_files(ds, str(refused.value)) == entry.resolved.paths
