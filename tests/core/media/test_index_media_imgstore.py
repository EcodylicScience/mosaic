"""Tests for imgstore discovery in Dataset.index_media."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline.media_index import (
    MediaIndexScope,
    StoreGroupingError,
    media_members_from_rows,
    read_media_index,
    refuse_merged_store_recordings,
)
from tests.helpers import MOTIF_SYNC_UUID, MakeStore, make_dataset, write_mpeg4_mp4

pytest.importorskip("imgstore")

from mosaic_media import VIDEO_EXTENSIONS, MediaProbeError  # noqa: E402

from mosaic.core.media.imgstore_io import imgstore_store_identity  # noqa: E402

# No ``media_raw``: ``index_media`` resolves through ``resolve_media_root``, so a
# dataset declaring only ``media`` writes the originals index to
# ``media/index.csv``, which is where these tests read it back from. ``frames``
# is where ``extract_frames`` lands its PNGs.
_ROOTS = ("media", "tracks", "frames")


def _camera_meta(serial: str, uuid: str) -> dict[str, object]:
    """Motif document-root metadata for one camera of a synced recording."""
    return {
        "camera_serial": serial,
        "synchronizationuuid": uuid,
        "synchronization": "framenumber",
    }


# Probes the store's video chunk to measure it, which shells out.
@pytest.mark.media
def test_index_media_discovers_store_and_excludes_chunks(tmp_path, make_imgstore):
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    store_dir, _ = make_imgstore(name="rec1", nframes=12, parent=search)
    write_mpeg4_mp4(search / "plain.mp4")

    # Include .npy so the store's internal chunk files would be picked up by the
    # glob unless the chunk-exclusion guard works.
    out_csv = ds.index_media([search], extensions=(".mp4", ".npy"))
    df = pd.read_csv(out_csv)

    # Exactly one imgstore entry, pointing at the store directory.
    store_rows = df[df["media_type"] == "imgstore"]
    assert len(store_rows) == 1
    assert Path(store_rows.iloc[0]["abs_path"]).resolve() == store_dir.resolve()

    # The plain mp4 is a normal video entry.
    video_rows = df[df["media_type"] == "video"]
    assert len(video_rows) == 1
    assert Path(video_rows.iloc[0]["abs_path"]).name == "plain.mp4"

    # No row lives *inside* the store directory (chunks excluded).
    store_resolved = store_dir.resolve()
    for ap in df["abs_path"]:
        p = Path(ap).resolve()
        assert store_resolved not in p.parents

    # The store row carries probed metadata.
    assert int(store_rows.iloc[0]["width"]) == 64
    assert int(store_rows.iloc[0]["height"]) == 48


def test_resolve_media_returns_store_dir(tmp_path, make_imgstore):
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    store_dir, _ = make_imgstore(name="seqA", nframes=8, parent=search)
    ds.index_media([search])

    paths = ds.resolve_media("", "seqA").paths
    assert len(paths) == 1
    assert paths[0].resolve() == store_dir.resolve()


@pytest.mark.media
@pytest.mark.slow
def test_index_media_excludes_every_supported_extension_inside_a_store(
    tmp_path: Path,
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
) -> None:
    """A store's contents are never indexed, whatever they are named.

    The exclusion is a parent-path test (``Dataset._probe_dir_rows``) applied
    before the extension filter, so it is extension-agnostic by construction.
    Asserting it over one suffix understates it, and no single suffix can be the
    realistic one: imgstore writes video chunks as ``.avi`` (mjpeg) or ``.mp4``
    (avc1), and which of those a machine can produce depends on the codecs its
    OpenCV build can reach. So the store is written in the format that needs no
    H.264 encoder, and a decoy of every extension mosaic-media recognizes is
    placed beside the real chunks.

    A decoy is empty on purpose: the skip happens before the probe, so a file
    that reaches ffprobe would prove the exclusion had already failed.
    """
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    store_dir, _ = make_imgstore(
        name="vid_store", nframes=12, fmt="mjpeg/avi", chunksize=5, parent=search
    )
    assert list(store_dir.glob("*.avi")), "expected video chunks inside the store"

    for extension in sorted(VIDEO_EXTENSIONS):
        (store_dir / f"decoy{extension}").write_bytes(b"")
    write_mpeg4_mp4(search / "plain.mp4")

    out_csv = ds.index_media([search], extensions=tuple(sorted(VIDEO_EXTENSIONS)))
    df = pd.read_csv(out_csv)

    assert (df["media_type"] == "imgstore").sum() == 1
    assert (df["media_type"] == "video").sum() == 1  # only the plain mp4 beside it
    store_resolved = store_dir.resolve()
    for ap in df["abs_path"]:
        assert store_resolved not in Path(ap).resolve().parents


def test_index_media_synced_cameras_collapse_to_one_sequence(tmp_path, make_imgstore):
    """Two Motif stores sharing a synchronizationuuid are one sequence, two cameras."""
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    make_imgstore(
        name="rec.CAMA",
        parent=search,
        extra_metadata=_camera_meta("CAMA", MOTIF_SYNC_UUID),
    )
    make_imgstore(
        name="rec.CAMB",
        parent=search,
        extra_metadata=_camera_meta("CAMB", MOTIF_SYNC_UUID),
    )

    out_csv = ds.index_media([search])
    df = pd.read_csv(out_csv, keep_default_na=False)
    store_rows = df[df["media_type"] == "imgstore"]

    assert len(store_rows) == 2
    # One sequence (serial stripped from the dir name), two camera rows.
    assert set(store_rows["sequence"]) == {"rec"}
    assert set(store_rows["camera"]) == {"CAMA", "CAMB"}
    # Shared, populated provenance.
    assert set(store_rows["sync_uuid"]) == {MOTIF_SYNC_UUID}
    # Each camera is numbered from 0 -- never a cross-camera concatenation.
    assert set(store_rows["video_order"].astype(int)) == {0}


def test_index_media_gives_each_store_its_own_mint(tmp_path, make_imgstore):
    """Each camera's row carries its own ``__store.uuid``, not the shared sync id.

    The two are deliberately different things and the whole reason open item O5
    could be answered yes: ``sync_uuid`` is shared across a recording's cameras
    and safe to collide, so it can never name a single store, while
    ``__store.uuid`` differs per store and can. Asserted against the metadata
    rather than a literal, because the uuid is minted at store creation and is
    different on every run -- which is itself the limitation
    ``STORE_IDENTITY_SCHEME`` exists to advertise.
    """
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    dirs = {
        serial: make_imgstore(
            name=f"rec.{serial}",
            parent=search,
            extra_metadata=_camera_meta(serial, MOTIF_SYNC_UUID),
        )[0]
        for serial in ("CAMA", "CAMB")
    }

    out_csv = ds.index_media([search])
    df = pd.read_csv(out_csv, keep_default_na=False)
    store_rows = df[df["media_type"] == "imgstore"]

    minted = {
        serial: imgstore_store_identity(path).store_uuid
        for serial, path in dirs.items()
    }
    assert all(minted.values()), "every imgstore carries a __store.uuid"
    assert minted["CAMA"] != minted["CAMB"], "a mint names one store, not a recording"
    assert dict(zip(store_rows["camera"], store_rows["video_uuid"])) == minted
    # A store has no elementary stream, so nothing to hash for duplicates.
    assert set(store_rows["content_digest"]) == {""}
    # And the value says which regime produced it, so a reader can tell a
    # declared identity from a measured one without knowing what a store is.
    schemes = {
        json.loads(cell)["identity_scheme"] for cell in store_rows["media_facts"]
    }
    assert schemes == {"imgstore/1"}


def test_index_media_unsynced_stores_stay_separate_sequences(tmp_path, make_imgstore):
    """Non-Motif stores (no sync metadata) each stay their own sequence."""
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    make_imgstore(name="alpha", parent=search)
    make_imgstore(name="beta", parent=search)

    out_csv = ds.index_media([search])
    df = pd.read_csv(out_csv, keep_default_na=False)
    store_rows = df[df["media_type"] == "imgstore"]

    assert set(store_rows["sequence"]) == {"alpha", "beta"}
    assert set(store_rows["camera"]) == {""}
    assert set(store_rows["sync_uuid"]) == {""}


def test_index_media_dotted_name_no_sync_keeps_full_name(tmp_path, make_imgstore):
    """A dotted store name with no sync metadata keys on its full name.

    Regression for the ``Path.stem`` collapse: ``session01.cam`` must index as
    sequence ``session01.cam``, not ``session01`` (which would silently merge
    distinct recordings).
    """
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    make_imgstore(name="session01.cam", parent=search)

    out_csv = ds.index_media([search])
    df = pd.read_csv(out_csv, keep_default_na=False)
    store_rows = df[df["media_type"] == "imgstore"]

    assert len(store_rows) == 1
    assert store_rows.iloc[0]["sequence"] == "session01.cam"
    assert store_rows.iloc[0]["camera"] == ""


def test_resolve_media_multicamera_requires_camera(tmp_path, make_imgstore):
    """A camera-less resolve over a synced recording fails loud; camera= selects one."""
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    for serial in ("CAMA", "CAMB"):
        make_imgstore(
            name=f"rec.{serial}",
            parent=search,
            extra_metadata=_camera_meta(serial, MOTIF_SYNC_UUID),
        )
    ds.index_media([search])

    # Concatenating two cameras would fabricate a timeline, so resolve raises.
    with pytest.raises(MediaProbeError):
        ds.resolve_media("", "rec")

    one = ds.resolve_media("", "rec", camera="CAMA")
    assert len(one.paths) == 1
    assert one.paths[0].name == "rec.CAMA"

    # The scope enumerates one entry per camera, each its own single-store media.
    scope = ds.resolve_media_scope(None)
    assert sorted(e.camera for e in scope) == ["CAMA", "CAMB"]
    assert all(len(e.resolved.paths) == 1 for e in scope)


def test_extract_frames_runs_per_camera(tmp_path, make_imgstore):
    """extract-frames processes each camera into its own subdir, no concatenation."""
    from mosaic.tracking import extract_frames
    from mosaic.tracking.frame_extraction.dataset_runs import get_frame_paths

    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    (tmp_path / "frames").mkdir(exist_ok=True)
    search = tmp_path / "raw"
    for serial in ("CAMA", "CAMB"):
        make_imgstore(
            name=f"rec.{serial}",
            nframes=12,
            parent=search,
            extra_metadata=_camera_meta(serial, MOTIF_SYNC_UUID),
        )
    ds.index_media([search])

    extract_frames(ds, n_frames=4, method="uniform")

    # One run dir; the sequence dir holds a subdir per camera.
    run_dirs = [d for d in (tmp_path / "frames" / "uniform").iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    cam_dirs = sorted(d.name for d in (run_dirs[0] / "rec").iterdir() if d.is_dir())
    assert cam_dirs == ["CAMA", "CAMB"]

    # The frames index records both cameras of the one sequence.
    fidx = pd.read_csv(run_dirs[0].parent / "index.csv", keep_default_na=False)
    rec = fidx[fidx["sequence"] == "rec"]
    assert sorted(rec["camera"]) == ["CAMA", "CAMB"]

    # The read path descends the camera subdirs and returns both cameras' frames.
    pngs = get_frame_paths(ds, method="uniform", group="", sequence="rec")
    assert len(pngs) == 8  # 4 per camera, no concatenated timeline


def _store_rows(out_csv: Path) -> pd.DataFrame:
    """The imgstore rows of the media index at *out_csv*, read as text."""
    df = pd.read_csv(out_csv, keep_default_na=False, dtype=str)
    return df[df["media_type"] == "imgstore"]


def _unsynced_meta(serial: str) -> dict[str, object]:
    """Motif document-root metadata for a store recorded by one camera alone."""
    return {"camera_serial": serial, "synchronization": "none"}


def test_index_media_records_each_store_camera_serial(
    tmp_path: Path, make_imgstore: MakeStore
) -> None:
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    for serial in ("CAMA", "CAMB"):
        _ = make_imgstore(
            name=f"rec.{serial}",
            parent=search,
            extra_metadata=_camera_meta(serial, MOTIF_SYNC_UUID),
        )
    _ = make_imgstore(name="lone", parent=search, extra_metadata=_unsynced_meta("C1"))
    _ = make_imgstore(name="plain", parent=search)

    rows = _store_rows(ds.index_media([search]))

    serials = dict(zip(rows["name"], rows["camera_serial"], strict=True))
    assert serials == {
        "rec.CAMA": "CAMA",
        "rec.CAMB": "CAMB",
        "lone": "C1",
        "plain": "",
    }


def test_write_media_index_records_the_camera_serial(
    tmp_path: Path, make_imgstore: MakeStore
) -> None:
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    scope_dir = tmp_path / "raw" / "tank1"
    _ = make_imgstore(name="rec", parent=scope_dir, extra_metadata=_unsynced_meta("C1"))

    result = ds.write_media_index(
        [MediaIndexScope(directory=scope_dir, group="g", sequence="tank1")]
    )

    rows = _store_rows(result.index_path)
    assert list(rows["camera_serial"]) == ["C1"]


def test_the_camera_serial_leaves_the_media_composition_unchanged() -> None:
    row: dict[str, object] = {
        "group": "g",
        "sequence": "s",
        "camera": "C1",
        "video_order": "0",
        "video_uuid": "a" * 32,
    }

    without = media_members_from_rows([row])
    with_serial = media_members_from_rows([{**row, "camera_serial": "C1"}])

    assert with_serial == without


def test_index_media_refuses_a_sync_set_with_divergent_base_names(
    tmp_path: Path, make_imgstore: MakeStore
) -> None:
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    _ = make_imgstore(
        name="recA.CAMA",
        parent=search,
        extra_metadata=_camera_meta("CAMA", MOTIF_SYNC_UUID),
    )
    _ = make_imgstore(
        name="recB.CAMB",
        parent=search,
        extra_metadata=_camera_meta("CAMB", MOTIF_SYNC_UUID),
    )

    with pytest.raises(StoreGroupingError, match="recA.*recB"):
        _ = ds.index_media([search])
    assert read_media_index(tmp_path / "media" / "index.csv") == []


def test_index_media_refuses_a_sync_set_with_an_empty_serial(
    tmp_path: Path, make_imgstore: MakeStore
) -> None:
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    _ = make_imgstore(
        name="rec.CAMA",
        parent=search,
        extra_metadata=_camera_meta("CAMA", MOTIF_SYNC_UUID),
    )
    _ = make_imgstore(
        name="rec", parent=search, extra_metadata=_camera_meta("", MOTIF_SYNC_UUID)
    )

    with pytest.raises(StoreGroupingError, match="camera_serial"):
        _ = ds.index_media([search])
    assert read_media_index(tmp_path / "media" / "index.csv") == []


def test_index_media_indexes_a_lone_synced_store_without_a_serial(
    tmp_path: Path, make_imgstore: MakeStore
) -> None:
    """A sync set of one store needs no serial to tell its cameras apart."""
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    _ = make_imgstore(
        name="rec", parent=search, extra_metadata=_camera_meta("", MOTIF_SYNC_UUID)
    )

    rows = _store_rows(ds.index_media([search]))

    assert list(rows["sequence"]) == ["rec"]
    assert list(rows["camera"]) == [""]
    assert list(rows["camera_serial"]) == [""]


@pytest.mark.parametrize("serials", [("C1", "C2"), ("C1", "C1"), ("", "")])
def test_index_media_refuses_unsynced_stores_that_derive_one_entry(
    tmp_path: Path, make_imgstore: MakeStore, serials: tuple[str, str]
) -> None:
    """Two stores named by their start time alone stay two entries.

    A scan's entry is derived from the name, so it refuses the stores of one
    camera too: a copy of one store derives the same entry as the original.
    """
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    for camera, serial in zip(("cam1", "cam2"), serials, strict=True):
        _ = make_imgstore(
            name="20210219_112253",
            parent=search / camera,
            extra_metadata=_unsynced_meta(serial) if serial else None,
        )

    with pytest.raises(StoreGroupingError, match="20210219_112253"):
        _ = ds.index_media([search])
    assert read_media_index(tmp_path / "media" / "index.csv") == []


def test_index_media_refuses_a_store_that_joins_the_entry_of_an_earlier_scan(
    tmp_path: Path, make_imgstore: MakeStore
) -> None:
    """A scan of each camera's directory compares the entry the first scan wrote."""
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    for camera in ("cam1", "cam2"):
        _ = make_imgstore(
            name="20210219_112253",
            parent=search / camera,
            extra_metadata=_unsynced_meta(camera.upper()),
        )
    index_path = ds.index_media([search / "cam1"])
    first = read_media_index(index_path)

    with pytest.raises(StoreGroupingError, match="cam1.*cam2"):
        _ = ds.index_media([search / "cam2"])
    assert read_media_index(index_path) == first


def test_index_media_joins_a_sync_set_scanned_one_camera_at_a_time(
    tmp_path: Path, make_imgstore: MakeStore
) -> None:
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    for camera, serial in (("cam1", "CAMA"), ("cam2", "CAMB")):
        _ = make_imgstore(
            name=f"rec.{serial}",
            parent=search / camera,
            extra_metadata=_camera_meta(serial, MOTIF_SYNC_UUID),
        )
    _ = ds.index_media([search / "cam1"])

    rows = _store_rows(ds.index_media([search / "cam2"]))

    assert list(rows["sequence"]) == ["rec", "rec"]
    assert sorted(rows["camera"]) == ["CAMA", "CAMB"]


@pytest.mark.parametrize("serials", [("", ""), ("CAMA", ""), ("", "CAMB")])
def test_index_media_refuses_a_serialless_camera_across_scans(
    tmp_path: Path, make_imgstore: MakeStore, serials: tuple[str, str]
) -> None:
    """A store without a serial cannot join a sync set an earlier scan wrote."""
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    for camera, serial in zip(("cam1", "cam2"), serials, strict=True):
        _ = make_imgstore(
            name=f"rec.{serial}" if serial else "rec",
            parent=search / camera,
            extra_metadata=_camera_meta(serial, MOTIF_SYNC_UUID),
        )
    index_path = ds.index_media([search / "cam1"])
    first = read_media_index(index_path)

    with pytest.raises(StoreGroupingError, match="camera_serial"):
        _ = ds.index_media([search / "cam2"])
    assert read_media_index(index_path) == first


def test_merged_store_refusal_names_a_kept_camera_by_its_camera_column() -> None:
    """A kept row of an index older than camera_serial is named by its camera."""

    def store(path: str, camera: str, serial: str | None) -> dict[str, object]:
        row: dict[str, object] = {
            "media_type": "imgstore",
            "assignment_source": "scan-imgstore-sync",
            "group": "",
            "sequence": "rec",
            "camera": camera,
            "sync_uuid": MOTIF_SYNC_UUID,
            "abs_path": path,
        }
        if serial is not None:
            row["camera_serial"] = serial
        return row

    refuse_merged_store_recordings(
        [store("cam2/rec.CAMB", "CAMB", "CAMB")],
        beside=[store("cam1/rec.CAMA", "CAMA", None)],
    )
    with pytest.raises(StoreGroupingError, match="camera_serial"):
        refuse_merged_store_recordings(
            [store("cam2/rec.CAMB", "CAMB", "CAMB")],
            beside=[store("cam1/rec", "", None)],
        )


def test_merged_store_refusal_compares_kept_rows_only_on_written_entries() -> None:
    """A kept entry that this write does not touch is left as it is."""

    def store(path: str, sequence: str) -> dict[str, object]:
        return {
            "media_type": "imgstore",
            "assignment_source": "scan-imgstore-sync",
            "group": "",
            "sequence": sequence,
            "sync_uuid": "",
            "abs_path": path,
        }

    kept = [store("cam1/a", "a"), store("cam2/a", "a")]

    refuse_merged_store_recordings([store("cam1/b", "b")], beside=kept)
    with pytest.raises(StoreGroupingError, match="cam1/a.*cam2/a.*cam3/a"):
        refuse_merged_store_recordings([store("cam3/a", "a")], beside=kept)


def test_index_media_keeps_the_assigned_entries_of_stores_that_share_a_name(
    tmp_path: Path, make_imgstore: MakeStore
) -> None:
    """A scan restores assigned identity before it compares derived entries."""
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    search = tmp_path / "raw"
    scopes: list[MediaIndexScope] = []
    for camera in ("cam1", "cam2"):
        _ = make_imgstore(
            name="20210219_112253",
            parent=search / camera,
            extra_metadata=_unsynced_meta(camera.upper()),
        )
        scopes.append(
            MediaIndexScope(directory=search / camera, group=camera, sequence="rec")
        )
    _ = ds.write_media_index(scopes)

    rows = _store_rows(ds.index_media([search]))

    assert sorted(zip(rows["group"], rows["sequence"], strict=True)) == [
        ("cam1", "rec"),
        ("cam2", "rec"),
    ]


def test_write_media_index_refuses_unsynced_cameras_under_one_scope(
    tmp_path: Path, make_imgstore: MakeStore
) -> None:
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    scope_dir = tmp_path / "raw" / "upload"
    for serial in ("C1", "C2"):
        _ = make_imgstore(
            name=f"rec_{serial}",
            parent=scope_dir,
            extra_metadata=_unsynced_meta(serial),
        )

    with pytest.raises(StoreGroupingError, match="C1.*C2"):
        _ = ds.write_media_index(
            [MediaIndexScope(directory=scope_dir, group="", sequence="rec")]
        )
    assert read_media_index(tmp_path / "media" / "index.csv") == []


def test_write_media_index_joins_one_camera_stores_as_clips(
    tmp_path: Path, make_imgstore: MakeStore
) -> None:
    ds = make_dataset(tmp_path, roots=_ROOTS, save=False)
    scope_dir = tmp_path / "raw" / "upload"
    for name in ("part1", "part2"):
        _ = make_imgstore(
            name=name, parent=scope_dir, extra_metadata=_unsynced_meta("C1")
        )

    result = ds.write_media_index(
        [MediaIndexScope(directory=scope_dir, group="", sequence="rec")]
    )

    rows = _store_rows(result.index_path).sort_values("name")
    assert set(rows["sequence"]) == {"rec"}
    assert set(rows["camera"]) == {"C1"}
    assert list(rows["video_order"]) == ["0", "1"]
