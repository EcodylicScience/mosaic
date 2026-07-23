"""Tests for imgstore discovery in Dataset.index_media."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("imgstore")

from mosaic_media import MediaProbeError  # noqa: E402

from mosaic.core.dataset import Dataset  # noqa: E402

_SYNC_UUID = "f064059f9ea046429f227bc7addab1eb"


def _camera_meta(serial: str, uuid: str) -> dict[str, object]:
    """Motif document-root metadata for one camera of a synced recording."""
    return {
        "camera_serial": serial,
        "synchronizationuuid": uuid,
        "synchronization": "framenumber",
    }


def _make_dataset(tmp_path: Path) -> Dataset:
    for sub in ("media", "tracks", "frames"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media": str(tmp_path / "media"),
            "tracks": str(tmp_path / "tracks"),
            "frames": str(tmp_path / "frames"),
        },
    )


def _write_plain_mp4(path: Path, nframes: int = 6) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(nframes):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def test_index_media_discovers_store_and_excludes_chunks(tmp_path, make_imgstore):
    ds = _make_dataset(tmp_path)
    search = tmp_path / "raw"
    store_dir, _ = make_imgstore(name="rec1", nframes=12, parent=search)
    _write_plain_mp4(search / "plain.mp4")

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
    ds = _make_dataset(tmp_path)
    search = tmp_path / "raw"
    store_dir, _ = make_imgstore(name="seqA", nframes=8, parent=search)
    ds.index_media([search])

    paths = ds.resolve_media("", "seqA").paths
    assert len(paths) == 1
    assert paths[0].resolve() == store_dir.resolve()


@pytest.mark.slow
def test_index_media_excludes_mp4_chunks_from_video_store(tmp_path, make_imgstore):
    """A VideoImgStore has real .mp4 chunks inside it; they must not be indexed."""
    ds = _make_dataset(tmp_path)
    search = tmp_path / "raw"
    store_dir, _ = make_imgstore(
        name="vid_store", nframes=12, fmt="avc1/mp4", chunksize=5, parent=search
    )
    # Sanity: the store really does contain .mp4 chunk files.
    assert list(store_dir.glob("*.mp4")), "expected mp4 chunks inside the store"
    _write_plain_mp4(search / "plain.mp4")

    out_csv = ds.index_media([search], extensions=(".mp4",))
    df = pd.read_csv(out_csv)

    assert (df["media_type"] == "imgstore").sum() == 1
    assert (df["media_type"] == "video").sum() == 1  # only the plain mp4
    store_resolved = store_dir.resolve()
    for ap in df["abs_path"]:
        assert store_resolved not in Path(ap).resolve().parents


def test_index_media_synced_cameras_collapse_to_one_sequence(tmp_path, make_imgstore):
    """Two Motif stores sharing a synchronizationuuid are one sequence, two cameras."""
    ds = _make_dataset(tmp_path)
    search = tmp_path / "raw"
    make_imgstore(
        name="rec.CAMA", parent=search, extra_metadata=_camera_meta("CAMA", _SYNC_UUID)
    )
    make_imgstore(
        name="rec.CAMB", parent=search, extra_metadata=_camera_meta("CAMB", _SYNC_UUID)
    )

    out_csv = ds.index_media([search])
    df = pd.read_csv(out_csv, keep_default_na=False)
    store_rows = df[df["media_type"] == "imgstore"]

    assert len(store_rows) == 2
    # One sequence (serial stripped from the dir name), two camera rows.
    assert set(store_rows["sequence"]) == {"rec"}
    assert set(store_rows["camera"]) == {"CAMA", "CAMB"}
    # Shared, populated provenance.
    assert set(store_rows["sync_uuid"]) == {_SYNC_UUID}
    # Each camera is numbered from 0 -- never a cross-camera concatenation.
    assert set(store_rows["video_order"].astype(int)) == {0}


def test_index_media_unsynced_stores_stay_separate_sequences(tmp_path, make_imgstore):
    """Non-Motif stores (no sync metadata) each stay their own sequence."""
    ds = _make_dataset(tmp_path)
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
    ds = _make_dataset(tmp_path)
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
    ds = _make_dataset(tmp_path)
    search = tmp_path / "raw"
    for serial in ("CAMA", "CAMB"):
        make_imgstore(
            name=f"rec.{serial}",
            parent=search,
            extra_metadata=_camera_meta(serial, _SYNC_UUID),
        )
    ds.index_media([search])

    # Concatenating two cameras would fabricate a timeline, so resolve raises.
    with pytest.raises(MediaProbeError):
        ds.resolve_media("", "rec")

    one = ds.resolve_media("", "rec", camera="CAMA")
    assert len(one.paths) == 1
    assert one.paths[0].name == "rec.CAMA"

    # The scope enumerates one entry per camera, each its own single-store media.
    scope = ds.resolve_media_scope(None, None)
    assert sorted(e.camera for e in scope) == ["CAMA", "CAMB"]
    assert all(len(e.resolved.paths) == 1 for e in scope)


def test_extract_frames_runs_per_camera(tmp_path, make_imgstore):
    """extract-frames processes each camera into its own subdir, no concatenation."""
    from mosaic.tracking import extract_frames
    from mosaic.tracking.frame_extraction.dataset_runs import get_frame_paths

    ds = _make_dataset(tmp_path)
    (tmp_path / "frames").mkdir(exist_ok=True)
    search = tmp_path / "raw"
    for serial in ("CAMA", "CAMB"):
        make_imgstore(
            name=f"rec.{serial}",
            nframes=12,
            parent=search,
            extra_metadata=_camera_meta(serial, _SYNC_UUID),
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
