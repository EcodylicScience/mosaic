"""Tests for imgstore discovery in Dataset.index_media."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("imgstore")

from mosaic.core.dataset import Dataset  # noqa: E402


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
