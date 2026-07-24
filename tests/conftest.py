"""Shared pytest fixtures."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest

from mosaic.core.dataset import Dataset


@pytest.fixture
def read_index_header() -> Callable[[Path], list[str]]:
    """Factory reading an index CSV's header line, before any schema widening.

    The only place a file's real column set survives: every reader in the
    toolkit widens to ``MEDIA_INDEX_COLUMNS``, so an absent column and an empty
    one are indistinguishable afterwards. Returns a callable
    ``(index_path) -> [column]``.
    """

    def _read(index_path: Path) -> list[str]:
        no_header: list[str] = []
        with index_path.open(newline="") as handle:
            return next(csv.reader(handle), no_header)

    return _read


@pytest.fixture
def write_cfr_mp4() -> Callable[..., None]:
    """Factory writing a small constant-frame-rate mp4 (parent dirs created).

    The shape every media test needs: a real file ffprobe can measure, cheap
    enough to write per test. Returns a callable ``(path, frames=, size=)``.
    """

    def _write(path: Path, frames: int = 6, size: tuple[int, int] = (64, 48)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # VideoWriter.fourcc rather than the module-level VideoWriter_fourcc
        # alias: the two return the same code, but only the classmethod is typed.
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, size)
        for _ in range(frames):
            writer.write(np.zeros((size[1], size[0], 3), np.uint8))
        writer.release()

    return _write


@pytest.fixture
def make_media_dataset() -> Callable[[Path], Dataset]:
    """Factory building a saved Dataset with ``media_raw``, ``media`` and
    ``tracks`` roots.

    The manifest is written to disk, not merely named: ``base_dir`` treats a
    ``manifest_path`` that is not an existing file as the base directory itself
    and creates it, which would make every root-relative ``abs_path`` resolve one
    level too deep. The ``tracks`` root is present because ``index_media`` reads
    its index to derive each media file's ``(group, sequence)``, so a transcode
    test that indexes real media needs it. Returns a callable
    ``(base_dir) -> Dataset``.
    """

    def _make(base: Path) -> Dataset:
        ds = Dataset(
            manifest_path=base / "dataset.yaml",
            roots={
                "media_raw": str(base / "media_raw"),
                "media": str(base / "media"),
                "tracks": str(base / "tracks"),
            },
        )
        ds.ensure_roots()
        ds.save()
        return ds

    return _make


@pytest.fixture
def make_imgstore(tmp_path: Path) -> Callable[..., tuple[Path, list[np.ndarray]]]:
    """Factory writing a synthetic imgstore for tests (no Motif required).

    Each frame is tagged uniquely in its first pixel (``frame[0, 0, 0] == i``)
    so read-back order/identity can be asserted. Defaults to the ``npy``
    (DirectoryImgStore) format, which is lossless and needs no codec/ffmpeg.
    ``extra_metadata`` writes document-root keys into ``metadata.yaml`` (e.g.
    Motif ``camera_serial`` / ``synchronizationuuid`` / ``synchronization``) so a
    multi-camera recording can be simulated.

    Returns a callable ``(name=, nframes=, fmt=, shape=, dtype=, chunksize=,
    parent=, extra_metadata=) -> (store_dir, frames)``.
    """
    imgstore = pytest.importorskip("imgstore")

    def _make(
        name: str = "store",
        nframes: int = 12,
        fmt: str = "npy",
        shape: tuple[int, ...] = (48, 64, 3),
        dtype: type = np.uint8,
        chunksize: int = 5,
        parent: Path | None = None,
        fps: float = 30.0,
        extra_metadata: Mapping[str, object] | None = None,
    ) -> tuple[Path, list[np.ndarray]]:
        base = parent if parent is not None else tmp_path
        base.mkdir(parents=True, exist_ok=True)
        dest = base / name
        # imgstore merges a passed metadata dict at the document root (its own
        # block lives under __store), so extra_metadata lands where is_imgstore /
        # imgstore_store_identity read Motif keys. Pass it only when set: a None
        # metadata would blow up the store's __store merge.
        extra = {"metadata": dict(extra_metadata)} if extra_metadata else {}
        store = imgstore.new_for_format(
            fmt,
            path=str(dest),
            mode="w",
            imgshape=shape,
            imgdtype=dtype,
            chunksize=chunksize,
            **extra,
        )
        frames: list[np.ndarray] = []
        for i in range(nframes):
            img = np.zeros(shape, dtype=dtype)
            img.reshape(-1)[0] = i % 256  # unique per-frame tag at [0, 0(, 0)]
            frames.append(img)
            store.add_image(img, frame_number=i, frame_time=float(i) / fps)
        store.close()
        return dest, frames

    return _make
