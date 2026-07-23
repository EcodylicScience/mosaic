"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Callable

import numpy as np
import pytest


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
