"""Shared pytest fixtures."""

from __future__ import annotations

import csv
import importlib.util
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest

# Modules the CI workflow installs through extras (`.[wavelets,imgstore]`).
# `imgstore` gates 35 tests behind ``pytest.importorskip``, so its absence
# presents as a skip rather than a failure -- a green CI that ran less than the
# workflow installed for. That is not hypothetical: the test step used to invoke
# `uv run pytest`, which re-synced the environment from `uv.lock` and pruned
# both extras before the first test ran.
CI_REQUIRED_MODULES = ("imgstore", "pywt")


def pytest_configure() -> None:
    """Under CI, a missing optional dependency is an error rather than a skip.

    Local runs are unaffected: a developer without ``imgstore`` installed still
    gets skips, which is the point of ``importorskip``. Only CI, which installs
    them explicitly, treats their absence as a broken environment.
    """
    if not os.environ.get("CI"):
        return
    missing = [
        name for name in CI_REQUIRED_MODULES if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise pytest.UsageError(
            f"CI installs {', '.join(missing)} through extras, but they are not "
            "importable. The suite would skip silently instead of failing. Check "
            "that the test step does not re-sync the environment away "
            "(uv run --no-sync)."
        )


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


def add_track_sequences(dataset: Dataset, *sequences: str, n_rows: int = 40) -> None:
    """Write a track parquet per sequence and rewrite ``tracks/index.csv``.

    Sequences accumulate: calling this again with a further name leaves the
    existing parquets in place, which is what lets a scenario widen a scope and
    then assert what was and was not recomputed.

    The group is empty, so the composite key renders as the bare sequence name
    and the parquet is ``<sequence>.parquet``.
    """
    tracks = dataset.get_root("tracks")
    tracks.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        frame = np.arange(n_rows, dtype=np.int64)
        pd.DataFrame(
            {
                "frame": frame,
                "time": frame / 30.0,
                "id": np.zeros(n_rows, dtype=np.int64),
                "feat_a": np.linspace(0.0, 1.0, n_rows),
            }
        ).to_parquet(tracks / f"{sequence}.parquet")
    present = sorted(tracks.glob("*.parquet"))
    index = pd.DataFrame(
        {
            "group": ["" for _ in present],
            "sequence": [path.stem for path in present],
            "abs_path": [str(path) for path in present],
        }
    )
    index.to_csv(tracks / "index.csv", index=False)


def track_sequences(dataset: Dataset) -> list[str]:
    """The sequence names currently present in ``tracks/``."""
    return sorted(p.stem for p in dataset.get_root("tracks").glob("*.parquet"))


@pytest.fixture
def scenario_dataset(tmp_path: Path) -> Dataset:
    """A real dataset with two synthetic track sequences.

    The backdrop the hashing workflows reference. A real ``Dataset`` rather than
    a stand-in, so scenario assertions exercise the same root resolution and
    index handling the control plane and notebooks do.
    """
    manifest = new_dataset_manifest(name="scenario", base_dir=tmp_path / "dataset")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    add_track_sequences(dataset, "seq_a", "seq_b")
    return dataset


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
