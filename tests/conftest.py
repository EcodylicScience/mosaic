"""Shared pytest fixtures."""

from __future__ import annotations

import csv
import importlib.util
import os
import shutil
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

# The same argument, for a binary rather than a module. Probing shells out to a
# system ffprobe, so every test that indexes real media hard-*fails* without one
# rather than skipping -- and the failure names a codec, not a missing tool.
# ``requires_ffprobe`` turns that into a skip locally; under CI a missing binary
# is a broken environment, exactly as a missing extra is.
CI_REQUIRED_BINARIES = ("ffprobe",)


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
    absent = [name for name in CI_REQUIRED_BINARIES if shutil.which(name) is None]
    if absent:
        raise pytest.UsageError(
            f"CI installs {', '.join(absent)} through ffmpeg, but it is not on "
            "PATH. Every media test would skip instead of running."
        )


@pytest.fixture
def requires_ffprobe() -> None:
    """Skip a test that needs to measure real media when ffprobe is absent.

    Requested by fixtures rather than by tests, so a test inherits the guard from
    the media it asks for. Under CI ``pytest_configure`` has already refused to
    start, so this never fires there.
    """
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not on PATH")


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


def add_tracks_variant(
    dataset: Dataset, run_id: str, *sequences: str, n_rows: int = 40
) -> None:
    """Write a variant-addressed track table per sequence, through the real writer.

    The counterpart to :func:`add_track_sequences`, which stays deliberately
    unlabelled -- it is the pre-Stage-3 dataset every existing analysis has, and
    keeping one fixture in that shape is what keeps proving that such a dataset
    still resolves and still hashes the same. This one is the shape a conversion
    writes today: tables under ``tracks/<run_id>/`` and rows naming the recipe.

    Uses ``write_tracks_row`` rather than a hand-built CSV, so the index it
    produces is the index production produces -- including the dedup that decides
    whether a second call adds a row or replaces one.
    """
    from mosaic.core.helpers import make_entry_key
    from mosaic.core.pipeline.tracks_identity import tracks_variant_root
    from mosaic.core.pipeline.tracks_index import write_tracks_row

    root = tracks_variant_root(dataset.get_root("tracks"), run_id)
    root.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        # Schema-valid ``trex_v1`` with two individuals, rather than the four
        # columns ``add_track_sequences`` writes. That is what lets a
        # *registered* feature actually run on this fixture -- including the
        # social ones, which need a sequence to hold at least two ids -- which
        # the chain-runner parity assertions depend on. ``feat_a`` stays for the
        # scenario mock features that read it.
        frame = np.tile(np.arange(n_rows, dtype=np.int64), 2)
        identity = np.repeat(np.arange(2, dtype=np.int64), n_rows)
        total = len(frame)
        columns: dict[str, object] = {
            "frame": frame,
            "time": frame / 30.0,
            "id": identity,
            "group": [""] * total,
            "sequence": [sequence] * total,
            "X#wcentroid": np.linspace(0.0, 10.0, total) + identity,
            "Y#wcentroid": np.linspace(0.0, 5.0, total) + identity,
            "feat_a": np.linspace(0.0, 1.0, total),
        }
        for keypoint in range(7):
            columns[f"poseX{keypoint}"] = np.linspace(0.0, 10.0, total) + keypoint
            columns[f"poseY{keypoint}"] = np.linspace(0.0, 5.0, total) + keypoint
        out_path = root / f"{make_entry_key('', sequence)}.parquet"
        pd.DataFrame(columns).to_parquet(out_path)
        write_tracks_row(
            dataset,
            run_id=run_id,
            group="",
            sequence=sequence,
            out_path=out_path,
            producer=run_id.split(".")[0],
            std_format="trex_v1",
            n_rows=n_rows,
        )


def track_sequences(dataset: Dataset) -> list[str]:
    """The sequence names the tracks index currently names.

    Read from the index rather than globbed off the root, so it answers the same
    for a flat legacy layout and for variant directories.
    """
    from mosaic.core.pipeline.tracks_index import read_tracks_index

    return sorted({str(name) for name in read_tracks_index(dataset)["sequence"]})


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


def add_media_sequence(
    dataset: Dataset,
    sequence: str,
    *,
    videos: tuple[str, ...] = ("a.mp4", "b.mp4"),
    frames: int = 6,
) -> None:
    """Give *sequence* real videos under ``media_raw`` and index them.

    Driven through ``Dataset.write_media_index``, the assignment path the control
    plane uses, so the media index and the composition it projects are the ones
    production produces rather than a hand-built stand-in.

    Each video's content varies with its filename. Two all-black videos are
    byte-identical and therefore share one ``video_uuid`` by design, so a
    composition over them is genuinely unchanged by a reorder -- which would make
    an ordering assertion pass while testing nothing.
    """
    from mosaic.core.pipeline.media_index import MediaIndexScope

    directory = dataset.get_root("media_raw") / sequence
    directory.mkdir(parents=True, exist_ok=True)
    for name in videos:
        shade = sum(name.encode()) % 200 + 20
        writer = cv2.VideoWriter(
            str(directory / name), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (64, 48)
        )
        for _ in range(frames):
            writer.write(np.full((48, 64, 3), shade, np.uint8))
        writer.release()

    _ = dataset.write_media_index(
        [
            MediaIndexScope(
                directory=directory,
                group="",
                sequence=sequence,
                order_by_name={name: i for i, name in enumerate(videos)},
            )
        ],
        extensions=(".mp4",),
    )


@pytest.fixture
def scenario_dataset_with_media(
    scenario_dataset: Dataset, requires_ffprobe: None
) -> Dataset:
    """``scenario_dataset``, plus two videos on ``seq_a``.

    **Composed rather than widened.** Three modules use the track-only
    ``scenario_dataset``, and giving it media would give all of them an ffprobe
    dependency for scenarios that never open a video. A scenario that needs media
    asks for it, and inherits the skip guard by asking.
    """
    add_media_sequence(scenario_dataset, "seq_a")
    return scenario_dataset
