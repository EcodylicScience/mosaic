"""Shared pytest fixtures."""

from __future__ import annotations

import os as _os

# Pin OpenMP to one thread before any library that links it is imported.
#
# torch and xgboost each bundle their own OpenMP runtime, and a process holding
# both *segfaults* on macOS -- in either import order, inside whichever of them
# is asked to do real work first. Nothing imported torch until the T-Rex
# checkpoint tests arrived, so the suite never met this; with them, a full run
# dies at 92% with SIGSEGV in `load_state_dict` or in an xgboost booster call.
#
# The two OpenMP copies are the disease (`torch/lib/libomp.dylib` and
# `sklearn/.dylibs/libomp.dylib` both ship in the venv). Single-threading them
# is the mitigation that actually holds: `KMP_DUPLICATE_LIB_OK=TRUE`, the usual
# advice, does *not* stop the crash here, and it is documented as being able to
# produce silently wrong results -- not a trade worth making in this codebase.
# One thread costs the suite ~15s and changes no numerics.
#
# `setdefault`, so a caller who wants threads back can say so.
#
# This is a real hazard for users too: xgboost is a core dependency and torch an
# optional extra, so a session that trains an identity model and an XGBoost
# classifier hits the same crash outside the tests. That half is not fixable
# from here -- it needs a venv with one OpenMP runtime.
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import dataclasses
import importlib.metadata
import importlib.util
import os
import re
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaFacts, derive
from mosaic_media.io.writer import FFmpegVideoWriter
from mosaic_media.transcode import Target

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import facts_to_row, store_facts

# Modules the CI workflow installs through extras (`.[wavelets,imgstore,sleap]`).
# `imgstore` gates 35 tests behind ``pytest.importorskip``, so its absence
# presents as a skip rather than a failure -- a green CI that ran less than the
# workflow installed for. That is not hypothetical: the test step used to invoke
# `uv run pytest`, which re-synced the environment from `uv.lock` and pruned
# both extras before the first test ran.
#
# `h5py` is here because the list was already one module behind the suite: the
# SLEAP integration arrived depending on it, correctly guarded by
# ``importorskip``, and CI neither installed it nor demanded it -- so nine tests
# (every SLEAP marker and reuse test, the three converter tests, and the SLEAP
# provenance test) skipped green. **A new optional dependency joins the install
# line and this tuple in the same change**, or its tests stop being evidence:
# adding it to the install alone leaves nothing to notice when it next vanishes.
CI_REQUIRED_MODULES = ("imgstore", "pywt", "h5py")

# The same rule, scoped to one job. `torch` (via the `identity` extra) is a
# ~200 MB wheel, so requiring it of every CI run would slow all of them down for
# tests only one job runs. It gets its own job instead, which sets
# MOSAIC_CI_IDENTITY=1 -- and inside that job the absence of torch is an error
# for exactly the reason above: `pytest.importorskip("torch")` would otherwise
# skip the T-Rex checkpoint tests green, and those are the only thing standing
# between a refactor and a silently randomly-initialised network inside T-Rex.
CI_IDENTITY_MODULES = ("torch",)

# The same argument again, for the job that installs `pose`. `lap` is the one
# that matters: Ultralytics imports it at module scope in its tracker matching
# and pip-installs it at run time when it is absent, so a job without it would
# skip the drift and reset checks green while proving nothing about the
# declaration that exists to stop that install happening.
CI_TRACKING_MODULES = ("ultralytics", "lap")

# The same argument, for a binary rather than a module. Probing shells out to a
# system ffprobe, so every test that indexes real media hard-*fails* without one
# rather than skipping -- and the failure names a codec, not a missing tool.
# ``requires_ffprobe`` turns that into a skip locally; under CI a missing binary
# is a broken environment, exactly as a missing extra is.
CI_REQUIRED_BINARIES = ("ffprobe",)


def _refuse_two_opencvs() -> None:
    """Two distributions providing ``cv2`` is a broken environment, everywhere.

    ``albumentations`` and ``albucore`` require ``opencv-python-headless``;
    ``mosaic-behavior`` and ``ultralytics`` require ``opencv-python``. Install both
    and pip is happy: they are different distributions, so nothing conflicts -- but
    they ship the *same* import package, so one overwrites the other's files and both
    leave their bundled native libraries in one ``cv2/.dylibs``. That directory then
    holds two ffmpeg builds (``libavcodec.61.19.100`` beside ``.101``), and the suite
    dies with ``Trace/BPT trap: 5`` at a different place on every run.

    Unlike the checks below this fires outside CI too, because the failure mode is a
    crash rather than a skip, and because the other half of it is silent: whichever
    wheel wins may be the headless one, which has no ``imshow``, so interactive
    playback breaks with no dependency error anywhere.
    """
    providers = sorted(
        name
        for dist in importlib.metadata.distributions()
        if "opencv" in (name := (dist.metadata["Name"] or "").lower())
    )
    if len(providers) > 1:
        raise pytest.UsageError(
            f"{len(providers)} distributions provide cv2 ({', '.join(providers)}). "
            "They share one import package, so their bundled ffmpeg libraries land in "
            "one directory and the suite crashes nondeterministically. Keep exactly "
            "one -- `uv pip uninstall opencv-python opencv-python-headless` then "
            "`uv pip install 'opencv-python>=4.7'`, which is the build whose imshow "
            "playback needs."
        )


def pytest_configure() -> None:
    """Under CI, a missing optional dependency is an error rather than a skip.

    Local runs are unaffected: a developer without ``imgstore`` installed still
    gets skips, which is the point of ``importorskip``. Only CI, which installs
    them explicitly, treats their absence as a broken environment.
    """
    _refuse_two_opencvs()
    if not os.environ.get("CI"):
        return
    required = CI_REQUIRED_MODULES
    if os.environ.get("MOSAIC_CI_IDENTITY"):
        required += CI_IDENTITY_MODULES
    if os.environ.get("MOSAIC_CI_TRACKING"):
        required += CI_TRACKING_MODULES
    missing = [name for name in required if importlib.util.find_spec(name) is None]
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

    The shape every media test needs: a real file ffprobe can measure, and
    cheap enough to write per test at roughly 14 ms a clip. Returns a callable
    ``(path, frames=, size=)``.

    Written through the toolkit's own writer, so the fixture encodes AV1. The
    codec is load-bearing rather than incidental: the read-target gate refuses
    an ``"analysis"`` read whose verdict carries
    ``unverified_frame_correspondence``, which every codec outside the measured
    frame-exact set does, and AV1 is in that set. The writer is also the one
    mosaic already ships and the codec its analysis transcode targets, so a
    fixture and a real derivative are the same kind of file.

    An OpenCV ``VideoWriter`` is not used, but not because it could not be: its
    ``mp4v`` fourcc encodes MPEG-4, which is not frame exact, though its ``VP80``
    and ``VP90`` fourccs are. Those write WebM rather than the mp4 this
    fixture's name and its callers' ``.mp4`` paths promise.
    """

    def _write(path: Path, frames: int = 6, size: tuple[int, int] = (64, 48)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with FFmpegVideoWriter(path, width=size[0], height=size[1], fps=30.0) as writer:
            for _ in range(frames):
                writer.write(np.zeros((size[1], size[0], 3), np.uint8))

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

    ``X``/``Y`` are here because the features these scenarios run need them. Without
    them every entry's ``apply`` raised, and because a per-entity failure used to be
    swallowed, the run reported success having computed nothing -- so tests asserted
    on the ``params.json`` of a run with no outputs.
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
                "X": np.linspace(0.0, 10.0, n_rows),
                "Y": np.linspace(10.0, 0.0, n_rows),
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


def write_trex_npz(
    path: Path,
    *,
    individual: int | None = None,
    n: int = 8,
    cm_per_pixel: float = 1.0,
    **columns: np.ndarray,
) -> None:
    """Write a per-individual TREx export carrying what TREx always writes.

    Six near-identical builders used to sit in six test modules, and every one of
    them omitted the two fields that decide what a TREx table *means*:
    ``cm_per_pixel``, which says whether its positions are centimetres, and the
    ``#wcentroid`` pair, which is the body centre. A file without them is not a
    file TREx produces, so tests built on one were measuring a shape that cannot
    occur.

    ``cm_per_pixel`` and ``id`` are written as one-element arrays because that is
    how TREx writes them -- as ``std::vector`` of one, not as scalars -- which is
    what makes them arrive NaN-padded rather than broadcast.

    The bare ``X``/``Y`` are given the same values as ``#wcentroid`` by default.
    In a real export they differ (bare is the head), but most callers only need
    *a* position; a caller testing the head-versus-centre distinction passes them
    explicitly through *columns*.

    ``individual`` defaults to the trailing digits of the filename, because TREx
    names each file for the individual it holds -- ``myseq_fish0.npz`` beside
    ``myseq_fish1.npz``. Defaulting it to a constant instead would give a
    sequence's several files one id and quietly collapse them into one animal.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if individual is None:
        match = re.search(r"(\d+)$", path.stem)
        individual = int(match.group(1)) if match else 0
    centre_x = np.linspace(0.0, 1.0, n)
    centre_y = np.linspace(1.0, 0.0, n)
    fields: dict[str, np.ndarray] = {
        "frame": np.arange(n, dtype=np.int64),
        "time": np.arange(n, dtype=float) / 30.0,
        "id": np.array([individual]),
        "cm_per_pixel": np.array([cm_per_pixel]),
        "X": centre_x,
        "Y": centre_y,
        "X#wcentroid": centre_x,
        "Y#wcentroid": centre_y,
        "poseX0": centre_x,
        "poseY0": centre_y,
    }
    fields.update(columns)
    np.savez(path, **fields)


def add_tracks_variant(
    dataset: Dataset,
    run_id: str,
    *sequences: str,
    n_rows: int = 40,
    consumed_source_roots: tuple[str, ...] = ("tracks_raw",),
    std_format: str = "trex_v1",
) -> None:
    """Write a variant-addressed track table per sequence, through the real writer.

    ``std_format`` names the schema the rows claim. It defaults to the legacy
    ``trex_v1`` so existing callers are unchanged; a scenario about a dataset
    part-way through a migration sets it per call, which is the only way to build
    one index holding two schema families.

    ``consumed_source_roots`` defaults to what all three conversion writers pass,
    so a row this produces answers "which root would a change have to be under?"
    the way a converted row does. Overridable to ``()`` for a scenario about a
    row that predates the column.

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
        # A schema-valid table with two individuals, rather than the four columns
        # ``add_track_sequences`` writes. That is what lets a *registered*
        # feature actually run on this fixture -- including the social ones,
        # which need a sequence to hold at least two ids -- which the
        # chain-runner parity assertions depend on. ``feat_a`` stays for the
        # scenario mock features that read it.
        #
        # X/Y are the body centre and every converter emits them. This fixture
        # carried only the ``#wcentroid`` pair, a shape no converter produces,
        # so tests built on it were measuring a table that cannot exist.
        # ``#wcentroid`` stays, holding the identical values, because that is
        # what a TREx table looks like: one body centre under both names.
        frame = np.tile(np.arange(n_rows, dtype=np.int64), 2)
        identity = np.repeat(np.arange(2, dtype=np.int64), n_rows)
        total = len(frame)
        centre_x = np.linspace(0.0, 10.0, total) + identity
        centre_y = np.linspace(0.0, 5.0, total) + identity
        columns: dict[str, object] = {
            "frame": frame,
            "time": frame / 30.0,
            "id": identity,
            "group": [""] * total,
            "sequence": [sequence] * total,
            "X": centre_x,
            "Y": centre_y,
            "X#wcentroid": centre_x,
            "Y#wcentroid": centre_y,
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
            std_format=std_format,
            n_rows=n_rows,
            consumed_source_roots=consumed_source_roots,
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

    ``fill`` tags the *whole* frame with ``i`` rather than only its first pixel.
    A one-pixel tag cannot survive an encode to 4:2:0, whose chroma planes are
    subsampled 2x2, so a test that reads frames back through a lossy codec (the
    store export) needs frames that are uniform enough to stay distinguishable.
    The ``frame[0, 0, 0] == i`` invariant holds either way.

    Returns a callable ``(name=, nframes=, fmt=, shape=, dtype=, chunksize=,
    parent=, fill=, extra_metadata=) -> (store_dir, frames)``.
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
        fill: bool = False,
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
            if fill:
                # Spread across the middle of the range rather than using i
                # directly. Consecutive small integers are indistinguishable
                # after a lossy encode -- everything below the limited-range
                # floor comes back as 0 -- and the point of a filled frame is to
                # stay identifiable through one.
                value = 16 + (i * 200) // max(1, nframes - 1)
                img = np.full(shape, value, dtype=dtype)
            else:
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
        with FFmpegVideoWriter(
            directory / name, width=64, height=48, fps=30.0
        ) as writer:
            for _ in range(frames):
                writer.write(np.full((48, 64, 3), shade, np.uint8))

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

    ``seq_b`` deliberately stays media-less, and the track-only fixture stays
    track-only: two H3 scenarios *are* the transition from no media to media, and
    seeding it would make one vacuous and fail the other outright.
    """
    add_media_sequence(scenario_dataset, "seq_a")
    return scenario_dataset


def clean_facts_cells(video_uuid: str = "") -> dict[str, object]:
    """A complete, verdict-clean set of media-facts cells for one index row.

    The tracker marker suites all need a media row a tracker will actually run
    against: probed dimensions, a container and pixel format that derive to a
    clean verdict, and -- when *video_uuid* is given -- the content identity that
    lets a marker tell a video replaced in place from one merely renamed.
    """
    facts: MediaFacts = store_facts(
        width=640,
        height=480,
        fps=30.0,
        frame_count=100,
        codec="h264",
        duration=100 / 30.0,
        video_uuid=video_uuid,
        identity_scheme="video/1" if video_uuid else "",
    )
    facts = dataclasses.replace(
        facts,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        pixel_format="yuv420p",
        moov_at_start=True,
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


def write_media_index(
    dataset: Dataset,
    sequences: list[str],
    *,
    filenames: dict[str, str] | None = None,
    uids: dict[str, str] | None = None,
) -> None:
    """Index one stub video per sequence, with full facts cells.

    The bytes are a placeholder: every tracker marker suite fakes the tool, so
    nothing decodes them. *filenames* and *uids* are what the rename-versus-replace
    scenarios vary -- the same file under a new name keeps its uid, a replacement
    changes it.
    """
    media_root = dataset.get_root(dataset.resolve_media_root())
    media_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seq in sequences:
        filename = (filenames or {}).get(seq, f"{seq}.mp4")
        video = media_root / filename
        if not video.exists():
            _ = video.write_bytes(b"fake")
        rows.append(
            {
                "name": filename,
                "group": "",
                "sequence": seq,
                "group_safe": "",
                "sequence_safe": seq,
                "abs_path": dataset.relative_to_root(video),
                "size_bytes": 4,
                "mtime_iso": "",
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "codec": "h264",
                "media_type": "video",
                "video_order": 0,
                **clean_facts_cells((uids or {}).get(seq, "")),
            }
        )
    pd.DataFrame(rows).to_csv(media_root / "index.csv", index=False)


def add_transcode_derivative(
    dataset: Dataset, sequence: str, *, target: Target = "playback"
) -> Path:
    """Register a derivative for *sequence*'s first video, without encoding one.

    A stub, because nothing being tested reads a derivative's bytes -- what is
    read is its *name*, so it is written under the scheme the transcode op uses
    and the recipe is computed through the op's own function rather than
    hard-coded (the recipe folds environment-driven thresholds, so a literal
    would pin the suite to one machine).

    Both links are written, in the order the op writes them: the back-link row
    into the ``media`` index, then the forward-link cell onto the original.

    ``playback`` by default, matching the scenario this exists for -- a proxy
    made so a browser can play the video, which the tracker, frame extraction,
    crops and every feature ignore.
    """
    from mosaic_media import CHROME_149
    from mosaic_media.transcode import ANALYSIS_ENCODING, PLAYBACK_ENCODING

    from mosaic.core.media.facts_columns import (
        MEDIA_INDEX_COLUMNS,
        derivative_column_for_target,
    )
    from mosaic.core.pipeline.media_index import (
        frame_from_rows,
        read_media_index,
        write_media_index_rows,
    )
    from mosaic.core.pipeline.transcode import (
        TRANSCODE_KIND_DIRECTORY,
        TranscodeParams,
        transcode_recipe_hash,
    )
    from mosaic.media_probe_config import media_thresholds

    raw_index = dataset.get_root("media_raw") / "index.csv"
    originals = [dict(row) for row in read_media_index(raw_index)]
    matches = [row for row in originals if row.get("sequence") == sequence]
    if not matches:
        raise AssertionError(f"no media_raw row for sequence {sequence!r}")
    original = matches[0]
    video_uuid = original["video_uuid"]

    recipe = transcode_recipe_hash(
        TranscodeParams(entry=("", sequence), target=target),
        ANALYSIS_ENCODING if target == "analysis" else PLAYBACK_ENCODING,
        CHROME_149,
        media_thresholds(),
    )
    transcode_root = dataset.get_root("media") / TRANSCODE_KIND_DIRECTORY
    transcode_root.mkdir(parents=True, exist_ok=True)
    derivative = transcode_root / f"{video_uuid}.{recipe}.{target}.mp4"
    _ = derivative.write_bytes(b"stub")

    media_index = dataset.get_root("media") / "index.csv"
    rows = [dict(row) for row in read_media_index(media_index)]
    row: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
    row.update(
        {
            "name": derivative.name,
            "group": original.get("group", ""),
            "sequence": sequence,
            "abs_path": dataset.relative_to_root(str(derivative)),
            "source_video_uuid": video_uuid,
            "recipe_hash": recipe,
        }
    )
    rows.append(row)
    write_media_index_rows(media_index, frame_from_rows(rows))

    column = derivative_column_for_target(target)
    for candidate in originals:
        if candidate.get("video_uuid") == video_uuid:
            candidate[column] = f"{TRANSCODE_KIND_DIRECTORY}/{derivative.name}"
    write_media_index_rows(raw_index, frame_from_rows(list(originals)))
    return derivative
