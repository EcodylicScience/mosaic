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
import importlib.metadata
import importlib.util
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest

from mosaic_media.io.writer import FFmpegVideoWriter

from mosaic.core.dataset import Dataset, new_dataset_manifest

# The plain helpers live in `tests.helpers`; only the ones the fixtures below
# call are imported here. Test modules import from the facade, never from
# this file.
from tests.helpers.environment import (
    FFMPEG_TOOLCHAIN,
    missing_ffmpeg_tools,
    require_ffmpeg as _require_ffmpeg,
)
from tests.helpers.media import MediaClip, add_media_sequence, write_media_index
from tests.helpers.tracks import add_track_sequences

# Modules every CI job must have, installed through the `test` dependency group.
# `imgstore` gates 35 tests behind ``pytest.importorskip``, so its absence
# presents as a skip rather than a failure -- a green CI that ran less than the
# workflow installed for. That is not hypothetical: the test step used to invoke
# `uv run pytest`, which re-synced the environment from `uv.lock` and pruned
# both extras before the first test ran.
#
# **A new optional dependency joins the install line and this tuple in the same
# change**, or its tests stop being evidence: adding it to the install alone
# leaves nothing to notice when it next vanishes.
#
# `pywt`, `h5py` and `tables` used to be here. They are base dependencies now, so
# requiring them of CI would assert something `pip install -e .` already
# guarantees -- and guarding them in a test is an error the coverage suite below
# reports, because a guard that can never fire masks a broken install.
#
# That rule is no longer only prose. ``test_optional_dependency_coverage.py``
# reads the suite's own ``importorskip`` calls out of its AST and fails when one
# names a module no tuple here requires -- which is how ``timm`` and ``tables``
# were found, both guarded and neither installed by any job, and how ``yaml`` was
# found being guarded despite being a *core* dependency, a guard that could never
# fire and would have masked a broken install.
CI_REQUIRED_MODULES = ("imgstore",)

# The same rule, scoped to one job. `torch` (via the `deep-learning` extra) is a
# ~200 MB wheel, so requiring it of every CI run would slow all of them down for
# tests only one job runs. It gets its own job instead, which sets
# MOSAIC_CI_IDENTITY=1 -- and inside that job the absence of torch is an error
# for exactly the reason above: `pytest.importorskip("torch")` would otherwise
# skip the T-Rex checkpoint tests green, and those are the only thing standing
# between a refactor and a silently randomly-initialised network inside T-Rex.
CI_IDENTITY_MODULES = ("torch", "timm")

# The same argument again for the tracking job, with one difference that changes
# its shape entirely: what that job installs is not a module. Ultralytics is
# AGPL-3.0 and mosaic never imports it -- it runs in an environment the user
# builds, reached as a subprocess -- so there is no import name to demand of
# *this* environment and deliberately no tuple below. Folding one in would claim
# a CI job installs Ultralytics here, and would then excuse an
# `importorskip("ultralytics")` that guards the wrong environment entirely.
#
# The rule itself is unchanged, pointed at the environment instead: under
# MOSAIC_CI_TRACKING an Ultralytics environment that does not resolve is a broken
# environment rather than a reason to skip. `test_ultralytics_preflight.py` skips
# its drift check when it cannot find one, and a job that builds the environment
# and then skips that check has proved nothing about the tracker tables it exists
# to compare.

# And again for the job that installs `feral`, which became possible at all only
# when FERAL started publishing to PyPI. `feral` is probed with
# ``importlib.util.find_spec`` rather than ``pytest.importorskip``, because two of
# the tests assert the *absence* path -- the ImportError naming the extra -- which
# ``importorskip`` cannot express. That probe is audited the same way; see
# ``test_optional_dependency_coverage.py``.
#
# Its own job for the usual reason and one more: FERAL pins its dependencies
# exactly, so the environment it produces is not the one the other jobs install.
CI_FERAL_MODULES = ("feral",)

# The same argument, for binaries rather than modules. Probing shells out to the
# system toolchain, so every test that indexes real media hard-*fails* without it
# rather than skipping -- and the failure names a codec, not a missing tool.
# ``requires_ffmpeg`` turns that into a skip locally; under CI a missing binary
# is a broken environment, exactly as a missing extra is.
#
# **Both binaries, not just ffprobe.** The guard was named and scoped for
# ``ffprobe`` alone, but the transcode op and the raw-H.264 packet scan shell out
# to ``ffmpeg``, and it is the one that goes missing first: with a stripped PATH
# the media suites died on `FileNotFoundError: 'ffmpeg'` while the guard reported
# the toolchain present.
#
# The list and the probe live in `tests.helpers.environment`, because the plain
# helpers there need the same answer and a second copy is how the two drift.
CI_REQUIRED_BINARIES = FFMPEG_TOOLCHAIN


def _refuse_an_unresolvable_external_environment() -> None:
    """Under the tracking job, an environment that is not there is an error.

    **Both** of them: the tracker and pose inference run upstream Ultralytics, and
    point inference runs the POLO fork, which cannot share an environment with it.
    Each is checked, because a skip in either place is a suite that reports green
    having compared nothing -- the tracker-table drift check against the release
    that runs it, and the point-inference path against a real fork, which nothing
    exercised at all before it had an environment to run in.

    Resolved through `tool_invocation` -- the same five-step ladder a real run
    walks -- rather than by reading the four variables here, so this check and the
    lookup it stands in for cannot come to disagree about what "the environment is
    there" means.

    Imported inside the call: only the tracking job asks, and the import reaches
    the whole pipeline package.
    """
    from mosaic.tracking.common.toolenv import ToolNotFoundError, tool_invocation
    from mosaic.tracking.common.ultralytics_env import POLO_ENV, ULTRALYTICS_ENV

    for env in (ULTRALYTICS_ENV, POLO_ENV):
        try:
            _ = tool_invocation(env, executable="python")
        except ToolNotFoundError as absent:
            raise pytest.UsageError(
                f"CI builds the {env.tool} environment for this job, but it does "
                f"not resolve: {absent} The checks that need it would skip "
                "silently instead of running against the release that will run "
                f"them. Check that the environment was built and that "
                f"{env.bin_var} or {env.conda_env_var} names it."
            ) from absent


def _refuse_two_opencv_builds() -> None:
    """Two *builds* of ``cv2`` in one environment is a broken install, everywhere.

    ``albumentations`` and ``albucore`` require ``opencv-python-headless``;
    ``mosaic-behavior`` and ``ultralytics`` require ``opencv-python``. Install both
    wheels and pip is happy: they are different distributions, so nothing conflicts --
    but they ship the *same* import package, so one overwrites the other's files and
    both leave their bundled native libraries in one ``cv2/.dylibs``. That directory
    then holds two ffmpeg builds (``libavcodec.61.19.100`` beside ``.101``), and the
    suite dies with ``Trace/BPT trap: 5`` at a different place on every run.

    **Counting distribution names is not the test; counting builds is.** conda-forge
    ships one ``py-opencv`` that deliberately registers *both* ``opencv-python`` and
    ``opencv-python-headless`` metadata for its single build -- which is exactly what
    lets one install satisfy mosaic's requirement and albumentations' at once, with no
    vendored ffmpeg at all because it links the environment's shared one. Refusing on
    the name count would reject the environment that has this right and accept nothing
    in exchange. So conda's providers collapse to the one build they describe, and
    every wheel counts on its own. A wheel installed *over* a conda build still fails
    the check, which is correct: it overwrites those files.

    Unlike the checks below this fires outside CI too, because the failure mode is a
    crash rather than a skip, and because the other half of it is silent: whichever
    wheel wins may be the headless one, which has no ``imshow``, so interactive
    playback breaks with no dependency error anywhere.

    Not covered, deliberately: ``av`` and ``opencv-python`` each vendor a *complete*
    ffmpeg, so two of those wheels collide the same way even when only one provides
    ``cv2``. That is the environment CI installs and has never crashed on, so it is a
    documented hazard rather than a refusal here -- see the conda-forge recipe in
    CLAUDE.md, which is what removes it.
    """
    conda: list[str] = []
    wheels: list[str] = []
    for dist in importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if "opencv" not in name:
            continue
        installer = (dist.read_text("INSTALLER") or "").strip().lower()
        (conda if installer == "conda" else wheels).append(name)

    builds = sorted(wheels) + ([f"conda ({', '.join(sorted(conda))})"] if conda else [])
    if len(builds) > 1:
        raise pytest.UsageError(
            f"{len(builds)} builds of cv2 are installed ({', '.join(builds)}). "
            "They share one import package, so their bundled ffmpeg libraries land in "
            "one directory and the suite crashes nondeterministically. Keep exactly "
            "one -- `pip uninstall opencv-python opencv-python-headless` then either "
            "`pip install 'opencv-python>=4.7'`, or `conda install -c conda-forge "
            "py-opencv`, which links the environment's shared ffmpeg and registers "
            "both names for one build."
        )


def pytest_configure() -> None:
    """Under CI, a missing optional dependency is an error rather than a skip.

    Local runs are unaffected: a developer without ``imgstore`` installed still
    gets skips, which is the point of ``importorskip``. Only CI, which installs
    them explicitly, treats their absence as a broken environment. The tracking
    job's dependency is an external environment rather than an import, and is
    held to the same standard by the call at the end.
    """
    _refuse_two_opencv_builds()
    if not os.environ.get("CI"):
        return
    required = CI_REQUIRED_MODULES
    if os.environ.get("MOSAIC_CI_IDENTITY"):
        required += CI_IDENTITY_MODULES
    if os.environ.get("MOSAIC_CI_FERAL"):
        required += CI_FERAL_MODULES
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise pytest.UsageError(
            f"CI installs {', '.join(missing)} through extras, but they are not "
            "importable. The suite would skip silently instead of failing. Check "
            "that the test step does not re-sync the environment away "
            "(uv run --no-sync)."
        )
    if os.environ.get("MOSAIC_CI_TRACKING"):
        _refuse_an_unresolvable_external_environment()
    absent = [name for name in CI_REQUIRED_BINARIES if shutil.which(name) is None]
    if absent:
        raise pytest.UsageError(
            f"CI installs {', '.join(absent)} through ffmpeg, but it is not on "
            "PATH. Every media test would skip instead of running."
        )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``media``-marked tests when the ffmpeg toolchain is absent.

    The fixture below covers a test that reaches the toolchain *through* a
    fixture. This covers the rest: a module that shells out directly marks itself
    ``pytestmark = pytest.mark.media`` and gets the same outcome. Two routes to
    one answer, rather than one missing binary producing a skip in some files and
    a bare ``FileNotFoundError`` in others.

    Under CI ``pytest_configure`` has already refused to start, so this is a
    local-only path.
    """
    missing = missing_ffmpeg_tools()
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"not on PATH: {', '.join(missing)}")
    for item in items:
        if "media" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def requires_ffmpeg() -> None:
    """Skip a test that needs the ffmpeg toolchain when it is not on ``PATH``.

    Requested by fixtures rather than by tests, so a test inherits the guard from
    the media it asks for. Under CI ``pytest_configure`` has already refused to
    start, so this never fires there.

    Both binaries: probing shells to ``ffprobe``, while the transcode op and the
    raw-H.264 packet scan shell to ``ffmpeg``. A PATH carrying one and not the
    other used to produce a bare ``FileNotFoundError`` naming whichever came
    first, from inside a test that never mentions either.

    The fixture form for a test that reaches media through a fixture;
    ``tests.helpers.require_ffmpeg`` is the same guard for a helper a test calls
    directly. One answer, two shapes.
    """
    _require_ffmpeg()


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
def write_cfr_mp4(requires_ffmpeg: None) -> Callable[..., None]:
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
def make_media_dataset(requires_ffmpeg: None) -> Callable[[Path], Dataset]:
    """Factory building a saved Dataset with ``media_raw``, ``media`` and
    ``tracks`` roots.

    Guarded on the ffmpeg toolchain because a media-shaped dataset exists to be
    indexed, and indexing shells out. The guard used to sit only on
    ``scenario_dataset_with_media``, so the 14 files reaching media through this
    factory met a bare ``FileNotFoundError`` where the other files skipped.

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

    The guard stays *inside* the fixture and carries an explicit reason. It
    cannot move to module scope -- ``conftest.py`` is imported by every run, so a
    top-level import would make ``imgstore`` mandatory for the whole suite -- and
    it should not move to the files that request this fixture: only 4 of
    ``test_media_reprobe.py``'s 44 tests and 4 of
    ``test_media_identity_matching.py``'s 9 touch a store, so a module-level
    guard there would skip 45 tests to protect 8. What was actually wrong is that
    the skip named nothing; ``-ra`` in ``addopts`` now prints this reason in
    every run's summary.
    """
    imgstore = pytest.importorskip(
        "imgstore",
        reason="imgstore is not installed (pip install --group test)",
    )

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


@pytest.fixture
def scenario_dataset_with_media(
    scenario_dataset: Dataset, requires_ffmpeg: None
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


@pytest.fixture
def dataset_without_index(tmp_path: Path) -> Dataset:
    """An initialized dataset whose originals index does not exist.

    What proves an entries-only scope needs no index: resolving one here must
    not raise.
    """
    manifest = new_dataset_manifest("no-index", base_dir=tmp_path)
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


@pytest.fixture
def two_entry_dataset(tmp_path: Path) -> Dataset:
    """Media rows for (A, one), (A, two) and (B, one), each one video.

    Group B repeats the sequence name 'one' on purpose: it is what makes a
    sequences-only selector resolve to two entries, and what a cross product
    cannot express.
    """
    manifest = new_dataset_manifest("two-entry", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    write_media_index(
        dataset,
        [
            MediaClip(
                filename="a1.mp4", group="A", sequence="one", video_uuid="uid-a1"
            ),
            MediaClip(
                filename="a2.mp4", group="A", sequence="two", video_uuid="uid-a2"
            ),
            MediaClip(
                filename="b1.mp4", group="B", sequence="one", video_uuid="uid-b1"
            ),
        ],
    )
    return dataset


@pytest.fixture
def two_camera_dataset(tmp_path: Path) -> Dataset:
    """One entry, (A, one), with two media rows differing only by camera."""
    manifest = new_dataset_manifest("two-camera", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    write_media_index(
        dataset,
        [
            MediaClip(
                filename="cam0.mp4",
                group="A",
                sequence="one",
                camera="cam0",
                video_uuid="uid-cam0",
            ),
            MediaClip(
                filename="cam1.mp4",
                group="A",
                sequence="one",
                camera="cam1",
                video_uuid="uid-cam1",
            ),
        ],
    )
    return dataset
