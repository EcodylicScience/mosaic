"""Where tracker intermediates live, and what may be done with them -- item 8.1.

One root for every tracking-stage intermediate, and one place that says so. Item
8.1 asked for the root literal collapsed into a single constant "so the
relocation is a single edit"; by the time three trackers had landed there were
**six** copies of it -- one per tool in ``default_roots``, one per tool in each
runner's self-creating ``set_root`` -- and the add-a-tracker recipe had written
the duplication down as a checklist item rather than removing it.

**A table rather than a constant, because the sweeper needs more than a path.**
Item 8.4's retention is *per artifact class, not per root*: a ``.pv`` and its
settings are expensive and reusable, inference output is audit-only. Written as a
branch per tool that is three branches and then four; written as a column here it
is data, and a fifth tracker joins by adding a row.

**Naming: ``_tracking``, not ``tmp``.** The contents are generated and safe to
delete, but nothing evicts them automatically and no code may assume they are
ephemeral. The leading underscore marks the root as machine-generated without
hiding it, so a user browsing the dataset can see what is safe to remove.

**This module deliberately holds no index-row classes.** ``core`` does not import
``tracking`` -- the constraint ``provenance.py`` states and the reason its walk
leaves extracted frames out -- so a table naming ``TRexIndexRow`` would invert
the layering for the sake of a type annotation. What the sweeper and the
reconciler need from a row is ``prune_missing`` and ``drop_entries``, which every
``IndexCSV`` has whatever it holds; the row classes stay where they are written
and reach this table through registration, not import.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Final, Literal

from mosaic_media.transcode import TranscodeError

from mosaic.core.pipeline.markers import PhaseName
from mosaic.core.pipeline.refusal import Refusal

__all__ = [
    "DECODE_PROBE_IMPORT_FAILED",
    "TRACKING_ROOT",
    "TRACKING_ROOTS",
    "RetentionClass",
    "ToolCodecError",
    "TrackingPhase",
    "TrackingRoot",
    "is_under_tracking_root",
    "tracking_output_schema",
    "tracking_root",
    "tracking_root_default",
]

TRACKING_ROOT: Final = "_tracking"
"""The parent root, and the one string this programme wants written once.

Also the name a user-content scan must never descend into (item 8.1's exclusion
clause), which is why it is a bare component name rather than a path: the check
is against ``Path.parts``, since a directory exclusion cannot be expressed as a
basename pattern.
"""

RetentionClass = Literal["tracker", "inference", "conversion"]
"""Which retention window an intermediate falls under (item 8.4).

``tracker`` is the expensive, reusable, correctable output -- a ``.pv`` and its
settings, a ``.slp``, a predictions CSV. Its window is long and is ended by
promotion (item 8.6) rather than by age. ``inference`` is audit-only: neither
reused nor edited, kept so someone can see what a detector emitted before schema
coercion, and evicted on a shorter clock.

``conversion`` is the *input* to a tracker run rather than its output: a
detection pass shared by every run that tracks the same pixels under the same
detection settings. It is the most expensive artifact in the tree and the one
several runs read at once, so age alone must not reclaim it -- a slot still
named by a surviving tracker directory is refused whatever its age, and the
window only decides how long it lingers after its last reader is itself gone.

A closed alias rather than a bare ``str``, because the window a class maps to is
a policy decision and an unrecognized value must not silently fall through to the
longer one.
"""


@dataclass(frozen=True, slots=True)
class TrackingPhase:
    """One gated phase a producer completes, and what a re-run of it must remove.

    ``clear_globs`` is deliberately not ``TrackingRoot.outputs``. ``outputs`` is
    the sweeper's evidence that a directory holds real tracker output; these are
    the files a re-run of *this phase* must delete before it starts, which
    includes byproducts that are evidence of nothing (TREx's ``average_*.png``)
    and splits by phase what ``outputs`` lists in one flat tuple. A killed phase
    leaves partial files behind, and they must not be mistaken for -- or merged
    with -- the new run's.

    A glob matching a directory removes it as a tree, so a tool whose phase
    output is a session directory rather than a file is expressible.
    """

    name: PhaseName
    clear_globs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolDecoder:
    """Declare the decoder that a tool reads video with, and the codecs that it reads.

    mosaic chooses the codec of every file that it hands to an external tool, and
    does not control the decoder that opens the file. Some of those decoders
    include a software AV1 decoder and some do not. libavcodec's native ``av1``
    decoder wraps a hardware accelerator, and the software decoders
    (``libdav1d``, ``libaom-av1``) are external libraries that a build may omit.

    Each tool declares here the codecs that it reads, beside the other facts
    about each producer. TREx and Ultralytics declare AV1. SLEAP reads AV1 when
    the OpenCV in its environment links dav1d. Lightning Pose reads AV1 when the
    installed DALI's ``fn.readers.video`` handles it, and DALI 2.3 does not on
    any GPU.
    Those two declare a probe, which tests the environment that a run uses.

    Attributes:
        stack: The component that decodes, named in a refusal.
        also_reads: Codecs that this tool reads beyond the baseline in
            ``SOFTWARE_DECODABLE_CODECS``. Empty is the conservative answer and
            the default.
        probe: A Python program that the interpreter of the tool's environment
            runs with a file's path as its only argument. It decodes one frame
            with the reader that the tool uses and exits 0. When it cannot, it
            prints the reader's error and exits non-zero. When the reader does
            not import, it exits :data:`DECODE_PROBE_IMPORT_FAILED`. A file in a
            codec outside the declared set is handed to the tool only after the
            probe decodes it. Empty means the tool is not tested, and the
            declared set decides.
        remedy: What an operator can do about a refusal. Empty when the refusal
            cannot be remedied.
    """

    stack: str
    also_reads: frozenset[str] = frozenset()
    probe: str = ""
    remedy: str = ""


CONSERVATIVE_DECODER: Final = ToolDecoder(
    stack="a decoder mosaic neither installs nor configures",
    remedy=(
        "re-make the file in a codec every libavcodec build decodes, or set "
        "MOSAIC_ALLOW_TOOL_CODECS if this environment does handle it"
    ),
)
"""What a producer that has not declared a decoder is assumed to hold.

Assumes nothing beyond the baseline, so a tracker added without a declaration
refuses AV1 rather than being trusted with it. `tests/test_tracker_conformance.py`
turns that silence into a named failure.
"""


class ToolCodecError(Refusal, TranscodeError):
    """A tool is being handed a file its decoder stack may not open.

    Its own class rather than a reuse of ``StoreExportMissingError`` or
    ``JoinedExportMissingError``, because the remedy differs: those say build the
    file, and this one says the file exists and is in the wrong codec.

    A refusal, ``undecodable_codec``, raised when a tracker reaches the entry.
    The attempt ends there, and entries before it in the same run may already be
    tracked and published.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="undecodable_codec")


DECODE_PROBE_IMPORT_FAILED: Final = 3
"""The exit status of a decode probe whose interpreter does not import the reader.

That interpreter is not an environment of the tool, or the reader's installation
in it is broken, and the probe did not reach the file. It is told apart from a
reader that fails on the file, whose remedy is a decoder rather than a placement
or a repair.
"""


def _importing(reader: str, statements: str) -> str:
    """The head of a decode probe, which imports *reader* with *statements*.

    The import is outside the read, and an import that raises exits
    :data:`DECODE_PROBE_IMPORT_FAILED` with the error.
    """
    return (
        "import sys\n\n"
        "try:\n"
        f"{textwrap.indent(statements, '    ')}"
        "except Exception as exc:\n"
        f'    print(f"{reader} did not import: {{type(exc).__name__}}: {{exc}}", '
        "file=sys.stderr)\n"
        f"    sys.exit({DECODE_PROBE_IMPORT_FAILED})\n\n"
    )


_SLEAP_DECODE_PROBE: Final = _importing(
    "sleap_io", "import numpy as np\nimport sleap_io as sio\n"
) + (
    """\
try:
    frame = sio.load_video(sys.argv[1])[0]
except Exception as exc:
    sys.exit(f"sleap_io could not read frame 0: {type(exc).__name__}: {exc}")
if not isinstance(frame, np.ndarray) or frame.size == 0:
    sys.exit(f"sleap_io read frame 0 as {frame!r:.200}")
print(f"sleap_io read frame 0 with shape {frame.shape}")
"""
)
"""SLEAP's decode probe, which reads frame 0 with ``sleap_io.load_video``.

``sleap-nn track`` reads video through sleap-io, and sleap-io reads through OpenCV
whenever OpenCV is importable. When OpenCV cannot decode the codec, sleap-io
raises, and the probe exits 1 with its message.
"""

_LITPOSE_DECODE_PROBE: Final = _importing(
    "nvidia.dali", "from nvidia.dali import fn, pipeline_def, types\n"
) + (
    """\
try:
    @pipeline_def(batch_size=1, num_threads=1, device_id=0)
    def read_one_frame():
        return fn.readers.video(
            device="gpu",
            filenames=[sys.argv[1]],
            sequence_length=1,
            normalized=False,
            dtype=types.DALIDataType.FLOAT,
            file_list_include_preceding_frame=True,
            skip_vfr_check=True,
        )

    pipe = read_one_frame()
    pipe.build()
    pipe.run()
except Exception as exc:
    reason = str(exc).split("Stacktrace (", 1)[0].rstrip()
    sys.exit(f"DALI could not read a frame: {type(exc).__name__}: {reason}")
print("DALI read one frame")
"""
)
"""Lightning Pose's decode probe, which reads one frame through a DALI pipeline.

The pipeline runs on the GPU. Its reader takes the arguments of Lightning Pose's
prediction reader (``fn.readers.video`` in ``lightning_pose/data/dali.py``) that
affect decoding: ``device="gpu"``, ``normalized=False``, a float ``dtype``,
``file_list_include_preceding_frame=True`` and ``skip_vfr_check=True``. It reads
a sequence of one frame, and leaves out the batching, shuffling and padding
arguments. DALI 2.3's reader raises "Unhandled codec 225" for AV1 on every GPU.

DALI appends a native stacktrace to its error. The probe prints the error without
it, and its output ends with the reason.
"""


@dataclass(frozen=True, slots=True)
class TrackingRoot:
    """One tool's intermediate root, and what the sweeper needs to know about it.

    ``outputs`` are the globs that identify *real* output inside an entry working
    directory. They are not a completeness test -- a completion marker is, and
    that is item 8.2's -- but they are what distinguishes a directory a tracker
    wrote from a directory something else left behind.

    ``path_columns`` are this root's path-bearing index columns *beyond*
    ``abs_path``. They live here rather than in a table beside ``default_roots``
    because that table is what a new tracker forgets: a column missing from it
    silently stops being portable, and the add-a-tracker recipe had to carry a
    checklist item asking people to remember. One row per tracker, and the
    portability passes read it.

    ``phase_outputs`` is every gated phase this producer completes, in order,
    with what each one owns. The sweeper needs *all* of them before it will call
    a directory finished: without that, a TREx run whose conversion completed and
    whose tracking was killed reads as complete on the convert marker alone, and
    gets reclaimed at its age, taking a conversion someone is still using. The
    per-phase globs are here for the same reason the phase names are -- "what
    does this tool leave, and when" is producer knowledge, and this is where the
    machinery is allowed to have it without importing the producer.

    ``output_schema`` is the track schema this producer's bridged tables answer
    to -- the tracker-side counterpart of ``TrackConverter.output_schema``, and
    for the same reason. The bridge used to spell one module-level constant for
    every tracker, so ``meta.tracks.standard_format`` had no effect on any
    tracked table and a tracker whose columns genuinely differed had nowhere to
    say so. One row per producer, and the bridge reads it.

    ``model_sets`` is whether this producer runs several models as one set, named
    by a digest over all of them that equals no member's. SLEAP's top-down pair
    is the one such set. A variant of any other producer is named by each model's
    own identity, so a record of one that predates the recorded models is still
    decided by comparing a model's digest with the one it names.
    """

    key: str
    retention: RetentionClass
    outputs: tuple[str, ...]
    phase_outputs: tuple[TrackingPhase, ...]
    path_columns: tuple[str, ...] = ()
    output_schema: str = "trex_v1"
    decoder: ToolDecoder = CONSERVATIVE_DECODER
    model_sets: bool = False

    @property
    def phases(self) -> tuple[PhaseName, ...]:
        """Every gated phase this producer completes, in order."""
        return tuple(phase.name for phase in self.phase_outputs)

    @property
    def default_path(self) -> str:
        """This root's location, relative to the dataset base directory."""
        return f"{TRACKING_ROOT}/{self.key}"

    def clear_globs(self, phase: PhaseName) -> tuple[str, ...]:
        """What a re-run of *phase* must delete first, empty if it declares none."""
        for declared in self.phase_outputs:
            if declared.name == phase:
                return declared.clear_globs
        return ()


TRACKING_ROOTS: Final[dict[str, TrackingRoot]] = {
    root.key: root
    for root in (
        # `.pv` + settings from the convert phase, `.results` + per-individual
        # `data/*.npz` from the track phase. The background image is a convert
        # byproduct: cleared with the phase that writes it, and evidence of
        # nothing, so it is not in `outputs`.
        TrackingRoot(
            key="trex",
            decoder=ToolDecoder(
                stack="its own environment's libavcodec, which it links directly",
                also_reads=frozenset({"av1"}),
                remedy=(
                    "measured against a conda TREx environment, whose libavcodec "
                    "links libdav1d; if yours does not, "
                    "`ffmpeg -decoders | grep dav1d` in that environment says so"
                ),
            ),
            retention="tracker",
            output_schema="trex_v2",
            outputs=("*.pv", "*.settings", "*.results", "data/*.npz"),
            phase_outputs=(
                TrackingPhase("convert", ("*.pv", "*.settings", "average_*.png")),
                TrackingPhase("track", ("*.results", "data/*.npz")),
            ),
            path_columns=("video_abs_path", "pv_path"),
        ),
        # The shared conversion cache: one `.pv` per (detection settings, source
        # content), read by every tracker run whose convert-phase parameters and
        # media agree. A slot is addressed by both terms, so it is published
        # once and never rewritten -- which is what lets several runs read one
        # while a sixth is tracking off it.
        #
        # `*.results` is in the clear globs and in nothing else. TRex's
        # conversion writes one unconditionally, mosaic deletes it at publish,
        # and it must never sit beside a shared `.pv`: a results load with no
        # explicit path falls back to the *input* folder, so leaving one here
        # would put a stale tracking state where a later run could reach it.
        TrackingRoot(
            key="trex-convert",
            decoder=ToolDecoder(
                stack="its own environment's libavcodec, which it links directly",
                also_reads=frozenset({"av1"}),
                remedy=(
                    "measured against a conda TREx environment, whose libavcodec "
                    "links libdav1d; if yours does not, "
                    "`ffmpeg -decoders | grep dav1d` in that environment says so"
                ),
            ),
            retention="conversion",
            # Inert: nothing bridges from this root, and it is spelled rather
            # than defaulted because the default is the legacy centimetre schema.
            output_schema="trex_v2",
            outputs=("*.pv", "*.settings"),
            phase_outputs=(
                TrackingPhase(
                    "convert",
                    (
                        "*.pv",
                        "*.settings",
                        "average_*.png",
                        "*.results",
                        "*.results.meta",
                        ".incoming-*",
                    ),
                ),
            ),
            path_columns=("video_abs_path", "pv_path", "settings_path"),
        ),
        # The analysis export has no phase of its own -- it is ensured rather
        # than gated -- so the `.h5` is cleared with the inference it derives
        # from. Leaving it would strand a stale export from a superseded `.slp`
        # that the existence-gated export then declines to regenerate.
        TrackingRoot(
            key="sleap",
            decoder=ToolDecoder(
                stack="sleap-io, which reads video through OpenCV",
                probe=_SLEAP_DECODE_PROBE,
                remedy=(
                    "In the SLEAP environment, run `pip uninstall -y "
                    "opencv-python opencv-python-headless`, then `conda install "
                    "-c conda-forge py-opencv`, and add `--update-all` when the "
                    "solve fails on packages that the environment pins. The "
                    "Linux OpenCV wheel from PyPI does not decode AV1, and "
                    "sleap-io reads video through OpenCV whenever OpenCV is "
                    "importable"
                ),
            ),
            retention="tracker",
            output_schema="mosaic_v1",
            outputs=("*.predictions.slp", "*.analysis.h5"),
            phase_outputs=(
                TrackingPhase("track", ("*.predictions.slp", "*.analysis.h5")),
            ),
            path_columns=("video_abs_path", "slp_path", "analysis_h5_path"),
            # A top-down pair: the centroid model, then the centered-instance one.
            model_sets=True,
        ),
        TrackingRoot(
            key="litpose",
            decoder=ToolDecoder(
                stack="NVIDIA DALI's fn.readers.video, which decodes on the GPU",
                probe=_LITPOSE_DECODE_PROBE,
                remedy=(
                    "Hand Lightning Pose an H.264 file, such as a media variant "
                    'made with "codec": "h264". The fn.readers.video of DALI 2.3 '
                    'does not handle AV1 on any GPU, and fails with "Unhandled '
                    'codec 225"'
                ),
            ),
            retention="tracker",
            output_schema="mosaic_v1",
            outputs=("*.predictions.csv",),
            phase_outputs=(TrackingPhase("track", ("*.predictions.csv",)),),
            path_columns=("video_abs_path", "csv_path"),
        ),
        # The tracker configuration this run used lives at the *run* root, beside
        # run_params.json, rather than in an entry directory -- it is one value
        # for the whole run -- so it is neither evidence of a tracked entry nor
        # something re-running one must clear. The request and response the tool
        # exchanged are the opposite: byproducts of one attempt, cleared when the
        # phase re-runs so a stale request cannot sit beside fresh output, and
        # kept out of `outputs`, which is the sweeper's evidence of real output.
        TrackingRoot(
            key="ultralytics",
            decoder=ToolDecoder(
                stack="mosaic-media's own PyAV reader, inside the tool's environment",
                also_reads=frozenset({"av1"}),
            ),
            retention="tracker",
            output_schema="mosaic_v1",
            outputs=("*.predictions.parquet",),
            phase_outputs=(
                TrackingPhase(
                    "track",
                    (
                        "*.predictions.parquet",
                        "track-request.json",
                        "track-response.json",
                    ),
                ),
            ),
            path_columns=("video_abs_path", "predictions_path"),
        ),
        # Model inference (item 8.7). Audit-only: the parquet is what a detector
        # emitted *before* schema coercion, which is what you want when debugging
        # a bad model -- and nothing reads it back, so it is a byproduct on a
        # shorter clock rather than a cache. One root per inference kind, because
        # each is a separate op with its own identifiers.
        #
        # The two Ultralytics ops additionally exchange a JSON request and
        # response with the environment their model runs in, and those are
        # byproducts of one attempt: cleared when the phase re-runs so a stale
        # request cannot sit beside fresh output, and kept out of `outputs`,
        # which is the sweeper's evidence of real output. `infer-localizer` runs
        # in mosaic's own process and exchanges nothing.
        TrackingRoot(
            key="infer-pose",
            decoder=ToolDecoder(
                stack="mosaic-media's own PyAV reader, inside the tool's environment",
                also_reads=frozenset({"av1"}),
            ),
            retention="inference",
            output_schema="mosaic_v1",
            outputs=("predictions.parquet",),
            phase_outputs=(
                TrackingPhase(
                    "infer",
                    (
                        "predictions.parquet",
                        "infer-request.json",
                        "infer-response.json",
                    ),
                ),
            ),
        ),
        TrackingRoot(
            key="infer-points",
            decoder=ToolDecoder(
                stack="mosaic-media's own PyAV reader, inside the tool's environment",
                also_reads=frozenset({"av1"}),
            ),
            retention="inference",
            output_schema="mosaic_v1",
            outputs=("predictions.parquet",),
            phase_outputs=(
                TrackingPhase(
                    "infer",
                    (
                        "predictions.parquet",
                        "infer-request.json",
                        "infer-response.json",
                    ),
                ),
            ),
        ),
        TrackingRoot(
            key="infer-localizer",
            decoder=ToolDecoder(
                stack="mosaic-media's own PyAV reader, inside the tool's environment",
                also_reads=frozenset({"av1"}),
            ),
            retention="inference",
            output_schema="mosaic_v1",
            outputs=("predictions.parquet",),
            phase_outputs=(TrackingPhase("infer", ("predictions.parquet",)),),
        ),
    )
}
"""Every root under ``_tracking``, keyed by root key.

The keys are the op kinds, so ``ds.get_root(kind)`` answers for a tracker and an
inference op alike and no caller has to know which it is holding.
"""


def tracking_root(key: str) -> TrackingRoot:
    """The registered root of producer *key*.

    Raises:
        KeyError: If *key* is not registered, naming the keys that are. A
            producer that never joined the table has declared nothing, and a
            guess at its root, schema or table shape would be recorded as its
            declaration.
    """
    if key not in TRACKING_ROOTS:
        known = ", ".join(sorted(TRACKING_ROOTS))
        raise KeyError(f"unknown tracking root {key!r}; registered roots are {known}")
    return TRACKING_ROOTS[key]


def tracking_root_default(key: str) -> str:
    """The default location of tracker root *key*, relative to ``base_dir``.

    Raises on an unregistered key rather than composing a path for it. A tracker
    that has not joined the table is one the sweeper cannot see, and minting its
    root here would put output somewhere nothing reclaims.
    """
    return tracking_root(key).default_path


def tracking_output_schema(key: str) -> str:
    """The track schema producer *key* writes, for the caller that validates it.

    Raises on an unregistered key for the same reason
    :func:`tracking_root_default` does: guessing a schema for a producer that
    never joined the table would validate its tables against a contract nobody
    declared for them, and record that guess on every row.
    """
    return tracking_root(key).output_schema


def is_under_tracking_root(parts: tuple[str, ...]) -> bool:
    """Does a path with these components pass through ``_tracking``?

    Component-wise, never by prefix string: a scan is handed arbitrary search
    directories, and ``str.startswith`` would both miss a match below the search
    root and fire on a sibling named ``_tracking_backup``.
    """
    return TRACKING_ROOT in parts
