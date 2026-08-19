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

from dataclasses import dataclass
from typing import Final, Literal

from mosaic.core.pipeline.markers import PhaseName

__all__ = [
    "TRACKING_ROOT",
    "TRACKING_ROOTS",
    "RetentionClass",
    "TrackingPhase",
    "TrackingRoot",
    "is_under_tracking_root",
    "tracking_output_schema",
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

    ``joins_sources`` is whether this tool can read an entry's several clips as
    one continuous video. It lives here, beside the other producer knowledge,
    rather than as a check against the tool's name in the scope builder: "what
    can this tool do" is exactly what this table is for, and a fifth tracker
    copying a row has to answer it rather than inherit someone else's answer by
    matching a string. ``False`` means the scope builder truncates an entry to
    its first clip and says so, which is what every tracker did before any of
    them could join.
    """

    key: str
    retention: RetentionClass
    outputs: tuple[str, ...]
    phase_outputs: tuple[TrackingPhase, ...]
    path_columns: tuple[str, ...] = ()
    output_schema: str = "trex_v1"
    joins_sources: bool = False

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
            retention="tracker",
            output_schema="trex_v2",
            outputs=("*.pv", "*.settings", "*.results", "data/*.npz"),
            phase_outputs=(
                TrackingPhase("convert", ("*.pv", "*.settings", "average_*.png")),
                TrackingPhase("track", ("*.results", "data/*.npz")),
            ),
            path_columns=("video_abs_path", "pv_path"),
            # TRex's `source` is a PathArray, and its VideoSource sums the frame
            # counts of every file it names into one length -- so a session's
            # clips convert into a single `.pv` with one continuous frame index,
            # and identities never break at a clip boundary.
            joins_sources=True,
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
            # Mirrors the `trex` row: a joined session converts once, into one
            # slot addressed by the composition digest of its ordered clips.
            joins_sources=True,
        ),
        # The analysis export has no phase of its own -- it is ensured rather
        # than gated -- so the `.h5` is cleared with the inference it derives
        # from. Leaving it would strand a stale export from a superseded `.slp`
        # that the existence-gated export then declines to regenerate.
        TrackingRoot(
            key="sleap",
            retention="tracker",
            output_schema="mosaic_v1",
            outputs=("*.predictions.slp", "*.analysis.h5"),
            phase_outputs=(
                TrackingPhase("track", ("*.predictions.slp", "*.analysis.h5")),
            ),
            path_columns=("video_abs_path", "slp_path", "analysis_h5_path"),
        ),
        TrackingRoot(
            key="litpose",
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


def tracking_root_default(key: str) -> str:
    """The default location of tracker root *key*, relative to ``base_dir``.

    Raises on an unregistered key rather than composing a path for it. A tracker
    that has not joined the table is one the sweeper cannot see, and minting its
    root here would put output somewhere nothing reclaims.
    """
    if key not in TRACKING_ROOTS:
        known = ", ".join(sorted(TRACKING_ROOTS))
        raise KeyError(f"unknown tracking root {key!r}; registered roots are {known}")
    return TRACKING_ROOTS[key].default_path


def tracking_output_schema(key: str) -> str:
    """The track schema producer *key* writes, for the caller that validates it.

    Raises on an unregistered key for the same reason
    :func:`tracking_root_default` does: guessing a schema for a producer that
    never joined the table would validate its tables against a contract nobody
    declared for them, and record that guess on every row.
    """
    if key not in TRACKING_ROOTS:
        known = ", ".join(sorted(TRACKING_ROOTS))
        raise KeyError(f"unknown tracking root {key!r}; registered roots are {known}")
    return TRACKING_ROOTS[key].output_schema


def is_under_tracking_root(parts: tuple[str, ...]) -> bool:
    """Does a path with these components pass through ``_tracking``?

    Component-wise, never by prefix string: a scan is handed arbitrary search
    directories, and ``str.startswith`` would both miss a match below the search
    root and fire on a sibling named ``_tracking_backup``.
    """
    return TRACKING_ROOT in parts
