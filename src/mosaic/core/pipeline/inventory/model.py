"""What the inventory reports: coverage, artifact records, and one status rule.

**Coverage is not a boolean, and its key type is not one type.** A step covering
50 of 90 sequences is not "done", and it becomes less done the moment the scope
widens to 120 -- so what is reported is *which* entries exist, never a flag. And
the thing covered differs by artifact kind: a feature run covers
``(group, sequence)``, a frame run covers a ``(group, sequence, camera)`` because
the cameras of one recording share an entry, and a trained model is one artifact
that is covered or is not. Transcode covers *media rows*, and has no
run-addressed directory at all.

That last case is why :class:`Coverage` is generic in its key and why the lookup
takes a per-kind ``ref``. Under one signature -- ``coverage(storage, run_id)`` --
transcode has no run identifier to pass, its "run root" does not exist, and an
already-clean corpus reads as permanently incomplete: nothing is missing because
nothing was ever supposed to be produced, but a directory-shaped check finds no
directory and says zero. A decider acting on that resubmits the same work every
tick, forever.

**Status is derived, never stored.** Every value here is recomputed from the
record each time it is asked for. Nothing in this package writes a status cell,
because a stored one goes stale and forks from the artifacts it describes --
which is the failure mode that makes people mark tasks succeeded by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, overload

from .params import ParamsState, RunParams

__all__ = [
    "AnyRecord",
    "ArtifactKind",
    "ArtifactRecord",
    "ArtifactRef",
    "ArtifactStatus",
    "CameraEntry",
    "CameraRecord",
    "Coverage",
    "DatasetInventory",
    "Entry",
    "EntryRecord",
    "FeatureRunRef",
    "FrameRunRef",
    "InventoryScope",
    "LabelsVariantRef",
    "MediaDerivativeRef",
    "TrackerRunRef",
    "TracksVariantRef",
    "TrainedModelRef",
    "UnitRecord",
    "classify",
]

type Entry = tuple[str, str]
"""``(group, sequence)`` -- what a feature run, a tracks table or a tracker covers."""

type CameraEntry = tuple[str, str, str]
"""``(group, sequence, camera)`` -- what a frame run covers.

The camera axis is part of the key rather than a detail: the cameras of one
recording share a ``(group, sequence)``, so without it a run that extracted one
camera would read as covering the entry and the other camera would never be seen
as missing.
"""

ArtifactKind = Literal[
    "feature",
    "tracks-variant",
    "labels-variant",
    "tracker-run",
    "frame-run",
    "trained-model",
    "media-derivative",
]
"""Every kind of artifact a dataset can hold, named once."""

ArtifactStatus = Literal[
    "absent",
    "partial",
    "complete",
    "complete-but-drifted",
    "inconsistent",
]
"""What an artifact's coverage and consistency add up to. See :func:`classify`."""

Target = Literal["analysis", "playback"]
"""Which transcode a media derivative serves."""


@dataclass(frozen=True, slots=True)
class Coverage[KeyT]:
    """Which of the wanted keys an artifact actually holds.

    ``covered`` and ``missing`` are derived rather than stored, so the three
    cannot drift apart. ``present`` is stored rather than only ``covered``,
    because a run holding entries nobody asked for is real information -- it is
    what a widened scope will find already computed -- and because the
    empty-target case has to mean "anything at all", which is inexpressible from
    ``covered`` alone.
    """

    target: frozenset[KeyT]
    present: frozenset[KeyT]
    covers_all: bool = False
    """A single artifact answering for every key, whatever the target holds.

    What a global fit's ``__global__`` marker means: one output, not one per
    entry, so asking which entries it covers is the wrong question.
    """

    @property
    def covered(self) -> frozenset[KeyT]:
        """The wanted keys this artifact holds."""
        return self.present & self.target

    @property
    def missing(self) -> frozenset[KeyT]:
        """The wanted keys it does not."""
        return self.target - self.present

    @property
    def is_satisfied(self) -> bool:
        """Does this artifact answer for everything that was wanted?

        An empty target means nothing was asked for, and the honest answer is
        then "whatever is here" -- which is the rule the completeness predicate
        this replaced already applied, preserved rather than tidied away.
        """
        if self.covers_all:
            return True
        if not self.target:
            return bool(self.present)
        return not self.missing


# --- refs: the per-kind key an artifact is looked up by ----------------------


@dataclass(frozen=True, slots=True)
class FeatureRunRef:
    """One feature run: ``features/<name>/<run_id>/``."""

    kind: ClassVar[ArtifactKind] = "feature"
    name: str
    run_id: str


@dataclass(frozen=True, slots=True)
class TracksVariantRef:
    """One tracks recipe: ``tracks/<variant>/``. ``""`` names the unlabelled tables."""

    kind: ClassVar[ArtifactKind] = "tracks-variant"
    run_id: str


@dataclass(frozen=True, slots=True)
class LabelsVariantRef:
    """One converted-label variant: ``labels/<kind>/<run_id>/``."""

    kind: ClassVar[ArtifactKind] = "labels-variant"
    label_kind: str
    run_id: str


@dataclass(frozen=True, slots=True)
class TrackerRunRef:
    """One tracker or inference run under ``_tracking/<root_key>/<run_id>/``."""

    kind: ClassVar[ArtifactKind] = "tracker-run"
    root_key: str
    run_id: str


@dataclass(frozen=True, slots=True)
class FrameRunRef:
    """One frame-extraction run: ``frames/<method>/<run_id>/``."""

    kind: ClassVar[ArtifactKind] = "frame-run"
    method: str
    run_id: str


@dataclass(frozen=True, slots=True)
class TrainedModelRef:
    """One trained model: ``models/<op_kind>/<run_id>/``."""

    kind: ClassVar[ArtifactKind] = "trained-model"
    op_kind: str
    run_id: str


@dataclass(frozen=True, slots=True)
class MediaDerivativeRef:
    """Every media row's derivative for one target.

    The one ref carrying no run identifier, because transcode mints none that
    addresses anything: its output is named by recipe and reuse is gated by the
    filename plus the forward link on the source row. There is one of these per
    target per dataset, not one per run.
    """

    kind: ClassVar[ArtifactKind] = "media-derivative"
    target: Target


type ArtifactRef = (
    FeatureRunRef
    | TracksVariantRef
    | LabelsVariantRef
    | TrackerRunRef
    | FrameRunRef
    | TrainedModelRef
    | MediaDerivativeRef
)


# --- the record ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArtifactRecord[KeyT]:
    """Everything the inventory knows about one artifact.

    Generic in the coverage key so the per-kind difference is in the type rather
    than in a comment. ``orphan_rows`` and ``orphan_files`` are kept apart
    because they mean opposite things: a row with no file is damage, a file with
    no row is the ordinary state of a run still writing.
    """

    ref: ArtifactRef
    name: str
    """A human-facing name: the feature storage, the label kind, the op kind."""
    run_id: str
    """Empty for a media derivative, which has no addressing run identifier."""
    coverage: Coverage[KeyT]
    status: ArtifactStatus
    run_root: Path | None = None
    """Where the outputs live, or ``None`` for a kind with no run directory."""
    index_path: Path | None = None
    params: RunParams | None = None
    params_state: ParamsState = "absent"
    rows: frozenset[KeyT] = frozenset()
    """What the index says this run holds, as against what disk holds."""
    orphan_rows: frozenset[KeyT] = frozenset()
    orphan_files: frozenset[KeyT] = frozenset()
    drift: tuple[Entry, ...] = ()
    """Entries whose recorded source composition no longer matches the present.

    Complete and loadable, but superseded. Reported rather than refused: a
    drifted run is exactly what a comparison across revisions needs.
    """
    identity_scheme: str = ""
    started_at: str = ""
    finished_at: str = ""
    upstreams: tuple[str, ...] = ()
    extra: dict[str, frozenset[str]] = field(default_factory=dict)
    """Per-kind detail with nowhere else to live, e.g. transcode's two remedies.

    Deliberately narrow and deliberately named as an exception: anything a second
    kind needs becomes a field.
    """


type EntryRecord = ArtifactRecord[Entry]
type CameraRecord = ArtifactRecord[CameraEntry]
type UnitRecord = ArtifactRecord[str]
type AnyRecord = EntryRecord | CameraRecord | UnitRecord


# --- status, decided in exactly one place -------------------------------------


def classify(
    *,
    satisfied: bool,
    any_covered: bool,
    orphan_rows: bool,
    orphan_files: bool,
    drifted: bool,
    finished: bool,
) -> ArtifactStatus:
    """The one place an artifact's status is decided.

    Takes facts rather than the record, so the rule is readable as a rule and
    every caller reaches the same verdict from the same evidence.

    Precedence, highest first:

    1. ``inconsistent`` -- the index and the files disagree. **Except** while the
       run is unfinished: outputs are written before their index rows, so files
       ahead of rows is what a run in progress looks like, and calling that
       damage would make every live run red.
    2. ``complete-but-drifted`` -- satisfied, but a source moved underneath it.
       Superseded is not invalid.
    3. ``complete``.
    4. ``partial`` -- some of it is here and some is not. Its own value rather
       than folded into ``absent``, because "nothing has run" and "89 of 90" call
       for different actions and reporting the second as the first is a lie a
       coverage bar cannot recover from.
    5. ``absent``.

    Args:
        satisfied: Whether coverage answers for everything wanted.
        any_covered: Whether it answers for anything wanted.
        orphan_rows: Whether the index names entries disk does not hold.
        orphan_files: Whether disk holds entries the index does not name.
        drifted: Whether any recorded source composition has moved.
        finished: Whether the run recorded a finish. An unfinished run is
            expected to have files ahead of rows.

    Returns:
        The status, which is never stored anywhere.
    """
    if orphan_rows or (orphan_files and finished):
        return "inconsistent"
    if satisfied and drifted:
        return "complete-but-drifted"
    if satisfied:
        return "complete"
    if any_covered:
        return "partial"
    return "absent"


# --- the inventory ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InventoryScope:
    """What an inventory was asked about, carried so its answers can be read.

    ``entries`` narrows what counts as wanted; ``None`` means the whole dataset.
    ``tracks_run_id`` decides which tracks variant defines the entry universe,
    and it is not decoration: measuring a variant-pinned run against every entry
    in the index makes it read permanently incomplete, which is the same shape of
    wrong answer the transcode case is.
    """

    kinds: frozenset[ArtifactKind]
    entries: frozenset[Entry] | None = None
    tracks_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetInventory:
    """What one dataset holds, as read at one moment.

    A cache and never a source of truth: the ``index.csv`` files and the files
    themselves are that, and this is a view over them. Stale is safe here -- a
    stale view causes redundant or delayed work, never wrong work -- which is why
    it is never written anywhere it could be mistaken for the record.
    """

    dataset_root: Path
    scope: InventoryScope
    records: tuple[AnyRecord, ...] = ()
    unavailable_kinds: frozenset[ArtifactKind] = frozenset()
    """Kinds that were asked for and have no contributor registered.

    Never silently empty and never an error: a caller importing only
    ``mosaic.core`` has not imported the modules that register the ops half, and
    reporting zero tracker runs would be a wrong answer rather than a missing one.
    """
    errors: tuple[str, ...] = ()

    def of_kind(self, kind: ArtifactKind) -> tuple[AnyRecord, ...]:
        """Every record of one kind, in the order they were scanned."""
        return tuple(record for record in self.records if record.ref.kind == kind)

    def record(self, ref: ArtifactRef) -> AnyRecord | None:
        """The record for *ref*, or ``None`` when the dataset holds no such artifact."""
        for record in self.records:
            if record.ref == ref:
                return record
        return None

    @overload
    def coverage(
        self,
        ref: FeatureRunRef | TracksVariantRef | LabelsVariantRef | TrackerRunRef,
    ) -> Coverage[Entry]: ...

    @overload
    def coverage(self, ref: FrameRunRef) -> Coverage[CameraEntry]: ...

    @overload
    def coverage(self, ref: TrainedModelRef | MediaDerivativeRef) -> Coverage[str]: ...

    def coverage(
        self, ref: ArtifactRef
    ) -> Coverage[Entry] | Coverage[CameraEntry] | Coverage[str]:
        """What *ref* covers. **Never raises.**

        An artifact the dataset does not hold answers with empty coverage rather
        than a ``KeyError``, which is what makes absence something a caller can
        act on instead of something it has to guard every lookup against.
        """
        found = self.record(ref)
        if found is None:
            return Coverage[Entry](target=frozenset(), present=frozenset())
        return found.coverage

    def status(self, ref: ArtifactRef) -> ArtifactStatus:
        """*ref*'s status, ``absent`` when the dataset holds no such artifact."""
        found = self.record(ref)
        return "absent" if found is None else found.status
