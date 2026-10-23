"""Plan the runs that an op needs before it reads its entries, and refuse the rest.

:func:`plan_prerequisites` puts each entry through the checks that the op's
run makes before its tool reads the entry. Where a run clears a failed
check, the plan answers the check with that run: ``export-joined`` when the
op reads clips as their join, ``export-store`` when it reads an imgstore
through its export, and ``preprocess`` when a media variant lacks a current
file of the entry. Any other failed check is the entry's refusal, with its
cause and a reason naming the remedy where one exists.

The plan reads the media index, the stores' metadata, the media variants' rows
and recipes, and the files on disk with their headers. It does not write a file
or decode a frame. One entry's failure refuses that entry alone.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from mosaic_media import MediaProbeError
from mosaic_media.transcode import TranscodeError
from pydantic import ValidationError

from mosaic.core.entry import Entry
from mosaic.core.helpers import is_nameless_entry
from mosaic.core.media.facts_columns import AnalysisDerivativeMissingError
from mosaic.core.params import Params
from mosaic.core.pipeline.consumed_camera import (
    ResolvedEntryMedia,
    consumed_cameras,
    first_cameras,
    resolve_entry_media,
)
from mosaic.core.pipeline.joined_export import (
    EntryJoinMissingError,
    JoinedExportParams,
    join_frame_total,
    join_needs_reencode,
    join_to_read,
    joined_run_id,
)
from mosaic.core.pipeline.media_input import MediaInputParams
from mosaic.core.pipeline.ops import OPS
from mosaic.core.pipeline.preprocess import (
    PreprocessParams,
    PreprocessRefused,
    preprocess_entry_failures,
    preprocess_identity,
)
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantDriftedError,
    MediaVariantMissingError,
)
from mosaic.core.pipeline.preprocess_layout import (
    PREPROCESS_KIND,
    media_variant_recipe_path,
)
from mosaic.core.pipeline.store_export import (
    StoreExportParams,
    entry_stores,
    export_run_id,
    unnamed_stores,
)
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS, tracking_root
from mosaic.core.pipeline.variant_source import VariantLookup
from mosaic.core.scope import Scope
from mosaic.tracking.common.scope import (
    JoinedSourceMismatchError,
    refuse_unjoinable,
    run_entries,
)
from mosaic.tracking.common.tool_input import (
    StoreExportMissingError,
    codecs_never_read,
    handed_codecs,
    refuse_unreadable_entry,
)
from mosaic.tracking.registration import register_ops

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset, ResolvedScopeEntry

__all__ = [
    "EntryRefusal",
    "PrerequisiteKind",
    "PrerequisitePlan",
    "PrerequisiteRun",
    "RefusalCause",
    "indexed_entries",
    "plan_prerequisites",
]

type PrerequisiteKind = Literal["export-joined", "export-store", "preprocess"]
"""The ops that a plan runs before the op it is made for."""

type RefusalCause = Literal[
    "no_media",
    "needs_analysis_transcode",
    "unresolved_media",
    "needs_media_reprobe",
    "several_joins",
    "unidentified_clips",
    "unjoinable",
    "multi_camera_join",
    "multi_camera_variant",
    "no_media_raw",
    "preprocess_refused",
    "no_variant_recipe",
    "undecodable_codec",
]
"""Why the planned runs leave an entry unreadable. A closed set.

- ``no_media``: the entry's media is not indexed, or another entry with the same
  entry key is indexed first and takes its output name.
- ``needs_analysis_transcode``: its media requires an analysis transcode that has
  not run.
- ``unresolved_media``: its media or a store's metadata cannot be read.
- ``needs_media_reprobe``: the clips do not report frame counts, or a store lacks
  a ``video_uuid``. ``reprobe-media --apply`` measures them.
- ``several_joins``: two current joins of the clips exist, and the user keeps one.
- ``unidentified_clips``: a clip lacks a content identity. A join of the clips
  cannot be addressed without it.
- ``unjoinable``: the clips cannot be read as one video.
- ``multi_camera_join``: a camera of an entry of several cameras needs a join, and
  ``export-joined`` joins an entry of one camera.
- ``multi_camera_variant``: a camera of a nameless entry lacks a current file of
  the media variant, and a run of the variant writes another camera of the entry.
- ``no_media_raw``: a store needs an export, and the dataset does not declare a
  ``media_raw`` root to index one in.
- ``preprocess_refused``: the variant's recipe does not fit the entry.
- ``no_variant_recipe``: the named variant lacks the entry, and its runs do not
  record a recipe that makes it.
- ``undecodable_codec``: the named variant is in a codec that the tool cannot
  decode in any environment.
"""

_EXTRACT_FRAMES: Final = "extract-frames"
"""The op outside the tracking roots that a plan answers for over every camera."""

_ONE_CAMERA_REMEDY: Final = (
    "Index the entry's media under a sequence name. A tool then reads one camera "
    "of the entry."
)
"""The remedy for a nameless entry's camera that the planned runs leave unreadable."""


@dataclass(frozen=True, slots=True)
class PrerequisiteRun:
    """One run that must finish before the op reads its entries.

    Attributes:
        kind: The op to run.
        params: Its params.
        entries: Its scope, as the ``(group, sequence)`` pairs of the media
            index. ``mosaic run`` does not take a camera scope. A run therefore
            covers every camera of an entry. A nameless entry is named here by
            its indexed key, and elsewhere in an unscoped plan by its label.
        run_id: The run id that the run is planned to record. Export run ids
            name the recipe alone and repeat across entries.
        after: The positions in :attr:`PrerequisitePlan.runs` of the runs that
            must finish first.
    """

    kind: PrerequisiteKind
    params: JoinedExportParams | StoreExportParams | PreprocessParams
    entries: tuple[Entry, ...]
    run_id: str
    after: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EntryRefusal:
    """One entry that the op cannot read once the planned runs finish.

    Attributes:
        group: The entry's group.
        sequence: The entry's sequence.
        camera: The camera refused, or ``""`` when the entry's media does not
            resolve.
        cause: Why, as a member of a closed set that a caller can branch on.
        reason: The refusal's message, naming the remedy where one exists.
    """

    group: str
    sequence: str
    camera: str
    cause: RefusalCause
    reason: str


@dataclass(frozen=True, slots=True)
class PrerequisitePlan:
    """What an op needs before it reads its entries.

    Attributes:
        kind: The op that the plan is made for.
        media: The ``media`` that the op must run with, ``""`` for the entry
            media.
        runs: The runs to make first, in dependency order.
        refusals: The entries that the op cannot read, sorted by group,
            sequence, camera and cause.
        entries: The op's scope once the runs finish, sorted. It contains each
            entry without a refusal.
    """

    kind: str
    media: str
    runs: tuple[PrerequisiteRun, ...]
    refusals: tuple[EntryRefusal, ...]
    entries: tuple[Entry, ...]


@dataclass(frozen=True, slots=True)
class _CameraRead:
    """One camera that the op reads, named as the plan and as a run's scope name it.

    Attributes:
        entry: The entry as the plan names it.
        camera: The camera, resolved.
        run_entry: The entry as a planned run's scope names it.
        several_cameras: Whether *run_entry* has more than one camera.
    """

    entry: Entry
    camera: ResolvedScopeEntry
    run_entry: Entry
    several_cameras: bool


@dataclass(slots=True)
class _Draft:
    """The runs and refusals of a plan, in the order they are found."""

    runs: list[PrerequisiteRun] = field(default_factory=list[PrerequisiteRun])
    refusals: list[EntryRefusal] = field(default_factory=list[EntryRefusal])


def plan_prerequisites(
    ds: Dataset,
    kind: str,
    params: Params | Mapping[str, object] | None = None,
    *,
    entries: Iterable[Entry] | None = None,
) -> PrerequisitePlan:
    """Return the runs that *kind* needs before it reads *entries*, and the refusals.

    A tracker or an ``infer-*`` op is answered for each camera that its run
    reads, the cameras of every entry reducing together to one per entry key as
    its run reduces them. The run labels each camera of a nameless entry by the
    camera's first file's stem, and reads each. ``preprocess`` is answered for
    the one camera of each entry that it writes, and ``extract-frames`` for
    every camera. An unscoped plan answers ``preprocess`` for every camera of a
    nameless entry, each under its label. The run writes one of them.

    An op that reads a media variant needs one ``preprocess`` run of the
    variant's recorded recipe, over each entry whose file of the variant is
    missing or out of date. Each such entry is put through the checks that the
    run makes before it decodes one. The run writes one camera of each entry.
    Each other camera of a nameless entry without a current file of the variant
    is refused. A chained variant's run comes after the run of its upstream, for
    the entries whose file of the upstream is missing or out of date. A tool
    reads the variant that its ``media`` names. When the plan cannot find a
    recipe that makes a variant, each entry without a current file of the
    variant is refused. A codec that a tool cannot decode in any environment is
    undecodable to it. Every entry is refused a variant in an undecodable codec.
    When a tool reading the entry media is handed an undecodable codec for some
    cameras, it reads the H.264 variant of the entry media instead,
    ``PreprocessParams(steps=[], codec="h264")``, provided the variant's run
    writes at least one of those cameras and fits its entry, or one of them has
    a current file of the variant. Otherwise each of those cameras is refused.
    ``preprocess`` is put through the checks of the recipe in its params.

    An op reading the entry media is put through the checks that its run makes
    before it reads the entry:

    * clips that the op reads as their join, without a current one, need an
      ``export-joined`` run, with ``reencode`` set when the clips differ in
      stream profile;
    * a store that the op reads through its export, without one, needs an
      ``export-store`` run, which exports every camera of the entry.

    An export that its run would refuse is the entry's refusal instead: a join
    of a camera of an entry with several cameras, a join of clips that report no
    frame counts, an export on a dataset without a ``media_raw`` root, and an
    export of a store without a ``video_uuid``. Every other check that fails is
    the entry's refusal, as is an entry whose media does not resolve. One run is
    planned per op and entry, and one per variant. The plan is the same whatever
    the order of *entries*.

    The plan does not probe a file, and a run's probe can still refuse an entry
    that the plan passes. An ``entry-files`` runner refuses a store without an
    export when a probe measures one of its chunks at another frame count than
    the store's index gives. SLEAP refuses a file that its decode probe does not
    decode, such as AV1 where the OpenCV of its environment does not link dav1d.

    Args:
        ds: The dataset, which the plan reads and leaves unchanged.
        kind: The op to plan for: a tracking op that reads entry media,
            ``extract-frames`` or ``preprocess``.
        params: The op's params, or ``None``. A tracking op's ``media`` is read
            from them. ``extract-frames`` and ``preprocess`` read all of their
            params, which must validate. The listed ``frames`` of
            ``extract-frames`` narrow the entries.
        entries: The entries to plan for, or ``None`` for every indexed entry,
            each named by its label as an unscoped run names it. A run planned
            for a nameless entry is still scoped by its indexed key, because no
            run's scope names the label.

    Returns:
        The plan, with the ``media`` that the op must run with.

    Raises:
        ValueError: If *kind* is not an op this answers for, or ``media`` is not
            a string.
        pydantic.ValidationError: If ``extract-frames`` or ``preprocess`` params
            do not validate.
        ListedFramesRefused: If ``extract-frames`` lists frames of none of
            *entries*.
    """
    register_ops()
    answered = sorted(
        {key for key in TRACKING_ROOTS if key in OPS}
        | {_EXTRACT_FRAMES, PREPROCESS_KIND}
    )
    if kind not in answered:
        message = (
            f"plan_prerequisites answers for {', '.join(answered)}, and {kind!r} "
            f"is none of them."
        )
        raise ValueError(message)
    scope = None if entries is None else sorted(set(entries))
    recipe: PreprocessParams | None = None
    media = ""
    if kind == _EXTRACT_FRAMES:
        scope = _listed_scope(params, scope)
    elif kind == PREPROCESS_KIND:
        recipe = _validated(PreprocessParams, params)
        media = recipe.media
    else:
        media = _requested_media(params)

    resolved = resolve_entry_media(ds, scope)
    draft = _Draft()
    reads = _camera_reads(ds, kind, resolved, draft)
    if recipe is not None:
        _ = _fit(ds, kind, recipe, reads, draft)
    elif kind == _EXTRACT_FRAMES:
        _read_entry_media(ds, kind, reads, draft)
    else:
        if not media:
            media, reads = _codec_variant(ds, kind, reads, draft)
        if media:
            _read_variant(ds, kind, media, reads, draft)
        else:
            _read_entry_media(ds, kind, reads, draft)

    refused = {(refusal.group, refusal.sequence) for refusal in draft.refusals}
    return PrerequisitePlan(
        kind=kind,
        media=media,
        runs=tuple(draft.runs),
        refusals=tuple(sorted(draft.refusals, key=_refusal_order)),
        entries=tuple(sorted(entry for entry in resolved if entry not in refused)),
    )


def _validated[P: Params](
    model: type[P], params: Params | Mapping[str, object] | None
) -> P:
    """Return *params* validated as *model*.

    Raises:
        pydantic.ValidationError: If *params* do not validate.
    """
    return model.model_validate(
        params if isinstance(params, Params) else dict(params or {})
    )


def _requested_media(params: Params | Mapping[str, object] | None) -> str:
    """Return the ``media`` of a tracking op's *params*, ``""`` when unset.

    Raises:
        ValueError: If a mapping's ``media`` is not a string.
    """
    if isinstance(params, MediaInputParams):
        return params.media
    if params is None or isinstance(params, Params):
        return ""
    media = params.get("media", "")
    if not isinstance(media, str):
        message = f"media names a media variant by its run id, and {media!r} is not one"
        raise ValueError(message)
    return media


def _listed_scope(
    params: Params | Mapping[str, object] | None, scope: list[Entry] | None
) -> list[Entry] | None:
    """Return *scope* narrowed to the entries that ``extract-frames`` lists frames of.

    Raises:
        pydantic.ValidationError: If *params* do not validate.
        ListedFramesRefused: If the list names none of *scope*.
    """
    # Local: importing this module leaves frame extraction unloaded, and
    # `register_ops` loads it before a plan reads it.
    from mosaic.tracking.frame_extraction.dataset_runs import (
        ExtractFramesParams,
        listed_entries,
        listed_frames,
    )

    listed = listed_frames(_validated(ExtractFramesParams, params))
    if listed is None:
        return scope
    return listed_entries(listed, scope)[0]


def _camera_reads(
    ds: Dataset,
    kind: str,
    resolved: Mapping[Entry, ResolvedEntryMedia],
    draft: _Draft,
) -> list[_CameraRead]:
    """Return each camera that *kind* reads, and refuse each entry left without one.

    A read's run entry is the entry that a run's scope names for its camera
    (:func:`~mosaic.tracking.common.scope.run_entries`). An unscoped plan labels
    each camera of a nameless entry by its first file's stem, and a run's scope
    names the entry's indexed key, which covers every camera resolved under it.

    Args:
        ds: The dataset.
        kind: The op that reads the cameras.
        resolved: Each entry's cameras, or why it has none.
        draft: The plan that each refusal is added to.
    """
    cameras = [
        (entry, camera)
        for entry, media in resolved.items()
        if isinstance(media, list)
        for camera in media
    ]
    run_entry_of = {
        camera.resolved.paths[0]: run_entry
        for (_, camera), run_entry in zip(
            cameras, run_entries(ds, cameras), strict=True
        )
    }
    camera_counts = Counter(run_entry_of.values())
    reads: list[_CameraRead] = []
    for entry, read in _cameras_read(kind, resolved).items():
        if not isinstance(read, list):
            draft.refusals.append(_unresolved(entry, read))
            continue
        for camera in read:
            run_entry = run_entry_of[camera.resolved.paths[0]]
            several = camera_counts[run_entry] > 1
            reads.append(_CameraRead(entry, camera, run_entry, several))
    return reads


def indexed_entries(
    ds: Dataset, entries: Iterable[Entry]
) -> tuple[tuple[Entry, ...], tuple[Entry, ...]]:
    """Return *entries* as a run's scope names them, and the labels that it cannot name.

    An unscoped plan names an entry by its label, and labels each camera of a
    nameless entry by its first file's stem. A run's scope names the entry by
    its indexed key instead, and covers every camera under that key. A nameless
    entry's key is named when *entries* contains the label of each of its
    cameras. Otherwise the scope cannot name a camera's label that is no indexed
    key. Every other entry is named as given.

    Args:
        ds: The dataset.
        entries: The entries, such as the :attr:`PrerequisitePlan.entries` of an
            unscoped plan.

    Returns:
        The entries that ``mosaic run --entries`` resolves, and the labels of
        *entries* that it cannot name, each sorted.
    """
    given = set(entries)
    ungrouped, nameless = _nameless_cameras(ds)
    labels = {
        entry: {(camera.group, camera.sequence) for camera in media}
        for entry, media in nameless.items()
    }
    labeled = {label for named in labels.values() for label in named}
    keys = {entry for entry, named in labels.items() if named <= given}
    covered = {label for key in keys for label in labels[key]}
    as_given = {entry for entry in given if entry not in labeled or entry in ungrouped}
    unnamed = given - covered - as_given
    return tuple(sorted(keys | as_given)), tuple(sorted(unnamed))


def _nameless_cameras(
    ds: Dataset,
) -> tuple[set[Entry], dict[Entry, list[ResolvedScopeEntry]]]:
    """Return the entries indexed in the empty group, and each nameless one's cameras.

    A nameless entry whose media does not resolve is left out of the cameras.
    """
    try:
        indexed = ds.resolve_scope(Scope(groups=[""])).entries
    except FileNotFoundError:
        return set(), {}
    nameless = sorted(entry for entry in indexed if is_nameless_entry(*entry))
    if not nameless:
        return indexed, {}
    return indexed, {
        entry: media
        for entry, media in resolve_entry_media(ds, nameless).items()
        if isinstance(media, list)
    }


def _cameras_read(
    kind: str, resolved: Mapping[Entry, ResolvedEntryMedia]
) -> dict[Entry, ResolvedEntryMedia]:
    """Return the cameras that *kind* reads of each entry, or why it reads none.

    ``extract-frames`` reads every camera. Any other op reads the cameras that
    its run keeps of every entry, one per entry key, and ``preprocess`` reads
    the first camera that it keeps of each entry.
    """
    if kind == _EXTRACT_FRAMES:
        return dict(resolved)
    consumed = consumed_cameras(kind, resolved, report_skipped=False)
    if kind != PREPROCESS_KIND:
        return consumed
    return {
        entry: media
        if isinstance(media, (FileNotFoundError, MediaProbeError))
        else [media]
        for entry, media in first_cameras(consumed).items()
    }


def _scoped_entries(reads: Iterable[_CameraRead]) -> list[Entry]:
    """Return the entries that a run over *reads* is scoped by, sorted."""
    return sorted({read.run_entry for read in reads})


# --- the entry media ------------------------------------------------------------


def _read_entry_media(
    ds: Dataset, kind: str, reads: Sequence[_CameraRead], draft: _Draft
) -> None:
    """Plan the joins and store exports that let *kind* read the entry media.

    Each read is answered by :func:`_answer`. One run is planned per op and
    entry, and the runs are ordered by op, then by entry.
    """
    runs: dict[tuple[PrerequisiteKind, Entry], PrerequisiteRun] = {}
    for read in reads:
        answer = _answer(ds, kind, read)
        if isinstance(answer, EntryRefusal):
            draft.refusals.append(answer)
        elif answer is not None:
            _ = runs.setdefault((answer.kind, read.run_entry), answer)
    draft.runs.extend(sorted(runs.values(), key=_run_order))


def _answer(
    ds: Dataset, kind: str, read: _CameraRead
) -> PrerequisiteRun | EntryRefusal | None:
    """Return the run that lets *kind* read the camera of *read*, or its refusal.

    Returns ``None`` when *kind* reads the camera as it is.
    """
    entry, camera = read.entry, read.camera
    try:
        if kind == _EXTRACT_FRAMES:
            refuse_unjoinable(
                ds,
                kind,
                camera.group,
                camera.sequence,
                camera.resolved.paths,
                camera.resolved.facts,
                hands_over_path=False,
            )
            _ = join_to_read(ds, camera, asker=kind)
        else:
            refuse_unreadable_entry(ds, camera, kind=kind)
    except JoinedSourceMismatchError as refusal:
        return _refusal(entry, camera, "unjoinable", str(refusal))
    except EntryJoinMissingError as refusal:
        match refusal.cause:
            case "no_join" | "superseded_join":
                return _join_run(kind, read)
            case "several_joins" | "unidentified_clips" as cause:
                return _refusal(entry, camera, cause, str(refusal))
    except StoreExportMissingError:
        return _store_export_run(ds, kind, read)
    except MediaProbeError as refusal:
        return _refusal(entry, camera, "unresolved_media", str(refusal))
    return None


def _join_run(kind: str, read: _CameraRead) -> PrerequisiteRun | EntryRefusal:
    """Return the ``export-joined`` run of the camera's clips, or why its run fails.

    ``export-joined`` resolves the run's entry without a camera, which refuses an
    entry of several cameras, and checks the join against the clips' summed
    frame counts.
    """
    entry, camera = read.entry, read.camera
    label = _label(kind, entry)
    if read.several_cameras:
        # The op of a tracking root takes `media`, and `extract-frames` does not.
        # A run of a media variant writes one camera of a nameless entry.
        remedy = ""
        if kind in TRACKING_ROOTS:
            remedy = (
                f" {_ONE_CAMERA_REMEDY} Read that camera through a media variant."
                if is_nameless_entry(*read.run_entry)
                else " Read the camera through a media variant instead."
            )
        reason = (
            f"{label} camera {camera.camera} is read as the join of its clips, "
            f"and export-joined joins the clips of an entry of one camera.{remedy}"
        )
        return _refusal(entry, camera, "multi_camera_join", reason)
    facts = camera.resolved.facts
    if join_frame_total(facts) <= 0:
        reason = (
            f"{label} is read as the join of its clips, and the clips report no "
            f"frame counts to check a join against. Run 'mosaic reprobe-media "
            f"--apply' to measure them."
        )
        return _refusal(entry, camera, "needs_media_reprobe", reason)
    params = JoinedExportParams(reencode=join_needs_reencode(facts))
    run_id = joined_run_id(params)
    return PrerequisiteRun("export-joined", params, (read.run_entry,), run_id, ())


def _store_export_run(
    ds: Dataset, kind: str, read: _CameraRead
) -> PrerequisiteRun | EntryRefusal:
    """Return the ``export-store`` run of the read's entry, or why its run fails.

    The run exports the store of every camera of the entry, each named by its
    ``video_uuid``, and indexes the exports beside a ``media_raw`` root.
    """
    entry, camera = read.entry, read.camera
    label = _label(kind, entry)
    if not ds.routes_derivatives():
        reason = (
            f"{label} is read through the export of its imgstore, and this "
            f"dataset does not declare a media_raw root. Its originals index is "
            f"media/index.csv, where an export is not read."
        )
        return _refusal(entry, camera, "no_media_raw", reason)
    try:
        unnamed = unnamed_stores(entry_stores(ds, *read.run_entry))
    except FileNotFoundError as refusal:
        return _refusal(entry, camera, "unresolved_media", str(refusal))
    if unnamed:
        reason = (
            f"{label} is read through the export of its imgstore, and the media "
            f"index does not record a video_uuid for "
            f"{', '.join(store.name for store in unnamed)} to name an export by. "
            f"Run 'mosaic reprobe-media --apply' first."
        )
        return _refusal(entry, camera, "needs_media_reprobe", reason)
    params = StoreExportParams()
    run_id = export_run_id(params)
    return PrerequisiteRun("export-store", params, (read.run_entry,), run_id, ())


# --- media variants -------------------------------------------------------------


def _codec_recipe() -> PreprocessParams:
    """Return the recipe of the variant that re-encodes the entry media in H.264.

    Each call returns a new instance, because a plan's runs contain their params.
    """
    return PreprocessParams(steps=[], codec="h264")


def _codec_variant(
    ds: Dataset, kind: str, reads: Sequence[_CameraRead], draft: _Draft
) -> tuple[str, list[_CameraRead]]:
    """Return the H.264 variant when *kind* reads it, and the reads still to answer.

    A codec that the tool of *kind* cannot decode in any environment is
    undecodable. *kind* reads the variant when its tool is handed an undecodable
    codec for at least one camera that qualifies. A camera qualifies when the
    variant's run writes and accepts it, or when it has a current file of the
    variant. The variant's files are looked up only when its run writes and
    accepts none of those cameras. ``media`` names one variant for the whole
    run. Every camera then reads it. When none of those cameras qualifies, a
    camera that the run does not write is refused with
    ``multi_camera_variant``, and the rest with the run's failure. A camera
    whose handed codec cannot be read is refused too. A refused camera is not
    answered further.

    Returns:
        The variant's run id, or ``""`` for the entry media, and the reads that
        are not refused.
    """
    never = codecs_never_read(tracking_root(kind).decoder)
    if not never:
        return "", list(reads)
    readable: list[_CameraRead] = []
    unread: list[tuple[_CameraRead, frozenset[str]]] = []
    for read in reads:
        try:
            codecs = handed_codecs(ds, read.camera, kind=kind)
        except (MediaProbeError, TranscodeError) as error:
            refusal = _refusal(read.entry, read.camera, "unresolved_media", str(error))
            draft.refusals.append(refusal)
            continue
        if codecs & never:
            unread.append((read, codecs & never))
        else:
            readable.append(read)
    if not unread:
        return "", readable
    recipe = _codec_recipe()
    variant = preprocess_identity(recipe).run_id
    unread_reads = [read for read, _ in unread]
    failures = preprocess_entry_failures(ds, recipe, _scoped_entries(unread_reads))
    written = _written_cameras(ds, unread_reads)
    run_writes = any(
        read.run_entry not in failures and _other_camera_written(written, read) is None
        for read in unread_reads
    )
    if run_writes or len(_lacking_reads(ds, variant, unread_reads)[0]) < len(unread):
        return variant, [*readable, *unread_reads]
    for read, codecs in unread:
        handed = ", ".join(sorted(codecs))
        other = _other_camera_written(written, read)
        if other is not None:
            draft.refusals.append(_unwritten(kind, variant, read, other, handed))
            continue
        context = (
            f"{_label(kind, read.entry)} is handed {handed}, which {kind} never "
            f"reads, and the H.264 variant of its media cannot be written. "
        )
        draft.refusals.append(_unfit(read, failures[read.run_entry], context))
    return "", readable


def _read_variant(
    ds: Dataset, kind: str, variant: str, reads: Sequence[_CameraRead], draft: _Draft
) -> None:
    """Plan the runs that write *variant* for each read without a current file.

    The tool reads the variant's file, without a join or an export. The
    variant's codec is its recipe's, or else that of its current files. Every
    read is refused when the tool cannot decode that codec in any environment.
    Otherwise a read without a current file is refused when the variant's run
    does not write its camera (:func:`_written_cameras`).
    """
    lacking, codecs = _lacking_reads(ds, variant, reads)
    recipe = _variant_recipe(ds, variant)
    if isinstance(recipe, PreprocessParams):
        codecs = {recipe.codec}
    decoder = tracking_root(kind).decoder
    undecodable = ", ".join(sorted(codecs & codecs_never_read(decoder)))
    if not undecodable:
        written = _written_cameras(ds, lacking)
        covered: list[_CameraRead] = []
        for read in lacking:
            other = _other_camera_written(written, read)
            if other is None:
                covered.append(read)
            else:
                draft.refusals.append(_unwritten(kind, variant, read, other))
        _ = _variant_run(ds, kind, variant, recipe, covered, draft)
        return
    remedy = f" {decoder.remedy}." if decoder.remedy else ""
    for read in reads:
        reason = (
            f"{_label(kind, read.entry)} reads the media variant {variant}, which "
            f"is {undecodable}. {kind} decodes with {decoder.stack}. That reader "
            f"decodes {undecodable} in no environment.{remedy}"
        )
        draft.refusals.append(
            _refusal(read.entry, read.camera, "undecodable_codec", reason)
        )


def _lacking_reads(
    ds: Dataset, variant: str, reads: Sequence[_CameraRead]
) -> tuple[list[_CameraRead], set[str]]:
    """Look up the file of *variant* for each read's camera, in one read of its index.

    Returns:
        The reads without a current file, and the codecs of the current files.
    """
    lookup = VariantLookup.read(
        ds,
        variant,
        sorted({(read.camera.group, read.camera.sequence) for read in reads}),
    )
    lacking: list[_CameraRead] = []
    codecs: set[str] = set()
    for read in reads:
        try:
            codecs.add(lookup.resolve(ds, read.camera).facts.codec_name)
        except (MediaVariantMissingError, MediaVariantDriftedError):
            lacking.append(read)
    return lacking, codecs


def _written_cameras(
    ds: Dataset, reads: Iterable[_CameraRead]
) -> dict[Entry, ResolvedScopeEntry]:
    """Return the camera that a run of a variant writes of each entry of *reads*.

    A run of a variant writes one camera of each entry in its scope
    (:func:`~mosaic.core.pipeline.consumed_camera.first_cameras`). The result
    is keyed by run entry, and contains each run entry of several cameras whose
    media resolves. An op reads every camera of a nameless entry, which the run
    covers in part.
    """
    several = _scoped_entries(read for read in reads if read.several_cameras)
    if not several:
        return {}
    return {
        entry: media[0]
        for entry, media in _cameras_read(
            PREPROCESS_KIND, resolve_entry_media(ds, several)
        ).items()
        if isinstance(media, list)
    }


def _other_camera_written(
    written: Mapping[Entry, ResolvedScopeEntry], read: _CameraRead
) -> ResolvedScopeEntry | None:
    """Return the camera that a variant's run writes in place of the read's camera.

    Returns ``None`` when the run writes the read's camera, or when *written*
    lacks the read's run entry.

    Args:
        written: What :func:`_written_cameras` returned.
        read: The read.
    """
    camera = written.get(read.run_entry)
    if camera is None or camera.resolved.paths[0] == read.camera.resolved.paths[0]:
        return None
    return camera


def _unwritten(
    kind: str,
    variant: str,
    read: _CameraRead,
    written: ResolvedScopeEntry,
    handed: str = "",
) -> EntryRefusal:
    """Return the refusal of *read*, whose camera a run of *variant* does not write.

    Args:
        kind: The op that reads the camera.
        variant: The variant's run id.
        read: The read refused.
        written: The camera of the entry that the run writes.
        handed: The codecs that the tool is handed for the camera and does not
            read, or ``""`` when the op names *variant* in ``media``.
    """
    because = f", because {kind} does not read {handed}" if handed else ""
    reason = (
        f"{_label(kind, read.entry)} camera {read.camera.camera} needs a file of "
        f"the media variant {variant}{because}. {kind} reads every camera of a "
        f"nameless entry, and a run of the variant writes camera {written.camera} "
        f"of the entry alone. {_ONE_CAMERA_REMEDY}"
    )
    return _refusal(read.entry, read.camera, "multi_camera_variant", reason)


def _variant_recipe(ds: Dataset, variant: str) -> PreprocessParams | str:
    """Return the recipe that makes *variant*, or why *variant* lacks one.

    The recipe is the H.264 recipe when *variant* is its run id, and otherwise
    the one that the variant's runs record. A recorded recipe must still make
    *variant*, which a version bump changes.
    """
    codec_recipe = _codec_recipe()
    if variant == preprocess_identity(codec_recipe).run_id:
        return codec_recipe
    path = media_variant_recipe_path(ds, variant)
    try:
        recorded = path.read_bytes()
    except OSError:
        return f"it does not record a readable recipe at {path}"
    try:
        recipe = PreprocessParams.model_validate_json(recorded)
    except ValidationError as error:
        return f"the recipe that it records at {path} does not validate: {error}"
    made = preprocess_identity(recipe).run_id
    if made != variant:
        return (
            f"the recipe that it records at {path} now makes {made} instead of "
            f"{variant}"
        )
    return recipe


def _variant_run(
    ds: Dataset,
    kind: str,
    variant: str,
    recipe: PreprocessParams | str,
    reads: Sequence[_CameraRead],
    draft: _Draft,
) -> tuple[list[_CameraRead], tuple[int, ...]]:
    """Plan the run that writes *variant* for *reads*, after any run of its upstream.

    Args:
        ds: The dataset.
        kind: The op that the plan is made for, named in a refusal.
        variant: The variant's run id.
        recipe: The variant's recipe, or why the variant lacks one.
        reads: The reads that lack a current file of *variant*.
        draft: The plan that the runs and refusals are added to.

    Returns:
        The reads that the run covers, and its position in the plan's runs.
        Both are empty without a planned run.
    """
    if not reads:
        return [], ()
    if isinstance(recipe, str):
        for read in reads:
            reason = (
                f"{_label(kind, read.entry)} needs a current file of the media "
                f"variant {variant}, and {recipe}. Run preprocess over the entry "
                f"with the parameters that made {variant}."
            )
            draft.refusals.append(
                _refusal(read.entry, read.camera, "no_variant_recipe", reason)
            )
        return [], ()
    covered, after = _fit(ds, kind, recipe, reads, draft)
    if not covered:
        return [], ()
    run = PrerequisiteRun(
        "preprocess", recipe, tuple(_scoped_entries(covered)), variant, after
    )
    draft.runs.append(run)
    return covered, (len(draft.runs) - 1,)


def _fit(
    ds: Dataset,
    kind: str,
    recipe: PreprocessParams,
    reads: Sequence[_CameraRead],
    draft: _Draft,
) -> tuple[list[_CameraRead], tuple[int, ...]]:
    """Check *reads* against *recipe*, as its run checks them before decoding one.

    A read whose file of the upstream variant is not current is covered once
    the upstream's run is planned first. Its fit against the upstream's
    placement stays unchecked, as the run leaves it. Every other failure is the
    read's refusal.

    Returns:
        The reads that a run of *recipe* covers, and the position of the
        upstream's run in the plan's runs, empty when none is planned.
    """
    failures = preprocess_entry_failures(ds, recipe, _scoped_entries(reads))
    covered: list[_CameraRead] = []
    upstream: list[_CameraRead] = []
    for read in reads:
        failure = failures.get(read.run_entry)
        if failure is None:
            covered.append(read)
        elif recipe.media and isinstance(
            failure, (MediaVariantMissingError, MediaVariantDriftedError)
        ):
            upstream.append(read)
        else:
            draft.refusals.append(_unfit(read, failure))
    if not upstream:
        return covered, ()
    upstream_recipe = _variant_recipe(ds, recipe.media)
    made, after = _variant_run(ds, kind, recipe.media, upstream_recipe, upstream, draft)
    return [*covered, *made], after


def _unfit(read: _CameraRead, failure: Exception, context: str = "") -> EntryRefusal:
    """Return the refusal of *read*, which a run of a recipe fails with *failure*.

    *context* opens the reason, before the failure's message.
    """
    cause: RefusalCause
    if isinstance(failure, PreprocessRefused):
        cause = "preprocess_refused"
    elif isinstance(failure, MediaProbeError):
        cause = "unjoinable"
    else:
        cause = "no_media"
    return _refusal(read.entry, read.camera, cause, f"{context}{failure}")


# --- refusals and order ---------------------------------------------------------


def _label(kind: str, entry: Entry) -> str:
    """Return the prefix that names *kind* and *entry* in a refusal's reason."""
    return f"[{kind}] ({entry[0]}, {entry[1]})"


def _refusal(
    entry: Entry, camera: ResolvedScopeEntry, cause: RefusalCause, reason: str
) -> EntryRefusal:
    """Return the refusal of *camera* of *entry*."""
    return EntryRefusal(entry[0], entry[1], camera.camera, cause, reason)


def _unresolved(
    entry: Entry, error: FileNotFoundError | MediaProbeError
) -> EntryRefusal:
    """Return the refusal of an entry whose media does not resolve."""
    cause: RefusalCause
    if isinstance(error, FileNotFoundError):
        cause = "no_media"
    elif isinstance(error, AnalysisDerivativeMissingError):
        cause = "needs_analysis_transcode"
    else:
        cause = "unresolved_media"
    return EntryRefusal(entry[0], entry[1], "", cause, str(error))


def _run_order(run: PrerequisiteRun) -> tuple[str, tuple[Entry, ...]]:
    """Order runs by kind, then by entry."""
    return (run.kind, run.entries)


def _refusal_order(refusal: EntryRefusal) -> tuple[str, str, str, str]:
    """Order refusals by group, sequence, camera and cause."""
    return (refusal.group, refusal.sequence, refusal.camera, refusal.cause)
