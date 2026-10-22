"""Plan the runs that an op needs before it reads its entries, and refuse the rest.

:func:`plan_prerequisites` puts each entry through the checks that the op's run
makes before its tool reads the entry. A check that a run clears is answered with
that run: ``export-joined`` for clips that are read as their join, and
``export-store`` for an imgstore that is read through its export. A check that no
run clears is the entry's refusal, with its cause and a reason that names the
remedy where one exists.

The plan reads the media index, the stores' metadata, and the files on disk with
their headers. It writes no file and decodes no frame. One entry's failure
refuses that entry alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from mosaic_media import MediaProbeError

from mosaic.core.entry import Entry
from mosaic.core.helpers import is_nameless_entry
from mosaic.core.media.facts_columns import AnalysisDerivativeMissingError
from mosaic.core.params import Params
from mosaic.core.pipeline.consumed_camera import (
    ResolvedEntryMedia,
    consumed_cameras,
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
from mosaic.core.pipeline.preprocess import PreprocessParams
from mosaic.core.pipeline.store_export import (
    StoreExportParams,
    entry_stores,
    export_run_id,
    unnamed_stores,
)
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.core.scope import Scope
from mosaic.tracking.common.scope import JoinedSourceMismatchError, refuse_unjoinable
from mosaic.tracking.common.tool_input import (
    StoreExportMissingError,
    refuse_unreadable_entry,
)
from mosaic.tracking.registration import register_ops

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.dataset import Dataset, ResolvedScopeEntry

__all__ = [
    "EntryRefusal",
    "PrerequisiteKind",
    "PrerequisitePlan",
    "PrerequisiteRun",
    "RefusalCause",
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
    "no_media_raw",
    "preprocess_refused",
    "no_variant_recipe",
    "undecodable_codec",
]
"""Why no planned run makes an entry readable. A closed set.

- ``no_media``: the entry has no indexed media, or another entry with the same
  entry key is indexed first and takes its output name.
- ``needs_analysis_transcode``: its media requires an analysis transcode that has
  not run.
- ``unresolved_media``: its media or a store's metadata cannot be read.
- ``needs_media_reprobe``: clips report no frame counts, or a store carries no
  ``video_uuid``. ``reprobe-media --apply`` measures them.
- ``several_joins``: two current joins of the clips exist, and the user keeps one.
- ``unidentified_clips``: a clip carries no content identity, so no join of the
  clips can be addressed.
- ``unjoinable``: the clips cannot be read as one video.
- ``multi_camera_join``: a camera of an entry of several cameras needs a join, and
  ``export-joined`` joins an entry of one camera.
- ``no_media_raw``: a store needs an export, and the dataset has no ``media_raw``
  root to index one in.
- ``preprocess_refused``: the variant's recipe does not fit the entry.
- ``no_variant_recipe``: the named variant lacks the entry, and no recorded recipe
  makes it.
- ``undecodable_codec``: the named variant is in a codec that the tool never reads.
"""

_EXTRACT_FRAMES: Final = "extract-frames"
"""The one op outside the tracking roots that a plan answers for."""


@dataclass(frozen=True, slots=True)
class PrerequisiteRun:
    """One run that must finish before the op reads its entries.

    Attributes:
        kind: The op to run.
        params: Its params.
        entries: Its scope, as the ``(group, sequence)`` pairs of the media
            index. ``mosaic run`` takes no camera scope, so a run covers every
            camera of an entry. A nameless entry is named here by its indexed
            key, and elsewhere in an unscoped plan by its label.
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
    """One entry that the op cannot read, and that no planned run clears.

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
        entries: The op's scope once the runs finish, sorted. It holds each
            entry without a refusal.
    """

    kind: str
    media: str
    runs: tuple[PrerequisiteRun, ...]
    refusals: tuple[EntryRefusal, ...]
    entries: tuple[Entry, ...]


def plan_prerequisites(
    ds: Dataset,
    kind: str,
    params: Params | Mapping[str, object] | None = None,
    *,
    entries: Iterable[Entry] | None = None,
) -> PrerequisitePlan:
    """Return the runs that *kind* needs before it reads *entries*, and the refusals.

    A tracker or an ``infer-*`` op is answered for the camera that it reads of
    each entry, the cameras of every entry reducing together as its run reduces
    them. ``extract-frames`` is answered for every camera. Each is put through
    the checks that its run makes before it reads the entry media:

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
    planned per op and entry. The plan is the same whatever the order of
    *entries*.

    Args:
        ds: The dataset, which the plan reads and leaves unchanged.
        kind: The op to plan for: a tracking op that reads entry media, or
            ``extract-frames``.
        params: The op's params, or ``None``. A tracking op's ``media`` is read
            from them. ``extract-frames`` reads all of its params, which must
            validate, and its listed ``frames`` narrow the entries.
        entries: The entries to plan for, or ``None`` for every indexed entry,
            each named by its label as an unscoped run names it. A run planned
            for a nameless entry is still scoped by its indexed key, because no
            run's scope names the label.

    Returns:
        The plan, with ``media`` empty.

    Raises:
        ValueError: If *kind* is not an op this answers for, or ``media`` is not
            a string.
        pydantic.ValidationError: If ``extract-frames`` params do not validate.
        ListedFramesRefused: If ``extract-frames`` lists frames of none of
            *entries*.
        NotImplementedError: If the params name a media variant in ``media``.
    """
    register_ops()
    answered = sorted({key for key in TRACKING_ROOTS if key in OPS} | {_EXTRACT_FRAMES})
    if kind not in answered:
        message = (
            f"plan_prerequisites answers for {', '.join(answered)}, and {kind!r} "
            f"is none of them."
        )
        raise ValueError(message)
    scope = None if entries is None else sorted(set(entries))
    if kind == _EXTRACT_FRAMES:
        scope = _listed_scope(params, scope)
    else:
        media = _requested_media(params)
        if media:
            message = (
                f"{kind} reads the media variant {media}, and the runs that a "
                f"media variant needs are not planned."
            )
            raise NotImplementedError(message)

    resolved = resolve_entry_media(ds, scope)
    nameless_keys = {} if scope is not None else _nameless_keys(ds)
    runs: dict[tuple[PrerequisiteKind, Entry], PrerequisiteRun] = {}
    refusals: list[EntryRefusal] = []
    for entry, read in _cameras_read(kind, resolved).items():
        if not isinstance(read, list):
            refusals.append(_unresolved(entry, read))
            continue
        cameras = resolved[entry]
        indexed = (entry, len(cameras) if isinstance(cameras, list) else 0)
        for camera in read:
            run_entry, camera_count = nameless_keys.get(
                tuple(camera.resolved.paths), indexed
            )
            answer = _answer(
                ds,
                kind,
                entry,
                camera,
                run_entry=run_entry,
                several_cameras=camera_count > 1,
            )
            if isinstance(answer, EntryRefusal):
                refusals.append(answer)
            elif answer is not None:
                _ = runs.setdefault((answer.kind, run_entry), answer)

    refused = {(refusal.group, refusal.sequence) for refusal in refusals}
    return PrerequisitePlan(
        kind=kind,
        media="",
        runs=tuple(sorted(runs.values(), key=_run_order)),
        refusals=tuple(sorted(refusals, key=_refusal_order)),
        entries=tuple(sorted(entry for entry in resolved if entry not in refused)),
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

    validated = ExtractFramesParams.model_validate(
        params if isinstance(params, Params) else dict(params or {})
    )
    listed = listed_frames(validated)
    if listed is None:
        return scope
    return listed_entries(listed, scope)[0]


def _nameless_keys(ds: Dataset) -> dict[tuple[Path, ...], tuple[Entry, int]]:
    """Return each nameless entry's key and camera count, keyed by its cameras' clips.

    An unscoped plan names an entry by its label, and labels each camera of a
    nameless entry by its first file's stem. A run's scope names the entry by
    its indexed key instead, and resolves every camera under that key. The
    clips identify the camera even where its label equals another entry's key.
    """
    try:
        indexed = ds.resolve_scope(Scope(groups=[""])).entries
    except FileNotFoundError:
        return {}
    nameless = sorted(entry for entry in indexed if is_nameless_entry(*entry))
    if not nameless:
        return {}
    return {
        tuple(camera.resolved.paths): (entry, len(media))
        for entry, media in resolve_entry_media(ds, nameless).items()
        if isinstance(media, list)
        for camera in media
    }


def _cameras_read(
    kind: str, resolved: Mapping[Entry, ResolvedEntryMedia]
) -> dict[Entry, ResolvedEntryMedia]:
    """Return the cameras that *kind* reads of each entry, or why it reads none.

    ``extract-frames`` reads every camera. Any other op reads one camera per
    entry, reduced over every entry as its run reduces them.
    """
    if kind == _EXTRACT_FRAMES:
        return dict(resolved)
    consumed = consumed_cameras(kind, resolved, report_skipped=False)
    return {
        entry: media
        if isinstance(media, (FileNotFoundError, MediaProbeError))
        else [media]
        for entry, media in consumed.items()
    }


def _answer(
    ds: Dataset,
    kind: str,
    entry: Entry,
    camera: ResolvedScopeEntry,
    *,
    run_entry: Entry,
    several_cameras: bool,
) -> PrerequisiteRun | EntryRefusal | None:
    """Return the run that lets *kind* read *camera*, its refusal, or ``None``.

    Args:
        ds: The dataset.
        kind: The op that reads the camera.
        entry: The entry as the plan names it.
        camera: One camera of the entry, resolved.
        run_entry: The entry as a planned run's scope names it.
        several_cameras: Whether *run_entry* has more than one camera.
    """
    try:
        if kind == _EXTRACT_FRAMES:
            refuse_unjoinable(
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
                return _join_run(
                    kind, entry, camera, run_entry, several_cameras=several_cameras
                )
            case "several_joins" | "unidentified_clips" as cause:
                return _refusal(entry, camera, cause, str(refusal))
    except StoreExportMissingError:
        return _store_export_run(ds, kind, entry, camera, run_entry)
    # A nameless entry's store raises a bare FileNotFoundError. Its export is
    # looked up by the entry's label, which matches no index row.
    except (FileNotFoundError, MediaProbeError) as refusal:
        return _refusal(entry, camera, "unresolved_media", str(refusal))
    return None


def _join_run(
    kind: str,
    entry: Entry,
    camera: ResolvedScopeEntry,
    run_entry: Entry,
    *,
    several_cameras: bool,
) -> PrerequisiteRun | EntryRefusal:
    """Return the ``export-joined`` run of *camera*'s clips, or why its run fails.

    ``export-joined`` resolves *run_entry* without a camera, which refuses an
    entry of several cameras, and checks the join against the clips' summed
    frame counts.
    """
    label = _label(kind, entry)
    if several_cameras:
        reason = (
            f"{label} camera {camera.camera} is read as the join of its clips, "
            f"and export-joined joins the clips of an entry of one camera. Read "
            f"the camera through a media variant instead."
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
    return PrerequisiteRun("export-joined", params, (run_entry,), run_id, ())


def _store_export_run(
    ds: Dataset,
    kind: str,
    entry: Entry,
    camera: ResolvedScopeEntry,
    run_entry: Entry,
) -> PrerequisiteRun | EntryRefusal:
    """Return the ``export-store`` run of *run_entry*, or why its run fails.

    The run exports the store of every camera of the entry, each named by its
    ``video_uuid``, and indexes the exports beside a ``media_raw`` root.
    """
    label = _label(kind, entry)
    if not ds.routes_derivatives():
        reason = (
            f"{label} is read through the export of its imgstore, and this "
            f"dataset has no media_raw root. Its originals index is "
            f"media/index.csv, where an export is not read."
        )
        return _refusal(entry, camera, "no_media_raw", reason)
    try:
        unnamed = unnamed_stores(entry_stores(ds, *run_entry))
    except FileNotFoundError as refusal:
        return _refusal(entry, camera, "unresolved_media", str(refusal))
    if unnamed:
        reason = (
            f"{label} is read through the export of its imgstore, and "
            f"{', '.join(store.name for store in unnamed)} has no video_uuid in "
            f"the media index to name an export by. Run 'mosaic reprobe-media "
            f"--apply' first."
        )
        return _refusal(entry, camera, "needs_media_reprobe", reason)
    params = StoreExportParams()
    run_id = export_run_id(params)
    return PrerequisiteRun("export-store", params, (run_entry,), run_id, ())


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
