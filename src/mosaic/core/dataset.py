# dataset.py
from __future__ import annotations

import datetime
import hashlib
import itertools
import json
import os
import re
import sys
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    cast,
    overload,
)


import numpy as np
import pandas as pd
from mosaic_media import (
    VIDEO_EXTENSIONS,
    MediaFacts,
    MediaProbeError,
)

from .helpers import (
    detect_label_format,
    is_nameless_entry,
    make_entry_key,
    parse_entry_key,
    place_label_payload,
    text_cell,
    to_safe_name,
)
from .media.drift import MeasurementOrigin, MediaDrift, classify_identity
from .media.facts_columns import (
    MEDIA_INDEX_COLUMNS as MEDIA_INDEX_COLUMNS,  # re-exported for API/tests
    AnalysisDerivativeMissingError,
    AssignmentSource,
    ProbeMetadata,
    derivative_path_for_target,
    media_row_entry,
    media_row_key,
    media_row_uuid,
    read_link_cell,
    row_division,
    row_facts_or_none,
    row_mapping,
    row_to_facts,
    series_facts_or_none,
)
from .media.probe_row import probe_video_metadata, row_from_facts
from .media.store_division import (
    DIVISION_ASSIGNMENT_SOURCE,
    CommittedDivision,
    DivisionPlan,
    DivisionRefused,
    DivisionRule,
    HeldBack,
    MediaRowKey,
    RefusedSpan,
    StoreWindow,
    held_back_of,
    refused_of,
    store_base_name,
)
from .media.store_rate import store_rate_mismatch
from .media.uniformity import UniformityVerdict, camera_uniformity
from .schema import (
    TrackSchemaError,
    TRACK_SCHEMAS,
    TrackSchema,
    ensure_track_schema,
)
from .stored_paths import (
    manifest_file_path,
    remap_single_path as _remap_single_path,
    resolve_stored_path,
)
from .track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    ensure_track_converters_registered,
    get_track_converter,
    merge_on_column_union,
)
from .label_converter import (
    LabelConverter,
    LabelConvertParams,
    LabelEntry,
    get_label_converter,
    validate_label_format,
)
from .media.prune import (
    PruneReport,
    declined_report,
    prune_media as _prune_media,
)
from .media.reprobe import (
    ReprobeAbort,
    ReprobeReport,
    reprobe_media as _reprobe_media,
)
from .entry import (
    Entry,
    EntryKeyCollisionError,
    entry_key_collision,
    entry_token,
    validate_entry_name,
)
from .individuals import refuse_repeated_rows
from .recording import (
    OrderFacts,
    Recording,
    RecordingKey,
    RecordingKind,
    adjacent,
    order_members,
    recording_kind,
    recording_name,
    recording_name_collision,
    recording_of,
    recordings,
    sequence_frame,
)
from .json_value import JsonValue
from .manifest import (
    AnyScanSource,
    DatasetManifest,
    DatasetTag,
    LabelsScanSource,
    LibraryLink,
    MANIFEST_VERSION,
    MediaDivision,
    MediaLayout,
    MediaScanSource,
    RawScanSource,
    ScanSource,
    ScanSources,
    SourceKind,
    SplitRole,
    default_roots,
    empty_root_template,
    legacy_tracking_roots,
    new_manifest,
    now_stamp,
    overlapping_sources,
    read_manifest,
    resolve_manifest_path,
    validate_root_inside,
    write_manifest,
)
from .scope import Scope
from .typed_attribute import TypedAttributeType, TypedAttributeValue
from .pipeline._utils import (
    ResolvedScope,
    atomic_write,
    coerce_np as _coerce_np,
    now_iso as _now_iso,
)
from .pipeline.placement import place_converted
from .pipeline.scan_claim import ScanClaim
from .pipeline.artifact_kinds import ArtifactKind
from .pipeline.index_stamp import IndexStamp
from .pipeline.media_index import (
    MEDIA_NUMERIC_COLUMNS,
    MediaIndexScope,
    RowStat,
    StoreGroupingError,
    build_media_index_row,
    build_prior_order,
    media_members_from_rows,
    densify_video_order,
    frame_from_rows,
    mtime_iso,
    read_media_index as _read_media_index,
    refuse_merged_store_recordings,
    unnamed_cameras_error,
    write_media_index_rows,
)
from .pipeline.tracks_identity import (
    convert_variant_payload,
    converter_op,
    tracks_run_id,
    tracks_variant_root,
    write_tracks_variant,
)
from .pipeline.labels_identity import (
    label_convert_variant_payload,
    label_converter_op,
    labels_run_id,
    labels_variant_root,
    write_labels_variant,
)
from .pipeline.composition import (
    labels_raw_composition,
    media_composition,
    tracks_raw_composition,
)
from .pipeline.index_csv import index_records
from .pipeline.index_lock import index_lock
from .pipeline.label_series import revision_of, series_spec
from .pipeline.label_series_index import (
    carried_series_digests,
    ensure_series_root,
    read_label_series,
    row_for_revision_file,
    series_index_path,
    write_label_series_rows,
)
from .pipeline.sequence_index import (
    SequenceLabelRow,
    SourceRoot,
    amend_sequence_labels,
    read_sequence_labels,
    sequence_index_path,
    sequence_label_path,
    sequence_labels,
    write_sequence_compositions,
)
from .pipeline.dataset_indexes import iter_dataset_indexes, label_kinds
from .pipeline.entry_keys import (
    entries_kept_beside,
    entries_of_rows,
    index_entries,
    media_index_entries,
    root_index_entries,
)
from .pipeline.preprocess_layout import media_variant_filter, media_variants_root
from .pipeline.promotion import correction_revision
from .pipeline.tracking_roots import (
    TRACKING_ROOT,
    TRACKING_ROOTS,
    is_under_tracking_root,
)
from .pipeline.tracks_index import (
    TRACKS_INDEX_PATH_COLUMNS,
    CellBackfill,
    adopt_legacy_columns,
    backfill_frame_extents,
    backfill_frames_read,
    backfill_known_tail_loss,
    backfill_media_frames,
    FrameAxisMismatch,
    FrameTailShortfall,
    frame_axis_mismatches,
    frame_tail_shortfalls,
    consumed_composition_for,
    legacy_view,
    read_frame_extent,
    read_tracks_index,
    select_variant_rows,
    tracks_index_path,
    write_tracks_row,
)
from .pipeline.labels_index import (
    LABELS_INDEX_PATH_COLUMNS,
    labels_index_path,
    read_labels_index,
    select_label_variant_rows,
    write_labels_row,
)
from .pipeline.tracks_raw_index import (
    TracksRawIndexRow,
    TracksRawIndexScope,
    build_tracks_raw_row,
    frame_from_rows as _tracks_frame_from_rows,
    iter_track_files,
    load_tracks_raw_index_frame,
    read_tracks_raw_index as _read_tracks_raw_index,
    source_members_from_rows,
    write_tracks_raw_index_rows,
)
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.pipeline._utils import atomic_savez
from mosaic.user_paths import user_path

if TYPE_CHECKING:
    from .pipeline.inventory import ArtifactRef
    from .pipeline.job import CancelToken
    from .pipeline.progress import ProgressCallback
    from .pipeline.reconcile import ReconcileReport
    from .media.prune_joined import JoinedPruneReport
    from .pipeline.sweep import SweepClass, SweepReport


def _normalize_patterns(pats) -> tuple[str, ...]:
    if pats is None:
        return tuple()
    if isinstance(pats, str):
        return (pats,)
    try:
        return tuple(pats)
    except TypeError:
        return (str(pats),)


def _normalize_path_map(path_map: Mapping[str, str]) -> list[tuple[Path, Path]]:
    normalized: list[tuple[Path, Path]] = []
    for src, dst in path_map.items():
        if not src or not dst:
            continue
        normalized.append((user_path(src), user_path(dst)))
    normalized = [pair for pair in normalized if pair[0] != pair[1]]
    normalized.sort(key=lambda pair: len(pair[0].as_posix()), reverse=True)
    return normalized


# Per-root, the path-bearing columns beyond ``abs_path``. One table rather than a
# special case per root, because both path passes read raw CSVs and have no row
# class to ask -- so a column missing from here silently stops being portable,
# which is what happened to the tracks index's ``source_abs_path`` until now.
#
# The tracker entries come from the tracking-root registry rather than being
# listed here: three hand-written tuples beside a table of three roots is the
# arrangement where adding a fourth tracker means remembering a second place,
# and the add-a-tracker recipe had a checklist item for exactly that.
#
# The ``models`` entry is a union across the two row types living under that
# root -- a trained model and a converted dataset -- because the lookup is by
# root key and both are ``models/<kind>/index.csv``. Naming a column a given
# index does not have is harmless: both passes intersect against the frame's
# real columns rather than creating what is missing.
_CORRECTION_COL: Final = "_correction_revision"
"""Groupby key naming which promoted revision a raw row is, 0 for an upload.

Derived per call rather than stored: it is a property of the filename promotion
chose, and a column in ``tracks_raw/index.csv`` would be a second place for it to
disagree from.
"""

_GROUP_FROM_DEFAULT: Final = "filename"
"""The group policy a track conversion applies when its caller names none.

The default of ``LabelConvertParams.group_from``, so a file that feeds both the
tracks and the labels root, as a CalMS21 ``.npy`` does, names each entry alike in
both. ``filename`` takes the file-level group the scan recorded, which it records
only where the source declares a grouping rule, and the group inside the file
everywhere else.
"""

_INDEX_PATH_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = {
    "tracks": TRACKS_INDEX_PATH_COLUMNS,
    # ``labels`` belongs here for the same reason ``tracks`` does: its rows carry
    # a ``source_abs_path`` naming the upload they were converted from. Left out,
    # every portability pass walked the labels index and rewrote only
    # ``abs_path``, so a dataset moved between machines kept a labels row
    # pointing at the old one's filesystem -- and reported itself portable.
    "labels": LABELS_INDEX_PATH_COLUMNS,
    "models": (
        "best_model_path",
        "metrics_path",
        "artifact_path",
        "data_yaml",
        "data_path",
    ),
    **{key: root.path_columns for key, root in TRACKING_ROOTS.items()},
}

# The track-converter registry moved to ``core.track_converter``. That is what
# closes the cycle noted at the foot of this file: a converter no longer imports
# ``dataset`` in order to register itself.
#
# ``register_track_converter`` is deliberately NOT re-exported here. It is now a
# class decorator rather than ``(src_format, fn)``, so a name that still
# resolved but meant something else would be worse than an ImportError --
# import it from ``mosaic.core.track_converter``.
#
# ``TRACK_SEQ_ENUM`` is gone too. It was a second dict keyed by the same
# src_format string and populated by the same modules, so it could register an
# enumerator for a format with no converter; enumeration is now a method on the
# converter class, behind its ``enumerable`` flag.


# The label-converter registry moved to ``mosaic.core.label_converter`` (imported
# at the top), the same relocation ``track_converter`` made and for the same
# reason: a converter importing ``dataset`` to register itself while ``dataset``
# imported the registry to dispatch was the cycle. The typed ``labels/<kind>/``
# schema lives on ``LabelsIndexRow`` in ``labels_index`` -- there is no
# ``LABEL_INDEX_COLUMNS`` list to keep in step with it any more.


def _md5(path: Path, chunk=1 << 20) -> str:
    """The raw-source checksum recorded in ``tracks_raw/index.csv``.

    Still md5, deliberately, though identity elsewhere is sha1 (``hash_params``)
    and blake2b (``mosaic-media``'s content digest). The column is named ``md5``
    and its value is copied verbatim into ``TracksIndexRow.source_md5``, the
    label index, and every one of the six registered label converters -- one of
    which, ``label_converter_template.py``, is the documented extension point a
    third-party converter is written against. Renaming it there would break an
    out-of-tree converter with no error, only an empty cell, which is the rename
    the migration rule forbids. The algorithm rides *inside* the composition
    payload instead (see ``composition``), so switching it later changes every
    digest rather than producing an incomparable one that looks comparable.
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _format_merges_per_sequence(src_format: str) -> bool:
    """Does this format's converter declare that several files are one sequence?

    Tolerant where :func:`get_track_converter` is strict, and deliberately so.
    ``index_tracks_raw`` resolves a format the caller has just *chosen*, where a
    typo is worth refusing; this reads one back out of an index it did not
    necessarily write, for a bulk gesture that warns per row and keeps going. An
    unregistered format is one warning per file, not an aborted run over the
    rest of the dataset.
    """
    ensure_track_converters_registered()
    converter_cls = TRACK_CONVERTERS.get(src_format)
    return converter_cls is not None and converter_cls.merges_per_sequence


def _track_variant_id(
    converter: TrackConverter[TrackConvertParams],
    params: TrackConvertParams,
    correction: int,
) -> str:
    """The tracks variant one conversion recipe names, computed without I/O."""
    cls = type(converter)
    payload = convert_variant_payload(params.identity_dump(), correction)
    return tracks_run_id(converter_op(cls.src_format), cls.version, payload)


def _label_variant_id(
    converter: LabelConverter[LabelConvertParams],
    params: LabelConvertParams,
    label_kind: str,
) -> str:
    """The labels variant one label conversion recipe names, computed without I/O."""
    cls = type(converter)
    payload = label_convert_variant_payload(label_kind, params.identity_dump())
    return labels_run_id(label_converter_op(cls.src_format), cls.version, payload)


def _stem_as_sequence(stem: str) -> str:
    """One file is one sequence, and the stem names it.

    The naming rule for a source that has no rule of its own -- every label
    source, and every track format whose converter does not declare otherwise.
    Named rather than inlined so that both callers of ``_index_raw`` state their
    rule the same way and neither carries a branch about which one it is.
    """
    return stem


def _dataset_base_dir(ds) -> Path:
    """
    Resolve the directory that holds dataset-level config (sibling to dataset manifest).
    """
    base = getattr(ds, "manifest_path", None)
    if base is not None:
        base = Path(base)
        base = base.parent if base.is_file() else base
    else:
        base = Path(ds.get_root("features")).parent
    base.mkdir(parents=True, exist_ok=True)
    return base


def _unresolved_root_message(
    key: str, roots: Mapping[str, str], manifest_path: Path
) -> str:
    """Why ``get_root`` could not resolve *key*, and the verb that repairs it.

    Three faults used to share one sentence, and the sentence described the
    rarest: it called the root "not set in manifest" and then listed that same
    root among the "available" ones. A ``Dataset`` is built around a manifest
    *path* and starts from :func:`~mosaic.core.manifest.empty_root_template` --
    every declared key present, every value empty -- so the reader was sent to a
    manifest file that was correct, when what was missing was the load.

    So the state picks the sentence, and the key that failed never appears
    inside a list the reader will scan as "roots that work":

    - Nothing anywhere holds a value: the ``Dataset`` nobody loaded. Lists
      nothing, and names both ways a dataset comes to have roots.
    - Declared with an empty value: the manifest's own spelling of unset. Lists
      the roots that do hold a value.
    - Not declared: a misspelling, or a root this manifest predates. Lists the
      declared keys, which is where a near miss shows.

    The text carries no ``"`` and no newline. ``KeyError`` reprs its argument, so
    either one would be escaped back at the reader instead of read.
    """
    if not any(roots.values()):
        return (
            f"dataset root {key!r} cannot be resolved: this dataset declares "
            "no roots at all. Constructing a Dataset only points at a manifest "
            f"-- read {manifest_path} with open_dataset() or load() to pick up "
            "the roots it declares, or pass roots= to the constructor when the "
            "dataset is not on disk yet."
        )
    if key in roots:
        with_values = ", ".join(sorted(name for name, value in roots.items() if value))
        return (
            f"dataset root {key!r} is declared with an empty value, which reads "
            f"as unset. Roots that hold a value: {with_values}. Give it a path "
            "with set_root(), or test has_root() first where an unset root is a "
            "state the caller expects."
        )
    declared = ", ".join(sorted(roots))
    return (
        f"dataset root {key!r} is not declared by this manifest. Declared "
        f"roots: {declared}. Check the spelling against that list, or declare "
        "the root with set_root()."
    )


############# DATASET

_RAW_ROOT_FOR_KIND: Final[Mapping[str, str]] = {
    "media": "media_raw",
    "tracks": "tracks_raw",
    "labels": "labels_raw",
}
"""Which root a declared source of each kind feeds."""


# The manifest format -- the root table, its backfills, the
# inside-the-dataset rule and the legacy-layout query -- lives in
# ``core.manifest``, which knows nothing about ``Dataset``. They are
# re-exported here because every caller already imports them from this
# module, and because what a root *is* stays a property of the dataset that
# declares it.


def _tag_type_of(value: TypedAttributeValue) -> TypedAttributeType:
    """The tag type a value alone names, for a tag defined by its first value."""
    if value is None:
        return "label"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "text"
    msg = (
        f"a tag defined by its value takes a string, number, boolean or null, "
        f"not {value!r}; define it with mosaic tags define"
    )
    raise ValueError(msg)


def new_dataset_manifest(
    name: str,
    base_dir: str | Path,
    roots: Mapping[str, str | Path] = default_roots,
    version: str = "0.1.0",
    outfile: str | Path | None = None,
    *,
    notes: str = "",
    tags: Sequence[DatasetTag] = (),
    sources: ScanSources | None = None,
) -> Path:
    """Create a dataset manifest and the root directories it declares.

    Every root is normalized to a path relative to *base_dir*, which is the
    portable form, and created. A root resolving outside the dataset raises:
    that root's own ``index.csv`` would land outside too, and then the dataset
    stops being the thing you can copy, archive or sync. Storage elsewhere is
    reached by declaring a :class:`~mosaic.core.manifest.ScanSources` entry for
    it, whose files are indexed by absolute ``abs_path`` into an index that stays
    inside.

    Args:
        name: What the dataset is called.
        base_dir: The dataset directory. Created if absent.
        roots: Root declarations. Defaults to every root mosaic knows.
        version: The dataset's own version string.
        outfile: Where to write. Defaults to ``<base_dir>/dataset.yaml``; the
            suffix chooses YAML or JSON.
        notes: Free text for the dataset.
        tags: Typed attributes describing the dataset.
        sources: Declared scan recipes. Unlike roots, these may point anywhere.

    Returns:
        The path written.

    Raises:
        ValueError: If a root resolves outside *base_dir*.
    """
    base_dir = user_path(base_dir).resolve()
    normalized: dict[str, str] = {}
    for key, declared in roots.items():
        # The validator's return value is what gets joined, not `declared`: it is
        # the expanded spelling, and rebuilding from the raw one would put back
        # the `~` the validator just judged.
        checked = validate_root_inside(base_dir, declared, key)
        full = (base_dir / checked).resolve()
        full.mkdir(parents=True, exist_ok=True)
        normalized[key] = str(full.relative_to(base_dir))

    manifest = new_manifest(
        name,
        version=version,
        roots=normalized,
        notes=notes,
        tags=tags,
        sources=sources,
    )
    target = user_path(outfile) if outfile is not None else base_dir / "dataset.yaml"
    write_manifest(target, manifest)
    return target


def open_dataset(path: str | Path, *, ensure_roots: bool = True) -> Dataset:
    """Open the dataset at *path* -- ``Dataset(path).load()`` as one call.

    The constructor deliberately reads nothing: it takes a manifest *path*, so a
    caller can point at a dataset that does not exist yet and create it. That
    makes ``Dataset(path)`` alone a working expression yielding an object whose
    roots are all empty, and every accessor on it then fails against a manifest
    file that is perfectly correct. This is the front door for the far more
    common intent, and the counterpart to :func:`new_dataset_manifest`: that one
    creates a dataset, this one opens one.

    Args:
        path: The manifest file, or the directory holding it.
        ensure_roots: Create the declared root directories. Pass ``False`` on a
            read-only mount, or wherever opening a dataset must not write to it.

    Returns:
        The loaded dataset.

    Raises:
        FileNotFoundError: If no manifest is at *path*.
        ManifestVersionError: If the manifest is newer than this mosaic.
    """
    return Dataset(manifest_path=path).load(ensure_roots=ensure_roots)


# --------------------------
# Dataset manifest + manager
# --------------------------


def _entry_mask(df: "pd.DataFrame", entries: Iterable[Entry]) -> "pd.Series":
    """Mask selecting the media-index rows whose entry is one of *entries*.

    Compares on the ``(group, sequence)`` pair alone, which keeps every camera
    and every clip of one entry. :meth:`Dataset.resolve_media_scope` narrows
    the index with it before grouping the surviving rows onto the camera axis.
    """
    wanted = {(str(group), str(sequence)) for group, sequence in entries}
    pairs = pd.MultiIndex.from_arrays([df["group"], df["sequence"]])
    return pd.Series(pairs.isin(wanted), index=df.index)


def _media_cell(row: "pd.Series", key: str) -> str:
    """Read a media-index cell of a ``Series`` row as a trimmed string.

    The ``Series``-shaped adapter over :func:`read_link_cell`, which owns the
    rule that empty, ``"nan"`` and a float NaN all mean absent. Kept as one
    delegation rather than a second implementation: two copies of that rule are
    free to drift, and a cell read as absent by one and real by the other is
    exactly how a link resolves to the wrong file.
    """
    return read_link_cell(row_mapping(row), key)


def _facts_or_stale_probe_error(
    drow: "pd.Series", group: str, sequence: str
) -> MediaFacts:
    """Reconstruct a derivative row's :class:`MediaFacts`, or raise on stale facts.

    A ``media_facts`` cell that no longer reconstructs raises ``TypeError`` from
    :func:`row_to_facts`; convert it to the
    :class:`~mosaic_media.MediaProbeError` callers catch, naming the entry and
    the remedy. The ``try`` wraps only the reconstruction call, so an unrelated
    ``TypeError`` is never masked.

    The cause is stated as a stored measurement rather than as the identity
    fields specifically. Reconstruction fails the same way whichever key is
    missing -- a measurement added upstream is as much a reason as an identity
    value -- so naming one of them would send an operator looking for the wrong
    thing on every other cause.
    """
    try:
        return row_to_facts(row_mapping(drow))
    except TypeError as exc:
        message = (
            f"entry {group}/{sequence} has a derivative row whose stored facts "
            "no longer reconstruct; re-probe the media index"
        )
        raise MediaProbeError(message) from exc


@dataclass(frozen=True)
class ProbedEntry:
    """One probed media file or imgstore, before identity is assigned.

    Produced by :meth:`Dataset._probe_dir_rows` and consumed by both
    :meth:`Dataset.index_media` (scan-and-derive) and
    :meth:`Dataset.write_media_index` (assignment-driven). ``camera_serial``
    and ``sync_uuid`` are store *facts* read from imgstore metadata (empty for a
    plain video). The caller derives the sequence identity and the camera axis
    from them, and the probe assigns no identity.

    ``origin`` says where ``probe`` came from. The media drift check reads it to
    tell a file whose bytes moved from a caller injecting a measurement of a
    different file. The field is last and has a default, which leaves the
    keyword arguments of the imgstore site unaffected.
    """

    path: Path
    stat: RowStat
    probe: ProbeMetadata
    media_type: str
    camera_serial: str = ""
    sync_uuid: str = ""
    origin: MeasurementOrigin = "probed"


class AmbiguousMediaMatchError(MediaProbeError):
    """A sequence with no index row of its own matched several entries.

    Raised by the last matching tier in :meth:`Dataset._match_media_rows` when the
    rows it matches belong to more than one ``(group, sequence)`` -- two groups
    holding a sequence of the same name, asked for without a group. Returning
    either would bind the request to one group's media on no evidence, and
    returning both would concatenate unrelated recordings into one timeline.

    A subclass of :class:`~mosaic_media.MediaProbeError` so a caller resolving
    many entries -- a migration reporting faults per entry rather than aborting --
    can vent it per entry and keep going, alongside the transcode and measurement
    faults that resolution already raises as that type.
    """


@dataclass(frozen=True)
class UnplacedSequence:
    """Why a sequence of a divided recording has no first frame on its frame axis."""

    reason: str


@dataclass(frozen=True)
class ResolvedMedia:
    """Resolved media file paths for one (group, sequence), plus stored facts.

    ``paths`` are ordered by ``video_order`` (one element for a single-file
    sequence). ``facts`` is a parallel list of
    :class:`~mosaic_media.MediaFacts`, one per path, that readers inject instead
    of re-probing. ``windows`` is parallel to them too: the chunks of a store
    that a division row reads, and ``None`` for a file read whole. A reader
    opens each path on its window, and :attr:`row_keys` finds each path's row.

    ``placement`` is ``None`` for an undivided recording, and why the
    sequence cannot be placed when such a recording holds a division row of a
    store. For a sequence of a continuous group, whose tracks count ``frame``
    across the group, it is the sequence's first frame on the recording's
    frame axis, or why the sequence cannot be placed there. The media contains
    the frames of this sequence alone. A reader subtracts :attr:`first_frame` from a tracks frame
    (:meth:`sequence_frame`). A consumer that reads the media without a tracks
    frame does not ask for it, and resolves a sequence that cannot be placed.
    """

    paths: list[Path]
    facts: list[MediaFacts]
    windows: list[StoreWindow | None]
    placement: int | UnplacedSequence | None = None
    time_offset: float | None = None
    """The sequence's first frame on its recording's time axis, in seconds.

    A division continues the division before it at that division's rate, and
    takes it from the store clock after a gap or a store boundary
    (:meth:`Dataset.division_facts`). ``None`` for every other sequence, whose
    first frame over its rate is the offset.
    """

    def __post_init__(self) -> None:
        """Refuse windows that are not parallel to the paths."""
        if len(self.windows) != len(self.paths):
            message = (
                f"{len(self.windows)} windows for {len(self.paths)} media paths; "
                "they are parallel"
            )
            raise ValueError(message)

    @property
    def row_keys(self) -> list[MediaRowKey]:
        """The media index key of each path's row, parallel to :attr:`paths`.

        :meth:`Dataset.media_rows_of` takes them.
        """
        return [
            MediaRowKey.of(str(path), window)
            for path, window in zip(self.paths, self.windows, strict=True)
        ]

    @property
    def first_frame(self) -> int:
        """The sequence's first frame on its recording's frame axis, 0 when undivided.

        Raises:
            MediaProbeError: If the sequence cannot be placed on the axis.
        """
        if isinstance(self.placement, UnplacedSequence):
            raise MediaProbeError(self.placement.reason)
        return 0 if self.placement is None else self.placement

    @property
    def frame_count(self) -> int | None:
        """The frames of every clip, summed, or ``None`` when a clip's count is unknown."""
        counts = [int(clip.frame_count) for clip in self.facts]
        if not counts or min(counts) <= 0:
            return None
        return sum(counts)

    def sequence_frame(self, frame: int) -> int:
        """Return the media frame that recording frame *frame* is.

        A frame past the end of an undivided recording's media is returned
        unchanged, and a reader stops at the end of the file. In a divided
        recording that frame belongs to a later sequence.

        Raises:
            FrameOutsideSequenceError: If the sequence's media does not contain
                *frame*.
            MediaProbeError: If the sequence cannot be placed on its
                recording's frame axis.
        """
        return sequence_frame(
            frame,
            first_frame=self.first_frame,
            frame_count=None if self.placement is None else self.frame_count,
        )


@dataclass(frozen=True)
class ResolvedScopeEntry:
    """One resolved ``(group, sequence, camera)`` entry from a scoped enumeration.

    ``camera`` is the within-sequence camera axis (``""`` for single-camera
    media); a multi-camera recording yields one entry per camera, each with its
    own ``resolved`` media, so a consumer never concatenates two cameras into
    one timeline. A dataclass (not a tuple) so later phases can add per-entry
    fields (calibration, session clock) without re-breaking every consumer.
    """

    group: str
    sequence: str
    camera: str
    resolved: ResolvedMedia


@dataclass(frozen=True)
class _ScopedMediaRows:
    """One scoped ``(group, sequence, camera)`` of the originals index, unrouted.

    ``group`` and ``sequence`` are the entry's cells as indexed.
    ``sequence_label`` is the sequence a media scope reports, which differs from
    ``sequence`` only for a nameless entry. ``rows`` are this camera's index
    rows, ``video_order``-sorted.
    """

    group: str
    sequence: str
    sequence_label: str
    camera: str
    rows: pd.DataFrame

    @property
    def entry(self) -> Entry:
        """The ``(group, sequence)`` pair as indexed."""
        return (self.group, self.sequence)


def _one_value(values: Collection[str] | None) -> str:
    """Return the one value of *values*, ``""`` for none or several."""
    if values is None or len(values) != 1:
        return ""
    return next(iter(values))


def _row_window(row: pd.Series) -> StoreWindow | None:
    """Return the chunks of a store that a media index row reads, ``None`` when whole.

    Raises:
        MediaProbeError: If the row's chunk cells do not read.
    """
    try:
        return media_row_key(row_mapping(row)).window
    except ValueError as exc:
        raise MediaProbeError(str(exc)) from exc


def _recording_divided(
    group: str, group_rows: pd.DataFrame
) -> dict[str, OrderFacts] | UnplacedSequence:
    """Return what the division rows of the recording *group* record, by sequence.

    *group_rows* are every originals row of the recording, each read once. A
    recording whose rows are all division rows is divided, and one with no
    division row is not.

    A division's ``time_offset`` continues the division before it when its
    frames do (:func:`~mosaic.core.recording.adjacent`): that division's offset
    plus its frames at its row's rate, so time steps across the boundary as it
    does inside a division. Without a gap since the recording's first division,
    and at one rate, it is ``first_frame / fps``. After a gap, a store boundary
    or a division missing from the index, the offset is taken from the store
    clock: the division's start less the start of the recording's first
    division, plus that division's first frame at its rate.

    Returns:
        The facts of each division, nothing for an undivided recording, or why
        no sequence of the recording can be placed: it mixes division rows with
        other rows.

    Raises:
        MediaProbeError: If a row's division cells do not read.
    """
    try:
        read = [
            (row_division(row_mapping(row)), _row_rate(row))
            for _, row in group_rows.iterrows()
        ]
    except ValueError as exc:
        raise MediaProbeError(str(exc)) from exc
    divisions = [(division, rate) for division, rate in read if division is not None]
    if not divisions:
        return {}
    if len(divisions) < len(read):
        reason = (
            f"the recording {group!r} holds {len(divisions)} division rows "
            f"of a store and {len(read) - len(divisions)} other rows. A recording "
            "is the divisions of its stores or the sequences of its files, and "
            "not both."
        )
        return UnplacedSequence(reason)
    ordered = sorted(divisions, key=lambda found: found[0].first_frame)
    origin, origin_rate = ordered[0]
    lead: float | None = 0.0
    if origin.first_frame:
        lead = origin.first_frame / origin_rate if origin_rate > 0 else None
    facts: dict[str, OrderFacts] = {}
    previous: tuple[OrderFacts, float] | None = None
    for division, rate in ordered:
        placed = OrderFacts(
            first_frame=division.first_frame,
            start_utc=division.start_utc,
            frame_count=division.frame_count,
            follows_gap=division.follows_gap,
        )
        offset = _continued_offset(previous, placed)
        if offset is None and lead is not None:
            offset = (division.start_utc - origin.start_utc).total_seconds() + lead
        placed = dataclasses.replace(placed, time_offset=offset)
        facts[division.sequence] = placed
        previous = (placed, rate)
    return facts


def _continued_offset(
    previous: tuple[OrderFacts, float] | None, division: OrderFacts
) -> float | None:
    """Return *division*'s time offset continuing *previous* at its rate.

    *previous* is the division before it on the frame axis, with its row's
    rate. ``None`` when *division* does not continue its frames, or the
    previous offset or rate is unknown.
    """
    if previous is None:
        return None
    before, rate = previous
    if (
        before.time_offset is None
        or before.frame_count is None
        or rate <= 0
        or not adjacent(before, division)
    ):
        return None
    return before.time_offset + before.frame_count / rate


def _row_rate(row: pd.Series) -> float:
    """The ``fps`` cell of a media index row, ``0.0`` when it does not read."""
    try:
        rate = float(row.get("fps", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return rate if np.isfinite(rate) else 0.0


def _undeclared_division(group: str, matched: pd.DataFrame) -> UnplacedSequence | None:
    """Return why a division of an undeclared recording has no frame axis.

    *matched* are the sequence's own rows in the recording *group*, which the
    manifest does not declare continuous. Such a recording has no frame axis, so
    a division row among them is not placed at frame 0.

    Returns:
        Why the sequence cannot be placed, or ``None`` when no row is a division
        row.

    Raises:
        MediaProbeError: If a row's division cells do not read.
    """
    try:
        divisions = [row_division(row_mapping(row)) for _, row in matched.iterrows()]
    except ValueError as exc:
        raise MediaProbeError(str(exc)) from exc
    if all(division is None for division in divisions):
        return None
    reason = (
        f"the recording {group!r} holds division rows of a store but is not "
        "declared continuous, so its sequences have no frame axis. "
        "Dataset.set_continuous_groups declares it."
    )
    return UnplacedSequence(reason)


def _division_placement(
    group: str,
    divided: Mapping[str, OrderFacts] | UnplacedSequence,
    matched: pd.DataFrame,
) -> int | UnplacedSequence | None:
    """Return a sequence's first frame from its division row, ``None`` when undivided.

    A divided store's rows record each division's ``first_frame`` on the
    recording's frame axis. *divided* is :func:`_recording_divided` of the
    recording *group*, and *matched* are the sequence's own rows. ``None``
    means that the recording holds no division row, and its sequences are
    placed by their media frames (:class:`_MemberPlacement`).

    Returns:
        The first frame, ``None``, or why the sequence cannot be placed: the
        recording mixes division rows with other rows, or the sequence has more
        than one division row.

    Raises:
        MediaProbeError: If a row's division cells do not read.
    """
    if isinstance(divided, UnplacedSequence):
        return divided
    if not divided:
        return None
    try:
        own = [row_division(row_mapping(row)) for _, row in matched.iterrows()]
    except ValueError as exc:
        raise MediaProbeError(str(exc)) from exc
    divisions = [division for division in own if division is not None]
    if len(divisions) != 1:
        reason = (
            f"a sequence of the recording {group!r} has {len(divisions)} division "
            "rows, and a division is one row"
        )
        return UnplacedSequence(reason)
    return divisions[0].first_frame


class _MemberPlacement:
    """Places the sequences of one divided recording, for one camera, on its frame axis.

    The members are :meth:`Dataset.recording_members` of the recording, in member
    order without order facts: a natural sort on their names, as for a
    recording that no tracks run is pinned for. A member's clips are routed the
    first time a later member is placed.
    """

    def __init__(
        self,
        group: str,
        camera: str,
        members: Sequence[str],
        rows: pd.DataFrame,
        tracks: pd.DataFrame,
        route: Callable[[str, pd.DataFrame], ResolvedMedia],
    ) -> None:
        """Hold *group*'s *members*, its originals *rows* and its *tracks* index rows.

        *rows* only route a member's clips. *route* resolves a member's
        ``video_order``-sorted rows, as :meth:`Dataset.resolve_media` routes them.
        """
        self._group = group
        self._camera = camera
        self._rows = rows[rows["camera"] == camera]
        self._tracks = tracks
        self._route = route
        self._members = tuple(members)
        self._clips: dict[str, tuple[MediaFacts, ...] | str | None] = {}
        self._extents: dict[str, dict[str, tuple[int, int]]] | None = None

    def place(self, sequence: str, own: Sequence[MediaFacts]) -> int | UnplacedSequence:
        """Return *sequence*'s first frame, the media frames of the members before it.

        *own* are the facts of *sequence*'s clips. Time on the recording's axis
        is counted from frames at one rate. The clips of the members before
        *sequence* and *own* must share a rate within the store tolerance
        (:func:`~mosaic.core.media.store_rate.store_rate_mismatch`). A tracks
        variant that numbers the members' frames in another order than their
        names leaves *sequence* unplaced.

        Returns:
            The first frame, or why *sequence* cannot be placed: a member before
            it lacks media, has a clip of unknown frame count or unreadable
            clips, or the clips differ in frame rate.
        """
        disorder = self._frame_order_disagreement(sequence)
        if disorder is not None:
            return UnplacedSequence(disorder)
        earlier = self._earlier(sequence)
        total = 0
        rates: list[float] = []
        for member in earlier:
            clips = self._member_clips(member)
            if clips is None:
                why = "lacks media" + (
                    f" of camera {self._camera!r}" if self._camera else ""
                )
                return UnplacedSequence(self._unplaced(sequence, member, why))
            if isinstance(clips, str):
                return UnplacedSequence(self._unplaced(sequence, member, clips))
            counts = [int(clip.frame_count) for clip in clips]
            if min(counts) <= 0:
                why = "has a clip whose frame count is unknown"
                return UnplacedSequence(self._unplaced(sequence, member, why))
            total += sum(counts)
            rates.extend(float(clip.fps) for clip in clips)
        if not earlier:
            return 0
        own_rates = [float(clip.fps) for clip in own]
        if store_rate_mismatch([*rates, *own_rates]) is not None:
            return UnplacedSequence(self._mixed_rates(sequence, rates, own_rates))
        return total

    def opening(self, sequence: str) -> int | UnplacedSequence | None:
        """Return 0 when *sequence* starts the recording, read without its media.

        *sequence* starts the recording when no member sorts before it, and
        need not be a member. A tracks variant that numbers the frames of
        another member before it leaves it unplaced, as :meth:`place` does.

        Returns:
            0, why a tracks variant leaves *sequence* unplaced, or ``None``
            when a member sorts before it.
        """
        if self._earlier(sequence):
            return None
        disorder = self._frame_order_disagreement(sequence)
        return 0 if disorder is None else UnplacedSequence(disorder)

    def _earlier(self, sequence: str) -> tuple[str, ...]:
        """Return the members before *sequence* in member order."""
        named = {(self._group, member) for member in (*self._members, sequence)}
        order = tuple(member for _, member in order_members(named, {}))
        return order[: order.index(sequence)]

    def _member_clips(self, member: str) -> tuple[MediaFacts, ...] | str | None:
        """Return the routed facts of *member*'s clips, why they cannot be read, or ``None``.

        ``None`` means that the member lacks media of this camera.
        """
        if member not in self._clips:
            rows = self._rows[self._rows["sequence"] == member]
            if rows.empty:
                self._clips[member] = None
            else:
                sorted_rows = rows.sort_values("video_order", kind="stable")
                try:
                    self._clips[member] = tuple(self._route(member, sorted_rows).facts)
                except MediaProbeError as error:
                    self._clips[member] = f"has clips that cannot be read ({error})"
        return self._clips[member]

    def _frame_order_disagreement(self, sequence: str) -> str | None:
        """Say why a tracks variant's frame order of the members places *sequence* apart.

        A variant counts when it records disjoint frame extents for *sequence* and
        at least one other member. ``None`` when every such variant numbers the
        members before *sequence* as their names order them.
        """
        place = {member: index for index, member in enumerate(self._members)}
        for run_id, extents in self._variant_extents().items():
            if sequence not in extents or len(extents) < 2:
                continue
            by_frame = sorted(extents, key=lambda member: extents[member])
            spans = [extents[member] for member in by_frame]
            if any(left[1] >= right[0] for left, right in itertools.pairwise(spans)):
                continue
            by_name = sorted(extents, key=place.__getitem__)
            before_by_frame = set(by_frame[: by_frame.index(sequence)])
            if before_by_frame == set(by_name[: by_name.index(sequence)]):
                continue
            return (
                f"({self._group!r}, {sequence!r}) cannot be placed on the frame "
                f"axis of the divided recording {self._group!r}. Its sequences "
                f"are placed in the order of their names ({', '.join(by_name)}), "
                f"and the tracks variant {str(run_id)!r} numbers their frames in "
                f"the order {', '.join(by_frame)}. Rename the sequences so that "
                "their names sort in recording order."
            )
        return None

    def _variant_extents(self) -> dict[str, dict[str, tuple[int, int]]]:
        """Return each tracks variant's recorded frame extent of each member, read once."""
        if self._extents is None:
            members = set(self._members)
            self._extents = {
                str(run_id): {
                    str(row["sequence"]): extent
                    for _, row in rows.iterrows()
                    if str(row["sequence"]) in members
                    and (extent := read_frame_extent(row)) is not None
                }
                for run_id, rows in self._tracks.groupby("run_id", sort=True)
            }
        return self._extents

    def _mixed_rates(
        self, sequence: str, earlier: Sequence[float], own: Sequence[float]
    ) -> str:
        """Say why *sequence* cannot be placed: its clips and the earlier ones differ in rate."""
        rates = sorted({*earlier, *own})
        listed = ", ".join(f"{rate:g}" for rate in rates)
        if store_rate_mismatch(own) is not None:
            subject = f"the clips of ({self._group!r}, {sequence!r}) differ"
        else:
            subject = (
                f"({self._group!r}, {sequence!r}) and the sequences before it in "
                f"the divided recording {self._group!r} differ"
            )
        return (
            f"{subject} in frame rate ({listed} fps), and a frame's time on the "
            "recording's axis needs one rate."
        )

    def _unplaced(self, sequence: str, earlier: str, why: str) -> str:
        """Say why *sequence* cannot be placed: the member *earlier* before it *why*."""
        return (
            f"({self._group!r}, {sequence!r}) starts after the frames of the "
            f"sequences before it in the divided recording {self._group!r}, and "
            f"the sequence {earlier!r} before it {why}. Index that sequence's "
            "media, or remove it from the group."
        )


@dataclass(frozen=True)
class MediaIndexDisagreement:
    """A scoped file whose stored video_uuid differs from the injected one.

    The file on disk is no longer the file the injected facts describe. Reported
    rather than raised: by the time write_media_index runs, the file is already
    in place and the write must complete. The caller logs it and leaves a
    re-probe as the operator's repair.
    """

    basename: str
    prior_uuid: str
    injected_uuid: str


@dataclass(frozen=True)
class MediaIndexResult:
    """The written index path, plus what the write noticed about its own inputs.

    ``disagreements`` are stale overrides: a caller injected a measurement of a
    different file than the row described. ``drift`` is a file whose bytes
    moved under a stable path, found because the write re-probed it and still
    held what the row recorded. The two are kept apart because they
    call for different responses: one is an ordinary re-upload, the other is a
    source changing outside the system.

    Both are reported and neither aborts. By the time this runs the file is
    already in place, and refusing the write would leave the index describing a
    file that no longer exists -- strictly worse. Blocking a *derivation* from a
    changed source is item 6.2's job, and it fires from ``run_feature``.
    """

    index_path: Path
    disagreements: list[MediaIndexDisagreement]
    drift: list[MediaDrift] = field(default_factory=list)
    divisions: tuple[DivisionPlan, ...] = ()
    """The plan of each recording that a division scope divided."""

    @property
    def held_back(self) -> tuple[HeldBack, ...]:
        """The trailing sequences that :attr:`divisions` hold back, each with its reason."""
        return held_back_of(self.divisions)

    @property
    def refused(self) -> tuple[RefusedSpan, ...]:
        """The divisions and stores that :attr:`divisions` refuse, each with its reason."""
        return refused_of(self.divisions)


@dataclass(frozen=True)
class MediaScanResult:
    """The media index a scan wrote, and the recordings it divided.

    Attributes:
        path: The index written.
        divisions: The plan of each recording that a dividing source divided.
    """

    path: Path
    divisions: tuple[DivisionPlan, ...] = ()

    @property
    def held_back(self) -> tuple[HeldBack, ...]:
        """The trailing sequences that :attr:`divisions` hold back, each with its reason."""
        return held_back_of(self.divisions)

    @property
    def refused(self) -> tuple[RefusedSpan, ...]:
        """The divisions and stores that :attr:`divisions` refuse, each with its reason."""
        return refused_of(self.divisions)


@dataclass(frozen=True)
class _CommittedDivisionRow:
    """One division row of the media index, read for a write that divides.

    Attributes:
        path: The store's resolved path.
        camera_serial: The row's ``camera_serial``.
        division: The division, its rule and its recording.
    """

    path: Path
    camera_serial: str
    division: CommittedDivision


@dataclass(frozen=True)
class _ScannedMedia:
    """The rows that one source's scan derived, and the recordings it divided."""

    rows: list[dict[str, object]]
    divided: _DividedRows


@dataclass(frozen=True)
class _DividedRows:
    """The rows of the recordings a write divides.

    Attributes:
        rows: The division rows to write.
        plans: The plan of each recording.
        paths: The resolved paths of the stores divided, whose committed rows
            the division rows replace.
        redivided: Each recording divided again, with why, which the write
            reports once its rows replace the earlier divisions.
        display_names: The wall-clock label of each division, by entry.
    """

    rows: list[dict[str, object]] = dataclasses.field(default_factory=list)
    plans: tuple[DivisionPlan, ...] = ()
    paths: frozenset[Path] = frozenset()
    redivided: tuple[tuple[str, str], ...] = ()
    display_names: Mapping[Entry, str] = dataclasses.field(
        default_factory=dict[Entry, str]
    )

    @property
    def recordings(self) -> list[str]:
        """The names of the recordings that have a division to write."""
        return [plan.recording for plan in self.plans if plan.divisions]


@dataclass(frozen=True, slots=True)
class ConversionOutcome:
    """What one ``convert_all_tracks`` call actually did.

    ``convert_all_tracks`` used to return ``None`` and print its warnings, so a
    caller -- the CLI included -- could not tell a run that converted everything
    from one that converted nothing, and reported ``{"status": "ok"}`` either way.
    """

    converted: int
    failed: int
    variants: tuple[str, ...] = ()
    """The tracks variants this call minted, sorted. One per converter and
    correction revision it reached, whether its tables were written or skipped
    as current."""

    @property
    def ok(self) -> bool:
        """Did every sequence the run attempted convert?"""
        return self.failed == 0


@dataclass(frozen=True, slots=True)
class LabelConversionOutcome:
    """What one ``convert_all_labels`` call did.

    Attributes:
        variant: The labels variant the call minted and wrote into.
        written: Label files written by this call.
        skipped: Label files already present and kept, because the call did not
            overwrite.
    """

    variant: str
    written: int
    skipped: int


@dataclass(frozen=True, slots=True)
class MediaRows:
    """An originals media index, with the position of each listed row.

    Attributes:
        frame: The index as :meth:`Dataset.read_media_rows` read it.
        positions: The position in :attr:`frame` of each listed row, by its key:
            the resolved path of its file, or of its analysis derivative, and
            the chunks of a division row.
    """

    frame: pd.DataFrame
    positions: Mapping[MediaRowKey, int]

    def rows_of(
        self, keys: Iterable[MediaRowKey]
    ) -> dict[MediaRowKey, dict[str, object]]:
        """Return the row of each of *keys* that the index lists.

        Each row is a new ``{column: cell}`` mapping. A key that the index does
        not list is left out.
        """
        return {
            key: row_mapping(self.frame.iloc[self.positions[key]])
            for key in set(keys)
            if key in self.positions
        }


type _MediaRowsKey = tuple[IndexStamp, Path, tuple[tuple[Path, Path], ...]]
"""The originals index, the media root and the path map that rows were read under."""


class LibraryMismatchError(ValueError):
    """A library link resolves to a dataset other than the one it was made to.

    The link records the library manifest's ``uuid``. A different ``uuid`` at the
    linked path means another dataset now sits there, and resolving a model
    through it would hand back weights under a name they do not own.
    """


@dataclass(frozen=True, slots=True)
class LinkedLibrary:
    """A library link and the dataset it opened."""

    link: LibraryLink
    dataset: Dataset


def _series_of(source: ScanSource) -> str | None:
    """The label series *source* claims, or ``None`` for every other source."""
    return source.series if isinstance(source, LabelsScanSource) else None


_REPEATED_ROW_REMEDY: Final = (
    "Files of one sequence that are each numbered from frame 0, such as one file "
    "per clip, repeat their frames when merged. Number the frames across the "
    "recording in the converter."
)


def _placed_conversion(
    df: pd.DataFrame,
    hints: EntryHints,
    entry: Entry,
    source: str,
    division: OrderFacts | None,
) -> pd.DataFrame:
    """Return a converted table on its recording's axis, refusing a repeated row.

    Run at every convert site before the schema check. A division's table is
    numbered from the division's first frame and placed at it, and a table that
    reaches past the division's frames is refused, since it is numbered on the
    recording's axis already.

    Args:
        df: The converter's table.
        hints: The entry the table was converted for, placed at its division.
        entry: The entry the table is written under.
        source: The source, which a refusal names.
        division: The entry's division facts (:meth:`Dataset.division_facts`),
            ``None`` for an entry that is not a division.

    Raises:
        ValueError: If the table cannot be placed, or an individual has two rows
            of one frame. The message names *entry* and *source*.
    """
    placed = place_converted(
        df, hints, media_frame_count=None if division is None else division.frame_count
    )
    if "frame" in placed.columns:
        refuse_repeated_rows(
            placed,
            f"the converted table of entry {entry_token(*entry)!r} ({source})",
            remedy=_REPEATED_ROW_REMEDY,
        )
    return placed


_TRACKING_OUTPUT_KINDS: Final[tuple[ArtifactKind, ...]] = (
    "tracker-run",
    "frame-run",
    "frame-predictions",
    "frame-selection",
)
"""The output kinds that ``mosaic.tracking.register_ops`` registers, keyed by entry.

The prediction scores, prepared datasets and trained models that it also
registers are keyed by run, and a label series by revision, so their records
name no entry to list. Each is built from the frames of a frame run, or from the
predictions over them, which are listed for their entries while they exist.
"""

_OUTPUT_KINDS: Final[tuple[ArtifactKind, ...]] = (
    "tracks-variant",
    "labels-variant",
    "feature",
    "media-variant",
    *_TRACKING_OUTPUT_KINDS,
)
"""The artifact kinds that :meth:`Dataset.entry_outputs` lists."""

_TRACKING_OUTPUT_ROOTS: Final = ("frames", "frame_predictions", "selections", "models")
"""The roots besides ``TRACKING_ROOTS`` whose files only tracking code lists."""


def _output_label(ref: ArtifactRef) -> str:
    """Name an artifact by its kind and identifiers, as a refusal lists it."""
    return " ".join([ref.kind, *(str(value) for value in dataclasses.astuple(ref))])


def _refuse_assigned_store(rows: Iterable[Mapping[str, object]]) -> None:
    """Refuse a scan's division of a store whose identity a caller assigned.

    Raises:
        DivisionRefused: ``assigned_store``, naming the store and its entry.
    """
    for row in rows:
        if read_link_cell(row, "assignment_source") != "assigned":
            continue
        entry = entry_token(
            read_link_cell(row, "group"), read_link_cell(row, "sequence")
        )
        message = (
            f"the store {read_link_cell(row, 'abs_path')!r} is indexed as {entry!r} "
            "by Dataset.write_media_index, which owns its identity. Divide it "
            "there with a division MediaIndexScope, or scan with reassign=True."
        )
        raise DivisionRefused("assigned_store", message)


def _placed_hints(hints: EntryHints, division: OrderFacts | None) -> EntryHints:
    """Return *hints* placed at *division*'s first frame and time offset."""
    if division is None or division.first_frame is None:
        return hints
    return dataclasses.replace(
        hints, first_frame=division.first_frame, time_offset=division.time_offset
    )


def _continuous_groups_with(
    manifest: DatasetManifest, names: Sequence[str]
) -> tuple[str, ...]:
    """Return *manifest*'s continuous groups with *names* added.

    Raises:
        ValueError: If the manifest refuses the groups.
    """
    groups = tuple(dict.fromkeys([*manifest.continuous_groups, *names]))
    _check_manifest(manifest.model_copy(update={"continuous_groups": groups}))
    return groups


def _prepare_divided_manifest(manifest: DatasetManifest, names: Sequence[str]) -> None:
    """Raise *manifest* to version 4 once it accepts *names* as continuous groups.

    In place. An index of division rows is written only under a version-4
    manifest, which older code refuses to open, because it would read each
    division row as its whole store.

    Raises:
        ValueError: If the manifest refuses the groups.
    """
    _ = _continuous_groups_with(manifest, names)
    manifest.manifest_version = MANIFEST_VERSION


def _declare_divided_groups(manifest: DatasetManifest, names: Sequence[str]) -> None:
    """Add *names* to *manifest*'s continuous groups, at version 4. In place.

    Raises:
        ValueError: If the manifest refuses the groups.
    """
    manifest.continuous_groups = _continuous_groups_with(manifest, names)
    manifest.manifest_version = MANIFEST_VERSION


def _check_manifest(manifest: DatasetManifest) -> None:
    """Validate *manifest* whole, after a ``model_copy`` that ran no validators.

    Re-parsed from a dump, which is what runs every validator. That round trip is
    safe for a manifest holding file sources because a source's dump leaves out
    the knobs it never declared -- see
    :meth:`~mosaic.core.manifest.ScanSource._omit_what_was_never_declared`. Before
    it did, this re-parse made every file source look as though it had declared a
    walk, and the dataset refused every manifest edit from then on.
    """
    _ = DatasetManifest.model_validate(manifest.model_dump())


class Dataset:
    """A mosaic dataset: a manifest, the roots it declares, and the work over them.

    Constructed around a manifest *path*, not around loaded content -- ``load()``
    is a separate step, so a caller can point at a dataset that does not exist
    yet and create it.

    The manifest's own fields are reached through properties rather than copied
    onto this object, so there is one representation of what the file says.
    ``roots`` and ``meta`` hand back the live mappings, because callers across
    the toolkit mutate them in place and a copy would silently discard the edit.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        name: str = "unnamed",
        version: str = "0.1",
        roots: Mapping[str, str] | None = None,
        meta: Mapping[str, JsonValue] | None = None,
        manifest: DatasetManifest | None = None,
    ) -> None:
        """Build a dataset around *manifest_path*.

        Args:
            manifest_path: The manifest file, or the directory holding it.
            name: Dataset name, when constructing without loading.
            version: Dataset version, when constructing without loading.
            roots: Root declarations. Defaults to one empty value per known
                root, which reads as "declared nothing yet" -- ``get_root``
                treats an empty root as unset.
            meta: Structured metadata, when constructing without loading.
            manifest: A manifest to adopt wholesale. Overrides the other fields.
        """
        self.manifest_path: Path = user_path(manifest_path)
        if manifest is not None:
            self.manifest: DatasetManifest = manifest
        else:
            self.manifest = DatasetManifest(
                name=name,
                version=version,
                roots=dict(roots) if roots is not None else empty_root_template(),
                meta=dict(meta) if meta is not None else {},
            )
        self._path_map: list[tuple[Path, Path]] = []
        self._linked: (
            tuple[tuple[LibraryLink, ...], tuple[LinkedLibrary, ...]] | None
        ) = None
        self._media_rows: dict[str, tuple[_MediaRowsKey, MediaRows]] = {}

    def __repr__(self) -> str:
        return f"Dataset(manifest_path={self.manifest_path!r}, name={self.name!r})"

    # ---- Manifest fields ----

    @property
    def name(self) -> str:
        """What the dataset is called."""
        return self.manifest.name

    @name.setter
    def name(self, value: str) -> None:
        self.manifest.name = value

    @property
    def version(self) -> str:
        """The dataset's own version string, for whoever curates it."""
        return self.manifest.version

    @version.setter
    def version(self, value: str) -> None:
        self.manifest.version = value

    @property
    def uuid(self) -> str | None:
        """Minted once at creation and never rewritten."""
        return self.manifest.uuid

    @property
    def created_at(self) -> str | None:
        """When the manifest was first written, ISO-8601 UTC."""
        return self.manifest.created_at

    @property
    def roots(self) -> dict[str, str]:
        """The declared roots, live.

        Mutating the returned mapping edits the manifest, which is deliberate:
        callers across the toolkit assign into ``ds.roots[key]`` directly, and
        handing back a copy would swallow the edit rather than report it.
        Persisting the change is still :meth:`save`'s job.
        """
        return self.manifest.roots

    @roots.setter
    def roots(self, value: dict[str, str]) -> None:
        self.manifest.roots = value

    @property
    def meta(self) -> dict[str, JsonValue]:
        """Structured per-subsystem metadata, live. See :attr:`roots`."""
        return self.manifest.meta

    @meta.setter
    def meta(self, value: dict[str, JsonValue]) -> None:
        self.manifest.meta = value

    def meta_section(self, key: str) -> dict[str, JsonValue]:
        """The nested mapping ``meta[key]``, creating it if absent.

        ``meta`` is deliberately open-ended, so a value read out of it is
        anything JSON can hold. The two shapes mosaic actually stores there are a
        nested section and a number, and these two accessors are where that gets
        checked -- once, rather than at each of the call sites that would
        otherwise index into an unknown.

        A key holding something that is *not* a mapping is replaced by an empty
        one. The alternative is raising deep inside a converter over a manifest
        somebody hand-edited, which helps nobody: the section is metadata about
        derived files, and the converter is about to rewrite it anyway.
        """
        section = self.meta.get(key)
        if not isinstance(section, dict):
            section = {}
            self.meta[key] = section
        return section

    @overload
    def meta_float(self, key: str) -> float | None: ...

    @overload
    def meta_float(self, key: str, default: float) -> float: ...

    def meta_float(self, key: str, default: float | None = None) -> float | None:
        """``meta[key]`` as a number, or *default* when it is absent or is not one.

        Overloaded so a caller supplying a fallback gets a ``float`` rather than
        an optional one, and does not have to re-handle a ``None`` it already
        ruled out.

        ``bool`` is excluded: ``isinstance(True, int)`` is true, and a stray
        ``fps_default: true`` becoming 1.0 frames per second is worse than
        falling back to the default.
        """
        value = self.meta.get(key)
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        return default

    @property
    def sources(self) -> ScanSources:
        """The declared scan sources. Change them through the scan-source methods."""
        return self.manifest.sources

    # ---- Notes and typed tags ----

    @property
    def notes(self) -> str:
        """Free text that travels with the dataset."""
        return self.manifest.notes

    def set_notes(self, text: str, *, save: bool = True) -> None:
        """Replace the dataset's notes.

        Raises:
            ValueError: If *text* is longer than the manifest allows. Long prose
                belongs in a file the dataset references, not in the manifest
                every reader parses.
        """
        self._adopt_manifest({"notes": text})
        if save:
            self.save()

    @property
    def tags(self) -> tuple[DatasetTag, ...]:
        """The dataset's typed tags, ordered as they are written.

        These describe the *dataset*. The per-sequence tags that group sequences
        for analysis are a different thing, owned by mosaic-api.
        """
        return self.manifest.ordered_tags()

    def tag(self, name: str) -> DatasetTag | None:
        """The tag called *name*, matched case-insensitively, or ``None``."""
        return self.manifest.tag(name)

    def has_tag(self, name: str) -> bool:
        """Whether *name* is attached. The question a ``label`` tag answers."""
        return self.manifest.tag(name) is not None

    def tag_value(self, name: str) -> TypedAttributeValue:
        """The value of tag *name*, or ``None`` if it is absent or is a label."""
        found = self.manifest.tag(name)
        return None if found is None else found.value

    @property
    def continuous_groups(self) -> tuple[str, ...]:
        """Groups whose sequences are time divisions of one recording."""
        return self.manifest.continuous_groups

    def is_continuous_group(self, group: str) -> bool:
        """Whether *group*'s sequences divide one recording. See the manifest."""
        return self.manifest.is_continuous_group(group)

    def recording_kind(self, group: str) -> RecordingKind:
        """Return whether an entry of *group* belongs to a divided recording."""
        return recording_kind(self.manifest.continuous_groups, group)

    def recording_of(self, group: str, sequence: str) -> RecordingKey:
        """Return the key of the recording that ``(group, sequence)`` belongs to.

        ``(group, "")`` for a sequence of a continuous group, and the entry's
        own pair otherwise. See :mod:`mosaic.core.recording`.
        """
        return recording_of(self.manifest.continuous_groups, group, sequence)

    def entry_key_collision(
        self, written: Iterable[Entry], existing: Iterable[Entry]
    ) -> str | None:
        """Return why a new entry of *written* cannot be indexed beside *existing*.

        A new entry may not share its key with another entry
        (:func:`~mosaic.core.entry.entry_key_collision`), and an undivided one may
        not take the name of a continuous group
        (:func:`~mosaic.core.recording.recording_name_collision`). An entry of
        *existing* is not new.

        Returns:
            The reason, or ``None`` when every new entry may be written.
        """
        held = set(existing)
        batch = list(written)
        return entry_key_collision(batch, held) or recording_name_collision(
            self.manifest.continuous_groups, batch, held
        )

    def refuse_entry_key_collision(
        self, written: Iterable[Entry], existing: Iterable[Entry]
    ) -> None:
        """Refuse the collision that :meth:`entry_key_collision` reports.

        Raises:
            EntryKeyCollisionError: Naming the new entry and what it collides with.
        """
        reason = self.entry_key_collision(written, existing)
        if reason is not None:
            raise EntryKeyCollisionError(reason)

    def recordings(
        self,
        entries: Iterable[Entry] | None = None,
        *,
        order_facts: Mapping[Entry, OrderFacts] | None = None,
    ) -> tuple[Recording, ...]:
        """Return every recording that *entries* make up, sorted by key.

        Args:
            entries: The entries to group. ``None`` reads the media index and the
                tracks index and takes every entry either names.
            order_facts: What places each member in its recording. ``None``
                orders members by a natural sort on their names. The tracks
                frame extents of a pinned run order a hand-declared group:
                ``facts_from_frame_extents(read_frame_extents(ds, run_id))``.
        """
        universe = self.recording_entries() if entries is None else entries
        return recordings(self.manifest.continuous_groups, universe, order_facts)

    def recording_members(
        self,
        key: RecordingKey,
        entries: Iterable[Entry] | None = None,
        *,
        order_facts: Mapping[Entry, OrderFacts] | None = None,
    ) -> tuple[Entry, ...]:
        """Return the members of the recording that *key* names, in recording order.

        An undivided recording has its one entry. A divided one has every entry
        of its group among *entries*, ordered as :meth:`recordings` orders them.
        A key that names no recording has no members: ``(G, "")`` for an
        undeclared group, or ``(G, "s")`` for a declared one.

        Args:
            key: The recording.
            entries: The entries to find a divided recording's members among.
                ``None`` reads the media index and the tracks index and takes
                every entry either names.
            order_facts: What places each member in its recording, as for
                :meth:`recordings`.
        """
        if self.recording_kind(key.group) == "undivided":
            return ((key.group, key.sequence),) if key.sequence else ()
        if key.sequence:
            return ()
        universe = self.recording_entries() if entries is None else entries
        group_entries = (entry for entry in universe if entry[0] == key.group)
        return order_members(group_entries, order_facts or {})

    def recording_entries(self) -> set[Entry]:
        """Return every entry that the media index or the tracks index names.

        These are the entries that :meth:`recordings` groups, a media-only entry
        among them. Media entries are named as media resolution names them.
        """
        universe = root_index_entries(self, "tracks")
        # Named as media resolution names them, once the index contains a row.
        if media_index_entries(self):
            universe |= self.media_entries()
        return universe

    def _indexed_entries(self) -> set[Entry]:
        """Return every entry of the media, tracks_raw, tracks and labels indexes."""
        return (
            media_index_entries(self)
            | root_index_entries(self, "tracks_raw")
            | self._tracks_and_labels_entries()
        )

    def _tracks_and_labels_entries(self) -> set[Entry]:
        """Return every entry of the tracks and labels indexes."""
        found = root_index_entries(self, "tracks")
        for kind in label_kinds(self):
            found |= index_entries(labels_index_path(self, kind))
        return found

    def _entries_beside_media_write(
        self,
        written: set[Entry],
        committed: Iterable[Mapping[str, object]],
        preserved: Iterable[Mapping[str, object]],
    ) -> set[Entry]:
        """Return the entries whose keys a replacing media write must not reuse."""
        return entries_kept_beside(
            written,
            committed,
            preserved,
            held_elsewhere=self._indexed_entries_beside_media(),
        )

    def _entries_beside_tracks_raw_write(
        self,
        written: set[Entry],
        committed: Iterable[Mapping[str, object]],
        preserved: Iterable[Mapping[str, object]],
    ) -> set[Entry]:
        """Return the entries whose keys a replacing tracks_raw write must not reuse."""
        media = media_index_entries(self)
        held_elsewhere = media | self._tracks_and_labels_entries()
        return media | entries_kept_beside(
            written, committed, preserved, held_elsewhere=held_elsewhere
        )

    def measure_frame_extents(self, *, dry_run: bool = False) -> "pd.DataFrame":
        """Record the frame extent of every tracks row that lacks one.

        A dataset converted before the extent was recorded reads blank, and an
        ``overlap_frames`` run refuses a continuous group with a blank extent. Run
        this once to make such a dataset eligible.
        Returns the rows filled, with their measured values.
        """
        return backfill_frame_extents(self, dry_run=dry_run)

    def measure_media_frames(self, *, dry_run: bool = False) -> "CellBackfill":
        """Rewrite every row's count of frames the tool should have read.

        The media side of the comparison that :meth:`frame_axis_mismatches`
        makes, for tables already on disk. Only TRex can re-bridge such a table
        without re-tracking (a republish). A row is rewritten by the rule its
        producer records it by: a count where the run read the whole entry, and
        a blank where it read less on purpose or no media at all. A row whose run
        cannot be established keeps its value, which the result reports in
        ``not_established``. So does a tracker row in a process that has not
        called ``mosaic.tracking.register_ops``, which the result reports in
        ``unregistered``. See :func:`backfill_media_frames`.
        """
        return backfill_media_frames(self, dry_run=dry_run)

    def measure_frames_read(self, *, dry_run: bool = False) -> "CellBackfill":
        """Rewrite every row's count of frames the tool read.

        The tool's side of the comparison that :meth:`frame_axis_mismatches`
        makes, from what each run left on disk (:func:`backfill_frames_read`). A
        row whose run left nothing that tells keeps its value, and so does a
        tracker row in a process that has not called
        ``mosaic.tracking.register_ops``, which the result reports in
        ``unregistered``.
        """
        return backfill_frames_read(self, dry_run=dry_run)

    def measure_known_tail_loss(self, *, dry_run: bool = False) -> "CellBackfill":
        """Rewrite every row's known loss at the end of the file its tool read.

        A shortfall within it counts as a known tail loss rather than a mismatch
        (:func:`backfill_known_tail_loss`). A row keeps its value when its file
        cannot be found or its header read, or its media has changed since the
        run. So does a tracker row in a process that has not called
        ``mosaic.tracking.register_ops``, which the result reports in
        ``unregistered``.
        """
        return backfill_known_tail_loss(self, dry_run=dry_run)

    def frame_axis_mismatches(
        self, run_id: str | None = None
    ) -> "tuple[FrameAxisMismatch, ...]":
        """Tracks tables whose tool's frame count differs from their media's.

        A shortfall within a known tail loss is left out
        (:meth:`frame_tail_shortfalls`).

        One record per disagreeing table, naming its variant as well as its
        entry, from what the index already records -- a table measured on
        neither side is absent rather than reported as agreeing. See
        :func:`~mosaic.core.pipeline.tracks_index.frame_axis_mismatches`.
        """
        return frame_axis_mismatches(self, run_id)

    def frame_tail_shortfalls(
        self, run_id: str | None = None
    ) -> "tuple[FrameTailShortfall, ...]":
        """Tracks tables whose tool read short by no more than its producer declares.

        The shortfalls that :meth:`frame_axis_mismatches` leaves out, as a known
        loss at the end of a file. See
        :func:`~mosaic.core.pipeline.tracks_index.frame_tail_shortfalls`.
        """
        return frame_tail_shortfalls(self, run_id)

    def set_continuous_groups(
        self, groups: Iterable[str], *, save: bool = True
    ) -> None:
        """Declare which groups are each one recording divided into sequences.

        The list replaces the declared one. Each name must already be the group
        of an indexed entry. A manifest declaring a name refused here still
        loads.

        A recording whose store the media index divides stays declared: its
        division rows place its sequences.

        Raises:
            ValueError: If a name is empty or repeated, contains ``__`` or
                ``:``, is the name of a sequence in the empty group, or is the
                group of no entry in the media, tracks_raw, tracks or labels
                indexes, or if the list leaves out a recording whose store the
                media index divides.
        """
        named = tuple(str(group) for group in groups)
        seen: set[str] = set()
        for name in named:
            if not name:
                msg = (
                    "continuous_groups names the empty group. A continuous group "
                    "names the recording that its sequences divide."
                )
                raise ValueError(msg)
            if name in seen:
                msg = f"continuous_groups lists {name!r} more than once."
                raise ValueError(msg)
            seen.add(name)
        indexed = self._indexed_entries()
        groups_held = {group for group, _ in indexed}
        empty_group_keys = {
            to_safe_name(sequence) for group, sequence in indexed if not group
        }
        for name in named:
            for separator in ("__", ":"):
                if separator in name:
                    msg = (
                        f"continuous group {name!r} contains {separator!r}, which "
                        "joins a group to a sequence in an entry key or token. "
                        "Rename the group."
                    )
                    raise ValueError(msg)
            if to_safe_name(name) in empty_group_keys:
                msg = (
                    f"continuous group {name!r} has the name of the empty-group "
                    f"sequence {name!r}, and the two recordings would share one "
                    "name. Rename one of them."
                )
                raise ValueError(msg)
            if name not in groups_held:
                msg = (
                    f"continuous group {name!r} is the group of no entry in the "
                    "media, tracks_raw, tracks or labels indexes. Index its "
                    "sequences first."
                )
                raise ValueError(msg)
        dropped = sorted(self._divided_groups() - set(named))
        if dropped:
            msg = (
                f"continuous_groups leaves out {dropped}, whose stores the media "
                "index divides. Their division rows place their sequences on "
                "one frame axis."
            )
            raise ValueError(msg)
        self._declare_continuous_groups(named, save=save)

    def _divided_groups(self) -> set[str]:
        """The groups of the originals index's division rows."""
        return {
            read_link_cell(row, "group")
            for row in self.read_media_index()
            if media_row_key(row).window is not None
        }

    def _declare_continuous_groups(
        self, groups: tuple[str, ...], *, save: bool = True
    ) -> None:
        """Record *groups* as the continuous groups under the manifest rules alone.

        Raises:
            ValueError: If a name is empty or repeated.
        """
        self._adopt_manifest({"continuous_groups": groups})
        if save:
            self.save()

    @property
    def splits(self) -> dict[str, SplitRole]:
        """The role of each named group in a held-out protocol, as a copy.

        See :mod:`mosaic.core.splits` for reading one group's role.
        """
        return dict(self.manifest.splits)

    def set_splits(self, role: SplitRole, groups: Iterable[str]) -> None:
        """Give each of *groups* the role *role*, replacing any role it had.

        The manifest is re-read and written under the index lock, so two
        operators changing roles at once both land.

        Raises:
            ValueError: If *role* is not ``train``, ``test`` or ``sealed``, or a
                group name is empty. The manifest is left as it was.
        """
        named = [str(group) for group in groups]

        def assign(manifest: DatasetManifest) -> None:
            merged: dict[str, SplitRole] = {
                **manifest.splits,
                **dict.fromkeys(named, role),
            }
            _check_manifest(manifest.model_copy(update={"splits": merged}))
            manifest.splits = merged

        self.mutate_manifest(assign)

    def remove_splits(self, groups: Iterable[str]) -> list[str]:
        """Drop the role of each of *groups*. Returns the groups that had one.

        Takes the same lock as :meth:`set_splits`. A group with no role is
        skipped rather than refused, so removing twice is the same as once.
        """
        named = [str(group) for group in groups]
        removed: list[str] = []

        def drop(manifest: DatasetManifest) -> None:
            removed.clear()
            kept = dict(manifest.splits)
            for group in named:
                if kept.pop(group, None) is not None:
                    removed.append(group)
            manifest.splits = kept

        self.mutate_manifest(drop)
        return removed

    def define_tag(self, tag: DatasetTag, *, save: bool = True) -> None:
        """Declare *tag*, replacing any existing one of the same name.

        Redefining re-validates the value against the new constraints and raises
        if it no longer fits. That is the whole of what mosaic-api's narrowing
        machinery does, one row wide: here there is exactly one holder, so there
        are no other assignments for a constraint change to invalidate.
        """
        others = [
            existing
            for existing in self.manifest.tags
            if existing.name.casefold() != tag.name.casefold()
        ]
        self._adopt_manifest({"tags": (*others, tag)})
        if save:
            self.save()

    def set_tag_value(
        self, name: str, value: TypedAttributeValue, *, save: bool = True
    ) -> None:
        """Set the value of an already-defined tag, keeping its declared type.

        Raises:
            KeyError: If no tag of that name is defined. Defining and valuing are
                separate because a value alone cannot say what type it is, and
                guessing one would let ``"12"`` and ``12`` mean different things
                on different days.
            ValueError: If *value* does not satisfy the tag's constraints.
        """
        found = self.manifest.tag(name)
        if found is None:
            declared = sorted(existing.name for existing in self.manifest.tags)
            msg = f"no tag named {name!r}; defined: {declared or 'none'}"
            raise KeyError(msg)
        # Rebuilt rather than copied with an update: ``model_copy`` does not run
        # validators, so an off-vocabulary value would only be caught later by
        # the whole-manifest check, which reports it by list position instead of
        # by name.
        replacement = DatasetTag(
            name=found.name,
            type=found.type,
            type_constraints=dict(found.type_constraints),
            value=value,
            description=found.description,
            display_order=found.display_order,
        )
        self.define_tag(replacement, save=save)

    def remove_tag(self, name: str, *, save: bool = True) -> bool:
        """Drop the tag called *name*. Returns whether one was there."""
        remaining = [
            existing
            for existing in self.manifest.tags
            if existing.name.casefold() != name.casefold()
        ]
        if len(remaining) == len(self.manifest.tags):
            return False
        self.manifest = self.manifest.model_copy(update={"tags": tuple(remaining)})
        if save:
            self.save()
        return True

    def _adopt_manifest(self, update: Mapping[str, object]) -> None:
        """Replace the fields named in *update* once the manifest they make is valid.

        A refused manifest is not held. A later :meth:`save` therefore cannot
        write it.

        Raises:
            ValueError: If the manifest with *update* applied fails validation.
        """
        candidate = self.manifest.model_copy(update=dict(update))
        _check_manifest(candidate)
        self.manifest = candidate

    def mutate_manifest(self, change: Callable[[DatasetManifest], None]) -> None:
        """Re-read the manifest, apply *change*, and write it back under a lock.

        The read-modify-write seam, for anything editing a manifest another
        process may also be editing -- mosaic-api setting notes while a CLI
        declares a source, say. :meth:`save` writes the file whole and
        atomically, so without the lock the loser of a race is not a merged file
        but a silently discarded one.

        *change* is handed the freshly read manifest and mutates it in place.
        Whatever this dataset held is replaced by the result, so the in-memory
        copy and the file agree afterwards.
        """
        with index_lock(self.manifest_path):
            current = (
                read_manifest(self.manifest_path)
                if self.manifest_path.exists()
                else self.manifest
            )
            change(current)
            self.manifest = current
            self._ensure_roots()
            write_manifest(self.manifest_path, current)

    # ---- Linked libraries ----

    @property
    def libraries(self) -> tuple[LibraryLink, ...]:
        """The datasets a model reference may resolve through, in lookup order."""
        return self.manifest.libraries

    def add_library(self, link: LibraryLink, *, save: bool = True) -> LibraryLink:
        """Declare *link* and persist the manifest. Returns the link as stored.

        The target is opened once, here, so a link that names nothing fails at
        the gesture that made it and not at the first inference that needs it.
        A link declared without a ``uuid`` records the target's, so that the
        library moving and a different dataset taking its place are told apart
        from then on.

        Args:
            link: The link to declare.
            save: Write the manifest. Pass ``False`` to batch several changes.

        Raises:
            ValueError: If the id is taken, or the link points at this dataset.
            FileNotFoundError: If no manifest is at the linked path.
            LibraryMismatchError: If *link* names a ``uuid`` the target does not
                carry.
        """
        if any(existing.id == link.id for existing in self.manifest.libraries):
            msg = f"a library named {link.id!r} is already linked"
            raise ValueError(msg)
        target = self._open_library(link)
        if target.manifest_path.resolve() == self.manifest_path.resolve():
            msg = f"library {link.id!r} points at this dataset; a dataset cannot link itself"
            raise ValueError(msg)
        stored = link.model_copy(
            update={
                "uuid": link.uuid or (target.uuid or ""),
                "added_at": link.added_at or now_stamp(),
            }
        )
        self._adopt_manifest({"libraries": (*self.manifest.libraries, stored)})
        if save:
            self.save()
        return stored

    def remove_library(self, link_id: str, *, save: bool = True) -> bool:
        """Drop the link called *link_id*. Returns whether one was there.

        Nothing on disk changes. A run that named a model by run identifier
        keeps the identifier; it stops resolving here until a link, or a copy of
        the model in this dataset's own ``models/``, provides it again.
        """
        remaining = tuple(
            existing for existing in self.manifest.libraries if existing.id != link_id
        )
        if len(remaining) == len(self.manifest.libraries):
            return False
        self.manifest = self.manifest.model_copy(update={"libraries": remaining})
        if save:
            self.save()
        return True

    def linked_libraries(self) -> tuple[LinkedLibrary, ...]:
        """Every linked library that is reachable now, opened, in lookup order.

        One level deep: a library's own links are never followed, so a lookup
        terminates and a model is served by a dataset this manifest names.

        An unreachable link is skipped with a warning, the tolerance a source on
        an unmounted share gets, because a dataset must stay usable while a
        library is offline. A reachable target carrying the wrong ``uuid`` is not
        tolerated: that is a different dataset, not an absent one.

        Raises:
            LibraryMismatchError: If a link's recorded ``uuid`` disagrees with
                the dataset found at its path.
        """
        links = self.manifest.libraries
        if self._linked is not None and self._linked[0] == links:
            return self._linked[1]
        opened: list[LinkedLibrary] = []
        for link in links:
            try:
                target = self._open_library(link)
            except FileNotFoundError:
                where = self._resolve_declared_path(link.path)
                print(
                    f"[libraries] {link.id!r} is not reachable at {where}; "
                    "models it holds will not resolve until it is.",
                    file=sys.stderr,
                )
                continue
            opened.append(LinkedLibrary(link=link, dataset=target))
        found = tuple(opened)
        # Cached only when every link opened. A library that was offline must be
        # looked for again, or a long-lived process would go on reporting its
        # models missing after the share came back.
        if len(found) == len(links):
            self._linked = (links, found)
        return found

    def _open_library(self, link: LibraryLink) -> Dataset:
        """Open the dataset *link* names, checking it is the one the link recorded."""
        where = self._resolve_declared_path(link.path)
        target = Dataset(manifest_path=resolve_manifest_path(where)).load(
            ensure_roots=False
        )
        if link.uuid and target.uuid and link.uuid != target.uuid:
            msg = (
                f"library {link.id!r} was linked to dataset {link.uuid}, but "
                f"{where} holds dataset {target.uuid} ({target.name!r}). If the "
                "library moved, repoint the link at where it is now; if this is "
                "the intended dataset, remove the link and add it again."
            )
            raise LibraryMismatchError(msg)
        return target

    # ---- Declared scan sources ----

    def scan_sources(
        self, kind: SourceKind, *, only: Sequence[str] = ()
    ) -> tuple[AnyScanSource, ...]:
        """The declared sources feeding *kind*, or just those *only* names.

        Args:
            kind: Which raw root's sources to return.
            only: Source ids to restrict to. Empty means all of them.

        Returns:
            The selected sources, in declaration order.

        Raises:
            KeyError: If *only* names an id this kind does not declare.
        """
        return self.manifest.sources.select(kind, only)

    def add_scan_source(self, source: AnyScanSource, *, save: bool = True) -> None:
        """Declare *source* and persist the manifest.

        **Not validated inside the dataset**, unlike :meth:`set_root`. A source
        exists to name storage elsewhere; its files are recorded by absolute
        ``abs_path`` into an index that stays inside, which is the arrangement
        that replaced an outside root. Its directory is never created either --
        a scan discovers whether it is there.

        Args:
            source: The source to declare. Its ``kind`` picks the list.
            save: Write the manifest. Pass ``False`` to batch several changes.

        Raises:
            ValueError: If the id is already taken for this kind, or the source
                would claim files another source of the same kind already claims.
        """
        kind: SourceKind = source.kind
        existing = list(self.manifest.sources.of_kind(kind))
        if any(candidate.id == source.id for candidate in existing):
            msg = f"a {kind} source named {source.id!r} is already declared"
            raise ValueError(msg)
        stamped = source.model_copy(update={"added_at": source.added_at or now_stamp()})
        proposed = [*existing, stamped]
        overlap = overlapping_sources(proposed, self._resolve_declared_path)
        if overlap is not None:
            left, right = overlap
            msg = (
                f"{kind} sources {left!r} and {right!r} would claim the same "
                "files, which makes it ambiguous which recipe identifies them. "
                "Narrow one of the paths, or list files explicitly so the two "
                "claims are disjoint."
            )
            raise ValueError(msg)
        self.manifest.sources = self.manifest.sources.with_kind(kind, proposed)
        if save:
            self.save()

    def remove_scan_source(
        self, kind: SourceKind, source_id: str, *, save: bool = True
    ) -> int:
        """Undeclare a source. Returns how many index rows it was claiming.

        The rows stay. Undeclaring is a statement about future scans; dropping
        rows would delete composition membership and move downstream identity
        hashes, which is a repair worth asking for explicitly (``mosaic reindex``
        and ``mosaic prune-media`` both exist and both default to a dry run).

        Raises:
            KeyError: If *kind* declares no source with that id.
        """
        existing = list(self.manifest.sources.of_kind(kind))
        match = next((s for s in existing if s.id == source_id), None)
        if match is None:
            declared = sorted(s.id for s in existing)
            msg = (
                f"no {kind} source named {source_id!r}; declared: {declared or 'none'}"
            )
            raise KeyError(msg)
        orphaned = self._rows_claimed_by(
            kind, self.source_claim(match), series=_series_of(match)
        )
        remaining = [s for s in existing if s.id != source_id]
        self.manifest.sources = self.manifest.sources.with_kind(kind, remaining)
        if save:
            self.save()
        return orphaned

    def declare_scan_source(self, source: AnyScanSource) -> None:
        """Declare *source* in the manifest on disk, under the manifest's lock.

        :meth:`add_scan_source` applied to the manifest as it reads under the
        lock (:meth:`mutate_manifest`), so a writer in another process loses
        nothing. The writer that an agent's tools call.

        Raises:
            ValueError: As :meth:`add_scan_source` raises.
        """
        self._edit_manifest_locked(lambda: self.add_scan_source(source, save=False))

    def undeclare_scan_source(self, kind: SourceKind, source_id: str) -> int:
        """Undeclare a source under the manifest's lock, as :meth:`declare_scan_source`.

        Returns:
            How many index rows the source was claiming. They stay, as
            :meth:`remove_scan_source` keeps them.

        Raises:
            KeyError: If *kind* declares no source with that id.
        """
        return self._edit_manifest_locked(
            lambda: self.remove_scan_source(kind, source_id, save=False)
        )

    def update_tags(
        self,
        values: Mapping[str, TypedAttributeValue],
        *,
        remove: Sequence[str] = (),
    ) -> None:
        """Set and remove dataset tags under the manifest's lock.

        A value for a defined tag is checked against its type, as
        :meth:`set_tag_value` checks it. A value for a tag not defined defines
        one, typed by the value: ``None`` a ``label``, a boolean ``bool``, an
        integer ``int``, a number ``float`` and a string ``text``.

        Raises:
            KeyError: *remove* names a tag that is not defined.
            ValueError: A value does not fit its tag.
        """

        def edit() -> None:
            for name, value in values.items():
                if self.manifest.tag(name) is None:
                    tag = DatasetTag(name=name, type=_tag_type_of(value), value=value)
                    self.define_tag(tag, save=False)
                else:
                    self.set_tag_value(name, value, save=False)
            for name in remove:
                if not self.remove_tag(name, save=False):
                    declared = sorted(tag.name for tag in self.manifest.tags)
                    msg = f"no tag named {name!r}; defined: {declared or 'none'}"
                    raise KeyError(msg)

        self._edit_manifest_locked(edit)

    def _edit_manifest_locked[T](self, edit: Callable[[], T]) -> T:
        """Run *edit*, which changes ``self.manifest`` without saving, under the lock.

        *edit* sees the manifest as it reads under the lock, and its result is
        written. An edit that replaced the manifest object (``model_copy``)
        is copied onto the one :meth:`mutate_manifest` writes.
        """
        found: list[T] = []

        def change(current: DatasetManifest) -> None:
            self.manifest = current
            found.append(edit())
            edited = self.manifest
            if edited is not current:
                for name in type(current).model_fields:
                    setattr(current, name, getattr(edited, name))

        self.mutate_manifest(change)
        return found[0]

    def add_source_files(
        self, kind: SourceKind, source_id: str, files: Sequence[str]
    ) -> int:
        """Add *files* to a file-mode source. Returns how many were new.

        The durable form of an import that selects some of a folder's contents:
        a second selection extends the source that already describes that folder
        rather than minting one per gesture.

        Raises:
            KeyError: If no such source is declared.
            ValueError: If the source is a directory source, which claims its
                files by glob and has no list to extend.
        """
        return self._edit_source_files(kind, source_id, files, adding=True)

    def remove_source_files(
        self, kind: SourceKind, source_id: str, files: Sequence[str]
    ) -> int:
        """Drop *files* from a file-mode source. Returns how many were removed.

        **Their index rows go too**, unlike :meth:`remove_scan_source`, which
        keeps them. The difference is how specific the gesture is: undeclaring a
        whole source says nothing about which of its rows are still wanted, while
        naming individual files is the un-import of exactly those. Leaving them
        behind would make the index describe files the dataset no longer claims
        from anywhere, with nothing left to rescan and clean them up.
        """
        removed = self._edit_source_files(kind, source_id, files, adding=False)
        if removed:
            source = next(
                s for s in self.manifest.sources.of_kind(kind) if s.id == source_id
            )
            base = self.resolve_source_path(source)
            self.drop_claimed_rows(
                kind,
                ScanClaim.over_files(base / entry for entry in files),
                series=_series_of(source),
            )
        return removed

    def drop_claimed_rows(
        self, kind: SourceKind, claim: ScanClaim, *, series: str | None = None
    ) -> int:
        """Remove every row of *kind*'s index inside *claim*. Returns the count.

        Public because ``mosaic sources remove --drop-rows`` is the gesture that
        wants it: undeclaring keeps rows by default, and this is the opt-in.

        *series* names the label series the source claimed. Its rows are in that
        series' index, and dropping them touches no composition, because a series
        row never entered one.
        """
        if series is not None:
            index_path = series_index_path(self, series)
            if not index_path.exists():
                return 0
            with index_lock(index_path):
                revisions = index_records(read_label_series(self, series))
                remaining = [
                    row for row in revisions if not self._row_claimed(row, claim)
                ]
                if len(remaining) == len(revisions):
                    return 0
                write_label_series_rows(index_path, remaining)
            return len(revisions) - len(remaining)
        if kind == "media":
            index_path = self.get_root(self.resolve_media_root()) / "index.csv"
            with index_lock(index_path):
                committed = _read_media_index(index_path)
                kept: list[dict[str, object]] = [
                    dict(row) for row in committed if not self._row_claimed(row, claim)
                ]
                if len(kept) == len(committed):
                    return 0
                write_media_index_rows(index_path, frame_from_rows(kept))
            self._write_media_compositions(kept)
            return len(committed) - len(kept)

        index_path = self.get_root(_RAW_ROOT_FOR_KIND[kind]) / "index.csv"
        with index_lock(index_path):
            committed = _read_tracks_raw_index(index_path)
            raw_kept: list[dict[str, object]] = [
                dict(row) for row in committed if not self._row_claimed(row, claim)
            ]
            if len(raw_kept) == len(committed):
                return 0
            write_tracks_raw_index_rows(index_path, _tracks_frame_from_rows(raw_kept))
        if kind == "labels":
            self._write_labels_raw_compositions(raw_kept)
        else:
            self._write_tracks_raw_compositions(raw_kept)
        return len(committed) - len(raw_kept)

    def _edit_source_files(
        self, kind: SourceKind, source_id: str, files: Sequence[str], *, adding: bool
    ) -> int:
        existing = list(self.manifest.sources.of_kind(kind))
        match = next((s for s in existing if s.id == source_id), None)
        if match is None:
            declared = sorted(s.id for s in existing)
            msg = (
                f"no {kind} source named {source_id!r}; declared: {declared or 'none'}"
            )
            raise KeyError(msg)
        if match.mode != "files":
            msg = (
                f"{kind} source {source_id!r} is a directory source: it claims "
                "whatever its globs match, so there is no file list to edit"
            )
            raise ValueError(msg)
        current = list(match.files)
        if adding:
            additions = [entry for entry in files if entry not in current]
            updated = [*current, *additions]
            changed = len(additions)
        else:
            removals = {entry for entry in files if entry in current}
            updated = [entry for entry in current if entry not in removals]
            changed = len(removals)
        if not changed:
            return 0
        replacement = match.model_copy(update={"files": tuple(updated)})
        self.manifest.sources = self.manifest.sources.with_kind(
            kind, [replacement if s.id == source_id else s for s in existing]
        )
        self.save()
        return changed

    def scan_media(
        self,
        *,
        only: Sequence[str] = (),
        reassign: bool = False,
        prune_unsourced: bool = False,
        index_filename: str = "index.csv",
        include_partial: bool = False,
    ) -> MediaScanResult:
        """Rescan every declared media source and rewrite ``media_raw/index.csv``.

        Each source is scanned with its own recipe -- its extensions, its layout,
        its match mode -- so one dataset can draw from a NAS folder of ``.mp4``
        under one layout and an explicit import selection under another. The
        replace scope is the union of what those sources claim, so a row under
        none of them survives untouched.

        The originals index, never the derivative one: the destination is
        :meth:`resolve_media_root`, which is ``media_raw`` for any dataset that
        declares it and ``media`` only on one predating the split. Transcode
        derivatives are written to ``media/index.csv`` by the transcode op and
        are never touched here.

        A source with a ``division`` divides each imgstore recording it finds
        into sequences of whole chunks, as a division ``MediaIndexScope`` does
        (:meth:`write_media_index`), and declares each recording continuous. A
        rescan measures each store again, keeps its divisions, and adds the
        divisions that closed since. A store whose identity a caller assigned is
        refused unless *reassign* is set.

        Args:
            only: Restrict to these source ids. The declaration is not changed.
            reassign: Let the scan re-derive identity for rows a caller assigned.
            prune_unsourced: Also drop rows no scanned source claims. Off by
                default: those rows are usually an assignment or an external
                reference, not garbage.
            index_filename: Output filename within the media root.
            include_partial: Write each divided recording's trailing sequence
                that is still recording, which is otherwise held back.

        Returns:
            The index path written, and the plan of each recording divided.

        Raises:
            KeyError: If *only* names a source that is not declared.
            ValueError: If no media source is declared at all.
            DivisionRefused: If a dividing source's store cannot be divided.
                Nothing is written.
        """
        selected = self.scan_sources("media", only=only)
        if not selected:
            msg = (
                "no media scan sources are declared, so there is nothing to "
                "rescan. Declare one with add_scan_source(), or call "
                "index_media(search_dirs=...) for a one-off pass."
            )
            raise ValueError(msg)

        claim = ScanClaim()
        rows: list[dict[str, object]] = []
        plans: list[DivisionPlan] = []
        redivided: list[tuple[str, str]] = []
        display_names: dict[Entry, str] = {}
        committed = _read_media_index(
            self.get_root(self.resolve_media_root()) / index_filename
        )
        for source in selected:
            if not isinstance(source, MediaScanSource):
                continue
            claim = claim | self.source_claim(source)
            scanned = self._scan_media_dirs(
                self._search_paths(source),
                source,
                committed=committed,
                include_partial=include_partial,
                reassign=reassign,
            )
            rows.extend(scanned.rows)
            plans.extend(scanned.divided.plans)
            redivided.extend(scanned.divided.redivided)
            display_names.update(scanned.divided.display_names)
            claim = claim | ScanClaim.over_files(scanned.divided.paths)
        divided = _DividedRows(
            plans=tuple(plans),
            redivided=tuple(redivided),
            display_names=display_names,
        )
        # One dedupe across every source, not one per source: two sources may
        # legitimately reach the same file through different paths, and the first
        # recipe that named it wins, deterministically in declaration order.
        path = self._write_scanned_media(
            self._dedupe_scope_rows(rows),
            claim=claim,
            index_filename=index_filename,
            reassign=reassign,
            prune_unsourced=prune_unsourced,
            divided=divided,
        )
        self._declare_divided_recordings(divided)
        return MediaScanResult(path=path, divisions=divided.plans)

    def scan_tracks(
        self,
        *,
        only: Sequence[str] = (),
        prune_unsourced: bool = False,
        index_filename: str = "index.csv",
    ) -> Path:
        """Rescan every declared tracks source and rewrite ``tracks_raw/index.csv``.

        Each source carries its own ``src_format``, patterns and grouping rule,
        so one dataset can hold TREx output beside CalMS21 arrays. Scanning them
        one at a time through :meth:`index_tracks_raw` could not: each write
        replaced the one before it.

        Named for the source *kind* it scans, not the root it writes: the kind is
        ``tracks`` (:data:`~mosaic.core.manifest.SourceKind`, the same word the
        manifest and ``mosaic scan --kind`` use) and the root it feeds is
        ``tracks_raw``, joined by ``_RAW_ROOT_FOR_KIND``. A ``_raw`` suffix marks
        a name only where two indexes would otherwise answer to one -- which is
        why the readers carry it (:meth:`read_tracks_raw_index` beside
        ``read_tracks_index``) and the scanners do not. Nothing scans the
        converted tables under ``tracks/``; they are computed, not found.
        """
        return self._scan_raw_sources(
            "tracks",
            only=only,
            prune_unsourced=prune_unsourced,
            index_filename=index_filename,
        )

    def scan_labels(
        self,
        *,
        only: Sequence[str] = (),
        prune_unsourced: bool = False,
        index_filename: str = "index.csv",
    ) -> Path:
        """Rescan every declared labels source and rewrite ``labels_raw/index.csv``.

        Named for the ``labels`` source kind rather than the ``labels_raw`` root,
        for the reason given on :meth:`scan_tracks`.
        """
        return self._scan_raw_sources(
            "labels",
            only=only,
            prune_unsourced=prune_unsourced,
            index_filename=index_filename,
        )

    def _scan_raw_sources(
        self,
        kind: SourceKind,
        *,
        only: Sequence[str],
        prune_unsourced: bool,
        index_filename: str,
    ) -> Path:
        """Drive the raw scan for *kind*, one declared source at a time.

        Each source is a separate ``_index_raw`` pass carrying that source's own
        recipe. That is safe here where it was not before, because each pass now
        replaces only what its own claim covers -- so the second source's write
        preserves the first source's rows instead of erasing them.

        Raises:
            KeyError: If *only* names a source that is not declared.
            ValueError: If no source of this kind is declared.
        """
        selected = self.scan_sources(kind, only=only)
        if not selected:
            msg = (
                f"no {kind} scan sources are declared, so there is nothing to "
                f"rescan. Declare one with add_scan_source()."
            )
            raise ValueError(msg)
        scan = self.index_tracks_raw if kind == "tracks" else self.index_labels_raw
        written = self.get_root(_RAW_ROOT_FOR_KIND[kind]) / index_filename
        position = 0
        for source in selected:
            if not isinstance(source, RawScanSource):
                continue
            if isinstance(source, LabelsScanSource) and source.series is not None:
                # A versioned series has its own index and its own rule: one row
                # per revision, never a per-sequence composition. It is also
                # never pruned as unsourced, because the rows a dataset wrote
                # for its own saves sit under no source at all.
                written = self.index_label_series(source)
                continue
            position += 1
            written = scan(
                self._search_paths(source),
                patterns=list(source.patterns),
                src_format=source.src_format,
                index_filename=index_filename,
                recursive=source.recursive,
                multi_sequences_per_file=source.multi_sequences_per_file,
                group_from=source.group_from,
                group_pattern=source.group_pattern,
                exclude_patterns=list(source.exclude_patterns),
                compute_md5=source.md5,
                claim=self.source_claim(source),
                # Only the first pass may prune: a later one would see the
                # earlier sources' rows as unclaimed and delete what this same
                # scan had just written. Counted over the per-sequence sources
                # alone, so a series source declared first does not use it up.
                prune_unsourced=prune_unsourced and position == 1,
            )
        return written

    def _search_paths(self, source: ScanSource) -> list[str | Path]:
        """What to walk for *source*: its directory, or each listed file.

        A file-mode source hands the walker its individual files. The prober
        accepts a file as readily as a directory, which is what lets one scan
        path serve both modes rather than branching all the way down.
        """
        if source.mode == "files":
            return list(self._source_files(source))
        return [self.resolve_source_path(source)]

    def _source_files(self, source: ScanSource) -> list[Path]:
        """The listed files of a file-mode source that are actually on disk.

        A listed file that has gone is reported and skipped: its row leaves the
        index because the file did, while the declaration stays, because a share
        being unmounted is not a decision to un-import.
        """
        base = self.resolve_source_path(source)
        present: list[Path] = []
        for entry in source.files:
            candidate = base / entry
            if candidate.exists():
                present.append(candidate)
            else:
                print(
                    f"[scan] source {source.id!r}: {candidate} is listed but "
                    "missing; its row will leave the index",
                    file=sys.stderr,
                )
        return present

    def _rows_claimed_by(
        self, kind: SourceKind, claim: ScanClaim, *, series: str | None = None
    ) -> int:
        """How many rows of *kind*'s index fall inside *claim*.

        *series* names the label series a source claimed, whose rows live in that
        series' own index rather than in ``labels_raw/index.csv``.
        """
        if series is not None:
            frame = read_label_series(self, series)
            return sum(
                1 for row in index_records(frame) if self._row_claimed(row, claim)
            )
        try:
            rows = (
                self.read_media_index()
                if kind == "media"
                else _read_tracks_raw_index(
                    self.get_root(_RAW_ROOT_FOR_KIND[kind]) / "index.csv"
                )
            )
        except KeyError:
            return 0
        return sum(1 for row in rows if self._row_claimed(row, claim))

    # ---- Load and save ----

    def load(self, ensure_roots: bool = True) -> Dataset:
        """Read the manifest at :attr:`manifest_path` into this dataset.

        **Reading never writes.** A manifest older than the current format is
        migrated in memory and stays as it is on disk until something saves, so a
        read-only mount works and looking at a legacy dataset does not rewrite
        it.

        Args:
            ensure_roots: Create the declared root directories. Sources are never
                created, whatever this says -- a source names storage that may be
                unmounted, and a scan is what discovers whether it is there.

        Returns:
            This dataset, so the call chains off the constructor.

        Raises:
            FileNotFoundError: If no manifest is at *manifest_path*.
            ManifestVersionError: If the manifest is newer than this mosaic.
        """
        resolved = resolve_manifest_path(self.manifest_path)
        self.manifest_path = resolved
        self.manifest = read_manifest(resolved)
        if ensure_roots:
            self._ensure_roots()
        return self

    def save(self) -> None:
        """Write the manifest back, atomically.

        Rewrites the whole file, so it goes out through ``atomic_write``: a
        reader must never see half a manifest, and an interrupted write must not
        leave a dataset without one at all.

        Unknown top-level keys read from disk are written back unchanged. That is
        what lets a field this version stopped modeling survive a load-and-save
        round trip rather than being annihilated by it.
        """
        self._ensure_roots()
        write_manifest(self.manifest_path, self.manifest)

    # ---- Helpers ----
    @property
    def base_dir(self) -> Path:
        """Directory holding the dataset manifest (``dataset.yaml``'s parent).

        The dataset-level anchor for config and internal state. Used to locate the
        Job-Contract run-logs (``<base_dir>/.mosaic/runs/``) and to resolve
        root-relative ``abs_path`` values.
        """
        return _dataset_base_dir(self)

    def get_root(self, key: str) -> Path:
        """Return the absolute path for a named dataset root.

        Args:
            key: Root name (e.g. ``"media"``, ``"tracks"``, ``"features"``).

        Returns:
            Absolute path to the root directory.

        Raises:
            KeyError: If *key* does not resolve to a root. The message
                distinguishes the three ways that happens -- undeclared,
                declared with an empty value, and a ``Dataset`` whose roots are
                empty throughout because nothing has read the manifest yet --
                and names the repair for each.
        """
        if key not in self.roots or not self.roots[key]:
            msg = _unresolved_root_message(key, self.roots, self.manifest_path)
            raise KeyError(msg)
        p = user_path(self.roots[key])
        if not p.is_absolute():
            return (_dataset_base_dir(self) / p).resolve()
        return p

    def has_root(self, key: str) -> bool:
        """Return True if *key* is a set (non-empty) root."""
        return key in self.roots and bool(self.roots[key])

    def resolve_media_root(self) -> str:
        """Return the root key that holds actual video files.

        Prefers ``media_raw`` (original uploads) when set, falls back to
        ``media`` for backward compatibility with older datasets.
        """
        if self.has_root("media_raw"):
            return "media_raw"
        return "media"

    def set_root(self, key: str, path: str | Path) -> None:
        """Set a named dataset root and create the directory if needed.

        **A root must resolve inside the dataset** (item 9.1, implementing rule
        P7). An outside root puts that root's ``index.csv`` outside the dataset
        too, which is the scattering the rule removes: the dataset stops being
        the thing you can copy, archive or sync.

        Absolute *is* allowed, so long as it lands inside -- the rule is about
        where a root is, not how it is spelled, and the portability pass
        relativizes an inside-absolute root on its next run.

        **What this does not pin is ``abs_path``, and that is deliberate.** An
        individual file may be referenced by absolute path from an index that
        lives inside the dataset -- which is how a second dataset references a
        video living inside a first one without copying it, and what the future
        import gesture will use. Open item O2 resolved to that arrangement rather
        than to shared membership, so the separation is load-bearing: pinning
        ``abs_path`` too would remove the mechanism, not tighten it.

        Validated here rather than on read, the same boundary rule item 2.5
        applies to entry names: a dataset that already holds an outside root
        keeps resolving, so looking at a legacy dataset does not raise. What
        refuses to act on one is the sweeper, which declines rather than
        deleting.

        Args:
            key: Root name (e.g. ``"media_raw"``, ``"tracks"``).
            path: Directory path (absolute or relative to dataset root).

        Raises:
            ValueError: If *path* resolves outside the dataset directory.
        """
        self.roots[key] = str(validate_root_inside(self.base_dir, path, key))
        self._ensure_roots()

    def _ensure_roots(self) -> None:
        # Expanded, not refused: a `~` root cannot be written any more, but a
        # dataset that already holds one must keep loading. Validated on write,
        # tolerated on read -- the same rule an outside root already follows.
        for p in self.roots.values():
            if p:
                path = user_path(p)
                if not path.is_absolute():
                    path = _dataset_base_dir(self) / path
                path.mkdir(parents=True, exist_ok=True)

    def ensure_roots(self) -> None:
        """Public wrapper so callers can trigger directory creation after mutations."""
        self._ensure_roots()

    def remap_roots(self, path_map: Mapping[str, str]) -> None:
        """
        Remap dataset roots by replacing the longest matching path prefixes using path_map.
        path_map entries are {source_prefix: dest_prefix}.
        """
        if not path_map:
            return
        normalized = _normalize_path_map(path_map)
        if not normalized:
            return
        updated: dict[str, str] = {}
        for key, raw_path in self.roots.items():
            if not raw_path:
                continue
            current = user_path(raw_path)
            new_value = _remap_single_path(current, normalized)
            if new_value is not None:
                updated[key] = str(new_value)
        self.roots.update(updated)
        self._path_map = list(normalized)

    def remap_path(self, path: str | Path) -> Path:
        """Remap a single path using the dataset's path_map.

        Applies the longest-matching prefix replacement from ``path_map``
        (set during ``load()``). Returns the path unchanged if no prefix matches.

        Args:
            path: Path to remap.

        Returns:
            Remapped path, or the original if no mapping applies.
        """
        p = user_path(path)
        if not self._path_map:
            return p
        new_value = _remap_single_path(p, self._path_map)
        return new_value if new_value is not None else p

    def resolve_path(self, stored_path: str | Path, anchor: Path | None = None) -> Path:
        """Resolve a stored path (absolute or relative) to an absolute path.

        Relative paths are resolved against *anchor* (default: dataset root).
        Absolute paths that exist are returned as-is; absolute paths that don't
        exist are tried through this dataset's ``path_map``.
        """
        p = user_path(stored_path)
        # _dataset_base_dir() creates the directory it returns, so the default
        # anchor is resolved only when a relative path actually needs one; the
        # resolver never reads the anchor for an absolute path.
        base = anchor
        if base is None:
            base = p if p.is_absolute() else _dataset_base_dir(self)
        return resolve_stored_path(p, base, path_map=self._path_map)

    def _relative_to_root(self, abs_path: Path) -> str:
        """Convert an absolute path to relative-to-dataset-root for storage.

        Internal paths (inside dataset tree) become relative strings.
        External paths (outside dataset tree) remain absolute.
        """
        root = _dataset_base_dir(self)
        try:
            return str(abs_path.resolve().relative_to(root))
        except ValueError:
            return str(abs_path.resolve())

    def relative_to_root(self, path: str | Path) -> str:
        """Public: dataset-root-relative storage form of *path* (abs stays abs).

        Preferred at cross-module index-writer call sites so stored ``abs_path``
        values are portable across machines / synced datasets. The reader side
        (:meth:`resolve_path`) reverses it. External paths (outside the dataset
        tree) are returned absolute unchanged.
        """
        return self._relative_to_root(user_path(path))

    def rewrite_index_paths(
        self, path_map: Mapping[str, str], dry_run: bool = False
    ) -> dict[str, int]:
        """
        Permanently rewrite abs_path in all index CSV files on disk.

        Args:
            path_map: {old_prefix: new_prefix} mapping
            dry_run: If True, report what would change without writing

        Returns:
            Dict of {index_path: num_paths_changed}
        """
        normalized = _normalize_path_map(path_map)
        if not normalized:
            return {}

        def rewrite_index(idx_path: Path, extra_columns: Sequence[str] = ()) -> int:
            """Remap ``abs_path``, plus any *extra_columns* holding a path.

            Most index rows carry exactly one path. The rows that carry a
            second one (the tracker's source video, its ``.pv``) name it here
            rather than in the row class, because this is a raw CSV pass with
            no row type to consult.
            """
            if not idx_path.exists():
                return 0
            df = self._read_index_frame_or_none(idx_path)
            if df is None:
                return 0
            columns = [c for c in ("abs_path", *extra_columns) if c in df.columns]
            if not columns:
                return 0
            changed = 0
            for col in columns:
                col_changed = 0
                new_paths = []
                for p in df[col]:
                    if pd.isna(p) or not str(p):
                        new_paths.append(p)
                        continue
                    remapped = _remap_single_path(user_path(str(p)), normalized)
                    if remapped is not None and str(remapped) != p:
                        new_paths.append(str(remapped))
                        col_changed += 1
                    else:
                        new_paths.append(p)
                if col_changed > 0:
                    changed += col_changed
                    if not dry_run:
                        df[col] = new_paths
            if changed > 0 and not dry_run:
                df.to_csv(idx_path, index=False)
            return changed

        # Which files are indexes is one question, answered in one place (item
        # 6.1). This pass and ``make_portable`` each used to enumerate them in
        # their own closures, and the tracker roots had to be appended to both by
        # hand -- so a new tracker was portable in one pass and not the other,
        # with nothing failing to say so.
        results: dict[str, int] = {}
        for index in iter_dataset_indexes(self, _INDEX_PATH_COLUMNS):
            count = rewrite_index(index.path, index.path_columns)
            if count > 0:
                results[str(index.path)] = count

        return results

    def make_portable(self, dry_run: bool = False) -> dict[str, int]:
        """Convert all internal absolute paths to relative (to dataset root).

        Only needed for datasets created before relative-path support.
        Idempotent — safe to call multiple times. Already-relative paths
        are left unchanged.

        Args:
            dry_run: If True, report what would change without writing.

        Returns:
            Dict of ``{file_path: num_paths_changed}``.
        """
        root = _dataset_base_dir(self)
        results: dict[str, int] = {}

        def _make_rel(abs_str: str) -> tuple[str, bool]:
            """Try to make *abs_str* relative to dataset root.
            Returns (new_str, changed)."""
            p = Path(abs_str)
            if not p.is_absolute():
                return abs_str, False  # already relative
            try:
                rel = str(p.resolve().relative_to(root))
                return rel, rel != abs_str
            except ValueError:
                return abs_str, False  # external — keep absolute

        # --- 8a. Roots in dataset.yaml ---
        roots_changed = 0
        new_roots = {}
        for k, v in self.roots.items():
            if not v:
                new_roots[k] = v
                continue
            new_val, changed = _make_rel(v)
            new_roots[k] = new_val
            if changed:
                roots_changed += 1
        if roots_changed > 0:
            if not dry_run:
                self.roots = new_roots
                self.save()
            results["dataset.yaml (roots)"] = roots_changed

        # --- 8b. Index CSVs: convert abs_path column ---
        def _convert_index(idx_path: Path, extra_columns: Sequence[str] = ()) -> int:
            """Convert abs_path, plus any *extra_columns* holding a path, to relative."""
            if not idx_path.exists():
                return 0
            df = self._read_index_frame_or_none(idx_path)
            if df is None:
                return 0
            total_changed = 0
            for col in ("abs_path", *extra_columns):
                if col not in df.columns:
                    continue
                col_changed = 0
                new_vals = []
                for val in df[col]:
                    if pd.isna(val) or not str(val):
                        new_vals.append(val)
                        continue
                    new_val, changed = _make_rel(str(val))
                    new_vals.append(new_val)
                    if changed:
                        col_changed += 1
                if col_changed > 0:
                    total_changed += col_changed
                    if not dry_run:
                        df[col] = new_vals
            if total_changed > 0 and not dry_run:
                df.to_csv(idx_path, index=False)
            return total_changed

        # One enumeration, shared with ``rewrite_index_paths`` (item 6.1). The
        # hand-written copy this replaces had already drifted: it visited
        # ``labels`` and ``features`` in bespoke blocks, appended each tracker
        # root separately, and omitted the inference roots entirely.
        for index in iter_dataset_indexes(self, _INDEX_PATH_COLUMNS):
            count = _convert_index(index.path, index.path_columns)
            if count > 0:
                results[str(index.path)] = count

        # --- 8c. run_info.json files (frame extraction manifests) ---
        # Not an index, so it is not in the enumeration above: a run manifest is
        # a JSON blob a frame-extraction run wrote, and the shared walk is about
        # ``index.csv`` files.
        if self.has_root("frames"):
            frp = self.get_root("frames")
            if frp.exists():
                for ri_path in frp.rglob("run_info.json"):
                    try:
                        data = json.loads(ri_path.read_text())
                    except Exception:
                        continue
                    changed = 0
                    # output_dir -> relative to dataset root
                    if "output_dir" in data:
                        new_val, did_change = _make_rel(data["output_dir"])
                        if did_change:
                            data["output_dir"] = new_val
                            changed += 1
                    # video_path -> relative to dataset root
                    if "video_path" in data:
                        new_val, did_change = _make_rel(data["video_path"])
                        if did_change:
                            data["video_path"] = new_val
                            changed += 1
                    # files[].path -> relative to run_info.json
                    for f in data.get("files", []):
                        if "path" in f:
                            recorded = str(f["path"])
                            relative = manifest_file_path(recorded)
                            if relative != recorded:
                                f["path"] = relative
                                changed += 1
                    if changed > 0:
                        if not dry_run:
                            ri_path.write_text(json.dumps(data, indent=2, default=str))
                        results[str(ri_path)] = changed

        return results

    def _entry_stamps(self) -> dict[tuple[str, str], str]:
        """``(group, sequence) -> latest started_at`` across that entry's rows.

        ``started_at`` is stamped fresh by ``RunIndexRowBase`` on every append, so
        comparing it across a conversion says which entries that conversion
        actually rewrote -- which set membership alone cannot, since a superseded
        row stays in the index rather than disappearing from it.

        Aggregated per entry rather than per row, now that an entry can carry
        several variants. The dict comprehension this replaces also produced one
        stamp per entry -- last row wins -- and happened to pick the right one,
        because ``IndexCSV`` removes a rewritten row and re-appends it at the
        end, so a touched entry's newest stamp was always last. That is a real
        property of the writer and an unstated dependency here: nothing declares
        it, nothing tests it, and it is invisible at this call site. Asking for
        the latest stamp says what is meant instead of relying on where the row
        sits.
        """
        stamps: dict[tuple[str, str], str] = {}
        for _, row in read_tracks_index(self).iterrows():
            entry = (str(row["group"]), str(row["sequence"]))
            stamp = str(row["started_at"])
            if stamp > stamps.get(entry, ""):
                stamps[entry] = stamp
        return stamps

    @staticmethod
    def _read_index_frame_or_none(path: Path) -> pd.DataFrame | None:
        """An index CSV as a frame, or ``None`` when it holds no header at all.

        The raw-CSV counterpart of the "absent is empty" rule the typed indexes
        follow. A truncated write, an interrupted scan, or a sync that landed the
        inode without the bytes leaves a present file saying nothing; reading it
        raised ``EmptyDataError`` out of the whole-dataset hygiene passes, so one
        such file stopped ``reindex`` and ``make_portable`` over every other root
        with a message that named no path.

        Every cell is read as its text, an empty cell as ``""``. A pass that
        rewrites one column then writes every other cell back as it was read:
        a ``follows_gap`` of ``false`` stays ``false``, and a blank cell beside a
        ``chunk_first`` of ``0`` leaves it ``0`` rather than ``0.0``.
        """
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            return None

    def _conversion_is_current(self, run_id: str, group: str, sequence: str) -> bool:
        """Whether an existing table was made from what its sources hold *now*.

        The existence of the output says only that *this recipe* has *a* table for
        this entry; it says nothing about which bytes went into it. A scan updates
        ``tracks_raw/index.csv`` and the per-sequence composition, and without this
        the conversion that follows skips on existence alone -- leaving a table,
        and an index row claiming a composition, that disagree with the files they
        name, under a command that reported success.

        Answers **True when it cannot tell**, so the recorded composition is used
        as evidence of staleness and never as evidence of currency: a legacy row
        that predates the projection, or a dataset whose ``sequences.csv`` has not
        been written, keeps skipping exactly as before rather than recomputing
        every entry on every call.
        """
        current = consumed_composition_for(self, group, sequence, ("tracks_raw",))
        if not current:
            return True
        index = read_tracks_index(self)
        if index.empty:
            return True
        rows = index[
            (index["run_id"] == run_id)
            & (index["group"] == group)
            & (index["sequence"] == sequence)
        ]
        if rows.empty:
            return True
        recorded = text_cell(rows.iloc[-1].get("consumed_composition", ""))
        return not recorded or recorded == current

    def _warn_superseded_entries(self, covered: set[tuple[str, str]]) -> None:
        """Say when the index holds entries no current raw source claims.

        A converter that changes how it spells an entry writes rows under the new
        names without touching the old ones, so both resolve and every feature
        runs over each sequence twice. That is the visible consequence of
        ``calms21_npy`` 0.2, which stopped spelling its ids as slash paths.

        *covered* is every entry this conversion reached -- written, or skipped
        because it was already current -- collected as the branches walked them
        rather than predicted, since an enumerable format expands one file into
        many sequences that its raw row does not name. Anything indexed and not
        in it is a spelling no source produces any more.

        The previous test was "an entry this call did not rewrite", which named
        **every** entry on a cached re-run: a conversion that correctly writes
        nothing has rewritten nothing. It then printed a deletion remedy for
        tables that were perfectly good.

        Reported rather than repaired: deleting tables this call did not write is
        exactly the rename M1's migration rule forbids.
        """
        if not covered:
            return
        indexed = set(self._entry_stamps())
        gone = sorted(indexed - covered)
        if not gone:
            return
        listing = ", ".join(f"({g!r}, {s!r})" for g, s in gone[:5])
        more = f" and {len(gone) - 5} more" if len(gone) > 5 else ""
        remedy = ", ".join(f"({g!r}, {s!r})" for g, s in gone[:5])
        print(
            f"[convert_all_tracks] {len(gone)} entr"
            f"{'y' if len(gone) == 1 else 'ies'} in tracks/index.csv "
            f"{'is' if len(gone) == 1 else 'are'} claimed by no current raw "
            f"source: {listing}{more}.\n"
            "  If a converter changed how it spells its entries, these are the "
            "old spellings and both will resolve until you remove them:\n"
            f"    ds.drop_entries(scope=Scope(entries=[{remedy}]), "
            'delete_files=True, run_id="")\n'
            '  The run_id="" names the unlabelled tables explicitly. Omitting it '
            "means every\n  variant of those entries, which would delete the "
            "conversions you just made.",
            file=sys.stderr,
        )

    def drop_entries(
        self,
        *,
        scope: Scope,
        delete_files: bool = False,
        run_id: Optional[str] = None,
    ) -> int:
        """Remove ``(group, sequence)`` rows from the standardized-tracks index.

        The cleanup half of a rename. When a converter changes how it spells an
        entry -- as ``calms21_npy`` did at version 0.2 -- a re-conversion writes
        rows under the new names while the old ones stay, pointing at parquets
        that also stay. Both then resolve, and every feature runs over each
        sequence twice.

        Nothing removes them automatically: conversion deleting tables it did not
        write is exactly the rename this milestone's migration rule forbids. So
        the conversion warns, naming what it superseded, and this is the one call
        that acts on it.

        Args:
            scope: Which entries to drop, as
                :class:`~mosaic.core.scope.Scope`. Required, and it must name
                entries. This call deletes index rows, and an omitted selector
                would mean every row of the index. A camera-addressed selector
                narrows to the ``(group, sequence)`` pairs it names, since a
                tracks row is keyed without a camera.
            delete_files: Also unlink each row's parquet. Off by default -- an
                orphaned table is recoverable, a deleted one is not.
            run_id: Drop only this variant's rows for those entries. ``None``,
                the default, drops every variant -- which is what a rename
                cleanup wants, and what this did when an entry could only have
                one row. Name one to retire a single recipe and keep the rest,
                which is also how an entry stops being ambiguous.

        Returns:
            How many index rows were dropped.

        Raises:
            ValueError: *scope* names groups or sequences instead of entries.
        """
        if scope.groups is not None or scope.sequences is not None:
            raise ValueError(
                f"drop_entries deletes the rows it is given and does not "
                f"enumerate an index to find them, and {scope!r} names groups "
                f"or sequences. Resolve them with Dataset.resolve_scope first "
                f"and pass Scope(entries=[...])."
            )
        wanted = scope.entry_pairs or set()
        if not wanted:
            return 0
        path = tracks_index_path(self)
        if not path.exists():
            return 0
        with index_lock(path):
            frame = adopt_legacy_columns(pd.read_csv(path, keep_default_na=False))
            keep_mask = [
                (str(row["group"]), str(row["sequence"])) not in wanted
                or (run_id is not None and str(row["run_id"]) != run_id)
                for _, row in frame.iterrows()
            ]
            if all(keep_mask):
                return 0
            if delete_files:
                for keep, (_, row) in zip(keep_mask, frame.iterrows()):
                    if keep:
                        continue
                    target = self.resolve_path(str(row["abs_path"]))
                    target.unlink(missing_ok=True)
            kept = frame[keep_mask]
            atomic_write(path, lambda p: kept.to_csv(p, index=False))
            return len(frame) - len(kept)

    def reindex_features(
        self,
        feature: str | None = None,
        *,
        dry_run: bool = True,
    ) -> dict[str, int]:
        """Reconcile feature index CSVs with the parquet files on disk.

        Drops index rows whose ``abs_path`` no longer resolves to an existing
        file (e.g. outputs deleted by hand), leaving every still-present entry
        intact. Paths are resolved with :meth:`resolve_path`, so rows that are
        merely *relocated* (a moved or synced dataset) are **kept**, not pruned
        -- for those, use :meth:`make_portable` / :meth:`rewrite_index_paths`.
        Never deletes parquet files; touches the ``index.csv`` files only. The
        ``index.csv`` is the source of truth for what-ran, so this fully
        reconciles state (there is no separate store to keep in sync).

        Args:
            feature: Restrict to a single feature storage name. If None, every
                feature under ``features/`` is reconciled.
            dry_run: If True (default), report what would be dropped without
                writing. Set False to actually rewrite the indexes.

        Returns:
            ``{index_csv_path: num_rows_dropped}`` for every index with drops.
        """
        from .pipeline.dataset_indexes import feature_storages
        from .pipeline.index import feature_index, feature_index_path

        # Guarded before the named branch, not only the listing one:
        # ``feature_storages`` answers an unset root with ``[]``, but
        # ``feature_index_path`` would raise ``KeyError`` from ``get_root``.
        if not self.roots.get("features"):
            return {}

        names = [feature] if feature is not None else feature_storages(self)

        results: dict[str, int] = {}
        for name in names:
            idx_path = feature_index_path(self, name)
            if not idx_path.exists():
                continue
            dropped = feature_index(idx_path).prune_missing(
                self.resolve_path, dry_run=dry_run
            )
            if len(dropped) == 0:
                continue
            results[str(idx_path)] = len(dropped)

        return results

    def reindex(
        self, root: str | None = None, *, dry_run: bool = True
    ) -> dict[str, int]:
        """Reconcile every index in the dataset against the files on disk.

        Item 6.1's reconciler, root-agnostic. Drops index rows whose ``abs_path``
        no longer resolves to an existing file and leaves every still-present row
        intact -- the same rule :meth:`reindex_features` applies to ``features/``
        alone, over every root that has an ``IndexCSV`` behind it: ``tracks``, and
        each tracker and inference root under ``_tracking``.

        **Relocated is not missing.** Paths resolve through :meth:`resolve_path`,
        so a moved or synced dataset keeps its rows; for those,
        :meth:`make_portable` is the pass that wants running. Never deletes
        anything but rows.

        **Why the ``_tracking`` roots matter here.** Until this existed they were
        reached by no reindex, prune or portability pass at all, so a tracker
        working directory removed by hand left a row naming it forever. It is
        also the half item 8.4's sweeper deliberately does not own: the sweeper
        drops the rows *it* invalidates, and everything deleted by hand is this
        pass's to repair.

        Args:
            root: Restrict to one root key. If None, every registered root is
                reconciled.
            dry_run: If True (default), report what would be dropped without
                writing.

        Returns:
            ``{index_csv_path: num_rows_dropped}`` for every index with drops.
        """
        from .pipeline.dataset_indexes import iter_dataset_indexes, reconcilable_index

        results: dict[str, int] = {}
        for index in iter_dataset_indexes(self):
            if root is not None and index.root_key != root:
                continue
            factory = reconcilable_index(index.root_key)
            if factory is None or not index.path.exists():
                continue
            dropped = factory(index.path).prune_missing(
                self.resolve_path, dry_run=dry_run
            )
            if len(dropped) > 0:
                results[str(index.path)] = len(dropped)
        return results

    def reconcile(
        self,
        *,
        apply: bool = False,
        force: bool = False,
        only: tuple[str, ...] = (),
    ) -> ReconcileReport:
        """Recompute every artifact's identifier and re-address what moved.

        The forward pass over the identity machinery: for each feature (and, as
        their reconcilers land, tracks and labels) run, recompute its ``run_id``
        from the *current* code, compare it against the recorded one, and -- where
        the recorded provenance confirms the inputs did not change -- re-address the
        artifact under its new identifier rather than recomputing it. A run whose
        inputs cannot be confirmed unchanged is reported and left, to be recomputed
        by an ordinary run; the current version is never stamped onto history that
        cannot be verified.

        Where ``reindex`` reconciles the index against the *disk* (dropping rows
        for missing files), this reconciles the on-disk identifiers against the
        *code*. It is the pass to run after a hashing-scheme change: it reads the
        ``.identity_scheme`` marker each run was minted under, so it is idempotent
        and resumable, and a re-run over an already-migrated dataset reports every
        run ``ok``.

        Args:
            apply: If False (default), report what would change without touching
                anything. If True, refresh stale markers and perform every
                confirmed re-address (a directory move plus an index rewrite, with
                the index backed up first).
            force: Reserved for the destructive path (deleting derivatives whose
                identity moved but could not be re-addressed); not yet wired.
            only: Restrict to these artifact kinds (e.g. ``("features",)``). Empty
                means every registered kind.

        Returns:
            A :class:`~mosaic.core.pipeline.reconcile.ReconcileReport` classifying
            every artifact and recording what ``apply`` did.
        """
        import dataclasses

        from .pipeline.reconcile import identity_reconcilers, run_reconcile

        # identity_reconcilers imports the built-in reconciler modules for their
        # registration side effect, the same seam FEATURES/OPS use.
        reconcilers = identity_reconcilers(self, only)
        report = run_reconcile(reconcilers, apply=apply, force=force)

        # Compose the cheap index-hygiene passes so one command brings a dataset
        # fully current. Only on a full run -- a narrowed ``only`` is asking about
        # one artifact kind, not the whole tree. The heavier media/tracking passes
        # (``reprobe-media``, ``prune-media``, ``sweep-tracking``) stay separate
        # commands: they probe or delete, and have their own reports and abort
        # semantics.
        if only:
            return report
        # ``reindex`` deletes a row whose file it cannot resolve, and a dataset
        # copied from another machine holds rows naming that machine's absolute
        # paths -- present files, unresolvable cells. Pruning those would destroy
        # the index this pass exists to bring forward, and neither pass here can
        # repair them: ``make_portable`` relativizes against *this* root, which a
        # foreign prefix does not sit under. Only ``rewrite_index_paths``, with a
        # prefix map the caller supplies, can. So skip the pruning and say so.
        stale = self._foreign_path_indexes()
        pruned: dict[str, int] = {}
        if stale:
            listing = ", ".join(sorted(stale)[:3])
            more = f" and {len(stale) - 3} more" if len(stale) > 3 else ""
            print(
                f"[reconcile] skipped dropping dangling rows: {len(stale)} index"
                f"{'' if len(stale) == 1 else 'es'} hold paths from another "
                f"machine ({listing}{more}).\n"
                "  Those rows name files that exist here under different paths, "
                "and pruning them would\n  delete the index rather than repair "
                "it. Repair them first, then re-run:\n"
                "    ds.rewrite_index_paths({'<their prefix>': '<this dataset>'})"
                "; ds.make_portable()",
                file=sys.stderr,
            )
        else:
            pruned = self.reindex(dry_run=not apply)
        repathed = self.make_portable(dry_run=not apply)
        return dataclasses.replace(report, pruned=pruned, repathed=repathed)

    def _foreign_path_indexes(self) -> set[str]:
        """Indexes holding an absolute path that resolves to nothing here.

        The signature of a dataset copied or synced from another machine. Such a
        cell is not a dangling row -- the file is usually present, under this
        dataset, at a path the cell does not name -- so it must not be pruned.
        """
        stale: set[str] = set()
        for index in iter_dataset_indexes(self, _INDEX_PATH_COLUMNS):
            if not index.path.exists():
                continue
            frame = self._read_index_frame_or_none(index.path)
            if frame is None:
                continue
            for column in ("abs_path", *index.path_columns):
                if column not in frame.columns:
                    continue
                for cell in frame[column]:
                    text = "" if pd.isna(cell) else str(cell)
                    if text and Path(text).is_absolute() and not Path(text).exists():
                        stale.add(str(index.path))
                        break
        return stale

    def list_groups(self) -> list[str]:
        """The group names in ``tracks/index.csv``, sorted. Empty when there are none.

        Absent and empty answer alike: both mean "this dataset has no
        standardized tracks", and answering them differently is what left six
        readers of this file with four different policies. A caller that wants to
        tell a human to convert first checks for an empty result -- that check is
        the same for both spellings.
        """
        return sorted({str(g) for g in read_tracks_index(self)["group"]})

    def list_entries(self, group: str | None = None) -> list[Entry]:
        """The ``(group, sequence)`` entries in ``tracks/index.csv``, sorted.

        Optionally within one group. Empty when there are none; see
        :meth:`list_groups` on why absent is not an error.
        """
        df = read_tracks_index(self)
        if group is not None:
            df = df[df["group"] == group]
        return sorted(
            {(str(g), str(s)) for g, s in zip(df["group"], df["sequence"], strict=True)}
        )

    def list_sequences(self, group: str | None = None) -> list[str]:
        """The sequence names in ``tracks/index.csv``, optionally within one group.

        A name that two groups share is listed once. Empty when there are none;
        see :meth:`list_groups` on why absent is not an error.
        """
        return sorted({sequence for _, sequence in self.list_entries(group)})

    def get_sequence_metadata(
        self,
        level_names: list[str] | None = None,
        separator: str = "__",
    ) -> pd.DataFrame:
        """
        Return a DataFrame with all sequences and optionally parsed hierarchy columns.

        This method provides a way to view the full dataset structure and filter
        by arbitrary hierarchy levels, supporting datasets with different organizational
        structures (2, 3, 4+ levels).

        Note
        ----
        Hierarchy parsing reads structure out of the ``__``-delimited group/sequence
        names. This is *legacy convenience* for datasets that encode factors in
        names. The canonical, redefinable way to group/categorize sequences is
        tags (owned by mosaic-api); a tag-resolved subset is run via
        ``run_feature(scope=Scope(entries=[(group, sequence), ...]))``.

        Parameters
        ----------
        level_names : list[str], optional
            Names for hierarchy levels. If provided, parses the full path
            (group + sequence) into columns with these names.
            E.g., ["individual", "speed", "loop"] for a 3-level hierarchy.
        separator : str, default "__"
            The separator used in compound names.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - group, sequence: Original values from index
            - group_safe, sequence_safe: URL-encoded versions
            - abs_path: Path to the parquet file
            - Additional columns from index (n_rows, etc.)
            - If level_names provided: one column per level name

        Examples
        --------
        >>> # Basic usage - get all sequences
        >>> meta = ds.get_sequence_metadata()
        >>> meta[['group', 'sequence']].head()

        >>> # Parse into hierarchy levels
        >>> meta = ds.get_sequence_metadata(level_names=["individual", "speed", "loop"])
        >>> meta.groupby("speed")["sequence"].count()

        >>> # 4-level hierarchy for continuous recordings
        >>> meta = ds.get_sequence_metadata(
        ...     level_names=["experiment", "arena", "day", "hour"]
        ... )
        """
        from .helpers import parse_hierarchy

        # legacy_view re-derives the safe-name columns this method's docstring
        # promises. They are no longer stored -- they were a cache of a pure
        # function of group/sequence, recomputed on every write anyway.
        df = legacy_view(read_tracks_index(self))

        # The label beside the token, never instead of it. Every other column
        # here is keyed on group/sequence and a caller filters with them, so
        # replacing them would break the thing this method is for; a
        # display_name column is additive and lets a listing show both.
        #
        # parse_hierarchy still reads the **token**. validate_entry_name's error
        # text steers users to encode hierarchy there with "__", and a factor
        # parsed out of a freely-relabelled string would change meaning the
        # moment someone renamed a sequence for readability.
        labels = self.display_names()
        df["display_name"] = [
            labels.get((str(group), str(sequence)), str(sequence))
            for group, sequence in zip(df["group"], df["sequence"])
        ]

        if level_names:
            # Parse each row into hierarchy levels
            parsed_rows = []
            for _, row in df.iterrows():
                parsed = parse_hierarchy(
                    row["group"], row["sequence"], level_names, separator
                )
                parsed_rows.append(parsed)

            # Add parsed columns to DataFrame
            parsed_df = pd.DataFrame(parsed_rows)
            df = pd.concat([df, parsed_df], axis=1)

        return df

    def query_sequences(
        self,
        group_contains: str | None = None,
        group_startswith: str | None = None,
        group_endswith: str | None = None,
        sequence_contains: str | None = None,
        sequence_startswith: str | None = None,
        sequence_endswith: str | None = None,
    ) -> list[tuple[str, str]]:
        """
        Return (group, sequence) pairs matching the specified criteria.

        Provides flexible filtering for hierarchical datasets where group and/or
        sequence names encode multiple factors.

        Note
        ----
        This is *legacy convenience* based on substring/prefix matching of the
        ``__``-delimited names. The canonical, redefinable grouping of sequences
        is tags (owned by mosaic-api). The returned ``(group, sequence)`` pairs
        can be passed straight to ``run_feature(scope=Scope(entries=...))`` to
        run a feature over exactly that subset.

        Parameters
        ----------
        group_contains : str, optional
            Filter groups containing this substring
        group_startswith : str, optional
            Filter groups starting with this prefix
        group_endswith : str, optional
            Filter groups ending with this suffix
        sequence_contains : str, optional
            Filter sequences containing this substring
        sequence_startswith : str, optional
            Filter sequences starting with this prefix
        sequence_endswith : str, optional
            Filter sequences ending with this suffix

        Returns
        -------
        list[tuple[str, str]]
            List of (group, sequence) pairs matching all criteria

        Examples
        --------
        >>> # Get all sequences for individual_01
        >>> pairs = ds.query_sequences(group_startswith="individual_01")

        >>> # Get all speed_3 recordings across all individuals
        >>> pairs = ds.query_sequences(sequence_startswith="speed_3")

        >>> # Get all loop_1 recordings at speed_3
        >>> pairs = ds.query_sequences(
        ...     sequence_contains="speed_3",
        ...     sequence_endswith="loop_1"
        ... )
        """
        df = read_tracks_index(self)

        mask = pd.Series([True] * len(df))

        if group_contains is not None:
            mask &= df["group"].str.contains(group_contains, na=False)
        if group_startswith is not None:
            mask &= df["group"].str.startswith(group_startswith, na=False)
        if group_endswith is not None:
            mask &= df["group"].str.endswith(group_endswith, na=False)
        if sequence_contains is not None:
            mask &= df["sequence"].str.contains(sequence_contains, na=False)
        if sequence_startswith is not None:
            mask &= df["sequence"].str.startswith(sequence_startswith, na=False)
        if sequence_endswith is not None:
            mask &= df["sequence"].str.endswith(sequence_endswith, na=False)

        filtered = df[mask]
        return list(zip(filtered["group"], filtered["sequence"]))

    # ----------------------------
    # Media indexing (no symlinks)
    # ----------------------------
    def _probe_dir_rows(
        self,
        search_dirs: Iterable[str | Path],
        exts: set[str],
        recursive: bool,
        facts_by_name: Mapping[str, MediaFacts] | None = None,
        cached_facts_by_key: Mapping[MediaRowKey, MediaFacts] | None = None,
    ) -> list[ProbedEntry]:
        """Probe every media file + imgstore under *search_dirs* (identity-free).

        Returns one :class:`ProbedEntry` per entry -- plain video files first
        (deterministically ordered by resolved path), then one entry per
        imgstore directory (sorted). An imgstore entry carries its Motif
        ``camera_serial`` and ``sync_uuid``
        (= ``synchronizationuuid``) read from store metadata; a plain video
        carries neither. Identity assignment (group/sequence) is left to the
        caller. Shared by :meth:`index_media` (scan-and-derive) and
        :meth:`write_media_index` (assignment-driven scope re-probe).

        *facts_by_name* maps a plain video's basename to an already-measured
        :class:`~mosaic_media.MediaFacts` -- a caller's own measurement, which
        always wins; a file present in it has its row built from those facts and
        is never probed. *cached_facts_by_key* maps the key of a prior undivided
        row, by its resolved path, to the row's measurement, reused only for a
        file with no injected facts;
        the caller admits an entry only when the file's size and mtime still
        match what that row recorded. Everything else is probed;
        :meth:`index_media` passes neither map, so it probes every file. The
        precedence is injection, then cache, then probe.
        """
        facts_map = facts_by_name or {}
        cache = cached_facts_by_key or {}
        from .media.imgstore_io import (
            imgstore_probe,
            imgstore_store_identity,
            is_imgstore,
            store_stat,
        )

        search = [user_path(d) for d in search_dirs]

        # Skip this dataset's media variants. A recursive source rooted at the
        # media root otherwise indexes them as originals.
        is_media_variant = media_variant_filter(self)

        # Discover imgstore directories first. A store is a directory (not a file
        # with an extension) that contains its own chunk video files -- so we
        # must (a) emit one entry per store and (b) exclude those internal chunks
        # from the plain file glob below.
        imgstore_dirs: set[Path] = set()
        for d in search:
            if not d.exists():
                continue
            candidates = [d, *(d.rglob("*") if recursive else d.glob("*"))]
            for cand in candidates:
                # The directory test runs first, because resolving a symlink loop
                # raises and a loop is never a directory.
                if (
                    not cand.is_dir()
                    or is_under_tracking_root(cand.parts)
                    or is_media_variant(cand)
                ):
                    continue
                if is_imgstore(cand):
                    imgstore_dirs.add(cand.resolve())

        # Serial glob: collect (path, stat) probe_candidates only. Probing
        # (ffprobe / MediaFacts, I/O bound) happens afterward through a bounded
        # thread pool so many-file search dirs index in parallel.
        probe_candidates: list[tuple[Path, os.stat_result]] = []
        tracking_skipped = 0
        variants_skipped = 0
        for d in search:
            if not d.exists():
                print(f"[WARN] search dir missing: {d}", file=sys.stderr)
                continue
            # A file-mode source hands over its individual files rather than a
            # directory to walk. Globbing a file yields nothing, so it would
            # silently probe zero entries -- accept it as its own candidate and
            # let the same filters below decide.
            it = [d] if d.is_file() else (d.rglob("*") if recursive else d.glob("*"))
            for p in it:
                if not p.is_file():
                    continue
                # Skip macOS resource forks (._* files)
                if p.name.startswith("._"):
                    continue
                # Never descend into `_tracking` (item 8.1). A tracker writes
                # debug frames and re-encoded clips into its working directory,
                # and this glob filters on extension alone -- so a generated
                # `.mp4` would be indexed as *source* media, giving a derived
                # file a `video_uuid` and a place in a sequence's composition.
                if is_under_tracking_root(p.parts):
                    tracking_skipped += 1
                    continue
                # Skip files that live inside an imgstore directory (its chunks).
                if imgstore_dirs and any(
                    sd in p.resolve().parents for sd in imgstore_dirs
                ):
                    continue
                if p.suffix.lower() not in exts:
                    continue
                # Skip this dataset's media variants. The extension filter above
                # has already excluded the index and claim files beside them, and
                # the count names only video files.
                if is_media_variant(p):
                    variants_skipped += 1
                    continue
                try:
                    st = p.stat()
                except OSError as e:
                    print(f"[WARN] skip {p}: {e}", file=sys.stderr)
                    continue
                probe_candidates.append((p, st))

        if tracking_skipped:
            print(
                f"[INFO] skipped {tracking_skipped} generated file(s) under "
                f"{TRACKING_ROOT}/ -- tracker output is not source media",
                file=sys.stderr,
            )
        if variants_skipped:
            print(
                f"[INFO] skipped {variants_skipped} media variant file(s) under "
                f"{media_variants_root(self)}, which are not source media",
                file=sys.stderr,
            )

        # Probe deterministically by resolved path so pool completion order
        # never affects the returned order.
        probe_candidates.sort(key=lambda item: str(item[0].resolve()))
        results: list[ProbedEntry] = []
        max_probe_workers = min(4, (os.cpu_count() or 2))
        with ThreadPoolExecutor(max_workers=max_probe_workers) as executor:
            # Only files with no injected facts reach the pool; an injected file
            # builds its row inline. Iterating probe_candidates (already sorted)
            # keeps the returned order deterministic regardless.
            futures = {
                p: executor.submit(probe_video_metadata, p)
                for p, _st in probe_candidates
                if p.name not in facts_map
                and MediaRowKey.of(str(p.resolve()), None) not in cache
            }
            for p, st in probe_candidates:
                injected = facts_map.get(p.name)
                if injected is not None:
                    results.append(
                        ProbedEntry(
                            p, st, row_from_facts(injected), "video", origin="injected"
                        )
                    )
                    continue
                cached = cache.get(MediaRowKey.of(str(p.resolve()), None))
                if cached is not None:
                    results.append(
                        ProbedEntry(
                            p, st, row_from_facts(cached), "video", origin="cached"
                        )
                    )
                    continue
                try:
                    probe = futures[p].result()
                except (OSError, MediaProbeError) as e:
                    print(f"[WARN] skip {p}: {e}", file=sys.stderr)
                    continue
                results.append(ProbedEntry(p, st, probe, "video", origin="probed"))

        # One entry per imgstore directory (one camera of a recording). The
        # Motif camera_serial / synchronizationuuid ride along as store facts;
        # index_media groups the cameras of one recording into one sequence.
        for store_dir in sorted(imgstore_dirs):
            try:
                store_st = store_stat(store_dir)
                probe = imgstore_probe(store_dir)
            except OSError as e:
                print(f"[WARN] skip imgstore {store_dir}: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(
                    f"[WARN] failed to probe imgstore {store_dir}: {e}",
                    file=sys.stderr,
                )
                continue
            identity = imgstore_store_identity(store_dir)
            results.append(
                ProbedEntry(
                    store_dir,
                    store_st,
                    probe,
                    "imgstore",
                    camera_serial=identity.camera_serial,
                    sync_uuid=(
                        identity.sync_uuid
                        if identity.synchronization.lower() != "none"
                        else ""
                    ),
                )
            )

        return results

    def index_media(
        self,
        search_dirs: Iterable[str | Path],
        extensions: Tuple[str, ...] = (".mp4", ".avi"),
        index_filename: str = "index.csv",
        recursive: bool = True,
        sequence_match_mode: str = "exact",
        media_layout: MediaLayout | str = "stem",
        *,
        claim: ScanClaim | None = None,
        reassign: bool = False,
        prune_unsourced: bool = False,
    ) -> Path:
        """
        Scan search_dirs for media files with given extensions and write ``media_raw/index.csv``.

        The one-off counterpart to :meth:`scan_media`, for directories the
        manifest does not declare as sources. Both write the **originals** index:
        :meth:`resolve_media_root`, which answers ``media_raw`` for any dataset
        declaring that root and ``media`` only on one predating the split.
        Transcode derivatives live in a second index, ``media/index.csv``, and
        nothing here writes it.

        - No symlinks created; absolute paths recorded.
        - **Replaces what the scan claims and preserves everything else.** A row
          under none of the scanned directories survives: an assignment-written
          row for a sequence this pass was not given, and a row whose
          ``abs_path`` points outside the dataset -- the mechanism one dataset
          uses to reference another's video without copying it.
        - imgstore directories (Motif / Loopbio) are discovered natively: each
          store becomes one entry (``media_type="imgstore"``). The cameras of one
          synchronized recording (a shared Motif ``synchronizationuuid``) collapse
          into a single sequence with one ``camera`` row per store; each store's
          internal chunk files are excluded from the plain file glob. A scan
          refuses, with :class:`StoreGroupingError`, a synchronized recording
          whose stores have different base names, a synchronized recording of
          two or more stores that include one without a ``camera_serial``, and
          stores of different recordings that derive one ``(group, sequence)``,
          such as two cameras' stores both named by their start time. The last
          two compare the stores of this scan with those the index keeps under
          each entry the scan writes. Each
          unsynchronized store is a recording of its own here, whatever its
          ``camera_serial``.
        - Columns: name, group, sequence, group_safe, sequence_safe, camera,
          sync_uuid, camera_serial, abs_path, size_bytes, mtime_iso, width,
          height, fps, codec, media_type, frame_count, analysis_transcode,
          stream_transcode, analysis_derivative_path, playback_derivative_path,
          source_path, media_facts, video_order. ``camera`` is the
          within-sequence camera axis (``""`` for single-camera media),
          ``sync_uuid`` the recording id that groups a recording's cameras, and
          ``camera_serial`` the serial an imgstore's metadata records (``""``
          for a plain video). ``media_facts`` is the full
          injectable MediaFacts serialized as JSON; the other new columns
          duplicate a few of its fields (plus the verdict) for untyped pandas
          readers and routing.

        Parameters
        ----------
        search_dirs : Iterable[str | Path]
            Directories to scan for media files.
        extensions : tuple of str
            File extensions to include.
        index_filename : str
            Output CSV filename within media root.
        recursive : bool
            Whether to search subdirectories.
        sequence_match_mode : str
            How to match video filenames to known sequences from tracks/index.csv.
            - "exact" (default): video stem must exactly match a sequence name.
            - "prefix": video stem is matched to the longest sequence name that
              is a prefix of the stem. This handles split recordings where files
              are named like ``session01_001.mp4``, ``session01_002.mp4`` mapping
              to sequence ``session01``.
        """
        if media_layout not in ("stem", "per_sequence"):
            raise ValueError(
                f"media_layout must be 'stem' or 'per_sequence', got {media_layout!r}"
            )
        if sequence_match_mode not in {"exact", "prefix"}:
            raise ValueError(
                f"sequence_match_mode must be 'exact' or 'prefix', got '{sequence_match_mode}'"
            )

        # The recipe this ad-hoc pass runs under, as the same object a declared
        # source would be -- so the walk has one code path whether its knobs came
        # from the manifest or from these arguments. Built through
        # ``model_validate`` because ``media_layout`` and ``sequence_match_mode``
        # arrive as open strings for backwards compatibility, and the model is
        # what narrows them to their closed sets.
        recipe = MediaScanSource.model_validate(
            {
                "id": "ad-hoc",
                "path": ".",
                "extensions": tuple(
                    e if e.startswith(".") else f".{e}" for e in extensions
                ),
                "recursive": recursive,
                "layout": media_layout,
                "match_mode": sequence_match_mode,
            }
        )
        # Materialized once. `search_dirs` is an Iterable, and this used to be
        # consumed twice -- a list here and a generator in the claim below -- so
        # a generator argument produced an empty claim, and the write silently
        # became an append instead of the replace it reports.
        search = [user_path(d) for d in search_dirs]
        rows = self._scan_media_dirs(search, recipe).rows
        if claim is None:
            claim = ScanClaim.over_directories(search)
        return self._write_scanned_media(
            rows,
            claim=claim,
            index_filename=index_filename,
            reassign=reassign,
            prune_unsourced=prune_unsourced,
        )

    def _scan_media_dirs(
        self,
        search_dirs: Sequence[str | Path],
        source: MediaScanSource,
        *,
        committed: Sequence[Mapping[str, object]] = (),
        include_partial: bool = False,
        reassign: bool = False,
    ) -> _ScannedMedia:
        """Probe *search_dirs* into media index rows, using *source*'s recipe.

        The identity half of a media scan: walk, probe, and decide each row's
        ``(group, sequence)``. Split from the write so a pass over several
        declared sources can accumulate rows under several different recipes and
        commit them together -- writing the index once per source would make each
        source's write replace the one before it.

        A source with a ``division`` divides the stores it finds
        (:meth:`_divided_rows`) against the *committed* rows of the index.
        """
        exts = {
            e.lower() if e.startswith(".") else f".{e.lower()}"
            for e in source.extensions
        }
        recursive = source.recursive
        media_layout = source.layout
        sequence_match_mode = source.match_mode
        seq_key_map = self._build_media_sequence_keymap()

        # Probe every file + imgstore under the search dirs (identity-free), then
        # derive each row's (group, sequence) from the track keymap here. Plain
        # videos key off their stem; imgstore stores are grouped by sync_uuid so
        # the cameras of one recording share a sequence (see _imgstore_rows).
        rows: list[dict[str, object]] = []
        imgstore_entries: list[ProbedEntry] = []
        for entry in self._probe_dir_rows(search_dirs, exts, recursive):
            if entry.media_type == "imgstore":
                imgstore_entries.append(entry)
                continue
            meta = self._match_media_sequence(
                seq_key_map, entry.path.stem, mode=sequence_match_mode
            )
            # Where identity comes from when no track table names this file.
            #
            # ``per_sequence`` reads it from the *directory*, which is item 9.2's
            # declared layout: ``<media_raw>/<entry key>/*.mp4``, one level named
            # by ``make_entry_key``. That is the layout the control plane already
            # writes -- and until now nothing in mosaic could read it back, so a
            # multi-clip sequence re-derived as one sequence per file.
            #
            # ``stem`` stays the default, which grandfathers the flat layout
            # rather than migrating it. ``sequence_match_mode="prefix"`` exists
            # to serve split recordings under that layout, and a default flip
            # would silently re-identify every dataset that relies on it.
            if media_layout == "per_sequence":
                fallback_group, fallback_seq = parse_entry_key(entry.path.parent.name)
                fallback_safe = to_safe_name(fallback_seq)
                if meta is None and fallback_group:
                    meta = {
                        "group": fallback_group,
                        "sequence": fallback_seq,
                        "group_safe": to_safe_name(fallback_group),
                        "sequence_safe": fallback_safe,
                    }
            else:
                # When no track match, use the stem as sequence so each entry is
                # its own sequence (not all lumped together under an empty key).
                fallback_seq = entry.path.stem
                fallback_safe = to_safe_name(entry.path.stem)
            rows.append(
                build_media_index_row(
                    path=entry.path,
                    stat=entry.stat,
                    to_store_path=self.relative_to_root,
                    group=meta.get("group", "") if meta else "",
                    sequence=meta.get("sequence", fallback_seq)
                    if meta
                    else fallback_seq,
                    group_safe=meta.get("group_safe", "") if meta else "",
                    sequence_safe=meta.get("sequence_safe", fallback_safe)
                    if meta
                    else fallback_safe,
                    probe=entry.probe,
                    media_type=entry.media_type,
                    assignment_source="scan-keymap" if meta else "scan-stem",
                )
            )
        divided = _DividedRows()
        if source.division is None:
            rows.extend(
                self._imgstore_rows(imgstore_entries, seq_key_map, sequence_match_mode)
            )
        elif imgstore_entries:
            divided = self._divided_rows(
                [
                    (stores, source.division)
                    for stores in self._recordings_to_divide(
                        imgstore_entries,
                        source.division,
                        self._committed_divisions(committed),
                    )
                ],
                include_partial=include_partial,
                committed=committed,
                assignment_source=DIVISION_ASSIGNMENT_SOURCE,
                reassign=reassign,
            )
            rows.extend(divided.rows)

        # De-duplicate by row key: the stored path, and a division's chunks.
        seen: set[MediaRowKey] = set()
        dedup: list[dict[str, object]] = []
        for r in rows:
            k = media_row_key(r)
            if k in seen:
                continue
            seen.add(k)
            dedup.append(r)

        return _ScannedMedia(rows=dedup, divided=divided)

    def _write_scanned_media(
        self,
        dedup: list[dict[str, object]],
        *,
        claim: ScanClaim,
        index_filename: str = "index.csv",
        reassign: bool = False,
        prune_unsourced: bool = False,
        divided: _DividedRows | None = None,
    ) -> Path:
        """Commit scanned media rows, replacing only what *claim* covers.

        The write half of a media scan, shared by the ad-hoc search-dir pass and
        the declared-source pass so the preserve rule cannot be right in one and
        wrong in the other. The manifest is raised for *divided*'s recordings
        once the write's refusals pass (:meth:`_prepare_divided_manifest`).
        """
        media_root = self.get_root(self.resolve_media_root())
        out_csv = media_root / index_filename
        # Scanning and probing above is the expensive, read-only phase and is
        # deliberately unlocked; from here it is in-memory work plus one terminal
        # ``atomic_write``. Not a requirement of ``index_lock`` any more -- the
        # lock is on a sidecar the rename never touches -- but of throughput: the
        # lock's timeout is tuned for a CSV rewrite, and holding it across
        # ffprobe would serialize every uploader on I/O that is not the write.
        #
        # **This is a replace over what the scan claims, not over the file.** A
        # row the scan does not claim survives: rows written by an assignment
        # scope this scan was not given, and rows whose ``abs_path`` points
        # outside the dataset entirely -- the mechanism that lets one dataset
        # reference another's video without copying it. Replacing the whole file
        # deleted both, so scanning dir A and then dir B kept only B.
        #
        # ``prior_order`` is read, not discarded. Passing an empty prior map made
        # every row an unknown-order prior, which sorts by *name* -- so scanning
        # a corpus whose order came from an arranged write silently permuted it,
        # and the media composition hash computed from that order moved with no
        # content change. The lookup key is (group, sequence, basename) and a
        # rescan re-derives the same (group, sequence) for the same file, so the
        # keys match. They miss legitimately when the tracks keymap changed
        # between scans and a file's sequence name moved with it: the prior order
        # genuinely no longer describes that sequence, and falling back to name
        # is correct there.
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        touched = {
            (str(row.get("group", "")), str(row.get("sequence", ""))) for row in dedup
        }
        # What the walk found, added to where it looked. See walked_claim: a
        # symlinked source stores each row under the target path, which the
        # directory claim does not cover, so without this the scan preserves its
        # own rows and appends a duplicate set on every pass.
        claim = claim | self.walked_claim(dedup)
        with index_lock(out_csv):
            committed = _read_media_index(out_csv)
            prior_order = build_prior_order(committed)
            preserved: list[dict[str, object]] = [
                dict(row) for row in committed if not self._row_claimed(row, claim)
            ]
            if not reassign:
                self._keep_assigned_identity(dedup, committed)
                dedup = self._keep_assigned_divisions(dedup, committed)
            # Unconditional, and deliberately outside the ``reassign`` guard: a
            # calibration is not identity. ``reassign`` says the scan may
            # re-derive which sequence a file belongs to, which no more licenses
            # discarding how many centimetres its pixels are than it licenses
            # discarding the file.
            self._keep_calibration(dedup, committed)
            if prune_unsourced:
                preserved = []
            refuse_merged_store_recordings(dedup, beside=preserved)
            self._carry_forward_derivative_links(dedup, out_csv)
            # Checked on the entries the rows are written under: a filename's
            # guess that an assigned identity replaced above writes nothing.
            written = entries_of_rows(dedup)
            self.refuse_entry_key_collision(
                written, self._entries_beside_media_write(written, committed, preserved)
            )
            if divided is not None:
                self._prepare_divided_manifest(divided)
            merged: list[dict[str, object]] = [*preserved, *dedup]
            # Densified over the sequences this scan touched, not over the whole
            # file: ``densify_video_order`` renumbers every row it is handed, so
            # passing all of them re-numbers untouched sequences and moves their
            # composition hashes for no reason.
            in_scope = [
                row
                for row in merged
                if (str(row.get("group", "")), str(row.get("sequence", ""))) in touched
            ]
            densify_video_order(in_scope, session_positions={}, prior_order=prior_order)
            df_out = frame_from_rows(merged)
            write_media_index_rows(out_csv, df_out)
        # ``merged``, not ``dedup``: ``write_sequence_compositions`` *replaces*
        # the per-sequence index, so projecting only the scanned rows would
        # delete the composition row of every sequence this pass preserved.
        self._write_media_compositions(merged)

        multi_count = 0
        if not df_out.empty:
            # Count temporal-chunk multiplicity per camera; two cameras of one
            # recording are one sequence with two rows, not a multi-video chunked
            # sequence, so they must not be conflated here.
            seq_counts = df_out.groupby(["group", "sequence", "camera"]).size()
            multi_count = int((seq_counts > 1).sum())
        print(
            f"[index_media] Wrote {len(df_out)} entries -> {out_csv}"
            + (f" ({multi_count} multi-video sequences)" if multi_count else "")
        )
        return out_csv

    @staticmethod
    def _refuse_unsynced_cameras(
        scope: MediaIndexScope, entries: Sequence[ProbedEntry]
    ) -> None:
        """Refuse a scope that would join independent cameras into one sequence.

        Unsynchronized imgstores under one scope become clips of one sequence,
        which is right for one camera's stores. Stores of distinct non-empty
        ``camera_serial`` values record separate recordings.

        Raises:
            StoreGroupingError: Naming the scope's entry and the serials.
        """
        serials = sorted(
            {
                entry.camera_serial
                for entry in entries
                if entry.media_type == "imgstore"
                and not entry.sync_uuid
                and entry.camera_serial
            }
        )
        if len(serials) < 2:
            return
        message = (
            f"the scope for ({scope.group!r}, {scope.sequence!r}) holds "
            f"unsynchronized imgstores of cameras {serials}, which are separate "
            "recordings. Give each camera's stores a scope of their own."
        )
        raise StoreGroupingError(message)

    @staticmethod
    def _store_base_name(entry: ProbedEntry, *, strip_camera: bool) -> str:
        """The sequence base name of an imgstore directory.

        For a synchronized recording (*strip_camera* true) the shared sequence
        name is the dir name minus its ``.<camera_serial>`` suffix
        (:func:`~mosaic.core.media.store_division.store_base_name`). An
        unsynchronized store keeps its **full** dir name so two separate
        recordings that happen to share a base (``rec.A`` / ``rec.B``, different
        or absent sync ids) never collapse into one bogus 2-camera sequence.
        """
        if strip_camera:
            return store_base_name(entry.path.name, entry.camera_serial)
        return entry.path.name

    def _imgstore_rows(
        self,
        entries: list[ProbedEntry],
        seq_key_map: dict[str, list[dict[str, str]]],
        sequence_match_mode: str,
    ) -> list[dict[str, object]]:
        """Build media-index rows for imgstore stores, grouping cameras.

        Stores sharing a non-empty ``sync_uuid`` are one recording -> one
        sequence with a ``camera`` row per store, the camera named by the store's
        ``camera_serial``; an unsynced store forms its own singleton sequence.
        The sequence name is each store's base name (:meth:`_store_base_name`),
        matched once against the track keymap so a keymap hit assigns every
        camera the same ``(group, sequence)``. Emission is deterministic: groups
        and their member stores sort by ``(sync_uuid, camera_serial, path)``.
        Two recordings that derive one entry, and a store without a serial that
        joins the cameras of its recording kept in the index, are refused by
        the caller after assigned identity is restored
        (:func:`refuse_merged_store_recordings`).

        Raises:
            StoreGroupingError: If the stores of one synchronized recording
                reduce to different base names, or two or more of them include
                a store with an empty ``camera_serial``.
        """
        groups: dict[str, list[ProbedEntry]] = {}
        for entry in entries:
            # An unsynced store keys on its own path so distinct stores never
            # merge; the "\x00" prefix keeps such keys disjoint from any uuid.
            key = entry.sync_uuid or f"\x00{entry.path}"
            groups.setdefault(key, []).append(entry)

        rows: list[dict[str, object]] = []
        for _key, members in sorted(groups.items()):
            members = sorted(
                members, key=lambda e: (e.sync_uuid, e.camera_serial, str(e.path))
            )
            # A non-empty group key is a shared sync_uuid: these stores are one
            # recording's cameras, so strip the serial to their shared base name.
            synced = bool(members[0].sync_uuid)
            if synced and len(members) > 1:
                unnamed = [str(e.path) for e in members if not e.camera_serial]
                if unnamed:
                    raise unnamed_cameras_error(members[0].sync_uuid, unnamed)
            bases = {self._store_base_name(e, strip_camera=synced) for e in members}
            if len(bases) > 1:
                message = (
                    f"imgstore recording {members[0].sync_uuid!r} has cameras "
                    f"with divergent base names {sorted(bases)}. Name each store "
                    "<recording>.<camera_serial>."
                )
                raise StoreGroupingError(message)
            base = bases.pop()
            meta = self._match_media_sequence(
                seq_key_map, base, mode=sequence_match_mode
            )
            group = meta.get("group", "") if meta else ""
            sequence = meta.get("sequence", base) if meta else base
            group_safe = meta.get("group_safe", "") if meta else ""
            sequence_safe = (
                meta.get("sequence_safe", to_safe_name(base))
                if meta
                else to_safe_name(base)
            )
            for entry in members:
                rows.append(
                    build_media_index_row(
                        path=entry.path,
                        stat=entry.stat,
                        to_store_path=self.relative_to_root,
                        group=group,
                        sequence=sequence,
                        group_safe=group_safe,
                        sequence_safe=sequence_safe,
                        camera=entry.camera_serial,
                        sync_uuid=entry.sync_uuid,
                        camera_serial=entry.camera_serial,
                        probe=entry.probe,
                        media_type=entry.media_type,
                        # A store's identity is its directory name (minus the
                        # camera serial) whether or not the keymap matched, so
                        # both outcomes are the same kind of derivation and get
                        # one value -- unlike a plain video, where a keymap hit
                        # and a stem fallback are genuinely different claims.
                        assignment_source="scan-imgstore-sync",
                    )
                )
        return rows

    def _store_row(
        self, entry: ProbedEntry, assignment_source: AssignmentSource
    ) -> dict[str, object]:
        """The undivided media index row of a probed store, before its identity."""
        return build_media_index_row(
            path=entry.path,
            stat=entry.stat,
            to_store_path=self.relative_to_root,
            group="",
            sequence="",
            group_safe="",
            sequence_safe="",
            camera=entry.camera_serial,
            sync_uuid=entry.sync_uuid,
            camera_serial=entry.camera_serial,
            probe=entry.probe,
            media_type="imgstore",
            assignment_source=assignment_source,
        )

    def _committed_divisions(
        self, committed: Sequence[Mapping[str, object]]
    ) -> list[_CommittedDivisionRow]:
        """The division rows of *committed*, each with its store's resolved path.

        Raises:
            ValueError: If a row's division cells do not read.
        """
        found: list[_CommittedDivisionRow] = []
        for row in committed:
            division = row_division(row)
            key = self._stored_row_key(row)
            if division is None or key is None:
                continue
            found.append(
                _CommittedDivisionRow(
                    path=Path(key.abs_path),
                    camera_serial=read_link_cell(row, "camera_serial"),
                    division=CommittedDivision(
                        row=division,
                        rule=DivisionRule.from_cell(
                            read_link_cell(row, "division_rule")
                        ),
                        recording=read_link_cell(row, "group"),
                    ),
                )
            )
        return found

    def _row_on_paths(self, row: Mapping[str, object], paths: Collection[Path]) -> bool:
        """Whether *row* indexes one of the resolved store *paths*."""
        key = self._stored_row_key(row)
        return key is not None and Path(key.abs_path) in paths

    def _recordings_to_divide(
        self,
        stores: Sequence[ProbedEntry],
        division: MediaDivision,
        committed: Sequence[_CommittedDivisionRow],
    ) -> list[list[ProbedEntry]]:
        """Group probed stores into the recordings that *division* makes of them.

        By store, each store is a recording. By camera, a recording is every
        store of one camera serial, with the stores of the serial that the
        index divides by camera already and that still exist. A store without
        a serial is a recording alone, which the planner refuses by camera.
        """
        if division.identity == "store":
            return [[store] for store in stores]
        by_serial: dict[str, list[ProbedEntry]] = {}
        alone: list[list[ProbedEntry]] = []
        for store in stores:
            if store.camera_serial:
                by_serial.setdefault(store.camera_serial, []).append(store)
            else:
                alone.append([store])
        for serial, members in by_serial.items():
            known = {member.path.resolve() for member in members}
            earlier = sorted(
                {
                    committed_row.path
                    for committed_row in committed
                    if committed_row.camera_serial == serial
                    and committed_row.division.rule.identity == "camera"
                    and committed_row.path not in known
                    and committed_row.path.is_dir()
                }
            )
            for path in earlier:
                members.extend(
                    found
                    for found in self._probe_dir_rows([path], set(), recursive=False)
                    if found.media_type == "imgstore"
                )
        return [by_serial[serial] for serial in sorted(by_serial)] + alone

    def _divided_rows(
        self,
        recordings: Sequence[tuple[Sequence[ProbedEntry], MediaDivision]],
        *,
        include_partial: bool,
        committed: Sequence[Mapping[str, object]],
        assignment_source: AssignmentSource,
        reassign: bool = False,
    ) -> _DividedRows:
        """Divide each recording of *recordings* into its media index rows.

        Every refusal is raised before a row is returned, so a write divides
        all of its recordings or none.

        Args:
            recordings: Each recording's probed stores and its declared division.
            include_partial: Whether to write each recording's trailing sequence
                that is still recording.
            committed: The media index rows before the write.
            assignment_source: ``"assigned"`` for a division scope, which the
                rows keep. A scan writes ``scan-imgstore-division`` rows.
            reassign: Whether a scan may divide a store whose identity a caller
                assigned.

        Raises:
            DivisionRefused: As
                :func:`~mosaic.core.media.division_index.plan_recording` raises
                it, or for a store a caller assigned, a name that another
                recording has, a recording that holds other rows, or a
                re-division of a recording whose entries have outputs.
            MediaProbeError: If a store does not open.
        """
        from .media.division_index import (
            PlannedRecording,
            StoreToDivide,
            plan_recording,
            recording_rows,
        )

        divisions = self._committed_divisions(committed)
        planned: list[tuple[PlannedRecording, list[StoreToDivide]]] = []
        redivided: list[tuple[str, str]] = []
        for stores, division in recordings:
            to_divide = [
                StoreToDivide(
                    path=store.path.resolve(),
                    store_row=self._store_row(store, assignment_source),
                    sync_uuid=store.sync_uuid,
                )
                for store in stores
            ]
            paths = {store.path for store in to_divide}
            own = [row for row in committed if self._row_on_paths(row, paths)]
            if assignment_source != "assigned" and not reassign:
                _refuse_assigned_store(own)
            frozen = [
                dataclasses.replace(
                    found.division,
                    row=dataclasses.replace(
                        found.division.row, abs_path=self.relative_to_root(found.path)
                    ),
                )
                for found in divisions
                if found.path in paths
            ]
            result = plan_recording(
                to_divide,
                frames=division.frames,
                identity=division.identity,
                name=division.name,
                include_partial=include_partial,
                committed=frozen,
            )
            recording = result.plan.recording
            if result.redivision is not None:
                self._refuse_redivision_with_outputs(
                    recording, result.redivision, entries_of_rows(own)
                )
                redivided.append((recording, result.redivision))
            whole = [row for row in own if row_division(row) is None]
            if whole:
                whole_paths = sorted({read_link_cell(row, "abs_path") for row in whole})
                why = f"the stores {whole_paths} are indexed whole"
                self._refuse_redivision_with_outputs(
                    recording, why, entries_of_rows(whole)
                )
                redivided.append((recording, why))
            planned.append((result, to_divide))

        divided = frozenset(
            store.path for _, divided_stores in planned for store in divided_stores
        )
        plans = tuple(result.plan for result, _ in planned)
        self._refuse_taken_recording_names(plans, committed, divided)
        rows: list[dict[str, object]] = []
        for result, divided_stores in planned:
            rows.extend(recording_rows(result, divided_stores))
        for row in rows:
            row["assignment_source"] = (
                "assigned"
                if assignment_source == "assigned"
                else DIVISION_ASSIGNMENT_SOURCE
            )
        return _DividedRows(
            rows=rows,
            plans=plans,
            paths=divided,
            redivided=tuple(redivided),
            display_names={
                (result.plan.recording, sequence): name
                for result, _ in planned
                for sequence, name in result.display_names().items()
            },
        )

    def _refuse_redivision_with_outputs(
        self, recording: str, why: str, entries: Iterable[Entry]
    ) -> None:
        """Refuse to divide *recording* again while *entries* have outputs.

        Raises:
            DivisionRefused: Naming each entry's outputs.
        """
        outputs = self.entry_outputs(entries)
        held = {entry: found for entry, found in outputs.items() if found}
        if not held:
            return
        listed = "; ".join(
            f"{entry_token(*entry)}: {', '.join(found)}"
            for entry, found in sorted(held.items())
        )
        message = (
            f"the recording {recording!r} would be divided again, because {why}, "
            f"and the entries it replaces have outputs ({listed}). Remove them "
            "first, or keep the recording's division."
        )
        raise DivisionRefused("redivision_with_outputs", message)

    def _refuse_taken_recording_names(
        self,
        plans: Sequence[DivisionPlan],
        committed: Sequence[Mapping[str, object]],
        divided: Collection[Path],
    ) -> None:
        """Refuse a recording name that another recording or entry has.

        A name is taken by another recording of the write, by the rows of a
        store other than the divided ones, and by an empty-group sequence of
        that name. A continuous group whose sequences are not divisions of the
        divided stores is a recording declared by hand.

        Raises:
            DivisionRefused: ``recording_name_taken`` or ``mixed_recording``.
        """
        seen: set[str] = set()
        for plan in plans:
            if plan.recording in seen:
                message = (
                    f"two recordings of this write are named {plan.recording!r}. "
                    "Give one of them another name."
                )
                raise DivisionRefused("recording_name_taken", message)
            seen.add(plan.recording)
        elsewhere = [row for row in committed if not self._row_on_paths(row, divided)]
        own = entries_of_rows(
            row for row in committed if self._row_on_paths(row, divided)
        ) | {
            (plan.recording, division.row.sequence)
            for plan in plans
            for division in plan.divisions
        }
        indexed = self._indexed_entries_beside_media()
        named = entries_of_rows(elsewhere) | indexed
        for plan in plans:
            name = plan.recording
            stores = sorted(
                read_link_cell(row, "abs_path")
                for row in elsewhere
                if read_link_cell(row, "group") == name
                and row_division(row) is not None
            )
            members = sorted(
                {
                    sequence
                    for group, sequence in entries_of_rows(
                        row for row in elsewhere if row_division(row) is None
                    )
                    | indexed
                    if group == name and (group, sequence) not in own
                }
            )
            if members and not stores and self.recording_kind(name) == "divided":
                message = (
                    f"the continuous group {name!r} holds the sequences {members}, "
                    "which are not divisions of its store. A recording is the "
                    "divisions of its stores or sequences declared by hand, and "
                    "not both."
                )
                raise DivisionRefused("mixed_recording", message)
            taken = (
                f"the recording of the stores {sorted(set(stores))}"
                if stores
                else f"the group of {members}"
                if members
                else "an empty-group sequence"
                if ("", name) in named
                else None
            )
            if taken is not None:
                message = (
                    f"the recording name {name!r} is {taken} already. Give the "
                    "division a name of its own."
                )
                raise DivisionRefused("recording_name_taken", message)

    def _indexed_entries_beside_media(self) -> set[Entry]:
        """Every entry of the tracks_raw, tracks and labels indexes."""
        return (
            root_index_entries(self, "tracks_raw") | self._tracks_and_labels_entries()
        )

    def _prepare_divided_manifest(self, divided: _DividedRows) -> None:
        """Raise the manifest to 4 before *divided*'s rows are written.

        Called under the media index lock once the write's refusals pass, so a
        refused write leaves the manifest as it was. The groups that
        :meth:`_declare_divided_recordings` adds after the write are checked
        here, so the declaration does not refuse once the rows are written.

        Raises:
            ValueError: If the manifest refuses the recordings as continuous
                groups. Nothing is written.
        """
        names = divided.recordings
        if names:
            self.mutate_manifest(
                lambda manifest: _prepare_divided_manifest(manifest, names)
            )

    def _declare_divided_recordings(self, divided: _DividedRows) -> None:
        """Declare each recording of *divided* continuous once its rows are written.

        Each division without a sequence label is labeled by its wall-clock
        start. A later write labels the divisions written before it. Each
        recording divided again is reported on stderr, with why.
        """
        names = divided.recordings
        if names:
            self.mutate_manifest(
                lambda manifest: _declare_divided_groups(manifest, names)
            )
        self._label_divisions(divided.display_names)
        for recording, why in divided.redivided:
            print(
                f"[divide] the recording {recording!r} is divided again, because "
                f"{why}. Its entries have no outputs, and its earlier divisions "
                "are replaced.",
                file=sys.stderr,
            )

    def _label_divisions(self, display_names: Mapping[Entry, str]) -> None:
        """Record each label of *display_names* for an entry without a label.

        An entry's other label cells, such as its promotion lineage, are kept.
        """

        def label(row: SequenceLabelRow) -> SequenceLabelRow | None:
            if row.display_name:
                return None
            name = display_names[row.group, row.sequence]
            return dataclasses.replace(row, display_name=name)

        amend_sequence_labels(self, sorted(display_names), label)

    def entry_outputs(self, entries: Iterable[Entry]) -> dict[Entry, list[str]]:
        """Return the outputs each of *entries* has, by artifact.

        An output is a tracks variant, a converted labels variant, a feature run
        or a media variant that covers or indexes the entry, and, when the
        tracking kinds are registered (``mosaic.tracking.register_ops``), a
        tracker run, a frame run, a frame-predictions run or a frame
        selection. Each output is named by its kind and identifiers. An entry
        without outputs maps to an empty list.

        Raises:
            DivisionRefused: ``tracking_unregistered``, when the tracking kinds
                are not registered and a root they own holds files.
        """
        from .pipeline.inventory import inventory

        wanted = sorted(set(entries))
        found: dict[Entry, list[str]] = {entry: [] for entry in wanted}
        if not wanted:
            return found
        held = inventory(self, kinds=_OUTPUT_KINDS)
        unlisted = held.unavailable_kinds & frozenset(_TRACKING_OUTPUT_KINDS)
        if unlisted:
            occupied = self._tracking_roots_with_files()
            if occupied:
                message = (
                    f"the roots {occupied} hold tracking outputs that this process "
                    "cannot list. Call mosaic.tracking.register_ops() first."
                )
                raise DivisionRefused("tracking_unregistered", message)
        for record in held.records:
            covered = {
                (key[0], key[1])
                for key in record.rows | record.coverage.present
                if not isinstance(key, str)
            }
            label = _output_label(record.ref)
            for entry in wanted:
                if entry in covered:
                    found[entry].append(label)
        return found

    def _tracking_roots_with_files(self) -> list[str]:
        """The tracking-owned roots that hold a file, by root key."""
        keys = [*TRACKING_ROOTS, *_TRACKING_OUTPUT_ROOTS]
        occupied: list[str] = []
        for key in keys:
            if not self.has_root(key):
                continue
            root = self.get_root(key)
            if root.is_dir() and any(path.is_file() for path in root.rglob("*")):
                occupied.append(key)
        return occupied

    def _divide_scopes(
        self,
        scopes: Sequence[MediaIndexScope],
        *,
        exts: set[str],
        recursive: bool,
        include_partial: bool,
        committed: Sequence[Mapping[str, object]],
    ) -> _DividedRows:
        """Divide the store that each division scope names (:meth:`_divided_rows`).

        Scopes that share a ``group`` and a division are one recording by
        camera, and a scope with an empty ``group`` derives its recording's
        name.

        Raises:
            ValueError: If a division scope names a sequence, or two names, or
                two scopes of one group declare different divisions.
            DivisionRefused: ``not_one_store`` for a directory that is not one
                store, and as :meth:`_divided_rows` raises it.
        """
        declared: dict[tuple[str, MediaDivision], list[ProbedEntry]] = {}
        for scope in scopes:
            division = scope.division
            if division is None:
                continue
            if scope.sequence:
                msg = (
                    f"the division scope of {scope.directory} names the sequence "
                    f"{scope.sequence!r}. A division names its own sequences; "
                    "give sequence=''."
                )
                raise ValueError(msg)
            if scope.group and division.name and scope.group != division.name:
                msg = (
                    f"the division scope of {scope.directory} names the recording "
                    f"{scope.group!r} and its division names {division.name!r}"
                )
                raise ValueError(msg)
            name = scope.group or division.name
            found = self._probe_dir_rows([scope.directory], exts, recursive)
            store = scope.directory.resolve()
            if [entry.path.resolve() for entry in found] != [store] or (
                found[0].media_type != "imgstore"
            ):
                message = (
                    f"a division scope names one imgstore directory, and "
                    f"{scope.directory} holds {len(found)} media entries"
                    + ("" if found else ", or is not a store")
                )
                raise DivisionRefused("not_one_store", message)
            ruled = division.model_copy(update={"name": name})
            for (other, other_division), _ in declared.items():
                if name and other == name and other_division != ruled:
                    msg = (
                        f"the division scopes of the recording {name!r} declare "
                        "different divisions"
                    )
                    raise ValueError(msg)
            declared.setdefault((name or "", ruled), []).extend(found)
        committed_divisions = self._committed_divisions(committed)
        recordings = [
            (stores, division)
            for (_, division), entries in declared.items()
            for stores in self._recordings_to_divide(
                entries, division, committed_divisions
            )
        ]
        return self._divided_rows(
            recordings,
            include_partial=include_partial,
            committed=committed,
            assignment_source="assigned",
        )

    def write_media_index(
        self,
        scopes: Iterable[MediaIndexScope],
        *,
        extensions: Tuple[str, ...] = (".mp4", ".avi"),
        index_filename: str = "index.csv",
        recursive: bool = True,
        include_partial: bool = False,
    ) -> MediaIndexResult:
        """Project explicit sequence assignments into a valid media index.

        The assignment-driven counterpart to :meth:`index_media`: rather than
        deriving each file's (group, sequence) from the track keymap, the caller
        passes one :class:`MediaIndexScope` per affected (group, sequence) -- its
        media_raw subdir, the explicit identity, and this session's arranged
        order. Every file found under a scope directory is (re)probed and given
        that scope's identity; ``video_order`` is densified per
        (group, sequence, camera) as "existing videos first by prior order, then
        this session's videos by arranged position", **over the passed scopes'
        sequences only**; every index row not under any scope directory (other
        sequences, external ``abs_path`` values) is preserved verbatim, order
        cell included; and the file is written atomically with root-relative
        ``abs_path``. This is the single entry point the API's
        upload finalize calls -- the API owns none of these semantics itself.

        Overlapping scope directories are caller error, as they are for
        :meth:`write_tracks_raw_index`: a file scanned by two scopes is written
        once, first occurrence winning, and the collision is reported to stderr.
        The return carries the written index path and any
        :class:`MediaIndexDisagreement`s -- files whose stored uuid differed from
        the injected one, reported not raised.

        Unsynchronized imgstores under one scope join its sequence as clips when
        they share a ``camera_serial``. A scope holding unsynchronized stores of
        two or more distinct serials is refused before anything is written.

        A scope with a ``division`` divides its one store into sequences of
        whole chunks (:class:`MediaIndexScope`). Its rows replace every row of
        the store, the undivided one included, in the one index write, and the
        recording is then declared continuous. *include_partial* writes each
        recording's trailing sequence that is still recording, which is
        otherwise held back. The result's ``divisions``, ``held_back`` and
        ``refused`` report each recording's plan. A recording divided before
        keeps its divisions. A division with another rule or name divides the
        recording again, which is refused while its entries have outputs
        (:meth:`entry_outputs`), and the division of a store indexed whole is
        refused while that store's entries have outputs. A re-division without
        outputs is reported on stderr. Call ``mosaic.tracking.register_ops()``
        first, so that the tracking outputs are listed. The manifest is raised
        to version 4 before the index is written, and the recordings are
        declared continuous after it.

        Raises:
            StoreGroupingError: If a scope holds unsynchronized imgstores of
                distinct non-empty ``camera_serial`` values.
            DivisionRefused: If a division scope's store cannot be divided, each
                refusal with its ``reason``. Nothing is written.
        """
        media_root = self.get_root(self.resolve_media_root())
        index_path = media_root / index_filename
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
        scope_list = list(scopes)
        scope_dirs = [scope.directory.resolve() for scope in scope_list]
        division_scopes = [scope for scope in scope_list if scope.division is not None]

        # --- Probe phase: expensive, read-only, and deliberately unlocked. ---
        #
        # Everything here either probes the filesystem or builds an advisory map,
        # and probing a large upload takes minutes. Holding the index lock across
        # it would blow index_lock's DEFAULT_TIMEOUT_S, which is tuned for a CSV
        # rewrite measured in milliseconds, so a concurrent writer would fail on
        # a healthy system. The authoritative read -- the rows to preserve and
        # the prior order they carry -- happens again inside the lock below.
        existing = _read_media_index(index_path)

        # The prior row for each scoped file, and -- when its file has not moved
        # -- the measurement that row already holds. Built in one pass, on one
        # key, because they answer two halves of the same question.
        #
        # Keyed on the **resolved path**, never the basename. Two sequences can
        # each hold a "video.mp4", and a basename key would serve one sequence's
        # measurement for the other's file. This replaces a basename-keyed uuid
        # map that degraded to reporting nothing on a collision: correct, but it
        # meant the case most in need of a comparison was the one case that got
        # none, twenty lines above a cache that was path-keyed for this reason.
        #
        # The cache is the WRITE path being cheap, and it is allowed to be. The
        # audit path -- reprobe-media -- re-probes unconditionally and is what
        # detects a file replaced in place on a share. Do not apply this shortcut
        # there: a cache that skips the probe skips the comparison that IS the
        # check. What escapes here is narrower than that sounds -- a replacement
        # preserving size *and* mtime exactly -- and is recorded in `drift.py`.
        prior_row_by_key: dict[MediaRowKey, dict[str, str]] = {}
        cached_facts_by_key: dict[MediaRowKey, MediaFacts] = {}
        for row in existing:
            if not self._row_under_dirs(row, scope_dirs):
                continue
            stored = read_link_cell(row, "abs_path")
            if not stored:
                continue
            resolved = self.resolve_path(stored).resolve()
            key = dataclasses.replace(media_row_key(row), abs_path=str(resolved))
            prior_row_by_key[key] = dict(row)
            facts = row_facts_or_none(row)
            if facts is None:
                continue
            try:
                stat_result = resolved.stat()
            except OSError:
                continue
            if str(row.get("size_bytes", "")) != str(stat_result.st_size):
                continue
            if read_link_cell(row, "mtime_iso") != mtime_iso(stat_result.st_mtime):
                continue
            cached_facts_by_key[key] = facts

        disagreements: list[MediaIndexDisagreement] = []
        drift: list[MediaDrift] = []

        # Probe each scope directory and assign its explicit identity; collect
        # this session's arranged positions keyed (group, sequence, basename).
        divided = (
            self._divide_scopes(
                division_scopes,
                exts=exts,
                recursive=recursive,
                include_partial=include_partial,
                committed=existing,
            )
            if division_scopes
            else _DividedRows()
        )
        fresh: list[dict[str, object]] = []
        session_positions: dict[tuple[str, str, str], int] = {}
        for scope in scope_list:
            if scope.division is not None:
                continue
            group_safe = to_safe_name(scope.group) if scope.group else ""
            sequence_safe = to_safe_name(scope.sequence)
            entries = self._probe_dir_rows(
                [scope.directory],
                exts,
                recursive,
                scope.facts_by_name,
                cached_facts_by_key,
            )
            self._refuse_unsynced_cameras(scope, entries)
            for entry in entries:
                # A probed imgstore supplies its own camera_serial/sync_uuid from
                # store metadata (read once in _probe_dir_rows); scope.camera is
                # only an override for a plain video the caller tags with a camera.
                #
                # The same inequality means two different things, and only
                # `entry.origin` tells them apart. Against an INJECTED
                # measurement it is a caller describing a different file than the
                # row did -- an ordinary re-upload, which is what
                # MediaIndexDisagreement has always reported. Against a PROBED
                # one it is the bytes having moved under a stable path, which is
                # drift. A CACHED measurement *is* the stored one, so comparing
                # it could only ever produce a false positive.
                prior = prior_row_by_key.get(
                    MediaRowKey.of(str(entry.path.resolve()), None)
                )
                if prior is not None and entry.origin != "cached":
                    change = classify_identity(prior, entry.probe)
                    if change in ("content_digest_changed", "video_uuid_changed"):
                        if entry.origin == "injected":
                            disagreements.append(
                                MediaIndexDisagreement(
                                    basename=entry.path.name,
                                    prior_uuid=media_row_uuid(prior),
                                    injected_uuid=str(
                                        entry.probe.get("video_uuid", "")
                                    ),
                                )
                            )
                        else:
                            drift.append(
                                MediaDrift(
                                    stored_path=read_link_cell(prior, "abs_path"),
                                    resolved_path=entry.path,
                                    change=change,
                                    recorded_uuid=media_row_uuid(prior),
                                    measured_uuid=str(
                                        entry.probe.get("video_uuid", "")
                                    ),
                                    recorded_digest=read_link_cell(
                                        prior, "content_digest"
                                    ),
                                    measured_digest=str(
                                        entry.probe.get("content_digest", "")
                                    ),
                                )
                            )
                fresh.append(
                    build_media_index_row(
                        path=entry.path,
                        stat=entry.stat,
                        to_store_path=self.relative_to_root,
                        group=scope.group,
                        sequence=scope.sequence,
                        group_safe=group_safe,
                        sequence_safe=sequence_safe,
                        camera=entry.camera_serial or scope.camera,
                        sync_uuid=entry.sync_uuid,
                        camera_serial=entry.camera_serial,
                        probe=entry.probe,
                        media_type=entry.media_type,
                        assignment_source="assigned",
                    )
                )
            for name, position in scope.order_by_name.items():
                session_positions[(scope.group, scope.sequence, name)] = position

        # Overlapping scope directories are caller error, and the sibling
        # write_tracks_raw_index already says so and dedupes on abs_path. Media
        # did not, so one file scanned by two scopes landed twice under two
        # sequence names -- which puts one video uid into two sequences' media
        # compositions, or twice into one.
        #
        # Fresh-vs-fresh only: _row_under_dirs already guarantees preserved and
        # fresh are disjoint by resolved path, and a blanket dedup across both
        # would drop a legitimately duplicated external row. Reported rather than
        # silently collapsed, because nothing else tells the caller its scopes
        # overlap. Note this is not the same thing as two byte-identical files in
        # one sequence, which legitimately share one video_uuid and are two rows.
        fresh = self._dedupe_scope_rows(fresh)
        fresh.extend(divided.rows)

        # --- Commit phase: cheap, and locked. ---
        #
        # Re-read authoritatively inside the lock rather than reusing the probe
        # phase's snapshot: another writer may have landed rows while this one
        # was probing, and they are exactly the rows this write must preserve.
        # Everything from here is in-memory work plus one terminal
        # ``atomic_write``. Not because ``index_lock`` requires it any more -- it
        # holds a sidecar no rename touches -- but because the probe phase above
        # is unbounded I/O and the lock's timeout is tuned for a CSV rewrite.
        #
        # ``_read_media_index`` uses ``csv.DictReader`` and yields ``[]`` for the
        # zero-byte file the lock leaves on a first write. A pandas reader moved
        # inside this block would raise ``EmptyDataError`` there.
        index_path.parent.mkdir(parents=True, exist_ok=True)
        touched = {
            (scope.group, scope.sequence)
            for scope in scope_list
            if scope.division is None
        } | entries_of_rows(divided.rows)
        with index_lock(index_path):
            committed = _read_media_index(index_path)
            prior_order = build_prior_order(committed)
            preserved = [
                row
                for row in committed
                if not self._row_under_dirs(row, scope_dirs)
                and not self._row_on_paths(row, divided.paths)
            ]
            written = entries_of_rows(fresh)
            self.refuse_entry_key_collision(
                written, self._entries_beside_media_write(written, committed, preserved)
            )
            self._prepare_divided_manifest(divided)
            # A store's scale is a fact about its pixels, which a division keeps.
            self._keep_calibration(divided.rows, committed)

            # Carry transcode derivative links onto the fresh rows (a re-finalize
            # of a transcoded sequence must not drop its routing links), merge
            # with the preserved rows, densify video_order, and write atomically.
            #
            # Densified over the sequences THIS write was given, not over the
            # whole file. ``densify_video_order`` renumbers every row it is
            # handed, and ``build_prior_order`` skips a blank ``video_order``
            # cell -- so an untouched sequence carrying blank cells was being
            # renumbered by name during someone else's upload, which contradicts
            # this method's own "preserved verbatim" contract and moves a media
            # composition hash for a sequence nobody named.
            #
            # The partition key is membership in *touched*, not
            # preserved-vs-fresh: a preserved row can belong to a touched
            # sequence when its ``abs_path`` lives outside the scope directory,
            # and leaving it out would collide its order with the fresh rows.
            # ``densify_video_order`` mutates in place and its return is
            # discarded, so *merged* keeps its construction order and the file's
            # row order is unchanged.
            #
            # One behaviour change to know: a legacy index with gappy orders is
            # no longer globally re-densified as a side effect of an unrelated
            # write. That cleanup was doing damage; offering it deliberately is a
            # repair command, not a byproduct of an upload.
            self._carry_forward_derivative_links(fresh, index_path)
            merged: list[dict[str, object]] = [dict(row) for row in preserved]
            merged.extend(fresh)
            in_scope = [
                row
                for row in merged
                if (str(row.get("group", "")), str(row.get("sequence", ""))) in touched
            ]
            densify_video_order(
                in_scope,
                session_positions=session_positions,
                prior_order=prior_order,
            )
            write_media_index_rows(index_path, frame_from_rows(merged))
        self._declare_divided_recordings(divided)
        self._write_media_compositions(merged)
        for moved in drift:
            print(
                f"[write_media_index] drift: {moved.stored_path} "
                f"({moved.change}) recorded={moved.recorded_uuid or '-'} "
                f"measured={moved.measured_uuid or '-'}",
                file=sys.stderr,
            )
        return MediaIndexResult(
            index_path=index_path,
            disagreements=disagreements,
            drift=drift,
            divisions=divided.plans,
        )

    def _write_media_compositions(self, rows: list[dict[str, object]]) -> None:
        """Project the media rows just committed into ``media_raw/sequences.csv``.

        Computed from the in-memory rows rather than by re-reading the file:
        cheaper, and it is exactly the state this process committed. The repair
        path recomputes from disk and both go through the same functions, so the
        two cannot diverge.

        Skipped when the media root resolves to ``media`` -- a legacy dataset
        with no ``media_raw``, where the index holds derivatives. A derivative is
        named by its source's identity and has no composition of its own (rule
        P6), so the honest answer there is no row at all.

        Called after the index write and outside its lock. See
        :mod:`mosaic.core.pipeline.sequence_index` for why that order cannot be
        inverted.
        """
        if self.resolve_media_root() != "media_raw":
            return
        members = media_members_from_rows(rows)
        compositions = {key: media_composition(group) for key, group in members.items()}
        _ = write_sequence_compositions(self, "media_raw", compositions=compositions)

    def _write_tracks_raw_compositions(self, rows: list[dict[str, object]]) -> None:
        """Project the raw-track rows just committed into ``tracks_raw/sequences.csv``."""
        members = source_members_from_rows(rows)
        compositions = {
            key: tracks_raw_composition(group) for key, group in members.items()
        }
        _ = write_sequence_compositions(self, "tracks_raw", compositions=compositions)

    def _write_labels_raw_compositions(self, rows: list[dict[str, object]]) -> None:
        """Project the raw-label rows just committed into ``labels_raw/sequences.csv``.

        The ``tracks_raw`` sibling, differing only in which composition function
        stamps the members: the raw-source index schema and its member grouping
        are shared, so ``source_members_from_rows`` serves both roots.
        """
        members = source_members_from_rows(rows)
        compositions = {
            key: labels_raw_composition(group) for key, group in members.items()
        }
        _ = write_sequence_compositions(self, "labels_raw", compositions=compositions)

    def display_names(self) -> dict[tuple[str, str], str]:
        """Every recorded sequence label, keyed by its ``(group, sequence)`` token.

        One read for a listing, rather than one per row. Sequences with no
        recorded label are absent from the mapping, so a caller falls back to the
        token by ``.get(key, default)`` rather than by testing for emptiness.
        """
        frame = read_sequence_labels(self)
        return {
            (str(row["group"]), str(row["sequence"])): str(row["display_name"])
            for _, row in frame.iterrows()
            if str(row["display_name"])
        }

    def display_name(self, group: str, sequence: str) -> str:
        """What to call this sequence to a human: its label, or its token.

        The fallback is the whole design. A token is a perfectly good name until
        someone chooses a better one, so a dataset that has never recorded a
        label reads exactly as it did before labels existed -- and no caller
        needs to branch.
        """
        return self.display_names().get(
            (group, sequence), make_entry_key(group, sequence)
        )

    def set_display_name(
        self, group: str, sequence: str, name: str, *, display_group: str = ""
    ) -> None:
        """Record what to call *(group, sequence)*. Touches no file but this one.

        That is item 4.1's entire point: relabelling is metadata, so it must not
        move a directory, rewrite an ``abs_path``, or change the token that every
        filename and every index join key is built from. Passing ``""`` clears
        the label and returns the sequence to being called by its token. The
        sequence's ``derived_from`` lineage is kept.
        """
        entry = (
            validate_entry_name(group, "group"),
            validate_entry_name(sequence, "sequence"),
        )
        amend_sequence_labels(
            self,
            [entry],
            lambda row: dataclasses.replace(
                row, display_group=display_group, display_name=name
            ),
        )

    def rebuild_sequence_index(self, root: SourceRoot) -> Path | None:
        """Recompute *root*'s per-sequence index from that root's ``index.csv``.

        The repair entry point, for a projection left absent or stale by a crash
        between the two writes, or by a hand-edited index. Returns the written
        path, or ``None`` when the root is unset.

        It reads from disk where the writers project from the rows they just
        committed, and both go through the same composition functions -- so this
        is also the tests' oracle: what a writer wrote and what a rebuild
        produces must agree, and a divergence is a bug in one of them.
        """
        try:
            _ = self.get_root(root)
        except KeyError:
            return None
        if root == "media_raw":
            if self.resolve_media_root() != "media_raw":
                return None
            rows = [dict(row) for row in self.read_media_index()]
            self._write_media_compositions(rows)
        elif root == "labels_raw":
            index_path = self.get_root(root) / "index.csv"
            rows = [dict(row) for row in _read_tracks_raw_index(index_path)]
            self._write_labels_raw_compositions(rows)
        else:
            index_path = self.get_root(root) / "index.csv"
            rows = [dict(row) for row in _read_tracks_raw_index(index_path)]
            self._write_tracks_raw_compositions(rows)
        return sequence_index_path(self, root)

    def _prior_digests(self, index_path: Path) -> dict[Path, str]:
        """A prior row's ``md5``, reusable when the file it describes has not moved.

        Keyed on the resolved absolute path and admitted only when the stored
        ``size_bytes`` and ``mtime_iso`` both still match the file on disk --
        the same shape, and for the same reason, as the media write path's facts
        cache. Without it, turning checksums on by default would re-hash every
        raw file under a scope directory on every re-finalize, which for a corpus
        on a slow share is the difference between a usable default and one
        everybody turns off.

        A size-and-mtime match is **not** proof of identical content, and this
        is the write path, which is allowed to be cheap. The audit that catches a
        file replaced in place with the same size and timestamp is a separate
        unconditional re-hash, exactly as ``reprobe-media`` is for measurements;
        do not apply this shortcut there, because a cache that skips the read
        skips the comparison that IS the check.
        """
        digests: dict[Path, str] = {}
        for row in _read_tracks_raw_index(index_path):
            stored = str(row.get("abs_path", "") or "").strip()
            digest = str(row.get("md5", "") or "").strip()
            if not stored or not digest:
                continue
            resolved = self.resolve_path(stored).resolve()
            try:
                stat_result = resolved.stat()
            except OSError:
                continue
            if str(row.get("size_bytes", "")) != str(stat_result.st_size):
                continue
            if str(row.get("mtime_iso", "")) != mtime_iso(stat_result.st_mtime):
                continue
            digests[resolved] = digest
        return digests

    @staticmethod
    def _dedupe_scope_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """Keep one row per row key, first occurrence winning.

        The key is the stored ``abs_path`` and a division's chunks
        (:func:`~mosaic.core.media.facts_columns.media_row_key`), so the
        divisions of one store are each kept. Two :class:`MediaIndexScope`s over
        one directory each probe the same files and each stamp their own
        ``(group, sequence)``, so the same file lands twice under two
        identities. Reported to stderr rather than collapsed silently: the
        scopes come from a caller who believes they are disjoint, and nothing
        else would tell them otherwise.
        """
        seen: dict[MediaRowKey, dict[str, object]] = {}
        kept: list[dict[str, object]] = []
        for row in rows:
            stored = media_row_key(row)
            first = seen.get(stored)
            if first is not None:
                print(
                    f"[write_media_index] {stored.abs_path} is under two scopes "
                    f"({first.get('group', '')!r}, {first.get('sequence', '')!r}) "
                    f"and ({row.get('group', '')!r}, {row.get('sequence', '')!r}); "
                    f"keeping the first.",
                    file=sys.stderr,
                )
                continue
            seen[stored] = row
            kept.append(row)
        return kept

    def read_media_index(
        self, index_filename: str = "index.csv"
    ) -> list[dict[str, str]]:
        """Read the **originals** index as string-cell records (empty if absent).

        A dataset has two media indexes, and this reads one of them: the
        originals under :meth:`resolve_media_root` -- ``media_raw/index.csv``
        wherever that root is declared, ``media/index.csv`` only on a dataset
        predating the split. The other, ``media/index.csv`` holding one row per
        transcode derivative, is reached through :meth:`media_routing_context`.

        The bare name is deliberate: it cannot address the derivative index, so
        there is nothing here to disambiguate. What varies is which *root* holds
        the originals, which is why this resolves rather than naming one.
        """
        media_root = self.get_root(self.resolve_media_root())
        return _read_media_index(media_root / index_filename)

    def sweep_tracking(
        self,
        *,
        apply: bool,
        roots: Sequence[str] | None = None,
        retention_overrides: Mapping[str, float] | None = None,
        execution_id: str = "",
        now: "datetime.datetime | None" = None,
    ) -> "SweepReport":
        """Reclaim finished tracker intermediates under ``_tracking`` (item 8.4).

        Dry-run unless *apply*. Walks ``<_tracking>/<tool>/<run_id>/<entry>/``,
        classifies each entry directory from its markers, its index row and its
        age, and removes only the classes the decision module authorizes -- rows
        first, then files, so a crash between them leaves rows naming absent
        files (which :meth:`reindex` repairs) rather than files nothing names.

        **The claim is the gate.** A directory a live execution holds is never a
        candidate, and neither is one this dataset's index does not yet name:
        several producers append their rows only after the whole batch, so
        mid-run most finished directories are unrowed and every one of them is
        work in progress.

        Three gates decline before anything is read, each returning a report with
        ``considered=False`` rather than raising. The third is the one that
        matters most: a dataset whose tracker root still points inside
        ``tracks_raw`` (the pre-item-8.1 layout) is refused outright, because
        deleting under it would delete user uploads.

        Root resolution and the writes live here; the decisions live in
        :mod:`~mosaic.core.pipeline.sweep`, which knows nothing about datasets.

        Args:
            apply: Perform the deletions. Default is a report.
            roots: Restrict to these tracker root keys.
            retention_overrides: Per retention class, in days.
            execution_id: The asking execution, so its own claims read as
                ``mine``. A sweeper normally passes none, which matches nothing.
            now: Override for the current instant, for tests.

        Returns:
            A :class:`~mosaic.core.pipeline.sweep.SweepReport`.
        """
        from typing import get_args

        from .pipeline.sweep import (
            DeclineReason,
            RetentionClass,
            SweepEntry,
            classify_entry,
            declined_sweep,
            deletable,
            retention_days,
            summarize,
        )
        from .pipeline.tracking_roots import TRACKING_ROOTS

        if not self.has_root(TRACKING_ROOT):
            return declined_sweep("no-tracking-root")
        if legacy_tracking_roots(self.roots):
            return declined_sweep("legacy-layout")

        base = self.base_dir.resolve()
        wanted = set(roots) if roots is not None else set(TRACKING_ROOTS)
        overrides: dict[RetentionClass, float] = {}
        # From the alias rather than a literal pair: a window nobody can set is
        # a window silently ignored, and adding a retention class must not be
        # the kind of change that needs this line remembered.
        classes: tuple[RetentionClass, ...] = get_args(RetentionClass)
        for name, days in (retention_overrides or {}).items():
            if name in classes:
                overrides[name] = days

        keys = sorted(wanted & set(TRACKING_ROOTS))
        conversion_keys = [
            key for key in keys if TRACKING_ROOTS[key].retention == "conversion"
        ]
        other_keys = [key for key in keys if key not in set(conversion_keys)]
        promoted_runs = self._promoted_from()

        decided: list[SweepEntry] = []

        def classify_root(key: str, pinned: frozenset[Path]) -> DeclineReason | None:
            if not self.has_root(key):
                return None
            root = self.get_root(key)
            if base not in root.resolve().parents and root.resolve() != base:
                return "root-outside-dataset"
            if not root.exists():
                return None
            rowed = self._rowed_entries(key)
            window = retention_days(key, overrides)
            for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                for entry_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
                    decided.append(
                        classify_entry(
                            entry_dir,
                            root_key=key,
                            run_id=run_dir.name,
                            entry=entry_dir.name,
                            run_log_base=self.base_dir,
                            execution_id=execution_id,
                            rowed=(run_dir.name, entry_dir.name) in rowed,
                            promoted=run_dir.name in promoted_runs,
                            max_age_days=window,
                            pinned=entry_dir.resolve() in pinned,
                            now=now,
                        )
                    )
            return None

        # Consumers first, then the conversions they pin. One pass either way:
        # a slot whose last reader is reclaimed in *this* run is reclaimed with
        # it, and one whose reader survives is refused.
        for key in other_keys:
            reason = classify_root(key, frozenset())
            if reason is not None:
                return declined_sweep(reason)
        if conversion_keys:
            going = {
                entry.path.resolve() for entry in decided if deletable(entry.verdict)
            }
            pinned_slots = self._pinned_conversions(going)
            for key in conversion_keys:
                reason = classify_root(key, pinned_slots)
                if reason is not None:
                    return declined_sweep(reason)

        report = summarize(decided, applied=apply)
        if not apply:
            return report
        return self._perform_sweep(report, deletable)

    def _pinned_conversions(self, going: set[Path]) -> frozenset[Path]:
        """Shared conversion slots a surviving tracker directory still reads.

        The pin is **derived from markers, never from an index row**, and the
        difference is the whole point. Several producers append their rows only
        after the whole batch, so mid-run a live tracking run has written its
        convert marker and no row at all. Keying this on the index would leave
        exactly that run's conversion looking unreferenced, and reclaim a `.pv`
        out from under a track phase that is reading it.

        *going* is what this sweep has already decided to remove, so a slot
        whose last reader is reclaimed in the same pass is not pinned by it --
        which is what makes the cascade single-pass rather than needing a second
        run to catch up.

        Walks every ``retention == "tracker"`` root **regardless of any root
        narrowing the caller asked for**: `--root trex-convert` must still read
        the tracker roots, or restricting the sweep to the cache would reclaim
        all of it.
        """
        from .pipeline.markers import read_phase_marker
        from .pipeline.tracking_roots import TRACKING_ROOTS

        pinned: set[Path] = set()
        for key, root_spec in sorted(TRACKING_ROOTS.items()):
            if root_spec.retention != "tracker" or not self.has_root(key):
                continue
            root = self.get_root(key)
            if not root.exists():
                continue
            for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                for entry_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
                    if entry_dir.resolve() in going:
                        continue
                    marker = read_phase_marker(entry_dir, "convert")
                    if marker is None or not marker.recorded_output:
                        continue
                    pinned.add(
                        self.resolve_path(marker.recorded_output).parent.resolve()
                    )
        return frozenset(pinned)

    def _promoted_from(self) -> set[str]:
        """Producer runs a promoted correction has superseded (items 8.4 / 8.6).

        Item 8.4 makes promotion the *primary* eviction signal and age the
        fallback, for a reason worth keeping: once a corrected track set is in
        ``tracks_raw``, the tracker output it was corrected from has served its
        purpose and its retention window is beside the point. The link is
        ``derived_from`` on the dataset-level ``sequences.csv`` -- the column item
        4.1 declared and left unused for exactly this.

        An absent label file means nothing has been promoted, which is the
        ordinary state and not an error.
        """
        path = sequence_label_path(self)
        if not path.exists():
            return set()
        try:
            frame = sequence_labels(path).read()
        except (OSError, pd.errors.ParserError):
            return set()
        if "derived_from" not in frame.columns:
            return set()
        # One cell can name several: a sequence corrected twice supersedes both
        # producer runs, and the label row that records them is one per sequence.
        return {
            run
            for value in frame["derived_from"]
            for run in str(value).split(",")
            if run
        }

    def _rowed_entries(self, root_key: str) -> set[tuple[str, str]]:
        """``(run_id, entry key)`` pairs this root's index names.

        An unreadable or absent index yields the empty set, which classifies
        every directory as ``unrowed`` -- and unrowed is refused, so the failure
        direction of not being able to read the index is "delete nothing".
        """
        from .pipeline.dataset_indexes import reconcilable_index

        factory = reconcilable_index(root_key)
        index_path = self.get_root(root_key) / "index.csv"
        if factory is None or not index_path.exists():
            return set()
        try:
            frame = pd.read_csv(index_path)
        except (OSError, pd.errors.ParserError):
            return set()
        if "run_id" not in frame.columns:
            return set()
        rowed: set[tuple[str, str]] = set()
        for _, row in frame.iterrows():
            run_id = str(row["run_id"])
            stored = str(row.get("abs_path", ""))
            rowed.add((run_id, Path(stored).name))
        return rowed

    def _perform_sweep(
        self,
        report: "SweepReport",
        is_deletable: Callable[["SweepClass"], bool],
    ) -> "SweepReport":
        """Drop the rows, then remove the directories, for the authorized set.

        Rows before files, the rule ``IndexCSV.drop_entries`` documents and
        ``delete_set`` already follows: a crash between the two leaves rows
        naming files that are gone, which :meth:`reindex` repairs, rather than
        files nothing names, which nothing finds.
        """
        import shutil
        from dataclasses import replace

        from .pipeline.dataset_indexes import reconcilable_index

        candidates = [e for e in report.entries if is_deletable(e.verdict)]
        by_root: dict[str, list[tuple[str, str]]] = {}
        for entry in candidates:
            by_root.setdefault(entry.root_key, []).append((entry.run_id, entry.entry))

        rows_dropped = 0
        for root_key, pairs in sorted(by_root.items()):
            factory = reconcilable_index(root_key)
            index_path = self.get_root(root_key) / "index.csv"
            if factory is None or not index_path.exists():
                continue
            index = factory(index_path)
            for run_id, entry in pairs:
                # A working directory is named by the composite entry *key*,
                # ``make_entry_key(group, sequence)``, while ``drop_entries``
                # matches a ``(group, sequence)`` pair. Passing the key as a bare
                # sequence matched nothing whenever the group was non-empty, so
                # the sweep removed the directory and left its row behind --
                # exactly the state this method's "rows before files" ordering
                # exists to avoid. It agreed only for an empty group, where the
                # key *is* the sequence, which is every dataset the control plane
                # creates and so every dataset the tests covered.
                dropped = index.drop_entries(
                    scope=Scope(entries=[parse_entry_key(entry)]), run_id=run_id
                )
                rows_dropped += len(dropped)

        removed: list[Path] = []
        reclaimed = 0
        for entry in candidates:
            try:
                shutil.rmtree(entry.path)
            except OSError as exc:
                print(f"[sweep] could not remove {entry.path}: {exc}", file=sys.stderr)
                continue
            removed.append(entry.path)
            reclaimed += entry.bytes_on_disk

        return replace(
            report,
            removed=removed,
            rows_dropped=rows_dropped,
            bytes_reclaimed=reclaimed,
        )

    def sequence_uniformity(
        self,
        group: str,
        sequence: str,
        *,
        order_by_name: Mapping[str, int] | None = None,
        index_filename: str = "index.csv",
    ) -> dict[str, UniformityVerdict]:
        """Which of a sequence's cameras a reader would refuse, under a proposed order.

        Item 6.5's precheck. *order_by_name* is the arrangement to test, mapping
        a clip's basename to its position exactly as :class:`MediaIndexScope`
        carries it; ``None`` tests the order the index already holds. Run it
        against the arrangement a caller is *about* to commit, because a reorder
        can move a marginal sequence between readable and unreadable with no
        artifact deleted and nothing else to notice.

        Returns one entry per camera **that has something to report** -- a
        mismatch a reader would raise on, or clips whose stored facts could not
        be rebuilt. An empty mapping means every camera agrees and every clip
        contributed a measurement, which are two different facts a caller should
        not have to separate itself (see :class:`UniformityVerdict`).

        A read, never a probe: the facts come out of the media index. Reported
        against the originals, which is what a rearrangement moves; a sequence a
        reader would open through analysis transcodes has those derived per
        video, so their uniformity follows the recipe rather than the order.
        """
        rows = [
            row
            for row in self.read_media_index(index_filename)
            if str(row.get("group", "")) == group
            and str(row.get("sequence", "")) == sequence
        ]
        by_camera: dict[str, list[Mapping[str, object]]] = {}
        for row in rows:
            by_camera.setdefault(str(row.get("camera", "") or ""), []).append(row)

        reported: dict[str, UniformityVerdict] = {}
        for camera, camera_rows in sorted(by_camera.items()):
            verdict = camera_uniformity(camera_rows, order_by_name=order_by_name)
            if verdict.mismatch is not None or not verdict.established:
                reported[camera] = verdict
        return reported

    def reprobe_media(
        self, *, apply: bool, skip_unreadable: bool = False
    ) -> ReprobeReport:
        """Re-probe the media this dataset's index already lists, in place.

        The counterpart to :meth:`index_media` for a media index that already
        exists and is authoritative: every file it lists is measured again and
        the fresh measurement written back into that row, minting the per-file
        identity columns and migrating an index written before a column existed
        to the current schema. The index owns group, sequence, order and paths;
        none of them is re-derived, and no row is added or removed.

        Dry-run unless *apply*; *skip_unreadable* leaves a row whose media is
        missing or unprobeable verbatim instead of aborting.

        Resolves the roots and hands plain paths to
        :func:`~mosaic.core.media.reprobe.reprobe_media`, which knows nothing
        about datasets, so root resolution has exactly one home.

        **Re-projects ``media_raw/sequences.csv`` after an applied run**, because
        ``video_uuid`` is a term of the media composition and this is the one
        command that rewrites it. Left out, the single pass whose purpose is
        correcting identity would leave every composition it invalidated on disk
        still claiming to be current -- and item 6.2 compares exactly those
        values, so the staleness would be invisible rather than loud.

        Sequential, never nested: the index is written and released inside
        :func:`~mosaic.core.media.reprobe.reprobe_media`, and the projection
        follows. Writing the projection first would record a composition for an
        index state that had not committed.

        This does not make the audit path agree with the write path on caching,
        and must not be read as a step toward that. ``drift``'s module docstring
        keeps the three apart deliberately -- *scan has no baseline, write has one
        and uses it, audit re-probes unconditionally* -- and the audit's no-cache
        rule is the only thing that catches a replacement preserving size and
        mtime. What is added here is the projection refresh the audit owed, not a
        shared cache.
        """
        index_filename = "index.csv"
        try:
            media_raw_root = self.get_root(self.resolve_media_root())
        except KeyError as error:
            message = f"the dataset manifest has no usable media root: {error}"
            raise ReprobeAbort(message) from error
        index_path = media_raw_root / index_filename
        # Derivatives get their own index only when the `media` root is a
        # distinct place from the originals root; a legacy media-only dataset
        # has one index and no derivative pass.
        # Compared resolved: get_root returns an absolute path but does not
        # normalize it, so a symlinked or ".."-containing root spelling would
        # make one file look like two indexes and be backed up and rewritten
        # twice in a single run.
        derivative_index_path: Path | None = None
        if self.has_root("media"):
            candidate = self.get_root("media") / index_filename
            if candidate.resolve() != index_path.resolve():
                derivative_index_path = candidate
        report = _reprobe_media(
            index_path,
            derivative_index_path=derivative_index_path,
            media_raw_root=media_raw_root,
            base_directory=_dataset_base_dir(self),
            apply=apply,
            skip_unreadable=skip_unreadable,
        )
        if report.applied:
            _ = self.rebuild_sequence_index("media_raw")
        return report

    def prune_media(
        self,
        *,
        apply: bool,
        min_age_hours: float = 24.0,
        relink: bool = False,
        include_stray: bool = False,
    ) -> PruneReport:
        """Delete the transcode derivatives no forward link reaches.

        A retuned recipe writes a new derivative and overwrites the link cell,
        leaving the previous file and its row behind with nothing addressing
        them; this is what removes them. Deleting one costs a re-encode that was
        going to happen anyway -- the op's reuse gate needs the link as well as
        the file -- so *relink* is where the value is: it writes the cell for an
        unreferenced file a current recipe would reproduce, turning the next
        run's encode into a skip, and clears a cell whose file is gone.

        Dry-run unless *apply*. *min_age_hours* holds back anything modified
        inside the window, so a prune cannot race a running encode.

        Four gates decline before anything is read, each returning a report with
        ``considered=False`` rather than raising -- "would prune 0" on a dataset
        that can never hold a derivative reads as an invitation to pass
        ``--apply``, which is the wrong thing to tell someone.

        Root resolution lives here and the decisions live in
        :mod:`~mosaic.core.media.prune`, which knows nothing about datasets.
        """
        from mosaic_media import CHROME_149
        from mosaic_media.transcode import ANALYSIS_ENCODING, PLAYBACK_ENCODING

        from mosaic.core.media.prune import TARGETS
        from mosaic.core.pipeline.transcode import (
            TRANSCODE_KIND_DIRECTORY,
            TranscodeParams,
            transcode_recipe_hash,
        )
        from mosaic.media_probe_config import media_thresholds

        if not self.has_root("media"):
            return declined_report("no-media-root")
        # `resolve_media_root` answers "media_raw" the moment that root is set,
        # so both of the next two are needed: the first rejects a dataset that
        # never had one, the second a manifest that points both names at one
        # place. Compared resolved, as reprobe does -- an unnormalized ".." or a
        # symlink would make one file look like two indexes, and this function
        # writes two whole-file projections, originals then derivatives. Against
        # one path the second erases the first, entire, with no error. (It is no
        # longer a lock problem: `index_lock` holds a sidecar, so the two writes
        # under one re-entrant lock would be safe. It is that neither projection
        # describes the other's rows.)
        if self.resolve_media_root() != "media_raw":
            return declined_report("single-root")
        media_root = self.get_root("media")
        media_raw_root = self.get_root("media_raw")
        if media_root.resolve() == media_raw_root.resolve():
            return declined_report("one-index")
        transcode_root = (media_root / TRANSCODE_KIND_DIRECTORY).resolve()
        # Roots are free-form strings and `frames` defaults to `media/frames`, so
        # a manifest may legally nest one inside another. Nested under the kind
        # directory, originals or extracted frames would be walked as derivative
        # candidates.
        nested = [media_raw_root.resolve()]
        if self.has_root("frames"):
            nested.append(self.get_root("frames").resolve())
        if any(
            root == transcode_root or transcode_root in root.parents for root in nested
        ):
            return declined_report("nested-root")

        live_recipes = {
            target: transcode_recipe_hash(
                # These params state the recipe and nothing about coverage.
                # A run's entries are its scope, and `allow_hardware` is
                # HASH_EXCLUDE. What this builds is therefore the recipe every
                # current run of this target names its output after.
                TranscodeParams(target=target),
                ANALYSIS_ENCODING if target == "analysis" else PLAYBACK_ENCODING,
                CHROME_149,
                media_thresholds(),
            )
            for target in TARGETS
        }
        return _prune_media(
            media_raw_root / "index.csv",
            derivative_index_path=media_root / "index.csv",
            media_root=media_root,
            transcode_root=transcode_root,
            base_directory=_dataset_base_dir(self),
            live_recipes=live_recipes,
            apply=apply,
            min_age_hours=min_age_hours,
            relink=relink,
            include_stray=include_stray,
        )

    def prune_joined_exports(
        self,
        *,
        apply: bool,
        min_age_hours: float = 24.0,
        include_stray: bool = False,
        now: "datetime.datetime | None" = None,
    ) -> "JoinedPruneReport":
        """Delete the joined exports an earlier version of ``export-joined`` made.

        A tracker reads a join only under a recipe the op writes today, so when
        the op's version moves, every join on disk is read by nothing and
        re-joining writes the current one beside it. This removes the old ones.
        A join of a clip set no entry resolves to, and a second current join of
        one clip set, are listed and kept. See
        :mod:`~mosaic.core.media.prune_joined`.

        Dry-run unless *apply*. *min_age_hours* holds back anything modified
        inside the window. *include_stray* also deletes the op's leftover working
        files once they are past it.

        Declines, with ``considered=False``, on a dataset with no media root and
        on one where another root resolves inside the joined kind directory.
        Unlike :meth:`prune_media` it runs on a single-root dataset, where
        ``media`` holds the originals and joins can still be written.
        """
        from .media.prune_joined import declined_joined_report, prune_joined
        from .pipeline.joined_export import (
            JOINED_KIND_DIRECTORY,
            current_joined_recipes,
        )

        if not self.has_root("media"):
            return declined_joined_report("no-media-root")
        joined_root = (self.get_root("media") / JOINED_KIND_DIRECTORY).resolve()
        # Roots are free-form strings, so a manifest may legally nest one inside
        # another. Nested under the kind directory, its files would be walked as
        # candidates for deletion.
        others = [self.get_root(key).resolve() for key in self.roots if key != "media"]
        if any(root == joined_root or joined_root in root.parents for root in others):
            return declined_joined_report("nested-root")

        clip_sets, unresolved = self._joined_clip_sets()
        return prune_joined(
            joined_root,
            clip_sets=clip_sets,
            current_recipes=current_joined_recipes(),
            apply=apply,
            min_age_hours=min_age_hours,
            include_stray=include_stray,
            unresolved=unresolved,
            now=now,
        )

    def _joined_clip_sets(self) -> tuple[dict[str, str], list[str]]:
        """Every clip set a join could hold today, mapped to its entry's label.

        Each single-camera entry's clips are routed as ``export-joined`` routes
        them through :meth:`resolve_media`. The digests then match the names that
        the op writes. One :meth:`resolve_media_scope` read routes them all, and
        places a divided recording's sequences once. A multi-camera entry is
        skipped, since the op cannot join one. An entry that fails to resolve
        is returned in the second list with its error. Its joins then read as
        unsourced and are kept. A failure here can only prevent a deletion.
        """
        from .pipeline.joined_export import joined_source_uid

        try:
            df = self._load_media_index()
        except FileNotFoundError:
            return {}, []
        single_camera = sorted(
            (str(group), str(sequence))
            for (group, sequence), rows in df.groupby(["group", "sequence"])
            if len({camera for camera in rows["camera"] if camera}) <= 1
        )
        failed: dict[Entry, MediaProbeError] = {}
        try:
            scope = self.resolve_media_scope(single_camera, errors=failed)
        except FileNotFoundError:
            return {}, []
        clip_sets: dict[str, str] = {}
        for media in scope:
            if len(media.resolved.facts) < 2:
                continue
            source_uid = joined_source_uid(list(media.resolved.facts))
            if source_uid:
                clip_sets[source_uid] = f"{media.group}/{media.sequence}"
        unresolved = [
            f"{group}/{sequence}: {failed[group, sequence]}"
            for group, sequence in sorted(failed)
        ]
        return clip_sets, unresolved

    # Identity cells: what a row says it *is*, as opposed to what was measured
    # about the file. A scan re-measures freely and must not re-decide these.
    _IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
        "group",
        "sequence",
        "group_safe",
        "sequence_safe",
        "camera",
        "sync_uuid",
        "video_order",
        "assignment_source",
    )

    def _keep_assigned_identity(
        self,
        fresh: list[dict[str, object]],
        committed: Sequence[Mapping[str, object]],
    ) -> int:
        """Restore assigned identity onto freshly scanned rows. In place.

        A scan derives ``(group, sequence)`` from a filename stem, a directory
        name or a track keymap -- all guesses, and all recorded as such in
        ``assignment_source``. ``"assigned"`` means a caller said so through a
        :class:`MediaIndexScope`, which is the one cycle-free source of identity
        in the index.

        **A guess must not overwrite a fact.** Without this, declaring
        ``media_raw`` as a scan source would silently repartition every project
        the control plane manages and move every media composition digest with
        it. The measured cells -- size, mtime, probe, facts -- are still taken
        from this scan, because those genuinely are what the scan just observed.

        Rows are matched by key: the resolved path and a division's chunks.

        Returns:
            How many rows kept an assigned identity.
        """
        assigned: dict[MediaRowKey, Mapping[str, object]] = {}
        for row in committed:
            if read_link_cell(row, "assignment_source") != "assigned":
                continue
            key = self._stored_row_key(row)
            if key is not None:
                assigned[key] = row
        if not assigned:
            return 0
        kept = 0
        for row in fresh:
            key = self._stored_row_key(row)
            prior = None if key is None else assigned.get(key)
            if prior is None:
                continue
            for column in self._IDENTITY_COLUMNS:
                if column in prior:
                    row[column] = prior[column]
            kept += 1
        return kept

    def _keep_assigned_divisions(
        self,
        fresh: Sequence[dict[str, object]],
        committed: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Keep the division rows a caller assigned to a store a scan indexes whole.

        A division scope's rows are an assigned identity, as the cells
        :meth:`_keep_assigned_identity` restores are. A scan whose source does
        not divide finds the store and proposes its undivided row, and the
        store's committed division rows are kept in its place.
        """
        assigned: dict[Path, list[dict[str, object]]] = {}
        for row in committed:
            key = self._stored_row_key(row)
            if (
                key is None
                or key.window is None
                or read_link_cell(row, "assignment_source") != "assigned"
            ):
                continue
            assigned.setdefault(Path(key.abs_path), []).append(dict(row))
        if not assigned:
            return list(fresh)
        kept: list[dict[str, object]] = []
        for row in fresh:
            key = self._stored_row_key(row)
            divisions = (
                None
                if key is None or key.window is not None
                else assigned.pop(Path(key.abs_path), None)
            )
            kept.extend([row] if divisions is None else divisions)
        return kept

    def set_media_calibration(
        self,
        cm_per_pixel: float | None,
        *,
        recording: str | None = None,
        camera: str | None = None,
        group: str | None = None,
        sequence: str | None = None,
        index_filename: str = "index.csv",
    ) -> int:
        """Record how many centimetres one pixel spans, per media row.

        The scale a physical unit is computed from. It lives here, on the video,
        because that is what it is a property of: the camera and the rig it was
        mounted on. A dataset that mixes rigs, zoom levels or cameras carries a
        different value per row and nothing has to be told twice; a tracks table
        derived from the video inherits the answer rather than restating it, and
        a reconversion does not lose it.

        Not applied to any table. Positions in ``tracks/`` are pixels, and
        converting them is a feature's job, where the choice is recorded in a
        run identifier. This records the input that feature reads.

        Args:
            cm_per_pixel: Centimetres per pixel, or ``None`` to clear the value
                back to uncalibrated. Must be positive when given.
            recording: Restrict to the member rows of this recording, named as
                the dataset summary names it: a continuous group, or an
                undivided entry's token (``group:sequence``, or the sequence in
                the empty group). ``None`` means every recording.
            camera: Restrict to this camera. ``None`` means every camera.
            group: Restrict to this group. ``None`` means every group.
            sequence: Restrict to this sequence. ``None`` means every sequence.
            index_filename: Output filename within the media root.

        Returns:
            How many rows were updated.

        Raises:
            ValueError: If *cm_per_pixel* is not positive, or *recording* is
                given with *group* or *sequence*.
        """
        if cm_per_pixel is not None and not cm_per_pixel > 0.0:
            raise ValueError(
                f"cm_per_pixel={cm_per_pixel} is not a usable scale. Pass None to "
                "clear it back to uncalibrated."
            )
        if recording is not None and (group is not None or sequence is not None):
            message = (
                f"recording={recording!r} names the rows to calibrate, and so do "
                "group= and sequence=. Pass the recording alone."
            )
            raise ValueError(message)
        cell = "" if cm_per_pixel is None else repr(float(cm_per_pixel))

        out_csv = self.get_root(self.resolve_media_root()) / index_filename
        updated = 0
        with index_lock(out_csv):
            rows = [dict(row) for row in _read_media_index(out_csv)]
            for row in rows:
                row_group = read_link_cell(row, "group")
                row_sequence = read_link_cell(row, "sequence")
                if group is not None and row_group != group:
                    continue
                if sequence is not None and row_sequence != sequence:
                    continue
                if camera is not None and read_link_cell(row, "camera") != camera:
                    continue
                if recording is not None and recording != recording_name(
                    self.recording_of(row_group, row_sequence)
                ):
                    continue
                row["cm_per_pixel"] = cell
                updated += 1
            write_media_index_rows(out_csv, frame_from_rows(rows))
        return updated

    def media_calibration(
        self, group: str, sequence: str, camera: str | None = None
    ) -> float | None:
        """The ``cm_per_pixel`` recorded for *(group, sequence)*, or ``None``.

        ``None`` means uncalibrated -- no row says -- which a caller needing a
        physical unit must treat as a refusal rather than as ``1.0``.

        Args:
            group: The sequence's group.
            sequence: The sequence.
            camera: The camera whose rows to read. ``None`` reads every camera,
                which must then record one scale.

        Raises:
            ValueError: If one camera's rows record two scales, or *camera* is
                ``None`` and the sequence's cameras record different scales. One
                camera of one sequence has one scale.
        """
        by_camera: dict[str, set[str]] = {}
        for row in self.read_media_index():
            row_camera = read_link_cell(row, "camera")
            if (
                read_link_cell(row, "group") != group
                or read_link_cell(row, "sequence") != sequence
                or (camera is not None and row_camera != camera)
            ):
                continue
            values = by_camera.setdefault(row_camera, set())
            if text := read_link_cell(row, "cm_per_pixel"):
                values.add(text)
        for found, values in sorted(by_camera.items()):
            if len(values) > 1:
                shown = f"camera {found!r} of sequence" if found else "Sequence"
                message = (
                    f"{shown} ({group!r}, {sequence!r}) records {len(values)} "
                    f"different cm_per_pixel values ({sorted(values)}). One "
                    "camera of one sequence has one scale."
                )
                raise ValueError(message)
        scales = {next(iter(values), "") for values in by_camera.values()}
        if len(scales) > 1:
            message = (
                f"the cameras {sorted(by_camera)} of sequence ({group!r}, "
                f"{sequence!r}) record different cm_per_pixel values. Pass "
                "camera= to read one of them."
            )
            raise ValueError(message)
        scale = next(iter(scales), "")
        return float(scale) if scale else None

    def _keep_calibration(
        self,
        fresh: list[dict[str, object]],
        committed: Sequence[Mapping[str, object]],
    ) -> int:
        """Carry a recorded ``cm_per_pixel`` onto freshly scanned rows. In place.

        Separate from :meth:`_keep_assigned_identity`, and keyed on the value
        being present rather than on ``assignment_source``. The two answer
        different questions: that one protects a *fact* about which sequence a
        file belongs to from being overwritten by a *guess*, and only an
        assigned identity is a fact. A calibration has no guessed counterpart --
        no probe measures it, so a scan never proposes one -- and it is equally
        real whether the row's identity came from a caller or from a filename.

        A fresh row takes the calibration of the committed row with its key: its
        resolved path and its division's chunks. A fresh row without such a row
        takes the calibration of its file when every committed row of the file
        records one value, so a store divided after it was calibrated keeps its
        scale on each division, and a division row's scale reaches its store's
        undivided row. Failing that, a fresh row of a continuous group takes the
        calibration of its camera when every committed row of that recording and
        camera records one value. A recording that grows by a store or a
        sequence then keeps each camera's scale.

        Returns:
            How many rows kept a calibration.
        """
        by_key: dict[MediaRowKey, str] = {}
        by_file: dict[str, set[str]] = {}
        by_camera: dict[tuple[str, str], set[str]] = {}
        for row in committed:
            key = self._stored_row_key(row)
            if key is None:
                continue
            value = read_link_cell(row, "cm_per_pixel")
            if value:
                by_key[key] = value
            by_file.setdefault(key.abs_path, set()).add(value)
            recording_camera = self._continuous_camera(row)
            if recording_camera is not None:
                by_camera.setdefault(recording_camera, set()).add(value)
        if not by_key:
            return 0
        kept = 0
        for row in fresh:
            key = self._stored_row_key(row)
            if key is None:
                continue
            prior = by_key.get(key) or _one_value(by_file.get(key.abs_path))
            recording_camera = self._continuous_camera(row)
            if not prior and recording_camera is not None:
                prior = _one_value(by_camera.get(recording_camera))
            if not prior:
                continue
            row["cm_per_pixel"] = prior
            kept += 1
        return kept

    def _continuous_camera(self, row: Mapping[str, object]) -> tuple[str, str] | None:
        """Return *row*'s divided recording and camera, ``None`` outside one."""
        group = read_link_cell(row, "group")
        if self.recording_kind(group) != "divided":
            return None
        return (group, read_link_cell(row, "camera"))

    def _stored_row_key(self, row: Mapping[str, object]) -> MediaRowKey | None:
        """Return *row*'s key with its stored path fully resolved, ``None`` without one."""
        key = media_row_key(row)
        if not key.abs_path:
            return None
        resolved = self.resolve_path(key.abs_path).resolve()
        return dataclasses.replace(key, abs_path=str(resolved))

    def _row_claimed(self, row: Mapping[str, object], claim: ScanClaim) -> bool:
        """True if *row*'s ``abs_path`` falls inside *claim*.

        Resolve the stored path first (it may be root-relative), then ask the
        claim -- the resolver decoupling that keeps the check correct whether
        ``abs_path`` is stored relative or absolute.

        A row with no ``abs_path`` is claimed by nobody, so it survives every
        scan. That is the conservative answer: a row a scan cannot locate is not
        a row a scan may delete.
        """
        abs_cell = str(row.get("abs_path", "") or "").strip()
        if not abs_cell:
            return False
        return claim.claims(self.resolve_path(abs_cell).resolve())

    def walked_claim(self, rows: Iterable[Mapping[str, object]]) -> ScanClaim:
        """The claim over the files a scan actually walked, by stored path.

        A source's declared claim covers *where it looked*; this covers *what it
        found*, and the two are not always the same set. A symlink inside a
        scanned directory is the case that separates them: the walk finds
        ``<dataset>/media_raw/<entry>/clip.mp4``, but the row records the
        symlink's target, because :meth:`_relative_to_root` resolves before
        testing containment and a target outside the dataset is stored absolute.
        The directory claim then does not cover the row the walk just produced.

        Left at that, a scan **disowns rows it created**: they are preserved as
        somebody else's, and the same files are appended again on the next pass.
        The index grows by a full copy per scan, silently, and the duplicates
        reach the tracker -- a joined TREx conversion is handed every clip twice,
        which doubles a session's frame count and repeats its footage mid-timeline
        with nothing raising.

        Unioning this in fixes that without touching the preserve rule that
        motivates it. A row an ordinary scan walked was already claimed by the
        directory, so nothing moves for it; a row **no** scan walked -- one an
        assignment scope wrote, or one referencing another dataset's video --
        appears in no walk and is still preserved. Deletion is the one case this
        cannot answer: a file removed from a symlinked source is neither walked
        nor under the directory claim, so its row survives until
        ``prune_unsourced``.

        A path that cannot be resolved is skipped rather than raised on.
        ``Path.resolve()`` is non-strict about a *missing* target and about an
        untraversable parent, but it rejects an embedded NUL with ``ValueError``
        on every supported interpreter, and on 3.12 it additionally raises
        ``RuntimeError`` on a symlink **loop** -- 3.13 returns such a path
        unchanged. So building this claim could abort a whole scan over one bad
        cell. Skipping is not a shrug: a path
        this process cannot resolve is one no claim can cover, which is exactly
        the conservative answer :meth:`_row_claimed` already gives an empty
        cell. The row then falls through to being preserved, which is the safe
        direction -- the failure mode this whole method exists to remove is a
        duplicate, and the one it must never introduce is a deletion.

        Resolved here rather than left to :meth:`ScanClaim.over_files`, which
        resolves what it is given: the guard has to wrap the call that actually
        raises. It also makes this symmetric with :meth:`_row_claimed`, whose
        ``resolve_path(cell).resolve()`` is what these paths are compared against.
        """
        claimed: list[Path] = []
        for row in rows:
            cell = str(row.get("abs_path", "") or "").strip()
            if not cell:
                continue
            try:
                claimed.append(self.resolve_path(cell).resolve())
            except (OSError, RuntimeError, ValueError):
                continue
        return ScanClaim.over_files(claimed)

    def _row_under_dirs(self, row: Mapping[str, object], dirs: list[Path]) -> bool:
        """True if *row*'s resolved ``abs_path`` lives under any of *dirs*.

        The directory case of :meth:`_row_claimed`, kept for the assignment-driven
        writers, which are always given scope directories rather than a claim.
        """
        return self._row_claimed(row, ScanClaim.over_directories(dirs))

    def _resolve_declared_path(self, path: str) -> Path:
        """A declared source path as an absolute one, creating and checking nothing."""
        declared = user_path(path)
        return declared if declared.is_absolute() else self.base_dir / declared

    def resolve_source_path(self, source: ScanSource) -> Path:
        """Where a declared source's *path* actually is.

        Relative resolves against the dataset, absolute is taken as written.
        Nothing is created and nothing is checked: a source names storage that
        may be on an unmounted share, and finding out is a scan's job, not a
        load's.

        Note the deliberate asymmetry with :meth:`get_root`. A root is validated
        to lie inside the dataset, because that root holds an ``index.csv`` that
        has to travel with it. A source is expected to lie outside, because that
        is the whole reason it exists.
        """
        return self._resolve_declared_path(source.path)

    def source_claim(self, source: ScanSource) -> ScanClaim:
        """What *source* is responsible for, and so what a scan of it replaces."""
        base = self.resolve_source_path(source)
        if source.mode == "files":
            return ScanClaim.over_files(base / entry for entry in source.files)
        return ScanClaim.over_directories([base])

    def _carry_forward_derivative_links(
        self, rows: list[dict[str, object]], index_path: Path
    ) -> None:
        """Preserve per-target derivative links across a media reindex.

        A reindex freshly measures every column -- which would reset the
        transcode-written ``analysis_derivative_path`` /
        ``playback_derivative_path`` links to empty. Those links record a
        transcode decision, not a measurement, so carry them forward: for each
        freshly probed row matching a row in the prior index (*index_path*, read
        before it is overwritten), copy the two link cells over. A link whose
        derivative file no longer exists is dropped rather than carried as a
        dangling reference.

        A row is matched by ``video_uuid``, on both sides. There is no path
        fallback: the two-leaf key that stood in for one while a corpus was
        unminted is not unique -- ``rec1/cam0/video.mp4`` and
        ``rec2/cam0/video.mp4`` answer to the same key -- so carrying a link
        through it could attach one recording's derivative to another's row,
        routing analysis reads at a different video with no error. Re-probing an
        index is what makes every row carry an identity; a row carrying none
        carries no link either.
        """
        if not index_path.exists() or not self.has_root("media"):
            return
        prior = _read_media_index(index_path)
        if not prior:
            return
        link_columns = ["analysis_derivative_path", "playback_derivative_path"]
        media_root = self.get_root("media")
        prior_links_by_uuid: dict[str, dict[str, str]] = {}
        for prior_row in prior:
            uuid = media_row_uuid(prior_row)
            if not uuid:
                continue
            carried: dict[str, str] = {}
            for column in link_columns:
                link = read_link_cell(prior_row, column)
                if link and (media_root / link).exists():
                    carried[column] = link
            if carried:
                prior_links_by_uuid[uuid] = carried
        if not prior_links_by_uuid:
            return
        for row in rows:
            uuid = media_row_uuid(row)
            if not uuid:
                continue
            links = prior_links_by_uuid.get(uuid)
            if not links:
                continue
            for column, link in links.items():
                row[column] = link

    def _match_media_rows(
        self,
        df: "pd.DataFrame",
        group: str,
        sequence: str,
        camera: str | None = None,
    ) -> "pd.DataFrame | None":
        """Return the media-index rows for (group, sequence), video_order-sorted.

        Matches on direct ``(group, sequence)``, then safe-name, then a
        case-insensitive comparison against each row's own ``sequence`` cell,
        which also accepts the request carrying a media extension the entry's
        own name lacks (``VIDEO_EXTENSIONS``, and no other suffix). That last
        tier reads identity cells like the two above it and never a row's
        filename, so an entry is matched whole -- every file of a multi-file
        recording -- and never through a file that happens to be named for a
        different sequence. A non-empty *group* narrows it to that group,
        case-insensitively; an empty *group* names no namespace and admits any
        group but a continuous one.
        With more than one entry still standing after that narrowing, the tier
        refuses rather than choosing. An empty *sequence* matches nothing there.

        When *camera* is given, the matched rows are further filtered to that
        camera (``""`` selects the blank-camera rows), and an empty result after
        that filter returns ``None``. Returns ``None`` when nothing matches.

        Raises:
            AmbiguousMediaMatchError: If the sequence-cell tier leaves rows of
                more than one ``(group, sequence)`` standing.
        """
        # Untyped so the pandas ``df[mask]`` (Series | DataFrame in the stubs)
        # widens by inference rather than tripping a declared-type mismatch, as
        # the rest of this module's index masking does.
        matched = None
        if "group" in df.columns and "sequence" in df.columns:
            df_match = df[
                (df["group"].fillna("") == str(group))
                & (df["sequence"].fillna("") == str(sequence))
            ]
            if not df_match.empty:
                matched = df_match

        if matched is None and {"group_safe", "sequence_safe"}.issubset(df.columns):
            safe_group = to_safe_name(group) if group else ""
            safe_sequence = to_safe_name(sequence)
            df_match = df[
                (df["group_safe"].fillna("") == safe_group)
                & (df["sequence_safe"].fillna("") == safe_sequence)
            ]
            if not df_match.empty:
                matched = df_match

        if matched is None:
            # Compared against the row's own ``sequence`` cell, never its
            # ``name``: a filename is not an identity -- entry "session1" may
            # hold "trial.mp4" -- so matching the file would answer for an
            # unrelated sequence, and would answer with the single chunk whose
            # name fits rather than the whole of a multi-file recording. What is
            # bridged here is the request differing from an entry's own name by
            # case or by a media extension, which is all the exact and
            # safe-name tiers above miss. Every comparison is a string equality,
            # so a request holding regex metacharacters is matched literally
            # rather than compiled as a pattern.
            wanted = sequence.casefold()
            if not wanted:
                # An empty sequence names no entry; left to compare it would
                # answer for every row that shares its emptiness.
                return None
            # Only a media extension is stripped, never whatever follows the
            # last dot: entry names carry dots routinely ("cam1.left",
            # "session.v2"), so stripping any suffix would let "trial.1" answer
            # with entry "trial"'s media -- the same wrong-entry fault one suffix
            # further in. VIDEO_EXTENSIONS is the set the probe itself reads, so
            # a raw elementary stream is bridged like any container; ".264" and
            # ".265" belong to it, and a request ending in one is read as such a
            # stream rather than as a numbered name. The suffix is sliced off
            # rather than taken as Path.stem, which would also discard leading
            # directories and let "sub/trial.mp4" answer with entry "trial".
            suffix = Path(wanted).suffix
            bare = wanted[: -len(suffix)] if suffix in VIDEO_EXTENSIONS else wanted
            row_sequences = [str(value).casefold() for value in df["sequence"]]
            candidates = df[
                pd.Series(
                    [value in (wanted, bare) for value in row_sequences],
                    index=df.index,
                )
            ]
            # A request naming a group keeps this tier inside it: what the
            # tier bridges are differences within an entry, never across two. An
            # empty group asks for no particular namespace and matches any but a
            # continuous group, whose sequences are named with their recording.
            # The comparison is case-insensitive like the one above. The tier
            # bridges case on both axes rather than only one.
            if not group:
                candidates = candidates[
                    pd.Series(
                        [
                            self.recording_kind(str(value)) == "undivided"
                            for value in candidates["group"].fillna("")
                        ],
                        index=candidates.index,
                    )
                ]
            else:
                wanted_group = str(group).casefold()
                candidates = candidates[
                    pd.Series(
                        [
                            str(value).casefold() == wanted_group
                            for value in candidates["group"]
                        ],
                        index=candidates.index,
                    )
                ]
            if candidates.empty:
                return None
            entries = {
                (str(row_group), str(row_sequence))
                for row_group, row_sequence in zip(
                    candidates["group"].fillna(""),
                    candidates["sequence"].fillna(""),
                )
            }
            if len(entries) > 1:
                spelled = ", ".join(
                    f"({entry_group!r}, {entry_sequence!r})"
                    for entry_group, entry_sequence in sorted(entries)
                )
                raise AmbiguousMediaMatchError(
                    f"({group!r}, {sequence!r}) has no media-index row of its "
                    f"own, and matches {len(entries)} entries {spelled}; give "
                    f"the pair a row of its own, or resolve one of those "
                    f"entries by its own (group, sequence)"
                )
            matched = candidates

        if camera is not None:
            matched = matched[matched["camera"].fillna("") == camera]
            if matched.empty:
                return None
        # Stable, so rows tied on video_order -- every row of an index that
        # predates the column, which reads back as all-zero -- keep the index's
        # order instead of pandas' default quicksort permuting them per run.
        return matched.sort_values("video_order", kind="stable")

    def _load_media_index(self, index_filename: str = "index.csv") -> "pd.DataFrame":
        """Read and normalize the originals media index for scoped resolution.

        Reads ``<resolve_media_root()>/<index_filename>`` and normalizes every
        non-numeric schema column (``MEDIA_INDEX_COLUMNS`` minus
        ``MEDIA_NUMERIC_COLUMNS``) so downstream masking and grouping are
        string-typed and NaN-free: each such column, when present, is filled to
        ``""`` and coerced to ``str``, and is created empty when absent (a
        legacy or hand-seeded CSV written before a column existed). Separately,
        ``video_order`` is coerced to an int column (created as ``0`` when
        absent). Shared by :meth:`match_media_rows` and
        :meth:`resolve_media_scope` so the read-and-normalize lives in one place.

        Raises:
            FileNotFoundError: If the index file does not exist.
        """
        media_key = self.resolve_media_root()
        idx_path = self.get_root(media_key) / index_filename
        if not idx_path.exists():
            raise FileNotFoundError(
                f"{media_key}/{index_filename} not found; run index_media() first."
            )
        # Read the non-numeric schema columns as text rather than coercing after
        # the fact. A sequence named "0066" is inferred as int64 by a bare
        # read_csv, and the .astype(str) below then yields "66" -- the padding is
        # already gone by the time it runs. That renames the entry for every
        # caller of this method (frame extraction, transcode, tracker scope), so
        # the run lands under a key that matches neither the media index it came
        # from nor the tracks index beside it. dtype keys naming a column the
        # file does not carry are ignored, so this stays schema-driven.
        df = pd.read_csv(
            idx_path,
            dtype={
                column: str
                for column in MEDIA_INDEX_COLUMNS
                if column not in MEDIA_NUMERIC_COLUMNS
            },
        )
        # Back-fill every non-numeric schema column by schema, not by a fixed
        # list: an index written before a column existed must read it back as an
        # empty string rather than KeyError or a float NaN, and a future column
        # addition should not need to touch this method. video_order is numeric
        # and coerced separately below.
        for column in MEDIA_INDEX_COLUMNS:
            if column in MEDIA_NUMERIC_COLUMNS:
                continue
            if column in df.columns:
                df[column] = df[column].fillna("").astype(str)
            else:
                df[column] = ""
        if "video_order" not in df.columns:
            df["video_order"] = 0
        else:
            df["video_order"] = df["video_order"].fillna(0).astype(int)
        return df

    def match_media_rows(
        self,
        group: str,
        sequence: str,
        camera: str | None = None,
        index_filename: str = "index.csv",
    ) -> "pd.DataFrame":
        """Return the originals-index rows for (group, sequence), video_order-sorted.

        Reads the originals index (:meth:`resolve_media_root`) and matches by
        direct ``(group, sequence)``, then safe-name, then the row's own
        ``sequence`` cell case-insensitively -- **without** applying
        transcode-verdict routing (unlike :meth:`resolve_media`). *camera*, when
        given, further restricts the match to one camera of a multi-camera
        recording. The transcode job needs the originals, not their derivatives.

        A sequence of a continuous group matches its own rows. With none, the
        refusal says that its recording's media is indexed per sequence.

        Raises:
            FileNotFoundError: If the index is missing/empty or no row matches.
            AmbiguousMediaMatchError: If the last tier leaves rows of more than
                one ``(group, sequence)`` standing.
        """
        df = self._load_media_index(index_filename)
        return self._matched_media_rows(df, group, sequence, camera)

    def _matched_media_rows(
        self,
        df: "pd.DataFrame",
        group: str,
        sequence: str,
        camera: str | None,
    ) -> "pd.DataFrame":
        """Return :meth:`match_media_rows`'s rows from the loaded index *df*.

        Raises:
            FileNotFoundError: If *df* is empty or no row matches.
            AmbiguousMediaMatchError: As :meth:`_match_media_rows` raises it.
        """
        if df.empty:
            raise FileNotFoundError("Media index is empty.")
        matched = self._match_media_rows(df, group, sequence, camera)
        if matched is None:
            message = f"No media file found matching sequence '{sequence}'."
            if self.recording_kind(group) == "divided":
                message += (
                    f" ({group!r}, {sequence!r}) is a sequence of the divided "
                    f"recording {group!r}, whose media is indexed per sequence: "
                    "index this sequence's media under the group."
                )
            raise FileNotFoundError(message)
        return matched

    def media_rows_of(
        self, keys: Iterable[MediaRowKey], index_filename: str = "index.csv"
    ) -> dict[MediaRowKey, dict[str, object]]:
        """Return the originals-index row of each of *keys* that the index lists.

        The rows are :meth:`read_media_rows`'s. The dataset keeps its last read
        of each index, and reads the index again when the index's
        :class:`~mosaic.core.pipeline.index_stamp.IndexStamp`, the media root or
        the path map changed. Looking up the entries one at a time therefore
        reads the index once.

        Args:
            keys: The rows to look up, as :attr:`ResolvedMedia.row_keys` names
                them.
            index_filename: The originals index to read.

        Returns:
            The row of each listed key, as a ``{column: cell}`` mapping. A key
            that the index does not list is left out, and so is every key when
            the index does not exist.
        """
        wanted = set(keys)
        if not wanted:
            return {}
        index = self.get_root(self.resolve_media_root()) / index_filename
        # The stamp precedes the read. A write during the read then leaves the
        # stamp stale, and the next lookup reads the index again.
        stamp = IndexStamp.of(index)
        if not stamp.exists:
            return {}
        key = (stamp, self.get_root("media"), tuple(self._path_map))
        last = self._media_rows.get(index_filename)
        if last is not None and last[0] == key:
            return last[1].rows_of(wanted)
        try:
            rows = self.read_media_rows(index_filename)
        except FileNotFoundError:
            return {}
        self._media_rows[index_filename] = (key, rows)
        return rows.rows_of(wanted)

    def read_media_rows(self, index_filename: str = "index.csv") -> MediaRows:
        """Read the originals index, and find the row of each file that it lists.

        A row lists its file and its analysis derivative. A read opens the
        derivative of a row that requires an analysis transcode. A path is
        therefore found as :meth:`resolve_media_scope` returns it, however the
        caller names its entry. Several rows can list one file. The first row
        that records it as ``abs_path`` answers for it, and otherwise the first
        row linking it. Resolving each ``abs_path`` cell (:meth:`resolve_path`)
        costs a stat or a path resolution. A row is keyed by its resolved path
        and its division's chunks, so the divisions of one store are told apart.

        Args:
            index_filename: The originals index to read.

        Raises:
            FileNotFoundError: If the index does not exist.
            ValueError: If a row's chunk cells do not read
                (:func:`~mosaic.core.media.facts_columns.media_row_key`).
        """
        df = self._load_media_index(index_filename)
        media_root = self.get_root("media")
        own: dict[MediaRowKey, int] = {}
        derived: dict[MediaRowKey, int] = {}
        for position, (_, row) in enumerate(df.iterrows()):
            cells = row_mapping(row)
            _ = own.setdefault(self._resolved_row_key(cells), position)
            derivative = derivative_path_for_target(cells, "analysis", media_root)
            if derivative is not None:
                _ = derived.setdefault(MediaRowKey.of(str(derivative), None), position)
        return MediaRows(frame=df, positions={**derived, **own})

    def _resolved_row_key(self, row: Mapping[str, object]) -> MediaRowKey:
        """Return *row*'s key with its stored path resolved (:meth:`resolve_path`).

        Raises:
            ValueError: If the row's chunk cells do not read.
        """
        key = media_row_key(row)
        return dataclasses.replace(key, abs_path=str(self.resolve_path(key.abs_path)))

    def indexed_entries_of(
        self, keys: Iterable[MediaRowKey]
    ) -> dict[MediaRowKey, Entry]:
        """Return the indexed ``(group, sequence)`` of each of *keys* that a row lists.

        The pair is the entry's key in the media index, which a run's scope names.
        A media scope labels a nameless entry
        (:func:`~mosaic.core.helpers.is_nameless_entry`) by its first file's stem
        instead, and a scope does not resolve that label. Each key is looked up by
        :meth:`media_rows_of`, and a key that the index does not list is left out.
        """
        return {
            key: media_row_entry(row) for key, row in self.media_rows_of(keys).items()
        }

    def indexed_entry(self, key: MediaRowKey, named: Entry) -> Entry:
        """Return the indexed key of *key*'s entry, or *named* when unlisted.

        The key is :meth:`indexed_entries_of`'s. *named* is the entry as the caller
        names it.
        """
        return self.indexed_entries_of([key]).get(key, named)

    def _derivative_facts(
        self,
        group: str,
        sequence: str,
        derivative_path: Path,
        original_video_uuid: str,
        derivative_df: "pd.DataFrame | None",
    ) -> MediaFacts:
        """Look up an analysis derivative's stored facts in the ``media`` index.

        Two passes, exact file first. Pass 1 returns the row whose ``abs_path``
        resolves to *derivative_path* -- the unambiguous match, since that is the
        file the caller opens. Pass 2 is a fallback for a row that lacks a
        resolvable ``abs_path``: it matches on the source's ``video_uuid``,
        restricted to a row whose ``abs_path`` basename equals the requested
        derivative's, so a per-target sibling -- which shares its
        ``source_video_uuid`` -- can never cross into the wrong target's facts.
        An original carrying no uuid resolves through pass 1 or raises.

        **Nothing mosaic writes reaches pass 2.** A derivative row is minted
        only by ``_set_back_link``, whose ``abs_path`` and whose original's
        forward link are both derived from the transcode's ``output_path``, so
        the two reverse to one file and pass 1 always matches. The fallback
        covers an index edited outside mosaic, and is kept for that alone.

        Raises :class:`~mosaic_media.MediaProbeError` when the derivative file,
        its row, or its stored facts cannot be found.
        """
        if not derivative_path.exists():
            message = (
                f"entry {group}/{sequence} points at derivative "
                f"{derivative_path} which does not exist"
            )
            raise MediaProbeError(message)
        if derivative_df is None or derivative_df.empty:
            message = (
                f"entry {group}/{sequence} has a derivative but the media index "
                f"holds no derivative rows"
            )
            raise MediaProbeError(message)

        target_name = derivative_path.name

        # Pass 1: exact resolved-file match on abs_path.
        exact = self.derivative_row(derivative_path, derivative_df)
        if exact is not None:
            return _facts_or_stale_probe_error(exact, group, sequence)

        # Pass 2, uuid form: match the derivative row by the source's video_uuid,
        # keeping the basename guard so the analysis and playback siblings --
        # which share a source_video_uuid -- never cross into each other's facts.
        if original_video_uuid:
            for _, drow in derivative_df.iterrows():
                abs_cell = _media_cell(drow, "abs_path")
                if abs_cell and Path(abs_cell).name != target_name:
                    continue
                if _media_cell(drow, "source_video_uuid") == original_video_uuid:
                    return _facts_or_stale_probe_error(drow, group, sequence)

        message = (
            f"entry {group}/{sequence} derivative {derivative_path} has no "
            f"matching row with stored facts in the media index"
        )
        raise MediaProbeError(message)

    def derivative_row(
        self, derivative_path: Path, derivative_df: "pd.DataFrame | None"
    ) -> "pd.Series | None":
        """Return the ``media`` index row of the derivative at *derivative_path*.

        The row is the one whose ``abs_path`` resolves to *derivative_path*, the
        back-link that registration writes.

        Args:
            derivative_path: The derivative's file.
            derivative_df: The ``media`` index, as :meth:`media_routing_context`
                returns it.

        Returns:
            The row, or ``None`` when *derivative_df* is ``None`` or does not
            list that file.
        """
        if derivative_df is None:
            return None
        target = derivative_path.resolve()
        for _, drow in derivative_df.iterrows():
            abs_cell = _media_cell(drow, "abs_path")
            if abs_cell and self.resolve_path(abs_cell).resolve() == target:
                return drow
        return None

    def routes_derivatives(self) -> bool:
        """Return whether media reads route to derivatives under the ``media`` root.

        They do only beside a distinct ``media_raw`` root, which contains the
        originals. Without one, ``media/index.csv`` is the originals index, and a
        derivative's row written there is never routed to.
        """
        return self.resolve_media_root() == "media_raw"

    def media_routing_context(
        self, index_filename: str = "index.csv"
    ) -> tuple[bool, "pd.DataFrame | None"]:
        """Load the shared context for routing media-index rows by transcode verdict.

        Returns ``(route_derivatives, derivative_df)``:

        * ``route_derivatives`` is :meth:`routes_derivatives`.
        * ``derivative_df`` is the ``media`` index (one row per derivative) when
          it exists, else ``None``.

        Load once, then pass both to :meth:`route_media_row` for each row, so a
        batch of routings reads each index only once.
        """
        route_derivatives = self.routes_derivatives()
        derivative_df: pd.DataFrame | None = None
        if route_derivatives:
            derivative_idx = self.get_root("media") / index_filename
            if derivative_idx.exists():
                derivative_df = pd.read_csv(derivative_idx)
        return route_derivatives, derivative_df

    def route_media_row(
        self,
        group: str,
        sequence: str,
        row: "pd.Series",
        route_derivatives: bool,
        derivative_df: "pd.DataFrame | None",
    ) -> tuple[Path, MediaFacts]:
        """Route one media-index row to the file a per-frame read must open.

        mosaic's per-frame reads are analysis reads, so this routes on the
        ``analysis_transcode`` verdict and the ``analysis_derivative_path`` link:

        * a row marked ``analysis_transcode="required"`` resolves to its
          registered analysis derivative under the ``media`` root, carrying the
          derivative's stored facts;
        * a clean row resolves to the original file, carrying its stored facts.

        A ``required`` row is defective for analysis reads, so it must resolve to
        a clean derivative or fail. If it has no analysis derivative -- whether
        the link is missing or the dataset is legacy ``media``-only with no
        ``media_raw`` split (*route_derivatives* is ``False``) -- this raises
        :class:`~mosaic.core.media.facts_columns.AnalysisDerivativeMissingError`
        telling the caller to transcode first. A clean row whose stored
        measurement cannot be reconstructed raises its base class,
        :class:`~mosaic_media.MediaProbeError`, telling the caller to re-probe
        the media index instead. Transcoding and re-probing remedy distinct
        problems, and the two failures differ in type and in text. This method
        opens neither the defective original nor a file without a measurement
        to hand the reader.

        *route_derivatives* and *derivative_df* come from
        :meth:`media_routing_context`; load them once and reuse across a batch.
        """
        if _media_cell(row, "analysis_transcode") == "required":
            if route_derivatives:
                derivative_path = derivative_path_for_target(
                    row_mapping(row), "analysis", self.get_root("media")
                )
                if derivative_path is not None:
                    facts = self._derivative_facts(
                        group,
                        sequence,
                        derivative_path,
                        media_row_uuid(row_mapping(row)),
                        derivative_df,
                    )
                    return derivative_path, facts
            message = (
                f"entry {group}/{sequence} requires an analysis transcode but "
                f"has no analysis derivative; transcode it first"
            )
            raise AnalysisDerivativeMissingError(message)

        routed = self.resolve_path(row["abs_path"])
        facts = series_facts_or_none(row)
        if facts is None:
            message = (
                f"entry {group}/{sequence} file {routed} carries no reconstructable "
                f"measurement in the media index; run "
                f"'mosaic reprobe-media --apply' to re-probe it"
            )
            raise MediaProbeError(message)
        return routed, facts

    def resolve_media(
        self,
        group: str,
        sequence: str,
        camera: str | None = None,
        index_filename: str = "index.csv",
    ) -> ResolvedMedia:
        """Resolve media for (group, sequence), routing by transcode verdict.

        Reads the originals index (:meth:`resolve_media_root`), then routes each
        matched row: a row whose stored verdict marks
        ``analysis_transcode="required"`` resolves to its analysis derivative
        under the ``media`` root; a clean row resolves to the original file.
        Stored facts travel with each path (see :class:`ResolvedMedia`), so
        readers need not re-probe. Paths are ordered by ``video_order``; a
        single-file sequence yields one element.

        A multi-camera recording has more than one camera under one
        ``(group, sequence)``; concatenating them would fabricate a timeline, so
        *camera* must select one and a ``camera=None`` call over such a sequence
        raises rather than returning both. Single-camera and temporal-chunk
        sequences (every row ``camera=""``) resolve unchanged.

        A sequence of a continuous group resolves to its own media too. A
        division of a store is placed at the ``first_frame`` its row records,
        and reads its window of the store's chunks. A recording that mixes
        division rows with other rows cannot be placed, and neither can a
        division of a recording that the manifest does not declare continuous.
        Any other sequence's ``placement`` counts the media frames of the
        group's sequences before it, in member order: a natural sort on the
        names of the sequences that the media index or the tracks index names.
        A sequence that cannot be placed resolves, and
        :attr:`ResolvedMedia.first_frame` raises.

        Raises:
            FileNotFoundError: If the index is missing/empty or no row matches.
            MediaProbeError: If a row requires a transcode but has no derivative,
                a derivative's file/facts cannot be found, a matched row's
                stored measurement cannot be reconstructed, or *camera* is
                ``None`` while the sequence spans more than one camera.
            AmbiguousMediaMatchError: If the last matching tier leaves rows of
                more than one ``(group, sequence)`` standing.
        """
        df = self._load_media_index(index_filename)
        matched = self._matched_media_rows(df, group, sequence, camera)
        if camera is None:
            cameras = {c for c in matched["camera"].fillna("").astype(str) if c}
            if len(cameras) > 1:
                raise MediaProbeError(
                    f"sequence ({group!r}, {sequence!r}) spans {len(cameras)} "
                    f"cameras {sorted(cameras)}; pass camera= to select one"
                )
        route_derivatives, derivative_df = self.media_routing_context(index_filename)
        resolved = self._resolve_matched_rows(
            group, sequence, matched, route_derivatives, derivative_df
        )
        first = matched.iloc[0]
        entry_group, entry_sequence = str(first["group"]), str(first["sequence"])
        if self.recording_kind(entry_group) == "undivided":
            unplaced = _undeclared_division(entry_group, matched)
            if unplaced is None:
                return resolved
            return dataclasses.replace(resolved, placement=unplaced)
        divided = _recording_divided(entry_group, df[df["group"] == entry_group])
        placement = _division_placement(entry_group, divided, matched)
        if isinstance(placement, int) and not isinstance(divided, UnplacedSequence):
            facts = divided.get(entry_sequence)
            return dataclasses.replace(
                resolved,
                placement=placement,
                time_offset=None if facts is None else facts.time_offset,
            )
        if placement is None:
            members = self._member_placement(
                df,
                entry_group,
                str(first["camera"]),
                (route_derivatives, derivative_df),
            )
            placement = members.place(entry_sequence, resolved.facts)
        return dataclasses.replace(resolved, placement=placement)

    def _member_placement(
        self,
        df: "pd.DataFrame",
        group: str,
        camera: str,
        routing: "tuple[bool, pd.DataFrame | None]",
    ) -> _MemberPlacement:
        """Return the placement of the divided recording *group*'s sequences, for *camera*.

        The sequences are :meth:`recording_members`, found among the rows of
        *df* and of the tracks index. *df* is the originals index, which routes
        their clips, and *routing* is :meth:`media_routing_context`.
        """
        tracks = (
            read_tracks_index(self)
            if self.has_root("tracks")
            else pd.DataFrame(columns=["run_id", "group", "sequence"])
        )
        route_derivatives, derivative_df = routing

        def route(sequence: str, rows: pd.DataFrame) -> ResolvedMedia:
            return self._resolve_matched_rows(
                group, sequence, rows, route_derivatives, derivative_df
            )

        rows = df[df["group"] == group]
        # Named as recording_entries names them: media rows as media resolution
        # names them, tracks rows as their index's cells read.
        indexed = {(group, str(sequence)) for sequence in rows["sequence"]} | {
            (text_cell(row_group), text_cell(sequence))
            for row_group, sequence in zip(
                tracks["group"], tracks["sequence"], strict=True
            )
        }
        members = self.recording_members(RecordingKey(group, ""), indexed)
        return _MemberPlacement(
            group,
            camera,
            [sequence for _, sequence in members],
            rows,
            tracks[tracks["group"] == group],
            route,
        )

    def _resolve_matched_rows(
        self,
        group: str,
        sequence: str,
        matched: "pd.DataFrame",
        route_derivatives: bool,
        derivative_df: "pd.DataFrame | None",
    ) -> ResolvedMedia:
        """Route one entry's matched rows into a :class:`ResolvedMedia`.

        Shared body for :meth:`resolve_media` and :meth:`resolve_media_scope`:
        routes each row through :meth:`route_media_row`, so every routed row
        carries its stored measurement. *matched* must already be
        ``video_order``-sorted.
        """
        paths: list[Path] = []
        facts: list[MediaFacts] = []
        windows: list[StoreWindow | None] = []
        for _, row in matched.iterrows():
            routed_path, routed_facts = self.route_media_row(
                group, sequence, row, route_derivatives, derivative_df
            )
            paths.append(routed_path)
            facts.append(routed_facts)
            windows.append(_row_window(row))
        return ResolvedMedia(paths=paths, facts=facts, windows=windows)

    def resolve_scope(
        self,
        scope: Scope | None = None,
        index_filename: str = "index.csv",
    ) -> ResolvedScope:
        """Enumerate the entries *scope* names, against the originals index.

        A selector -- a command line, a request body, a recipe -- either names
        entries or names groups and sequences whose members only the index can
        list. ``groups=["A"]`` alone means every sequence in group A.

        The index is read only when *groups* or *sequences* is given. An
        explicit entry list needs no index, and a dataset that has none still
        resolves one. A camera-addressed selector resolves to its
        ``(group, sequence)`` pairs. The camera it named is recorded on
        ``ResolvedScope.selector``, and :meth:`resolve_media_scope` expands the
        camera axis for every resolved entry.

        Args:
            scope: What to cover. ``None`` and ``Scope()`` both mean every
                indexed entry, and both resolve to an empty entry set whose
                selector reports ``is_unset``.
            index_filename: The index to enumerate against.

        Returns:
            The resolved entries, beside the selector they were resolved from.

        Raises:
            FileNotFoundError: If *groups* or *sequences* is given and the
                originals index does not exist.
        """
        selector = scope if scope is not None else Scope()
        if selector.groups is None and selector.sequences is None:
            named = selector.entry_pairs or set()
            return ResolvedScope(entries=named, selector=selector)

        df = self._load_media_index(index_filename)
        mask = pd.Series(True, index=df.index)
        if selector.groups is not None:
            mask &= df["group"].isin({str(group) for group in selector.groups})
        if selector.sequences is not None:
            mask &= df["sequence"].isin(
                {str(sequence) for sequence in selector.sequences}
            )
        scoped = df[mask]
        enumerated = {
            (str(group), str(sequence))
            for group, sequence in zip(scoped["group"], scoped["sequence"], strict=True)
        }
        return ResolvedScope(entries=enumerated, selector=selector)

    def resolve_media_scope(
        self,
        entries: Iterable[Entry] | None,
        index_filename: str = "index.csv",
        *,
        errors: dict[Entry, MediaProbeError] | None = None,
    ) -> list[ResolvedScopeEntry]:
        """Enumerate the scoped ``(group, sequence, camera)`` entries with media.

        Reads the originals index once, restricts it to the rows whose
        ``(group, sequence)`` pair is in *entries* (``None`` keeps all), and
        returns one :class:`ResolvedScopeEntry` per distinct
        ``(group, sequence, camera)`` in deterministic order -- so the cameras of
        one recording become separate entries and are never concatenated into a
        single timeline. An explicit enumeration pins an arbitrary subset even
        when sequence names repeat across groups. :meth:`resolve_scope` is the
        first stage of the two, enumerating that list from a group or sequence
        selector. Each entry's :class:`ResolvedMedia` carries its
        ``video_order``-sorted paths and stored facts, routed by transcode
        verdict as :meth:`resolve_media` does, and its ``placement``
        as :meth:`resolve_media` places it. When an entry has an empty
        group and a sequence whose safe name is empty, the returned sequence
        label falls back to the first original file's stem.

        Args:
            entries: The ``(group, sequence)`` pairs to resolve, or ``None`` for
                every indexed entry.
            index_filename: The originals index to read.
            errors: When given, an entry whose media cannot be routed is
                recorded here under its ``(group, sequence)`` pair, mapped to the
                error that resolving it alone raises, and its cameras are not
                returned. The rest of the scope still resolves from the one
                read.

        Raises:
            FileNotFoundError: If the originals index does not exist.
            MediaProbeError: If *errors* is not given and an entry requires a
                transcode but has no derivative, a derivative's file/facts
                cannot be found, or a matched row's stored measurement cannot
                be reconstructed.
        """
        # Media-index resolution is a Dataset concern, so a scoped enumeration lives
        # here as a method (mirroring resolve_media) rather than as a free function in
        # a consumer package: the read-and-route logic stays in one layer instead of
        # being duplicated upward.
        df = self._load_media_index(index_filename)
        scoped = self._scoped_media_rows(df, entries)

        route_derivatives, derivative_df = self.media_routing_context(index_filename)
        resolved_entries: list[tuple[Entry, ResolvedScopeEntry]] = []
        failed: dict[Entry, MediaProbeError] = {}
        divided: dict[tuple[str, str], _MemberPlacement] = {}
        # One verdict per recording, so that placing its n sequences reads its
        # rows once rather than n times.
        verdicts: dict[
            str, dict[str, OrderFacts] | UnplacedSequence | MediaProbeError
        ] = {}
        for block in scoped:
            if block.entry in failed:
                continue
            try:
                resolved = self._resolve_matched_rows(
                    block.group,
                    block.sequence,
                    block.rows,
                    route_derivatives,
                    derivative_df,
                )
                if self.recording_kind(block.group) == "divided":
                    if block.group not in verdicts:
                        try:
                            verdicts[block.group] = _recording_divided(
                                block.group, df[df["group"] == block.group]
                            )
                        except MediaProbeError as exc:
                            verdicts[block.group] = exc
                    verdict = verdicts[block.group]
                    if isinstance(verdict, MediaProbeError):
                        raise verdict
                    placement = _division_placement(block.group, verdict, block.rows)
                    time_offset: float | None = None
                    if isinstance(placement, int) and not isinstance(
                        verdict, UnplacedSequence
                    ):
                        found = verdict.get(block.sequence)
                        time_offset = None if found is None else found.time_offset
                    if placement is None:
                        place = (block.group, block.camera)
                        if place not in divided:
                            divided[place] = self._member_placement(
                                df, *place, (route_derivatives, derivative_df)
                            )
                        placement = divided[place].place(block.sequence, resolved.facts)
                    resolved = dataclasses.replace(
                        resolved, placement=placement, time_offset=time_offset
                    )
                else:
                    unplaced = _undeclared_division(block.group, block.rows)
                    if unplaced is not None:
                        resolved = dataclasses.replace(resolved, placement=unplaced)
            except MediaProbeError as exc:
                if errors is None:
                    raise
                failed[block.entry] = exc
                continue
            resolved_entries.append(
                (
                    block.entry,
                    ResolvedScopeEntry(
                        block.group, block.sequence_label, block.camera, resolved
                    ),
                )
            )
        if errors is not None:
            errors.update(failed)
        return [media for entry, media in resolved_entries if entry not in failed]

    def division_facts(self, entries: Iterable[Entry]) -> dict[Entry, OrderFacts]:
        """Return what the division row of each of *entries* records.

        An entry of a declared continuous group whose rows are all division
        rows of its stores has facts: its first frame, start, frames, whether it
        follows a gap, and its first frame on the recording's time axis. Every
        other entry is left out.

        Raises:
            MediaProbeError: If a division row's cells do not read.
        """
        wanted = {
            entry for entry in entries if self.recording_kind(entry[0]) == "divided"
        }
        if not wanted:
            return {}
        try:
            df = self._load_media_index()
        except FileNotFoundError:
            return {}
        found: dict[Entry, OrderFacts] = {}
        for group in sorted({group for group, _ in wanted}):
            facts = _recording_divided(group, df[df["group"] == group])
            if isinstance(facts, UnplacedSequence):
                continue
            for sequence, fact in facts.items():
                if (group, sequence) in wanted:
                    found[group, sequence] = fact
        return found

    def sequence_first_frames(
        self,
        entries: Collection[Entry],
        *,
        errors: dict[Entry, str] | None = None,
    ) -> dict[Entry, frozenset[int]]:
        """Return each entry's first frame on its recording's frame axis, one per camera.

        Each camera's first frame is :attr:`ResolvedMedia.first_frame` as
        :meth:`resolve_media_scope` places it. An entry without indexed media is
        left out, as is a camera whose media cannot be routed or placed.

        Args:
            entries: The ``(group, sequence)`` pairs to place.
            errors: When given, each entry with a camera that cannot be routed
                or placed is recorded here with the reason.
        """
        if not entries:
            return {}
        failed: dict[Entry, MediaProbeError] = {}
        try:
            scope = self.resolve_media_scope(sorted(entries), errors=failed)
        except FileNotFoundError:
            return {}
        found: dict[Entry, set[int]] = {}
        reasons = {entry: str(exc) for entry, exc in failed.items()}
        for item in scope:
            entry = (item.group, item.sequence)
            try:
                first_frame = item.resolved.first_frame
            except MediaProbeError as exc:
                reasons.setdefault(entry, str(exc))
                continue
            found.setdefault(entry, set()).add(first_frame)
        if errors is not None:
            errors.update(reasons)
        return {entry: frozenset(frames) for entry, frames in found.items()}

    def _sequence_starts(self, entries: Iterable[Entry]) -> dict[Entry, int | str]:
        """Return the first frame of each divided-recording entry, or why it has none.

        The first frame is the latest of :meth:`sequence_first_frames` over the
        entry's cameras. An entry that no member of its recording precedes is
        read without its media (:meth:`_opening_starts`). An undivided entry
        starts at frame 0 and is left out.
        """
        divided = {
            entry for entry in entries if self.recording_kind(entry[0]) == "divided"
        }
        errors: dict[Entry, str] = {}
        found = self.sequence_first_frames(divided, errors=errors)
        opened = self._opening_starts(
            {entry for entry in divided if entry in errors or entry not in found}
        )
        starts: dict[Entry, int | str] = {}
        for entry in divided:
            if entry in opened:
                starts[entry] = opened[entry]
            elif entry in errors:
                starts[entry] = errors[entry]
            elif entry in found:
                starts[entry] = max(found[entry])
            else:
                starts[entry] = "its media is not indexed"
        return starts

    def _opening_starts(self, entries: Collection[Entry]) -> dict[Entry, int | str]:
        """Return the first frame of each entry that no member of its recording precedes.

        Such an entry is at frame 0 of its divided recording
        (:meth:`_MemberPlacement.opening`). Its media need be neither indexed
        nor routable, unless a tracks variant orders it apart, which is
        returned as the reason. An entry that a member precedes is left out.
        """
        if not entries:
            return {}
        try:
            df = self._load_media_index()
        except FileNotFoundError:
            df = pd.DataFrame(columns=["group", "sequence", "camera"])
        routing = self.media_routing_context()
        placements = {
            group: self._member_placement(df, group, "", routing)
            for group in {group for group, _ in entries}
        }
        starts: dict[Entry, int | str] = {}
        for group, sequence in entries:
            opening = placements[group].opening(sequence)
            if isinstance(opening, UnplacedSequence):
                starts[group, sequence] = opening.reason
            elif opening is not None:
                starts[group, sequence] = opening
        return starts

    def media_entries(self) -> frozenset[Entry]:
        """Every ``(group, sequence)`` the originals index holds media for.

        Each entry is named as :meth:`resolve_media_scope` names it, and the
        cameras of one recording are one entry. Nothing is routed, so an entry
        whose media cannot be read yet is listed too, such as one that requires
        an analysis transcode that has not run. The read that routes the entry
        reports why it cannot.

        Raises:
            FileNotFoundError: If the originals index does not exist.
        """
        return frozenset(
            (block.group, block.sequence_label)
            for block in self._scoped_media_rows(self._load_media_index(), None)
        )

    def _scoped_media_rows(
        self, df: "pd.DataFrame", entries: Iterable[Entry] | None
    ) -> list[_ScopedMediaRows]:
        """Split the originals index into one block per ``(group, sequence, camera)``.

        The one enumeration behind :meth:`resolve_media_scope` and
        :meth:`media_entries`, so both name an entry alike: a nameless entry
        (:func:`~mosaic.core.helpers.is_nameless_entry`) is labeled by its first
        original file's stem. Blocks come in ``(group, sequence, camera)`` order,
        and nothing is routed.

        Args:
            df: The originals index, as :meth:`_load_media_index` reads it.
            entries: The ``(group, sequence)`` pairs to keep, or ``None`` for
                every indexed entry.
        """
        mask = pd.Series(True, index=df.index)
        if entries is not None:
            mask &= _entry_mask(df, entries)
        blocks: list[_ScopedMediaRows] = []
        for (group, sequence, camera), sub in df[mask].groupby(
            ["group", "sequence", "camera"]
        ):
            group, sequence, camera = str(group), str(sequence), str(camera)
            rows = sub.sort_values("video_order")
            label = sequence
            if is_nameless_entry(group, sequence):
                label = self.resolve_path(str(rows.iloc[0]["abs_path"])).stem
            blocks.append(_ScopedMediaRows(group, sequence, label, camera, rows))
        return blocks

    def _build_media_sequence_keymap(self) -> dict[str, list[dict[str, str]]]:
        """
        Build a lookup of various sequence keys -> metadata for mapping media files to sequences.
        """
        df = legacy_view(read_tracks_index(self))
        keymap: dict[str, list[dict[str, str]]] = {}
        for _, row in df.iterrows():
            group = str(row["group"])
            sequence = str(row["sequence"])
            if not sequence:
                continue
            # Derived, not read-with-fallback. The old `row.get("group_safe") or
            # ...` did not fire on a present-but-empty cell -- a NaN is truthy,
            # so it returned the NaN and the next `.lower()` raised
            # AttributeError. legacy_view derives both unconditionally.
            group_safe = str(row["group_safe"])
            sequence_safe = str(row["sequence_safe"])
            tail = Path(sequence).name
            tail_safe = to_safe_name(tail) if tail else ""
            keys = {
                sequence,
                sequence.lower(),
                sequence_safe,
                sequence_safe.lower(),
                tail,
                tail.lower(),
                tail_safe,
                tail_safe.lower(),
            }
            meta = {
                "group": group,
                "sequence": sequence,
                "group_safe": group_safe,
                "sequence_safe": sequence_safe,
            }
            for key in keys:
                if not key:
                    continue
                keymap.setdefault(key, []).append(meta)
        return keymap

    @staticmethod
    def _match_media_sequence(
        seq_key_map: dict[str, list[dict[str, str]]],
        stem: str,
        mode: str = "exact",
    ) -> Optional[dict[str, str]]:
        if not seq_key_map or not stem:
            return None
        candidates = [
            stem,
            stem.lower(),
            to_safe_name(stem),
            to_safe_name(stem).lower(),
        ]

        # Exact match: try each candidate key directly
        for key in candidates:
            hits = seq_key_map.get(key)
            if not hits:
                continue
            # An ambiguous key matches nothing. Several entries can register the
            # same key -- the keymap adds ``Path(sequence).name`` as a shorthand,
            # so `task1/train/m010` and `task1/test/m010` both register `m010`,
            # and a file named `m010.mp4` belongs to neither more than the other.
            # Taking hits[0] bound it to whichever sorted first, silently and
            # differently on a re-index. Refusing to guess leaves the file
            # unmatched, which is visible; guessing wrong is not.
            distinct = {(hit["group"], hit["sequence"]) for hit in hits}
            if len(distinct) > 1:
                print(
                    f"[media] {stem!r} matches {len(distinct)} track entries via "
                    f"{key!r}: {sorted(distinct)}. Leaving it unmatched -- rename "
                    "the media file to the full sequence name to disambiguate.",
                    file=sys.stderr,
                )
                return None
            return hits[0]

        if mode == "prefix":
            # Prefix match: find the longest known key that is a prefix
            # of any candidate form of the stem. Longest wins to avoid
            # ambiguity (e.g. "session01" vs "session01_special").
            stem_lc = stem.lower()
            stem_safe = to_safe_name(stem).lower()
            best_key: Optional[str] = None
            best_len = 0
            for key in seq_key_map:
                key_lc = key.lower()
                if len(key_lc) <= best_len:
                    continue
                if stem_lc.startswith(key_lc) or stem_safe.startswith(key_lc):
                    best_key = key
                    best_len = len(key_lc)
            if best_key is not None:
                return seq_key_map[best_key][0]

        return None

    def index_tracks_raw(
        self,
        search_dirs: Iterable[str | Path],
        patterns: Iterable[str] | str = ("*.npy", "*.h5", "*.csv"),
        src_format: str = "calms21_npy",
        index_filename: str = "index.csv",
        recursive: bool = True,
        multi_sequences_per_file: bool = False,
        group_from: Optional[str] = None,
        group_pattern: Optional[str] = None,
        group_from_path: Optional[Callable[[Path], str]] = None,
        exclude_patterns: Optional[Iterable[str]] = None,
        compute_md5: bool = True,
        *,
        claim: ScanClaim | None = None,
        prune_unsourced: bool = False,
    ) -> Path:
        """
        Scan for original tracking files and write tracks_raw/index.csv
        Columns: group, sequence, abs_path, src_format, size_bytes, mtime_iso, md5

        Parameters
        ----------
        search_dirs : Iterable[str | Path]
            Directories to search for files
        patterns : Iterable[str] | str
            Glob patterns to match files
        src_format : str
            Source format identifier (e.g., "trex_npz", "calms21_npy"). Must
            name a registered converter: it is resolved here, both so that a
            typo fails with the known formats listed rather than writing an
            index nothing can convert, and because the converter is what says
            how a filename stem names its sequence.
        index_filename : str
            Name of output index file
        recursive : bool
            Whether to search recursively
        multi_sequences_per_file : bool
            If True (e.g., CalMS files), set 'group' from group_from and leave 'sequence' blank
        group_from : str | None
            For multi_sequences_per_file: 'filename' or 'parent'
        group_pattern : str | None
            Regex pattern to extract group from sequence name. Must have a capturing group.
            Examples:
                r'^(hex|OCI|OLE)_' -> extracts 'hex', 'OCI', or 'OLE' as group
                r'^([A-Za-z]+)_'   -> extracts letters before first underscore as group
            Applied AFTER sequence is determined (e.g., after stripping a TREx ID suffix like _id0).
        group_from_path : Callable[[Path], str] | None
            Derive the group from the raw file's path, for a grouping that is a
            *rule* rather than a substring -- "day 1 is baseline, anything else
            is treatment", a lookup table, a parent directory two levels up.
            ``group_pattern`` can only lift text that is already there, and a
            dataset whose grouping needs a conditional was previously forced to
            patch ``index.csv`` after the fact, which conversion then could not
            see and the next re-index silently undid.

            Called once per discovered file with its :class:`~pathlib.Path`, and
            its return value is used verbatim (an entry name, so it may not
            contain ``/``, ``\\`` or NUL -- ``build_tracks_raw_row`` enforces
            that). Applies in both ``multi_sequences_per_file`` modes and
            supersedes ``group_from`` there. Must be **deterministic**: it feeds
            the entry key, so an unstable answer moves every downstream
            identifier. An exception it raises propagates -- a file the rule
            cannot classify is an error worth seeing, not a silent ``""``.

            Mutually exclusive with ``group_pattern``: two ways to spell one
            answer, silently picking one, is how the two drift apart.
        exclude_patterns : Iterable[str] | None
            Glob patterns to exclude
        compute_md5 : bool
            Checksum each source file. **Default True**, because the
            ``tracks_raw`` composition hash (item 4.4) is over these checksums,
            so an off-by-default column leaves every sequence's composition
            unestablishable -- which is what it was for every dataset, since
            nothing in the toolkit ever passed True. A digest is carried forward
            from the existing row when the stored path, size and mtime all still
            match, so a re-index re-hashes only what changed. Pass False for a
            corpus too large or too slow to hash and accept an unestablished
            composition: an honest empty, not a wrong value.
        """
        return self._index_raw(
            target_root="tracks_raw",
            composition_writer=self._write_tracks_raw_compositions,
            search_dirs=search_dirs,
            patterns=patterns,
            src_format=src_format,
            # Resolved here, so a src_format naming no converter is refused with
            # the registered ones listed rather than writing an index nothing can
            # convert. Which files come several per sequence, and what names the
            # individual within one, are the converter's to declare -- neither is
            # anything this method knows.
            sequence_from_stem=get_track_converter(src_format).sequence_from_stem,
            index_filename=index_filename,
            recursive=recursive,
            multi_sequences_per_file=multi_sequences_per_file,
            group_from=group_from,
            group_pattern=group_pattern,
            group_from_path=group_from_path,
            exclude_patterns=exclude_patterns,
            compute_md5=compute_md5,
            claim=claim,
            prune_unsourced=prune_unsourced,
        )

    def _index_raw(
        self,
        *,
        target_root: str,
        composition_writer: Callable[[list[dict[str, object]]], None],
        search_dirs: Iterable[str | Path],
        patterns: Iterable[str] | str,
        src_format: str,
        sequence_from_stem: Callable[[str], str],
        index_filename: str = "index.csv",
        recursive: bool = True,
        multi_sequences_per_file: bool = False,
        group_from: Optional[str] = None,
        group_pattern: Optional[str] = None,
        group_from_path: Optional[Callable[[Path], str]] = None,
        exclude_patterns: Optional[Iterable[str]] = None,
        compute_md5: bool = True,
        claim: ScanClaim | None = None,
        prune_unsourced: bool = False,
    ) -> Path:
        """Scan a source root and write its ``<root>/index.csv`` + composition.

        The body shared by :meth:`index_tracks_raw` and :meth:`index_labels_raw`.
        A raw source file carries the same seven columns whichever root it lands
        in (``group, sequence, abs_path, src_format, size_bytes, mtime_iso,
        md5``), and the scan, row build and md5 carry-forward are identity-free,
        so only *target_root* and *composition_writer* -- which sequence-index to
        project into -- distinguish a tracks source from a label source.

        *sequence_from_stem* is the one thing that varies with the source's
        format, and it arrives as a rule rather than being decided here. A format
        whose files come several per sequence -- TRex writes one ``.npz`` per
        individual -- has to drop what names the individual before the rest can
        be recognized as one entry, and this body cannot ask which format it is
        holding: ``src_format`` names a *track* converter on one path and a
        *label* converter on the other, and the label registry is not even keyed
        the same way. So each caller resolves the rule and passes it, the same
        seam ``group_from_path`` is on the branch below. Required rather than
        defaulted, because a default is where a third caller would silently get
        the wrong answer.
        """
        out_csv = self.get_root(target_root) / index_filename
        rows: list[TracksRawIndexRow] = []
        # Advisory, and read before the lock like the media probe phase's caches:
        # a stale entry costs a re-hash, never a wrong digest, because every hit
        # is re-validated against the file's current size and mtime.
        carried = self._prior_digests(out_csv) if compute_md5 else {}

        def _digest(path: Path) -> str:
            if not compute_md5:
                return ""
            return carried.get(path.resolve()) or _md5(path)

        # Materialized once: `search_dirs` is an Iterable read both here and by
        # the claim below, so a generator argument left the claim empty.
        search = [user_path(d) for d in search_dirs]
        pat_list = _normalize_patterns(patterns)
        exc_list = _normalize_patterns(exclude_patterns)
        if group_pattern and group_from_path is not None:
            raise ValueError(
                "pass group_pattern or group_from_path, not both -- they are two "
                "spellings of one answer, and silently preferring one is how the "
                "two drift apart. A rule that needs the regex can call it itself."
            )
        group_re = re.compile(group_pattern) if group_pattern else None

        for p, st in iter_track_files(
            search,
            pat_list,
            recursive=recursive,
            exclude_patterns=exc_list,
        ):
            if multi_sequences_per_file:
                # put file-level grouping into 'group', leave sequence blank
                if group_from == "filename":
                    grp = p.stem
                elif group_from == "parent":
                    grp = p.parent.name
                else:
                    grp = ""
                seq = ""
            else:
                seq = sequence_from_stem(p.stem)

                # Extract group from sequence using pattern
                if group_re:
                    m = group_re.search(seq)
                    grp = m.group(1) if m else ""
                else:
                    grp = ""

            # Last, so it supersedes both branches above: a caller that supplies a
            # rule has said the derivations here do not express their grouping.
            if group_from_path is not None:
                grp = str(group_from_path(p))

            rows.append(
                build_tracks_raw_row(
                    path=p,
                    stat=st,
                    to_store_path=self.relative_to_root,
                    group=grp,
                    sequence=seq,
                    src_format=src_format,
                    md5=_digest(p),
                )
            )

        # iter_track_files already deduped by resolved path and sorted. The scan
        # above is the expensive, read-only phase; the lock covers the write
        # alone, which is what makes the rewrite safe against a concurrent one.
        #
        # **A replace over what the scan claims, not over the file** -- the same
        # rule the media scan follows, and for the same reasons. It is also what
        # lets one dataset hold two source formats: scanning a trex directory and
        # then a CalMS21 one used to leave only the second.
        if claim is None:
            claim = ScanClaim.over_directories(search)
        # The same widening the media scan applies, for the same reason: a
        # symlinked raw file records its target, which the directory claim does
        # not cover, and the scan would preserve the row it had just written.
        claim = claim | self.walked_claim(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with index_lock(out_csv):
            committed = _read_tracks_raw_index(out_csv)
            preserved: list[dict[str, object]] = (
                []
                if prune_unsourced
                else [
                    dict(row) for row in committed if not self._row_claimed(row, claim)
                ]
            )
            written = entries_of_rows(rows)
            self.refuse_entry_key_collision(
                written,
                self._entries_beside_tracks_raw_write(written, committed, preserved),
            )
            # Preserved rows come back from the reader as plain mappings, while
            # this pass built typed rows; the frame builder takes either.
            merged: list[dict[str, object] | TracksRawIndexRow] = [*preserved, *rows]
            df = _tracks_frame_from_rows(merged)
            write_tracks_raw_index_rows(out_csv, df)
        # The merged rows, not the scanned ones: the composition writer replaces
        # the per-sequence index, so projecting only what this pass walked would
        # delete the composition row of every sequence it preserved.
        composition_writer(df.to_dict("records"))
        print(f"[index_{target_root}] {len(df)} -> {out_csv}")
        return out_csv

    def index_label_series(self, source: LabelsScanSource) -> Path:
        """Index the revisions *source* claims into ``labels_raw/<series>/index.csv``.

        The series sibling of :meth:`index_labels_raw`, and deliberately not built
        on its body. That body derives a ``(group, sequence)`` per file and ends by
        rewriting the per-sequence composition; a revision has neither. A keypoint
        set spans sequences, and even a per-sequence series must stay out of the
        composition, because a composition that moved on every save would block
        every derivative each time an editor was closed.

        **A replace over what the source claims, and nothing else**, the rule every
        scan follows. Rows this dataset's own writer appended sit under no source
        and survive; a claimed revision that has gone leaves.

        Each revision is checked against the digest recorded when it was written.
        A file whose size and modification time have not moved since the last scan
        keeps the digest it had, so a rescan of many unchanged revisions reads
        none of them.

        Raises:
            ValueError: *source* declares no series.
            LabelSeriesTamperedError: A claimed revision's bytes have changed.
        """
        if source.series is None:
            msg = f"labels source {source.id!r} declares no series"
            raise ValueError(msg)
        spec = series_spec(source.series)
        index_path = series_index_path(self, spec.name)
        ensure_series_root(index_path.parent, spec)

        if source.mode == "files":
            payloads = sorted(self._source_files(source))
        else:
            base = self.resolve_source_path(source)
            found = (
                base.rglob(spec.payload_filename)
                if source.recursive
                else base.glob(spec.payload_filename)
            )
            payloads = sorted(
                path
                for path in found
                if path.is_file() and revision_of(path.parent.name)
            )

        carried = carried_series_digests(self, spec.name)
        rows = [
            row_for_revision_file(
                self,
                path,
                series=spec.name,
                source_id=source.id,
                known_digest=carried.get(path.resolve(), ""),
            )
            for path in payloads
        ]

        claim = self.source_claim(source)
        with index_lock(index_path):
            committed = index_records(read_label_series(self, spec.name))
            preserved = [row for row in committed if not self._row_claimed(row, claim)]
            merged: list[dict[str, object]] = [
                *[dict(row) for row in preserved],
                *[dataclasses.asdict(row) for row in rows],
            ]
            write_label_series_rows(index_path, merged)
        print(f"[index_label_series:{spec.name}] {len(merged)} -> {index_path}")
        return index_path

    def index_labels_raw(
        self,
        search_dirs: Iterable[str | Path],
        patterns: Iterable[str] | str = ("*.csv", "*.npy", "*.pkl"),
        src_format: str = "boris_aggregated_csv",
        index_filename: str = "index.csv",
        recursive: bool = True,
        multi_sequences_per_file: bool = False,
        group_from: Optional[str] = None,
        group_pattern: Optional[str] = None,
        group_from_path: Optional[Callable[[Path], str]] = None,
        exclude_patterns: Optional[Iterable[str]] = None,
        compute_md5: bool = True,
        *,
        claim: ScanClaim | None = None,
        prune_unsourced: bool = False,
    ) -> Path:
        """Scan for raw label files and write ``labels_raw/index.csv``.

        The label sibling of :meth:`index_tracks_raw`, and the source side item
        9.3 gives converted labels. ``src_format`` must name a registered *label*
        converter (e.g. ``boris_aggregated_csv``), the column a later
        :meth:`convert_all_labels` filters on; it is checked here, because a
        format nothing claims writes an index whose rows that filter then skips
        in silence -- forever, and without ever saying which rows. Only the
        format half of the ``(src_format, label_kind)`` key is checked: the pair
        is what a conversion resolves, and which kind is wanted is not a question
        indexing a source has been asked.

        Files are indexed where they lie and never moved, so a format
        registered as both a track and a label
        converter -- ``calms21_npy`` -- can be indexed into both roots without
        copying; membership is by row, and the two compositions stay independent
        (see :class:`~mosaic.core.pipeline.composition.SourceMember`).

        ``compute_md5`` defaults True for the same reason it does on
        :meth:`index_tracks_raw`: the ``labels_raw`` composition is over these
        checksums, and an empty column leaves it unestablishable.

        ``group_from_path`` is the same seam it is there -- a grouping that is a
        rule rather than a substring -- and carries the same contract.
        """
        return self._index_raw(
            target_root="labels_raw",
            composition_writer=self._write_labels_raw_compositions,
            search_dirs=search_dirs,
            patterns=patterns,
            # Checked against the *label* registry, which is keyed on
            # ``(src_format, label_kind)`` -- a track-registry lookup would be
            # wrong even where it happened to hit, as it does for
            # ``calms21_npy``.
            src_format=validate_label_format(src_format),
            # A label file is one sequence, and no label converter says
            # otherwise, so there is no rule to resolve from the converter the
            # way the tracks side does.
            sequence_from_stem=_stem_as_sequence,
            index_filename=index_filename,
            recursive=recursive,
            multi_sequences_per_file=multi_sequences_per_file,
            group_from=group_from,
            group_pattern=group_pattern,
            group_from_path=group_from_path,
            exclude_patterns=exclude_patterns,
            compute_md5=compute_md5,
            claim=claim,
            prune_unsourced=prune_unsourced,
        )

    def write_tracks_raw_index(
        self,
        scopes: Iterable[TracksRawIndexScope],
        *,
        patterns: Iterable[str] | str = ("*.npy", "*.h5", "*.csv"),
        index_filename: str = "index.csv",
        recursive: bool = True,
        exclude_patterns: Optional[Iterable[str]] = None,
        compute_md5: bool = True,
    ) -> Path:
        """Project explicit raw-track assignments into a valid tracks_raw index.

        Named for the ``tracks_raw`` root it writes, not the ``tracks`` one: the
        converted tables have an index of their own, and a bare
        ``write_tracks_index`` would name the wrong file to anyone who did not
        already know which of the two this was.

        The assignment-driven counterpart to :meth:`index_tracks_raw` and the
        tracks sibling of :meth:`write_media_index`: rather than deriving each
        file's identity from its name, the caller passes one
        :class:`TracksRawIndexScope` per affected (group, sequence) -- its
        tracks_raw subdir and the explicit (group, sequence, src_format). Every
        file under a scope directory matching *patterns* is (re)stat-ed and given
        that scope's identity (no ``_fishN`` strip -- the caller owns grouping);
        every existing row not under any scope directory (other sequences,
        external ``abs_path`` values) is preserved verbatim; and the index is
        written atomically with root-relative ``abs_path``. Tracks are unordered,
        so there is no ``video_order`` densifier.

        Overlapping scope directories are caller error: a file scanned by two
        scopes is written once (first occurrence wins, via the merged dedup).
        Passing no scopes rewrites the index verbatim (idempotent). This is the
        single entry point a future API raw-track-import flow calls -- it owns
        none of these semantics itself.
        """
        out_csv = self.get_root("tracks_raw") / index_filename
        scope_list = list(scopes)
        scope_dirs = [scope.directory.resolve() for scope in scope_list]
        pat_list = _normalize_patterns(patterns)
        exc_list = _normalize_patterns(exclude_patterns)

        # --- Scan phase: expensive, read-only, unlocked. ---
        # Stat-ing (and, with compute_md5, hashing) every file under every scope
        # is the slow part, so the lock is taken below for the merge and write
        # only. The rows to preserve are re-read authoritatively there; this read
        # is advisory, and a stale entry costs a re-hash rather than a wrong
        # digest because every hit is re-validated against size and mtime.
        carried = self._prior_digests(out_csv) if compute_md5 else {}

        def _digest(path: Path) -> str:
            if not compute_md5:
                return ""
            return carried.get(path.resolve()) or _md5(path)

        fresh: list[TracksRawIndexRow] = []
        for scope in scope_list:
            for path, st in iter_track_files(
                [scope.directory],
                pat_list,
                recursive=recursive,
                exclude_patterns=exc_list,
            ):
                fresh.append(
                    build_tracks_raw_row(
                        path=path,
                        stat=st,
                        to_store_path=self.relative_to_root,
                        group=scope.group,
                        sequence=scope.sequence,
                        src_format=scope.src_format,
                        md5=_digest(path),
                    )
                )

        # --- Commit phase: cheap, and locked. ---
        # Preserve every existing row no scope directory covers. _row_under_dirs
        # resolves the stored path first, so containment is correct whether
        # abs_path is stored root-relative or absolute. Read inside the lock: a
        # writer that landed rows while this one scanned wrote exactly the rows
        # this one must preserve.
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with index_lock(out_csv):
            existing = _read_tracks_raw_index(out_csv)
            preserved = [
                row for row in existing if not self._row_under_dirs(row, scope_dirs)
            ]
            written = entries_of_rows(fresh)
            self.refuse_entry_key_collision(
                written,
                self._entries_beside_tracks_raw_write(written, existing, preserved),
            )
            # Merge preserved (string records) + fresh (typed rows).
            # _row_under_dirs already guarantees the two are disjoint by resolved
            # path, so the dedup only ever fires fresh-vs-fresh across
            # overlapping scopes (caller error).
            merged = [*preserved, *fresh]
            frame = _tracks_frame_from_rows(merged).drop_duplicates(subset=["abs_path"])
            write_tracks_raw_index_rows(out_csv, frame)
        self._write_tracks_raw_compositions(frame.to_dict("records"))
        return out_csv

    def read_tracks_raw_index(
        self, index_filename: str = "index.csv"
    ) -> list[dict[str, str]]:
        """Read the raw-tracks index as string-cell records (empty list if absent).

        The ``tracks_raw`` root, not ``tracks`` -- see
        :meth:`write_tracks_raw_index` for why the name says so.
        """
        return _read_tracks_raw_index(self.get_root("tracks_raw") / index_filename)

    # ----------------------------
    # Convert one original -> standard (T-Rex-like)
    # ----------------------------
    def convert_one_track(
        self,
        raw_row: pd.Series,
        params: Optional[dict] = None,
        overwrite: bool = False,
        covered: set[tuple[str, str]] | None = None,
        variants: set[str] | None = None,
    ) -> Path:
        """
        Convert a single raw track file (row from tracks_raw/index.csv) to a
        standardized parquet in whatever schema the converter declares.
        Returns path to standardized file, updates tracks/index.csv.

        *covered* collects the ``(group, sequence)`` entries the file names, and
        *variants* the tracks variant it was converted under.
        """
        params = params or {}
        src_format = str(raw_row["src_format"])
        src_path = self.resolve_path(raw_row["abs_path"])

        converter = get_track_converter(src_format)
        conv_params = self._converter_params(converter, params)
        # Asked of the converter rather than of the manifest. A dataset-level
        # ``standard_format`` was the wrong altitude: one dataset legitimately
        # holds tables from several converters, and only the converter knows
        # what shape it emits. It also let the name the table was *validated*
        # against differ from the one its index row *recorded*.
        std_fmt = converter.output_schema
        # Minted and recorded once per recipe, not per entry, and carried onto
        # every row this call writes. Also names the directory the tables live in
        # and described in its params.json, so a variant is explicable from disk.
        variant = self._tracks_variant(converter, conv_params)
        if variants is not None:
            variants.add(variant)
        producer = converter_op(src_format)

        # Where to place standardized file: one directory per tracks variant,
        # holding <group>__<seq>.parquet. Two recipes for one sequence no longer
        # collide, which is what the exists()/overwrite skips below used to hide.
        variant_root = tracks_variant_root(self.get_root("tracks"), variant)

        # Both identities read once, here, and used by every branch below. The
        # row comes off a bare pandas ``Series``, so a blank cell is a float NaN
        # and each ``str()`` of it would mint the word "nan" -- once into a
        # filename, once into an index row, and the two need not even agree.
        seq_value = text_cell(raw_row.get("sequence", ""))
        group_value = text_cell(raw_row.get("group", ""))
        source_md5 = text_cell(raw_row.get("md5", ""))

        # If sequence missing/blank and the format can hold several, expand this
        # file into multiple per-sequence outputs
        if (not seq_value) and converter.enumerable:
            # policy: 'filename' (default), 'infile', 'both'
            policy = str(params.get("group_from", _GROUP_FROM_DEFAULT)).lower()
            if policy not in {"infile", "filename", "both"}:
                policy = _GROUP_FROM_DEFAULT

            pairs = converter.enumerate_sequences(src_path)
            if not pairs:
                raise ValueError(
                    f"No (group, sequence) pairs enumerated for {src_path}"
                )
            produced = []

            for g, s in pairs:
                # canonical (with '/')
                canon_seq = s
                # decide output group by policy
                canon_group_infile = g or ""
                out_group_canon = canon_group_infile
                if policy in {"filename", "both"} and group_value:
                    out_group_canon = group_value

                # output path -- make_entry_key does the safe-name encoding
                if covered is not None:
                    covered.add((out_group_canon, canon_seq))
                stem = make_entry_key(out_group_canon, canon_seq)
                out_path = variant_root / f"{stem}.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Respect overwrite flag when outputs already exist, unless the
                # sources moved under them -- see ``_conversion_is_current``.
                if (
                    out_path.exists()
                    and not overwrite
                    and self._conversion_is_current(variant, out_group_canon, canon_seq)
                ):
                    produced.append(out_path)
                    continue

                # Which entry to produce -- passed beside the params, never
                # merged into them, so the sequence name cannot reach a digest.
                division = self.division_facts([(out_group_canon, canon_seq)]).get(
                    (out_group_canon, canon_seq)
                )
                hints = _placed_hints(
                    EntryHints(group=canon_group_infile, sequence=canon_seq), division
                )
                df_std = converter.convert(src_path, conv_params, hints)

                # Overwrite group column in DataFrame to match policy
                if "group" in df_std.columns and out_group_canon != canon_group_infile:
                    df_std["group"] = out_group_canon

                df_std = _placed_conversion(
                    df_std,
                    hints,
                    (out_group_canon, canon_seq),
                    f"{src_path}::{canon_seq}",
                    division,
                )

                # Ensure schema, then write
                _, _schema_report = ensure_track_schema(
                    df_std,
                    std_fmt,
                    strict=conv_params.strict_schema,
                    source=f"{src_path}::{canon_seq}",
                )
                _ = write_parquet_atomic(df_std, out_path)

                # The file-level grouping hint the old row kept as 'collection'
                # is not recorded: it had no reader anywhere, and it stays
                # recoverable from tracks_raw/index.csv through source_abs_path.
                write_tracks_row(
                    self,
                    run_id=variant,
                    group=out_group_canon,
                    sequence=canon_seq,
                    out_path=out_path,
                    producer=producer,
                    std_format=std_fmt,
                    n_rows=int(len(df_std)),
                    source=src_path,
                    source_md5=source_md5,
                    consumed_source_roots=("tracks_raw",),
                )
                produced.append(out_path)

            return self.get_root("tracks") / "index.csv"

        # Normal single-sequence path (default). The safe names are not
        # computed here any more: make_entry_key derives them for the filename,
        # and the index stores the raw identity plus nothing derivable from it.
        if covered is not None:
            covered.add((group_value, seq_value))
        rel_name = f"{make_entry_key(group_value, seq_value)}.parquet"
        out_path = variant_root / rel_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if (
            out_path.exists()
            and not overwrite
            and self._conversion_is_current(variant, group_value, seq_value)
        ):
            return out_path

        # Which entry to produce. The hints always win: the caller knows the
        # authoritative entry. (The old dict merged them with ``setdefault``
        # here and with direct assignment in the branch above, so a
        # user-supplied ``params["group"]`` won in one path and lost in the
        # other -- one of the reasons entry identity does not belong in params.)
        division = self.division_facts([(group_value, seq_value)]).get(
            (group_value, seq_value)
        )
        hints = _placed_hints(
            EntryHints(group=group_value, sequence=seq_value), division
        )
        df_std = converter.convert(src_path, conv_params, hints)
        df_std = _placed_conversion(
            df_std, hints, (group_value, seq_value), str(src_path), division
        )

        # Validate/coerce against the declared standard format schema (if any)
        _, _schema_report = ensure_track_schema(
            df_std, std_fmt, strict=conv_params.strict_schema, source=str(src_path)
        )

        _ = write_parquet_atomic(df_std, out_path)

        # The same two values the filename above was built from. Reading the row
        # a second time here is what let the table and its index row disagree:
        # a blank group reached one as "" and the other as "nan".
        write_tracks_row(
            self,
            run_id=variant,
            group=group_value,
            sequence=seq_value,
            out_path=out_path,
            producer=producer,
            std_format=std_fmt,
            n_rows=int(len(df_std)),
            source=src_path,
            source_md5=str(raw_row.get("md5", "")),
            consumed_source_roots=("tracks_raw",),
        )
        return out_path

    def list_converters(self) -> Dict[str, type[TrackConverter[TrackConvertParams]]]:
        """Return registered raw->standard track converters."""
        ensure_track_converters_registered()
        return dict(TRACK_CONVERTERS)

    def _tracks_variant(
        self,
        converter: TrackConverter[TrackConvertParams],
        params: TrackConvertParams,
        observed: Optional[dict] = None,
        correction: int = 0,
    ) -> str:
        """Mint this conversion's tracks-variant identity and record it.

        Params-only and scope-free, so one value names one recipe however many
        sequences it covers. Recorded beside the tables in
        ``tracks/<run_id>/params.json``, which Stage 3.2 turns into the
        directory the parquets themselves live in.

        *correction* names the revision when what is being converted is a
        promoted correction, which makes it a variant of its own rather than
        more rows in the uncorrected one.
        """
        cls = type(converter)
        run_id = _track_variant_id(converter, params, correction)
        _ = write_tracks_variant(
            self.get_root("tracks"),
            run_id,
            converter_op(cls.src_format),
            cls.version,
            params.identity_dump(),
            observed,
        )
        return run_id

    def track_conversion_variant(
        self,
        source_format: str,
        params: Mapping[str, object] | None = None,
        *,
        correction: int = 0,
    ) -> str:
        """Name the tracks variant a conversion of *source_format* would write.

        The identity :meth:`convert_all_tracks` mints for one format, computed
        without writing anything, so a planner can name the variant before the
        conversion runs.

        Args:
            source_format: The registered track converter's ``src_format``.
            params: Converter overrides, as :meth:`convert_all_tracks` takes
                them. ``fps`` falls back to the dataset's ``fps_default``.
            correction: The promoted correction revision, ``0`` for an upload.

        Returns:
            The tracks variant identifier.

        Raises:
            ValueError: *source_format* names no registered converter.
        """
        converter = get_track_converter(source_format)
        conv_params = self._converter_params(
            converter, dict(params or {}), source_format
        )
        return _track_variant_id(converter, conv_params, correction)

    def _converter_params(
        self,
        converter: TrackConverter[TrackConvertParams],
        overrides: Optional[Mapping[str, Any]],
        src_format: str = "",
    ) -> TrackConvertParams:
        """Build a converter's typed params from user overrides.

        Two resolutions happen here, both *before* the values can be hashed,
        which is the same rule item 1.1 established for upstream references:

        - ``fps`` falls back to the dataset's ``fps_default``, so the value
          recorded in identity is the one that was actually used rather than a
          field default standing in for it.
        - Entry identity (``group`` / ``sequence``) and the caller-side
          ``group_from`` policy are dropped: they travel as ``EntryHints`` or
          are resolved by the caller, and neither belongs in a recipe.

        Unknown keys raise rather than being ignored, because ``Params`` forbids
        extras. On a single-format dataset that is what you want -- silently
        dropping ``neck_idx`` produced a wrong heading. On a mixed-format one,
        pass ``params_by_format`` so each converter sees only its own keys.
        """
        merged = dict(overrides or {})
        if src_format and isinstance(merged.get("params_by_format"), dict):
            by_format: dict = merged.pop("params_by_format")
            merged = {**merged, **dict(by_format.get(src_format, {}))}
        else:
            merged.pop("params_by_format", None)
        for dropped in ("group", "sequence", "group_from"):
            merged.pop(dropped, None)

        params_cls = type(converter).Params
        if "fps" in params_cls.model_fields and "fps" not in merged:
            fps_default = self.meta_float("fps_default")
            if fps_default is not None:
                merged["fps"] = float(fps_default)
        # The old dict spelled the same thing two ways; accept the legacy key
        # rather than silently ignoring it, then let the model reject anything
        # genuinely unrecognized.
        if "fps_default" in merged:
            merged.setdefault("fps", float(merged["fps_default"]))
            merged.pop("fps_default")
        return params_cls.from_overrides(merged)

    def list_schemas(self) -> Dict[str, TrackSchema]:
        """Return registered track schemas."""
        return dict(TRACK_SCHEMAS)

    # ----------------------------
    # Bulk convert
    # ----------------------------
    def _convert_rows_individually(
        self,
        rows: pd.DataFrame,
        params: dict[str, object] | None,
        group_from: str,
        overwrite: bool,
        failures: list[str] | None = None,
        *,
        variants: set[str] | None = None,
        on_failure: Callable[[str, BaseException], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
        advance: Callable[[int], None] | None = None,
    ) -> set[tuple[str, str]]:
        """Convert each raw row into a table of its own, warning per failure.

        The fallback both of :meth:`convert_all_tracks`' branches take when a set
        of rows has nothing to merge -- an index with no merging format at all,
        and a single group inside one that has. One copy, because two copies of
        "warn and keep going" are two chances for one of them to start raising.

        Returns:
            The ``(group, sequence)`` entries these rows cover -- written or
            skipped as already current. Filled by ``convert_one_track`` rather
            than predicted from the raw row, because an enumerable format expands
            one file into many sequences and the row's own ``sequence`` is blank.
            See :meth:`_warn_superseded_entries`.

        The keyword arguments are :meth:`convert_all_tracks`' hooks: *variants*
        collects the variants minted, *on_failure* hears each failed file from
        inside its ``except`` block, *cancel_check* runs before each row and
        *advance* after it.
        """
        covered: set[tuple[str, str]] = set()
        for _, row in rows.iterrows():
            if cancel_check is not None:
                cancel_check()
            try:
                call_params = dict(params) if params else {}
                call_params["group_from"] = group_from
                _ = self.convert_one_track(
                    row,
                    params=call_params,
                    overwrite=overwrite,
                    covered=covered,
                    variants=variants,
                )
            except TrackSchemaError:
                raise
            except Exception as e:
                print(
                    f"[WARN] convert failed for {row.get('abs_path')}: {e}",
                    file=sys.stderr,
                )
                if failures is not None:
                    failures.append(str(row.get("abs_path")))
                if on_failure is not None:
                    on_failure(str(row.get("abs_path")), e)
            if advance is not None:
                advance(1)
        return covered

    def convert_all_tracks(
        self,
        params: Optional[dict[str, object]] = None,
        overwrite: bool = False,
        merge_per_sequence: Optional[bool] = None,
        group_from: Optional[str] = None,
        *,
        source_format: str | None = None,
        progress: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
        on_failure: Callable[[str, BaseException], None] | None = None,
    ) -> ConversionOutcome:
        """
        Convert all raw track files (from tracks_raw/index.csv) to standard T-Rex-like parquet files.

        A format whose files come several per sequence -- TRex writes one .npz per
        individual -- declares ``merges_per_sequence`` on its converter, and those
        files are concatenated on the union of their columns into one parquet per
        (group, sequence). Every other format gets one table per file. The
        declaration is the converter's, so a format added later does not need this
        method edited, and a mixed-format index merges what merges without the
        rest turning it off.

        Parameters
        ----------
        params : dict | None
            Extra parameters to pass to converters.
        overwrite : bool
            If True, overwrite existing output files.
        merge_per_sequence : bool | None
            If None (the default), ask each format's converter: merge the groups
            whose format declares ``merges_per_sequence`` and convert the rest one
            file at a time. If True, merge every group whatever its converter
            declares -- the merge reads nothing about the format, so this is how a
            dataset whose files happen to come several per sequence says so
            without the converter claiming it everywhere. If False, merge nothing.
            A group with a blank sequence is never merged: it names no entry to
            merge into, and its file holds several sequences instead.
        group_from : {'infile','filename','both'} | None
            Controls which *group* ends up in the standardized output & index:
            - 'filename' (default): use the raw file-level group from tracks_raw/index.csv
                        (e.g., 'calms21_task1_test') where the scan recorded one, and the
                        group from inside the source file otherwise. The scan records one
                        only when the source declares a grouping rule.
            - 'infile' : use the group from inside the source file (e.g., 'annotator-id_0').
            - 'both'  : set output group to the raw file-level group, and still record in-file group in the data
                        (converters should already keep in-file columns; we always keep raw file-level hint in
                        the 'collection' column).
            If None, defaults to 'filename', as label conversion does, so one
            file feeding both roots names each entry alike.
        source_format : str | None
            Convert only the raw rows of this ``src_format``. None converts
            every row. A format with no row raises ``ValueError``.
        progress : callable | None
            Called as ``progress(done, total)`` with the raw rows handled so far.
        cancel_check : callable | None
            Called before each raw row or merged sequence. It stops the
            conversion by raising.
        on_failure : callable | None
            Called as ``on_failure(path, exc)`` for each raw file that failed,
            from inside the ``except`` block that caught it, so a caller can
            record the traceback.
        """
        raw_idx = self.get_root("tracks_raw") / "index.csv"
        if not raw_idx.exists():
            raise FileNotFoundError(
                "tracks_raw/index.csv not found; run index_tracks_raw first."
            )
        try:
            # Through the typed reader, not a bare read: a blank ``group`` cell
            # has to arrive as "" rather than as the float NaN whose ``str()``
            # is the word "nan", which is truthy and would reach a filename.
            df = load_tracks_raw_index_frame(raw_idx)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"tracks_raw/index.csv is empty or malformed: {raw_idx}\n"
                "This usually means index_tracks_raw() found no matching files.\n"
                "Check your search_dirs and patterns parameters."
            )
        if source_format is not None:
            df = df[[text_cell(fmt) == source_format for fmt in df["src_format"]]]
            if df.empty:
                msg = (
                    f"No rows in tracks_raw/index.csv with src_format="
                    f"{source_format!r}. Scan the dataset's track sources first."
                )
                raise ValueError(msg)

        total = int(len(df))
        handled = 0

        def advance(count: int) -> None:
            nonlocal handled
            handled += count
            if progress is not None:
                progress(handled, total)

        minted: set[str] = set()

        # Which formats merge per (group, sequence), asked of each converter
        # rather than of the index as a whole.
        #
        # The old default was "every row in the index is trex_npz", so one row of
        # any other format turned merging off for the trex ones too -- and the
        # per-individual files it then converted one at a time all named the same
        # (group, sequence) output, so the first landed and the rest were skipped
        # as already written.
        # A blank src_format names no converter, so it is not a format that could
        # merge. Dropped here rather than left to fall out of `dropna`, which no
        # longer sees it now that the reader spells a blank cell "".
        present: set[str] = {
            name
            for name in (text_cell(fmt) for fmt in df["src_format"].unique())
            if name
        }
        merging: set[str]
        if merge_per_sequence is None:
            merging = {fmt for fmt in present if _format_merges_per_sequence(fmt)}
        elif merge_per_sequence:
            merging = present
        else:
            merging = set()

        # normalize group_from
        group_from = (group_from or _GROUP_FROM_DEFAULT).lower()
        if group_from not in {"infile", "filename", "both"}:
            raise ValueError(
                f"group_from must be one of 'infile', 'filename', 'both'; got {group_from}"
            )

        # Sequences that raised, so the return value can say so. A refusal under
        # ``strict_schema`` never lands here: it propagates, because it is the
        # answer the caller asked for rather than a mishap to survive.
        failures: list[str] = []

        # Every entry the raw sources name, filled as the branches below walk
        # them. What is indexed but absent from this is an old spelling -- see
        # ``_warn_superseded_entries``.
        covered: set[tuple[str, str]] = set()

        if not merging:
            # Nothing in this index merges: one table per raw file.
            covered |= self._convert_rows_individually(
                df,
                params,
                group_from,
                overwrite,
                failures=failures,
                variants=minted,
                on_failure=on_failure,
                cancel_check=cancel_check,
                advance=advance,
            )
            self._warn_superseded_entries(covered)
            return ConversionOutcome(
                converted=len(covered),
                failed=len(failures),
                variants=tuple(sorted(minted)),
            )

        # The schema is *not* read here. This loop groups by ``src_format``, so
        # one call can span converters, and a single value chosen up front would
        # label one converter's tables with another's schema. It is read from
        # each group's own converter below, beside the variant it mints.

        # Merge per (group, sequence, src_format). The same rule the single-file
        # path applies per cell, applied here per column: these three both key
        # the groupby and name the merged table, so an untrimmed group would
        # split one sequence's files into two half-merges, and an untrimmed one
        # reaching make_entry_key below would name a file the index row for it
        # does not.
        groupby_cols = ["group", "sequence", "src_format"]
        df = df.copy()
        for col in groupby_cols:
            if col not in df.columns:
                df[col] = ""
            df[col] = [text_cell(value) for value in df[col]]

        # A promoted correction and the upload it corrects are the same format
        # for the same entry, so the three keys above put them in one group and
        # the merge below concatenated them -- one table holding both readings of
        # every frame, under one identifier. The revision joins the key so each
        # side converts separately, and reaches the variant identity so the two
        # tables are distinguishable rather than in conflict.
        df[_CORRECTION_COL] = [
            correction_revision(str(value)) for value in df["abs_path"]
        ]
        # Only the newest correction converts. The series is append-only history;
        # re-converting every revision would mint a variant per revision forever,
        # and the older ones are already on disk under their own.
        newest = df.groupby(groupby_cols)[_CORRECTION_COL].transform("max")
        df = df[(df[_CORRECTION_COL] == 0) | (df[_CORRECTION_COL] == newest)]
        groupby_cols = [*groupby_cols, _CORRECTION_COL]
        # Read once for every entry a merge may write, under either group cell.
        divisions = self.division_facts(
            {(text_cell(g), text_cell(q)) for g, q in zip(df["group"], df["sequence"])}
        )

        for keys, group_df in df.groupby(groupby_cols):
            group, sequence, src_format, correction = keys

            # Nothing to merge here: either this format's files come one per
            # sequence, or these rows name no sequence to merge them into -- a
            # blank sequence means the file holds several, which is
            # convert_one_track's expansion rather than this loop's concat.
            if src_format not in merging or not sequence:
                covered |= self._convert_rows_individually(
                    group_df,
                    params,
                    group_from,
                    overwrite,
                    failures=failures,
                    variants=minted,
                    on_failure=on_failure,
                    cancel_check=cancel_check,
                    advance=advance,
                )
                continue
            if cancel_check is not None:
                cancel_check()

            # Merge this sequence's several files into one table
            first_row = group_df.iloc[0]

            # Determine output path early so we can honor overwrite=False
            # without re-loading and merging every NPZ. The variant has to be
            # minted first now, because it names the directory the path is in --
            # so the skip below is asking "does *this recipe* already have this
            # table", where before it asked the weaker "does any recipe".
            # Minting is cheap and idempotent (`write_tracks_variant` rewrites
            # one small sidecar), and `convert_one_track` already mints above
            # both of its skips, so this only makes the two branches agree.
            converter = get_track_converter(src_format)
            conv_params = self._converter_params(converter, params, src_format)
            variant = self._tracks_variant(
                converter, conv_params, correction=int(correction)
            )
            minted.add(variant)

            raw_group_hint = text_cell(first_row.get("group", ""))
            out_group = group  # default: infile (already what we grouped by)
            if group_from in {"filename", "both"} and raw_group_hint:
                out_group = raw_group_hint
            covered.add((text_cell(out_group), text_cell(sequence)))
            variant_root = tracks_variant_root(self.get_root("tracks"), variant)
            rel_name = f"{make_entry_key(out_group, sequence)}.parquet"
            out_path = variant_root / rel_name
            if (
                out_path.exists()
                and not overwrite
                and self._conversion_is_current(variant, out_group, sequence)
            ):
                advance(int(len(group_df)))
                continue

            division = divisions.get((text_cell(out_group), text_cell(sequence)))
            hints = _placed_hints(
                EntryHints(group=text_cell(group), sequence=text_cell(sequence)),
                division,
            )

            dfs: list[pd.DataFrame] = []
            _merge_failed = False
            for _, row in group_df.iterrows():
                src_path = self.resolve_path(row["abs_path"])
                try:
                    df_std = converter.convert(src_path, conv_params, hints)
                except TrackSchemaError:
                    # The caller asked for strict validation, so this is an answer
                    # rather than a mishap. Warning and skipping would report
                    # success over a dataset quietly missing this sequence.
                    raise
                except Exception as e:
                    print(
                        f"[WARN] convert failed for {src_path}: {e}; "
                        f"skipping sequence ({group}, {sequence})",
                        file=sys.stderr,
                    )
                    failures.append(str(src_path))
                    if on_failure is not None:
                        on_failure(str(src_path), e)
                    _merge_failed = True
                    break
                dfs.append(df_std)
            if _merge_failed or not dfs:
                advance(int(len(group_df)))
                continue

            try:
                merged_df = _placed_conversion(
                    merge_on_column_union(dfs),
                    hints,
                    (text_cell(out_group), text_cell(sequence)),
                    f"{group}/{sequence} (merged)",
                    division,
                )
            except ValueError as e:
                print(
                    f"[WARN] convert failed for ({group}, {sequence}): {e}",
                    file=sys.stderr,
                )
                first_path = str(self.resolve_path(first_row["abs_path"]))
                failures.append(first_path)
                if on_failure is not None:
                    on_failure(first_path, e)
                advance(int(len(group_df)))
                continue
            _, _schema_report = ensure_track_schema(
                merged_df,
                converter.output_schema,
                strict=conv_params.strict_schema,
                source=f"{group}/{sequence} (merged)",
            )

            # Write output (out_path determined above for overwrite check)
            _ = write_parquet_atomic(merged_df, out_path)

            # source_abs_path names only the first of the N per-id files merged
            # here. Unchanged from before; the full set stays recoverable from
            # tracks_raw/index.csv by (group, sequence, src_format), and naming
            # every source is item 5.1's job rather than this column's.
            write_tracks_row(
                self,
                run_id=variant,
                group=out_group,
                sequence=sequence,
                out_path=out_path,
                producer=converter_op(src_format),
                std_format=converter.output_schema,
                n_rows=int(len(merged_df)),
                source=self.resolve_path(first_row["abs_path"]),
                source_md5=text_cell(first_row.get("md5", "")),
                consumed_source_roots=("tracks_raw",),
            )
            advance(int(len(group_df)))

        self._warn_superseded_entries(covered)
        outcome = ConversionOutcome(
            converted=len(covered),
            failed=len(failures),
            variants=tuple(sorted(minted)),
        )
        if failures:
            print(
                f"[convert-tracks] converted {outcome.converted} sequence(s); "
                f"{outcome.failed} failed: {', '.join(sorted(failures))}",
                file=sys.stderr,
            )
        return outcome

    # ----------------------------
    # Labels: conversion + indexing
    # ----------------------------
    def convert_all_labels(
        self,
        kind: str = "behavior",
        overwrite: bool = False,
        params: Optional[dict[str, object]] = None,
        source_format: Optional[str] = None,
        *,
        progress: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
        **kwargs: object,
    ) -> LabelConversionOutcome:
        """
        Convert labels from raw files using registered label converters.

        This method now uses a plugin architecture via the label_library.
        Converters are automatically registered for different source formats.

        Parameters
        ----------
        kind : str, default="behavior"
            Type of labels to convert (e.g., "behavior", "id_tags")
        overwrite : bool, default=False
            Whether to overwrite existing label files
        params : dict, optional
            Configuration parameters passed to converter
        source_format : str, optional
            Source format identifier (e.g., "calms21_npy", "boris_csv")
            Must match a registered converter's src_format
        progress : callable, optional
            Called as ``progress(done, total)`` with the raw files handled so far.
        cancel_check : callable, optional
            Called before each raw file. It stops the conversion by raising.
        **kwargs : additional keyword arguments
            Passed to converter (e.g., group_from, fps, etc.)

        Returns
        -------
        LabelConversionOutcome
            The variant written into, and how many label files were written and
            kept.

        Raises
        ------
        ValueError
            If no converter is registered for (source_format, kind) combination
        FileNotFoundError
            If labels_raw/index.csv is missing

        Examples
        --------
        Convert CalMS21 labels:
        >>> dataset.convert_all_labels(
        ...     kind="behavior",
        ...     source_format="calms21_npy",
        ...     group_from="filename"
        ... )

        Convert Boris labels (once implemented):
        >>> dataset.convert_all_labels(
        ...     kind="behavior",
        ...     source_format="boris_csv",
        ...     fps=30.0
        ... )
        """
        params = params or {}
        kind = str(kind or "").lower()
        src_format = source_format or params.get("source_format", "calms21_npy")

        # A missing pair is a caller error -- a typo, or a converter module
        # nothing imported -- and the resolver says so with the registered pairs
        # listed. Asked here rather than checked against the registry first: a
        # membership test read the registry before anything had filled it, and
        # then reported that no format exists.
        converter = get_label_converter(src_format, kind)
        conv_params = self._label_converter_params(
            converter, {**params, **kwargs}, src_format=src_format
        )

        # The source side is labels_raw, not tracks_raw: a label file is a label
        # source even when the same physical file is also a track source (item
        # 9.3). Read the labels_raw index and filter to this converter's format.
        raw_idx = self.get_root("labels_raw") / "index.csv"
        if not raw_idx.exists():
            raise FileNotFoundError(
                "labels_raw/index.csv not found; run index_labels_raw first."
            )
        # keep_default_na=False so an empty group cell stays "" rather than
        # becoming a float NaN that stringifies to "nan" and reaches an entry key.
        df_raw = pd.read_csv(raw_idx, keep_default_na=False, dtype=str)
        if "src_format" not in df_raw.columns:
            raise ValueError("labels_raw/index.csv missing 'src_format' column.")
        df_raw = df_raw[df_raw["src_format"].astype(str) == str(src_format)]
        if df_raw.empty:
            raise ValueError(
                f"No rows in labels_raw/index.csv with src_format='{src_format}'."
            )

        # Mint the label variant once -- params-only and scope-free, so one value
        # names one recipe across every sequence it covers -- and place its .npz
        # files under labels/<kind>/<run_id>/. A re-conversion with different
        # params mints a new variant beside the old rather than overwriting it.
        variant = self._labels_variant(converter, conv_params, label_kind=kind)
        kind_root = self.get_root("labels") / kind
        variant_root = labels_variant_root(kind_root, variant)
        variant_root.mkdir(parents=True, exist_ok=True)
        producer = label_converter_op(src_format)

        # Every entry is converted and placed before the first file is written,
        # so a refused entry leaves no part of the variant on disk.
        converted: list[tuple[LabelEntry, Path, Path, str]] = []
        skipped = 0
        total = int(len(df_raw))
        for done, (_, raw_row) in enumerate(df_raw.iterrows(), start=1):
            if cancel_check is not None:
                cancel_check()
            src_path = self.resolve_path(raw_row["abs_path"])
            source_md5 = str(raw_row.get("md5", "") or "")
            for entry in converter.convert(src_path, conv_params, raw_row):
                out_path = (
                    variant_root / f"{make_entry_key(entry.group, entry.sequence)}.npz"
                )
                if out_path.exists() and not overwrite:
                    skipped += 1
                    continue
                converted.append((entry, out_path, src_path, source_md5))
            if progress is not None:
                progress(done, total)

        # Only a dense payload depends on where its sequence starts.
        starts = self._sequence_starts(
            [
                (entry.group, entry.sequence)
                for entry, *_ in converted
                if detect_label_format(dict(entry.payload)) == "dense"
            ]
        )
        divisions = self.division_facts(
            {(entry.group, entry.sequence) for entry, *_ in converted}
        )
        pending: list[tuple[LabelEntry, Mapping[str, object], Path, Path, str]] = []
        for entry, out_path, src_path, source_md5 in converted:
            # A division's events are numbered from its first frame and move to
            # it. Every other entry's stay where the converter numbered them.
            division = divisions.get((entry.group, entry.sequence))
            payload = place_label_payload(
                entry.payload,
                first_frame=0
                if division is None or division.first_frame is None
                else division.first_frame,
                sequence_start=starts.get((entry.group, entry.sequence), 0),
                entry=entry_token(entry.group, entry.sequence),
            )
            pending.append((entry, payload, out_path, src_path, source_md5))

        for entry, payload, out_path, src_path, source_md5 in pending:
            atomic_savez(out_path, **dict(payload))
            write_labels_row(
                self,
                run_id=variant,
                group=entry.group,
                sequence=entry.sequence,
                out_path=out_path,
                producer=producer,
                label_kind=kind,
                label_format=converter.label_format,
                n_frames=entry.n_frames,
                label_ids=",".join(str(i) for i in entry.label_ids),
                label_names=",".join(str(n) for n in entry.label_names),
                source=src_path,
                source_md5=source_md5,
                consumed_source_roots=("labels_raw",),
            )
        written = len(pending)

        registration: dict[str, JsonValue] = {
            "run_id": variant,
            "label_format": converter.label_format,
            "updated_at": _now_iso(),
        }
        # A converter's metadata is JSON by contract, and the manifest writes it.
        registration.update(cast("dict[str, JsonValue]", converter.get_metadata()))

        def register(manifest: DatasetManifest) -> None:
            section = manifest.meta.get("labels")
            if not isinstance(section, dict):
                section = {}
                manifest.meta["labels"] = section
            section[kind] = registration

        # The write IS the registration. These NPZ files have no other record
        # than ``meta['labels'][kind]``, so a write that quietly did not happen
        # reports success and leaves a conversion nothing can find. It goes
        # through ``mutate_manifest`` because a pipeline step converting labels
        # races every other writer of the manifest.
        try:
            self.mutate_manifest(register)
        except OSError as exc:
            msg = (
                f"converted {written} sequences for label kind {kind!r}, but "
                f"could not record them in {self.manifest_path}: {exc}. The "
                "files are on disk and nothing indexes them; fix the manifest "
                "and re-run the conversion."
            )
            raise RuntimeError(msg) from exc

        print(
            f"[convert_all_labels] kind={kind} wrote {written} sequences as "
            f"variant {variant} using {src_format} (overwrite={overwrite})."
        )
        return LabelConversionOutcome(variant=variant, written=written, skipped=skipped)

    def _labels_variant(
        self,
        converter: LabelConverter[LabelConvertParams],
        params: LabelConvertParams,
        *,
        label_kind: str,
        observed: Optional[dict] = None,
    ) -> str:
        """Mint this label conversion's variant identity and record it.

        The label sibling of :meth:`_tracks_variant`. Params-only and scope-free,
        so one value names one recipe however many sequences it covers, recorded
        in ``labels/<kind>/<run_id>/params.json``. The ``kind`` term in the
        payload domain-separates two kinds that share a ``src_format``.
        """
        cls = type(converter)
        run_id = _label_variant_id(converter, params, label_kind)
        _ = write_labels_variant(
            self.get_root("labels") / label_kind,
            run_id,
            label_converter_op(cls.src_format),
            cls.version,
            label_kind,
            params.identity_dump(),
            observed,
        )
        return run_id

    def label_conversion_variant(
        self,
        kind: str,
        source_format: str,
        params: Mapping[str, object] | None = None,
    ) -> str:
        """Name the labels variant a conversion of *source_format* would write.

        The identity :meth:`convert_all_labels` mints, computed without writing
        anything, so a planner can name the variant before the conversion runs.

        Args:
            kind: The label kind, such as ``"behavior"``.
            source_format: The registered label converter's ``src_format``.
            params: Converter overrides, as :meth:`convert_all_labels` takes
                them.

        Returns:
            The labels variant identifier.

        Raises:
            ValueError: No converter is registered for the pair.
        """
        label_kind = str(kind or "").lower()
        converter = get_label_converter(source_format, label_kind)
        conv_params = self._label_converter_params(
            converter, dict(params or {}), src_format=source_format
        )
        return _label_variant_id(converter, conv_params, label_kind)

    def _label_converter_params(
        self,
        converter: LabelConverter[LabelConvertParams],
        overrides: Optional[Mapping[str, Any]],
        src_format: str = "",
    ) -> LabelConvertParams:
        """Build a label converter's typed params from user overrides.

        The label sibling of :meth:`_converter_params`, with one difference:
        ``group_from`` is a real ``LabelConvertParams`` field the converter reads
        (label converters legitimately choose which group string to assign), so it
        is *kept* here rather than dropped -- it is excluded from identity by being
        ``HASH_EXCLUDE`` on the model, not by being removed before hashing. Only
        the entry identity (``group`` / ``sequence``) is dropped, as it travels on
        the raw row, never in a recipe. ``fps`` falls back to the dataset's
        ``fps_default`` before it can be hashed, and ``params_by_format`` selects a
        converter's own keys on a mixed-format dataset.
        """
        merged = dict(overrides or {})
        if src_format and isinstance(merged.get("params_by_format"), dict):
            by_format: dict = merged.pop("params_by_format")
            merged = {**merged, **dict(by_format.get(src_format, {}))}
        else:
            merged.pop("params_by_format", None)
        for dropped in ("group", "sequence", "source_format"):
            merged.pop(dropped, None)

        params_cls = type(converter).Params
        if "fps" in params_cls.model_fields and "fps" not in merged:
            fps_default = self.meta_float("fps_default")
            if fps_default is not None:
                merged["fps"] = float(fps_default)
        if "fps_default" in merged:
            merged.setdefault("fps", float(merged["fps_default"]))
            merged.pop("fps_default")
        return params_cls.from_overrides(merged)

    def convert_labels_custom(
        self,
        converter_fn: Callable,
        kind: str = "behavior",
        label_format: str = "individual_pair_v1",
        overwrite: bool = False,
        **kwargs,
    ) -> int:
        """
        Convert labels using a custom converter function.

        This method provides flexibility for one-off datasets with unique label
        structures that don't fit the standard converter pattern. The Dataset
        handles all index.csv bookkeeping while you provide the conversion logic.

        Parameters
        ----------
        converter_fn : callable
            A function that performs the actual label conversion. Must have signature:

                converter_fn(dataset, labels_root, existing_pairs, overwrite, **kwargs)
                    -> list[dict]

            Where:
            - dataset: This Dataset instance (for accessing paths, metadata, etc.)
            - labels_root: Path to output directory (e.g., dataset/labels/behavior/)
            - existing_pairs: set of (group, sequence) tuples already converted
            - overwrite: bool, whether to overwrite existing files
            - **kwargs: Any additional arguments passed to convert_labels_custom

            Returns:
            - list[dict]: Index rows for each converted sequence. Each dict should have:
                - 'kind': str, label kind (e.g., "behavior")
                - 'label_format': str, format name (e.g., "individual_pair_v1")
                - 'group': str, group name
                - 'sequence': str, sequence name
                - 'group_safe': str, filesystem-safe group name
                - 'sequence_safe': str, filesystem-safe sequence name
                - 'abs_path': str, absolute path to output NPZ file
                - 'n_frames': int, number of unique frames with labels
                - 'n_events': int, total number of label events
                - 'label_ids': str, comma-separated label IDs (e.g., "0,1,2")
                - 'label_names': str, comma-separated label names (e.g., "none,troph,other")
                - (optional) additional metadata columns

        kind : str, default="behavior"
            Type of labels being converted (e.g., "behavior", "id_tags")

        label_format : str, default="individual_pair_v1"
            Format name for metadata. Should match what's saved in NPZ files.

        overwrite : bool, default=False
            Whether to overwrite existing label files

        **kwargs
            Additional arguments passed to converter_fn

        Returns
        -------
        int
            Number of sequences converted

        Examples
        --------
        >>> def my_converter(dataset, labels_root, existing_pairs, overwrite, **kwargs):
        ...     '''Custom converter for my unique dataset.'''
        ...     boris_path = kwargs['boris_path']
        ...     metadata_path = kwargs['metadata_path']
        ...     fps = kwargs.get('fps', 50.0)
        ...
        ...     # ... your conversion logic here ...
        ...     # Save NPZ files to labels_root
        ...     # Return list of index row dicts
        ...
        ...     return index_rows
        >>>
        >>> n_converted = dataset.convert_labels_custom(
        ...     converter_fn=my_converter,
        ...     kind="behavior",
        ...     boris_path=Path("/path/to/boris.tsv"),
        ...     metadata_path=Path("/path/to/metadata.json"),
        ...     fps=50.0,
        ... )

        NPZ File Format (individual_pair_v1)
        ------------------------------------
        The converter should save NPZ files with these keys:
        - 'group': str, group name
        - 'sequence': str, sequence name
        - 'label_format': str, "individual_pair_v1"
        - 'frames': int32 array, shape (n_events,), frame indices
        - 'labels': int32 array, shape (n_events,), label IDs
        - 'individual_ids': int32 array, shape (n_events, 2), [id1, id2] per event
          - For individual behaviors: [subject_id, -1]
          - For pair behaviors: [id1, id2] (symmetric: store both directions)
          - For scene-level: [-1, -1]
        - 'label_ids': int32 array, all label IDs (e.g., [0, 1, 2])
        - 'label_names': object array, label names (e.g., ["none", "troph", "other"])
        - 'background_id': int, the label ID a frame without an event means
          (e.g., 0 for "none"), or -1 when every label is a behavior and such a
          frame is unlabeled. A file without it makes every reader guess, and
          warn (``LegacyLabelFileWarning``).
        - 'fps': float, frames per second
        - (optional) additional metadata

        See Also
        --------
        convert_all_labels : For standard converters registered in label_library
        load_labels : Load converted labels
        """
        kind = str(kind or "behavior").lower()

        # An escape hatch for one-off datasets: the custom function still writes
        # its .npz files into labels/<kind>/ and returns index-row dicts. It is
        # *authored*, not scored -- its arbitrary Python has no recipe we could
        # honestly hash -- so the rows carry an empty run_id (unlabelled, the same
        # state a hand-authored id_tags row is in) and no consumed source root. A
        # later registered conversion of the same kind supersedes them by the
        # resolver's labelled-beats-unlabelled rule.
        labels_root = self.get_root("labels") / kind
        labels_root.mkdir(parents=True, exist_ok=True)

        existing = read_labels_index(self, kind)
        existing_pairs: set[tuple[str, str]] = {
            (str(row["group"]), str(row["sequence"])) for _, row in existing.iterrows()
        }

        new_rows = converter_fn(
            dataset=self,
            labels_root=labels_root,
            existing_pairs=existing_pairs,
            overwrite=overwrite,
            **kwargs,
        )

        for row in new_rows or []:
            abs_path = row.get("abs_path")
            if not abs_path:
                continue
            write_labels_row(
                self,
                run_id="",
                group=row.get("group", ""),
                sequence=row.get("sequence", ""),
                out_path=self.resolve_path(abs_path),
                producer="authored",
                label_kind=kind,
                label_format=str(row.get("label_format", label_format)),
                n_frames=int(row.get("n_frames", 0) or 0),
                label_ids=str(row.get("label_ids", "")),
                label_names=str(row.get("label_names", "")),
            )

        if new_rows:
            labels_meta = self.meta_section("labels")
            labels_meta[kind] = {
                "run_id": "",
                "label_format": label_format,
                "updated_at": _now_iso(),
            }
            # As in convert_all_labels: the save is what makes the written
            # files findable, so a failure has to be heard rather than passed.
            try:
                self.save()
            except OSError as exc:
                msg = (
                    f"wrote {len(new_rows)} sequences for label kind {kind!r}, "
                    f"but could not record them in {self.manifest_path}: {exc}. "
                    "The files are on disk and nothing indexes them; fix the "
                    "manifest and re-run the conversion."
                )
                raise RuntimeError(msg) from exc

        print(
            f"[convert_labels_custom] kind={kind} wrote {len(new_rows or [])} "
            f"sequences (authored, overwrite={overwrite})."
        )
        return len(new_rows or [])

    def save_id_labels(
        self,
        kind: str,
        group: str,
        sequence: str,
        per_id_labels: dict,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Persist per-(sequence, id) tags under labels/<kind>.

        per_id_labels: {id_value -> {"field": value, ...}}
        """
        if not per_id_labels:
            raise ValueError("per_id_labels must contain at least one entry.")
        labels_root = self.get_root("labels") / kind
        labels_root.mkdir(parents=True, exist_ok=True)

        fname = f"{make_entry_key(group, sequence)}.npz"
        out_path = labels_root / fname
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"ID labels already exist for ({group},{sequence}); set overwrite=True to replace."
            )

        id_keys = sorted(per_id_labels.keys(), key=lambda v: str(v))
        ids_array = np.asarray(id_keys, dtype=object)
        field_names = sorted(
            {name for tags in per_id_labels.values() for name in (tags or {}).keys()}
        )

        payload: dict[str, np.ndarray] = {"ids": ids_array}
        for field_name in field_names:
            values = []
            for key in id_keys:
                tags = per_id_labels.get(key) or {}
                values.append(tags.get(field_name))
            payload[field_name] = np.asarray(values, dtype=object)

        if metadata:
            for meta_key, meta_val in metadata.items():
                payload[f"meta__{meta_key}"] = np.asarray([meta_val], dtype=object)

        atomic_savez(out_path, **payload)

        # Authored in place from an external table that is in no raw index -- the
        # third label provenance (item 9.3): an empty run_id and no consumed
        # source root, neither scored nor derived, laid out flat rather than under
        # a variant directory because there is no recipe to name.
        write_labels_row(
            self,
            run_id="",
            group=group,
            sequence=sequence,
            out_path=out_path,
            producer="authored",
            label_kind=kind,
            label_format="id_tags_v1",
            n_frames=len(id_keys),
            label_ids=",".join(map(str, id_keys)),
            label_names=",".join(field_names),
        )
        return out_path

    def convert_id_tags_from_csv(
        self,
        csv_path: str | Path,
        csv_type: str = "focal",
        all_ids: Optional[list] = None,
        overwrite: bool = False,
        # Type-specific options:
        focal_id_column: str = "focal_id",
        id_column: str = "id",
        category_column: str = "category",
        field_columns: Optional[list[str]] = None,
    ) -> list[Path]:
        """
        Convert a CSV file to id_tags labels.

        This method supports different CSV formats for per-individual metadata:

        Supported csv_type values
        -------------------------
        "focal"
            One focal ID per sequence. CSV columns: group, sequence, focal_id.
            Creates boolean 'focal' field for all IDs (True for focal, False otherwise).
            Requires `all_ids` parameter to populate non-focal IDs.

        "category"
            Per-ID category labels. CSV columns: group, sequence, id, category.
            Creates 'category' field with the value from CSV.
            IDs not in CSV are skipped (or use all_ids to include them with None).

        "multi"
            Per-ID multiple fields. CSV columns: group, sequence, id, field1, field2...
            Creates one field per column specified in `field_columns`.

        Parameters
        ----------
        csv_path : str or Path
            Path to input CSV file
        csv_type : str
            One of "focal", "category", "multi"
        all_ids : list, optional
            List of all valid IDs. Required for csv_type="focal" to populate non-focal IDs.
            For other types, auto-detected from CSV if not provided.
        overwrite : bool
            Whether to overwrite existing id_tags files
        focal_id_column : str
            Column name for focal ID (csv_type="focal")
        id_column : str
            Column name for individual ID (csv_type="category" or "multi")
        category_column : str
            Column name for category value (csv_type="category")
        field_columns : list[str], optional
            List of column names to use as fields (csv_type="multi")

        Returns
        -------
        list[Path]
            Paths to created npz files

        Examples
        --------
        # Focal labels (one focal individual per sequence)
        >>> dataset.convert_id_tags_from_csv(
        ...     csv_path="focal_ids.csv",
        ...     csv_type="focal",
        ...     all_ids=list(range(8)),
        ...     overwrite=True,
        ... )

        # Category labels (e.g., strain per individual)
        >>> dataset.convert_id_tags_from_csv(
        ...     csv_path="strain_labels.csv",
        ...     csv_type="category",
        ...     category_column="strain",
        ... )

        # Multiple fields per individual
        >>> dataset.convert_id_tags_from_csv(
        ...     csv_path="individual_metadata.csv",
        ...     csv_type="multi",
        ...     field_columns=["strain", "treatment", "sex"],
        ... )
        """
        csv_path = user_path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Only the two identity columns are pinned as text. They name a file, so
        # a sequence "001" must not read back as the integer 1 and then name an
        # entry no tracks table has. The rest stay inferred on purpose: ``id``
        # keys the .npz that a tracks table's integer ``id`` column is looked up
        # in, and a blank ``focal_id`` has to stay NaN to be skipped rather than
        # become "" and reach ``int()``.
        df = pd.read_csv(csv_path, dtype={"group": str, "sequence": str})

        # Validate required columns
        if "group" not in df.columns or "sequence" not in df.columns:
            raise ValueError("CSV must have 'group' and 'sequence' columns")

        # Settled once, for all three branches below. These two key the groupby
        # the branches use and name the file it writes, and a blank cell is a
        # float NaN even under a pinned dtype. Left raw, ``groupby`` drops every
        # blank-group row before the loop body runs -- silently, and ``group`` is
        # empty on every dataset the control plane creates, so that is the common
        # case rather than an edge.
        for column in ("group", "sequence"):
            df[column] = [text_cell(value) for value in df[column]]
        unnamed = [int(i) + 2 for i in df.index[df["sequence"] == ""]]
        if unnamed:
            raise ValueError(
                f"{csv_path}: no sequence named on line(s) "
                f"{', '.join(str(line) for line in unnamed)}. A row naming no "
                "sequence names no entry, and would write a file with no name."
            )

        created: list[Path] = []

        if csv_type == "focal":
            # Focal type: one focal ID per sequence, boolean field for all IDs
            if all_ids is None:
                raise ValueError("all_ids is required for csv_type='focal'")
            if focal_id_column not in df.columns:
                raise ValueError(
                    f"CSV must have '{focal_id_column}' column for csv_type='focal'"
                )

            for _, row in df.iterrows():
                group = str(row["group"])
                seq = str(row["sequence"])
                focal_id = row[focal_id_column]

                # Convert focal_id to same type as all_ids elements for comparison
                if pd.notna(focal_id):
                    # Try to match type with all_ids
                    if all_ids and isinstance(all_ids[0], int):
                        focal_id = int(focal_id)

                per_id_labels = {
                    id_val: {"focal": (id_val == focal_id)} for id_val in all_ids
                }

                path = self.save_id_labels(
                    kind="id_tags",
                    group=group,
                    sequence=seq,
                    per_id_labels=per_id_labels,
                    overwrite=overwrite,
                )
                created.append(path)

        elif csv_type == "category":
            # Category type: per-ID category value
            if id_column not in df.columns:
                raise ValueError(
                    f"CSV must have '{id_column}' column for csv_type='category'"
                )
            if category_column not in df.columns:
                raise ValueError(
                    f"CSV must have '{category_column}' column for csv_type='category'"
                )

            # Group by (group, sequence), both already settled above
            for (group, seq), group_df in df.groupby(["group", "sequence"]):
                group = str(group)
                seq = str(seq)

                per_id_labels = {}
                for _, row in group_df.iterrows():
                    id_val = row[id_column]
                    if isinstance(id_val, float) and id_val.is_integer():
                        id_val = int(id_val)
                    cat_val = row[category_column]
                    per_id_labels[id_val] = {category_column: cat_val}

                # Add missing IDs with None if all_ids provided
                if all_ids is not None:
                    for id_val in all_ids:
                        if id_val not in per_id_labels:
                            per_id_labels[id_val] = {category_column: None}

                path = self.save_id_labels(
                    kind="id_tags",
                    group=group,
                    sequence=seq,
                    per_id_labels=per_id_labels,
                    overwrite=overwrite,
                )
                created.append(path)

        elif csv_type == "multi":
            # Multi type: multiple fields per ID
            if id_column not in df.columns:
                raise ValueError(
                    f"CSV must have '{id_column}' column for csv_type='multi'"
                )
            if field_columns is None:
                # Auto-detect: all columns except group, sequence, id
                field_columns = [
                    c for c in df.columns if c not in ["group", "sequence", id_column]
                ]
            if not field_columns:
                raise ValueError("No field columns found for csv_type='multi'")

            # Group by (group, sequence), both already settled above
            for (group, seq), group_df in df.groupby(["group", "sequence"]):
                group = str(group)
                seq = str(seq)

                per_id_labels = {}
                for _, row in group_df.iterrows():
                    id_val = row[id_column]
                    if isinstance(id_val, float) and id_val.is_integer():
                        id_val = int(id_val)
                    per_id_labels[id_val] = {col: row[col] for col in field_columns}

                # Add missing IDs with None values if all_ids provided
                if all_ids is not None:
                    for id_val in all_ids:
                        if id_val not in per_id_labels:
                            per_id_labels[id_val] = {col: None for col in field_columns}

                path = self.save_id_labels(
                    kind="id_tags",
                    group=group,
                    sequence=seq,
                    per_id_labels=per_id_labels,
                    overwrite=overwrite,
                )
                created.append(path)

        else:
            raise ValueError(
                f"Unknown csv_type: '{csv_type}'. Must be 'focal', 'category', or 'multi'."
            )

        print(f"Created {len(created)} id_tags files from {csv_path.name}")
        return created

    def load_id_labels(
        self,
        kind: str = "id_tags",
        *,
        scope: Scope | None = None,
        labels_run_id: Optional[str] = None,
    ) -> dict[tuple[str, str], dict]:
        """Load per-id labels of *kind*, for the entries *scope* names.

        Args:
            kind: Which labels root to read.
            scope: What to cover, as :class:`~mosaic.core.scope.Scope`.
                ``None``, the default, covers every entry the labels index
                names. A camera-addressed selector narrows to the
                ``(group, sequence)`` pairs it names, since a labels row is
                keyed without a camera.
            labels_run_id: Which label variant to read, or ``None`` for
                whichever variant each entry has.

        Returns:
            ``{(group, sequence): {"labels": {id: {field: value}},
            "sequence_safe": str, "path": str, "metadata": dict}}``.

        Raises:
            FileNotFoundError: No labels of *kind* exist.
        """
        selector = scope if scope is not None else Scope()
        df = select_label_variant_rows(read_labels_index(self, kind), labels_run_id)
        if df.empty:
            raise FileNotFoundError(
                f"No labels of kind='{kind}' found; author or convert them first."
            )
        if selector.groups is not None:
            wanted_groups = {str(group) for group in selector.groups}
            df = df[df["group"].astype(str).isin(wanted_groups)]
        if selector.sequences is not None:
            wanted_sequences = {str(sequence) for sequence in selector.sequences}
            df = df[df["sequence"].astype(str).isin(wanted_sequences)]
        wanted_entries = selector.entry_pairs
        if wanted_entries is not None:
            keyed = list(
                zip(df["group"].astype(str), df["sequence"].astype(str), strict=True)
            )
            df = df[[pair in wanted_entries for pair in keyed]]
        result: dict[tuple[str, str], dict] = {}
        for _, row in df.iterrows():
            group = str(row.get("group", "") or "")
            sequence = str(row.get("sequence", "") or "")
            safe_seq = to_safe_name(sequence)
            abs_path = str(row.get("abs_path", "")).strip()
            if not abs_path:
                continue
            path = self.resolve_path(abs_path)
            if not path.exists():
                continue
            with np.load(path, allow_pickle=True) as npz:
                ids = npz["ids"]
                meta = {}
                field_arrays: dict[str, np.ndarray] = {}
                for key in npz.files:
                    if key == "ids":
                        continue
                    if key.startswith("meta__"):
                        meta[key.split("meta__", 1)[1]] = _coerce_np(npz[key][0])
                        continue
                    field_arrays[key] = npz[key]
                per_id: dict[Any, dict[str, Any]] = {}
                for idx_id, raw_id in enumerate(ids):
                    id_value = _coerce_np(raw_id)
                    tags: dict[str, Any] = {}
                    for field, arr in field_arrays.items():
                        if arr.shape[0] == ids.shape[0]:
                            tags[field] = _coerce_np(arr[idx_id])
                        else:
                            tags[field] = _coerce_np(arr[0])
                    per_id[id_value] = tags
            result[(group, sequence)] = {
                "group": group,
                "sequence": sequence,
                "sequence_safe": safe_seq,
                "path": str(path),
                "labels": per_id,
                "metadata": meta,
            }
        return result

    def load_labels(
        self,
        group: str,
        sequence: str,
        kind: str = "behavior",
        labels_run_id: Optional[str] = None,
    ) -> dict:
        """
        Load behavior labels for a specific (group, sequence).

        Returns dict with keys:
        - frames: np.ndarray of frame indices
        - labels: np.ndarray of behavior IDs
        - individual_ids: np.ndarray of shape (n_events, 2) if individual_pair_v1 format
        - label_ids: np.ndarray of all possible label IDs
        - label_names: np.ndarray of label names
        - label_format: str indicating format version
        - group, sequence, sequence_key: metadata

        Which variant is read follows :func:`select_label_variant_rows`: pass
        ``labels_run_id`` to name one, otherwise a labelled variant supersedes an
        unlabelled row and two genuine recipes raise rather than silently taking
        the first.
        """
        df = select_label_variant_rows(read_labels_index(self, kind), labels_run_id)
        df = df[
            (df["group"].astype(str) == group)
            & (df["sequence"].astype(str) == sequence)
        ]

        if len(df) == 0:
            raise ValueError(
                f"No labels found for group='{group}', sequence='{sequence}', kind='{kind}'"
            )

        row = df.iloc[0]
        abs_path = str(row.get("abs_path", "")).strip()
        if not abs_path:
            raise ValueError(f"No abs_path in index for ({group}, {sequence})")

        path = self.resolve_path(abs_path)
        if not path.exists():
            raise FileNotFoundError(f"Label file not found: {path}")

        with np.load(path, allow_pickle=True) as npz:
            data = {key: npz[key] for key in npz.files}

        return data

    def get_label_map(
        self, kind: str = "behavior", labels_run_id: Optional[str] = None
    ) -> dict[int, str]:
        """
        Get the label map {id: name} for a label kind.

        Reads the vocabulary off the resolved variant's first row.
        """
        df = select_label_variant_rows(read_labels_index(self, kind), labels_run_id)
        if df.empty:
            raise FileNotFoundError(
                f"No labels of kind='{kind}' found; convert or author them first."
            )
        row = df.iloc[0]

        ids_str = str(row.get("label_ids", "")).strip()
        names_str = str(row.get("label_names", "")).strip()
        if not ids_str or not names_str:
            raise ValueError(f"No label_ids/label_names in index for kind='{kind}'")

        ids = [int(x) for x in ids_str.split(",")]
        names = names_str.split(",")
        return dict(zip(ids, names))

    def get_labels_for_individual(
        self,
        group: str,
        sequence: str,
        individual_id: int,
        kind: str = "behavior",
        frame_range: Optional[tuple[int, int]] = None,
    ) -> dict:
        """
        Get all label events for a specific individual.

        Parameters
        ----------
        group : str
            Group name
        sequence : str
            Sequence name
        individual_id : int
            Individual ID to filter by
        kind : str
            Label kind (default "behavior")
        frame_range : tuple[int, int], optional
            (start_frame, end_frame) to filter events

        Returns
        -------
        dict
            Dictionary with keys:
            - frames: np.ndarray of frame indices
            - labels: np.ndarray of behavior IDs
            - individual_ids: np.ndarray of shape (n_events, 2)
        """
        data = self.load_labels(group, sequence, kind)

        # Check format
        if "individual_ids" not in data:
            # Old format: backward compatibility
            # Return all frames assuming labels apply to this individual
            result = {
                "frames": data["frames"],
                "labels": data["labels"],
                "individual_ids": None,
            }
            if frame_range:
                start, end = frame_range
                mask = (data["frames"] >= start) & (data["frames"] <= end)
                result["frames"] = data["frames"][mask]
                result["labels"] = data["labels"][mask]
            return result

        # New format: filter by individual_id
        ids = data["individual_ids"]
        mask = (ids[:, 0] == individual_id) | (ids[:, 1] == individual_id)

        if frame_range:
            start, end = frame_range
            mask &= (data["frames"] >= start) & (data["frames"] <= end)

        return {
            "frames": data["frames"][mask],
            "labels": data["labels"][mask],
            "individual_ids": ids[mask],
        }

    def get_labels_at_frame(
        self,
        group: str,
        sequence: str,
        frame: int,
        kind: str = "behavior",
        individual_id: Optional[int] = None,
    ) -> dict:
        """
        Get all labels at a specific frame.

        Parameters
        ----------
        group : str
            Group name
        sequence : str
            Sequence name
        frame : int
            Frame index
        kind : str
            Label kind (default "behavior")
        individual_id : int, optional
            Filter by individual ID if provided

        Returns
        -------
        dict
            Dictionary with keys:
            - frames: np.ndarray of frame indices (should all equal frame)
            - labels: np.ndarray of behavior IDs
            - individual_ids: np.ndarray or None
        """
        data = self.load_labels(group, sequence, kind)

        mask = data["frames"] == frame

        if individual_id is not None and "individual_ids" in data:
            ids = data["individual_ids"]
            mask &= (ids[:, 0] == individual_id) | (ids[:, 1] == individual_id)

        result = {
            "frames": data["frames"][mask],
            "labels": data["labels"][mask],
        }

        if "individual_ids" in data:
            result["individual_ids"] = data["individual_ids"][mask]
        else:
            result["individual_ids"] = None

        return result

    # ----------------------------
    # Load tracks (by group/sequence)
    # ----------------------------
    def load_tracks(
        self,
        group: str,
        sequence: str,
        prefer: str = "standard",
        auto_convert: bool = True,
        convert_params: dict[str, object] | None = None,
        run_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load T-Rex-like standardized tracks if present; otherwise optionally auto-convert from raw.

        ``run_id`` names one tracks variant. Left ``None`` the entry resolves the
        same way it does for a feature run -- an unlabelled table loses to a
        labelled one, and two genuinely different recipes raise rather than pick.
        """
        # Try standardized index first. Through the typed reader and the shared
        # variant selector, not a bare read_csv: the old spelling matched on
        # ``len(hit) == 1`` and so, the moment an entry carried two variants,
        # fell through to *re-converting* rather than reading the table it had.
        rows = select_variant_rows(read_tracks_index(self), run_id)
        if not rows.empty:
            hit = rows[
                (rows["group"].astype(str) == group)
                & (rows["sequence"].astype(str) == sequence)
            ]
            if len(hit) == 1:
                return pd.read_parquet(self.resolve_path(hit.iloc[0]["abs_path"]))

        if prefer != "standard":
            raise FileNotFoundError(
                f"No non-standard loader implemented for prefer='{prefer}'"
            )

        # Fallback: find in raw index and convert
        raw_idx = self.get_root("tracks_raw") / "index.csv"
        if not raw_idx.exists():
            raise FileNotFoundError(
                "tracks_raw/index.csv not found; run index_tracks_raw first."
            )
        # Through the typed reader for the same reason convert_all_tracks is:
        # the mask below tolerated a NaN group, but the row it selects is handed
        # straight to convert_one_track, which would spell it "nan".
        df_raw = load_tracks_raw_index_frame(raw_idx)
        hit = df_raw[(df_raw["group"] == group) & (df_raw["sequence"] == sequence)]
        if len(hit) == 0:
            raise FileNotFoundError(
                f"No raw track for ({group}, {sequence}) found in tracks_raw/index.csv"
            )
        if not auto_convert:
            raise FileNotFoundError(
                f"Standardized track missing for ({group},{sequence}) and auto_convert=False"
            )

        std_path = self.convert_one_track(hit.iloc[0], params=convert_params or {})
        return pd.read_parquet(std_path)

    # --- Pipeline delegation methods ---

    def run_feature(
        self,
        feature: Any,
        overwrite: bool = False,
        parallel_workers: int | None = None,
        parallel_mode: str | None = "thread",
        overlap_frames: int = 0,
        filter_start_frame: int | None = None,
        filter_end_frame: int | None = None,
        filter_start_time: float | None = None,
        filter_end_time: float | None = None,
        check_output: bool = False,
        *,
        scope: Scope | None = None,
        tracks_run_id: str | None = None,
        labels_run_id: str | None = None,
        execution_id: str | None = None,
        owner: str = "",
        track: bool = True,
        progress_callback: "ProgressCallback | None" = None,
        cancel_token: "CancelToken | None" = None,
    ) -> Any:
        """Execute a feature extraction pipeline over the dataset.

        Runs the feature's ``fit()`` (if needed) and ``apply()`` phases over
        the chosen scope.  Input routing is determined by ``feature.inputs``:
        tracks (default), a single upstream feature result, or a multi-input set.

        Args:
            feature: Feature instance implementing the Feature protocol.
            scope: What to cover, as :class:`~mosaic.core.scope.Scope`.
                ``None``, the default, covers every entry the inputs resolve.
                ``Scope(entries=[...])`` names an arbitrary subset, unambiguous
                where sequence names repeat across groups.
                ``Scope(groups=[...], sequences=[...])`` narrows by name,
                combining as a cross product.
            tracks_run_id: Which tracks variant the ``"tracks"`` input
                resolves to. ``None`` lets each entry resolve to whichever
                variant it has; name one when a sequence carries two recipes and
                resolution declines to choose.
            overwrite: Re-run even if outputs exist for this run_id.
            parallel_workers: When >1 and the feature declares itself
                parallelizable, run the apply phase in parallel.
            parallel_mode: ``'thread'`` (default) or ``'process'`` execution
                backend when *parallel_workers* > 1.
            overlap_frames: Frames of context from the sequences on either side,
                for a feature computing across frames. Trimmed back off the
                output. Neighbors exist only inside a continuous group, whose
                sequences are time divisions of one recording on one frame axis.
                Any other entry reads only its own frames, and the run notes how
                many did on stderr. A feature that has not declared
                ``accepts_overlap`` is refused. Mutually exclusive with
                frame/time filters.
                See :func:`~mosaic.core.pipeline.run.run_feature`.
            filter_start_frame: Only include frames >= this value.
            filter_end_frame: Only include frames < this value.
            filter_start_time: Converted to start frame via *fps_default*
                from dataset metadata.
            filter_end_time: Converted to end frame via *fps_default*
                from dataset metadata.
            check_output: When True, deeply validate cached outputs before
                skipping them (default validator fully reads the parquet; a
                feature may override via ``check_output``). When False
                (default), a cache hit only requires the output to exist.
            execution_id: Reuse an externally minted ULID attempt id.
            owner: Free-form attribution recorded on the attempt's run-log.
            track: Record this attempt (status/progress/heartbeat) into an
                append-only JSONL run-log under ``<dataset_root>/.mosaic/runs/``
                (default). Set False to run without a trace.
            progress_callback: Optional progress backend (per-entry / per-epoch).
            cancel_token: Optional cooperative cancellation signal.

        Returns:
            A ``Result`` with ``feature`` and ``run_id`` (plus attempt-only
            ``execution_id``/``cache_hit``).  Pass directly to ``Inputs()`` to
            chain features.

        Example:
            ```python
            from mosaic.behavior.feature_library import SpeedAngvel, Inputs

            speed = SpeedAngvel()
            result = ds.run_feature(speed)

            # Chain into a downstream feature
            downstream = SomeFeature(Inputs((result,)))
            ds.run_feature(downstream)
            ```
        """
        from .pipeline.run import run_feature

        return run_feature(
            self,
            feature,
            scope=scope,
            overwrite=overwrite,
            parallel_workers=parallel_workers,
            parallel_mode=parallel_mode,
            overlap_frames=overlap_frames,
            filter_start_frame=filter_start_frame,
            filter_end_frame=filter_end_frame,
            filter_start_time=filter_start_time,
            filter_end_time=filter_end_time,
            check_output=check_output,
            tracks_run_id=tracks_run_id,
            labels_run_id=labels_run_id,
            execution_id=execution_id,
            owner=owner,
            track=track,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
        )


# The hand-written ``_ensure_labels_index`` / ``_append_labels_index`` are gone:
# ``labels/<kind>/index.csv`` is now the typed ``LabelsIndexRow`` index, written
# only through ``write_labels_row`` and read only through ``read_labels_index``.
