# dataset.py
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import sys
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Tuple,
)

import numpy as np
import pandas as pd
import yaml  # pip install pyyaml
from mosaic_media import (
    MediaFacts,
    MediaProbeError,
)

from .helpers import (
    ensure_text_column,
    make_entry_key,
    to_safe_name,
    validate_entry_name,
)
from .media.drift import MeasurementOrigin, MediaDrift, classify_identity
from .media.facts_columns import (
    MEDIA_INDEX_COLUMNS as MEDIA_INDEX_COLUMNS,  # re-exported for API/tests
    ProbeMetadata,
    derivative_path_for_target,
    media_row_uuid,
    read_link_cell,
    row_facts_or_none,
    row_mapping,
    row_to_facts,
    series_facts_or_none,
)
from .media.probe_row import probe_video_metadata, row_from_facts
from .media.uniformity import UniformityVerdict, camera_uniformity
from .schema import (
    TRACK_SCHEMAS,
    TrackSchema,
    ensure_track_schema,
)
from .stored_paths import remap_single_path as _remap_single_path, resolve_stored_path
from .track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    get_track_converter,
)

# This import used to sit at the foot of the file behind the one `noqa: E402`
# that could not be hoisted: `track_library.trex` imported
# `register_track_converter` from here, so the two formed a cycle and importing
# at the top ran the converter module before the decorator existed. The comment
# there said closing it needed the registry moved out of dataset.py rather than
# a reordering. Item 1.3 moved it to `core.track_converter`, so the cycle is
# gone and this is an ordinary import.
from .track_library.trex import strip_trex_seq
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
from .pipeline._utils import (
    atomic_write,
    coerce_np as _coerce_np,
    now_iso as _now_iso,
)
from .pipeline.media_index import (
    MEDIA_NUMERIC_COLUMNS,
    MediaIndexScope,
    build_media_index_row,
    build_prior_order,
    media_members_from_rows,
    densify_video_order,
    frame_from_rows,
    mtime_iso,
    read_media_index as _read_media_index,
    write_media_index_rows,
)
from .pipeline.tracks_identity import (
    convert_variant_payload,
    converter_op,
    tracks_run_id,
    tracks_variant_root,
    write_tracks_variant,
)
from .pipeline.composition import media_composition, tracks_raw_composition
from .pipeline.index_lock import index_lock
from .pipeline.sequence_index import (
    SequenceLabelRow,
    SourceRoot,
    read_sequence_labels,
    sequence_index_path,
    sequence_label_path,
    sequence_labels,
    write_sequence_compositions,
)
from .pipeline.dataset_indexes import iter_dataset_indexes
from .pipeline.tracking_roots import (
    TRACKING_ROOT,
    TRACKING_ROOTS,
    is_under_tracking_root,
)
from .pipeline.tracks_index import (
    TRACKS_INDEX_PATH_COLUMNS,
    adopt_legacy_columns,
    legacy_view,
    read_tracks_index,
    select_variant_rows,
    tracks_index_path,
    write_tracks_row,
)
from .pipeline.tracks_raw_index import (
    TracksRawIndexRow,
    TracksRawIndexScope,
    build_tracks_raw_row,
    frame_from_rows as _tracks_frame_from_rows,
    iter_track_files,
    read_tracks_raw_index as _read_tracks_raw_index,
    source_members_from_rows,
    write_tracks_raw_index_rows,
)

if TYPE_CHECKING:
    from .pipeline.job import CancelToken
    from .pipeline.progress import ProgressCallback


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
        normalized.append((Path(src).expanduser(), Path(dst).expanduser()))
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
_INDEX_PATH_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = {
    "tracks": TRACKS_INDEX_PATH_COLUMNS,
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


# ----------- Label converter registry -----------


class LabelConverter(Protocol):
    """Protocol for label converter plugins."""

    src_format: str  # e.g., "calms21_npy", "boris_csv"
    label_kind: str  # e.g., "behavior", "id_tags"
    label_format: str  # e.g., "calms21_behavior_v1"

    def convert(
        self,
        src_path: Path,
        raw_row: pd.Series,
        labels_root: Path,
        params: dict,
        overwrite: bool,
        existing_pairs: set[tuple[str, str]],
    ) -> list[dict]:
        """
        Convert a source file to label npz files.

        Returns: List of index row dicts for labels/index.csv
        """
        ...

    def get_metadata(self) -> dict:
        """Optional: return format-specific metadata for dataset.meta['labels'][kind]."""
        ...


# Registry: (src_format, label_kind) -> converter class
LABEL_CONVERTERS: dict[tuple[str, str], type] = {}


def register_label_converter(cls: type):
    """Decorator to register label converters."""
    key = (cls.src_format, cls.label_kind)
    LABEL_CONVERTERS[key] = cls
    return cls


# --- Standardized label metadata ---
BEHAVIOR_LABEL_MAP = {
    0: "attack",
    1: "investigation",
    2: "mount",
    3: "other_interaction",
}

LABEL_INDEX_COLUMNS = [
    "kind",
    "label_format",
    "group",
    "sequence",
    "group_safe",
    "sequence_safe",
    "abs_path",
    "source_abs_path",
    "source_md5",
    "n_frames",
    "label_ids",
    "label_names",
]


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


try:
    import yaml

    _YAML_OK = True
except Exception:
    _YAML_OK = False


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


############# DATASET

default_roots = {
    # ── raw (external, immutable) ──
    "media_raw": "media_raw",  # original uploaded videos — don't touch, may be on NAS
    "tracks_raw": "tracks_raw",  # original tracking files from external tools
    "labels": "labels",  # GT annotations: behavior labels, keypoints, individual IDs
    # ── derived (computed by mosaic, regenerable) ──
    "media": "media",  # derived media: low-res copies, re-encoded, thumbnails
    "tracks": "tracks",  # standardised parquet tracks (converted from tracks_raw)
    # Raw tracker output lives under _tracking/ (not tracks_raw/), so tracks_raw
    # holds only user-uploaded content. The per-tool roots come from the registry
    # rather than being spelled here: item 8.1 asked for one literal, and a dict
    # literal per tool is how it became six.
    TRACKING_ROOT: TRACKING_ROOT,  # parent of the per-tracker raw-output roots
    **{key: root.default_path for key, root in TRACKING_ROOTS.items()},
    "features": "features",  # per-sequence feature parquets (wavelets, projections, embeddings)
    "models": "models",  # trained models, reports, plots
    "frames": "media/frames",  # extracted video frames (PNGs), can be very large
}


def _backfill_tracking_roots(roots: Dict[str, str]) -> Dict[str, str]:
    """Add the tracking roots a manifest predating item 8.1 does not declare.

    ``load`` replaces ``roots`` wholesale, so a manifest written before the
    ``_tracking`` root existed carries neither it nor the per-tool keys beneath
    it. Left alone that is not a cosmetic gap: ``get_root("_tracking")`` raises,
    so the sweeper crashes on exactly the datasets that most need sweeping.

    **Absent keys are filled; present ones are never repointed.** A dataset whose
    ``trex`` root still reads ``tracks_raw/trex`` keeps it. Silently moving it
    would orphan every run already on disk *and* strand the index that names
    them -- and the legacy location is a state the sweeper must be able to
    recognize and decline, which it cannot do if loading has quietly erased the
    evidence. This is item 7.1's precedent applied to roots rather than to files:
    name the legacy class, refuse to act on it, leave it visible.

    In-place on the mapping ``load`` was handed, and returned for the caller to
    assign, so a manifest that needs no backfill is untouched and ``save`` round-
    trips it byte-identically.
    """
    for key in (TRACKING_ROOT, *TRACKING_ROOTS):
        _ = roots.setdefault(key, default_roots[key])
    return roots


def legacy_tracking_roots(roots: Mapping[str, str]) -> dict[str, str]:
    """Tracker roots still pointing at their pre-item-8.1 location.

    ``{root key: declared path}`` for every tracker root that resolves outside
    ``_tracking`` -- in practice ``{"trex": "tracks_raw/trex"}`` on a dataset
    converted before the relocation. Empty on a current one.

    Two callers need this and want opposite things from it. A raw-tracks scan
    wants to know that tracker output is sitting *inside* a source root, where
    its exclusion cannot reach; a sweeper wants to know that a root it is about
    to delete under holds user content, and to decline. Both questions are "which
    roots did not move", so they are answered once here rather than by two
    spellings of the same string comparison.
    """
    return {
        key: declared
        for key, declared in roots.items()
        if key in TRACKING_ROOTS
        and declared
        and TRACKING_ROOT not in PurePosixPath(str(declared).replace("\\", "/")).parts
    }


def new_dataset_manifest(
    name: str,
    base_dir: str | Path,
    roots: dict[str, str | Path] = default_roots,
    version: str = "0.1.0",
    index_format: str = "group/sequence",
    outfile: str | Path | None = None,
    # Continuous dataset support
    dataset_type: str = "discrete",  # "discrete" or "continuous"
    segment_duration: str | None = None,  # e.g., "1H", "30min", "1D"
    time_column: str | None = None,  # column name for timestamps, e.g., "timestamp"
) -> Path:
    """
    Create a minimal, extensible dataset manifest (YAML) with only a few required fields.
    - name: dataset name (e.g., "CALMS21")
    - base_dir: absolute or relative base directory for the dataset
    - roots: dict of subpaths you actually use NOW (e.g., {"media": "videos", "features": "features", "labels": "labels"})
    - index_format: how you think about addressing items ("group/sequence" is recommended)
    Returns the path to the created YAML.
    """
    base_dir = Path(base_dir).resolve()
    # Normalize roots -> relative paths (portable) when inside base_dir
    norm_roots = {}
    for k, v in roots.items():
        full = (base_dir / Path(v)).resolve()
        full.mkdir(parents=True, exist_ok=True)
        try:
            norm_roots[k] = str(full.relative_to(base_dir))
        except ValueError:
            norm_roots[k] = str(full)  # outside base_dir, keep absolute

    manifest = {
        "name": name,
        "version": version,
        "uuid": str(uuid.uuid4()),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "index_format": index_format,  # recommended: "group/sequence"
        "roots": norm_roots,  # required minimal roots you actually use now
        "dataset_type": dataset_type,  # "discrete" (default) or "continuous"
        # You can append optional fields later without placeholders
    }

    # Add continuous-specific fields if applicable
    if dataset_type == "continuous":
        if segment_duration:
            manifest["segment_duration"] = segment_duration
        if time_column:
            manifest["time_column"] = time_column

    header_comment = """# ==========================================================
# DATASET MANIFEST (extensible YAML)
# Minimal required fields above; append optional fields below
#
# DIRECTORY STRUCTURE (roots):
#   Raw (external, immutable — do not modify):
#     media_raw/    Original uploaded videos (may live on NAS)
#     tracks_raw/   Original tracking files from external tools
#     labels/       Ground truth: behavior labels, keypoints, individual IDs
#   Derived (computed by mosaic, regenerable):
#     media/        Derived media: low-res copies, re-encoded, thumbnails
#       frames/     Extracted video frames (PNGs), organised by method/run_id
#     tracks/       Standardised parquet tracks (converted from tracks_raw)
#     features/     Per-sequence feature parquets (wavelets, projections, embeddings)
#     models/       Trained models, reports, plots
#
# DATASET TYPES:
#   dataset_type: "discrete"     # Default: distinct recordings (trials, sessions)
#   dataset_type: "continuous"   # Long continuous recordings (days/months)
#     segment_duration: "1H"     # Segment size for continuous (e.g., "1H", "30min", "1D")
#     time_column: "timestamp"   # Column name for time-based operations
#
# Common OPTIONAL fields you may add later:
#   fps_default: 30.0
#   resolution_default: [1920, 1080]
#   n_animals_default: 2
#   species: ""
#   groups:                      # [{id, notes, condition, date, ...}]
#   sequences:                   # [{id, group, media_path, pose_path, fps, n_frames, n_animals, ...}]
#   splits:                      # {task1_train: [...], task1_test: [...], ...}
#   labels_map:                  # {0: attack, 1: investigation, ...}
#   skeleton:                    # [[p1, p2], ...]
#   bodyparts:                   # ["snout","neck",...]
#   processing:                  # [{step, time, params_hash, code_commit, ...}]
#   pose_model:                  # {name, engine, checkpoint, config}
#   behavior_model:              # {name, checkpoint, config}
#   provenance:                  # {repo, commit, env}
#   quality:                     # {missing_rate, drift, ...}
#   modalities:                  # ["video","pose","audio",...]
#   cameras:                     # {cam0: {intrinsics:..., extrinsics:...}, ...}
#   notes: |
#     Free-form notes about the dataset.
# ==========================================================
"""

    text = header_comment + yaml.safe_dump(
        manifest, sort_keys=False, default_flow_style=False
    )

    if outfile is None:
        outfile = base_dir / "dataset.yaml"
    else:
        outfile = Path(outfile)

    outfile.write_text(text, encoding="utf-8")
    print(f"Wrote dataset manifest -> {outfile}")
    return outfile


# --------------------------
# Dataset manifest + manager
# --------------------------


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

    A ``media_facts`` cell written before the current identity fields raises
    ``TypeError`` from :func:`row_to_facts`; convert it to the
    :class:`~mosaic_media.MediaProbeError` callers catch, naming the entry and
    the remedy. The ``try`` wraps only the reconstruction call, so an unrelated
    ``TypeError`` is never masked.
    """
    try:
        return row_to_facts(row_mapping(drow))
    except TypeError as exc:
        message = (
            f"entry {group}/{sequence} has a derivative row whose stored facts "
            "predate the current identity fields; re-probe the media index"
        )
        raise MediaProbeError(message) from exc


@dataclass(frozen=True)
class ProbedEntry:
    """One probed media file or imgstore, before identity is assigned.

    Produced by :meth:`Dataset._probe_dir_rows` and consumed by both
    :meth:`Dataset.index_media` (scan-and-derive) and
    :meth:`Dataset.write_media_index` (assignment-driven). ``camera`` and
    ``sync_uuid`` are store *facts* read from imgstore metadata (empty for a
    plain video); deriving the sequence identity from them is the caller's job,
    so this stays identity-free.

    ``origin`` says where ``probe`` came from, and it is load-bearing rather than
    diagnostic: item 5.2's drift check cannot tell a file whose bytes moved from a
    caller injecting a measurement of a different file without it. Last, with a
    default, so the imgstore site's keyword arguments are unaffected.
    """

    path: Path
    stat: os.stat_result
    probe: ProbeMetadata
    media_type: str
    camera: str = ""
    sync_uuid: str = ""
    origin: MeasurementOrigin = "probed"


@dataclass(frozen=True)
class ResolvedMedia:
    """Resolved media file paths for one (group, sequence), plus stored facts.

    ``paths`` are ordered by ``video_order`` (one element for a single-file
    sequence). ``facts`` is a parallel list of
    :class:`~mosaic_media.MediaFacts`, one per path, that readers inject instead
    of re-probing.
    """

    paths: list[Path]
    facts: list[MediaFacts]


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
    different file than the row described. ``drift`` is item 5.2's check -- a
    file whose bytes moved under a stable path, found because the write re-probed
    it and still held what the row recorded. The two are kept apart because they
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


@dataclass
class Dataset:
    manifest_path: Path
    name: str = "unnamed"
    version: str = "0.1"
    format: str = "yaml"
    roots: Dict[str, str] = field(
        default_factory=lambda: {
            "media_raw": "",
            "media": "",
            "tracks_raw": "",
            "tracks": "",
            TRACKING_ROOT: "",
            **{key: "" for key in TRACKING_ROOTS},
            "features": "",
            "labels": "",
            "models": "",
            "frames": "",
        }
    )
    meta: Dict[str, Any] = field(default_factory=dict)
    _path_map: list[tuple[Path, Path]] = field(
        default_factory=list, init=False, repr=False
    )

    # Continuous dataset support
    dataset_type: str = "discrete"  # "discrete" or "continuous"
    segment_duration: str | None = None  # e.g., "1H", "30min", "1D"
    time_column: str | None = None  # column name for timestamps

    # Manifest identity: written once by new_dataset_manifest and preserved
    # across a load -> save round-trip (save() dropping these is why callers
    # that need them stable had to avoid save() entirely).
    uuid: str | None = None
    created_at: str | None = None
    index_format: str | None = None

    def __post_init__(self) -> None:
        # Normalize manifest_path: callers may pass a str (e.g. from
        # os.path.join). Coercing to Path keeps methods like save() — which
        # call self.manifest_path.write_text(...) — working regardless of
        # whether a str or Path was supplied.
        self.manifest_path = Path(self.manifest_path)

    @property
    def is_continuous(self) -> bool:
        """Check if this is a continuous recording dataset."""
        return self.dataset_type == "continuous"

    # ---- Instance load method ----
    def load(self, ensure_roots: bool = True) -> "Dataset":
        """Load dataset metadata from self.manifest_path."""
        mp = Path(self.manifest_path)

        if mp.is_dir():
            # allow passing a dataset directory instead of a file
            for cand in ("dataset.yaml", "dataset.yml", "dataset.json"):
                candp = mp / cand
                if candp.exists():
                    mp = candp
                    break
            else:
                raise FileNotFoundError(f"No manifest found in directory: {mp}")

        if not mp.exists():
            raise FileNotFoundError(mp)

        if mp.suffix.lower() in (".yaml", ".yml"):
            if not _YAML_OK:
                raise RuntimeError("pyyaml not installed but manifest is YAML.")
            data = yaml.safe_load(mp.read_text())
            fmt = "yaml"
        elif mp.suffix.lower() == ".json":
            data = json.loads(mp.read_text())
            fmt = "json"
        else:
            # fallback: try yaml then json
            if _YAML_OK:
                try:
                    data = yaml.safe_load(mp.read_text())
                    fmt = "yaml"
                except Exception:
                    data = json.loads(mp.read_text())
                    fmt = "json"
            else:
                data = json.loads(mp.read_text())
                fmt = "json"

        # overwrite instance fields
        self.name = data.get("name", self.name)
        self.version = str(data.get("version", self.version))
        self.format = data.get("format", fmt)
        self.roots = _backfill_tracking_roots(data.get("roots", self.roots))
        self.meta = data.get("meta", self.meta)

        # Continuous dataset fields
        self.dataset_type = data.get("dataset_type", "discrete")
        self.segment_duration = data.get("segment_duration", None)
        self.time_column = data.get("time_column", None)

        # Manifest identity (preserved across the round-trip; absent in older
        # manifests, in which case the attribute stays None and save() omits it).
        self.uuid = data.get("uuid", self.uuid)
        self.created_at = data.get("created_at", self.created_at)
        self.index_format = data.get("index_format", self.index_format)

        if ensure_roots:
            self._ensure_roots()
        return self

    def save(self) -> None:
        """Persist manifest."""
        self._ensure_roots()
        payload: dict[str, object] = {
            "name": self.name,
            "version": self.version,
            "format": self.format,
            "roots": self.roots,
            "meta": self.meta,
            "dataset_type": self.dataset_type,
        }
        # Preserve manifest identity when present (a load->save round-trip must
        # not drop the uuid / created_at / index_format seeded at creation).
        if self.uuid:
            payload["uuid"] = self.uuid
        if self.created_at:
            payload["created_at"] = self.created_at
        if self.index_format:
            payload["index_format"] = self.index_format
        # Only include continuous-specific fields if set
        if self.segment_duration:
            payload["segment_duration"] = self.segment_duration
        if self.time_column:
            payload["time_column"] = self.time_column

        if self.format == "json":
            self.manifest_path.write_text(json.dumps(payload, indent=2))
        else:
            if not _YAML_OK:
                raise RuntimeError(
                    "pyyaml not installed; set format='json' or install pyyaml."
                )
            self.manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))

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
            KeyError: If *key* is not set in the manifest.
        """
        if key not in self.roots or not self.roots[key]:
            raise KeyError(
                f"Root '{key}' is not set in manifest. "
                f"Available roots: {list(self.roots.keys())}"
            )
        p = Path(self.roots[key])
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

        Args:
            key: Root name (e.g. ``"media_raw"``, ``"tracks"``).
            path: Directory path (absolute or relative to dataset root).
        """
        self.roots[key] = str(Path(path))
        self._ensure_roots()

    def _ensure_roots(self) -> None:
        for p in self.roots.values():
            if p:
                path = Path(p)
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
            current = Path(raw_path).expanduser()
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
        p = Path(str(path).strip())
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
        p = Path(str(stored_path).strip())
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
        return self._relative_to_root(Path(path))

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
            df = pd.read_csv(idx_path)
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
                    remapped = _remap_single_path(Path(str(p)), normalized)
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
            df = pd.read_csv(idx_path)
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
                    # files[].path -> filename only (they're siblings of run_info.json)
                    for f in data.get("files", []):
                        if "path" in f:
                            p = Path(f["path"])
                            if p.is_absolute():
                                f["path"] = p.name
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

    def _warn_superseded_entries(self, before: dict[tuple[str, str], str]) -> None:
        """Say when a conversion left older rows for the same tables behind.

        A converter that changes how it spells an entry writes rows under the new
        names without touching the old ones, so both resolve and every feature
        runs over each sequence twice. That is the visible consequence of
        ``calms21_npy`` 0.2, which stopped spelling its ids as slash paths.

        Detected as "an entry that was here before this call and that this call
        did not rewrite" -- so a normal re-conversion, which rewrites the same
        names, says nothing. Reported rather than repaired: deleting tables this
        call did not write is exactly the rename M1's migration rule forbids.
        """
        after = self._entry_stamps()
        gone = sorted(
            entry
            for entry, stamp in before.items()
            if after.get(entry) == stamp  # unchanged => this call did not touch it
        )
        if not gone:
            return
        listing = ", ".join(f"({g!r}, {s!r})" for g, s in gone[:5])
        more = f" and {len(gone) - 5} more" if len(gone) > 5 else ""
        print(
            f"[convert_all_tracks] {len(gone)} entr"
            f"{'y' if len(gone) == 1 else 'ies'} in tracks/index.csv were not "
            f"rewritten by this conversion: {listing}{more}.\n"
            "  If a converter changed how it spells its entries, these are the "
            "old spellings and both will resolve until you remove them:\n"
            "    ds.drop_entries([...], delete_files=True)",
            file=sys.stderr,
        )

    def drop_entries(
        self,
        entries: Iterable[tuple[str, str]],
        *,
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
            entries: The ``(group, sequence)`` pairs to drop.
            delete_files: Also unlink each row's parquet. Off by default -- an
                orphaned table is recoverable, a deleted one is not.
            run_id: Drop only this variant's rows for those entries. ``None``,
                the default, drops every variant -- which is what a rename
                cleanup wants, and what this did when an entry could only have
                one row. Name one to retire a single recipe and keep the rest,
                which is also how an entry stops being ambiguous.

        Returns:
            How many index rows were dropped.
        """
        wanted = {(str(g), str(s)) for g, s in entries}
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
        from .pipeline.index import feature_index, feature_index_path

        if not self.roots.get("features"):
            return {}
        fp = self.get_root("features")
        if not fp.exists():
            return {}

        if feature is not None:
            names = [feature]
        else:
            names = sorted(sub.name for sub in fp.iterdir() if sub.is_dir())

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

    def list_groups(self) -> list[str]:
        """The group names in ``tracks/index.csv``, sorted. Empty when there are none.

        Absent and empty answer alike: both mean "this dataset has no
        standardized tracks", and answering them differently is what left six
        readers of this file with four different policies. A caller that wants to
        tell a human to convert first checks for an empty result -- that check is
        the same for both spellings.
        """
        return sorted({str(g) for g in read_tracks_index(self)["group"]})

    def list_sequences(self, group: str | None = None) -> list[str]:
        """The sequences in ``tracks/index.csv``, optionally within one group.

        Empty when there are none; see :meth:`list_groups` on why absent is not
        an error.
        """
        df = read_tracks_index(self)
        if group is not None:
            df = df[df["group"] == group]
        return sorted({str(s) for s in df["sequence"]})

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
        ``run_feature(entries=[(group, sequence), ...])``.

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
        can be passed straight to ``run_feature(entries=...)`` to run a feature
        over exactly that subset.

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
        cached_facts_by_path: Mapping[Path, MediaFacts] | None = None,
    ) -> list[ProbedEntry]:
        """Probe every media file + imgstore under *search_dirs* (identity-free).

        Returns one :class:`ProbedEntry` per entry -- plain video files first
        (deterministically ordered by resolved path), then one entry per
        imgstore directory (sorted). An imgstore entry carries its Motif
        ``camera`` (= ``camera_serial``) and ``sync_uuid``
        (= ``synchronizationuuid``) read from store metadata; a plain video
        carries neither. Identity assignment (group/sequence) is left to the
        caller. Shared by :meth:`index_media` (scan-and-derive) and
        :meth:`write_media_index` (assignment-driven scope re-probe).

        *facts_by_name* maps a plain video's basename to an already-measured
        :class:`~mosaic_media.MediaFacts` -- a caller's own measurement, which
        always wins; a file present in it has its row built from those facts and
        is never probed. *cached_facts_by_path* maps a resolved source path to a
        prior row's measurement, reused only for a file with no injected facts;
        the caller admits an entry only when the file's size and mtime still
        match what that row recorded. Everything else is probed;
        :meth:`index_media` passes neither map, so it probes every file. The
        precedence is injection, then cache, then probe.
        """
        facts_map = facts_by_name or {}
        cache = cached_facts_by_path or {}
        from .media.imgstore_io import (
            imgstore_probe,
            imgstore_store_identity,
            is_imgstore,
        )

        search = [Path(d) for d in search_dirs]

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
                if is_under_tracking_root(cand.parts):
                    continue
                if cand.is_dir() and is_imgstore(cand):
                    imgstore_dirs.add(cand.resolve())

        # Serial glob: collect (path, stat) probe_candidates only. Probing
        # (ffprobe / MediaFacts, I/O bound) happens afterward through a bounded
        # thread pool so many-file search dirs index in parallel.
        probe_candidates: list[tuple[Path, os.stat_result]] = []
        tracking_skipped = 0
        for d in search:
            if not d.exists():
                print(f"[WARN] search dir missing: {d}", file=sys.stderr)
                continue
            it = d.rglob("*") if recursive else d.glob("*")
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
                if p.name not in facts_map and p.resolve() not in cache
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
                cached = cache.get(p.resolve())
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
                st = store_dir.stat()
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
                    st,
                    probe,
                    "imgstore",
                    camera=identity.camera_serial,
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
    ) -> Path:
        """
        Scan search_dirs for media files with given extensions and write an index CSV into media root.
        - No symlinks created; absolute paths recorded.
        - imgstore directories (Motif / Loopbio) are discovered natively: each
          store becomes one entry (``media_type="imgstore"``). The cameras of one
          synchronized recording (a shared Motif ``synchronizationuuid``) collapse
          into a single sequence with one ``camera`` row per store; each store's
          internal chunk files are excluded from the plain file glob.
        - Columns: name, group, sequence, group_safe, sequence_safe, camera,
          sync_uuid, abs_path, size_bytes, mtime_iso, width, height, fps, codec,
          media_type, frame_count, analysis_transcode, stream_transcode,
          analysis_derivative_path, playback_derivative_path, source_path,
          media_facts, video_order. ``camera`` is the within-sequence camera
          axis (``""`` for single-camera media) and ``sync_uuid`` the recording
          id that groups a recording's cameras. ``media_facts`` is the full
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
        if sequence_match_mode not in {"exact", "prefix"}:
            raise ValueError(
                f"sequence_match_mode must be 'exact' or 'prefix', got '{sequence_match_mode}'"
            )

        media_root = self.get_root(self.resolve_media_root())
        out_csv = media_root / index_filename
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
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
            # When no track match, use the stem as sequence so each entry is its
            # own sequence (not all lumped together under an empty key).
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
        rows.extend(
            self._imgstore_rows(imgstore_entries, seq_key_map, sequence_match_mode)
        )

        # De-duplicate by stored path.
        seen: set[object] = set()
        dedup: list[dict[str, object]] = []
        for r in rows:
            k = r["abs_path"]
            if k in seen:
                continue
            seen.add(k)
            dedup.append(r)

        # Carry transcode-written derivative links forward across the reindex,
        # then densify video_order per (group, sequence, camera) -- the same
        # ranking write_media_index uses, so a scan and an assignment-driven
        # write number identically.
        #
        # A rescan is not a session, so there are no session positions; but the
        # prior order is read, not discarded. Passing an empty prior map made
        # every row an unknown-order prior, which sorts by *name* -- so scanning
        # a corpus whose order came from an arranged write silently permuted it,
        # and the media composition hash (item 4.4) computed from that order
        # moved with no content change. The lookup key is
        # (group, sequence, basename) and a rescan re-derives the same
        # (group, sequence) for the same file, so the keys match. They miss
        # legitimately when the tracks keymap changed between scans and a file's
        # sequence name moved with it: the prior order genuinely no longer
        # describes that sequence, and falling back to name is correct there.
        #
        # Scanning and probing above is the expensive, read-only phase and is
        # deliberately unlocked; from here it is in-memory work plus one terminal
        # ``atomic_write``, which is the shape ``index_lock`` requires.
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with index_lock(out_csv):
            prior_order = build_prior_order(_read_media_index(out_csv))
            self._carry_forward_derivative_links(dedup, out_csv)
            densify_video_order(dedup, session_positions={}, prior_order=prior_order)
            df_out = frame_from_rows(dedup)
            write_media_index_rows(out_csv, df_out)
        self._write_media_compositions(dedup)

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
    def _store_base_name(entry: ProbedEntry, *, strip_camera: bool) -> str:
        """The sequence base name of an imgstore directory.

        For a synchronized recording (*strip_camera* true) the shared sequence
        name is the dir name minus its ``.<camera_serial>`` suffix
        (``sound_2_20210930_172126.23739656`` -> ``sound_2_20210930_172126``) --
        never ``Path.stem``, which would strip a dotted serial like a file
        extension. An unsynchronized store keeps its **full** dir name so two
        separate recordings that happen to share a base (``rec.A`` / ``rec.B``,
        different or absent sync ids) never collapse into one bogus 2-camera
        sequence.
        """
        if strip_camera and entry.camera:
            return entry.path.name.removesuffix("." + entry.camera)
        return entry.path.name

    def _imgstore_rows(
        self,
        entries: list[ProbedEntry],
        seq_key_map: dict[str, list[dict[str, str]]],
        sequence_match_mode: str,
    ) -> list[dict[str, object]]:
        """Build media-index rows for imgstore stores, grouping cameras.

        Stores sharing a non-empty ``sync_uuid`` are one recording -> one
        sequence with a ``camera`` row per store; an unsynced store forms its own
        singleton sequence. The sequence name is each store's base name
        (:meth:`_store_base_name`); all cameras of a recording reduce to the same
        base, and a divergent base warns and falls back to the deterministic
        minimum. The canonical base is matched once against the track keymap so a
        keymap hit assigns every camera the same ``(group, sequence)``. Emission
        is deterministic: groups and their member stores sort by
        ``(sync_uuid, camera_serial, path)``.
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
                members, key=lambda e: (e.sync_uuid, e.camera, str(e.path))
            )
            # A non-empty group key is a shared sync_uuid: these stores are one
            # recording's cameras, so strip the serial to their shared base name.
            synced = bool(members[0].sync_uuid)
            bases = {self._store_base_name(e, strip_camera=synced) for e in members}
            base = min(bases)
            if synced and len(bases) > 1:
                print(
                    f"[index_media] imgstore recording {members[0].sync_uuid!r} "
                    f"has cameras with divergent base names {sorted(bases)}; "
                    f"using {base!r}.",
                    file=sys.stderr,
                )
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
                        camera=entry.camera,
                        sync_uuid=entry.sync_uuid,
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

    def write_media_index(
        self,
        scopes: Iterable[MediaIndexScope],
        *,
        extensions: Tuple[str, ...] = (".mp4", ".avi"),
        index_filename: str = "index.csv",
        recursive: bool = True,
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
        """
        media_root = self.get_root(self.resolve_media_root())
        index_path = media_root / index_filename
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
        scope_list = list(scopes)
        scope_dirs = [scope.directory.resolve() for scope in scope_list]

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
        prior_row_by_path: dict[Path, dict[str, str]] = {}
        cached_facts_by_path: dict[Path, MediaFacts] = {}
        for row in existing:
            if not self._row_under_dirs(row, scope_dirs):
                continue
            stored = read_link_cell(row, "abs_path")
            if not stored:
                continue
            resolved = self.resolve_path(stored).resolve()
            prior_row_by_path[resolved] = dict(row)
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
            cached_facts_by_path[resolved] = facts

        disagreements: list[MediaIndexDisagreement] = []
        drift: list[MediaDrift] = []

        # Probe each scope directory and assign its explicit identity; collect
        # this session's arranged positions keyed (group, sequence, basename).
        fresh: list[dict[str, object]] = []
        session_positions: dict[tuple[str, str, str], int] = {}
        for scope in scope_list:
            group_safe = to_safe_name(scope.group) if scope.group else ""
            sequence_safe = to_safe_name(scope.sequence)
            for entry in self._probe_dir_rows(
                [scope.directory],
                exts,
                recursive,
                scope.facts_by_name,
                cached_facts_by_path,
            ):
                # A probed imgstore supplies its own camera/sync_uuid from store
                # metadata (read once in _probe_dir_rows); scope.camera is only
                # an override for a plain video the caller tags with a camera.
                #
                # The same inequality means two different things, and only
                # `entry.origin` tells them apart. Against an INJECTED
                # measurement it is a caller describing a different file than the
                # row did -- an ordinary re-upload, which is what
                # MediaIndexDisagreement has always reported. Against a PROBED
                # one it is the bytes having moved under a stable path, which is
                # drift (item 5.2). A CACHED measurement *is* the stored one, so
                # comparing it could only ever produce a false positive.
                prior = prior_row_by_path.get(entry.path.resolve())
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
                        camera=entry.camera or scope.camera,
                        sync_uuid=entry.sync_uuid,
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

        # --- Commit phase: cheap, and locked. ---
        #
        # Re-read authoritatively inside the lock rather than reusing the probe
        # phase's snapshot: another writer may have landed rows while this one
        # was probing, and they are exactly the rows this write must preserve.
        # Everything from here is in-memory work plus one terminal
        # ``atomic_write`` -- the shape ``index_lock`` requires, since a rename
        # over the path drops the block's grip on the inode it flocked.
        #
        # ``_read_media_index`` uses ``csv.DictReader`` and yields ``[]`` for the
        # zero-byte file the lock's ``O_CREAT`` leaves on a first write. A pandas
        # reader moved inside this block would raise ``EmptyDataError`` there.
        index_path.parent.mkdir(parents=True, exist_ok=True)
        touched = {(scope.group, scope.sequence) for scope in scope_list}
        with index_lock(index_path):
            committed = _read_media_index(index_path)
            prior_order = build_prior_order(committed)
            preserved = [
                row for row in committed if not self._row_under_dirs(row, scope_dirs)
            ]

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
        self._write_media_compositions(merged)
        for moved in drift:
            print(
                f"[write_media_index] drift: {moved.stored_path} "
                f"({moved.change}) recorded={moved.recorded_uuid or '-'} "
                f"measured={moved.measured_uuid or '-'}",
                file=sys.stderr,
            )
        return MediaIndexResult(
            index_path=index_path, disagreements=disagreements, drift=drift
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
        the label and returns the sequence to being called by its token.
        """
        path = sequence_label_path(self)
        sequence_labels(path).append(
            [
                SequenceLabelRow(
                    group=validate_entry_name(group, "group"),
                    sequence=validate_entry_name(sequence, "sequence"),
                    display_group=display_group,
                    display_name=name,
                )
            ]
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
        """Keep one row per stored ``abs_path``, first occurrence winning.

        Two :class:`MediaIndexScope`s over one directory each probe the same
        files and each stamp their own ``(group, sequence)``, so the same file
        lands twice under two identities. Reported to stderr rather than
        collapsed silently: the scopes come from a caller who believes they are
        disjoint, and nothing else would tell them otherwise.
        """
        seen: dict[str, dict[str, object]] = {}
        kept: list[dict[str, object]] = []
        for row in rows:
            stored = str(row.get("abs_path", ""))
            first = seen.get(stored)
            if first is not None:
                print(
                    f"[write_media_index] {stored} is under two scopes "
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
        """Read the media index as string-cell records (empty list if absent)."""
        media_root = self.get_root(self.resolve_media_root())
        return _read_media_index(media_root / index_filename)

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
        # symlink would make one file look like two indexes, and `index_lock` is
        # re-entrant per resolved path, so a run that took "both" locks would
        # hold one and perform two writes inside it.
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
                # `entry` and `allow_hardware` are both HASH_EXCLUDE, so any
                # entry yields the recipe every current run of this target would
                # name its output after.
                TranscodeParams(entry=("", ""), target=target),
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

    def _row_under_dirs(self, row: Mapping[str, object], dirs: list[Path]) -> bool:
        """True if *row*'s resolved ``abs_path`` lives under any of *dirs*.

        Resolve the stored path first (it may be root-relative), then test
        containment -- the resolver decoupling that keeps the check correct
        whether ``abs_path`` is stored relative or absolute.
        """
        abs_cell = str(row.get("abs_path", "") or "").strip()
        if not abs_cell:
            return False
        resolved = self.resolve_path(abs_cell).resolve()
        for directory in dirs:
            try:
                _ = resolved.relative_to(directory)
                return True
            except ValueError:
                continue
        return False

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

        Matches in the same order as the historical resolver: direct
        (group, sequence), then safe-name, then a filename-stem substring
        fallback. When *camera* is given, the matched rows are further filtered
        to that camera (``""`` selects the blank-camera rows), and an empty
        result after that filter returns ``None``. Returns ``None`` when nothing
        matches.
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
            stem = Path(sequence).name.lower()
            df = df.copy()
            df["name_lower"] = df["name"].astype(str).str.lower()
            candidates = df[df["name_lower"].str.contains(stem, na=False)]
            if candidates.empty:
                return None
            matched = candidates

        if camera is not None:
            matched = matched[matched["camera"].fillna("") == camera]
            if matched.empty:
                return None
        return matched.sort_values("video_order")

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
        df = pd.read_csv(idx_path)
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
        direct ``(group, sequence)``, then safe-name, then filename-stem
        substring -- **without** applying transcode-verdict routing (unlike
        :meth:`resolve_media`). *camera*, when given, further restricts the match
        to one camera of a multi-camera recording. The transcode job needs the
        originals, not their derivatives.

        Raises:
            FileNotFoundError: If the index is missing/empty or no row matches.
        """
        df = self._load_media_index(index_filename)
        if df.empty:
            raise FileNotFoundError("Media index is empty.")
        matched = self._match_media_rows(df, group, sequence, camera)
        if matched is None:
            raise FileNotFoundError(
                f"No media file found matching sequence '{sequence}'."
            )
        return matched

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
        An original carrying no uuid resolves through pass 1 or raises. Raises
        :class:`~mosaic_media.MediaProbeError` when the derivative file, its row,
        or its stored facts cannot be found.
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

        target = derivative_path.resolve()
        target_name = derivative_path.name

        # Pass 1: exact resolved-file match on abs_path.
        for _, drow in derivative_df.iterrows():
            abs_cell = _media_cell(drow, "abs_path")
            if abs_cell and self.resolve_path(abs_cell).resolve() == target:
                return _facts_or_stale_probe_error(drow, group, sequence)

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

    def media_routing_context(
        self, index_filename: str = "index.csv"
    ) -> tuple[bool, "pd.DataFrame | None"]:
        """Load the shared context for routing media-index rows by transcode verdict.

        Returns ``(route_derivatives, derivative_df)``:

        * ``route_derivatives`` is ``True`` only when a distinct ``media_raw``
          root holds the originals (so the ``media`` root can hold derivatives to
          route to); a legacy ``media``-only dataset yields ``False``.
        * ``derivative_df`` is the ``media`` index (one row per derivative) when
          it exists, else ``None``.

        Load once, then pass both to :meth:`route_media_row` for each row, so a
        batch of routings reads each index only once.
        """
        route_derivatives = self.resolve_media_root() == "media_raw"
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
        :class:`~mosaic_media.MediaProbeError` telling the caller to transcode
        first. A clean row whose stored measurement cannot be reconstructed
        raises the same way, telling the caller to re-probe the media index
        instead: transcoding and re-probing are distinct remedies for distinct
        problems, so the two failures stay textually distinct rather than
        collapsing into one message. There is no silent-degrade arm that opens
        the defective original, and there is no silent-degrade arm that opens a
        file without a measurement to hand the reader.

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
            raise MediaProbeError(message)

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

        Raises:
            FileNotFoundError: If the index is missing/empty or no row matches.
            MediaProbeError: If a row requires a transcode but has no derivative,
                a derivative's file/facts cannot be found, a matched row's
                stored measurement cannot be reconstructed, or *camera* is
                ``None`` while the sequence spans more than one camera.
        """
        matched = self.match_media_rows(group, sequence, camera, index_filename)
        if camera is None:
            cameras = {c for c in matched["camera"].fillna("").astype(str) if c}
            if len(cameras) > 1:
                raise MediaProbeError(
                    f"sequence ({group!r}, {sequence!r}) spans {len(cameras)} "
                    f"cameras {sorted(cameras)}; pass camera= to select one"
                )
        route_derivatives, derivative_df = self.media_routing_context(index_filename)
        return self._resolve_matched_rows(
            group, sequence, matched, route_derivatives, derivative_df
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
        for _, row in matched.iterrows():
            routed_path, routed_facts = self.route_media_row(
                group, sequence, row, route_derivatives, derivative_df
            )
            paths.append(routed_path)
            facts.append(routed_facts)
        return ResolvedMedia(paths=paths, facts=facts)

    def resolve_media_scope(
        self,
        groups: Iterable[str] | None,
        sequences: Iterable[str] | None,
        entries: Iterable[tuple[str, str]] | None = None,
        index_filename: str = "index.csv",
    ) -> list[ResolvedScopeEntry]:
        """Enumerate the scoped ``(group, sequence, camera)`` entries with media.

        Reads the originals index once, filters it to the given *groups* /
        *sequences* scope (either may be ``None`` to keep all), and returns one
        :class:`ResolvedScopeEntry` per distinct ``(group, sequence, camera)`` in
        deterministic order -- so the cameras of one recording become separate
        entries and are never concatenated into a single timeline. When *entries*
        is given, the scope is further restricted to rows whose
        ``(group, sequence)`` pair is in that set -- an explicit enumeration that
        pins an arbitrary subset even when sequence names repeat across groups
        (unlike the *groups*/*sequences* cross-product). Each entry's
        :class:`ResolvedMedia` carries its ``video_order``-sorted paths and
        stored facts, routed by transcode verdict exactly as
        :meth:`resolve_media` does. When an entry has an empty group and a
        sequence whose safe name is empty, the returned sequence label falls back
        to the first original file's stem.

        Raises:
            FileNotFoundError: If the originals index does not exist.
            MediaProbeError: If an entry requires a transcode but has no
                derivative, a derivative's file/facts cannot be found, or a
                matched row's stored measurement cannot be reconstructed.
        """
        # Media-index resolution is a Dataset concern, so a scoped enumeration lives
        # here as a method (mirroring resolve_media) rather than as a free function in
        # a consumer package: the read-and-route logic stays in one layer instead of
        # being duplicated upward.
        df = self._load_media_index(index_filename)

        mask = pd.Series(True, index=df.index)
        if groups is not None:
            mask &= df["group"].isin({str(g) for g in groups})
        if sequences is not None:
            mask &= df["sequence"].isin({str(s) for s in sequences})
        if entries is not None:
            wanted = {(str(group), str(sequence)) for group, sequence in entries}
            pairs = pd.MultiIndex.from_arrays([df["group"], df["sequence"]])
            mask &= pd.Series(pairs.isin(wanted), index=df.index)
        scoped = df[mask]

        route_derivatives, derivative_df = self.media_routing_context(index_filename)
        resolved_entries: list[ResolvedScopeEntry] = []
        for (group, sequence, camera), sub in scoped.groupby(
            ["group", "sequence", "camera"]
        ):
            group, sequence, camera = str(group), str(sequence), str(camera)
            sub = sub.sort_values("video_order")
            resolved = self._resolve_matched_rows(
                group, sequence, sub, route_derivatives, derivative_df
            )
            if not group and not to_safe_name(sequence):
                sequence = self.resolve_path(str(sub.iloc[0]["abs_path"])).stem
            resolved_entries.append(
                ResolvedScopeEntry(group, sequence, camera, resolved)
            )
        return resolved_entries

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
        exclude_patterns: Optional[Iterable[str]] = None,
        compute_md5: bool = True,
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
            Source format identifier (e.g., "trex_npz", "calms21_npy")
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
        out_csv = self.get_root("tracks_raw") / index_filename
        rows: list[TracksRawIndexRow] = []
        # Advisory, and read before the lock like the media probe phase's caches:
        # a stale entry costs a re-hash, never a wrong digest, because every hit
        # is re-validated against the file's current size and mtime.
        carried = self._prior_digests(out_csv) if compute_md5 else {}

        def _digest(path: Path) -> str:
            if not compute_md5:
                return ""
            return carried.get(path.resolve()) or _md5(path)

        pat_list = _normalize_patterns(patterns)
        exc_list = _normalize_patterns(exclude_patterns)
        group_re = re.compile(group_pattern) if group_pattern else None

        for p, st in iter_track_files(
            map(Path, search_dirs),
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
                if src_format == "trex_npz":
                    seq = strip_trex_seq(p.stem)
                else:
                    seq = p.stem  # 1 file ~= 1 sequence default

                # Extract group from sequence using pattern
                if group_re:
                    m = group_re.search(seq)
                    grp = m.group(1) if m else ""
                else:
                    grp = ""

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
        # alone, which is what makes this whole-file rewrite safe against a
        # concurrent one.
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df = _tracks_frame_from_rows(rows)
        with index_lock(out_csv):
            write_tracks_raw_index_rows(out_csv, df)
        self._write_tracks_raw_compositions(df.to_dict("records"))
        print(f"[index_tracks_raw] {len(df)} -> {out_csv}")
        return out_csv

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
        self, raw_row: pd.Series, params: Optional[dict] = None, overwrite: bool = False
    ) -> Path:
        """
        Convert a single raw track file (row from tracks_raw/index.csv) to standard trex_v1 parquet.
        Returns path to standardized file, updates tracks/index.csv.
        """
        params = params or {}
        std_fmt = self.meta.get("tracks", {}).get("standard_format", "trex_v1")
        src_format = str(raw_row["src_format"])
        src_path = self.resolve_path(raw_row["abs_path"])

        converter = get_track_converter(src_format)
        conv_params = self._converter_params(converter, params)
        # Minted and recorded once per recipe, not per entry, and carried onto
        # every row this call writes. Also names the directory the tables live in
        # and described in its params.json, so a variant is explicable from disk.
        variant = self._tracks_variant(converter, conv_params)
        producer = converter_op(src_format)

        # Where to place standardized file: one directory per tracks variant,
        # holding <group>__<seq>.parquet. Two recipes for one sequence no longer
        # collide, which is what the exists()/overwrite skips below used to hide.
        variant_root = tracks_variant_root(self.get_root("tracks"), variant)

        # If sequence missing/blank and the format can hold several, expand this
        # file into multiple per-sequence outputs
        raw_seq_val = raw_row.get("sequence", "")
        seq_value = "" if _is_empty_like(raw_seq_val) else str(raw_seq_val).strip()
        if (not seq_value) and converter.enumerable:
            # policy: 'infile' (default), 'filename', 'both'
            policy = str(params.get("group_from", "infile")).lower()
            if policy not in {"infile", "filename", "both"}:
                policy = "infile"

            raw_collection = (
                str(raw_row.get("group", "")) if raw_row is not None else ""
            )
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
                if policy in {"filename", "both"} and raw_collection:
                    out_group_canon = raw_collection

                # output path -- make_entry_key does the safe-name encoding
                stem = make_entry_key(out_group_canon, canon_seq)
                out_path = variant_root / f"{stem}.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Respect overwrite flag when outputs already exist
                if out_path.exists() and not overwrite:
                    produced.append(out_path)
                    continue

                # Which entry to produce -- passed beside the params, never
                # merged into them, so the sequence name cannot reach a digest.
                df_std = converter.convert(
                    src_path,
                    conv_params,
                    EntryHints(group=canon_group_infile, sequence=canon_seq),
                )

                # Overwrite group column in DataFrame to match policy
                if "group" in df_std.columns and out_group_canon != canon_group_infile:
                    df_std["group"] = out_group_canon

                # Ensure schema, then write
                _, _schema_report = ensure_track_schema(
                    df_std,
                    std_fmt,
                    strict=conv_params.strict_schema,
                    source=f"{src_path}::{canon_seq}",
                )
                df_std.to_parquet(out_path, index=False)

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
                    source_md5=str(raw_row.get("md5", "")),
                    consumed_source_roots=("tracks_raw",),
                )
                produced.append(out_path)

            return self.get_root("tracks") / "index.csv"

        # Normal single-sequence path (default). The safe names are not
        # computed here any more: make_entry_key derives them for the filename,
        # and the index stores the raw identity plus nothing derivable from it.
        group_value = str(raw_row.get("group", "")) or ""
        rel_name = f"{make_entry_key(group_value, seq_value)}.parquet"
        out_path = variant_root / rel_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            return out_path

        # Which entry to produce. The hints always win: the caller knows the
        # authoritative entry. (The old dict merged them with ``setdefault``
        # here and with direct assignment in the branch above, so a
        # user-supplied ``params["group"]`` won in one path and lost in the
        # other -- one of the reasons entry identity does not belong in params.)
        df_std = converter.convert(
            src_path,
            conv_params,
            EntryHints(
                group=str(raw_row.get("group", "")),
                sequence=str(raw_row.get("sequence", "")),
            ),
        )

        # Validate/coerce against the declared standard format schema (if any)
        _, _schema_report = ensure_track_schema(
            df_std, std_fmt, strict=conv_params.strict_schema, source=str(src_path)
        )

        df_std.to_parquet(out_path, index=False)

        # group/sequence come straight off a pandas Series here, so they can be
        # numpy scalars; write_tracks_row stringifies them, which is what keeps
        # the dedup that holds this index to one row per entry working.
        write_tracks_row(
            self,
            run_id=variant,
            group=raw_row.get("group", ""),
            sequence=raw_row["sequence"],
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
        return dict(TRACK_CONVERTERS)

    def _tracks_variant(
        self,
        converter: TrackConverter[TrackConvertParams],
        params: TrackConvertParams,
        observed: Optional[dict] = None,
    ) -> str:
        """Mint this conversion's tracks-variant identity and record it.

        Params-only and scope-free, so one value names one recipe however many
        sequences it covers. Recorded beside the tables in
        ``tracks/<run_id>/params.json``, which Stage 3.2 turns into the
        directory the parquets themselves live in.
        """
        cls = type(converter)
        op = converter_op(cls.src_format)
        payload = convert_variant_payload(params.identity_dump())
        run_id = tracks_run_id(op, cls.version, payload)
        _ = write_tracks_variant(
            self.get_root("tracks"),
            run_id,
            op,
            cls.version,
            params.identity_dump(),
            observed,
        )
        return run_id

    def _converter_params(
        self,
        converter: TrackConverter[TrackConvertParams],
        overrides: Optional[dict],
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
            fps_default = self.meta.get("fps_default")
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
    def convert_all_tracks(
        self,
        params: Optional[dict] = None,
        overwrite: bool = False,
        merge_per_sequence: Optional[bool] = None,
        group_from: Optional[str] = None,
    ) -> None:
        """
        Convert all raw track files (from tracks_raw/index.csv) to standard T-Rex-like parquet files.

        By default, for src_format == 'trex_npz', files are merged per (group, sequence) into a single
        parquet file (one per unique (group, sequence)). For other formats, or if merge_per_sequence=False,
        each row is converted individually.

        Parameters
        ----------
        params : dict | None
            Extra parameters to pass to converters.
        overwrite : bool
            If True, overwrite existing output files.
        merge_per_sequence : bool | None
            If True, merge per (group, sequence) for formats that support it (currently trex_npz).
            If None, defaults to True if all rows are trex_npz, else False.
        group_from : {'infile','filename','both'} | None
            Controls which *group* ends up in the standardized output & index:
            - 'infile' (default): use the group from inside the source file (e.g., 'annotator-id_0').
            - 'filename'   : use the raw file-level group hint from tracks_raw/index.csv (e.g., 'calms21_task1_test').
            - 'both'  : set output group to the raw file-level group, and still record in-file group in the data
                        (converters should already keep in-file columns; we always keep raw file-level hint in
                        the 'collection' column).
            If None, defaults to 'infile'.
        """
        raw_idx = self.get_root("tracks_raw") / "index.csv"
        if not raw_idx.exists():
            raise FileNotFoundError(
                "tracks_raw/index.csv not found; run index_tracks_raw first."
            )
        try:
            df = pd.read_csv(raw_idx)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"tracks_raw/index.csv is empty or malformed: {raw_idx}\n"
                "This usually means index_tracks_raw() found no matching files.\n"
                "Check your search_dirs and patterns parameters."
            )

        # Decide merging default for trex
        if merge_per_sequence is None:
            merge_per_sequence = len(df) > 0 and (df["src_format"] == "trex_npz").all()

        # normalize group_from
        group_from = (group_from or "infile").lower()
        if group_from not in {"infile", "filename", "both"}:
            raise ValueError(
                f"group_from must be one of 'infile', 'filename', 'both'; got {group_from}"
            )

        before = self._entry_stamps()

        if not merge_per_sequence:
            # Convert each row individually
            for _, row in df.iterrows():
                try:
                    call_params = dict(params) if params else {}
                    call_params["group_from"] = group_from
                    self.convert_one_track(row, params=call_params, overwrite=overwrite)
                except Exception as e:
                    print(f"[WARN] convert failed for {row.get('abs_path')}: {e}")
            self._warn_superseded_entries(before)
            return

        # Merge per (group, sequence, src_format)
        groupby_cols = ["group", "sequence", "src_format"]
        df = df.copy()
        for col in groupby_cols:
            if col not in df.columns:
                df[col] = ""
            df[col] = (
                df[col]
                .astype("string")
                .fillna("")
                .replace({"nan": "", "None": ""}, regex=False)
                .str.strip()
            )

        for keys, group_df in df.groupby(groupby_cols):
            group, sequence, src_format = keys

            # Non-mergeable formats -> fall back to individual conversion
            if src_format != "trex_npz":
                for _, row in group_df.iterrows():
                    try:
                        call_params = dict(params) if params else {}
                        call_params["group_from"] = group_from
                        self.convert_one_track(
                            row, params=call_params, overwrite=overwrite
                        )
                    except Exception as e:
                        print(f"[WARN] convert failed for {row.get('abs_path')}: {e}")
                continue

            # Merge TRex NPZ per (group, sequence)
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
            variant = self._tracks_variant(converter, conv_params)

            raw_group_hint = str(first_row.get("group", "")) or ""
            out_group = group  # default: infile (already what we grouped by)
            if group_from in {"filename", "both"} and raw_group_hint:
                out_group = raw_group_hint
            variant_root = tracks_variant_root(self.get_root("tracks"), variant)
            rel_name = f"{make_entry_key(out_group, sequence)}.parquet"
            out_path = variant_root / rel_name
            if out_path.exists() and not overwrite:
                continue

            hints = EntryHints(group=group or "", sequence=sequence or "")

            dfs = []
            _merge_failed = False
            for _, row in group_df.iterrows():
                src_path = self.resolve_path(row["abs_path"])
                try:
                    df_std = converter.convert(src_path, conv_params, hints)
                except Exception as e:
                    print(
                        f"[WARN] convert failed for {src_path}: {e}; "
                        f"skipping sequence ({group}, {sequence})"
                    )
                    _merge_failed = True
                    break
                dfs.append(df_std)
            if _merge_failed or not dfs:
                continue

            # Align columns across IDs
            all_cols = sorted(set().union(*[set(d.columns) for d in dfs]))
            aligned = []
            for d in dfs:
                missing = [c for c in all_cols if c not in d.columns]
                if missing:
                    for mc in missing:
                        d[mc] = np.nan
                aligned.append(d[all_cols])
            merged_df = pd.concat(aligned, ignore_index=True)
            ensure_track_schema(
                merged_df,
                "trex_v1",
                strict=False,
                source=f"{group}/{sequence} (merged)",
            )

            # Write output (out_path determined above for overwrite check)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged_df.to_parquet(out_path, index=False)

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
                std_format="trex_v1",
                n_rows=int(len(merged_df)),
                source=self.resolve_path(first_row["abs_path"]),
                source_md5=str(first_row.get("md5", "")),
                consumed_source_roots=("tracks_raw",),
            )

        self._warn_superseded_entries(before)

    # ----------------------------
    # Labels: conversion + indexing
    # ----------------------------
    def convert_all_labels(
        self,
        kind: str = "behavior",
        overwrite: bool = False,
        params: Optional[dict] = None,
        source_format: Optional[str] = None,
        **kwargs,
    ) -> None:
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
        **kwargs : additional keyword arguments
            Passed to converter (e.g., group_from, fps, etc.)

        Raises
        ------
        ValueError
            If no converter is registered for (source_format, kind) combination
        FileNotFoundError
            If tracks_raw/index.csv is missing

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

        # Look up converter in registry
        converter_key = (src_format, kind)
        if converter_key not in LABEL_CONVERTERS:
            available = list(LABEL_CONVERTERS.keys())
            raise ValueError(
                f"No label converter registered for (src_format='{src_format}', kind='{kind}'). "
                f"Available converters: {available}\n"
                f"To add support for a new format, create a converter in label_library/ "
                f"and import it in label_library/__init__.py"
            )

        # Instantiate converter
        converter_cls = LABEL_CONVERTERS[converter_key]
        converter = converter_cls(params=params, **kwargs)

        # Load raw index
        raw_idx = self.get_root("tracks_raw") / "index.csv"
        if not raw_idx.exists():
            raise FileNotFoundError(
                "tracks_raw/index.csv not found; run index_tracks_raw first."
            )

        df_raw = pd.read_csv(raw_idx)
        if "src_format" not in df_raw.columns:
            raise ValueError("tracks_raw/index.csv missing 'src_format' column.")
        df_raw = df_raw[df_raw["src_format"].astype(str) == str(src_format)]
        if df_raw.empty:
            raise ValueError(
                f"No rows in tracks_raw/index.csv with src_format='{src_format}'."
            )

        # Setup output directory
        labels_root = self.get_root("labels") / kind
        labels_root.mkdir(parents=True, exist_ok=True)
        idx_path = labels_root / "index.csv"
        _ensure_labels_index(idx_path)

        # Load existing pairs
        existing_pairs: set[tuple[str, str]] = set()
        if idx_path.exists():
            df_idx = pd.read_csv(idx_path)
            if not df_idx.empty:
                grouped = df_idx.get("group", pd.Series(dtype=str)).fillna("")
                seqs = df_idx.get("sequence", pd.Series(dtype=str)).fillna("")
                existing_pairs = set(zip(grouped.astype(str), seqs.astype(str)))

        # Convert each raw file using the converter
        new_rows: list[dict] = []
        for _, raw_row in df_raw.iterrows():
            src_path = self.resolve_path(raw_row["abs_path"])
            created = converter.convert(
                src_path=src_path,
                raw_row=raw_row,
                labels_root=labels_root,
                params=params,
                overwrite=overwrite,
                existing_pairs=existing_pairs,
            )
            if created:
                new_rows.extend(created)

        # Update index and metadata
        if new_rows:
            # Store dataset-root-relative abs_path so the labels index stays
            # portable across machines / synced datasets (converters emit
            # absolute paths).
            for _r in new_rows:
                if _r.get("abs_path"):
                    _r["abs_path"] = self.relative_to_root(_r["abs_path"])
            _append_labels_index(idx_path, new_rows)

            # Update metadata with converter's metadata
            labels_meta = self.meta.setdefault("labels", {})
            labels_meta[kind] = {
                "index": str(idx_path.resolve()),
                "label_format": converter.label_format,
                "updated_at": _now_iso(),
            }

            # Add format-specific metadata if converter provides it
            if hasattr(converter, "get_metadata"):
                labels_meta[kind].update(converter.get_metadata())

            try:
                self.save()
            except Exception:
                pass

        print(
            f"[convert_all_labels] kind={kind} wrote {len(new_rows)} sequences using {src_format} converter (overwrite={overwrite})."
        )

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
        - 'fps': float, frames per second
        - (optional) additional metadata

        See Also
        --------
        convert_all_labels : For standard converters registered in label_library
        load_labels : Load converted labels
        """
        kind = str(kind or "behavior").lower()

        # Setup output directory
        labels_root = self.get_root("labels") / kind
        labels_root.mkdir(parents=True, exist_ok=True)
        idx_path = labels_root / "index.csv"
        _ensure_labels_index(idx_path)

        # Load existing pairs to avoid duplicates
        existing_pairs: set[tuple[str, str]] = set()
        if idx_path.exists():
            df_idx = pd.read_csv(idx_path)
            if not df_idx.empty:
                grouped = df_idx.get("group", pd.Series(dtype=str)).fillna("")
                seqs = df_idx.get("sequence", pd.Series(dtype=str)).fillna("")
                existing_pairs = set(zip(grouped.astype(str), seqs.astype(str)))

        # Call the custom converter
        new_rows = converter_fn(
            dataset=self,
            labels_root=labels_root,
            existing_pairs=existing_pairs,
            overwrite=overwrite,
            **kwargs,
        )

        # Update index and metadata
        if new_rows:
            # Store dataset-root-relative abs_path (see convert_all_labels).
            for _r in new_rows:
                if _r.get("abs_path"):
                    _r["abs_path"] = self.relative_to_root(_r["abs_path"])
            _append_labels_index(idx_path, new_rows)

            # Update dataset metadata
            labels_meta = self.meta.setdefault("labels", {})
            labels_meta[kind] = {
                "index": str(idx_path.resolve()),
                "label_format": label_format,
                "updated_at": _now_iso(),
            }

            try:
                self.save()
            except Exception:
                pass

        print(
            f"[convert_labels_custom] kind={kind} wrote {len(new_rows)} sequences (overwrite={overwrite})."
        )
        return len(new_rows)

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
        idx_path = labels_root / "index.csv"
        _ensure_labels_index(idx_path)

        safe_group = to_safe_name(group) if group else ""
        safe_seq = to_safe_name(sequence)
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

        np.savez_compressed(out_path, **payload)

        row = {
            "kind": kind,
            "label_format": "id_tags_v1",
            "group": group,
            "sequence": sequence,
            "group_safe": safe_group,
            "sequence_safe": safe_seq,
            "abs_path": self._relative_to_root(out_path),
            "source_abs_path": "",
            "source_md5": "",
            "n_frames": len(id_keys),
            "label_ids": ",".join(map(str, id_keys)),
            "label_names": ",".join(field_names),
        }
        _append_labels_index(idx_path, [row])
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
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate required columns
        if "group" not in df.columns or "sequence" not in df.columns:
            raise ValueError("CSV must have 'group' and 'sequence' columns")

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
                group = str(row["group"]) if pd.notna(row["group"]) else ""
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

            # Group by (group, sequence)
            for (group, seq), group_df in df.groupby(["group", "sequence"]):
                group = str(group) if pd.notna(group) else ""
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

            # Group by (group, sequence)
            for (group, seq), group_df in df.groupby(["group", "sequence"]):
                group = str(group) if pd.notna(group) else ""
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
        groups: Optional[Iterable[str]] = None,
        sequences: Optional[Iterable[str]] = None,
    ) -> dict[tuple[str, str], dict]:
        """
        Load per-id labels for the requested kind.
        Returns {(group, sequence): {"labels": {id: {field: value}}, "sequence_safe": str, "path": str, "metadata": dict}}
        """
        labels_root = self.get_root("labels") / kind
        idx_path = labels_root / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Labels index not found for kind='{kind}': {idx_path}"
            )
        df = pd.read_csv(idx_path)
        if groups is not None:
            groups = {str(g) for g in groups}
            df = df[df["group"].fillna("").astype(str).isin(groups)]
        if sequences is not None:
            sequences = {str(s) for s in sequences}
            df = df[df["sequence"].fillna("").astype(str).isin(sequences)]
        result: dict[tuple[str, str], dict] = {}
        for _, row in df.iterrows():
            group = str(row.get("group", "") or "")
            sequence = str(row.get("sequence", "") or "")
            safe_seq = row.get("sequence_safe") or to_safe_name(sequence)
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

    def load_labels(self, group: str, sequence: str, kind: str = "behavior") -> dict:
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

        For backward compatibility with old dense formats, individual_ids may not be present.
        """
        labels_root = self.get_root("labels") / kind
        idx_path = labels_root / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Labels index not found for kind='{kind}': {idx_path}"
            )

        df = pd.read_csv(idx_path)
        df = df[(df["group"].fillna("") == group) & (df["sequence"] == sequence)]

        if len(df) == 0:
            raise ValueError(
                f"No labels found for group='{group}', sequence='{sequence}', kind='{kind}'"
            )

        if len(df) > 1:
            print(
                f"Warning: Multiple label entries found for ({group}, {sequence}). Using first."
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

    def get_label_map(self, kind: str = "behavior") -> dict[int, str]:
        """
        Get the label map {id: name} for a label kind.

        Reads from the labels index.csv (first row).
        """
        idx_path = self.get_root("labels") / kind / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Labels index not found for kind='{kind}': {idx_path}"
            )

        df = pd.read_csv(idx_path, nrows=1)
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
        convert_params: Optional[dict] = None,
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
        df_raw = pd.read_csv(raw_idx)
        hit = df_raw[
            (df_raw["group"].fillna("") == group) & (df_raw["sequence"] == sequence)
        ]
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
        groups: Iterable[str] | None = None,
        sequences: Iterable[str] | None = None,
        entries: Iterable[tuple[str, str]] | None = None,
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
        tracks_run_id: str | None = None,
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
            groups: Scope filter — restrict to these group names.
            sequences: Scope filter — restrict to these sequence names.
            entries: Scope filter — restrict to these explicit
                ``(group, sequence)`` pairs. Selects an arbitrary subset
                (unambiguous when sequence names repeat across groups), e.g. a
                tag-resolved set of sequences. Intersects with
                ``groups``/``sequences`` when those are also given.
            tracks_run_id: Which tracks variant the ``"tracks"`` input
                resolves to. ``None`` lets each entry resolve to whichever
                variant it has; name one when a sequence carries two recipes and
                resolution declines to choose.
            overwrite: Re-run even if outputs exist for this run_id.
            parallel_workers: When >1 and the feature declares itself
                parallelizable, run the apply phase in parallel.
            parallel_mode: ``'thread'`` (default) or ``'process'`` execution
                backend when *parallel_workers* > 1.
            overlap_frames: Extra frames from adjacent segments for
                edge-effect handling.  Mutually exclusive with frame/time
                filters.
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
            groups=groups,
            sequences=sequences,
            entries=entries,
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
            execution_id=execution_id,
            owner=owner,
            track=track,
            progress_callback=progress_callback,
            cancel_token=cancel_token,
        )


def _is_empty_like(x: Optional[Any]) -> bool:
    """True for None/NaN/''/'nan'/'none' (case-insensitive)."""
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    if isinstance(x, str):
        s = x.strip().lower()
        return s in ("", "nan", "none")
    return False


def _ensure_labels_index(idx_path: Path):
    if not idx_path.exists():
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "kind": pd.Series(dtype="string"),
                "label_format": pd.Series(dtype="string"),
                "group": pd.Series(dtype="string"),
                "sequence": pd.Series(dtype="string"),
                "group_safe": pd.Series(dtype="string"),
                "sequence_safe": pd.Series(dtype="string"),
                "abs_path": pd.Series(dtype="string"),
                "source_abs_path": pd.Series(dtype="string"),
                "source_md5": pd.Series(dtype="string"),
                "n_frames": pd.Series(dtype="Int64"),
                "label_ids": pd.Series(dtype="string"),
                "label_names": pd.Series(dtype="string"),
            }
        ).to_csv(idx_path, index=False)


def _append_labels_index(idx_path: Path, rows: list[dict]):
    if not idx_path.exists():
        _ensure_labels_index(idx_path)
    df = pd.read_csv(idx_path)
    for col in LABEL_INDEX_COLUMNS:
        fill = "" if col != "n_frames" else None
        df = ensure_text_column(df, col, "" if fill is None else fill)
    updated = df.copy()
    for r in rows:
        row = dict(r)
        row.setdefault("kind", "")
        row.setdefault("label_format", "")
        row.setdefault("group", "")
        row.setdefault("sequence", "")
        if "group_safe" not in row:
            row["group_safe"] = to_safe_name(row["group"]) if row["group"] else ""
        if "sequence_safe" not in row:
            row["sequence_safe"] = (
                to_safe_name(row["sequence"]) if row["sequence"] else ""
            )
        row.setdefault("abs_path", "")
        row.setdefault("source_abs_path", "")
        row.setdefault("source_md5", "")
        if "n_frames" not in row:
            row["n_frames"] = ""
        row.setdefault("label_ids", "")
        row.setdefault("label_names", "")
        mask = (updated["group"].fillna("") == row["group"]) & (
            updated["sequence"].fillna("") == row["sequence"]
        )
        updated = updated[~mask]
        updated = pd.concat([updated, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(idx_path, index=False)
