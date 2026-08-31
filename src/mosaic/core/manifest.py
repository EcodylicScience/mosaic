"""The dataset manifest: what ``dataset.yaml`` holds, and how it is read and written.

The manifest is the file that makes a directory a mosaic dataset. It names the
dataset, declares its :data:`roots`, declares the :class:`ScanSources` it draws
raw media, tracks and labels from, and carries the dataset's own notes and typed
tags.

**Roots live inside the dataset; sources deliberately do not.** A root holds that
root's own ``index.csv``, so a root outside the dataset would put the index
outside too and the dataset would stop being the thing you can copy, archive or
sync -- :func:`validate_root_inside` refuses one. A *source* is the opposite: it
exists to name storage elsewhere, its files are recorded by absolute ``abs_path``
from an index that is inside, and it is never created, never validated inside,
and never walked at load time.

**This module holds the format, and nothing else.** It takes no import from
:class:`~mosaic.core.dataset.Dataset`, so the format can be read, written, tested
and reviewed without the 5000-line orchestrator that uses it; which manifest a
dataset has, and what its roots resolve to, belongs there rather than here.

The dependency direction is the point, not the process footprint: importing
anything under ``mosaic.core`` still pulls pandas, because ``mosaic.core``'s own
``__init__`` eagerly imports ``Dataset`` and the track-converter registry. This
module adds nothing to that, and would be cheap on its own if that ever changed.

Not to be confused with :mod:`mosaic.core.pipeline.manifest`, which builds the
in-memory table of per-sequence work a feature run will do. The two share a word
and nothing else.
"""

from __future__ import annotations

import datetime
import json
import re
import uuid as uuid_module
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import ClassVar, Final, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from mosaic.core.json_value import JsonValue
from mosaic.core.strict_model import StrictModel
from mosaic.core.pipeline._utils import atomic_write
from mosaic.user_paths import user_path
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOT, TRACKING_ROOTS
from mosaic.core.typed_attribute import (
    TypedAttributeType,
    TypedAttributeValue,
    validate_constraints,
    validate_typed_value,
)

__all__ = [
    "MANIFEST_FILENAMES",
    "MANIFEST_VERSION",
    "NOTES_MAX_CHARS",
    "TAGS_MAX_COUNT",
    "TAG_DESCRIPTION_MAX_LENGTH",
    "TAG_NAME_MAX_LENGTH",
    "AnyScanSource",
    "DatasetManifest",
    "DatasetTag",
    "GroupFrom",
    "LabelsScanSource",
    "ManifestVersionError",
    "MediaLayout",
    "MediaScanSource",
    "RawScanSource",
    "ScanSource",
    "ScanSources",
    "SequenceMatchMode",
    "SourceKind",
    "SourceMode",
    "TracksScanSource",
    "backfill_roots",
    "default_roots",
    "empty_root_template",
    "legacy_tracking_roots",
    "manifest_header",
    "manifest_payload",
    "manifest_text",
    "migrate_to_current",
    "new_manifest",
    "now_stamp",
    "overlapping_sources",
    "read_manifest",
    "resolve_manifest_path",
    "validate_root_inside",
    "write_manifest",
]


MANIFEST_VERSION: Final = 2
"""The manifest format this code writes.

Bumped only for a change that older code would misread. Reading tolerates any
version at or below this one and migrates it in memory; a *higher* one raises
:class:`ManifestVersionError` rather than being read with the wrong rules.
"""

MANIFEST_FILENAMES: Final = ("dataset.yaml", "dataset.yml", "dataset.json")
"""Names probed, in order, when a dataset *directory* is handed to a reader."""

NOTES_MAX_CHARS: Final = 65_536
TAGS_MAX_COUNT: Final = 200
TAG_NAME_MAX_LENGTH: Final = 64
TAG_DESCRIPTION_MAX_LENGTH: Final = 500
SOURCE_ID_MAX_LENGTH: Final = 64


MediaLayout = Literal["stem", "per_sequence"]
"""How a media scan derives identity for a file no track table names.

``stem`` is the historical heuristic: the filename stem is the sequence, so a
multi-clip sequence re-derives as one sequence per clip. ``per_sequence`` reads
the declared layout -- ``<media_raw>/<entry key>/`` -- which is what the control
plane already writes.

``stem`` remains the default deliberately: ``match_mode="prefix"`` exists to
serve split recordings under the flat layout, and flipping the default would
silently re-identify every dataset relying on it.
"""

SequenceMatchMode = Literal["exact", "prefix"]
"""How a media file's stem is matched against known sequence names.

``exact`` requires the stem to be a sequence name. ``prefix`` matches the longest
sequence name that prefixes the stem, which is how split recordings named
``session01_001.mp4``, ``session01_002.mp4`` reach sequence ``session01``.
"""

GroupFrom = Literal["filename", "parent"]
"""Where a raw scan reads the group for a file holding several sequences.

Applies only under ``multi_sequences_per_file``; outside it, ``group_pattern``
is the knob. A closed set rather than a bare string because a typo previously
reached the indexer and silently produced an empty group.
"""

SourceKind = Literal["media", "tracks", "labels"]
"""Which raw root a declared source feeds."""

SourceMode = Literal["directory", "files"]
"""Whether a source claims a whole directory or an explicit list of files.

``directory`` walks and globs. ``files`` claims exactly what it lists and nothing
else in the same directory, which is what an import that selects some of a
folder's contents needs -- no glob expresses an arbitrary subset.
"""


class ManifestVersionError(RuntimeError):
    """A manifest declares a format version this code does not know.

    Raised on read rather than guessing. A newer manifest may hold fields whose
    absence changes meaning, so reading it under this version's rules and then
    saving would drop them silently.
    """


# ---------------------------------------------------------------- roots


_DEFAULT_ROOTS: Final[dict[str, str]] = {
    # raw: uploaded or referenced, never written by mosaic
    "media_raw": "media_raw",
    "tracks_raw": "tracks_raw",
    "labels_raw": "labels_raw",
    "labels": "labels",
    # derived: computed by mosaic, regenerable
    "media": "media",
    "tracks": "tracks",
    # Raw tracker output lives under _tracking/, not tracks_raw/, so tracks_raw
    # holds only user content. The per-tool roots come from the registry rather
    # than being spelled here, so a tracker added there cannot be forgotten here.
    TRACKING_ROOT: TRACKING_ROOT,
    **{key: root.default_path for key, root in TRACKING_ROOTS.items()},
    "features": "features",
    "models": "models",
    "frames": "media/frames",
}

default_roots: Final[Mapping[str, str]] = MappingProxyType(_DEFAULT_ROOTS)
"""Every root a dataset declares, and where it sits relative to the dataset.

Read-only: it is the default argument of the manifest constructor and the
template every new dataset starts from, so a caller mutating it would repoint
every dataset created afterwards in the same process.
"""

_SOURCE_ROOT_KEYS: Final[tuple[str, ...]] = ("tracks_raw", "media_raw", "labels")


def empty_root_template() -> dict[str, str]:
    """One empty-valued key per declared root -- the "nothing loaded yet" state.

    Derived from :data:`default_roots` so the two cannot drift. An empty value
    reads as unset, which is what lets a partially declared manifest resolve the
    roots it does declare and raise only for the ones it does not.
    """
    return {key: "" for key in default_roots}


def backfill_roots(roots: dict[str, str]) -> dict[str, str]:
    """Add the roots a manifest predating them does not declare.

    **Absent keys are filled; present ones are never repointed.** A dataset whose
    ``trex`` root still reads ``tracks_raw/trex`` keeps it. Silently moving it
    would orphan every run already on disk *and* strand the index that names
    them -- and the legacy location is a state the sweeper must be able to
    recognize and decline, which it cannot do if loading has quietly erased the
    evidence.

    **``media_raw`` is deliberately not in the list, and cannot be.** Filling
    ``labels_raw`` adds an empty directory to a dataset that had no labels, which
    costs nothing. Filling ``media_raw`` on a dataset whose videos are in
    ``media/`` would point it at an empty directory and make its media vanish.
    So it is the one source root that may legitimately be absent -- which is why
    the media accessors resolve their root through
    :meth:`~mosaic.core.dataset.Dataset.resolve_media_root` while the tracks and
    labels accessors can pin theirs, and why the media names do not carry a
    ``_raw`` a caller could rely on.

    In place on the mapping it is handed, and returned for the caller to assign,
    so a manifest that needs no backfill round-trips unchanged.
    """
    for key in (TRACKING_ROOT, *TRACKING_ROOTS, "labels_raw"):
        _ = roots.setdefault(key, _DEFAULT_ROOTS[key])
    return roots


def validate_root_inside(base_dir: Path, path: str | Path, key: str) -> Path:
    """Return *path* unchanged, or raise because it leaves the dataset.

    Roots always live inside the dataset tree, and external storage is expressed
    as :class:`ScanSources` whose files are referenced by absolute ``abs_path``
    from an index that lives inside. An outside root puts that root's own
    ``index.csv`` outside too, and then the dataset is no longer the thing you
    can copy, archive or sync.

    Absolute is fine when it lands inside. The rule is about where a root is, not
    how it is written -- and the portability pass relativizes an inside-absolute
    root on its next run anyway.

    A ``~`` is expanded before the rule is applied, so it is judged on where it
    lands. Unexpanded it would be judged on how it is written: ``~/elsewhere`` is
    not absolute, so it would be read as a path *relative* to the dataset, land
    inside, and pass -- and the literal ``~`` would then be persisted as a root
    and recreated under the dataset on every load. Expanded, it is an outside
    root and refused like any other.

    The comparison resolves both sides. An unnormalized ``..`` or a symlink would
    otherwise let a root leave the dataset while reading as though it stayed.

    Args:
        base_dir: The dataset directory the root must land inside.
        path: The candidate root, absolute or relative to *base_dir*.
        key: The root's name, for the error message.

    Returns:
        *path* as a :class:`~pathlib.Path`, unchanged.

    Raises:
        ValueError: If *path* resolves outside *base_dir*.
    """
    candidate = user_path(path)
    absolute = candidate if candidate.is_absolute() else base_dir / candidate
    resolved = absolute.resolve()
    root = base_dir.resolve()
    if resolved != root and root not in resolved.parents:
        msg = (
            f"root {key!r} would resolve outside the dataset: {resolved} is not "
            f"under {root}. Roots live inside the dataset; to use storage "
            "elsewhere, declare a scan source for it -- those may point anywhere."
        )
        raise ValueError(msg)
    return candidate


def legacy_tracking_roots(roots: Mapping[str, str]) -> dict[str, str]:
    """Tracker roots still nested inside a *source* root -- the pre-relocation layout.

    ``{root key: declared path}``, in practice ``{"trex": "tracks_raw/trex"}`` on
    a dataset converted before the relocation. Empty on a current one.

    Two callers need this and want opposite things from it. A raw-tracks scan
    wants to know that tracker output is sitting inside a source root, where its
    exclusion cannot reach; a sweeper wants to know that a root it is about to
    delete under holds user content, and to decline. Both questions are "is this
    root somewhere uploads live", so they are answered once here.

    Nested-in-a-source-root, not merely absent-from ``_tracking``: a root
    pointing *outside the dataset* is a different fault with a different repair,
    and reporting it as legacy sends someone to fix the wrong thing.
    """
    return {
        key: declared
        for key, declared in roots.items()
        if key in TRACKING_ROOTS
        and declared
        and PurePosixPath(str(declared).replace("\\", "/")).parts[:1]
        in {(source,) for source in _SOURCE_ROOT_KEYS}
    }


# ---------------------------------------------------------------- sources


_SOURCE_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_DISCOVERY_FIELDS: Final = frozenset(
    {"recursive", "extensions", "patterns", "exclude_patterns"}
)
"""Knobs that describe a *walk*. Meaningless once a source lists its files."""


class ScanSource(StrictModel):
    """A declared place a dataset draws raw files from.

    A source may point anywhere -- that is its purpose. Its directory is never
    created and never required to exist at load time, so a manifest naming an
    unmounted NAS still opens.

    Attributes:
        id: Stable handle, used by ``--only`` and by the messages that name a
            source. Letters, digits, dot, dash and underscore.
        path: Where the files are. Absolute, or relative to the dataset.
        files: Paths relative to *path*. Non-empty makes this a ``files`` source,
            claiming exactly these and nothing else beside them.
        recursive: Walk subdirectories. Directory mode only.
        added_at: When the source was declared, for a human reading the file.
    """

    id: str
    path: str
    files: tuple[str, ...] = ()
    recursive: bool = True
    added_at: str = ""

    @property
    def mode(self) -> SourceMode:
        """``"files"`` when this source lists its files, else ``"directory"``."""
        return "files" if self.files else "directory"

    @field_validator("id")
    @classmethod
    def _id_is_a_token(cls, value: str) -> str:
        if not _SOURCE_ID_RE.match(value):
            msg = (
                f"source id {value!r} must start with a letter or digit and hold "
                f"only letters, digits, '.', '-' and '_' "
                f"(at most {SOURCE_ID_MAX_LENGTH} characters)"
            )
            raise ValueError(msg)
        return value

    @field_validator("path")
    @classmethod
    def _path_is_not_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "source path must not be empty"
            raise ValueError(msg)
        return value

    @field_validator("files")
    @classmethod
    def _files_stay_under_path(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Every listed file is relative, escapes nothing, and appears once.

        An absolute entry or one containing ``..`` would let the source claim
        files outside its own ``path``, which makes its claimed set unbounded --
        and the claimed set is exactly what the overlap check and the scan's
        replace scope are computed from.
        """
        seen: set[str] = set()
        for entry in value:
            if not entry.strip():
                msg = "source files must not contain an empty entry"
                raise ValueError(msg)
            if Path(entry).is_absolute():
                msg = f"source file {entry!r} must be relative to the source path"
                raise ValueError(msg)
            if ".." in PurePosixPath(entry.replace("\\", "/")).parts:
                msg = f"source file {entry!r} must not contain '..'"
                raise ValueError(msg)
            if entry in seen:
                msg = f"source file {entry!r} is listed twice"
                raise ValueError(msg)
            seen.add(entry)
        return value

    @model_validator(mode="after")
    def _a_file_list_declares_no_walk(self) -> Self:
        if not self.files:
            return self
        declared = _DISCOVERY_FIELDS & self.model_fields_set
        if declared:
            msg = (
                f"source {self.id!r} lists files, so {sorted(declared)} "
                "do not apply -- a listed file is claimed whatever a glob says. "
                "Drop them, or drop 'files' to make this a directory source."
            )
            raise ValueError(msg)
        return self


class MediaScanSource(ScanSource):
    """A source feeding ``media_raw``.

    Attributes:
        extensions: Which suffixes count as media. Directory mode only.
        layout: How identity is derived for a file no track table names.
        match_mode: How a stem is matched against known sequence names.
    """

    kind: Literal["media"] = "media"
    extensions: tuple[str, ...] = (".mp4", ".avi")
    layout: MediaLayout = "stem"
    match_mode: SequenceMatchMode = "exact"

    @field_validator("extensions")
    @classmethod
    def _dotted_and_lowercase(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for entry in value:
            stripped = entry.strip().lower()
            if not stripped or stripped == ".":
                msg = "source extensions must not contain an empty entry"
                raise ValueError(msg)
            normalized.append(stripped if stripped.startswith(".") else f".{stripped}")
        return tuple(normalized)


class RawScanSource(ScanSource):
    """Shared shape of a source feeding ``tracks_raw`` or ``labels_raw``.

    Attributes:
        patterns: Globs selecting files under *path*. Directory mode only.
        src_format: Which converter reads these files.
        exclude_patterns: Basename globs to skip. Directory mode only.
        multi_sequences_per_file: One file holds several sequences.
        group_from: Where the group comes from, under multi-sequence files only.
        group_pattern: Regex extracting the group from a path, outside them only.
        md5: Checksum each file. On by default because the composition hash is
            over these checksums, and a re-index re-hashes only what changed.
    """

    patterns: tuple[str, ...]
    src_format: str
    exclude_patterns: tuple[str, ...] = ()
    multi_sequences_per_file: bool = False
    group_from: GroupFrom | None = None
    group_pattern: str | None = None
    md5: bool = True

    @field_validator("src_format")
    @classmethod
    def _src_format_is_not_empty(cls, value: str) -> str:
        if not value.strip():
            msg = "source src_format must not be empty"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _one_grouping_rule(self) -> Self:
        """``group_from`` and ``group_pattern`` answer the same question.

        They apply on opposite sides of ``multi_sequences_per_file``, so
        declaring both means one of them is dead and which one depends on a flag
        elsewhere in the source. The indexer refuses the same combination.
        """
        if self.group_from is not None and self.group_pattern is not None:
            msg = (
                f"source {self.id!r} declares both group_from and group_pattern; "
                "group_from applies under multi_sequences_per_file and "
                "group_pattern outside it, so exactly one can take effect"
            )
            raise ValueError(msg)
        if self.group_from is not None and not self.multi_sequences_per_file:
            msg = (
                f"source {self.id!r} declares group_from without "
                "multi_sequences_per_file, where it has no effect; "
                "use group_pattern instead"
            )
            raise ValueError(msg)
        return self


class TracksScanSource(RawScanSource):
    """A source feeding ``tracks_raw``."""

    kind: Literal["tracks"] = "tracks"
    patterns: tuple[str, ...] = ("*.npy", "*.h5", "*.csv")
    src_format: str = "calms21_npy"


class LabelsScanSource(RawScanSource):
    """A source feeding ``labels_raw``."""

    kind: Literal["labels"] = "labels"
    patterns: tuple[str, ...] = ("*.csv", "*.npy", "*.pkl")
    src_format: str = "boris_aggregated_csv"


AnyScanSource = MediaScanSource | TracksScanSource | LabelsScanSource


class ScanSources(StrictModel):
    """Every declared source, by the root it feeds.

    Attributes:
        media: Sources feeding ``media_raw``.
        tracks: Sources feeding ``tracks_raw``.
        labels: Sources feeding ``labels_raw``.
    """

    media: tuple[MediaScanSource, ...] = ()
    tracks: tuple[TracksScanSource, ...] = ()
    labels: tuple[LabelsScanSource, ...] = ()

    @model_validator(mode="after")
    def _ids_unique_within_a_kind(self) -> Self:
        for kind in ("media", "tracks", "labels"):
            seen: set[str] = set()
            for source in self.of_kind(kind):
                if source.id in seen:
                    msg = f"two {kind} sources share the id {source.id!r}"
                    raise ValueError(msg)
                seen.add(source.id)
        return self

    def of_kind(self, kind: SourceKind) -> tuple[AnyScanSource, ...]:
        """The declared sources feeding *kind*, in declaration order."""
        if kind == "media":
            return self.media
        if kind == "tracks":
            return self.tracks
        return self.labels

    def with_kind(self, kind: SourceKind, sources: Iterable[AnyScanSource]) -> Self:
        """A copy with *kind*'s sources replaced, re-validating this collection.

        The sources are passed as model instances, not dumped to dictionaries and
        re-parsed. A dump names every field, including the ones a source left at
        its default -- and ``model_fields_set`` is exactly what tells a file-mode
        source apart from one that declared a walk. Round-tripping through a dump
        therefore made every existing file source look as though it had asked for
        ``extensions`` and ``recursive`` too, so declaring any *second* source
        failed by re-validating the first.

        The collection's own rules -- unique ids within a kind -- still run,
        which is the validation this call is for.
        """
        listed = tuple(sources)
        if kind == "media":
            return type(self)(
                media=tuple(s for s in listed if isinstance(s, MediaScanSource)),
                tracks=self.tracks,
                labels=self.labels,
            )
        if kind == "tracks":
            return type(self)(
                media=self.media,
                tracks=tuple(s for s in listed if isinstance(s, TracksScanSource)),
                labels=self.labels,
            )
        return type(self)(
            media=self.media,
            tracks=self.tracks,
            labels=tuple(s for s in listed if isinstance(s, LabelsScanSource)),
        )

    def select(
        self, kind: SourceKind, only: Sequence[str] = ()
    ) -> tuple[AnyScanSource, ...]:
        """The sources of *kind*, or just those *only* names.

        Args:
            kind: Which root's sources to return.
            only: Source ids to restrict to. Empty means all of them.

        Returns:
            The selected sources, in declaration order.

        Raises:
            KeyError: If *only* names an id this kind does not declare. The
                message lists what is declared, because the usual cause is a
                typo and the usual repair is reading the real ids.
        """
        declared = self.of_kind(kind)
        if not only:
            return declared
        wanted = set(only)
        known = {source.id for source in declared}
        missing = sorted(wanted - known)
        if missing:
            msg = (
                f"no {kind} source named {missing[0]!r}; "
                f"declared: {sorted(known) or 'none'}"
            )
            raise KeyError(msg)
        return tuple(source for source in declared if source.id in wanted)


def overlapping_sources(
    sources: Sequence[AnyScanSource],
    resolve: Callable[[str], Path],
) -> tuple[str, str] | None:
    """The first pair of sources whose claimed files could intersect, or ``None``.

    Two sources of one kind claiming the same file make the scan's replace scope
    ambiguous: which recipe identifies that file becomes a question about
    declaration order rather than about the manifest. Refusing at declaration
    keeps the claimed sets a partition, which is what lets a scan replace one
    source's rows without touching another's.

    The three cases, and why they differ:

    - **directory against directory** -- refused when either path contains the
      other. A recursive walk's set cannot be enumerated without touching disk,
      so containment is the conservative test.
    - **files against files** -- refused only on a shared resolved file. Several
      import batches under one directory are legal and expected; that is the
      case selective import produces.
    - **files against directory** -- refused when the file source's path is at or
      under the directory source's tree, since the walk would claim those files
      too.

    Args:
        sources: The sources of one kind, including the candidate.
        resolve: Turns a declared path into an absolute one.

    Returns:
        The ids of the first overlapping pair, or ``None``.
    """
    resolved = [(source, resolve(source.path)) for source in sources]
    for index, (left, left_path) in enumerate(resolved):
        for right, right_path in resolved[index + 1 :]:
            if _claims_intersect(left, left_path, right, right_path):
                return (left.id, right.id)
    return None


def _under_or_equal(inner: Path, outer: Path) -> bool:
    return inner == outer or outer in inner.parents


def _claims_intersect(
    left: AnyScanSource, left_path: Path, right: AnyScanSource, right_path: Path
) -> bool:
    if left.mode == "files" and right.mode == "files":
        left_files = {left_path / entry for entry in left.files}
        right_files = {right_path / entry for entry in right.files}
        return bool(left_files & right_files)
    if left.mode == "files":
        return _under_or_equal(left_path, right_path)
    if right.mode == "files":
        return _under_or_equal(right_path, left_path)
    return _under_or_equal(left_path, right_path) or _under_or_equal(
        right_path, left_path
    )


# ---------------------------------------------------------------- tags


class DatasetTag(StrictModel):
    """One typed attribute describing the dataset as a whole.

    The field names match the tag definitions and assignments mosaic-api keeps
    for sequences and individuals, so a tag means the same thing on either side
    and moving one across is a mapping rather than a translation. Definition and
    value collapse into a single entry because a dataset-level tag has exactly
    one holder: there is no vocabulary shared across many targets, and so nothing
    for a constraint change to narrow *against*. Redefining a tag re-validates
    its own value and raises if it no longer fits, which is that protection one
    row wide.

    Not to be confused with the per-sequence tags mosaic-api owns, which are how
    sequences are grouped for analysis. These describe the dataset itself.

    Attributes:
        name: Case-insensitively unique within the manifest.
        type: One of the six typed-attribute types.
        type_constraints: Bounds or options, whose legal keys depend on *type*.
        value: The value, satisfying *type* and *type_constraints*. Always
            ``None`` for a ``label``, which is presence-only.
        description: What the tag means, for whoever reads it next.
        display_order: Ordering hint. Ties break on *name*.
    """

    name: str
    type: TypedAttributeType
    type_constraints: dict[str, JsonValue] = Field(default_factory=dict)
    value: TypedAttributeValue = None
    description: str | None = None
    display_order: int = 0

    @field_validator("name")
    @classmethod
    def _name_is_bounded(cls, value: str) -> str:
        if not value.strip():
            msg = "tag name must not be empty"
            raise ValueError(msg)
        if len(value) > TAG_NAME_MAX_LENGTH:
            msg = f"tag name {value!r} exceeds {TAG_NAME_MAX_LENGTH} characters"
            raise ValueError(msg)
        return value

    @field_validator("description")
    @classmethod
    def _description_is_bounded(cls, value: str | None) -> str | None:
        if value is not None and len(value) > TAG_DESCRIPTION_MAX_LENGTH:
            msg = (
                f"tag description exceeds {TAG_DESCRIPTION_MAX_LENGTH} characters "
                f"({len(value)})"
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _constraints_then_value(self) -> Self:
        """Constraints first, then the value against them.

        In that order because a malformed constraint reported against a value
        sends the reader to the wrong field.

        **A ``None`` value means the tag is declared but not yet set**, and is
        legal for every type. Definition and assignment are one entry here, but
        they remain two acts: declaring that a dataset *has* a cohort is useful
        before anybody knows which cohort. ``label`` is the exception in the
        other direction -- it is presence-only, so ``None`` is not merely allowed
        but required, and that is checked rather than skipped.
        """
        try:
            validate_constraints(self.type, self.type_constraints)
            if self.value is not None or self.type == "label":
                validate_typed_value(self.type, self.type_constraints, self.value)
        except ValueError as exc:
            msg = f"tag {self.name!r}: {exc}"
            raise ValueError(msg) from exc
        return self


# ---------------------------------------------------------------- the manifest


class DatasetManifest(BaseModel):
    """Everything ``dataset.yaml`` holds.

    Attributes:
        manifest_version: The format version. See :data:`MANIFEST_VERSION`.
        name: What the dataset is called.
        version: The dataset's own version string, for whoever curates it.
        uuid: Minted once at creation and never rewritten.
        created_at: When the manifest was first written, ISO-8601 UTC.
        roots: Named directories, always inside the dataset.
        sources: Declared scan recipes, which point wherever the data is.
        notes: Free text that travels with the dataset.
        tags: Typed attributes describing the dataset.
        continuous_groups: Groups whose sequences are time divisions of one
            recording rather than independent trials. See
            :meth:`is_continuous_group`.
        meta: Structured per-subsystem metadata, written by converters.
        preserved: Top-level keys this version does not model, kept verbatim.
        migrated_from: The version read from disk, when it was not the current
            one. ``None`` for a manifest already at :data:`MANIFEST_VERSION`.
    """

    # Tolerant where every submodel is strict. A source is a replayable recipe
    # and a tag is a typed contract, where a mistyped key means the next scan
    # runs a different recipe or a tag loses a constraint. Those models derive
    # from ``StrictModel`` and raise. An unknown key here is somebody's future
    # field, kept rather than refused.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    manifest_version: int = MANIFEST_VERSION
    name: str = "unnamed"
    version: str = "0.1"
    uuid: str | None = None
    created_at: str | None = None
    roots: dict[str, str] = Field(default_factory=empty_root_template)
    sources: ScanSources = Field(default_factory=ScanSources)
    notes: str = ""
    tags: tuple[DatasetTag, ...] = ()
    continuous_groups: tuple[str, ...] = ()
    meta: dict[str, JsonValue] = Field(default_factory=dict)

    # Not written as themselves: `preserved` is re-emitted by name after the
    # modeled keys, and `migrated_from` is a fact about this read, not about the
    # dataset.
    preserved: dict[str, JsonValue] = Field(default_factory=dict, exclude=True)
    migrated_from: int | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _capture_unknown_keys(cls, data: dict[str, object]) -> dict[str, object]:
        """Move every unmodeled top-level key into ``preserved``.

        This is what makes retiring a field free. A key this version stopped
        modeling is not deleted from the file; it is carried through the
        load-and-save round trip untouched, so a manifest edited by a tool that
        knows about it keeps working.

        ``preserved`` is itself a modeled field, so it never lands in *unknown*
        and a hand-written one is simply replaced by what this read found.
        """
        known = set(cls.model_fields)
        unknown = {key: value for key, value in data.items() if key not in known}
        if not unknown:
            return data
        modeled = {key: value for key, value in data.items() if key in known}
        return {**modeled, "preserved": unknown}

    @field_validator("notes")
    @classmethod
    def _notes_is_bounded(cls, value: str) -> str:
        if len(value) > NOTES_MAX_CHARS:
            msg = (
                f"notes is {len(value)} characters, over the "
                f"{NOTES_MAX_CHARS} limit; long prose belongs in a file the "
                "dataset references rather than in the manifest"
            )
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _tags_are_uniquely_named_and_bounded(self) -> Self:
        if len(self.tags) > TAGS_MAX_COUNT:
            msg = f"manifest declares {len(self.tags)} tags, over the {TAGS_MAX_COUNT} limit"
            raise ValueError(msg)
        seen: dict[str, str] = {}
        for tag in self.tags:
            folded = tag.name.casefold()
            if folded in seen:
                msg = (
                    f"tags {seen[folded]!r} and {tag.name!r} differ only by case; "
                    "tag names are matched case-insensitively, as they are in "
                    "mosaic-api"
                )
                raise ValueError(msg)
            seen[folded] = tag.name
        return self

    @field_validator("continuous_groups")
    @classmethod
    def _continuous_groups_are_named_and_unique(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        """A continuous group must be named, and named once.

        The empty group -- legal everywhere else, because ``group`` is an
        optional namespace -- cannot be continuous. A continuous group *is* the
        recording its sequences divide, so it is the one place the name carries
        meaning rather than merely disambiguating.
        """
        seen: set[str] = set()
        for name in value:
            if not name:
                msg = (
                    "continuous_groups names the empty group. A continuous "
                    "group is the recording its sequences divide, so it has to "
                    "be named; the empty group is the absence of one."
                )
                raise ValueError(msg)
            if name in seen:
                msg = f"continuous_groups lists {name!r} more than once"
                raise ValueError(msg)
            seen.add(name)
        return value

    def is_continuous_group(self, group: str) -> bool:
        """Whether *group*'s sequences are time divisions of one recording.

        A continuous group asserts two things nothing else records. Its
        sequences are ordered in time and adjacent, so a feature may read across
        a sequence boundary (``overlap_frames``); and its ``frame`` column is one
        axis spanning the whole group rather than restarting per sequence, so its
        media resolves as one shared timeline rather than per sequence.

        The assertion is the caller's; mosaic verifies it against the recorded
        frame ranges before acting on it, and refuses where the two disagree.
        Declaration and measurement are both required and neither substitutes
        for the other: no measurement can establish that two sequences are
        divisions of one recording rather than two recordings that happen to be
        numbered consecutively, and no declaration can be trusted about an axis
        that is there to be read.
        """
        return group in self.continuous_groups

    def tag(self, name: str) -> DatasetTag | None:
        """The tag called *name*, matched case-insensitively, or ``None``."""
        folded = name.casefold()
        for candidate in self.tags:
            if candidate.name.casefold() == folded:
                return candidate
        return None

    def ordered_tags(self) -> tuple[DatasetTag, ...]:
        """The tags by ``display_order``, ties broken on name.

        The order they are written in and the order a reader should show them.
        """
        return tuple(sorted(self.tags, key=lambda tag: (tag.display_order, tag.name)))


# ---------------------------------------------------------------- read and write


def resolve_manifest_path(path: Path) -> Path:
    """The manifest file at or named by *path*.

    A directory is probed for :data:`MANIFEST_FILENAMES` in order, so a caller
    may hand over a dataset directory instead of the file inside it.

    Raises:
        FileNotFoundError: If *path* is a directory holding none of them, or
            names a file that does not exist.
    """
    if path.is_dir():
        for candidate in MANIFEST_FILENAMES:
            probe = path / candidate
            if probe.exists():
                return probe
        msg = f"No manifest found in directory: {path}"
        raise FileNotFoundError(msg)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


_PARSED_MANIFEST: Final = TypeAdapter(dict[str, JsonValue])
"""Turns whatever the parser produced into a typed mapping, or says why not.

A YAML or JSON parser returns an untyped object, and a manifest that is a list,
a bare string or a mapping with a non-string key is a real thing to find on
disk. Validating it here means the rest of the module works with
``dict[str, JsonValue]`` rather than re-checking at each use.
"""


def _parse(path: Path) -> dict[str, JsonValue]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        parsed: object = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        parsed = yaml.safe_load(text)
    else:
        # Unknown suffix: try YAML, which also parses JSON, then JSON for the
        # case where a stray tab or duplicate key makes YAML refuse it.
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError:
            parsed = json.loads(text)
    if parsed is None:
        return {}
    try:
        return _PARSED_MANIFEST.validate_python(parsed)
    except ValidationError as exc:
        msg = f"manifest {path} is a {type(parsed).__name__}, not a mapping"
        raise ValueError(msg) from exc


def migrate_to_current(
    data: dict[str, JsonValue], *, source: str = ""
) -> DatasetManifest:
    """Read *data* as a manifest of whatever version it declares, at the current one.

    Migration is in memory and additive. A retired key is not modeled, so it
    lands in ``preserved`` and is written back out; an absent root is backfilled;
    a version absent entirely reads as 1, because that is what every manifest
    written before versioning existed is.

    Args:
        data: The parsed manifest mapping.
        source: The file it came from, for the version error's message.

    Returns:
        A manifest at :data:`MANIFEST_VERSION`, with ``migrated_from`` set when
        the file on disk was older.

    Raises:
        ManifestVersionError: If *data* declares a version newer than this code.
    """
    declared_raw = data.get("manifest_version", 1)
    declared = declared_raw if isinstance(declared_raw, int) else 1
    if declared > MANIFEST_VERSION:
        where = f" ({source})" if source else ""
        msg = (
            f"manifest{where} declares version {declared}, but this mosaic "
            f"writes version {MANIFEST_VERSION}. Reading it under the older "
            "rules could drop fields it holds; upgrade mosaic instead."
        )
        raise ManifestVersionError(msg)

    manifest = DatasetManifest.model_validate({**data, "manifest_version": declared})
    manifest.roots = backfill_roots(manifest.roots)
    manifest.manifest_version = MANIFEST_VERSION
    manifest.migrated_from = declared if declared != MANIFEST_VERSION else None
    return manifest


def read_manifest(path: Path) -> DatasetManifest:
    """Read the manifest at *path*, migrated to the current version in memory.

    **Reading never writes.** A version-1 file stays a version-1 file on disk
    until something saves it, so a read-only mount works and looking at a legacy
    dataset does not rewrite it.

    Args:
        path: The manifest file, or the dataset directory holding it.

    Returns:
        The manifest, at :data:`MANIFEST_VERSION`.

    Raises:
        FileNotFoundError: If no manifest is there.
        ManifestVersionError: If the file is newer than this code.
        ValueError: If the file is not a mapping, or violates the schema.
    """
    resolved = resolve_manifest_path(path)
    return migrate_to_current(_parse(resolved), source=str(resolved))


def manifest_header() -> str:
    """The comment block written above every manifest.

    Regenerated on each write rather than preserved, because a YAML dump cannot
    carry comments through. That is also why the block says so: durable prose
    belongs in ``notes``, which does survive.
    """
    return f"""\
# ============================================================================
# mosaic dataset manifest (v{MANIFEST_VERSION})
#
# roots    Named directories, ALWAYS inside this dataset. Absent keys are
#          backfilled on load; a declared one is never repointed.
# sources  Declared scan recipes. A source may point OUTSIDE the dataset on
#          purpose; its files are indexed by absolute abs_path into an index
#          that stays inside. `mosaic scan` rescans exactly this set, and a
#          source directory is never created. Give `files:` to claim an
#          explicit selection rather than everything a glob matches.
# notes    Free text that travels with the dataset.
# tags     Typed dataset attributes: the same type / type_constraints / value
#          shape as the sequence and individual tags in mosaic-api. These
#          describe the DATASET; per-sequence tags live in mosaic-api.
# continuous_groups
#          Groups whose sequences are time divisions of ONE recording, not
#          independent trials. Their `frame` column is one axis spanning the
#          group, and their media resolves as one shared timeline. This is what
#          `overlap_frames` reads across; mosaic verifies it against the
#          recorded frame ranges and refuses where the two disagree.
# meta     Structured per-subsystem metadata, written by converters.
#
# Comments are NOT preserved across a save: this header is regenerated and
# anything else typed here is lost. Durable prose belongs in `notes`.
# Unknown top-level KEYS are preserved verbatim.
# ============================================================================
"""


def _source_payload(source: AnyScanSource) -> dict[str, JsonValue]:
    """One source as it is written, with the knobs its mode does not use dropped.

    ``kind`` is never written: which list a source is in already says it, and a
    second copy is a fact that can disagree with the first. The discovery knobs
    are dropped in file mode because the model refuses a source that declares
    both -- writing them would produce a file this code could not read back.
    """
    payload: dict[str, JsonValue] = {"id": source.id, "path": source.path}
    if source.mode == "files":
        payload["files"] = list(source.files)
    else:
        payload["recursive"] = source.recursive
        if isinstance(source, MediaScanSource):
            payload["extensions"] = list(source.extensions)
        else:
            payload["patterns"] = list(source.patterns)
            if source.exclude_patterns:
                payload["exclude_patterns"] = list(source.exclude_patterns)
    if isinstance(source, MediaScanSource):
        payload["layout"] = source.layout
        payload["match_mode"] = source.match_mode
    else:
        payload["src_format"] = source.src_format
        if source.multi_sequences_per_file:
            payload["multi_sequences_per_file"] = True
        if source.group_from is not None:
            payload["group_from"] = source.group_from
        if source.group_pattern is not None:
            payload["group_pattern"] = source.group_pattern
        payload["md5"] = source.md5
    if source.added_at:
        payload["added_at"] = source.added_at
    return payload


def _tag_payload(tag: DatasetTag) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {"name": tag.name, "type": tag.type}
    if tag.type_constraints:
        payload["type_constraints"] = dict(tag.type_constraints)
    # Written for every type but ``label``, which is presence-only. Testing the
    # type rather than the value keeps a legitimate False, 0 or "" from being
    # mistaken for "no value".
    if tag.type != "label":
        payload["value"] = tag.value
    if tag.description is not None:
        payload["description"] = tag.description
    if tag.display_order:
        payload["display_order"] = tag.display_order
    return payload


def manifest_payload(manifest: DatasetManifest) -> dict[str, JsonValue]:
    """The manifest as the mapping that gets serialized.

    Empty optional sections are omitted rather than written as placeholders, so
    a fresh dataset's manifest is short and the header is what teaches the
    format. Preserved keys come last, after everything this version models, so
    the modeled shape stays the first thing a reader sees.
    """
    payload: dict[str, JsonValue] = {
        "manifest_version": manifest.manifest_version,
        "name": manifest.name,
        "version": manifest.version,
    }
    if manifest.uuid:
        payload["uuid"] = manifest.uuid
    if manifest.created_at:
        payload["created_at"] = manifest.created_at
    payload["roots"] = dict(manifest.roots)

    declared: dict[str, JsonValue] = {}
    for kind in ("media", "tracks", "labels"):
        of_kind = manifest.sources.of_kind(kind)
        if of_kind:
            declared[kind] = [_source_payload(source) for source in of_kind]
    if declared:
        payload["sources"] = declared
    if manifest.notes:
        payload["notes"] = manifest.notes
    if manifest.tags:
        payload["tags"] = [_tag_payload(tag) for tag in manifest.ordered_tags()]
    if manifest.continuous_groups:
        payload["continuous_groups"] = list(manifest.continuous_groups)
    if manifest.meta:
        payload["meta"] = dict(manifest.meta)

    for key, value in manifest.preserved.items():
        if key not in payload:
            payload[key] = value
    return payload


class _ManifestDumper(yaml.SafeDumper):
    """A dumper for a file people read: block strings and indented lists.

    Subclassed rather than registered on ``SafeDumper`` so neither the
    representer nor the indentation leaks into every other ``yaml.safe_dump`` in
    the process -- the index writers dump YAML too and want the default.
    """

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        # PyYAML puts a sequence flush against the key that owns it. Correct
        # YAML, but a manifest is read by hand and a nested list of sources is
        # far easier to follow indented under its kind.
        _ = indentless
        super().increase_indent(flow=flow, indentless=False)


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Multi-line strings as literal blocks, so ``notes`` reads as prose.

    The node is built rather than routed through ``represent_scalar``, whose only
    extra work is recording the value for anchor reuse -- and a safe dumper never
    aliases a string, so that bookkeeping is a no-op here.
    """
    _ = dumper
    style = "|" if "\n" in data else None
    return yaml.ScalarNode("tag:yaml.org,2002:str", data, style=style)


_ManifestDumper.add_representer(str, _represent_str)


def manifest_text(manifest: DatasetManifest, *, as_json: bool = False) -> str:
    """The exact bytes :func:`write_manifest` would write, as a string."""
    payload = manifest_payload(manifest)
    if as_json:
        return json.dumps(payload, indent=2) + "\n"
    body = yaml.dump(
        payload,
        Dumper=_ManifestDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return manifest_header() + body


def write_manifest(path: Path, manifest: DatasetManifest) -> None:
    """Write *manifest* to *path*, atomically.

    The serialization is chosen by *path*'s suffix -- ``.json`` writes JSON,
    anything else writes YAML. There is no stored format field to disagree with
    the name of the file it describes.

    Atomic because the manifest is a single file rewritten whole: a reader must
    never see half of one, and an interrupted write must not leave a dataset
    without a manifest at all.
    """
    text = manifest_text(manifest, as_json=path.suffix.lower() == ".json")
    atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def new_manifest(
    name: str,
    *,
    version: str = "0.1.0",
    roots: Mapping[str, str] | None = None,
    notes: str = "",
    tags: Sequence[DatasetTag] = (),
    sources: ScanSources | None = None,
) -> DatasetManifest:
    """A fresh manifest, with its identity minted.

    ``uuid`` and ``created_at`` are set here and never rewritten afterwards, so
    they identify the dataset rather than its most recent edit.

    Args:
        name: What the dataset is called.
        version: The dataset's own version string.
        roots: Root declarations. Defaults to :data:`default_roots`.
        notes: Free text for the dataset.
        tags: Typed attributes describing the dataset.
        sources: Declared scan recipes.

    Returns:
        The manifest. Nothing is written; see :func:`write_manifest`.
    """
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return DatasetManifest(
        name=name,
        version=version,
        uuid=str(uuid_module.uuid4()),
        created_at=now,
        roots=dict(roots) if roots is not None else dict(default_roots),
        sources=sources if sources is not None else ScanSources(),
        notes=notes,
        tags=tuple(tags),
    )


def now_stamp() -> str:
    """An ISO-8601 UTC timestamp, for a source's ``added_at``."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
