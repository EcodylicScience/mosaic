"""The single writer of the raw-tracks index CSV.

Named ``tracks_raw_index`` rather than ``tracks_index``, because this is the
index of the *raw* uploads under ``tracks_raw/``, not of the converted tables
under ``tracks/``. ``tracks/index.csv`` has a typed row of its own, so each
index has to say which root it means. Every public name here carries
``tracks_raw`` for the same reason: a bare ``read_tracks_raw_index`` beside a
real converted-tracks index reads as the one the caller probably wanted, and
picking the wrong one is a silent wrong answer rather than an import error.

The raw-tracks index (``tracks_raw/index.csv``) is a mosaic-defined artifact
with invariants mosaic owns: the ``TRACKS_RAW_INDEX_COLUMNS`` schema (``group``,
``sequence``, ``abs_path``, ``src_format``, ``size_bytes``, ``mtime_iso``,
``md5`` and ``companions``), an ``abs_path`` stored root-relative when the file
lives inside the dataset tree (absolute otherwise), and atomic writes. The
``companions`` cell records the files beside a source that its converter may
read: a JSON array, sorted by name, of each one's ``name`` (its basename),
``size_bytes``, ``mtime_iso`` and ``md5``, or an empty cell when it lists
none. This module owns the Dataset-agnostic serialization of that artifact so a
single body of code enforces those invariants for every caller (today:
``Dataset.index_tracks_raw``).

It is the tracks sibling of :mod:`mosaic.core.pipeline.media_index`, but
simpler: tracks have no within-sequence ordering (no ``video_order``), no
ffprobe metadata, and no camera axis. Only a raw file's *format* and *grouping*
vary, and that variation is decided by the scanning caller, not here.

Like the media module, it stays Dataset-agnostic: the one place that needs
dataset context -- turning an absolute path into its stored form -- is supplied
by the caller as a ``to_store_path`` callable (``Dataset.relative_to_root``),
mirroring the resolver callable :meth:`IndexCSV.prune_missing` takes.

The assignment-driven projection ``Dataset.write_tracks_raw_index(scopes)`` (with
``TracksRawIndexScope``) is the tracks mirror of ``Dataset.write_media_index``
for a future API track-import flow that uploads raw tracker files, assigns each
an explicit ``(group, sequence, src_format)``, and imports them: it (re)stats
each file under a scope directory, stamps the caller's identity, preserves rows
outside the scopes, and atomic-writes -- without the media densifier, since
today's tracks are unordered. No API caller exists yet.

Reserved axes (additive, not built): multi-camera / time-chunked raw tracks
would add ``camera`` and ``sync_uuid`` as within-sequence *columns* -- as the
media index did in its multi-camera phase, where ``camera`` is a column (like
``id``) and identity stays the ``(group, sequence)`` 2-tuple, not a
``camera``-widened key. A per-file time-order would number chunks via the
shared ``assign_video_order`` ranker per ``(group, sequence, camera)`` (as
``densify_video_order`` does). One asymmetry to carry forward: imgstore media
*probes* ``camera``/``sync_uuid`` from store metadata, but raw track files
carry none, so tracks-``camera`` must be *assigned* through the scope, never
probed.
These land as new ``TRACKS_RAW_INDEX_COLUMNS`` entries plus new keyword scope
fields; ``write_tracks_raw_index_rows`` already drops any column outside the
schema, so nothing here needs a redesign when they arrive.
"""

from __future__ import annotations

import csv
import fnmatch
import json
import math
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Literal, TypedDict

import pandas as pd

from mosaic.core.pipeline.composition import SourceMember

from mosaic.core.entry import validate_entry_name
from mosaic.core.helpers import make_entry_key
from mosaic.core.narrowing import is_object_list, is_str_mapping
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.label_series import is_under_label_series
from mosaic.core.pipeline.definitions.store import is_under_definitions_root
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOT, is_under_tracking_root
from mosaic.core.pipeline.media_index import mtime_iso
from mosaic.core.pipeline.promotion import correction_revision

# Column schema and order for ``tracks_raw/index.csv``. Owned here (not in the
# ``core/media`` leaf like ``MEDIA_INDEX_COLUMNS``) because raw-tracks columns
# are pipeline-owned and carry no media-I/O dependency.
TRACKS_RAW_INDEX_COLUMNS = [
    "group",
    "sequence",
    "abs_path",
    "src_format",
    "size_bytes",
    "mtime_iso",
    "md5",
    "companions",
]

# The only numeric raw-tracks column; every other column is a text cell that
# must round trip through CSV as an empty string rather than a float NaN.
TRACKS_RAW_NUMERIC_COLUMNS = frozenset({"size_bytes"})


@dataclass(frozen=True)
class TracksRawIndexScope:
    """One affected ``(group, sequence)`` directory to (re)scan and assign.

    ``directory`` is the ``tracks_raw`` subdir scanned for this sequence's raw
    files; every matching file found is assigned ``group``/``sequence`` and
    stamped with ``src_format`` (the converter name) -- the assignment-driven
    counterpart to ``Dataset.index_tracks_raw``'s scan-and-derive. All files in
    one scope dir take that single identity, so multiple per-id raw files (e.g.
    TREx ``seq_fish0``/``seq_fish1``) become one multi-file sequence *without*
    the ``_fishN`` strip ``index_tracks_raw`` applies -- here the caller owns the
    grouping. This is the tracks sibling of :class:`MediaIndexScope`; it has no
    ``camera`` or ordering axis today (tracks are unordered). Multi-camera /
    time-chunked raw tracks would add an *assigned* ``camera`` (plus
    ``sync_uuid``) field here -- populating a within-sequence ``camera`` column,
    with identity still ``(group, sequence)`` -- and a per-file order, mirroring
    the media index's multi-camera model (see the module docstring).
    """

    directory: Path
    group: str
    sequence: str
    src_format: str


class TracksRawIndexRow(TypedDict):
    """One ``tracks_raw/index.csv`` row -- the typed surface a schema change
    cannot silently break.

    ``group``/``sequence`` are the raw file's assigned identity, ``src_format``
    names its converter, ``abs_path`` is the stored (root-relative or absolute)
    path, and ``md5`` is empty unless the caller opted into hashing.
    ``companions`` is the cell :func:`companions_cell` writes: the files beside
    the source that its converter may read, or empty when it lists none.
    """

    group: str
    sequence: str
    abs_path: str
    src_format: str
    size_bytes: int
    mtime_iso: str
    md5: str
    companions: str


@dataclass(frozen=True)
class CompanionRecord:
    """A file beside a raw source that its converter may read, as the scan saw it.

    Attributes:
        name: The companion's basename. A companion is always beside its source,
            and a basename survives a dataset moving between machines.
        size_bytes: Its size when scanned.
        mtime_iso: Its modification time when scanned, spelled as the index
            spells one.
        md5: Its checksum, empty when the scan did not hash.
    """

    name: str
    size_bytes: int
    mtime_iso: str
    md5: str


type CompanionRule = Callable[[Path], tuple[Path, ...]]
"""Lists the files beside a raw source that its converter may read."""


def no_companions(path: Path) -> tuple[Path, ...]:
    """The rule of a source whose converter reads nothing beside it.

    Every labels source follows it, and so does a format with no registered
    converter.
    """
    _ = path
    return ()


def _is_blank(value: object) -> bool:
    """Whether a cell holds nothing: absent, ``None`` or the NaN of a missing cell."""
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, float) and math.isnan(value))
    )


def companions_cell(records: Iterable[CompanionRecord]) -> str:
    """The ``companions`` cell for *records*: compact JSON sorted by name, or ``""``.

    One spelling per value, so the same companions give the same bytes whatever
    order they were listed in.
    """
    ordered = sorted(records, key=lambda record: record.name)
    if not ordered:
        return ""
    return json.dumps(
        [
            {
                "name": record.name,
                "size_bytes": record.size_bytes,
                "mtime_iso": record.mtime_iso,
                "md5": record.md5,
            }
            for record in ordered
        ],
        separators=(",", ":"),
    )


def _companion_from(entry: object) -> CompanionRecord | None:
    """The record an array entry spells, or ``None`` when it is not one."""
    if not is_str_mapping(entry) or set(entry) != {
        "name",
        "size_bytes",
        "mtime_iso",
        "md5",
    }:
        return None
    name, size_bytes = entry["name"], entry["size_bytes"]
    mtime, digest = entry["mtime_iso"], entry["md5"]
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or not isinstance(mtime, str)
        or not isinstance(digest, str)
    ):
        return None
    return CompanionRecord(
        name=name, size_bytes=size_bytes, mtime_iso=mtime, md5=digest
    )


def companions_of(row: Mapping[str, object]) -> tuple[CompanionRecord, ...]:
    """The companions a row's ``companions`` cell records, in the cell's order.

    A blank, absent, ``None`` or NaN cell reads as no companions, because an
    index written before the column holds no cell and rows built from one hold a
    missing value.

    Raises:
        ValueError: The cell is not the array :func:`companions_cell` writes.
            The message names the row's ``abs_path``.
    """
    cell = row.get("companions")
    if _is_blank(cell) or cell == "":
        return ()
    where = str(row.get("abs_path", "") or "")
    problem = (
        f"the companions cell of {where} is not the JSON array of objects with "
        "name, size_bytes, mtime_iso and md5 that a scan writes"
    )
    if not isinstance(cell, str):
        raise ValueError(problem)
    try:
        entries: object = json.loads(cell)
    except json.JSONDecodeError as error:
        raise ValueError(problem) from error
    if not is_object_list(entries):
        raise ValueError(problem)
    records: list[CompanionRecord] = []
    for entry in entries:
        record = _companion_from(entry)
        if record is None:
            raise ValueError(problem)
        records.append(record)
    return tuple(records)


def refuse_malformed_companions(rows: Iterable[Mapping[str, object]]) -> None:
    """Parse every row's ``companions`` cell, so a malformed one is refused.

    Raises:
        ValueError: A cell does not parse. The message names the row's file.
    """
    for row in rows:
        _ = companions_of(row)


def companion_records(
    path: Path, rule: CompanionRule, digest: Callable[[Path], str]
) -> tuple[CompanionRecord, ...]:
    """The records of the files *rule* lists beside *path*, sorted by name.

    *path* is the file the index stores, which a conversion reads. A listed file
    that is gone by the time it is read is left out, as it would be had the scan
    run a moment later.

    Args:
        path: The raw source file, resolved.
        rule: The converter's :meth:`~TrackConverter.companion_files`, or
            :func:`no_companions`.
        digest: Returns a file's checksum, or ``""`` when the scan does not hash.
    """
    records: list[CompanionRecord] = []
    for companion in rule(path):
        try:
            stat = companion.stat()
        except FileNotFoundError:
            continue
        records.append(
            CompanionRecord(
                name=companion.name,
                size_bytes=stat.st_size,
                mtime_iso=mtime_iso(stat.st_mtime),
                md5=digest(companion),
            )
        )
    return tuple(sorted(records, key=lambda record: record.name))


def read_tracks_raw_index(index_path: Path) -> list[dict[str, str]]:
    """Read a raw-tracks index CSV as string-cell records (empty list if absent).

    The record form (string cells, not pandas) is what identity-reading callers
    want; it is the drop-in a future API reader uses instead of its own
    ``csv.DictReader``.
    """
    if not index_path.exists():
        return []
    with index_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_tracks_raw_index_frame(index_path: Path) -> pd.DataFrame:
    """Read a raw-tracks index CSV into the full schema, text cells as ``str``.

    The frame-shaped reader every consumer of this index goes through, so that a
    blank ``group`` is structurally an empty string rather than a float NaN that
    ``str()`` spells ``"nan"`` and hands to an entry key. ``keep_default_na=False``
    plus pinned string dtypes is how the other index readers state that, and
    pinning the dtype is the half a post-hoc repair cannot do: a sequence named
    ``001`` has to read back as ``"001"``, not as the integer it looks like.

    Read as strings wholesale and the one numeric column put back afterwards,
    rather than by a per-column dtype mapping: the mapping is what says this, but
    pandas types it as a dict whose value type it will not accept, and one
    coercion is cheaper than the alternatives. A blank stays blank -- ``NaN``
    here is a size the index never recorded, not a size of zero.

    Missing columns are added, so an index written before a column existed still
    reads: an index of seven columns reads a blank ``companions`` cell, which is
    no companions.
    """
    df = pd.DataFrame(columns=TRACKS_RAW_INDEX_COLUMNS)
    if index_path.exists():
        try:
            df = pd.read_csv(index_path, keep_default_na=False, dtype=str)
        except pd.errors.EmptyDataError:
            # A file holding no header at all -- a truncated write, an
            # interrupted scan, a sync that landed the inode and not the bytes.
            # Read as the empty index a missing file reads as, rather than
            # raising a pandas message that names no path: this reader sits
            # under `reindex` and `make_portable`, where one such file took down
            # the whole pass over every other root.
            pass
    for column in TRACKS_RAW_INDEX_COLUMNS:
        if column not in df.columns:
            df[column] = ""
        elif column in TRACKS_RAW_NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def frame_from_rows(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Build a raw-tracks-index DataFrame with exactly ``TRACKS_RAW_INDEX_COLUMNS``.

    Accepts both freshly built :class:`TracksRawIndexRow` rows (a ``TypedDict``,
    hence a ``Mapping[str, object]``) and preserved string-cell records from
    :func:`read_tracks_raw_index`, so the projection can merge the two without
    per-row normalization.

    A text cell a row lacks is ``""``, not a float NaN, so a row read from an
    index written before a column carries a blank cell to whatever reads the
    frame next. ``size_bytes`` stays numeric.
    """
    frame = pd.DataFrame(rows, columns=TRACKS_RAW_INDEX_COLUMNS)
    for column in TRACKS_RAW_INDEX_COLUMNS:
        if column not in TRACKS_RAW_NUMERIC_COLUMNS:
            frame[column] = [
                "" if _is_blank(value) else value for value in frame[column].tolist()
            ]
    return frame


def _skip_reason(
    path: Path, exclude_patterns: Sequence[str]
) -> Literal["fork", "excluded", "tracking", "definitions", "series"] | None:
    """Why a scan leaves *path* out, or ``None`` to keep it.

    Checked in order of cost: the label-series marker is three ``stat`` calls.
    """
    if path.name.startswith("._"):
        return "fork"
    if any(fnmatch.fnmatch(path.name, ex) for ex in exclude_patterns):
        return "excluded"
    if is_under_tracking_root(path.parts):
        return "tracking"
    if is_under_definitions_root(path.parts):
        return "definitions"
    if is_under_label_series(path):
        return "series"
    return None


@dataclass(frozen=True)
class SourceFileFilter:
    """The per-file rule of a tracks or labels scan, for one declared base.

    A recursive filter accepts a file whose path below :attr:`base` matches
    ``**/<pattern>`` for some pattern, which is what ``base.rglob(pattern)``
    returns. A non-recursive one accepts a path that matches ``<pattern>``,
    which is what ``base.glob(pattern)`` returns. A pattern may carry path
    parts (``a/*.csv``, ``**/b/*.npz``). The skips apply on top: a basename that
    matches an exclude pattern, a macOS resource fork, a path through
    ``_tracking`` or ``feature_definitions``, and a file inside a label series.

    The walk (:func:`iter_track_files`) lists each pattern's glob and keeps the
    candidates :meth:`matches_below` accepts. :meth:`accepts` answers for a
    resolved path by its own name and against :attr:`base`, so a file symlink
    whose target has another name, or a file reached through a symlinked folder
    outside :attr:`base`, is walked but not accepted.

    Attributes:
        base: The source's declared directory, resolved by
            :func:`source_file_filter`. A path is judged relative to it.
        patterns: The globs of the source.
        recursive: Whether a pattern is searched at every depth below *base*.
        exclude_patterns: Basename globs the scan leaves out.
    """

    base: Path
    patterns: tuple[str, ...]
    recursive: bool
    exclude_patterns: tuple[str, ...]

    def matches_below(self, relative: PurePath) -> bool:
        """Whether *relative*, a path below :attr:`base`, fits a pattern.

        Looks at the path's text only. The skips need the file system and
        belong to :meth:`accepts` and the walk.
        """
        return any(
            relative.full_match(f"**/{pattern}" if self.recursive else pattern)
            for pattern in self.patterns
        )

    def accepts(self, resolved: Path) -> bool:
        """Whether a scan of this source indexes *resolved*.

        Args:
            resolved: An absolute, resolved path. One outside :attr:`base` is
                not accepted.
        """
        try:
            relative = resolved.relative_to(self.base)
        except ValueError:
            return False
        return (
            self.matches_below(relative)
            and _skip_reason(resolved, self.exclude_patterns) is None
        )

    def candidates(self, directory: Path) -> Iterator[Path]:
        """The paths below *directory* that a glob of a pattern lists.

        Each pattern is listed by its own glob, ``rglob`` for a recursive filter
        and ``glob`` for a non-recursive one, so only the paths a pattern can
        match are listed and a glob walks through a symlinked folder as it
        always has. A path matched by several patterns is listed once per
        pattern. :meth:`matches_below` is the filter over what this yields.
        """
        for pattern in self.patterns:
            yield from (
                directory.rglob(pattern) if self.recursive else directory.glob(pattern)
            )


def source_file_filter(
    base: Path,
    patterns: Sequence[str],
    *,
    recursive: bool,
    exclude_patterns: Sequence[str] = (),
) -> SourceFileFilter:
    """The per-file rule of a source declared over *base*.

    Args:
        base: The source's declared directory. The filter keeps it resolved,
            because :meth:`SourceFileFilter.accepts` takes resolved paths.
        patterns: The source's globs.
        recursive: Whether the patterns are searched at every depth.
        exclude_patterns: Basename globs to leave out.
    """
    return SourceFileFilter(
        base=base.resolve(),
        patterns=tuple(patterns),
        recursive=recursive,
        exclude_patterns=tuple(exclude_patterns),
    )


def iter_track_files(
    search_dirs: Iterable[Path],
    patterns: Sequence[str],
    *,
    recursive: bool = True,
    exclude_patterns: Sequence[str] = (),
    listed_skipped: list[Path] | None = None,
) -> list[tuple[Path, os.stat_result]]:
    """Scan *search_dirs* for raw-track files, deterministically.

    Yields ``(path, stat)`` for every file under *search_dirs* that
    :class:`SourceFileFilter` accepts for *patterns*: a match of any glob,
    skipping macOS resource forks (``._*``) and any basename matching
    *exclude_patterns*. A file matched by several patterns appears once
    (deduped by resolved path), and the result is sorted by resolved path so
    the written index is order-stable regardless of filesystem iteration order.
    The identity-free scan shared by :meth:`Dataset.index_tracks_raw`
    (scan-and-derive) and :meth:`Dataset.write_tracks_raw_index`
    (assignment-driven).

    An entry of *search_dirs* that is a file is a listed file, which a file
    source hands over in place of a directory. A listed file is claimed
    whatever a glob says, so *patterns* do not apply to it. It is kept unless a
    rule below skips it, and a skipped one is appended to *listed_skipped*, so
    the caller can say that a listed file was not indexed.

    **``_tracking`` is never descended into**. A tracker writes per-individual
    ``data/*.npz`` and predictions CSVs into its working directory, and this
    scan looks for exactly those extensions; reaching them would index
    generated intermediates as user raw tracks and fold their checksums into a
    sequence's ``tracks_raw`` composition, silently invalidating every
    downstream cache on each tracker re-run.

    It is a *component* test rather than an ``exclude_patterns`` entry, because
    ``exclude_patterns`` matches basenames and a directory exclusion cannot be
    written as one -- ``_tracking`` never appears as a file's name, so the glob
    would match nothing while the walk descended anyway. That is not a gap
    callers could have worked around; it was unexpressible.

    **A label series is never descended into either**, for the sibling reason.
    A series holds an editor's saved revisions, indexed one row per revision
    under their own rule. Read as uploaded label files, a projected scoring
    export would become a second, conflicting source for its sequence, and its
    checksum would move that sequence's composition on every save. It is
    recognized by the marker the series writer leaves, not by a directory name,
    so a folder a user happened to call ``keypoints`` is still walked.

    Skips are counted rather than announced per file: on a real tracker root
    that would be thousands of lines. The count goes to stderr in one line,
    because a scan that quietly returns less than the caller expected is the
    thing worth noticing.
    """
    walk = _walk_track_files(
        search_dirs,
        patterns,
        recursive=recursive,
        exclude_patterns=exclude_patterns,
        listed_skipped=listed_skipped,
    )
    if walk.tracking_skipped:
        print(
            f"[INFO] skipped {walk.tracking_skipped} generated file(s) under "
            f"{TRACKING_ROOT}/ -- tracker intermediates are not raw tracks",
            file=sys.stderr,
        )
    if walk.series_skipped:
        print(
            f"[INFO] skipped {walk.series_skipped} file(s) inside a versioned label "
            "series -- those are indexed per revision, by a source that declares "
            "`series`",
            file=sys.stderr,
        )
    return [walk.by_resolved[key] for key in sorted(walk.by_resolved)]


def walked_files(
    base: Path,
    patterns: Sequence[str],
    *,
    recursive: bool,
    exclude_patterns: Sequence[str] = (),
) -> frozenset[Path]:
    """The resolved paths of the files a walk of *base* finds.

    The walk of :func:`iter_track_files` for one directory source, without its stderr
    report. A file symlink resolves to its target, so a target with another name or
    outside *base* is in the set, which :meth:`SourceFileFilter.accepts` alone would not
    cover.
    """
    walk = _walk_track_files(
        [base], patterns, recursive=recursive, exclude_patterns=exclude_patterns
    )
    return frozenset(walk.by_resolved)


def walked_paths(
    search_dirs: Iterable[Path],
    patterns: Sequence[str],
    *,
    recursive: bool,
    exclude_patterns: Sequence[str] = (),
) -> list[Path]:
    """The paths a scan of *search_dirs* walks, sorted by resolved path.

    The files :func:`iter_track_files` returns, as the paths the walk found
    them at and without its stderr report, so a caller can ask of each what a
    scan would ask without writing an index.
    """
    walk = _walk_track_files(
        search_dirs, patterns, recursive=recursive, exclude_patterns=exclude_patterns
    )
    return [walk.by_resolved[key][0] for key in sorted(walk.by_resolved)]


@dataclass
class _TrackFileWalk:
    by_resolved: dict[Path, tuple[Path, os.stat_result]]
    tracking_skipped: int = 0
    series_skipped: int = 0


def _walk_track_files(
    search_dirs: Iterable[Path],
    patterns: Sequence[str],
    *,
    recursive: bool,
    exclude_patterns: Sequence[str],
    listed_skipped: list[Path] | None = None,
) -> _TrackFileWalk:
    """The files :func:`iter_track_files` returns, by resolved path, and its skips."""
    by_resolved: dict[Path, tuple[Path, os.stat_result]] = {}
    skipped = 0
    series_skipped = 0
    for directory in search_dirs:
        listed = directory.is_file()
        if listed:
            matches: Iterable[Path] = [directory]
        else:
            file_filter = source_file_filter(
                directory,
                patterns,
                recursive=recursive,
                exclude_patterns=exclude_patterns,
            )
            matches = (
                path
                for path in file_filter.candidates(directory)
                if file_filter.matches_below(path.relative_to(directory))
            )
        for path in matches:
            if not path.is_file():
                continue
            skip = _skip_reason(path, exclude_patterns)
            if skip is not None:
                if listed:
                    if listed_skipped is not None:
                        listed_skipped.append(path)
                elif skip == "tracking":
                    skipped += 1
                elif skip == "series":
                    series_skipped += 1
                continue
            resolved = path.resolve()
            if resolved in by_resolved:
                continue
            by_resolved[resolved] = (path, path.stat())
    return _TrackFileWalk(
        by_resolved=by_resolved,
        tracking_skipped=skipped,
        series_skipped=series_skipped,
    )


def _path_below(root: Path, path: Path) -> str:
    """*path* relative to *root* with ``/`` separators, or its name if elsewhere."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def stem_as_sequence(stem: str) -> str:
    """One file is one sequence, and the stem names it.

    The naming rule for a source that has no rule of its own: every label source,
    and every track format whose converter does not declare otherwise.
    """
    return stem


SequenceRule = Callable[[Path], str | None]
"""A function of a file's path that names its sequence, or ``None`` when it names none."""


class SequenceCaptureError(ValueError):
    """A ``sequence_path_pattern`` matched a path with a capture that names nothing.

    A group that did not take part in the match, or that matched the empty
    string, would name a blank sequence. The report of the raw files catches this
    class and no other, so a file the scan would refuse is listed as a miss.
    """


def source_label(source: str) -> str:
    """How a refusal names a scan's source: its id, or that there is none."""
    return f"source {source!r}" if source else "a scan of no declared source"


def sequence_rule(
    *,
    sequence_from_stem: Callable[[str], str],
    sequence_path_pattern: str | None,
    path_root: Path | None,
    source: str,
) -> SequenceRule:
    """The one rule that names a file's sequence, from a path pattern or from its stem.

    With *sequence_path_pattern*, the rule searches the path of a file below
    *path_root*, with ``/`` separators, and joins the pattern's capture groups
    with ``__`` in order. A path the pattern does not match has no sequence
    (``None``). Without it, the rule is *sequence_from_stem* of the file's stem,
    and never misses. The scan and the raw report build their rule here, so they
    name one sequence.

    Args:
        sequence_from_stem: The converter's rule for a stem, used without a
            pattern.
        sequence_path_pattern: A regular expression with at least one capture
            group, or ``None``.
        path_root: The folder the pattern is searched below. Required with a
            pattern.
        source: The id of the source scanned, for the messages of a refusal, or
            ``""`` for a scan of no declared source.

    Raises:
        ValueError: A pattern is given without *path_root* or without a capture
            group, or is not a regular expression.
        SequenceCaptureError: From the rule, when the pattern matches a path but
            a capture group did not take part or matched the empty string. The
            message names the file, the source and the pattern.
    """
    if sequence_path_pattern is None:
        return lambda path: sequence_from_stem(path.stem)
    if path_root is None:
        msg = (
            f"sequence_path_pattern {sequence_path_pattern!r} of "
            f"{source_label(source)} needs group_path_root, the folder it is "
            "searched below"
        )
        raise ValueError(msg)
    try:
        compiled = re.compile(sequence_path_pattern)
    except re.error as error:
        msg = (
            f"sequence_path_pattern {sequence_path_pattern!r} of "
            f"{source_label(source)} is not a regular expression: {error}"
        )
        raise ValueError(msg) from error
    if compiled.groups < 1:
        msg = (
            f"sequence_path_pattern {sequence_path_pattern!r} of "
            f"{source_label(source)} has no capture group, so it names no sequence"
        )
        raise ValueError(msg)

    def name_from_path(path: Path) -> str | None:
        found = compiled.search(_path_below(path_root, path))
        if found is None:
            return None
        captures = found.groups()
        if any(not capture for capture in captures):
            msg = (
                f"sequence_path_pattern {sequence_path_pattern!r} of "
                f"{source_label(source)} matches {path} but a capture group "
                f"names nothing (captures {captures!r}: one did not take part in "
                "the match or matched the empty string), so it names no sequence"
            )
            raise SequenceCaptureError(msg)
        return "__".join(capture for capture in captures if capture)

    return name_from_path


def group_for_path(
    path: Path,
    *,
    sequence_rule: SequenceRule,
    multi_sequences_per_file: bool,
    group_from: str | None,
    group_re: re.Pattern[str] | None,
    path_re: re.Pattern[str] | None,
    path_root: Path | None,
    group_from_path: Callable[[Path], str] | None,
) -> str | None:
    """The group a scan gives *path*, or ``None`` when its declared pattern misses it.

    A scan indexes a file whose group is a string and skips one whose group is
    ``None``. A file whose sequence *sequence_rule* does not name has no group
    and is skipped, whatever else is declared; a file of a
    *multi_sequences_per_file* source has no sequence to name, and the rule is
    not asked. Otherwise at most one rule applies, in this order:

    * *group_from_path*, when given, names the group and never misses.
    * *path_re*, when given, searches the path of *path* below *path_root*.
    * Under *multi_sequences_per_file*, the group is the file's stem or its
      parent folder's name per *group_from*, or empty, and *group_re* does not
      apply.
    * *group_re*, when given, searches the sequence that *sequence_rule* names.
    * Otherwise the group is empty.

    A declared pattern that finds nothing is a miss.
    """
    sequence = "" if multi_sequences_per_file else sequence_rule(path)
    if sequence is None:
        return None
    if group_from_path is not None:
        return str(group_from_path(path))
    if path_re is not None and path_root is not None:
        found = path_re.search(_path_below(path_root, path))
        return None if found is None else found.group(1) or ""
    if multi_sequences_per_file:
        if group_from == "filename":
            return path.stem
        if group_from == "parent":
            return path.parent.name
        return ""
    if group_re is not None:
        found = group_re.search(sequence)
        return None if found is None else found.group(1) or ""
    return ""


class RawEntryNamedTwice(ValueError):
    """Two raw files of one format name one entry, and a conversion would keep one.

    A conversion writes one table per entry and takes the first file that names
    it, so the recording of the other would be lost without a message.

    Attributes:
        entry: The ``(group, sequence)`` both files name.
        files: The two files, by resolved path, in path order.
        sources: How each file's source is described, in the order of *files*.
    """

    def __init__(
        self,
        entry: tuple[str, str],
        files: tuple[Path, Path],
        sources: tuple[str, str],
    ) -> None:
        self.entry = entry
        self.files = files
        self.sources = sources
        first, second = files
        super().__init__(
            f"{first} ({sources[0]}) and {second} ({sources[1]}) both name the "
            f"entry {make_entry_key(*entry)!r}, and a conversion would keep one "
            "of them and lose the other recording. Give them distinct names, "
            "narrow the source's patterns so one is not scanned, or name the "
            "sequence from the path with sequence_path_pattern."
        )


@dataclass(frozen=True)
class EntryNamedTwice:
    """An entry that two or more distinct files name.

    Attributes:
        entry: The ``(group, sequence)`` the files name.
        files: Every such file by resolved path, in path order, at least two.
    """

    entry: tuple[str, str]
    files: tuple[Path, ...]


def entries_named_twice(
    rows: Iterable[Mapping[str, object]],
    *,
    one_file_per_sequence: bool,
    resolve: Callable[[str], Path],
) -> list[EntryNamedTwice]:
    """The entries that two distinct files of *rows* name, in entry order.

    *rows* are the rows of one format. A row with no sequence names no entry (it
    is the ``multi_sequences_per_file`` shape, where the group names the file's
    sequences), and neither does a promoted correction, which promotion indexes
    under its upload's entry and a conversion keeps apart
    (:func:`~mosaic.core.pipeline.promotion.correction_revision`). Files are
    counted by their resolved path, so a symlink and its target, or a row
    listed twice, are one file. The result is the same for any order of *rows*.

    Args:
        rows: Rows with ``group``, ``sequence`` and ``abs_path`` cells.
        one_file_per_sequence: Whether the format's converter reads one file
            per sequence. ``False`` for a format that merges several files into
            one sequence or holds several sequences in one file, which names an
            entry twice by design: nothing is reported.
        resolve: Resolves a row's stored ``abs_path`` to the file it names.
    """
    if not one_file_per_sequence:
        return []
    named: dict[tuple[str, str], set[Path]] = {}
    for row in rows:
        sequence = str(row.get("sequence", "") or "")
        stored = str(row.get("abs_path", "") or "").strip()
        if not sequence or not stored or correction_revision(stored):
            continue
        entry = (str(row.get("group", "") or ""), sequence)
        named.setdefault(entry, set()).add(resolve(stored))
    return [
        EntryNamedTwice(entry=entry, files=tuple(sorted(files)))
        for entry, files in sorted(named.items())
        if len(files) > 1
    ]


def write_tracks_raw_index_rows(index_path: Path, df: pd.DataFrame) -> None:
    """Atomically write *df* projected onto ``TRACKS_RAW_INDEX_COLUMNS``.

    The projection fixes column order and drops any column outside the schema;
    the atomic write guarantees a concurrent reader never sees a partial file.
    """
    atomic_write(
        index_path, lambda p: df[TRACKS_RAW_INDEX_COLUMNS].to_csv(p, index=False)
    )


def build_tracks_raw_row(
    *,
    path: Path,
    stat: os.stat_result,
    to_store_path: Callable[[Path], str],
    group: str,
    sequence: str,
    src_format: str,
    md5: str = "",
    companions: Sequence[CompanionRecord] = (),
) -> TracksRawIndexRow:
    """Assemble one raw-tracks-index row.

    *stat* is the file's ``os.stat_result`` (size and mtime come from it).
    ``abs_path`` is produced by *to_store_path*, so the in-tree-relative /
    out-of-tree-absolute rule is enforced in one place. *md5* is passed by the
    caller (empty unless hashing was requested) rather than computed here, so
    this stays a stat-only, read-light assembler. *companions* are written as
    the row's ``companions`` cell.
    """
    return {
        "group": validate_entry_name(group, "group"),
        "sequence": validate_entry_name(sequence, "sequence"),
        "abs_path": to_store_path(path),
        "src_format": src_format,
        "size_bytes": stat.st_size,
        "mtime_iso": mtime_iso(stat.st_mtime),
        "md5": md5,
        "companions": companions_cell(companions),
    }


def source_members_from_rows(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, str], list[SourceMember]]:
    """Group raw-source rows into per-sequence composition members.

    The tracks_raw counterpart of ``media_members_from_rows``, and beside this
    schema for the same reason.

    ``name`` is the stored path's basename rather than the path itself, so the
    composition survives the dataset moving between machines -- a hashed value
    cannot be rewritten by the portability passes the way ``abs_path`` is.

    A row whose ``sequence`` is blank is **skipped**: that is the
    ``multi_sequences_per_file`` shape, where the grouping lives in ``group`` and
    the file covers many sequences at once. Composing them under one key
    ``(group, "")`` would mint a value describing no sequence in particular, and
    a fiction is worse here than an absence.

    A row's companions follow its own member, one member per companion name
    within a sequence: the files of a merged sequence sit beside each other, and
    one companion listed by several of them is one file. A companion without a
    digest leaves the composition incomplete, as a source without one does. A row
    without companions adds none, so the composition of a sequence whose converter
    lists none is what its files alone give.
    """
    members: dict[tuple[str, str], list[SourceMember]] = {}
    companion_names: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        sequence = str(row.get("sequence", "") or "")
        if not sequence:
            continue
        key = (str(row.get("group", "") or ""), sequence)
        stored = str(row.get("abs_path", "") or "")
        found = members.setdefault(key, [])
        found.append(
            SourceMember(
                name=Path(stored).name,
                digest=str(row.get("md5", "") or ""),
                algo="md5",
            )
        )
        seen = companion_names.setdefault(key, set())
        for companion in companions_of(row):
            if companion.name not in seen:
                seen.add(companion.name)
                found.append(
                    SourceMember(name=companion.name, digest=companion.md5, algo="md5")
                )
    return members
