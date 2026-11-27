"""What a dataset's indexes and run-logs already say about its raw data.

:func:`raw_report` answers six questions that are otherwise a script over the
raw files: which files are copies of one another, which were never hashed,
which files a sequence or group pattern leaves out, which tracks tables
disagree with their media about the frame axis, which are far longer or shorter
than their group's others, and which entries a method read with uneven time
steps.

It is a read. It opens the raw indexes, the tracks index and the run-logs, and
lists the directories of the declared tracks and labels sources by file name.
It opens no tracks table and no raw file, writes nothing and creates nothing: a
source whose path is missing is reported and not walked.

The module loads no pandas on import. The index readers and the manifest, which
load it, are imported inside the functions that read them.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Final, Literal

from pydantic import Field

from mosaic.core.scan_findings import RawKind, missing_path_finding
from mosaic.core.strict_model import StrictModel
from mosaic.runlog import read_runs, read_time_irregular, run_log_dir

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "RAW_REPORT_SAMPLE",
    "DuplicateSet",
    "FrameAxisRow",
    "MissingSource",
    "PatternMiss",
    "RawReport",
    "SpanOutlier",
    "TimeIrregularEntry",
    "UnhashedRaw",
    "raw_report",
]

RAW_REPORT_SAMPLE: Final = 10
"""The most items a list of the report keeps. Its count is of the whole."""

DUPLICATE_PATH_SAMPLE: Final = 5
"""The most paths a set of duplicates names."""

_KINDS: Final[tuple[RawKind, ...]] = ("tracks", "labels")
_KIND_ORDER: Final[dict[RawKind, int]] = {"tracks": 0, "labels": 1}


class DuplicateSet(StrictModel):
    """Raw files that share one checksum.

    Attributes:
        md5: The shared digest.
        count: How many distinct files carry it.
        sources: The declared sources that claim those files, as ``kind:id``. A
            file that no source claims names none.
        paths: Up to five of the files, in path order.
    """

    md5: str
    count: int
    sources: list[str]
    paths: list[str]


class UnhashedRaw(StrictModel):
    """The raw rows of one kind whose checksum is blank.

    A source declared with ``md5: false`` records no digest, so its files
    cannot be compared for duplicates and its composition is unestablished.

    Attributes:
        kind: Which raw index the rows are in.
        count: How many files have a blank checksum.
        paths: Up to ``RAW_REPORT_SAMPLE`` of them, in path order.
    """

    kind: RawKind
    count: int
    paths: list[str]


class PatternMiss(StrictModel):
    """The files a source's pattern names no sequence or group for.

    A source reports one miss per pattern that misses: its
    ``sequence_path_pattern``, its ``group_path_pattern``, or its
    ``group_pattern``. A scan skips such a file, so no index row records
    it.

    Attributes:
        kind: The kind of the source.
        source: The source's id.
        pattern: The pattern that missed, one of ``sequence_path_pattern``,
            ``group_path_pattern`` and ``group_pattern``. Its files are
            those it names no sequence or group for: no match, or, for the
            sequence pattern, a capture that names nothing, which a scan
            refuses. A file the sequence pattern misses is not counted
            against the group pattern.
        count: How many files the pattern misses.
        paths: Up to ``RAW_REPORT_SAMPLE`` of them, in path order.
    """

    kind: RawKind
    source: str
    pattern: str
    count: int
    paths: list[str]


class MissingSource(StrictModel):
    """A declared source whose path does not exist, such as an unmounted share.

    Attributes:
        kind: The kind of the source.
        source: The source's id.
        path: The path it declares, resolved against the dataset.
    """

    kind: RawKind
    source: str
    path: str


class FrameAxisRow(StrictModel):
    """One tracks table whose tool read other frames than its media holds.

    Attributes:
        verdict: ``mismatch`` for a disagreement, ``tail_short`` for a
            shortfall within what the producer's tool is known to lose at the
            end of a file.
        run_id: The tracks variant.
        group: The entry's group.
        sequence: The entry's sequence.
        producer: The row's producer.
        frames_read: The frames the tool read.
        media_frames: The frames the media holds.
    """

    verdict: Literal["mismatch", "tail_short"]
    run_id: str
    group: str
    sequence: str
    producer: str
    frames_read: int
    media_frames: int


class SpanOutlier(StrictModel):
    """One tracks table whose span is under half or over twice its peers'.

    Attributes:
        run_id: The tracks variant.
        group: The entry's group.
        sequence: The entry's sequence.
        span: ``frame_max - frame_min + 1`` of the row.
        median_span: The median span of the rows of its group and variant.
    """

    run_id: str
    group: str
    sequence: str
    span: int
    median_span: float


class TimeIrregularEntry(StrictModel):
    """One entry whose uneven time steps a method read as even.

    Attributes:
        entry: The entry, as the run-log names it.
        count: The uneven-time events recorded for it across run-logs.
        largest_step: The longest step in seconds of any of them.
        execution_id: The latest attempt that recorded one.
        recorded_at: When that attempt recorded it.
    """

    entry: str
    count: int
    largest_step: float
    execution_id: str
    recorded_at: str


class RawReport(StrictModel):
    """What the indexes and run-logs record about a dataset's raw data.

    Every list holds at most ``RAW_REPORT_SAMPLE`` items. The count beside it
    is of the whole, so a list shorter than its count was cut.

    Attributes:
        duplicates: Sets of raw files with one non-blank checksum, across the
            tracks and labels indexes, largest first.
        duplicate_sets: How many sets there are.
        unhashed: Per kind, the rows with a blank checksum.
        pattern_misses: Per pattern of a source that declares a sequence path
            pattern or a group pattern, the files it misses. A tracks source
            whose ``src_format`` has no registered converter is not checked,
            because its stem rule is unknown.
        pattern_miss_sources: How many source patterns miss a file. A source
            whose sequence and group patterns both miss counts twice.
        frame_axis: The tracks tables whose frame counts disagree with their
            media.
        frame_axis_tables: How many tables disagree.
        span_outliers: The tracks tables with an outlying frame span.
        span_outlier_tables: How many tables are outliers.
        time_irregular: The entries that a method read with uneven time steps.
        time_irregular_entries: How many entries there are.
        missing_sources: The declared tracks and labels sources with no path.
        missing_source_paths: How many sources have none.
    """

    duplicates: list[DuplicateSet] = Field(default_factory=list)
    duplicate_sets: int = 0
    unhashed: list[UnhashedRaw] = Field(default_factory=list)
    pattern_misses: list[PatternMiss] = Field(default_factory=list)
    pattern_miss_sources: int = 0
    frame_axis: list[FrameAxisRow] = Field(default_factory=list)
    frame_axis_tables: int = 0
    span_outliers: list[SpanOutlier] = Field(default_factory=list)
    span_outlier_tables: int = 0
    time_irregular: list[TimeIrregularEntry] = Field(default_factory=list)
    time_irregular_entries: int = 0
    missing_sources: list[MissingSource] = Field(default_factory=list)
    missing_source_paths: int = 0


@dataclass(frozen=True)
class _RawFile:
    """One raw index row: its kind, where its file is, and its checksum."""

    kind: RawKind
    path: Path
    md5: str


def raw_report(ds: Dataset) -> RawReport:
    """Build the raw report of *ds* from its indexes and run-logs.

    Reads the raw indexes, the tracks index and the run-logs, and lists the
    directory of each declared tracks or labels source by file name. Opens no
    tracks table or raw file, and writes and creates nothing.

    Args:
        ds: The dataset to report on.

    Returns:
        The report, with each list cut to ``RAW_REPORT_SAMPLE`` items and each
        count taken over the whole.
    """
    files = [*_raw_files(ds, "tracks"), *_raw_files(ds, "labels")]
    duplicates, duplicate_sets = _duplicates(ds, files)
    misses, missing = _source_findings(ds)
    frame_axis, span_outliers = _tracks_findings(ds)
    time_irregular = _time_irregular(ds)
    return RawReport(
        duplicates=duplicates,
        duplicate_sets=duplicate_sets,
        unhashed=_unhashed(files),
        pattern_misses=misses[:RAW_REPORT_SAMPLE],
        pattern_miss_sources=len(misses),
        frame_axis=frame_axis[:RAW_REPORT_SAMPLE],
        frame_axis_tables=len(frame_axis),
        span_outliers=span_outliers[:RAW_REPORT_SAMPLE],
        span_outlier_tables=len(span_outliers),
        time_irregular=time_irregular[:RAW_REPORT_SAMPLE],
        time_irregular_entries=len(time_irregular),
        missing_sources=missing[:RAW_REPORT_SAMPLE],
        missing_source_paths=len(missing),
    )


def _raw_files(ds: Dataset, kind: RawKind) -> list[_RawFile]:
    """The rows of the *kind* raw index, with each stored path resolved."""
    from mosaic.core.pipeline.tracks_raw_index import read_tracks_raw_index

    root = f"{kind}_raw"
    if not ds.has_root(root):
        return []
    rows = read_tracks_raw_index(ds.get_root(root) / "index.csv")
    rows = [row for row in rows if (row.get("abs_path") or "").strip()]
    paths = ds.resolve_paths([row["abs_path"].strip() for row in rows])
    return [
        _RawFile(kind=kind, path=path, md5=(row.get("md5") or "").strip())
        for row, path in zip(rows, paths, strict=True)
    ]


def _identity(path: Path) -> Path:
    """The path of one file whatever link reached it, as far as it resolves."""
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return path


def _duplicates(ds: Dataset, files: list[_RawFile]) -> tuple[list[DuplicateSet], int]:
    """The sets of distinct files sharing a checksum, and how many there are.

    Only the sets kept are matched against the declared sources, so a dataset
    with no duplicates builds no claim.
    """
    by_digest: dict[str, dict[Path, set[RawKind]]] = defaultdict(dict)
    for raw in files:
        if raw.md5:
            kinds = by_digest[raw.md5].setdefault(_identity(raw.path), set())
            kinds.add(raw.kind)
    sets = {md5: found for md5, found in by_digest.items() if len(found) > 1}
    ordered = sorted(sets, key=lambda md5: (-len(sets[md5]), md5))
    claiming = {kind: ds.sources_claiming(kind) for kind in _KINDS}
    kept = [
        DuplicateSet(
            md5=md5,
            count=len(sets[md5]),
            sources=sorted(
                {
                    f"{kind}:{source}"
                    for path, kinds in sets[md5].items()
                    for kind in kinds
                    for source in claiming[kind](path)
                }
            ),
            paths=[str(path) for path in sorted(sets[md5])[:DUPLICATE_PATH_SAMPLE]],
        )
        for md5 in ordered[:RAW_REPORT_SAMPLE]
    ]
    return kept, len(sets)


def _unhashed(files: list[_RawFile]) -> list[UnhashedRaw]:
    """Per kind, the files whose row records no checksum."""
    report: list[UnhashedRaw] = []
    for kind in _KINDS:
        blank = sorted(
            {_identity(raw.path) for raw in files if raw.kind == kind and not raw.md5}
        )
        if blank:
            report.append(
                UnhashedRaw(
                    kind=kind,
                    count=len(blank),
                    paths=[str(path) for path in blank[:RAW_REPORT_SAMPLE]],
                )
            )
    return report


def _sequence_rule(kind: RawKind, src_format: str) -> Callable[[str], str] | None:
    """How *src_format* names a sequence from a stem, ``None`` if unknown."""
    from mosaic.core.pipeline.tracks_raw_index import stem_as_sequence
    from mosaic.core.track_converter import get_track_converter

    if kind == "labels":
        return stem_as_sequence
    try:
        return get_track_converter(src_format).sequence_from_stem
    except ValueError:
        return None


def _source_findings(ds: Dataset) -> tuple[list[PatternMiss], list[MissingSource]]:
    """The files each pattern misses and the sources with no path.

    Each source is judged by the rules a scan applies (``sequence_rule`` and
    ``group_for_path``) to the files a scan would walk. A file whose sequence
    path pattern names no sequence is a miss of that pattern, and so is one
    whose capture names nothing, which a scan refuses. The report raises
    neither. A source whose path is missing is reported and not walked, and
    so is a source whose format has no registered converter, which cannot say
    how its files name a sequence.
    """
    from mosaic.core.pipeline.tracks_raw_index import (
        SequenceCaptureError,
        group_for_path,
        sequence_rule,
        walked_paths,
    )

    misses: list[PatternMiss] = []
    missing: list[MissingSource] = []
    for kind in _KINDS:
        for source in ds.raw_index_sources(kind):
            base = ds.resolve_source_path(source)
            absent = missing_path_finding(kind, source.id, base)
            if absent is not None:
                missing.append(
                    MissingSource(kind=kind, source=source.id, path=str(base))
                )
                continue
            group_declared = source.group_path_pattern or source.group_pattern
            stem_rule = _sequence_rule(kind, source.src_format)
            if (
                source.sequence_path_pattern is None and group_declared is None
            ) or stem_rule is None:
                continue
            name_sequence = sequence_rule(
                sequence_from_stem=stem_rule,
                sequence_path_pattern=source.sequence_path_pattern,
                path_root=base,
                source=source.id,
            )
            group_re = (
                re.compile(source.group_pattern) if source.group_pattern else None
            )
            path_re = (
                re.compile(source.group_path_pattern)
                if source.group_path_pattern
                else None
            )
            search = (
                [base / entry for entry in source.files] if source.files else [base]
            )
            sequence_missed: list[str] = []
            group_missed: list[str] = []
            for path in walked_paths(
                search,
                source.patterns,
                recursive=source.recursive,
                exclude_patterns=source.exclude_patterns,
            ):
                try:
                    named = name_sequence(path)
                except SequenceCaptureError:
                    named = None
                if named is None:
                    sequence_missed.append(str(path))
                elif (
                    group_for_path(
                        path,
                        sequence_rule=name_sequence,
                        multi_sequences_per_file=source.multi_sequences_per_file,
                        group_from=source.group_from,
                        group_re=group_re,
                        path_re=path_re,
                        path_root=base,
                        group_from_path=None,
                    )
                    is None
                ):
                    group_missed.append(str(path))
            for pattern, missed in (
                (source.sequence_path_pattern, sequence_missed),
                (group_declared, group_missed),
            ):
                if pattern is not None and missed:
                    missed.sort()
                    misses.append(
                        PatternMiss(
                            kind=kind,
                            source=source.id,
                            pattern=pattern,
                            count=len(missed),
                            paths=missed[:RAW_REPORT_SAMPLE],
                        )
                    )
    misses.sort(key=lambda miss: (_KIND_ORDER[miss.kind], miss.source))
    missing.sort(key=lambda gone: (_KIND_ORDER[gone.kind], gone.source))
    return misses, missing


def _tracks_findings(ds: Dataset) -> tuple[list[FrameAxisRow], list[SpanOutlier]]:
    """The frame-axis disagreements and the span outliers of the tracks index.

    Reads the index once. Each row answers for itself, so an entry held by
    several variants appears once per variant.
    """
    from mosaic.core.helpers import text_cell
    from mosaic.core.pipeline.tracks_index import (
        frame_axis_findings,
        read_frame_extent,
        tracks_rows,
    )

    if not ds.has_root("tracks"):
        return [], []
    rows = tracks_rows(ds)
    producers: dict[tuple[str, str, str], str] = {}
    spans: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for _, row in rows.iterrows():
        run_id, group, sequence = (
            str(row["run_id"]),
            str(row["group"]),
            str(row["sequence"]),
        )
        producers[(run_id, group, sequence)] = text_cell(row.get("producer"))
        extent = read_frame_extent(row)
        if extent is not None:
            spans[(run_id, group)].append((sequence, extent[1] - extent[0] + 1))
    findings = frame_axis_findings(rows)
    frame_axis = [
        FrameAxisRow(
            verdict="mismatch",
            run_id=found.run_id,
            group=found.group,
            sequence=found.sequence,
            producer=producers.get((found.run_id, found.group, found.sequence), ""),
            frames_read=found.read,
            media_frames=found.media,
        )
        for found in findings.mismatches
    ] + [
        FrameAxisRow(
            verdict="tail_short",
            run_id=found.run_id,
            group=found.group,
            sequence=found.sequence,
            producer=found.producer,
            frames_read=found.read,
            media_frames=found.media,
        )
        for found in findings.tail_shortfalls
    ]
    frame_axis.sort(key=lambda row: (row.run_id, row.group, row.sequence))
    outliers: list[SpanOutlier] = []
    for (run_id, group), members in spans.items():
        typical = median(span for _, span in members)
        outliers.extend(
            SpanOutlier(
                run_id=run_id,
                group=group,
                sequence=sequence,
                span=span,
                median_span=float(typical),
            )
            for sequence, span in members
            if span < typical / 2 or span > typical * 2
        )
    outliers.sort(key=lambda row: (row.run_id, row.group, row.sequence))
    return frame_axis, outliers


def _time_irregular(ds: Dataset) -> list[TimeIrregularEntry]:
    """Per entry, the uneven-time events recorded across the run-logs.

    Each run-log is read once for its snapshot. The events are read from the
    snapshots that count any, so a run-log without one is read once and adds
    nothing.
    """
    logs = run_log_dir(ds.base_dir)
    if not logs.is_dir():
        return []
    counts: dict[str, int] = defaultdict(int)
    largest: dict[str, float] = {}
    latest: dict[str, tuple[str, str]] = {}
    for snapshot in read_runs(logs):
        if snapshot["entries_time_irregular"] <= 0:
            continue
        execution_id = snapshot["execution_id"]
        for record in read_time_irregular(logs, execution_id)["items"]:
            entry = record["key"]
            counts[entry] += 1
            largest[entry] = max(largest.get(entry, 0.0), record["largest_step"])
            latest[entry] = max(
                latest.get(entry, ("", "")), (record["recorded_at"], execution_id)
            )
    return [
        TimeIrregularEntry(
            entry=entry,
            count=counts[entry],
            largest_step=largest[entry],
            execution_id=latest[entry][1],
            recorded_at=latest[entry][0],
        )
        for entry in sorted(counts)
    ]
