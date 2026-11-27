"""The raw report: what the indexes and run-logs say about raw data.

Each field is built from real indexes and run-logs. The report is a read, so
the tests also pin what it never opens and what it never creates.
"""

from __future__ import annotations

import builtins
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from mosaic import runlog
from mosaic.core.dataset import Dataset, new_dataset_manifest, open_dataset
from mosaic.core.manifest import LabelsScanSource, TracksScanSource
from mosaic.core.raw_report import (
    RAW_REPORT_SAMPLE,
    DuplicateSet,
    FrameAxisRow,
    MissingSource,
    PatternMiss,
    RawReport,
    SpanOutlier,
    TimeIrregularEntry,
    UnhashedRaw,
    raw_report,
)
from mosaic.core.strict_model import StrictModel
from mosaic.runlog import run_log_dir
from tests.helpers import (
    add_tracks_table,
    copy_of,
    make_dataset,
    modules_loaded_by,
    track_table,
)


def _dataset(root: Path) -> Dataset:
    manifest = new_dataset_manifest(name="raw", base_dir=root / "ds")
    return Dataset(manifest_path=manifest).load()


def _write(path: Path, content: bytes = b"\x00") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(content)
    return path


def _write_all(paths: Iterable[Path], content: Callable[[Path], bytes]) -> None:
    for path in paths:
        _ = _write(path, content(path))


def _tracks_source(ds: Dataset, source_id: str, path: Path, **fields: object) -> None:
    ds.add_scan_source(
        TracksScanSource.model_validate(
            {"id": source_id, "path": str(path), "patterns": ["*.npy"], **fields}
        )
    )


def _labels_source(ds: Dataset, source_id: str, path: Path, **fields: object) -> None:
    ds.add_scan_source(
        LabelsScanSource.model_validate(
            {"id": source_id, "path": str(path), "patterns": ["*.csv"], **fields}
        )
    )


def _log_lines(*events: dict[str, object]) -> str:
    return "\n".join(json.dumps(event) for event in events) + "\n"


def _irregular(key: str, at: str, largest: float) -> dict[str, object]:
    return {
        "ev": "time_irregular",
        "t": at,
        "key": key,
        "who": "pair-wavelet",
        "largest_step": largest,
        "first_frame": 9,
        "last_frame": 10,
        "median_step": 0.03,
        "rate": 30.0,
    }


def _write_run_log(ds: Dataset, execution_id: str, *events: dict[str, object]) -> None:
    started: dict[str, object] = {
        "ev": "started",
        "t": "2026-11-01T00:00:00Z",
        "kind": "feature",
        "target": "x",
    }
    finished: dict[str, object] = {"ev": "finished", "t": "2026-11-01T00:09:00Z"}
    path = run_log_dir(ds.base_dir) / f"{execution_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(_log_lines(started, *events, finished))


def _untyped(function: Callable[..., object]) -> Callable[..., object]:
    """*function* as a callable that takes whatever a patch forwards to it."""
    return function


def _tree(base: Path) -> list[str]:
    return sorted(str(path.relative_to(base)) for path in base.rglob("*"))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path.resolve()


class TestDuplicates:
    def test_rows_of_one_digest_across_both_kinds_form_one_set(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        tracks, labels = root / "cams", root / "ann"
        _ = _write(tracks / "one.npy", b"same bytes")
        _ = _write(tracks / "other.npy", b"unique tracks")
        _ = _write(labels / "one.csv", b"same bytes")
        _tracks_source(ds, "cams", tracks)
        _labels_source(ds, "ann", labels)
        _ = ds.scan_tracks()
        _ = ds.scan_labels()

        report = raw_report(ds)

        assert report.duplicate_sets == 1
        [found] = report.duplicates
        assert found.count == 2
        assert len(found.md5) == 32
        assert found.sources == ["labels:ann", "tracks:cams"]
        assert found.paths == [
            str(labels / "one.csv"),
            str(tracks / "one.npy"),
        ]

    def test_a_set_names_five_paths_and_counts_every_row(self, root: Path) -> None:
        ds = _dataset(root)
        folder = root / "cams"
        _write_all([folder / f"c{i}.npy" for i in range(7)], lambda _: b"same")
        _tracks_source(ds, "cams", folder)
        _ = ds.scan_tracks()

        [found] = raw_report(ds).duplicates

        assert found.count == 7
        assert found.paths == [str(folder / f"c{i}.npy") for i in range(5)]

    def test_sets_are_ordered_by_size_then_digest(self, root: Path) -> None:
        ds = _dataset(root)
        folder = root / "cams"
        _write_all([folder / f"a{i}.npy" for i in range(2)], lambda _: b"pair")
        _write_all([folder / f"b{i}.npy" for i in range(3)], lambda _: b"triple")
        _write_all([folder / f"c{i}.npy" for i in range(2)], lambda _: b"duo")
        _tracks_source(ds, "cams", folder)
        _ = ds.scan_tracks()

        report = raw_report(ds)

        counts = [found.count for found in report.duplicates]
        assert counts == [3, 2, 2]
        pairs = report.duplicates[1:]
        assert [found.md5 for found in pairs] == sorted(found.md5 for found in pairs)
        assert report == raw_report(ds)

    def test_two_sources_over_distinct_folders_are_both_named(self, root: Path) -> None:
        ds = _dataset(root)
        _ = _write(root / "a" / "x.npy", b"copy")
        _ = _write(root / "b" / "y.npy", b"copy")
        _tracks_source(ds, "first", root / "a")
        _tracks_source(ds, "second", root / "b")
        _ = ds.scan_tracks()

        [found] = raw_report(ds).duplicates

        assert found.sources == ["tracks:first", "tracks:second"]

    def test_blank_digests_never_form_a_set(self, root: Path) -> None:
        ds = _dataset(root)
        _write_all([root / "raw" / f"u{i}.npy" for i in range(3)], lambda _: b"same")
        _tracks_source(ds, "raw", root / "raw", md5=False)
        _ = ds.scan_tracks()

        report = raw_report(ds)

        assert report.duplicates == []
        assert report.duplicate_sets == 0


class TestUnhashed:
    def test_rows_of_a_source_declared_without_md5_are_counted_per_kind(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        _write_all(
            [root / "t" / f"u{i}.npy" for i in range(3)], lambda p: p.name.encode()
        )
        _ = _write(root / "t2" / "hashed.npy", b"hashed")
        _ = _write(root / "l" / "u.csv", b"labels")
        _tracks_source(ds, "t", root / "t", md5=False)
        _tracks_source(ds, "t2", root / "t2")
        _labels_source(ds, "l", root / "l", md5=False)
        _ = ds.scan_tracks()
        _ = ds.scan_labels()

        report = raw_report(ds)

        assert report.unhashed == [
            UnhashedRaw(
                kind="tracks",
                count=3,
                paths=[str(root / "t" / f"u{i}.npy") for i in range(3)],
            ),
            UnhashedRaw(kind="labels", count=1, paths=[str(root / "l" / "u.csv")]),
        ]

    def test_a_kind_keeps_the_sample_and_counts_the_rest(self, root: Path) -> None:
        ds = _dataset(root)
        _write_all(
            [root / "t" / f"u{i:02d}.npy" for i in range(12)], lambda p: p.name.encode()
        )
        _tracks_source(ds, "t", root / "t", md5=False)
        _ = ds.scan_tracks()

        [unhashed] = raw_report(ds).unhashed

        assert unhashed.count == 12
        assert unhashed.paths == [
            str(root / "t" / f"u{i:02d}.npy") for i in range(RAW_REPORT_SAMPLE)
        ]


class TestPatternMisses:
    def test_a_stem_pattern_lists_the_files_it_misses(self, root: Path) -> None:
        ds = _dataset(root)
        folder = root / "raw"
        for name in ("hex_1.npy", "hex_2.npy", "plain_0.npy", "plain_1.npy"):
            _ = _write(folder / name, name.encode())
        _tracks_source(ds, "s", folder, group_pattern=r"^(hex)_")
        _ = ds.scan_tracks()

        report = raw_report(ds)

        assert report.pattern_misses == [
            PatternMiss(
                kind="tracks",
                source="s",
                pattern=r"^(hex)_",
                count=2,
                paths=[str(folder / "plain_0.npy"), str(folder / "plain_1.npy")],
            )
        ]
        assert report.pattern_miss_sources == 1

    def test_a_path_pattern_lists_the_files_it_misses(self, root: Path) -> None:
        ds = _dataset(root)
        folder = root / "raw"
        _ = _write(folder / "hex" / "a.npy", b"a")
        _ = _write(folder / "other" / "m.npy", b"m")
        _tracks_source(ds, "s", folder, group_path_pattern=r"^(hex)/")
        _ = ds.scan_tracks()

        [miss] = raw_report(ds).pattern_misses

        assert miss.pattern == r"^(hex)/"
        assert miss.paths == [str(folder / "other" / "m.npy")]

    def test_the_converters_sequence_decides_a_stem_pattern(self, root: Path) -> None:
        """TRex strips ``_fish0``, so ``^(hex_7)$`` matches the sequence."""
        ds = _dataset(root)
        folder = root / "raw"
        _write_all(
            [folder / "hex_7_fish0.npz", folder / "hex_7_fish1.npz"],
            lambda p: p.name.encode(),
        )
        _ = _write(folder / "oci_3_fish0.npz", b"oci")
        ds.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(folder),
                patterns=("*.npz",),
                src_format="trex_npz",
                group_pattern=r"^(hex_7)$",
            )
        )
        _ = ds.scan_tracks()

        [miss] = raw_report(ds).pattern_misses

        assert miss.count == 1
        assert miss.paths == [str(folder / "oci_3_fish0.npz")]

    def test_a_labels_source_matches_the_stem_itself(self, root: Path) -> None:
        ds = _dataset(root)
        folder = root / "ann"
        _ = _write(folder / "hex_1.csv", b"1")
        _ = _write(folder / "plain.csv", b"2")
        _labels_source(ds, "ann", folder, group_pattern=r"^(hex)_")
        _ = ds.scan_labels()

        [miss] = raw_report(ds).pattern_misses

        assert (miss.kind, miss.source, miss.count) == ("labels", "ann", 1)
        assert miss.paths == [str(folder / "plain.csv")]

    def test_a_source_with_no_pattern_or_no_miss_is_not_listed(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        _ = _write(root / "a" / "x.npy", b"x")
        _ = _write(root / "b" / "hex_1.npy", b"h")
        _tracks_source(ds, "plain", root / "a")
        _tracks_source(ds, "matched", root / "b", group_pattern=r"^(hex)_")
        _ = ds.scan_tracks()

        report = raw_report(ds)

        assert report.pattern_misses == []
        assert report.pattern_miss_sources == 0

    def test_a_pattern_missing_every_file_lists_them_all(self, root: Path) -> None:
        ds = _dataset(root)
        _write_all(
            [root / "raw" / f"p{i:02d}.npy" for i in range(12)],
            lambda p: p.name.encode(),
        )
        _tracks_source(ds, "s", root / "raw", group_pattern=r"^(hex)_")
        _ = ds.scan_tracks()

        [miss] = raw_report(ds).pattern_misses

        assert miss.count == 12
        assert miss.paths == [
            str(root / "raw" / f"p{i:02d}.npy") for i in range(RAW_REPORT_SAMPLE)
        ]

    def test_a_file_source_applies_the_pattern_to_its_listed_files(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        folder = root / "raw"
        _ = _write(folder / "hex_1.npy", b"1")
        _ = _write(folder / "plain.npy", b"2")
        _ = _write(folder / "unlisted.npy", b"3")
        ds.add_scan_source(
            TracksScanSource(
                id="s",
                path=str(folder),
                files=("hex_1.npy", "plain.npy"),
                group_pattern=r"^(hex)_",
            )
        )
        _ = ds.scan_tracks()

        [miss] = raw_report(ds).pattern_misses

        assert miss.paths == [str(folder / "plain.npy")]

    def test_a_source_with_only_a_sequence_path_pattern_lists_its_misses(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        folder = root / "raw"
        _ = _write(folder / "a" / "x.npy", b"a")
        _ = _write(folder / "b" / "y.npy", b"b")
        _ = _write(folder / "stray.npy", b"s")
        _tracks_source(ds, "s", folder, sequence_path_pattern=r"^([^/]+)/")
        _ = ds.scan_tracks()

        report = raw_report(ds)

        assert report.pattern_misses == [
            PatternMiss(
                kind="tracks",
                source="s",
                pattern=r"^([^/]+)/",
                count=1,
                paths=[str(folder / "stray.npy")],
            )
        ]
        assert sorted(str(row["sequence"]) for row in ds.read_tracks_raw_index()) == [
            "a",
            "b",
        ]

    def test_a_sequence_miss_is_not_counted_against_the_group_pattern(
        self, root: Path
    ) -> None:
        """The group pattern searches the sequence the path pattern named."""
        ds = _dataset(root)
        folder = root / "raw"
        _ = _write(folder / "a" / "x.npy", b"a")
        _ = _write(folder / "b" / "y.npy", b"b")
        _ = _write(folder / "stray.npy", b"s")
        _tracks_source(
            ds,
            "s",
            folder,
            sequence_path_pattern=r"^([^/]+)/",
            group_pattern=r"^(a)$",
        )
        _ = ds.scan_tracks()

        report = raw_report(ds)

        assert report.pattern_misses == [
            PatternMiss(
                kind="tracks",
                source="s",
                pattern=r"^([^/]+)/",
                count=1,
                paths=[str(folder / "stray.npy")],
            ),
            PatternMiss(
                kind="tracks",
                source="s",
                pattern=r"^(a)$",
                count=1,
                paths=[str(folder / "b" / "y.npy")],
            ),
        ]
        assert report.pattern_miss_sources == 2
        assert [
            (str(row["group"]), str(row["sequence"]))
            for row in ds.read_tracks_raw_index()
        ] == [("a", "a")]

    def test_more_misses_than_the_sample_gives_a_cut_list_beside_the_whole_count(
        self, root: Path
    ) -> None:
        """Six sources, each missing both of its patterns, make twelve misses."""
        ds = _dataset(root)
        for number in range(RAW_REPORT_SAMPLE // 2 + 1):
            folder = root / f"raw{number:02d}"
            _ = _write(folder / f"a{number:02d}" / "x.npy", b"a")
            _ = _write(folder / f"b{number:02d}" / "y.npy", b"b")
            _ = _write(folder / "stray.npy", b"s")
            _tracks_source(
                ds,
                f"s{number:02d}",
                folder,
                sequence_path_pattern=r"^([^/]+)/",
                group_pattern=r"^(a)\d+$",
            )
        _ = ds.scan_tracks()

        report = raw_report(ds)

        assert len(report.pattern_misses) == RAW_REPORT_SAMPLE
        assert report.pattern_miss_sources == 2 * (RAW_REPORT_SAMPLE // 2 + 1)

    def test_a_capture_that_names_nothing_is_a_miss_and_not_an_error(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        folder = root / "raw"
        _ = _write(folder / "a" / "x.npy", b"a")
        _tracks_source(ds, "s", folder, sequence_path_pattern=r"^(x*)")

        [miss] = raw_report(ds).pattern_misses

        assert miss.pattern == r"^(x*)"
        assert miss.paths == [str(folder / "a" / "x.npy")]


class TestFrameAxis:
    def test_each_disagreeing_table_is_listed_with_its_variant_and_producer(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        table = track_table("g", "short", n_rows=2)
        add_tracks_table(ds, "tool", table, media_frames=100, frames_read=95)
        add_tracks_table(
            ds,
            "tool",
            track_table("g", "tail", n_rows=2),
            media_frames=100,
            frames_read=95,
            known_tail_loss=5,
        )
        add_tracks_table(
            ds,
            "tool",
            track_table("g", "agrees", n_rows=2),
            media_frames=100,
            frames_read=100,
        )
        add_tracks_table(ds, "tool", track_table("g", "unmeasured", n_rows=2))
        add_tracks_table(ds, "other", table, media_frames=100, frames_read=101)

        report = raw_report(ds)

        assert report.frame_axis == [
            FrameAxisRow(
                verdict="mismatch",
                run_id="other",
                group="g",
                sequence="short",
                producer="other",
                frames_read=101,
                media_frames=100,
            ),
            FrameAxisRow(
                verdict="mismatch",
                run_id="tool",
                group="g",
                sequence="short",
                producer="tool",
                frames_read=95,
                media_frames=100,
            ),
            FrameAxisRow(
                verdict="tail_short",
                run_id="tool",
                group="g",
                sequence="tail",
                producer="tool",
                frames_read=95,
                media_frames=100,
            ),
        ]
        assert report.frame_axis_tables == 3

    def test_the_list_keeps_the_sample_and_counts_every_table(self, root: Path) -> None:
        ds = _dataset(root)
        for i in range(12):
            add_tracks_table(
                ds,
                "tool",
                track_table("g", f"s{i:02d}", n_rows=2),
                media_frames=100,
                frames_read=90,
            )

        report = raw_report(ds)

        assert report.frame_axis_tables == 12
        assert [row.sequence for row in report.frame_axis] == [
            f"s{i:02d}" for i in range(RAW_REPORT_SAMPLE)
        ]


class TestSpanOutliers:
    @staticmethod
    def _spans(ds: Dataset, variant: str, group: str, spans: dict[str, int]) -> None:
        for sequence, span in spans.items():
            add_tracks_table(
                ds, variant, track_table(group, sequence, n_rows=span), n_rows=span
            )

    def test_a_span_outside_half_to_twice_its_groups_median_is_listed(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        self._spans(
            ds,
            "trex",
            "g",
            {"a": 100, "b": 100, "c": 100, "short": 49, "long": 201, "edge": 50},
        )

        report = raw_report(ds)

        assert report.span_outliers == [
            SpanOutlier(
                run_id="trex", group="g", sequence="long", span=201, median_span=100.0
            ),
            SpanOutlier(
                run_id="trex", group="g", sequence="short", span=49, median_span=100.0
            ),
        ]
        assert report.span_outlier_tables == 2

    def test_a_group_is_compared_within_itself_and_within_a_variant(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        self._spans(ds, "trex", "g1", {"a": 100, "b": 100})
        self._spans(ds, "trex", "g2", {"a": 10, "b": 10})
        self._spans(ds, "other", "g1", {"a": 400})
        self._spans(ds, "trex", "solo", {"only": 7})

        assert raw_report(ds).span_outliers == []

    def test_the_list_keeps_the_sample_and_counts_every_table(self, root: Path) -> None:
        ds = _dataset(root)
        self._spans(ds, "trex", "g", {f"n{i:02d}": 100 for i in range(13)})
        self._spans(ds, "trex", "g", {f"s{i:02d}": 10 for i in range(12)})

        report = raw_report(ds)

        assert report.span_outlier_tables == 12
        assert [row.sequence for row in report.span_outliers] == [
            f"s{i:02d}" for i in range(RAW_REPORT_SAMPLE)
        ]


class TestTimeIrregular:
    def test_two_run_logs_naming_one_entry_give_one_item(self, root: Path) -> None:
        ds = _dataset(root)
        _write_run_log(
            ds,
            "01AAAAAAAAAAAAAAAAAAAAAAAA",
            _irregular("g__a", "2026-11-01T00:01:00Z", 1.45),
            _irregular("g__b", "2026-11-01T00:02:00Z", 0.2),
        )
        _write_run_log(
            ds,
            "01BBBBBBBBBBBBBBBBBBBBBBBB",
            _irregular("g__a", "2026-11-02T00:01:00Z", 0.4),
        )
        _write_run_log(ds, "01CCCCCCCCCCCCCCCCCCCCCCCC")

        report = raw_report(ds)

        assert report.time_irregular == [
            TimeIrregularEntry(
                entry="g__a",
                count=2,
                largest_step=1.45,
                execution_id="01BBBBBBBBBBBBBBBBBBBBBBBB",
                recorded_at="2026-11-02T00:01:00Z",
            ),
            TimeIrregularEntry(
                entry="g__b",
                count=1,
                largest_step=0.2,
                execution_id="01AAAAAAAAAAAAAAAAAAAAAAAA",
                recorded_at="2026-11-01T00:02:00Z",
            ),
        ]
        assert report.time_irregular_entries == 2

    def test_each_run_log_is_read_once_and_again_only_for_its_events(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(root)
        _write_run_log(
            ds,
            "01AAAAAAAAAAAAAAAAAAAAAAAA",
            _irregular("g__a", "2026-11-01T00:01:00Z", 1.45),
        )
        _write_run_log(ds, "01CCCCCCCCCCCCCCCCCCCCCCCC")
        reads: list[str] = []
        real = runlog.read_run_records

        def counted(run_dir: Path | str, execution_id: str) -> list[object]:
            reads.append(execution_id)
            return list(real(run_dir, execution_id))

        monkeypatch.setattr(runlog, "read_run_records", counted)

        _ = raw_report(ds)

        assert sorted(reads) == [
            "01AAAAAAAAAAAAAAAAAAAAAAAA",
            "01AAAAAAAAAAAAAAAAAAAAAAAA",
            "01CCCCCCCCCCCCCCCCCCCCCCCC",
        ]

    def test_the_list_keeps_the_sample_and_counts_every_entry(self, root: Path) -> None:
        ds = _dataset(root)
        _write_run_log(
            ds,
            "01AAAAAAAAAAAAAAAAAAAAAAAA",
            *(
                _irregular(f"g__s{i:02d}", "2026-11-01T00:01:00Z", 1.0)
                for i in range(12)
            ),
        )

        report = raw_report(ds)

        assert report.time_irregular_entries == 12
        assert [item.entry for item in report.time_irregular] == [
            f"g__s{i:02d}" for i in range(RAW_REPORT_SAMPLE)
        ]


class TestDuplicateSampleBound:
    def test_the_list_keeps_the_sample_and_counts_every_set(self, root: Path) -> None:
        ds = _dataset(root)
        folder = root / "cams"
        for i in range(12):
            _write_all(
                [folder / f"s{i:02d}a.npy", folder / f"s{i:02d}b.npy"],
                lambda _, i=i: f"set {i}".encode(),
            )
        _tracks_source(ds, "cams", folder)
        _ = ds.scan_tracks()

        report = raw_report(ds)

        assert report.duplicate_sets == 12
        assert len(report.duplicates) == RAW_REPORT_SAMPLE


class TestARead:
    def test_no_tracks_table_or_raw_file_is_opened(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _dataset(root)
        tracks, labels = root / "cams", root / "ann"
        _write_all([tracks / f"c{i}.npy" for i in range(3)], lambda _: b"same")
        _ = _write(tracks / "plain.npy", b"plain")
        _ = _write(labels / "one.csv", b"same")
        _tracks_source(ds, "cams", tracks, group_pattern=r"^(c)\d")
        _labels_source(ds, "ann", labels)
        _ = ds.scan_tracks()
        _ = ds.scan_labels()
        add_tracks_table(
            ds, "tool", track_table("g", "a", n_rows=3), media_frames=9, frames_read=3
        )
        _write_run_log(
            ds,
            "01AAAAAAAAAAAAAAAAAAAAAAAA",
            _irregular("g__a", "2026-11-01T00:01:00Z", 1.45),
        )
        sources = (str(tracks), str(labels))

        def forbidden(target: object) -> bool:
            text = str(target) if isinstance(target, (str, Path)) else ""
            return text.endswith(".parquet") or text.startswith(sources)

        real_open = _untyped(builtins.open)
        real_path_open = _untyped(Path.open)

        def guarded_open(file: object, *args: object, **kwargs: object) -> object:
            assert not forbidden(file), f"opened {file}"
            return real_open(file, *args, **kwargs)

        def guarded_path_open(self: Path, *args: object, **kwargs: object) -> object:
            assert not forbidden(self), f"opened {self}"
            return real_path_open(self, *args, **kwargs)

        def refuse(*args: object, **kwargs: object) -> object:
            raise AssertionError("a table was read")

        import pandas as pd

        monkeypatch.setattr(pd, "read_parquet", refuse)
        monkeypatch.setattr(pq, "read_table", refuse)
        monkeypatch.setattr(builtins, "open", guarded_open)
        monkeypatch.setattr(Path, "open", guarded_path_open)

        report = raw_report(ds)

        assert report.duplicate_sets == 1
        assert report.pattern_miss_sources == 1
        assert report.frame_axis_tables == 1
        assert report.time_irregular_entries == 1

    def test_a_missing_source_path_is_reported_and_nothing_is_created(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        _ = _write(root / "present" / "a.npy", b"a")
        _tracks_source(ds, "present", root / "present")
        _ = ds.scan_tracks()
        _tracks_source(ds, "gone", root / "gone", group_pattern=r"^(hex)_")
        _labels_source(ds, "ann", root / "ann-share")
        read_only = open_dataset(ds.manifest_path, ensure_roots=False)
        before = _tree(root)

        report = raw_report(read_only)

        assert report.missing_sources == [
            MissingSource(kind="tracks", source="gone", path=str(root / "gone")),
            MissingSource(kind="labels", source="ann", path=str(root / "ann-share")),
        ]
        assert report.missing_source_paths == 2
        assert report.pattern_misses == []
        assert _tree(root) == before
        assert not (root / "gone").exists()

    def test_an_empty_dataset_reports_nothing_and_creates_nothing(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        before = _tree(root)

        report = raw_report(ds)

        assert report == RawReport()
        assert _tree(root) == before
        assert not run_log_dir(ds.base_dir).exists()


class TestTheModel:
    def test_it_is_a_strict_model_that_round_trips_through_json(
        self, root: Path
    ) -> None:
        ds = _dataset(root)
        folder = root / "cams"
        _write_all([folder / "a.npy", folder / "b.npy"], lambda _: b"same")
        _tracks_source(ds, "cams", folder)
        _ = ds.scan_tracks()
        report = raw_report(ds)

        assert issubclass(RawReport, StrictModel)
        assert issubclass(DuplicateSet, StrictModel)
        assert RawReport.model_validate_json(report.model_dump_json()) == report
        with pytest.raises(ValueError, match="extra"):
            _ = RawReport.model_validate({"unknown": 1})

    def test_the_sample_is_ten(self) -> None:
        assert RAW_REPORT_SAMPLE == 10


def test_the_module_loads_no_pandas_and_no_upper_layer() -> None:
    statement = "import mosaic.core.raw_report"

    assert modules_loaded_by(statement, ("pandas",)) == []
    assert (
        modules_loaded_by(
            statement,
            ("mosaic.agent", "mosaic.behavior", "mosaic.tracking", "mosaic.cli"),
        )
        == []
    )


def test_a_copy_reports_the_frame_axis_of_its_sources_tables(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "source", name="source")
    add_tracks_table(
        source,
        "tool",
        track_table("g", "short", n_rows=2),
        media_frames=100,
        frames_read=95,
    )
    copy = copy_of(source, tmp_path / "copy")

    report = raw_report(copy)

    assert report.frame_axis == raw_report(source).frame_axis
    assert report.frame_axis_tables == 1
