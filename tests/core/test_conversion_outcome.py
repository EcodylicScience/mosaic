"""``ConversionOutcome.converted`` counts the entries written or current.

An entry that failed is not one of them, and a run that converts none of what it
attempted is lost whole. ``current`` counts the entries among them that the call
found current and did not write."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import ConversionOutcome, Dataset
from mosaic.core.entry import make_entry_key
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.schema import TrackSchemaError
from mosaic.core.track_converter import EntryHints
from mosaic.core.track_library.calms21 import Calms21Converter, Calms21Params
from tests.helpers import (
    CONVERTED_ENTRIES,
    calms21_dataset,
    converted_source,
    copy_of,
    fail_calms21_sequence,
    make_dataset,
    write_tracktor_recording,
    write_trex_npz,
)


_THREE_PAIRS = {"train": {"a": {"s1": 4, "s2": 4, "s3": 4}}}


def _dataset(tmp_path: Path, good: tuple[str, ...], bad: tuple[str, ...]) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    for name in good:
        write_trex_npz(raw / f"{name}_fish0.npz")
    for name in bad:
        raw.mkdir(parents=True, exist_ok=True)
        _ = (raw / f"{name}_fish0.npz").write_bytes(b"not an npz")
    _ = ds.index_tracks_raw([raw], patterns=["*.npz"], src_format="trex_npz")
    return ds


def test_a_failed_entry_is_not_a_converted_one(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, good=("a", "b"), bad=("c",))

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed) == (2, 1)
    assert not outcome.ok
    assert not outcome.lost_everything


def test_every_entry_failing_converts_none(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, good=(), bad=("a", "b"))

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed) == (0, 2)
    assert outcome.lost_everything


def test_a_rerun_counts_the_entries_that_are_already_current(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, good=("a",), bad=("b",))
    _ = ds.convert_all_tracks()

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed) == (1, 1)
    assert outcome.current == 1


def test_a_second_conversion_keeps_the_tables_of_the_first(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, good=("a", "b", "c"), bad=())
    first = ds.convert_all_tracks()
    tables = sorted(ds.get_root("tracks").rglob("*.parquet"))
    assert len(tables) == 3
    kept = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tables}
    raw = ds.base_dir / "raw"
    for name in ("d", "e"):
        write_trex_npz(raw / f"{name}_fish0.npz")
    _ = ds.index_tracks_raw([raw], patterns=["*.npz"], src_format="trex_npz")

    second = ds.convert_all_tracks()

    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tables
    } == kept
    assert (first.converted, first.current) == (3, 0)
    assert (second.converted, second.current) == (5, 3)
    assert second.converted - second.current == 2


def test_a_file_per_sequence_format_counts_its_current_entry(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = write_tracktor_recording(ds.base_dir / "raw")
    _ = ds.index_tracks_raw(
        [ds.base_dir / "raw"],
        patterns=["*_tracked_fixed_final.tsv"],
        src_format="tracktor_corrected_tsv",
    )
    first = ds.convert_all_tracks()

    second = ds.convert_all_tracks()

    assert (first.converted, first.current) == (1, 0)
    assert (second.converted, second.current) == (1, 1)


def _assert_a_copy_converts_nothing(source: Dataset, copy: Dataset) -> None:
    """Convert *copy* and check it reports *source*'s tables current and adds none."""
    entries = len(read_tracks_index(source))

    outcome = copy.convert_all_tracks()

    assert (outcome.converted, outcome.current, outcome.failed) == (
        entries,
        entries,
        0,
    )
    assert read_tracks_index(copy).empty
    assert sorted(copy.get_root("tracks").rglob("*.parquet")) == []


def test_a_copy_reports_its_sources_merged_tables_current_and_writes_no_table(
    tmp_path: Path,
) -> None:
    source = converted_source(tmp_path / "source")
    assert len(read_tracks_index(source)) == len(CONVERTED_ENTRIES)

    _assert_a_copy_converts_nothing(source, copy_of(source, tmp_path / "copy"))


def test_a_copy_reports_its_sources_per_file_table_current_and_writes_no_table(
    tmp_path: Path,
) -> None:
    source = make_dataset(tmp_path / "source", name="source")
    _ = write_tracktor_recording(source.base_dir / "raw")
    _ = source.index_tracks_raw(
        [source.base_dir / "raw"],
        patterns=["*_tracked_fixed_final.tsv"],
        src_format="tracktor_corrected_tsv",
    )
    _ = source.convert_all_tracks()
    copy = copy_of(source, tmp_path / "copy")

    _assert_a_copy_converts_nothing(source, copy)

    (row,) = copy.read_tracks_raw_index()
    file_outcome = copy.convert_track_file(pd.Series(row), {}, False)
    (source_table,) = read_tracks_index(source)["abs_path"]
    assert file_outcome.path == source.resolve_path(source_table)
    assert file_outcome.tables == []


def test_a_copy_reports_its_sources_enumerated_tables_current_and_writes_no_table(
    tmp_path: Path,
) -> None:
    source = calms21_dataset(tmp_path / "source", _THREE_PAIRS)
    _ = source.convert_all_tracks()
    assert len(read_tracks_index(source)) == 3

    _assert_a_copy_converts_nothing(source, copy_of(source, tmp_path / "copy"))


def test_nothing_to_convert_is_not_a_loss() -> None:
    outcome = ConversionOutcome(converted=0, failed=0)

    assert outcome.ok
    assert not outcome.lost_everything


def _indexed_entries(ds: Dataset) -> list[tuple[str, str]]:
    index = read_tracks_index(ds)
    return list(zip(index["group"], index["sequence"], strict=True))


def test_a_file_failing_on_its_second_pair_converts_the_first_and_the_third(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = calms21_dataset(tmp_path / "ds", _THREE_PAIRS)
    fail_calms21_sequence(monkeypatch, "s2")

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed) == (2, 1)
    assert not outcome.lost_everything
    assert sorted(_indexed_entries(ds)) == [("train", "s1"), ("train", "s3")]
    assert sorted(
        path.name for path in ds.get_root("tracks").rglob("*.parquet")
    ) == sorted(f"{make_entry_key('train', name)}.parquet" for name in ("s1", "s3"))
    assert not list(ds.get_root("tracks").rglob(".stage-*"))


def test_the_failed_entry_of_a_file_is_the_pair_that_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = calms21_dataset(tmp_path / "ds", _THREE_PAIRS)
    fail_calms21_sequence(monkeypatch, "s2")
    (row,) = ds.read_tracks_raw_index()

    outcome = ds.convert_track_file(pd.Series(row), {}, False)

    assert outcome.failure is None
    assert outcome.covered == [("train", "s1"), ("train", "s2"), ("train", "s3")]
    ((entry, failure),) = outcome.entry_failures
    assert entry == ("train", "s2")
    assert failure.key == make_entry_key("train", "s2")
    assert "s2" in str(failure.error)
    assert [staged.row.sequence for staged in outcome.tables] == ["s1", "s3"]


def test_a_file_that_cannot_enumerate_its_sequences_is_one_failed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = calms21_dataset(tmp_path / "ds", _THREE_PAIRS)
    (row,) = ds.read_tracks_raw_index()

    def refuse(self: Calms21Converter, path: Path) -> list[tuple[str, str]]:
        msg = "the file cannot be read"
        raise ValueError(msg)

    monkeypatch.setattr(Calms21Converter, "enumerate_sequences", refuse)

    outcome = ds.convert_track_file(pd.Series(row), {}, False)

    assert outcome.failure is not None
    assert outcome.entry_failures == []
    assert outcome.failed_entry is None


def test_every_pair_failing_is_a_failed_entry_each_and_a_lost_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4, "s2": 4}}})

    def refuse(
        self: Calms21Converter, path: Path, params: Calms21Params, hints: EntryHints
    ) -> pd.DataFrame:
        msg = f"sequence {hints.sequence} cannot be converted"
        raise ValueError(msg)

    monkeypatch.setattr(Calms21Converter, "convert", refuse)

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed) == (0, 2)
    assert outcome.lost_everything
    assert _indexed_entries(ds) == []


def test_on_failure_receives_a_record_for_each_failed_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4, "s2": 4}}})
    fail_calms21_sequence(monkeypatch, "s2")
    records: list[tuple[str, str]] = []

    def record(key: str, error: BaseException) -> None:
        records.append((key, str(error)))

    _ = ds.convert_all_tracks(on_failure=record)

    assert records == [
        (make_entry_key("train", "s2"), "sequence s2 cannot be converted")
    ]


def test_a_failed_pair_is_named_on_standard_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4, "s2": 4}}})
    fail_calms21_sequence(monkeypatch, "s2")
    (row,) = ds.read_tracks_raw_index()

    _ = ds.convert_all_tracks()

    warning = next(
        line for line in capsys.readouterr().err.splitlines() if "[WARN]" in line
    )
    assert str(row["abs_path"]) in warning
    assert make_entry_key("train", "s2") in warning


def test_a_rerun_converts_the_pair_that_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = calms21_dataset(tmp_path / "ds", _THREE_PAIRS)
    with monkeypatch.context() as broken:
        fail_calms21_sequence(broken, "s2")
        _ = ds.convert_all_tracks()

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed, outcome.current) == (3, 0, 2)
    assert sorted(_indexed_entries(ds)) == [
        ("train", "s1"),
        ("train", "s2"),
        ("train", "s3"),
    ]


def test_a_pair_failing_the_schema_under_strict_schema_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = calms21_dataset(tmp_path / "ds", _THREE_PAIRS)
    convert = Calms21Converter.convert

    def without_frames(
        self: Calms21Converter, path: Path, params: Calms21Params, hints: EntryHints
    ) -> pd.DataFrame:
        table = convert(self, path, params, hints)
        if hints.sequence == "s2":
            return table.drop(columns=["frame"])
        return table

    monkeypatch.setattr(Calms21Converter, "convert", without_frames)

    with pytest.raises(TrackSchemaError):
        _ = ds.convert_all_tracks(params={"strict_schema": True})


def test_convert_one_track_raises_the_failed_pairs_error_after_committing_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = calms21_dataset(tmp_path / "ds", _THREE_PAIRS)
    fail_calms21_sequence(monkeypatch, "s2")
    (row,) = ds.read_tracks_raw_index()

    with pytest.raises(ValueError, match="sequence s2 cannot be converted"):
        _ = ds.convert_one_track(pd.Series(row))

    assert sorted(_indexed_entries(ds)) == [("train", "s1"), ("train", "s3")]
