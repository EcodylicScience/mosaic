"""``ConversionOutcome.converted`` counts the entries written or current.

An entry that failed is not one of them, and a run that converts none of what it
attempted is lost whole."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import ConversionOutcome, Dataset
from mosaic.core.track_converter import EntryHints
from mosaic.core.track_library.calms21 import Calms21Converter, Calms21Params
from mosaic.core.pipeline.tracks_index import read_tracks_index
from tests.helpers import calms21_dataset, make_dataset, write_trex_npz


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


def test_nothing_to_convert_is_not_a_loss() -> None:
    outcome = ConversionOutcome(converted=0, failed=0)

    assert outcome.ok
    assert not outcome.lost_everything


def _failing_on_the_second_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    convert = Calms21Converter.convert

    def convert_or_fail(
        self: Calms21Converter, path: Path, params: Calms21Params, hints: EntryHints
    ) -> pd.DataFrame:
        if hints.sequence == "s2":
            msg = "the second pair cannot be converted"
            raise ValueError(msg)
        return convert(self, path, params, hints)

    monkeypatch.setattr(Calms21Converter, "convert", convert_or_fail)


def test_a_file_failing_on_its_second_pair_converts_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4, "s2": 4}}})
    _failing_on_the_second_pair(monkeypatch)

    outcome = ds.convert_all_tracks()

    assert (outcome.converted, outcome.failed) == (1, 1)
    index = read_tracks_index(ds)
    assert list(zip(index["group"], index["sequence"], strict=True)) == [
        ("train", "s1")
    ]


def test_the_failed_entry_of_a_file_is_the_pair_that_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4, "s2": 4}}})
    _failing_on_the_second_pair(monkeypatch)
    (row,) = ds.read_tracks_raw_index()

    outcome = ds.convert_track_file(pd.Series(row), {}, False)

    assert outcome.failure is not None
    assert outcome.failed_entry == ("train", "s2")
    assert outcome.covered == [("train", "s1"), ("train", "s2")]
