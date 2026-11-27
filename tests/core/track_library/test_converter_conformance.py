"""``converter_problems`` reports each way a converter class breaks the contract.

Each case is a small class that breaks one rule, and the sentence that comes back
names that rule. The classes are never registered, so no test leaves a converter in
the registry. The rule that needs a registry entry puts one there through
``monkeypatch.setitem``, which restores the registry afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, ClassVar, override

import numpy as np
import pandas as pd
import pytest

from mosaic.core.params import Declared
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    TrackConverter,
    TrackConvertParams,
)
from mosaic.core.track_library.conformance import converter_problems
from mosaic.core.track_library.sleap import (
    SleapAnalysisH5Converter,
    SleapConvertParams,
)
from tests.helpers import write_sleap_analysis_h5

FORMAT = "conformance_probe"


def _valid_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [0, 1],
            "time": [0.0, 1.0],
            "id": [0, 0],
            "group": ["", ""],
            "sequence": ["s", "s"],
            "X": [1.0, 2.0],
            "Y": [3.0, 4.0],
        }
    )


class _Conforming(TrackConverter[TrackConvertParams]):
    """Declares everything and converts to a valid ``mosaic_v1`` table."""

    src_format = FORMAT
    version = "1.0"
    carries_tracklets = False
    output_schema = "mosaic_v1"

    @override
    def convert(
        self, path: Path, params: TrackConvertParams, hints: EntryHints
    ) -> pd.DataFrame:
        _ = (path, params, hints)
        return _valid_table()


class _WithoutVersion(TrackConverter[TrackConvertParams]):
    src_format = FORMAT
    carries_tracklets = False
    output_schema = "mosaic_v1"


class _WithoutCarriesTracklets(TrackConverter[TrackConvertParams]):
    src_format = FORMAT
    version = "1.0"
    output_schema = "mosaic_v1"


class _WithoutOutputSchema(TrackConverter[TrackConvertParams]):
    src_format = FORMAT
    version = "1.0"
    carries_tracklets = False


class _Parent(TrackConverter[TrackConvertParams]):
    """Declares the three attributes, which a child inherits."""

    version = "1.0"
    carries_tracklets = False
    output_schema = "mosaic_v1"


class _InheritingChild(_Parent):
    src_format = FORMAT


class _WithoutFormat(_Parent):
    pass


class _UnreadableVersion(_Conforming):
    version = "1.0-beta"


class _MergesWithoutStemRule(_Conforming):
    merges_per_sequence = True


class _MergesWithStemRule(_Conforming):
    merges_per_sequence = True

    @override
    def sequence_from_stem(self, stem: str) -> str:
        return stem.split("_")[0]


class _ClaimsBothDirections(_MergesWithStemRule):
    enumerable = True


class _SequenceParams(TrackConvertParams):
    sequence: Annotated[str, Declared("A name that belongs to the entry.")] = ""


class _HashesAnEntryName(_Conforming):
    Params: ClassVar[type[TrackConvertParams]] = _SequenceParams


class _MissingX(_Conforming):
    @override
    def convert(
        self, path: Path, params: TrackConvertParams, hints: EntryHints
    ) -> pd.DataFrame:
        _ = (path, params, hints)
        return _valid_table().drop(columns=["X"])


class _Raises(_Conforming):
    @override
    def convert(
        self, path: Path, params: TrackConvertParams, hints: EntryHints
    ) -> pd.DataFrame:
        _ = (path, params, hints)
        raise RuntimeError("the file is not a track file")


def test_a_conforming_class_has_no_problems() -> None:
    assert converter_problems(_Conforming) == []


@pytest.mark.parametrize(
    ("cls", "attribute"),
    [
        (_WithoutVersion, "version"),
        (_WithoutCarriesTracklets, "carries_tracklets"),
        (_WithoutOutputSchema, "output_schema"),
    ],
    ids=["version", "carries_tracklets", "output_schema"],
)
def test_an_attribute_left_to_the_base_class_is_reported(
    cls: type[TrackConverter[TrackConvertParams]], attribute: str
) -> None:
    problems = converter_problems(cls)

    assert len(problems) == 1
    assert f"does not declare {attribute}" in problems[0]
    assert cls.__name__ in problems[0]


def test_an_attribute_that_a_parent_declares_counts_as_declared() -> None:
    assert converter_problems(_InheritingChild) == []


def test_a_class_without_a_format_is_reported() -> None:
    problems = converter_problems(_WithoutFormat)

    assert len(problems) == 1
    assert "src_format" in problems[0]


def test_a_version_that_a_run_identifier_cannot_return_is_reported() -> None:
    problems = converter_problems(_UnreadableVersion)

    assert len(problems) == 1
    assert "'1.0-beta'" in problems[0]
    assert "run identifier" in problems[0]


def test_another_class_under_the_format_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TRACK_CONVERTERS, FORMAT, _WithoutVersion)

    problems = converter_problems(_Conforming)

    assert len(problems) == 1
    assert FORMAT in problems[0]
    assert "_WithoutVersion" in problems[0]


def test_the_registered_class_is_not_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TRACK_CONVERTERS, FORMAT, _Conforming)

    assert converter_problems(_Conforming) == []


def test_a_class_nobody_registered_gets_no_registry_verdict() -> None:
    assert FORMAT not in TRACK_CONVERTERS
    assert converter_problems(_Conforming) == []


def test_a_merging_format_without_a_stem_rule_is_reported() -> None:
    problems = converter_problems(_MergesWithoutStemRule)

    assert len(problems) == 1
    assert "sequence_from_stem" in problems[0]


def test_a_merging_format_with_a_stem_rule_conforms() -> None:
    assert converter_problems(_MergesWithStemRule) == []


def test_a_format_that_is_enumerable_and_merging_is_reported() -> None:
    problems = converter_problems(_ClaimsBothDirections)

    assert len(problems) == 1
    assert "enumerable" in problems[0]
    assert "merges_per_sequence" in problems[0]


def test_params_that_hash_an_entry_name_are_reported() -> None:
    problems = converter_problems(_HashesAnEntryName)

    assert len(problems) == 1
    assert "entry identity" in problems[0]
    assert "sequence" in problems[0]


def test_a_sample_that_converts_to_a_valid_table_conforms(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    _ = sample.write_text("frame\n0\n")

    assert converter_problems(_Conforming, sample) == []


def test_a_table_without_a_required_column_is_reported(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    _ = sample.write_text("frame\n0\n")

    problems = converter_problems(_MissingX, sample)

    assert len(problems) == 1
    assert "mosaic_v1" in problems[0]
    assert "X" in problems[0]


def test_a_conversion_that_raises_is_reported(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    _ = sample.write_text("frame\n0\n")

    problems = converter_problems(_Raises, sample)

    assert len(problems) == 1
    assert "RuntimeError" in problems[0]
    assert "the file is not a track file" in problems[0]


def test_a_sample_is_converted_with_the_params_it_is_given(tmp_path: Path) -> None:
    seen: list[TrackConvertParams] = []

    class Recording(_Conforming):
        @override
        def convert(
            self, path: Path, params: TrackConvertParams, hints: EntryHints
        ) -> pd.DataFrame:
            seen.append(params)
            return super().convert(path, params, hints)

    sample = tmp_path / "sample.csv"
    _ = sample.write_text("frame\n0\n")
    given = TrackConvertParams(strict_schema=True)

    assert converter_problems(Recording, sample, given) == []
    assert seen == [given]


def test_a_sample_is_not_converted_when_none_is_given() -> None:
    class Never(_Conforming):
        @override
        def convert(
            self, path: Path, params: TrackConvertParams, hints: EntryHints
        ) -> pd.DataFrame:
            raise AssertionError("converted without a sample")

    assert converter_problems(Never) == []


class _CountsFrames(_Conforming):
    """Converts to a two-frame table and reports a frame count of its own."""

    count: ClassVar[int | None] = None

    @override
    def frames_read(self, path: Path) -> int | None:
        _ = path
        return self.count


def _counting(count: int | None) -> type[_CountsFrames]:
    class Counting(_CountsFrames):
        pass

    Counting.count = count
    return Counting


def _sample(tmp_path: Path) -> Path:
    sample = tmp_path / "sample.csv"
    _ = sample.write_text("frame\n0\n")
    return sample


@pytest.mark.parametrize("count", [None, 2, 3], ids=["none", "equal", "larger"])
def test_a_frame_count_that_covers_the_table_conforms(
    tmp_path: Path, count: int | None
) -> None:
    assert converter_problems(_counting(count), _sample(tmp_path)) == []


def test_a_frame_count_below_the_table_is_reported_with_both_numbers(
    tmp_path: Path,
) -> None:
    problems = converter_problems(_counting(1), _sample(tmp_path))

    assert len(problems) == 1
    assert "Counting" in problems[0]
    assert "frames_read" in problems[0]
    assert "1" in problems[0]
    assert "2 distinct frames" in problems[0]


@pytest.mark.parametrize("error", [OSError, ValueError, RuntimeError])
def test_a_frames_read_that_raises_is_reported_naming_the_exception(
    tmp_path: Path, error: type[Exception]
) -> None:
    class Raising(_Conforming):
        @override
        def frames_read(self, path: Path) -> int | None:
            raise error("the header is unreadable")

    problems = converter_problems(Raising, _sample(tmp_path))

    assert len(problems) == 1
    assert "frames_read" in problems[0]
    assert error.__name__ in problems[0]
    assert "the header is unreadable" in problems[0]


def test_a_frame_count_is_not_read_without_a_sample() -> None:
    class Never(_Conforming):
        @override
        def frames_read(self, path: Path) -> int | None:
            raise AssertionError("read a count without a sample")

    assert converter_problems(Never) == []


def test_a_conversion_that_fails_reports_no_frame_count_problem(
    tmp_path: Path,
) -> None:
    class Both(_Raises):
        @override
        def frames_read(self, path: Path) -> int | None:
            return 0

    problems = converter_problems(Both, _sample(tmp_path))

    assert len(problems) == 1
    assert "RuntimeError" in problems[0]


def test_sleap_on_its_sample_has_no_problems(tmp_path: Path) -> None:
    tracks = np.full((3, 2, 2, 2), np.nan)
    tracks[:, 0] = 1.0
    tracks[1, 1] = 2.0
    sample = tmp_path / "vid.analysis.h5"
    write_sleap_analysis_h5(sample, tracks)

    problems = converter_problems(
        SleapAnalysisH5Converter, sample, SleapConvertParams(fps=30.0)
    )

    assert problems == []


def _listing(listed: Callable[[Path], tuple[Path, ...]]) -> type[_Conforming]:
    """A conforming class whose ``companion_files`` is *listed*."""

    class Listing(_Conforming):
        @override
        def companion_files(self, path: Path) -> tuple[Path, ...]:
            return listed(path)

    return Listing


def _beside(sample: Path, name: str) -> Path:
    companion = sample.parent / name
    _ = companion.write_text("companion\n")
    return companion


def test_a_companion_that_exists_beside_the_sample_conforms(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    companion = _beside(sample, "sample.extra")

    assert converter_problems(_listing(lambda _path: (companion,)), sample) == []


def test_companions_that_are_sorted_and_distinct_conform(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    first = _beside(sample, "a.extra")
    second = _beside(sample, "b.extra")

    problems = converter_problems(_listing(lambda _path: (first, second)), sample)

    assert problems == []


def test_resolved_companions_of_a_sample_in_a_symlinked_directory_conform(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    sample = _sample(real)
    _ = _beside(sample, "sample.extra")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    def resolved(path: Path) -> tuple[Path, ...]:
        return (path.resolve().with_name("sample.extra"),)

    assert converter_problems(_listing(resolved), link / sample.name) == []


def test_a_companion_that_is_missing_is_reported(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    missing = tmp_path / "missing.extra"

    problems = converter_problems(_listing(lambda _path: (missing,)), sample)

    assert len(problems) == 1
    assert "Listing" in problems[0]
    assert "companion_files" in problems[0]
    assert "missing.extra" in problems[0]
    assert "does not exist" in problems[0]


def test_a_companion_that_is_a_directory_is_reported(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    directory = tmp_path / "folder.extra"
    directory.mkdir()

    problems = converter_problems(_listing(lambda _path: (directory,)), sample)

    assert len(problems) == 1
    assert "folder.extra" in problems[0]
    assert "is not a file" in problems[0]


def test_a_companion_in_another_directory_is_reported(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    companion = elsewhere / "sample.extra"
    _ = companion.write_text("companion\n")

    problems = converter_problems(_listing(lambda _path: (companion,)), sample)

    assert len(problems) == 1
    assert "sample.extra" in problems[0]
    assert "directory" in problems[0]


def test_the_sample_listed_as_its_own_companion_is_reported(tmp_path: Path) -> None:
    sample = _sample(tmp_path)

    problems = converter_problems(_listing(lambda path: (path,)), sample)

    assert len(problems) == 1
    assert "sample.csv" in problems[0]
    assert "is the sample" in problems[0]


def test_companions_out_of_order_are_reported(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    first = _beside(sample, "a.extra")
    second = _beside(sample, "b.extra")

    problems = converter_problems(_listing(lambda _path: (second, first)), sample)

    assert len(problems) == 1
    assert "sorted" in problems[0]


def test_a_companion_listed_twice_is_reported(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    companion = _beside(sample, "a.extra")

    problems = converter_problems(
        _listing(lambda _path: (companion, companion)), sample
    )

    assert len(problems) == 1
    assert "twice" in problems[0] or "repeat" in problems[0]


def test_a_companion_files_that_raises_is_reported_naming_the_hook(
    tmp_path: Path,
) -> None:
    def refuse(path: Path) -> tuple[Path, ...]:
        raise OSError("the directory is unreadable")

    problems = converter_problems(_listing(refuse), _sample(tmp_path))

    assert len(problems) == 1
    assert "Listing" in problems[0]
    assert "companion_files" in problems[0]
    assert "sample.csv" in problems[0]
    assert "OSError" in problems[0]
    assert "the directory is unreadable" in problems[0]


def test_companions_are_not_listed_without_a_sample() -> None:
    def never(path: Path) -> tuple[Path, ...]:
        raise AssertionError("listed companions without a sample")

    assert converter_problems(_listing(never)) == []
