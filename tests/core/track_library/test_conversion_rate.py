"""A converter times a table at the explicit rate, else the media's, else the dataset's.

SLEAP analysis files, DeepLabCut exports and Ultralytics tracker output record no frame
rate, so ``time`` is ``frame / rate`` for a rate that the caller supplies. The order is
the explicit ``fps`` parameter, then the measured rate of the entry's media
(``EntryHints.fps``), then the dataset's ``fps_default`` (``fallback_fps``). With none
of the three the converter refuses and writes no table at an assumed rate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.track_converter import (
    EntryHints,
    FrameRateUnknownError,
    TimedTrackConvertParams,
    TrackConverter,
    TrackConvertParams,
    conversion_rate,
    get_track_converter,
)
from mosaic.core.track_library.deeplabcut import DlcParams
from mosaic.core.track_library.sleap import SleapConvertParams
from mosaic.core.track_library.ultralytics_tracks import UltralyticsTracksParams
from tests.helpers import (
    write_dlc_csv,
    write_sleap_analysis_h5,
    write_ultralytics_predictions,
)

FRAMES = 6


def _write_sleap(path: Path) -> None:
    write_sleap_analysis_h5(path, np.random.default_rng(0).random((FRAMES, 1, 1, 2)))


def _write_dlc(path: Path) -> None:
    _ = write_dlc_csv(path, ["nose"], n_frames=FRAMES)


def _write_ultralytics(path: Path) -> None:
    write_ultralytics_predictions(path, n_frames=FRAMES, n_ids=1)


@dataclass(frozen=True)
class _Case:
    src_format: str
    params: type[TimedTrackConvertParams]
    write: Callable[[Path], None]
    filename: str


CASES = {
    "sleap": _Case("sleap_analysis_h5", SleapConvertParams, _write_sleap, "t.h5"),
    "deeplabcut": _Case("deeplabcut", DlcParams, _write_dlc, "t.csv"),
    "ultralytics": _Case(
        "ultralytics_tracks",
        UltralyticsTracksParams,
        _write_ultralytics,
        "t.parquet",
    ),
}


@pytest.fixture(params=sorted(CASES))
def case(request: pytest.FixtureRequest) -> _Case:
    return CASES[request.param]


def _convert(
    case: _Case,
    tmp_path: Path,
    hints: EntryHints,
    **params: float,
) -> pd.DataFrame:
    path = tmp_path / case.filename
    case.write(path)
    converter: TrackConverter[TrackConvertParams] = get_track_converter(case.src_format)
    overrides: dict[str, object] = {**params}
    return converter.convert(path, case.params.from_overrides(overrides), hints)


def _expect_time(table: pd.DataFrame, rate: float) -> None:
    assert len(table) > 0
    np.testing.assert_allclose(
        np.asarray(table["time"], dtype=float),
        np.asarray(table["frame"], dtype=float) / rate,
    )


def test_the_measured_rate_times_the_table(case: _Case, tmp_path: Path) -> None:
    table = _convert(case, tmp_path, EntryHints(fps=60.0))

    _expect_time(table, 60.0)


def test_an_explicit_rate_outranks_the_measured_one(
    case: _Case, tmp_path: Path
) -> None:
    table = _convert(case, tmp_path, EntryHints(fps=60.0), fps=25.0)

    _expect_time(table, 25.0)


def test_the_fallback_times_a_table_without_a_measured_rate(
    case: _Case, tmp_path: Path
) -> None:
    table = _convert(case, tmp_path, EntryHints(), fallback_fps=24.0)

    _expect_time(table, 24.0)


def test_the_measured_rate_outranks_the_fallback(case: _Case, tmp_path: Path) -> None:
    table = _convert(case, tmp_path, EntryHints(fps=60.0), fallback_fps=24.0)

    _expect_time(table, 60.0)


def test_a_table_without_any_rate_is_refused(case: _Case, tmp_path: Path) -> None:
    with pytest.raises(FrameRateUnknownError, match="fps_default"):
        _ = _convert(case, tmp_path, EntryHints())


def test_an_unset_rate_leaves_the_identity(case: _Case) -> None:
    unset = case.params.from_overrides().identity_dump()
    assert "fps" not in unset
    assert "fallback_fps" not in unset
    assert case.params.from_overrides({"fps": 25.0}).identity_dump()["fps"] == 25.0
    fallback = case.params.from_overrides({"fallback_fps": 24.0}).identity_dump()
    assert fallback["fallback_fps"] == 24.0


@pytest.mark.parametrize("field", ["fps", "fallback_fps"])
@pytest.mark.parametrize("value", [0.0, -30.0])
def test_a_rate_that_is_not_positive_is_refused(
    case: _Case, field: str, value: float
) -> None:
    with pytest.raises(ValueError, match=field):
        _ = case.params.from_overrides({field: value})


# --- the rule itself ---------------------------------------------------------


def test_the_rule_takes_the_explicit_rate_first() -> None:
    assert conversion_rate(25.0, 24.0, EntryHints(fps=60.0), "x.h5") == 25.0


def test_the_rule_takes_the_measured_rate_before_the_fallback() -> None:
    assert conversion_rate(None, 24.0, EntryHints(fps=60.0), "x.h5") == 60.0


def test_the_rule_falls_back_last() -> None:
    assert conversion_rate(None, 24.0, EntryHints(), "x.h5") == 24.0


def test_the_refusal_names_the_source_and_the_three_remedies() -> None:
    with pytest.raises(FrameRateUnknownError) as raised:
        _ = conversion_rate(None, None, EntryHints(), "x.h5")

    message = str(raised.value)
    assert "x.h5" in message
    assert "fps=" in message
    assert "index" in message
    assert "meta.fps_default" in message


def test_the_refusal_is_a_value_error() -> None:
    assert issubclass(FrameRateUnknownError, ValueError)
