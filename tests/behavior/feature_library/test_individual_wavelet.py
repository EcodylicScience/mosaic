"""``individual-wavelet`` output: one wavelet spectrogram per individual.

The values are those of ``log_power``, which ``pair-wavelet`` shares, so a track
without gaps or NaN equals the same series run through ``pair-wavelet``. A missing
frame reads as NaN, and a coefficient is NaN where its wavelet support covers one.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from mosaic.behavior.feature_library.individual_wavelet import IndividualWavelet
from mosaic.behavior.feature_library.pair_wavelet import PairWavelet
from mosaic.behavior.feature_library.wavelet import (
    log_power,
    wavelet_band,
    wavelet_columns,
)
from mosaic.core.pipeline.types import Result

type FloatArray = npt.NDArray[np.float64]

FPS = 30.0
WAVELET = "cmor1.5-1.0"
IDENTITIES = (0, 1)
IDENTITY_COLUMNS = ["frame", "id", "sequence", "group"]

# A band whose longest wavelet spans fewer rows than the track holds, so that a NaN
# blanks a run of rows and the ends of the track stay finite.
NARROW_BAND: dict[str, object] = {"f_min": 2.0, "f_max": 10.0, "n_freq": 5}
N_LONG = 400


def _feature(params: dict[str, object] | None = None) -> IndividualWavelet:
    return IndividualWavelet(
        IndividualWavelet.Inputs((Result(feature="keypoint-kinematics"),)),
        params=params,
    )


def _tracks(
    n_frames: int = 96,
    names: tuple[str, ...] = ("a", "b"),
    *,
    seed: int = 7,
    fps: object = None,
) -> pd.DataFrame:
    """One row per ``(frame, id)`` of two individuals, in a shuffled row order."""
    rng = np.random.default_rng(seed)
    blocks: list[pd.DataFrame] = []
    for identity in IDENTITIES:
        columns: dict[str, object] = {
            "frame": np.arange(n_frames, dtype=np.int64),
            "id": identity,
            "sequence": "seq",
            "group": "grp",
        }
        for name in names:
            columns[name] = rng.normal(size=n_frames) + rng.normal() * 3.0
        if fps is not None:
            columns["fps"] = fps
        blocks.append(pd.DataFrame(columns))
    table = pd.concat(blocks, ignore_index=True)
    order = np.random.default_rng(seed + 1).permutation(len(table))
    return table.iloc[order].reset_index(drop=True)


def _series(
    table: pd.DataFrame, identity: int, columns: list[str], order: str = "frame"
) -> FloatArray:
    rows = table[table["id"] == identity].sort_values(order)
    return rows[columns].to_numpy(dtype=float)


def _expected(
    table: pd.DataFrame,
    identity: int,
    columns: list[str],
    *,
    fps: float = FPS,
    f_min: float = 0.2,
    f_max: float = 5.0,
    n_freq: int = 25,
    log_floor: float = -3.0,
) -> pd.DataFrame:
    band = wavelet_band(WAVELET, f_min, f_max, n_freq, fps)
    values = log_power(_series(table, identity, columns), band, WAVELET, fps, log_floor)
    return pd.DataFrame(values, columns=wavelet_columns(columns, n_freq))


def _w_columns(out: pd.DataFrame) -> list[str]:
    return [c for c in out.columns if c.startswith("W_")]


def _individual(out: pd.DataFrame, identity: int) -> pd.DataFrame:
    return out[out["id"] == identity].reset_index(drop=True)


def test_every_column_equals_the_log_power_of_its_series_apart_from_the_other_individual() -> (
    None
):
    table = _tracks()

    out = _feature().apply(table)

    for identity in IDENTITIES:
        expected = _expected(table, identity, ["a", "b"])
        mine = _individual(out, identity)
        assert _w_columns(mine) == list(expected.columns)
        for column in expected.columns:
            assert mine[column].dtype == np.float32
            np.testing.assert_array_equal(
                mine[column].to_numpy(), expected[column].to_numpy(), err_msg=column
            )
    first = _individual(out, 0)[_w_columns(out)].to_numpy()
    second = _individual(out, 1)[_w_columns(out)].to_numpy()
    assert not np.array_equal(first, second)


def test_a_nan_blanks_a_run_of_rows_that_narrows_as_the_frequency_rises() -> None:
    table = _tracks(N_LONG)
    nan_frame = N_LONG // 2
    table.loc[(table["id"] == 0) & (table["frame"] == nan_frame), "a"] = np.nan

    out = _feature(NARROW_BAND).apply(table)

    first = _individual(out, 0)
    widths: list[int] = []
    for fi in range(5):
        blank = np.flatnonzero(first[f"W_a_f{fi}"].isna().to_numpy())
        assert blank.size > 0
        assert blank[-1] - blank[0] + 1 == blank.size, f"f{fi} run is not contiguous"
        assert blank[0] <= nan_frame <= blank[-1]
        assert blank.size < N_LONG
        widths.append(int(blank.size))
        assert not first[f"W_b_f{fi}"].isna().any()
    assert widths == sorted(widths, reverse=True)
    assert widths[0] > widths[-1]
    second = _individual(out, 1)
    assert not second[_w_columns(second)].isna().any().any()


def test_a_removed_frame_reads_as_a_nan_in_every_column_and_leaves_no_row() -> None:
    table = _tracks(N_LONG)
    gap_frame = N_LONG // 2
    at_gap = (table["id"] == 0) & (table["frame"] == gap_frame)
    with_nan = table.copy()
    with_nan.loc[at_gap, ["a", "b"]] = np.nan
    without_row = table[~at_gap].reset_index(drop=True)

    expected = _feature(NARROW_BAND).apply(with_nan)
    out = _feature(NARROW_BAND).apply(without_row)

    assert len(out) == len(table) - 1
    first = _individual(out, 0)
    assert gap_frame not in first["frame"].to_numpy()
    kept = expected[~((expected["id"] == 0) & (expected["frame"] == gap_frame))]
    assert out[IDENTITY_COLUMNS].equals(kept[IDENTITY_COLUMNS].reset_index(drop=True))
    for column in _w_columns(out):
        np.testing.assert_array_equal(
            out[column].to_numpy(), kept[column].to_numpy(), err_msg=column
        )
    assert first[_w_columns(first)].isna().any().any()


def test_a_track_without_gaps_equals_pair_wavelet_on_the_same_series() -> None:
    table = _tracks()
    one = table[table["id"] == 0].sort_values("frame").reset_index(drop=True)
    pairs = one.drop(columns=["id"]).assign(perspective=0, id1=0, id2=1)

    out = _feature({"cols": ["a", "b"]}).apply(table)
    reference = PairWavelet(
        PairWavelet.Inputs((Result(feature="pair-posedistance-pca"),)),
        params={"cols": ["a", "b"]},
    ).apply(pairs)

    mine = _individual(out, 0)
    assert _w_columns(mine) == _w_columns(reference)
    for column in _w_columns(reference):
        np.testing.assert_array_equal(
            mine[column].to_numpy(), reference[column].to_numpy(), err_msg=column
        )


def test_columns_one_row_per_frame_and_id_and_the_sort() -> None:
    table = _tracks()

    out = _feature().apply(table)

    expected_w = [f"W_{col}_f{fi}" for col in ("a", "b") for fi in range(25)]
    assert list(out.columns) == expected_w + IDENTITY_COLUMNS
    assert not {"perspective", "id1", "id2"} & set(out.columns)
    assert len(out) == len(table)
    keys = out[["id", "frame"]]
    assert keys.equals(
        table[["id", "frame"]].sort_values(["id", "frame"], ignore_index=True)
    )
    assert out.index.equals(pd.RangeIndex(len(out)))
    assert out["sequence"].eq("seq").all()
    assert out["group"].eq("grp").all()
    assert list(out.attrs) == [
        "frequencies_hz",
        "scales",
        "wavelet",
        "fps",
        "pc_cols",
        "input_columns",
    ]
    band = wavelet_band(WAVELET, 0.2, 5.0, 25, FPS)
    assert out.attrs["frequencies_hz"] == band.frequencies.tolist()
    assert out.attrs["scales"] == band.scales.tolist()
    assert out.attrs["wavelet"] == WAVELET
    assert out.attrs["fps"] == FPS
    assert out.attrs["pc_cols"] == []
    assert out.attrs["input_columns"] == ["a", "b"]


def test_each_row_keeps_the_sequence_and_group_of_its_input_row() -> None:
    n_previous = 100
    table = _tracks(N_LONG)
    table["sequence"] = np.where(table["frame"] < n_previous, "previous", "current")
    table["group"] = np.where(table["id"] == 0, "left", "right")

    out = _feature(NARROW_BAND).apply(table)

    expected_sequence = np.where(out["frame"] < n_previous, "previous", "current")
    expected_group = np.where(out["id"] == 0, "left", "right")
    assert np.array_equal(out["sequence"].to_numpy(), expected_sequence)
    assert np.array_equal(out["group"].to_numpy(), expected_group)
    # The two sequences are one continuous track, so the transform spans both.
    expected = _expected(table, 0, ["a", "b"], f_min=2.0, f_max=10.0, n_freq=5)
    mine = _individual(out, 0)
    for column in expected.columns:
        np.testing.assert_array_equal(
            mine[column].to_numpy(), expected[column].to_numpy(), err_msg=column
        )


def test_time_orders_the_rows_when_frame_is_absent_and_consecutive_rows_are_samples() -> (
    None
):
    table = _tracks().rename(columns={"frame": "time"})
    table["time"] = table["time"] * 0.5

    out = _feature().apply(table)

    assert list(out.columns)[-4:] == ["time", "id", "sequence", "group"]
    assert "frame" not in out.columns
    for identity in IDENTITIES:
        expected = _expected(table.assign(frame=table["time"]), identity, ["a", "b"])
        mine = _individual(out, identity)
        assert mine["time"].is_monotonic_increasing
        for column in expected.columns:
            np.testing.assert_array_equal(
                mine[column].to_numpy(), expected[column].to_numpy(), err_msg=column
            )


def test_a_repeated_frame_of_one_individual_is_refused() -> None:
    table = _tracks()
    repeated = pd.concat([table, table.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=r"^individual-wavelet: id .* repeated frames"):
        _feature().apply(repeated)


def test_a_non_integer_frame_is_refused() -> None:
    table = _tracks()
    table["frame"] = table["frame"].astype(float) + 0.5

    with pytest.raises(ValueError, match=r"^individual-wavelet: .*whole frame numbers"):
        _feature().apply(table)


def test_a_pair_table_is_refused_for_its_missing_id_column() -> None:
    pairs = _tracks().rename(columns={"id": "id1"}).assign(id2=1, perspective=0)

    with pytest.raises(ValueError, match=r"Missing required columns: \['id'\]"):
        _feature().apply(pairs)


def test_an_empty_table_returns_a_frame_naming_the_order_column_and_id() -> None:
    out = _feature().apply(_tracks().iloc[:0])

    assert out.empty
    assert list(out.columns) == ["frame", "id"]


def test_cols_takes_precedence_over_the_pc_prefix_columns() -> None:
    table = _tracks(names=("PC0", "PC1", "speed"))

    out = _feature({"cols": ["speed"]}).apply(table)

    assert out.attrs["input_columns"] == ["speed"]
    assert out.attrs["pc_cols"] == []
    assert _w_columns(out) == [f"W_speed_f{fi}" for fi in range(25)]
    expected = _expected(table, 1, ["speed"])
    np.testing.assert_array_equal(
        _individual(out, 1)["W_speed_f3"].to_numpy(), expected["W_speed_f3"].to_numpy()
    )


def test_cols_naming_only_absent_columns_raises() -> None:
    with pytest.raises(
        ValueError, match=r"\[individual-wavelet\] None of the requested"
    ):
        _feature({"cols": ["absent"]}).apply(_tracks())


def test_the_pc_prefix_columns_come_before_other_numeric_columns() -> None:
    table = _tracks(names=("PC0", "PC1", "speed"))

    out = _feature().apply(table)

    assert out.attrs["input_columns"] == ["PC0", "PC1"]
    assert out.attrs["pc_cols"] == ["PC0", "PC1"]
    assert not any(c.startswith("W_speed") for c in out.columns)


def test_without_cols_or_pc_columns_every_numeric_feature_column_is_read() -> None:
    table = _tracks(names=("speed", "area"), fps=FPS)

    out = _feature().apply(table)

    # Sorted by name, with the identity columns and the ``fps`` column left out.
    assert out.attrs["input_columns"] == ["area", "speed"]


def test_a_table_with_no_numeric_feature_column_raises() -> None:
    with pytest.raises(
        ValueError, match=r"\[individual-wavelet\] Could not auto-detect"
    ):
        _feature().apply(_tracks(names=()))


def test_a_constant_fps_column_sets_the_frame_rate() -> None:
    table = _tracks(fps=60.0)

    out = _feature({"sampling": {"fps_default": 15.0}}).apply(table)

    assert out.attrs["fps"] == 60.0
    expected = _expected(table, 0, ["a", "b"], fps=60.0)
    np.testing.assert_array_equal(
        _individual(out, 0)["W_a_f10"].to_numpy(), expected["W_a_f10"].to_numpy()
    )


def test_without_a_fps_column_the_default_frame_rate_applies() -> None:
    out = _feature({"sampling": {"fps_default": 15.0}}).apply(_tracks())

    assert out.attrs["fps"] == 15.0


def test_a_fps_column_that_varies_falls_back_to_the_default() -> None:
    varying = _tracks()
    varying["fps"] = np.where(varying["frame"] % 2 == 0, 30.0, 60.0)

    out = _feature({"sampling": {"fps_default": 15.0}}).apply(varying)

    assert out.attrs["fps"] == 15.0


def test_twenty_frequencies_give_twenty_columns_per_input_column() -> None:
    out = _feature({"n_freq": 20}).apply(_tracks())

    w_columns = _w_columns(out)
    assert len(w_columns) == 2 * 20
    for column in ("a", "b"):
        assert [c for c in w_columns if c.startswith(f"W_{column}_")] == [
            f"W_{column}_f{fi}" for fi in range(20)
        ]
    assert len(out.attrs["frequencies_hz"]) == len(out.attrs["scales"]) == 20
