"""``pair-wavelet`` output, pinned column by column.

Every ``W_{column}_f{index}`` value is compared bit for bit with a reference that
repeats the feature's dtype sequence: float32 scales, one ``pywt.cwt`` call per
series, the power cast to float32 before the float32 ``tiny`` is added and the log
taken, then the clamp. A float64 reference of ``log(abs(coeffs) ** 2 + tiny)``
differs from the feature in the low bits, so it cannot stand in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import pywt

from mosaic.behavior.feature_library.pair_wavelet import PairWavelet
from mosaic.core.pipeline.manifest import CoreSelector
from mosaic.core.pipeline.types import Result
from mosaic.core.pipeline.writers import trim_feature_output

type FloatArray = npt.NDArray[np.float64]
type Float32Array = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Band:
    """The band, wavelet and floor params of a run, and the reference's copy of them."""

    f_min: float = 0.2
    f_max: float = 5.0
    n_freq: int = 25
    wavelet: str = "cmor1.5-1.0"
    log_floor: float = -3.0

    def params(self) -> dict[str, object]:
        return {
            "f_min": self.f_min,
            "f_max": self.f_max,
            "n_freq": self.n_freq,
            "wavelet": self.wavelet,
            "log_floor": self.log_floor,
        }


N_FRAMES = 96
FPS = 30.0
PAIRS = ((0, 1), (0, 2))
PERSPECTIVES = (0, 1)
IDENTITY_COLUMNS = [
    "frame",
    "time",
    "perspective",
    "id1",
    "id2",
    "sequence",
    "group",
]


def _feature(params: dict[str, object] | None = None) -> PairWavelet:
    return PairWavelet(
        PairWavelet.Inputs((Result(feature="pair-posedistance-pca"),)),
        params=params,
    )


def _pairs(
    payload: dict[str, FloatArray] | None = None,
    *,
    seed: int = 7,
    fps: float = FPS,
) -> pd.DataFrame:
    """Two pairs times two perspectives, in a shuffled row order.

    The rows are shuffled so that the sort by pair, perspective and frame is the
    feature's and not the input's. ``payload`` maps a column name to an array of
    ``N_FRAMES`` values, drawn once and repeated for every series.
    """
    rng = np.random.default_rng(seed)
    blocks: list[pd.DataFrame] = []
    for id1, id2 in PAIRS:
        for perspective in PERSPECTIVES:
            columns: dict[str, object] = {
                "frame": np.arange(N_FRAMES, dtype=np.int64),
                "time": np.arange(N_FRAMES) / fps,
                "perspective": perspective,
                "id1": id1,
                "id2": id2,
                "sequence": "seq",
                "group": "grp",
            }
            names = ["PC0", "PC1"] if payload is None else list(payload)
            for name in names:
                columns[name] = rng.normal(size=N_FRAMES) + rng.normal() * 3.0
            blocks.append(pd.DataFrame(columns))
    table = pd.concat(blocks, ignore_index=True)
    order = np.random.default_rng(seed + 1).permutation(len(table))
    return table.iloc[order].reset_index(drop=True)


def _reference(
    table: pd.DataFrame, in_cols: list[str], *, fps: float, band: Band = Band()
) -> pd.DataFrame:
    """The expected ``W_*`` columns, in the feature's sort order, by the dtype sequence."""
    freqs = np.logspace(
        math.log2(band.f_min), math.log2(band.f_max), band.n_freq, base=2.0
    )
    scales = (fps / (freqs * pywt.central_frequency(band.wavelet))).astype(np.float32)
    tiny = np.finfo(np.float32).tiny

    ordered = table.sort_values(["id1", "id2", "perspective", "frame"])
    blocks: list[pd.DataFrame] = []
    for _, series in ordered.groupby(["id1", "id2", "perspective"]):
        columns: dict[str, Float32Array] = {}
        for column in in_cols:
            coeffs, _ = pywt.cwt(
                series[column].to_numpy(dtype=float),
                scales,
                pywt.ContinuousWavelet(band.wavelet),
                sampling_period=1.0 / fps,
            )
            power = (np.abs(coeffs) ** 2).astype(np.float32)
            log_power = np.maximum(np.log(power + tiny), band.log_floor)
            for fi in range(band.n_freq):
                columns[f"W_{column}_f{fi}"] = log_power[fi]
        blocks.append(pd.DataFrame(columns))
    return pd.concat(blocks, ignore_index=True)


def _assert_bit_for_bit(out: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert list(expected.columns) == [c for c in out.columns if c.startswith("W_")]
    for column in expected.columns:
        assert out[column].dtype == np.float32
        np.testing.assert_array_equal(
            out[column].to_numpy(), expected[column].to_numpy(), err_msg=column
        )


@pytest.mark.parametrize(
    "band",
    [
        pytest.param(Band(), id="defaults"),
        pytest.param(
            Band(f_min=0.5, f_max=6.0, n_freq=7, wavelet="cmor2.0-1.5", log_floor=-1.0),
            id="band-wavelet-and-floor",
        ),
    ],
)
def test_every_column_equals_the_float32_reference_bit_for_bit(band: Band) -> None:
    table = _pairs(fps=FPS)

    out = _feature(band.params()).apply(table)

    expected = _reference(table, ["PC0", "PC1"], fps=FPS, band=band)
    assert len(out) == len(expected) == len(PAIRS) * len(PERSPECTIVES) * N_FRAMES
    _assert_bit_for_bit(out, expected)


def test_the_floor_clamps_and_the_unclamped_values_pass_through() -> None:
    table = _pairs(fps=FPS)

    out = _feature({"log_floor": -1.0}).apply(table)

    values = out[[c for c in out.columns if c.startswith("W_")]].to_numpy()
    assert (values == np.float32(-1.0)).any()
    assert (values > np.float32(-1.0)).any()
    assert values.min() == np.float32(-1.0)


def test_columns_order_sort_and_attrs() -> None:
    table = _pairs(fps=FPS)
    n_freq = 25

    out = _feature().apply(table)

    expected_w = [f"W_{col}_f{fi}" for col in ("PC0", "PC1") for fi in range(n_freq)]
    assert list(out.columns) == expected_w + IDENTITY_COLUMNS

    keys = out[["id1", "id2", "perspective", "frame"]]
    assert keys.equals(
        keys.sort_values(["id1", "id2", "perspective", "frame"]).reset_index(drop=True)
    )
    assert out.index.equals(pd.RangeIndex(len(out)))
    assert set(
        map(tuple, keys[["id1", "id2", "perspective"]].drop_duplicates().to_numpy())
    ) == {(id1, id2, perspective) for id1, id2 in PAIRS for perspective in PERSPECTIVES}
    assert out["sequence"].eq("seq").all()
    assert out["group"].eq("grp").all()

    freqs = np.logspace(math.log2(0.2), math.log2(5.0), n_freq, base=2.0)
    scales = FPS / (freqs * pywt.central_frequency("cmor1.5-1.0"))
    assert list(out.attrs) == [
        "frequencies_hz",
        "scales",
        "wavelet",
        "fps",
        "pc_cols",
        "input_columns",
    ]
    assert out.attrs["frequencies_hz"] == freqs.astype(np.float32).tolist()
    assert out.attrs["scales"] == scales.astype(np.float32).tolist()
    assert out.attrs["wavelet"] == "cmor1.5-1.0"
    assert out.attrs["fps"] == FPS
    assert out.attrs["pc_cols"] == ["PC0", "PC1"]
    assert out.attrs["input_columns"] == ["PC0", "PC1"]


def test_time_replaces_frame_as_the_order_column_when_frame_is_absent() -> None:
    table = _pairs(fps=FPS).drop(columns="frame")

    out = _feature().apply(table)

    assert "time" in out.columns
    assert "frame" not in out.columns
    assert (
        out.groupby(["id1", "id2", "perspective"])["time"]
        .apply(lambda column: column.is_monotonic_increasing)
        .all()
    )


def test_cols_takes_precedence_over_the_pc_prefix_columns() -> None:
    table = _pairs({"PC0": np.empty(0), "PC1": np.empty(0), "speed": np.empty(0)})

    out = _feature({"cols": ["speed"]}).apply(table)

    assert out.attrs["input_columns"] == ["speed"]
    assert out.attrs["pc_cols"] == []
    _assert_bit_for_bit(out, _reference(table, ["speed"], fps=30.0))


def test_cols_keeps_the_present_columns_in_the_order_given() -> None:
    table = _pairs({"PC0": np.empty(0), "speed": np.empty(0), "area": np.empty(0)})

    out = _feature({"cols": ["speed", "absent", "area"]}).apply(table)

    assert out.attrs["input_columns"] == ["speed", "area"]
    _assert_bit_for_bit(out, _reference(table, ["speed", "area"], fps=30.0))


def test_cols_naming_only_absent_columns_raises() -> None:
    table = _pairs()

    with pytest.raises(ValueError, match=r"\[pair-wavelet\] None of the requested"):
        _feature({"cols": ["absent", "also-absent"]}).apply(table)


def test_the_pc_prefix_columns_come_before_other_numeric_columns() -> None:
    table = _pairs({"PC0": np.empty(0), "PC1": np.empty(0), "speed": np.empty(0)})

    out = _feature().apply(table)

    assert out.attrs["input_columns"] == ["PC0", "PC1"]
    assert not any(c.startswith("W_speed") for c in out.columns)
    _assert_bit_for_bit(out, _reference(table, ["PC0", "PC1"], fps=30.0))


def test_the_pc_prefix_param_selects_the_columns_it_names() -> None:
    table = _pairs({"PC0": np.empty(0), "Q0": np.empty(0), "Q1": np.empty(0)})

    out = _feature({"pc_prefix": "Q"}).apply(table)

    assert out.attrs["input_columns"] == ["Q0", "Q1"]
    assert out.attrs["pc_cols"] == ["Q0", "Q1"]


def test_without_cols_or_pc_columns_every_numeric_feature_column_is_read() -> None:
    table = _pairs({"speed": np.empty(0), "area": np.empty(0)}, fps=FPS)

    out = _feature().apply(table)

    # Sorted by name, with the identity columns and the ``fps`` column left out.
    assert out.attrs["input_columns"] == ["area", "speed"]
    assert out.attrs["pc_cols"] == []
    _assert_bit_for_bit(out, _reference(table, ["area", "speed"], fps=FPS))


def test_a_table_with_no_numeric_feature_column_raises() -> None:
    table = _pairs({"PC0": np.empty(0)}).drop(columns=["PC0"])

    with pytest.raises(ValueError, match=r"\[pair-wavelet\] Could not auto-detect"):
        _feature().apply(table)


def test_the_frame_rate_is_measured_from_the_time_column() -> None:
    table = _pairs(fps=60.0)

    out = _feature().apply(table)

    assert out.attrs["fps"] == pytest.approx(60.0)
    _assert_bit_for_bit(out, _reference(table, ["PC0", "PC1"], fps=60.0))


def test_a_stated_fps_column_does_not_set_the_frame_rate() -> None:
    table = _pairs(fps=30.0).assign(fps=60.0)

    out = _feature().apply(table)

    assert out.attrs["fps"] == pytest.approx(30.0)


def test_a_table_without_time_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="pair-wavelet.*'time'"):
        _ = _feature().apply(_pairs().drop(columns="time"))


def test_a_sampling_setting_is_refused() -> None:
    with pytest.raises(ValueError, match="sampling"):
        _ = _feature({"sampling": {"fps_default": 15.0}})


def test_twenty_frequencies_give_twenty_columns_per_input_column() -> None:
    table = _pairs(fps=FPS)

    out = _feature({"n_freq": 20}).apply(table)

    w_columns = [c for c in out.columns if c.startswith("W_")]
    assert len(w_columns) == 2 * 20
    for column in ("PC0", "PC1"):
        assert [c for c in w_columns if c.startswith(f"W_{column}_")] == [
            f"W_{column}_f{fi}" for fi in range(20)
        ]
    assert len(out.attrs["frequencies_hz"]) == len(out.attrs["scales"]) == 20
    _assert_bit_for_bit(
        out, _reference(table, ["PC0", "PC1"], fps=FPS, band=Band(n_freq=20))
    )


def _segment(first_frame: int, sequence: str, group: str, *, seed: int) -> pd.DataFrame:
    """The pairs of ``_pairs`` on frames from *first_frame*, named as one entry."""
    table = _pairs(seed=seed, fps=FPS)
    table["frame"] += first_frame
    table["sequence"] = sequence
    table["group"] = group
    return table


def test_overlapped_rows_carry_the_entry_of_their_own_frames() -> None:
    """Context rows of the neighbors do not name the entry's output rows.

    Under overlap the frame handed to ``apply`` starts with the previous entry's
    rows and ends with the next entry's. The rows kept for the entry name the
    entry, and the pair and perspective of their own series.
    """
    previous = _segment(0, "seg1", "grp", seed=11)
    own = _segment(N_FRAMES, "seg2", "grp", seed=12)
    following = _segment(2 * N_FRAMES, "seg3", "grp", seed=13)
    table = pd.concat([previous, own, following], ignore_index=True)
    selector = CoreSelector(
        order_col="frame",
        first=N_FRAMES,
        last=2 * N_FRAMES - 1,
        entry_key="grp__seg2",
    )

    out = trim_feature_output(_feature().apply(table), selector)

    assert out is not None
    assert len(out) == len(own)
    assert out["sequence"].eq("seg2").all()
    assert out["group"].eq("grp").all()
    expected = own[["id1", "id2", "perspective", "frame"]].sort_values(
        ["id1", "id2", "perspective", "frame"], ignore_index=True
    )
    assert out[["id1", "id2", "perspective", "frame"]].equals(expected)
