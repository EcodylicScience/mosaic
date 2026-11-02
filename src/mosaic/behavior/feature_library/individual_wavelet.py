"""IndividualWavelet feature -- CWT spectrograms of numeric columns of each individual.

The columns are chosen by ``cols``, else the ``pc_prefix`` columns, else every
numeric feature column. The transform is the one ``pair-wavelet`` runs, shared in
:mod:`mosaic.behavior.feature_library.wavelet`.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, final

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.pipeline.column_contract import (
    ORDER_COLUMN,
    Column,
    Conditional,
    ParamColumn,
    Payload,
    Prefix,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    DependencyLookup,
    EmitsLevel,
    Inputs,
    InputStream,
    Result,
    TrackInput,
    resolve_order_col,
)
from mosaic.core.pipeline.types.feature import RelatesLevel

from .derivatives import frame_axis
from .helpers import ensure_columns
from .registry import register_feature
from .wavelet import (
    WaveletParams,
    constant_fps,
    log_power,
    wavelet_band,
    wavelet_columns,
    wavelet_input_columns,
)

type IntArray = npt.NDArray[np.int64]

_NAME = "individual-wavelet"
_WHO = f"[{_NAME}]"


@final
@register_feature
class IndividualWavelet:
    """
    CWT spectrograms of numeric columns of each individual.
    The columns are those ``cols`` names, else the ``pc_prefix`` columns, else
    every numeric feature column. A pair model reads them through a join to its
    pair rows.

    Expects input df to contain columns:
      - 'id'
      - 'frame' (preferred) or 'time' (if used as order column)
      - the payload columns to transform: ``cols``, else the ``pc_prefix``
        columns PC0..PC{k-1}, else every numeric feature column

    Returns a DataFrame with one row per input row, sorted by id and then the
    order column, with columns:
      - frame (or time if that was the order col)
      - id
      - W_{col}_f{fi} (log-power, clamped, for each column x frequency)
      and (optionally) the group and sequence of each input row.

    Stateless (no fitting). FPS is inferred from constant df['fps'] if
    present, otherwise from fps_default. Frequencies are geometrically spaced
    in [f_min, f_max] (equal steps in log2 frequency). The transform and its
    parameters are those of ``pair-wavelet``.

    **Gaps.** With a ``frame`` column, each individual's rows are placed on the
    contiguous frames from its first to its last, the frames it lacks reading as
    NaN, and the transform runs on that range. With only a ``time`` column,
    consecutive rows are read as consecutive samples. A coefficient is NaN where
    its wavelet support covers a missing frame or a NaN of the column, and finite
    otherwise. The high frequencies recover close to a gap, and the low ones stay
    blank longer: a dropout blanks each frequency for half its wavelet's support
    on each side.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.individual_wavelet.IndividualWavelet.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    # Each row keeps the group and sequence of its input row, and an individual's
    # rows from neighboring sequences are one track.
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    # apply transforms the columns that cols names and needs only one of them.
    # Unset, it transforms the pc_prefix columns, or else every payload column.
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        ORDER_COLUMN,
        ParamColumn("cols"),
        Payload(),
    )
    # One W_<column>_f<index> column per input column and frequency. The order
    # column keeps its name: frame, or time when frame is absent.
    provides: ClassVar[Provides] = Provides(
        adds=(Prefix("W_"), Column(C.id_col)),
        conditional=(
            *kept(C.frame_col, C.seq_col, C.group_col),
            Conditional(Column(C.time_col)),
        ),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(WaveletParams):
        pass

    def __init__(
        self,
        inputs: IndividualWavelet.Inputs,
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    @staticmethod
    def temporal_reach(params: IndividualWavelet.Params) -> int | None:
        """Return ``None``: the reach depends on a frame rate the data may set.

        One row reads half the longest wavelet, ``scale * (upper_bound -
        lower_bound) / 2`` samples at the scale of ``f_min``, and the scale is the
        frame rate over ``f_min`` times the central frequency. The frame rate is
        ``sampling.fps_default`` only when the input has no constant ``fps`` column,
        and a column of 60 against a default of 30 doubles the reach. A declaration
        from the default alone would understate it, so the reach is unknown.
        """
        return None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        order_col = resolve_order_col(df)
        if df.empty:
            return pd.DataFrame(columns=[order_col, C.id_col])
        ensure_columns(df, [C.id_col])
        fps = constant_fps(df, p.sampling.fps_default)
        in_cols = wavelet_input_columns(df, p, _WHO)
        band = wavelet_band(p.wavelet, p.f_min, p.f_max, p.n_freq, fps)

        if order_col == C.frame_col:
            # The group and sequence of a row are left out of the axis, so that an
            # individual's rows from neighboring sequences are one track.
            axis_columns = [C.frame_col, C.id_col, "camera"]
            axis = frame_axis(df[[c for c in axis_columns if c in df.columns]], _NAME)
            order, frames, starts, ends = (
                axis.order,
                axis.frames,
                axis.starts,
                axis.ends,
            )
        else:
            order, starts, ends = _rows_by_order(df, order_col)
            frames = None

        signal = df[in_cols].to_numpy(dtype=float)[order]
        values = np.empty((len(df), len(in_cols) * len(band.scales)), dtype=np.float32)
        for start, end in zip(starts, ends):
            if frames is None:
                values[start:end] = log_power(
                    signal[start:end], band, p.wavelet, fps, p.log_floor
                )
                continue
            at = frames[start:end] - frames[start]
            placed = np.full((int(at[-1]) + 1, len(in_cols)), np.nan)
            placed[at] = signal[start:end]
            values[start:end] = log_power(placed, band, p.wavelet, fps, p.log_floor)[at]

        keep = [
            c for c in (order_col, C.id_col, C.seq_col, C.group_col) if c in df.columns
        ]
        spectra = pd.DataFrame(
            values, columns=wavelet_columns(in_cols, len(band.scales))
        )
        identity = df[keep].iloc[order].reset_index(drop=True)
        out = pd.concat([spectra, identity], axis=1)

        out.attrs["frequencies_hz"] = band.frequencies.tolist()
        out.attrs["scales"] = band.scales.tolist()
        out.attrs["wavelet"] = str(p.wavelet)
        out.attrs["fps"] = float(fps)
        out.attrs["pc_cols"] = [c for c in in_cols if c.startswith(p.pc_prefix)]
        out.attrs["input_columns"] = list(map(str, in_cols))
        return out


def _rows_by_order(
    df: pd.DataFrame, order_col: str
) -> tuple[IntArray, IntArray, IntArray]:
    """Order the rows by id, then *order_col*, and find each id's rows.

    Returns:
        The row permutation, and the first row and one past the last row of each
        id in that order.
    """
    codes = df.groupby(C.id_col, sort=True, dropna=False).ngroup().to_numpy()
    order = np.lexsort((df[order_col].to_numpy(), codes)).astype(np.int64)
    boundaries = np.flatnonzero(np.diff(codes[order]) != 0) + 1
    starts = np.concatenate(([0], boundaries)).astype(np.int64)
    ends = np.concatenate((boundaries, [len(df)])).astype(np.int64)
    return order, starts, ends
