"""IndividualWavelet feature -- CWT spectrograms of numeric columns of each individual.

The columns are chosen by ``cols``, else the ``pc_prefix`` columns, else every
numeric feature column. The transform is the one ``pair-wavelet`` runs, shared in
:mod:`mosaic.behavior.feature_library.wavelet`.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, final

import numpy as np
import pandas as pd

from mosaic.core.pipeline.column_contract import (
    AnyOf,
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
    Inputs,
    InputStream,
    Result,
    TrackInput,
    resolve_order_col,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.feature_levels import RelatesLevel
from mosaic.core.spaces import TRACK_SPACES, TrackSpace
from mosaic.core.timebase import uniform_grid_rate

from .derivatives import IntArray, frame_axis
from .helpers import ensure_columns
from .registry import register_feature
from .wavelet import (
    WaveletParams,
    wavelet_attrs,
    wavelet_columns,
    wavelet_dtype,
    wavelet_input_columns,
    wavelet_spectra,
)

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
      - W_{col}_f{fi} (log-power or log amplitude, clamped, for each column x
        frequency)
      and (optionally) the group and sequence of each input row.

    Stateless (no fitting). The frame rate is measured from the ``time`` column
    (frames over the time span), and the feature warns when the steps are uneven,
    because the transform reads its rows as evenly spaced. Frequencies are
    geometrically spaced in [f_min, f_max] (equal steps in log2 frequency). The transform and its
    parameters are those of ``pair-wavelet``.

    **Method.** ``method`` chooses the transform. ``pywt``, the default, is the
    PyWavelets transform above, with float32 output. ``motionmapper`` is the
    Morlet wavelet of Berman et al. (2014) with central angular frequency
    ``omega0``, which that method requires and the other refuses: each column
    gives the natural log of its normalized amplitude, computed by convolving the
    zero-padded series with the wavelet in the Fourier domain, clamped from below
    at ``log_floor``, in float64. Under it a NaN, or a frame the table lacks,
    blanks the rows within 8 scales of it in the columns it touches, and no
    others, so each blank run is ``2 * ceil(8 * s * fps) + 1`` rows long. A
    ``wavelet`` other than its default is refused under ``motionmapper``.

    **Gaps.** With a ``frame`` column, each individual's rows are placed on the
    contiguous frames from its first to its last, the frames it lacks reading as
    NaN, and the transform runs on that range. With only a ``time`` column,
    consecutive rows are read as consecutive samples. A coefficient is NaN where
    its wavelet support covers a missing frame or a NaN of the column, and finite
    otherwise. The high frequencies recover close to a gap, and the low ones stay
    blank longer: a dropout blanks each frequency for half its wavelet's support
    on each side.

    **Track spaces.** The transform reads the columns it is given as series and
    interprets no coordinate, so it computes in every track space. Over
    ``mosaic_world_v1`` tracks a ``Z`` column is one more series.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.individual_wavelet.IndividualWavelet.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.2"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    # Each row keeps the group and sequence of its input row, and an individual's
    # rows from neighboring sequences are one track.
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "individual"
    follows_ids: bool = True
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(TRACK_SPACES)
    # apply transforms the columns that cols names and needs only one of them,
    # so cols sits in an AnyOf, where a ParamColumn alone needs every column.
    # Unset, it transforms the pc_prefix columns, or else every payload column.
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        Column(C.time_col),
        AnyOf((ParamColumn("cols"),)),
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
        """Return ``None``: the reach depends on a frame rate the data sets.

        Under ``pywt`` one row reads half the longest wavelet, ``scale *
        (upper_bound - lower_bound) / 2`` samples at the scale of ``f_min``, and
        the scale is the frame rate over ``f_min`` times the central frequency.
        Under ``motionmapper`` the support is ``8 s fps`` samples at the scale of
        ``f_min``, ``s = (omega0 + sqrt(2 + omega0^2)) / (4 pi f_min)`` seconds,
        and every row also depends within rounding on the whole series, through
        one FFT over the padded series. The frame rate is measured from the
        input's ``time`` column, and a table at 60 frames per second reads twice
        the samples of one at 30, so no params bound the reach.
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
        fps = uniform_grid_rate(df, self.name)
        in_cols = wavelet_input_columns(df, p, _WHO)

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
        values = np.empty((len(df), len(in_cols) * p.n_freq), dtype=wavelet_dtype(p))
        for start, end in zip(starts, ends):
            if frames is None:
                values[start:end] = wavelet_spectra(signal[start:end], p, fps)
                continue
            at = frames[start:end] - frames[start]
            placed = np.full((int(at[-1]) + 1, len(in_cols)), np.nan)
            placed[at] = signal[start:end]
            values[start:end] = wavelet_spectra(placed, p, fps)[at]

        keep = [
            c
            for c in dict.fromkeys(
                (order_col, C.time_col, C.id_col, C.seq_col, C.group_col)
            )
            if c in df.columns
        ]
        spectra = pd.DataFrame(values, columns=wavelet_columns(in_cols, p.n_freq))
        identity = df[keep].iloc[order].reset_index(drop=True)
        out = pd.concat([spectra, identity], axis=1)

        out.attrs.update(wavelet_attrs(p, fps, in_cols))
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
