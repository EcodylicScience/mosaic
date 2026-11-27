"""PairWavelet feature -- CWT spectrograms of numeric columns of pair rows.

The columns are chosen by ``cols``, else the ``pc_prefix`` columns, else every
numeric feature column.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, final

import pandas as pd

from mosaic.core.pipeline.feature_levels import RelatesLevel
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
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
    resolve_order_col,
)
from mosaic.core.spaces import TRACK_SPACES, TrackSpace
from mosaic.core.timebase import uniform_grid_rate

from .helpers import ensure_columns
from .registry import register_feature
from .wavelet import (
    WaveletParams,
    wavelet_attrs,
    wavelet_columns,
    wavelet_input_columns,
    wavelet_spectra,
)

_WHO = "[pair-wavelet]"


@final
@register_feature
class PairWavelet:
    """
    CWT spectrograms of numeric columns of pair rows.
    The columns are those ``cols`` names, else the ``pc_prefix`` columns, else
    every numeric feature column. It is the spectral stage of a behavioral map
    that follows the approach of Berman et al. (2014), MotionMapper.

    Expects input df to contain columns:
      - 'perspective' (0 = A->B, 1 = B->A)
      - 'frame' (preferred) or 'time' (if used as order column)
      - the payload columns to transform: ``cols``, else the ``pc_prefix``
        columns PC0..PC{k-1}, else every numeric feature column

    Returns a DataFrame with columns:
      - frame (or time if that was the order col)
      - perspective
      - id1, id2 when the input has both
      - W_{col}_f{fi} (log-power or log amplitude, clamped, for each component
        x frequency)
      and (optionally) passthrough group/sequence if present in df.

    Stateless (no fitting). The frame rate is measured from the ``time`` column
    (frames over the time span), and the feature warns when the steps are uneven,
    because the transform reads its rows as evenly spaced. Frequencies are
    geometrically spaced in [f_min, f_max] (equal steps in log2 frequency).

    The transform is a continuous wavelet transform with PyWavelets, by default
    the complex Morlet wavelet ``cmor1.5-1.0`` at 25 frequencies. Each column
    gives the natural log of the squared coefficient magnitude, clamped from
    below at ``log_floor``.

    Consecutive rows are read as consecutive samples, and missing frames are not
    filled. A NaN in a series makes every coefficient whose wavelet support
    covers it NaN.

    **Method.** ``method`` chooses the transform. ``pywt``, the default, is the
    PyWavelets transform above, with float32 output. ``motionmapper`` is the
    Morlet wavelet of Berman et al. (2014) with central angular frequency
    ``omega0``, which that method requires and the other refuses: each column
    gives the natural log of its normalized amplitude, computed by convolving the
    zero-padded series with the wavelet in the Fourier domain, clamped from below
    at ``log_floor``, in float64. A NaN, or a frame the table lacks, blanks the
    rows within 8 scales of it in the columns it touches, and no others. A
    ``wavelet`` other than its default is refused under ``motionmapper``.

    Differences from Berman et al. (2014):

    - The default transform is the PyWavelets log power, and ``motionmapper`` is
      the transform of the original.
    - The band defaults to 0.2 to 5 Hz, where the original used 1 to 50 Hz.
    - The spectrogram at each time point is not normalized to sum to one. The
      columns are the clamped log power, or the log amplitude under
      ``motionmapper``, and a ``global-scaler`` downstream standardizes each
      column across the templates instead.

    **Track spaces.** The transform reads the columns it is given as series and
    interprets no coordinate, so it computes in every track space. Over
    ``mosaic_world_v1`` tracks a ``Z`` column is one more series.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.pair_wavelet.PairWavelet.Params`.
    """

    category = "per-frame"
    name = "pair-wavelet"
    version = "0.3"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "arena-aware"
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    follows_ids: bool = True
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(TRACK_SPACES)
    # apply transforms the columns that cols names and needs only one of them,
    # so cols sits in an AnyOf, where a ParamColumn alone needs every column.
    # Unset, it transforms the pc_prefix columns, or else every payload column.
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column("perspective"),
        Column(C.time_col),
        AnyOf((ParamColumn("cols"),)),
        Payload(),
    )
    # One W_<column>_f<index> column per input column and frequency. The order
    # column keeps its name: frame, or time when frame is absent.
    provides: ClassVar[Provides] = Provides(
        adds=(Prefix("W_"), Column("perspective")),
        conditional=(
            *kept(C.frame_col, "id1", "id2", C.seq_col, C.group_col),
            Conditional(Column(C.time_col)),
        ),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(WaveletParams):
        pass

    def __init__(
        self,
        inputs: PairWavelet.Inputs,
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    @staticmethod
    def temporal_reach(params: PairWavelet.Params) -> int | None:
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
        fps = uniform_grid_rate(df, self.name)
        in_cols = wavelet_input_columns(df, p, _WHO)
        ensure_columns(df, ["perspective"])

        colnames = wavelet_columns(in_cols, p.n_freq)
        has_pair_ids = "id1" in df.columns and "id2" in df.columns
        group_keys = ["id1", "id2", "perspective"] if has_pair_ids else ["perspective"]

        out_blocks: list[pd.DataFrame] = []
        for _, g in df.groupby(group_keys):
            g = g.sort_values(order_col)
            persp = int(g["perspective"].iloc[0])
            signal = g[in_cols].to_numpy(dtype=float)  # (T, k)
            flat = wavelet_spectra(signal, p, fps)
            block = pd.DataFrame(flat, columns=colnames)
            block[order_col] = g[order_col].to_numpy()
            if C.time_col in g.columns:
                block[C.time_col] = g[C.time_col].to_numpy()
            block["perspective"] = persp

            if has_pair_ids:
                block["id1"] = g["id1"].iloc[0]
                block["id2"] = g["id2"].iloc[0]

            # Per row: under overlap the series spans the neighboring entries.
            for col in (C.seq_col, C.group_col):
                if col in g.columns:
                    block[col] = g[col].to_numpy()

            out_blocks.append(block)

        if not out_blocks:
            return pd.DataFrame(columns=[order_col, "perspective"])

        out = pd.concat(out_blocks, ignore_index=True)
        sort_keys: list[str] = []
        if "id1" in out.columns:
            sort_keys += ["id1", "id2"]
        sort_keys += ["perspective", order_col]
        out = out.sort_values(sort_keys).reset_index(drop=True)

        out.attrs.update(wavelet_attrs(p, fps, in_cols))
        return out
