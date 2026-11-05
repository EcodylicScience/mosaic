"""PairWavelet feature -- CWT spectrograms of numeric columns of pair rows.

The columns are chosen by ``cols``, else the ``pc_prefix`` columns, else every
numeric feature column.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, final

import pandas as pd

from mosaic.core.pipeline.types.feature import RelatesLevel
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
    EmitsLevel,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
    resolve_order_col,
)

from .helpers import ensure_columns
from .registry import register_feature
from .wavelet import (
    WaveletParams,
    constant_fps,
    log_power,
    wavelet_attrs,
    wavelet_band,
    wavelet_columns,
    wavelet_input_columns,
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
      - W_{col}_f{fi} (log-power, clamped, for each component x frequency)
      and (optionally) passthrough group/sequence if present in df.

    Stateless (no fitting). FPS is inferred from constant df['fps'] if
    present, otherwise from fps_default. Frequencies are geometrically spaced
    in [f_min, f_max] (equal steps in log2 frequency).

    The transform is a continuous wavelet transform with PyWavelets, by default
    the complex Morlet wavelet ``cmor1.5-1.0`` at 25 frequencies. Each column
    gives the natural log of the squared coefficient magnitude, clamped from
    below at ``log_floor``.

    Consecutive rows are read as consecutive samples, and missing frames are not
    filled. A NaN in a series makes every coefficient whose wavelet support
    covers it NaN.

    Differences from Berman et al. (2014):

    - The band defaults to 0.2 to 5 Hz, where the original used 1 to 50 Hz.
    - The spectrogram at each time point is not normalized to sum to one. The
      columns are the clamped log power, and a ``global-scaler`` downstream
      standardizes each column across the templates instead.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.pair_wavelet.PairWavelet.Params`.
    """

    category = "per-frame"
    name = "pair-wavelet"
    version = "0.2"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "arena-aware"
    accepts_overlap = True
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    follows_ids: bool = True
    # apply transforms the columns that cols names and needs only one of them.
    # Unset, it transforms the pc_prefix columns, or else every payload column.
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column("perspective"),
        ORDER_COLUMN,
        ParamColumn("cols"),
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
        fps = constant_fps(df, p.sampling.fps_default)
        in_cols = wavelet_input_columns(df, p, _WHO)
        ensure_columns(df, ["perspective"])

        band = wavelet_band(p.wavelet, p.f_min, p.f_max, p.n_freq, fps)
        colnames = wavelet_columns(in_cols, len(band.scales))
        has_pair_ids = "id1" in df.columns and "id2" in df.columns
        group_keys = ["id1", "id2", "perspective"] if has_pair_ids else ["perspective"]

        out_blocks: list[pd.DataFrame] = []
        for _, g in df.groupby(group_keys):
            g = g.sort_values(order_col)
            persp = int(g["perspective"].iloc[0])
            signal = g[in_cols].to_numpy(dtype=float)  # (T, k)
            flat = log_power(signal, band, p.wavelet, fps, p.log_floor)
            block = pd.DataFrame(flat, columns=colnames)
            block[order_col] = g[order_col].to_numpy()
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

        out.attrs.update(wavelet_attrs(band, p, fps, in_cols))
        return out
