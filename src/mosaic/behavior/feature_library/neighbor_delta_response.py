"""How a focal individual moves, against where each of its neighbours is.

``nn-delta-response`` pairs a focal's next movement with the position of its
nearest neighbour only. In a small group most of the social field is the other
neighbours: with four fish, three quarters of the neighbour positions a map of
where neighbours sit would show are not the nearest one. This feature emits one
row per focal, neighbour and frame instead, so that ``nn-delta-bins`` maps
occupancy, turning and speeding against every neighbour, or against the k
nearest by ``max_neighbor_rank``.

**The shuffled control.** A map of where neighbours sit is shaped by the arena
as well as by attraction: two fish confined to one dish are near each other more
often than chance in open water would put them. With ``time_shuffle`` each
neighbour's trajectory is shifted in time by its own random offset, circularly
over the sequence, before it is placed in the focal's frame. The focal's own
movement is untouched. What survives the shift is what the arena and each
animal's own habits produce; what disappears is the coordination between them.
A run with the shuffle and a run without it are two run identities, binned the
same way and compared.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Annotated, ClassVar, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.individuals import id_arenas, individual_pairs
from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.pipeline.column_contract import (
    AnyOf,
    Column,
    Conditional,
    ParamColumn,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.params import Declared, Params

from ._sampling_descriptions import SAMPLING_FPS_ONLY_DESCRIPTION
from .helpers import ego_rotate, ensure_columns, one_row_per_frame_and_id, wrap_angle
from .nn_delta_response import windowed_delta
from .registry import register_feature
from .types import SamplingConfig

_SPEED_COL_DESCRIPTION = (
    "The focal individual's speed column. Falls back to speed when absent."
)

_FOCAL_COL_DESCRIPTION = (
    "The column flagging a focal individual. Passed through for the focal "
    "and looked up for the neighbor as neighbor_focal when present, the "
    "category columns nn-delta-bins splits by."
)

_DIFF_NUMFRAMES_DESCRIPTION = "How far ahead the response delta is measured."

_MAX_NEIGHBOR_RANK_DESCRIPTION = (
    "Keep only the neighbors this close in rank, 1 being the nearest in "
    "that frame. 1 reproduces the nearest-neighbor pairing of "
    "nn-delta-response. Unset, every neighbor."
)

_WRAP_ANGLE_DESCRIPTION = (
    "Wrap the heading difference to [-pi, pi]. When False, the raw difference is kept."
)

_DIVIDE_DANGLE_BY_FRAMES_DESCRIPTION = (
    "Divide the heading change by the number of frames it spans. When "
    "False, the raw windowed difference is kept."
)

_SCALE_DANGLE_BY_FPS_DESCRIPTION = (
    "Multiply the heading change by the sampling frame rate. Together "
    "with divide_dangle_by_frames this gives radians per second."
)

_TIME_SHUFFLE_DESCRIPTION = (
    "Shift each neighbor's trajectory in time, circularly over the "
    "sequence and by its own random offset, before placing it in the "
    "focal's frame: the chance-level control of a neighbor map. The "
    "focal's own movement is not shifted."
)

_SHUFFLE_SEED_DESCRIPTION = (
    "Seed of the shuffle offsets, combined with the group and sequence so "
    "that each sequence draws its own offsets and every run draws the same."
)

_SHUFFLE_MIN_SHIFT_DESCRIPTION = (
    "The smallest time shift of a shuffled neighbor, in either direction. "
    "A sequence spanning no more than twice this shift refuses to shuffle."
)

_SPEED_FALLBACK = "speed"
"""The speed column read when the table lacks ``speed_col``."""

_NAME = "neighbor-delta-response"


@final
@register_feature
class NeighborDeltaResponse:
    """Change in a focal individual's position, heading and speed, by each neighbor.

    For every focal row with a valid future sample ``diff_numframes`` ahead,
    and every other individual present in that frame, one row: ``id1`` the
    focal, ``id2`` the neighbor, ``perspective`` 0 when the focal has the lower
    id. ``neighbor_x`` and ``neighbor_y`` place the neighbor in the focal's ego
    frame (heading along +x), ``neighbor_dist`` is their distance and
    ``neighbor_rank`` its rank among the focal's neighbors in that frame.
    ``dx``, ``dy``, ``dt``, ``dangle`` and ``dspeed`` are the focal's response,
    as ``nn-delta-response`` computes it, so ``nn-delta-bins`` reads either.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.neighbor_delta_response.NeighborDeltaResponse.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "arena-aware"
    accepts_overlap = False  # the shuffle wraps around one entry's whole span
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        Column(C.x_col),
        Column(C.y_col),
        Column(C.orientation_col),
        AnyOf((ParamColumn("speed_col"), Column(_SPEED_FALLBACK))),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.frame_col),
            Column("id1"),
            Column("id2"),
            Column("perspective"),
            Column("neighbor_x"),
            Column("neighbor_y"),
            Column("neighbor_dist"),
            Column("neighbor_rank"),
            Column("dx"),
            Column("dy"),
            Column("dt"),
            Column("dangle"),
            Column("dspeed"),
        ),
        conditional=(
            *kept(C.time_col, C.group_col, C.seq_col),
            Conditional(
                ParamColumn("focal_col"), when_present=ParamColumn("focal_col")
            ),
            Conditional(
                Column("neighbor_focal"), when_present=ParamColumn("focal_col")
            ),
        ),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        sampling: Annotated[SamplingConfig, Declared(SAMPLING_FPS_ONLY_DESCRIPTION)] = (
            Field(default_factory=SamplingConfig)
        )
        speed_col: Annotated[str, Declared(_SPEED_COL_DESCRIPTION)] = "SPEED"
        focal_col: Annotated[str, Declared(_FOCAL_COL_DESCRIPTION)] = "Focal_fish"
        diff_numframes: Annotated[
            int, Declared(_DIFF_NUMFRAMES_DESCRIPTION, unit="frames")
        ] = Field(default=4, ge=1)
        max_neighbor_rank: Annotated[
            int | None, Declared(_MAX_NEIGHBOR_RANK_DESCRIPTION)
        ] = Field(default=None, ge=1)
        wrap_angle: Annotated[bool, Declared(_WRAP_ANGLE_DESCRIPTION)] = True
        divide_dangle_by_frames: Annotated[
            bool, Declared(_DIVIDE_DANGLE_BY_FRAMES_DESCRIPTION)
        ] = True
        scale_dangle_by_fps: Annotated[
            bool, Declared(_SCALE_DANGLE_BY_FPS_DESCRIPTION)
        ] = True
        time_shuffle: Annotated[bool, Declared(_TIME_SHUFFLE_DESCRIPTION)] = False
        shuffle_seed: Annotated[int, Declared(_SHUFFLE_SEED_DESCRIPTION)] = 0
        shuffle_min_shift: Annotated[
            int, Declared(_SHUFFLE_MIN_SHIFT_DESCRIPTION, unit="frames")
        ] = Field(default=300, ge=1)

    def __init__(
        self,
        inputs: NeighborDeltaResponse.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

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
        if df.empty:
            return pd.DataFrame()
        p = self.params
        speed_col = p.speed_col if p.speed_col in df.columns else _SPEED_FALLBACK
        ensure_columns(
            df,
            [C.frame_col, C.id_col, C.x_col, C.y_col, C.orientation_col, speed_col],
        )
        rows = one_row_per_frame_and_id(df).reset_index(drop=True)
        ids = sorted(rows[C.id_col].unique().tolist())
        if len(ids) < 2:
            return pd.DataFrame()

        frames = rows[C.frame_col].to_numpy(dtype=np.int64)
        first = int(frames.min())
        n_frames = int(frames.max()) - first + 1
        frame_idx = frames - first
        # ids is sorted, so each id's slot on the grid is where it sorts into it.
        id_idx = np.searchsorted(np.asarray(ids), rows[C.id_col].to_numpy())

        # Every individual on one frame grid, NaN where it was not tracked.
        grid_x = np.full((n_frames, len(ids)), np.nan)
        grid_y = np.full((n_frames, len(ids)), np.nan)
        grid_x[frame_idx, id_idx] = rows[C.x_col].to_numpy(dtype=float)
        grid_y[frame_idx, id_idx] = rows[C.y_col].to_numpy(dtype=float)
        focal_flags = None
        if p.focal_col in rows.columns:
            focal_flags = np.full((n_frames, len(ids)), np.nan, dtype=object)
            focal_flags[frame_idx, id_idx] = rows[p.focal_col].to_numpy()

        if p.time_shuffle:
            shifts = self._shuffle_shifts(rows, len(ids), n_frames)
            for k, shift in enumerate(shifts):
                grid_x[:, k] = np.roll(grid_x[:, k], shift)
                grid_y[:, k] = np.roll(grid_y[:, k], shift)
                if focal_flags is not None:
                    focal_flags[:, k] = np.roll(focal_flags[:, k], shift)

        fps = p.sampling.fps_default
        lag = p.diff_numframes
        delta_cols = [C.x_col, C.y_col, C.orientation_col, speed_col, C.frame_col]
        i_x, i_y, i_ang, i_spd, i_frm = range(5)
        carried = [c for c in (C.time_col, C.group_col, C.seq_col) if c in rows]

        partners = set(individual_pairs(ids, id_arenas(df), ordered=True))
        outputs: list[pd.DataFrame] = []
        for focal_id in ids:
            g = rows[rows[C.id_col] == focal_id]
            vals = g[delta_cols].to_numpy(dtype=float)
            lags = np.full(len(g), lag, dtype=np.int64)
            fwd, ok = windowed_delta(vals, lags, i_frm, forward=True)
            if not ok.any():
                continue
            sel = g.loc[ok]
            fwd = fwd[ok]
            at = sel[C.frame_col].to_numpy(dtype=np.int64) - first
            others = [k for k, fid in enumerate(ids) if (focal_id, fid) in partners]
            if not others:
                continue

            # (rows, neighbors) offsets in the focal's ego frame.
            wdx = grid_x[at][:, others] - vals[ok, i_x][:, None]
            wdy = grid_y[at][:, others] - vals[ok, i_y][:, None]
            heading = np.repeat(vals[ok, i_ang][:, None], len(others), axis=1)
            ex, ey = ego_rotate(wdx, wdy, heading)
            dist = np.hypot(wdx, wdy)
            rank = _rank_rows(dist)

            keep = np.isfinite(ex) & np.isfinite(ey) & np.isfinite(heading)
            if p.max_neighbor_rank is not None:
                keep &= rank <= p.max_neighbor_rank
            r_idx, n_idx = np.nonzero(keep)
            if r_idx.size == 0:
                continue

            dangle = fwd[r_idx, i_ang]
            if p.wrap_angle:
                dangle = wrap_angle(dangle)
            if p.divide_dangle_by_frames:
                dangle = dangle / lag
            if p.scale_dangle_by_fps:
                dangle = dangle * fps
            neighbor_ids = np.asarray(ids, dtype=object)[np.asarray(others)[n_idx]]

            out = pd.DataFrame(
                {C.frame_col: sel[C.frame_col].to_numpy()[r_idx]},
            )
            for c in carried:
                out[c] = sel[c].to_numpy()[r_idx]
            out["id1"] = focal_id
            out["id2"] = neighbor_ids.tolist()
            out["perspective"] = np.array(
                [0 if focal_id < n else 1 for n in neighbor_ids], dtype=np.int64
            )
            out["neighbor_x"] = ex[r_idx, n_idx]
            out["neighbor_y"] = ey[r_idx, n_idx]
            out["neighbor_dist"] = dist[r_idx, n_idx]
            out["neighbor_rank"] = rank[r_idx, n_idx].astype(np.int64)
            out["dx"] = fwd[r_idx, i_x]
            out["dy"] = fwd[r_idx, i_y]
            out["dt"] = fwd[r_idx, i_frm]
            out["dangle"] = dangle
            out["dspeed"] = fwd[r_idx, i_spd]
            if focal_flags is not None:
                out[p.focal_col] = sel[p.focal_col].to_numpy()[r_idx]
                out["neighbor_focal"] = focal_flags[at][:, others][r_idx, n_idx]
            outputs.append(out)

        if not outputs:
            return pd.DataFrame()
        result = pd.concat(outputs, ignore_index=True)
        return result.sort_values(
            [C.frame_col, "id1", "id2"], kind="stable"
        ).reset_index(drop=True)

    def _shuffle_shifts(
        self, rows: pd.DataFrame, n_ids: int, n_frames: int
    ) -> np.ndarray:
        """One circular time shift per individual, seeded per sequence."""
        p = self.params
        min_shift = p.shuffle_min_shift
        if n_frames <= 2 * min_shift:
            raise ValueError(
                f"{self.name}: time_shuffle shifts each neighbor by at least "
                f"shuffle_min_shift={min_shift} frames in either direction, and "
                f"this sequence spans {n_frames}. Lower shuffle_min_shift."
            )
        entry = "/".join(
            str(rows[c].iloc[0]) if c in rows.columns else ""
            for c in (C.group_col, C.seq_col)
        )
        rng = np.random.default_rng([p.shuffle_seed, zlib.crc32(entry.encode())])
        return rng.integers(min_shift, n_frames - min_shift + 1, size=n_ids)


def _rank_rows(dist: np.ndarray) -> np.ndarray:
    """Rank each row's finite distances from 1, nearest first; 0 where not finite."""
    finite = np.isfinite(dist)
    order = np.argsort(np.where(finite, dist, np.inf), axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(dist.shape[0])[:, None]
    rank[rows, order] = np.arange(1, dist.shape[1] + 1)
    return np.where(finite, rank, 0)
