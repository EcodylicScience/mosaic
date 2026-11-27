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
from typing import Annotated, ClassVar, Self, final, override

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from mosaic.core.individuals import id_arenas, individual_pairs
from mosaic.core.pipeline.feature_levels import RelatesLevel
from mosaic.core.pipeline.column_contract import (
    AnyOf,
    Column,
    Conditional,
    ParamColumn,
    ParamIs,
    Provides,
    Requirement,
    When,
    kept,
)
from mosaic.core.pipeline.types import COLUMNS as C
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.params import Declared, Params, has_fields

from .helpers import ensure_columns, one_row_per_frame_and_id, wrap_angle
from .nn_delta_response import scaled_dangle, windowed_delta
from .registry import register_feature

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
    "Multiply the heading change by the window's own frame rate, its "
    "frames over the seconds between its two rows (the time column). "
    "Together with divide_dangle_by_frames this gives radians per "
    "second, whatever frames the track dropped."
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

_X_COL_DESCRIPTION = (
    "The x position of the focal and the neighbor, read for the offsets, the "
    "frame grid and the windowed delta."
)

_Y_COL_DESCRIPTION = (
    "The y position of the focal and the neighbor, read for the offsets, the "
    "frame grid and the windowed delta."
)

_DELTA_RESPONSE_DESCRIPTION = (
    "Write the focal's windowed delta: dx, dy, dt, dangle and dspeed. When "
    "False they are not computed, the window no longer limits the rows, and a "
    "non-default diff_numframes, wrap_angle, divide_dangle_by_frames or "
    "scale_dangle_by_fps is refused."
)

_REFERENCE_COLS_DESCRIPTION = (
    "Two columns [vx, vy], a vector per focal row that sets the frame of "
    "neighbor_x (along it) and neighbor_y (to its left) in place of the "
    "heading. Unset, the frame follows the heading column. A focal row whose "
    "vector is zero or missing has no frame, and its rows are left out."
)

_RESPONSE_COLS_DESCRIPTION = (
    "Two columns [ax, ay] of the focal's own row, projected on the frame as "
    "response_par and response_perp. A focal row where either is not finite "
    "is left out."
)

_PARTNER_COLS_DESCRIPTION = (
    "Two columns [vx, vy] of the neighbor's row in the same frame, projected "
    "on the focal's frame as partner_par and partner_perp, with their norm as "
    "partner_norm. They are NaN where the neighbor has no value. With "
    "time_shuffle they move with the neighbor's position."
)

_VECTOR_PAIR_FIELDS = ("reference_cols", "response_cols", "partner_cols")
"""The params fields that each name two columns, a vector's components."""

_DELTA_OPTION_FIELDS = (
    "diff_numframes",
    "wrap_angle",
    "divide_dangle_by_frames",
    "scale_dangle_by_fps",
)
"""The params that shape the windowed delta, which ``delta_response=False`` skips."""

_IDENTITY_OPTIONAL_FIELDS = (
    "x_col",
    "y_col",
    "delta_response",
    *_VECTOR_PAIR_FIELDS,
)
"""The fields that leave a run's identity while they hold their defaults."""

_SPEED_FALLBACK = "speed"
"""The speed column read when the table lacks ``speed_col``."""

_DELTA_COLUMNS = ("dx", "dy", "dt", "dangle", "dspeed")
_RESPONSE_COLUMNS = ("response_par", "response_perp")
_PARTNER_COLUMNS = ("partner_par", "partner_perp", "partner_norm")

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

    With ``reference_cols`` the frame follows a vector per focal row instead of
    the heading, ``response_cols`` projects the focal's own vector on it
    (``response_par``, ``response_perp``), and ``partner_cols`` projects the
    neighbor's vector in the same frame (``partner_par``, ``partner_perp``,
    ``partner_norm``). ``delta_response=False`` skips the windowed delta, and a
    focal row then counts where its response and its offsets are finite.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.neighbor_delta_response.NeighborDeltaResponse.Params`.
    """

    category = "per-frame"
    keywords: ClassVar[tuple[str, ...]] = ("social force", "force map")
    name = _NAME
    version = "0.2"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "arena-aware"
    accepts_overlap = False  # the shuffle wraps around one entry's whole span
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    follows_ids: bool = True
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        Column(C.id_col),
        ParamColumn("x_col"),
        ParamColumn("y_col"),
        ParamColumn("reference_cols"),
        ParamColumn("response_cols"),
        ParamColumn("partner_cols"),
        When(Column(C.orientation_col), (ParamIs("reference_cols", negated=True),)),
        When(Column(C.orientation_col), (ParamIs("delta_response"),)),
        When(Column(C.time_col), (ParamIs("delta_response"),)),
        When(
            AnyOf((ParamColumn("speed_col"), Column(_SPEED_FALLBACK))),
            (ParamIs("delta_response"),),
        ),
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
        ),
        conditional=(
            *(
                Conditional(Column(name), when_param="delta_response")
                for name in _DELTA_COLUMNS
            ),
            *(
                Conditional(Column(name), when_param="response_cols")
                for name in _RESPONSE_COLUMNS
            ),
            *(
                Conditional(Column(name), when_param="partner_cols")
                for name in _PARTNER_COLUMNS
            ),
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
        x_col: Annotated[str, Declared(_X_COL_DESCRIPTION)] = "X"
        y_col: Annotated[str, Declared(_Y_COL_DESCRIPTION)] = "Y"
        delta_response: Annotated[bool, Declared(_DELTA_RESPONSE_DESCRIPTION)] = True
        reference_cols: Annotated[
            list[str] | None, Declared(_REFERENCE_COLS_DESCRIPTION)
        ] = None
        response_cols: Annotated[
            list[str] | None, Declared(_RESPONSE_COLS_DESCRIPTION)
        ] = None
        partner_cols: Annotated[
            list[str] | None, Declared(_PARTNER_COLS_DESCRIPTION)
        ] = None
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

        @field_validator(*_VECTOR_PAIR_FIELDS, check_fields=False)
        @classmethod
        def _a_vector_is_two_distinct_columns(
            cls, value: list[str] | None
        ) -> list[str] | None:
            """Refuse a list of other than two columns, or of one column twice."""
            if value is None:
                return value
            if len(value) != 2:
                raise ValueError(
                    f"{_NAME}: a vector is two columns [x, y], and this lists "
                    f"{len(value)}."
                )
            if value[0] == value[1]:
                raise ValueError(
                    f"{_NAME}: a vector is two distinct columns, and this lists "
                    f"{value[0]!r} twice."
                )
            return value

        @model_validator(mode="after")
        def _delta_options_need_the_delta(self) -> Self:
            """Refuse a changed delta option while ``delta_response`` is False."""
            if not has_fields(self, "delta_response", *_DELTA_OPTION_FIELDS):
                return self
            if self.delta_response:
                return self
            fields = type(self).model_fields
            for name in _DELTA_OPTION_FIELDS:
                if getattr(self, name) != fields[name].default:
                    raise ValueError(
                        f"{_NAME}: {name} shapes the windowed delta, which "
                        "delta_response=False does not compute. Leave it at "
                        f"{fields[name].default!r}."
                    )
            return self

        @override
        def identity_dump(self) -> dict[str, object]:
            """The hashed params, less each new field while it holds its default.

            A run that sets none of them writes the output of the version
            before they existed, so it keeps its identifier.
            """
            dumped = super().identity_dump()
            fields = type(self).model_fields
            for name in _IDENTITY_OPTIONAL_FIELDS:
                if getattr(self, name) == fields[name].default:
                    del dumped[name]
            return dumped

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
        required = [C.frame_col, C.id_col, p.x_col, p.y_col]
        for pair in (p.reference_cols, p.response_cols, p.partner_cols):
            required.extend(pair or [])
        if p.delta_response:
            required.extend([C.time_col, speed_col])
        if p.delta_response or p.reference_cols is None:
            required.append(C.orientation_col)
        ensure_columns(df, required)
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

        def grid_of(column: str) -> np.ndarray:
            """One column of every individual on the frame grid, NaN where untracked."""
            grid = np.full((n_frames, len(ids)), np.nan)
            grid[frame_idx, id_idx] = rows[column].to_numpy(dtype=float)
            return grid

        grid_x = grid_of(p.x_col)
        grid_y = grid_of(p.y_col)
        partner_x, partner_y = (
            (grid_of(p.partner_cols[0]), grid_of(p.partner_cols[1]))
            if p.partner_cols
            else (None, None)
        )
        focal_flags = None
        if p.focal_col in rows.columns:
            focal_flags = np.full((n_frames, len(ids)), np.nan, dtype=object)
            focal_flags[frame_idx, id_idx] = rows[p.focal_col].to_numpy()

        if p.time_shuffle:
            shifts = self._shuffle_shifts(rows, len(ids), n_frames)
            for k, shift in enumerate(shifts):
                for grid in (grid_x, grid_y, partner_x, partner_y):
                    if grid is not None:
                        grid[:, k] = np.roll(grid[:, k], shift)
                if focal_flags is not None:
                    focal_flags[:, k] = np.roll(focal_flags[:, k], shift)

        lag = p.diff_numframes
        delta_cols = [
            p.x_col,
            p.y_col,
            C.orientation_col,
            speed_col,
            C.frame_col,
            C.time_col,
        ]
        i_x, i_y, i_ang, i_spd, i_frm, i_sec = range(6)
        carried = [c for c in (C.time_col, C.group_col, C.seq_col) if c in rows]

        partners = set(individual_pairs(ids, id_arenas(df), ordered=True))
        outputs: list[pd.DataFrame] = []
        for focal_id in ids:
            g = _rows_of(rows, focal_id)
            window = None
            ok = np.ones(len(g), dtype=bool)
            if p.delta_response:
                vals = g[delta_cols].to_numpy(dtype=float)
                lags = np.full(len(g), lag, dtype=np.int64)
                window, ok = windowed_delta(vals, lags, i_frm, forward=True)
            response = None
            if p.response_cols:
                response = g[p.response_cols].to_numpy(dtype=float)
                ok = ok & np.isfinite(response).all(axis=1)
            if not ok.any():
                continue
            sel = g.loc[ok]
            at = sel[C.frame_col].to_numpy(dtype=np.int64) - first
            others = [k for k, fid in enumerate(ids) if (focal_id, fid) in partners]
            if not others:
                continue

            ux, uy = _unit_vectors(sel, p.reference_cols)
            px, py = -uy, ux

            # (rows, neighbors) offsets in the focal's frame.
            wdx = grid_x[at][:, others] - sel[p.x_col].to_numpy(dtype=float)[:, None]
            wdy = grid_y[at][:, others] - sel[p.y_col].to_numpy(dtype=float)[:, None]
            ex = wdx * ux[:, None] + wdy * uy[:, None]
            ey = wdx * px[:, None] + wdy * py[:, None]
            dist = np.hypot(wdx, wdy)
            rank = _rank_rows(dist)

            keep = np.isfinite(ex) & np.isfinite(ey)
            if p.max_neighbor_rank is not None:
                keep &= rank <= p.max_neighbor_rank
            r_idx, n_idx = np.nonzero(keep)
            if r_idx.size == 0:
                continue

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
            if window is not None:
                fwd = window[ok]
                dangle = fwd[r_idx, i_ang]
                if p.wrap_angle:
                    dangle = wrap_angle(dangle)
                dangle = scaled_dangle(
                    dangle,
                    np.full(dangle.shape, lag, dtype=np.int64),
                    fwd[r_idx, i_sec],
                    divide_by_frames=p.divide_dangle_by_frames,
                    scale_by_rate=p.scale_dangle_by_fps,
                )
                out["dx"] = fwd[r_idx, i_x]
                out["dy"] = fwd[r_idx, i_y]
                out["dt"] = fwd[r_idx, i_frm]
                out["dangle"] = dangle
                out["dspeed"] = fwd[r_idx, i_spd]
            if response is not None:
                ax, ay = response[ok, 0], response[ok, 1]
                out["response_par"] = (ax * ux + ay * uy)[r_idx]
                out["response_perp"] = (ax * px + ay * py)[r_idx]
            if partner_x is not None and partner_y is not None:
                vx2 = partner_x[at][:, others]
                vy2 = partner_y[at][:, others]
                par = vx2 * ux[:, None] + vy2 * uy[:, None]
                perp = vx2 * px[:, None] + vy2 * py[:, None]
                out["partner_par"] = par[r_idx, n_idx]
                out["partner_perp"] = perp[r_idx, n_idx]
                out["partner_norm"] = np.sqrt(vx2**2 + vy2**2)[r_idx, n_idx]
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


def _rows_of(rows: pd.DataFrame, focal_id: object) -> pd.DataFrame:
    """The rows of one individual, in their order."""
    return rows[rows[C.id_col] == focal_id]


def _unit_vectors(
    rows: pd.DataFrame, reference_cols: list[str] | None
) -> tuple[np.ndarray, np.ndarray]:
    """The unit vector of each row's frame, along its reference.

    The reference is the heading's direction, or the vector in *reference_cols*.
    A zero or missing vector gives NaN, which leaves the row's neighbors out.
    """
    if reference_cols is None:
        heading = rows[C.orientation_col].to_numpy(dtype=float)
        return np.cos(heading), np.sin(heading)
    vx = rows[reference_cols[0]].to_numpy(dtype=float)
    vy = rows[reference_cols[1]].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.sqrt(vx**2 + vy**2)
        return vx / norm, vy / norm


def _rank_rows(dist: np.ndarray) -> np.ndarray:
    """Rank each row's finite distances from 1, nearest first; 0 where not finite."""
    finite = np.isfinite(dist)
    order = np.argsort(np.where(finite, dist, np.inf), axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(dist.shape[0])[:, None]
    rank[rows, order] = np.arange(1, dist.shape[1] + 1)
    return np.where(finite, rank, 0)
