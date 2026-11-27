"""
PairInteractionFilter -- detect pairwise interaction segments from trajectories.

Identifies frames where pairs of individuals meet configurable distance and
angular thresholds.  Applies morphological filtering to remove noise and
enforces a minimum interaction duration.

Typical use cases:
  - Detecting face-to-face interactions (distance + facing criterion)
  - Proximity-based pair detection (distance only, ``require_facing=False``)
  - Pre-filtering for expensive downstream processing (e.g. interaction crops)

All thresholds are parameterized and should be tuned per application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Final, Self, final, override

import numpy as np
import pandas as pd
from pydantic import model_validator
from scipy.ndimage import binary_closing, binary_opening

from mosaic.core.individuals import id_arenas, individual_pairs
from mosaic.core.pipeline.feature_levels import RelatesLevel
from mosaic.core.pipeline.column_contract import (
    ORDER_COLUMN,
    POSE_X,
    POSE_Y,
    AnyOf,
    Column,
    ParamColumn,
    ParamIs,
    Provides,
    Requirement,
    Template,
    When,
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
from mosaic.core.params import Declared, Params

from .helpers import ensure_columns, with_frame_times
from .registry import register_feature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_interaction_conditions(
    x_a: np.ndarray,
    y_a: np.ndarray,
    orient_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    orient_b: np.ndarray,
    shift_dist: float,
    max_dist: float,
    max_inv_orientation_diff_deg: float,
    require_facing: bool,
) -> np.ndarray:
    """Per-frame boolean: do two individuals meet distance + angle criteria?

    Both positions are shifted forward in their respective orientation
    directions before computing distance.  The angular test checks whether
    the individuals are roughly facing each other (inverse orientation
    difference < threshold).
    """
    # Shift positions forward along each individual's heading
    x_a_shifted = x_a + np.cos(orient_a) * shift_dist
    y_a_shifted = y_a + np.sin(orient_a) * shift_dist
    x_b_shifted = x_b + np.cos(orient_b) * shift_dist
    y_b_shifted = y_b + np.sin(orient_b) * shift_dist

    dist = np.hypot(x_a_shifted - x_b_shifted, y_a_shifted - y_b_shifted)
    meets = dist < max_dist

    if require_facing:
        # Circular orientation difference (shortest arc)
        orient_diff = np.abs(orient_a - orient_b)
        orient_diff = np.minimum(orient_diff, 2 * np.pi - orient_diff)

        # Check if individuals face roughly *opposite* directions
        inv_orient_diff = np.pi - orient_diff
        angle_ok = inv_orient_diff < np.deg2rad(max_inv_orientation_diff_deg)
        meets = meets & angle_ok

    return meets


def _remove_bool_islands(
    arr: np.ndarray,
    structure_length: int = 25,
) -> np.ndarray:
    """Remove short boolean islands via morphological close then open."""
    structure = np.ones(structure_length, dtype=bool)
    # Closing first: fills small False gaps inside True regions
    return binary_opening(binary_closing(arr, structure=structure), structure=structure)


def _find_long_true_runs(
    arr: np.ndarray,
    min_run: int = 250,
    n_frame_padding: int = 10,
) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for True runs >= *min_run* frames.

    Each run is padded by *n_frame_padding* on both sides (clamped to array
    bounds).  Indices are relative to *arr*.
    """
    if not arr.any():
        return []

    edges = np.flatnonzero(np.diff(arr.astype(np.int8)))
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges, len(arr) - 1]

    true_mask = arr[starts]
    true_starts = starts[true_mask]
    true_ends = ends[true_mask]
    lengths = true_ends - true_starts + 1

    keep = lengths >= min_run
    true_starts = true_starts[keep]
    true_ends = true_ends[keep]

    runs = []
    for s, e in zip(true_starts, true_ends):
        padded_start = max(0, s - n_frame_padding)
        padded_end = min(len(arr), e + 1 + n_frame_padding)
        runs.append((padded_start, padded_end))
    return runs


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------

_SHIFT_DIST_DESCRIPTION = (
    "Distance each individual is shifted forward along its heading "
    "before the distance check runs. 0 uses raw positions with no shift."
)

_MAX_DIST_DESCRIPTION = (
    "The maximum distance between the shifted positions for a pair to "
    "count as interacting."
)

_REQUIRE_FACING_DESCRIPTION = (
    "Require the pair to also face each other, using "
    "max_inv_orientation_diff_deg. False filters on distance alone."
)

_MAX_INV_ORIENTATION_DIFF_DEG_DESCRIPTION = (
    "The maximum inverse-orientation angle for a pair to count as "
    "facing. Ignored when require_facing is false."
)

_MIN_RUN_FRAMES_DESCRIPTION = (
    "The minimum length a continuous run meeting the interaction "
    "criteria needs to count as a segment."
)

_FRAME_PADDING_DESCRIPTION = "Padding added before and after each detected segment."

_MORPHOLOGICAL_STRUCTURE_SIZE_DESCRIPTION = (
    "Length of the structuring element used to close and open the "
    "per-frame interaction mask before segment extraction. 0 disables "
    "morphological filtering."
)

_PX_SCALE_DESCRIPTION = (
    "Scale factor applied to shift_dist and max_dist, for videos "
    "recorded at a different pixel resolution."
)

_USE_PIXEL_COORDS_DESCRIPTION = (
    "Use pixel-coordinate columns for the distance calculation. With "
    "pose_head_index set, reads that pose keypoint (poseX<i>/poseY<i>). "
    "Otherwise reads x_pixels/y_pixels where the table has them and X/Y "
    "where it does not, without a message, so a table that lacks the pixel "
    "columns is measured on X/Y. False always reads X/Y. Ignored while "
    "position_columns is set, and refused as False beside it."
)

_POSE_HEAD_INDEX_DESCRIPTION = (
    "Pose keypoint index used as the position for the distance "
    "calculation, together with use_pixel_coords: reads poseX<i> and "
    "poseY<i>. Unset, x_pixels/y_pixels or X/Y is read instead. Refused "
    "beside position_columns."
)

_POSITION_COLUMNS_DESCRIPTION = (
    "The two columns, x then y, that give each individual's position for "
    "the distance calculation, as ['X#head', 'Y#head'] for the head point "
    "that a TRex track keeps beside its body centroid in X/Y. Both columns "
    "are read as named: a table that lacks either is refused, with no "
    "fallback to X/Y. Unset, the columns follow use_pixel_coords and "
    "pose_head_index. Refused beside pose_head_index or use_pixel_coords "
    "set False."
)


_HEAD_X: Final = Template(POSE_X.prefix + "{pose_head_index}")
"""The x column of the keypoint that ``pose_head_index`` names."""

_HEAD_Y: Final = Template(POSE_Y.prefix + "{pose_head_index}")
"""The y column of the keypoint that ``pose_head_index`` names."""


_NAME: Final = "pair-interaction-filter"


@final
@register_feature
class PairInteractionFilter:
    """Detect pairwise interaction segments from trajectory data.

    For every unique pair of individuals in a sequence, tests per-frame
    distance and (optionally) angular criteria, applies morphological
    filtering, and extracts continuous interaction segments that meet a
    minimum duration.

    The position that the distance check reads is, in the order of the checks:
      - ``position_columns``: exactly the two columns it names, x then y, for
        example ``X#head``/``Y#head`` for the head point of a TRex track. A table
        that lacks either is refused.
      - ``use_pixel_coords`` with ``pose_head_index``: ``poseX<i>``/``poseY<i>``.
      - ``use_pixel_coords`` alone: ``x_pixels``/``y_pixels`` where the table has
        them, else ``X``/``Y`` without a message.
      - ``use_pixel_coords=False``: ``X``/``Y``.

    Output columns (two rows per frame per interaction segment, one per ordered
    pair, keyed by ``(frame, id1, id2, perspective)``):
      - frame: frame number
      - id1: the focal individual, id2: the other
      - perspective: 0 for the ordering with the lower id focal, 1 for its mirror
      - interaction_id: integer label for the segment within this pair
      - interaction_start: first frame of this segment
      - interaction_end: last frame (exclusive) of this segment

    The criteria are symmetric in the two individuals, so the mirrored row holds
    the same values. It is written anyway: one shape for every pair feature is
    what lets any two of them join, and a lone unordered row matches only half of
    an ordered partner's rows -- silently, since the join reads as a broadcast.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.pair_interaction_filter.PairInteractionFilter.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.3"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "arena-aware"
    accepts_overlap = False  # computes within a frame, so gains nothing
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "pair"
    follows_ids: bool = True
    # The position columns are the pair that position_columns names. Unset, they follow
    # use_pixel_coords and pose_head_index: the head keypoint, x_pixels/y_pixels
    # where present and else X/Y, or X/Y.
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.id_col),
        Column(C.orientation_col),
        ORDER_COLUMN,
        ParamColumn("position_columns"),
        *(
            When(position, (ParamIs("use_pixel_coords"), ParamIs("pose_head_index")))
            for position in (_HEAD_X, _HEAD_Y)
        ),
        *(
            When(
                AnyOf((Column(pixels), Column(center))),
                (
                    ParamIs("use_pixel_coords"),
                    ParamIs("pose_head_index", negated=True),
                    ParamIs("position_columns", negated=True),
                ),
            )
            for pixels, center in (("x_pixels", C.x_col), ("y_pixels", C.y_col))
        ),
        *(
            When(Column(center), (ParamIs("use_pixel_coords", negated=True),))
            for center in (C.x_col, C.y_col)
        ),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(
            Column(C.frame_col),
            Column("id1"),
            Column("id2"),
            Column("perspective"),
            Column("interaction_id"),
            Column("interaction_start"),
            Column("interaction_end"),
        ),
        conditional=kept(C.time_col, C.group_col, C.seq_col),
        # One segment, written on both rows of the pair.
        symmetric=(
            Column("interaction_id"),
            Column("interaction_start"),
            Column("interaction_end"),
        ),
    )

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        shift_dist: Annotated[float, Declared(_SHIFT_DIST_DESCRIPTION, unit="px")] = (
            15.0
        )
        max_dist: Annotated[float, Declared(_MAX_DIST_DESCRIPTION, unit="px")] = 40.0
        require_facing: Annotated[bool, Declared(_REQUIRE_FACING_DESCRIPTION)] = True
        max_inv_orientation_diff_deg: Annotated[
            float,
            Declared(_MAX_INV_ORIENTATION_DIFF_DEG_DESCRIPTION, unit="deg"),
        ] = 80.0
        min_run_frames: Annotated[
            int, Declared(_MIN_RUN_FRAMES_DESCRIPTION, unit="frames")
        ] = 250
        frame_padding: Annotated[
            int, Declared(_FRAME_PADDING_DESCRIPTION, unit="frames")
        ] = 10
        morphological_structure_size: Annotated[
            int, Declared(_MORPHOLOGICAL_STRUCTURE_SIZE_DESCRIPTION, unit="frames")
        ] = 25
        px_scale: Annotated[float, Declared(_PX_SCALE_DESCRIPTION)] = 1.0
        use_pixel_coords: Annotated[bool, Declared(_USE_PIXEL_COORDS_DESCRIPTION)] = (
            True
        )
        pose_head_index: Annotated[
            int | None, Declared(_POSE_HEAD_INDEX_DESCRIPTION, keypoint="index")
        ] = None
        position_columns: Annotated[
            tuple[str, str] | None, Declared(_POSITION_COLUMNS_DESCRIPTION)
        ] = None

        @model_validator(mode="after")
        def _check_position_columns(self) -> Self:
            """Refuse a named pair that another position setting contradicts."""
            if self.position_columns is None:
                return self
            x_name, y_name = self.position_columns
            if not x_name or not y_name:
                raise ValueError(
                    f"{_NAME}: position_columns names an empty column. Name the "
                    "x column and then the y column."
                )
            if x_name == y_name:
                raise ValueError(
                    f"{_NAME}: position_columns names {x_name!r} twice. Name the "
                    "x column and then a different y column."
                )
            if self.pose_head_index is not None:
                raise ValueError(
                    f"{_NAME}: position_columns names the position columns, which "
                    "pose_head_index would also choose. Set one of them."
                )
            if not self.use_pixel_coords:
                raise ValueError(
                    f"{_NAME}: position_columns names the position columns, which "
                    "use_pixel_coords=False would set to X/Y. Leave use_pixel_coords "
                    "unset."
                )
            return self

        @override
        def identity_dump(self) -> dict[str, object]:
            """The hashed params, with ``position_columns`` left out while it is unset.

            A run that names no columns has the identity it had before the field
            existed.
            """
            return self.drop_at_default(super().identity_dump(), ("position_columns",))

    def __init__(
        self,
        inputs: PairInteractionFilter.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

    # --- State (stateless feature) ---

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

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        order_col = resolve_order_col(df)
        ensure_columns(df, [C.id_col, C.orientation_col])

        p = self.params
        shift_dist = p.shift_dist * p.px_scale
        max_dist = p.max_dist * p.px_scale

        # Determine which columns to use for position
        if p.position_columns is not None:
            x_col, y_col = p.position_columns
        elif p.use_pixel_coords and p.pose_head_index is not None:
            x_col = f"{POSE_X.prefix}{p.pose_head_index}"
            y_col = f"{POSE_Y.prefix}{p.pose_head_index}"
            ensure_columns(df, [x_col, y_col])
        elif p.use_pixel_coords:
            # Pixel columns where the table has them, X/Y where it does not.
            x_col = "x_pixels" if "x_pixels" in df.columns else C.x_col
            y_col = "y_pixels" if "y_pixels" in df.columns else C.y_col
        else:
            x_col = C.x_col
            y_col = C.y_col

        ensure_columns(df, [x_col, y_col])

        # Sort by frame
        df = df.sort_values([C.id_col, order_col]).reset_index(drop=True)

        all_segments: list[pd.DataFrame] = []

        for _, gseq in (
            df.groupby(C.seq_col) if C.seq_col in df.columns else [(None, df)]
        ):
            ids = sorted(gseq[C.id_col].unique())
            if len(ids) < 2:
                continue

            for id_a, id_b in individual_pairs(ids, id_arenas(df)):
                segments = self._detect_pair_interactions(
                    gseq,
                    id_a,
                    id_b,
                    order_col,
                    x_col,
                    y_col,
                    shift_dist,
                    max_dist,
                )
                if segments is not None and not segments.empty:
                    all_segments.append(segments)

        if not all_segments:
            return pd.DataFrame(
                columns=[
                    C.frame_col,
                    "id1",
                    "id2",
                    "perspective",
                    "interaction_id",
                    "interaction_start",
                    "interaction_end",
                    C.group_col,
                    C.seq_col,
                ]
            )

        out = pd.concat(all_segments, ignore_index=True)
        # Attach group/sequence metadata if present
        for col in (C.group_col, C.seq_col):
            if col in df.columns:
                out[col] = df[col].iloc[0]
        return with_frame_times(out, df, resolve_order_col(df))

    def _detect_pair_interactions(
        self,
        gseq: pd.DataFrame,
        id_a,
        id_b,
        order_col: str,
        x_col: str,
        y_col: str,
        shift_dist: float,
        max_dist: float,
    ) -> pd.DataFrame | None:
        """Detect interaction segments for a single pair."""
        p = self.params

        df_a = gseq[gseq[C.id_col] == id_a][
            [order_col, x_col, y_col, C.orientation_col]
        ]
        df_b = gseq[gseq[C.id_col] == id_b][
            [order_col, x_col, y_col, C.orientation_col]
        ]

        if df_a.empty or df_b.empty:
            return None

        # Inner join on frame to align the two trajectories
        merged = df_a.merge(
            df_b,
            on=order_col,
            suffixes=("_a", "_b"),
            how="inner",
        )
        if merged.empty:
            return None

        frames = merged[order_col].to_numpy()
        x_a = merged[f"{x_col}_a"].to_numpy(dtype=float)
        y_a = merged[f"{y_col}_a"].to_numpy(dtype=float)
        orient_a = merged[f"{C.orientation_col}_a"].to_numpy(dtype=float)
        x_b = merged[f"{x_col}_b"].to_numpy(dtype=float)
        y_b = merged[f"{y_col}_b"].to_numpy(dtype=float)
        orient_b = merged[f"{C.orientation_col}_b"].to_numpy(dtype=float)

        # Per-frame interaction condition
        meets_cond = _check_interaction_conditions(
            x_a,
            y_a,
            orient_a,
            x_b,
            y_b,
            orient_b,
            shift_dist=shift_dist,
            max_dist=max_dist,
            max_inv_orientation_diff_deg=p.max_inv_orientation_diff_deg,
            require_facing=p.require_facing,
        )

        # Morphological filtering
        if p.morphological_structure_size > 0:
            meets_cond = _remove_bool_islands(
                meets_cond, p.morphological_structure_size
            )

        # Find long runs
        runs = _find_long_true_runs(
            meets_cond,
            min_run=p.min_run_frames,
            n_frame_padding=p.frame_padding,
        )

        if not runs:
            return None

        # Two rows per frame per interaction segment: the pair in both orderings.
        # `interaction_id` counts segments within the pair, so the mirror carries
        # the same one -- it is the same interaction seen the other way round.
        rows = []
        for seg_id, (start_idx, end_idx) in enumerate(runs):
            seg_frames = frames[start_idx:end_idx]
            start_frame = int(frames[start_idx])
            end_frame = int(frames[min(end_idx - 1, len(frames) - 1)])
            for f in seg_frames:
                for perspective, (focal, other) in enumerate(
                    ((id_a, id_b), (id_b, id_a))
                ):
                    rows.append(
                        {
                            C.frame_col: int(f),
                            "id1": focal,
                            "id2": other,
                            "perspective": perspective,
                            "interaction_id": seg_id,
                            "interaction_start": start_frame,
                            "interaction_end": end_frame,
                        }
                    )

        return pd.DataFrame(rows)
