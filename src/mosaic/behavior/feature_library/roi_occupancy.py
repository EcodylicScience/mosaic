"""RoiOccupancy -- which regions of interest each row's position lies in.

A region is a shape in source pixels, or a shape placed relative to the boundary
of the row's arena (:class:`~mosaic.core.regions.Roi`). It is defined inline in
the params, or stored on the arenas of the entry's layout and named. Each row
gets, per region, whether its position lies inside and its signed distance to
the region's edge, and ``roi`` names the first region it lies in.

Positions and distances are source pixels, the unit of every layout. A table in
centimeters is refused: read the pixel tracks, then scale the ``dist_*`` columns
with ``scale-to-cm`` and ``columns``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Literal, Self, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.arenas import Arena, ArenaLayout, region_boundary
from mosaic.core.params import Declared, Params, has_fields
from mosaic.core.pipeline.arena_index import (
    ArenaLookup,
    cached_arena_lookup,
    read_arena_layout,
)
from mosaic.core.pipeline.column_contract import (
    META_KEPT,
    Column,
    Conditional,
    ParamColumn,
    Prefix,
    Provides,
    Requirement,
    kept,
)
from mosaic.core.pipeline.storage_names import feature_slug_of
from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    InputStream,
    Result,
    TrackInputs,
)
from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.region_geometry import (
    Boundary,
    BoundaryEstimate,
    EstimateShape,
    boundary_center,
    estimate_boundary,
    inside,
    radius_norm,
    resolve_roi,
    signed_distance,
)
from mosaic.core.regions import (
    NO_REGION_NAME,
    CircleShape,
    RectShape,
    Roi,
    RoiName,
    Shape,
)
from mosaic.core.schema import ARENA_COLUMN, TrackSpace

from .helpers import ensure_columns, entry_identity
from .registry import register_feature

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

type FloatArray = npt.NDArray[np.float64]

BoundaryMode = Literal["arena", "estimate"]
"""Where an arena's boundary comes from: its layout, or a fit to the positions."""

CENTER_DIST: Final = "arena_center_dist"
RADIUS_NORM: Final = "arena_radius_norm"
ROI_COLUMN: Final = "roi"
FIT_PREFIX: Final = "boundary_"
FIT_ROWS: Final = "boundary_rows"
CIRCLE_FIT: Final = ("boundary_cx", "boundary_cy", "boundary_r")
RECT_FIT: Final = ("boundary_x0", "boundary_y0", "boundary_x1", "boundary_y1")

_SCALING_FEATURES: Final = frozenset({"scale-to-cm", "scale-to-bl"})
"""Producers whose output changes the unit of the tracks it was made from."""

_ROIS_DESCRIPTION = (
    "The regions of interest, at least one. A name is a region stored on the row's "
    "arena; an object is a region defined here. Each name is unique, and names "
    "the in_<name> and dist_<name> columns."
)

_PRIORITY_DESCRIPTION = (
    "The order in which roi names a row's region when it lies in several. Lists "
    "exactly the names of rois. Unset, the order of rois."
)

_BOUNDARY_DESCRIPTION = (
    "Where an arena's boundary comes from: its layout's boundary, else its "
    "region (arena), or a fit to the positions (estimate)."
)

_ESTIMATE_SHAPE_DESCRIPTION = (
    "The shape that boundary estimate fits. Left at circle unless boundary is estimate."
)

_ESTIMATE_QUANTILE_DESCRIPTION = (
    "The share of positions that the estimate trims at each end of each axis, "
    "from 0 up to but excluding 0.5. Left at 0 unless boundary is estimate."
)

_CENTER_DISTANCE_DESCRIPTION = (
    "Write arena_center_dist, the distance in pixels from the boundary's center, "
    "and arena_radius_norm, that distance as a fraction of the wall's along the "
    "same ray: 0 at the center, 1 on the wall. On a concave polygon a position "
    "past a notch reads above 1; use a circle or rect boundary, or regions, where "
    "that matters."
)

_X_COL_DESCRIPTION = "The column of each row's x position, such as a keypoint's."
_Y_COL_DESCRIPTION = "The column of each row's y position, such as a keypoint's."


def _name_of(roi: RoiName | Roi) -> str:
    return roi if isinstance(roi, str) else roi.name


@final
@register_feature
class RoiOccupancy:
    """Per-frame membership of each region of interest, and the distance to it.

    Outputs one row per input row: the metadata columns, ``arena`` when the
    input has it, then ``in_<name>`` (nullable boolean) and ``dist_<name>``
    (pixels, negative inside) for each region in ``rois`` order, and ``roi``,
    the first region in ``priority`` order that the row lies in, or ``none``.
    Regions may overlap, so several ``in_`` columns may be true. A row whose
    position is missing has null ``in_`` values, NaN distances and a null
    ``roi``.

    With an ``arena`` column each row reads its own arena's boundary and stored
    regions, and a row of no arena reads as a missing position. Without one the
    entry's layout must name one arena when the params read arena data.

    With ``boundary="estimate"`` each arena's boundary is fit to the positions of
    its rows instead, and the fit is written as columns constant within the
    arena: ``boundary_cx``, ``boundary_cy`` and ``boundary_r`` for a circle,
    ``boundary_x0``, ``boundary_y0``, ``boundary_x1`` and ``boundary_y1`` for a
    rect, and ``boundary_rows``, the positions the fit read. An arena with no
    position fails the entry. A row of no arena has null fit columns. The fit
    pools every frame and id of the arena, so a sparse trajectory underestimates
    it, and the recorded size shows it.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.roi_occupancy.RoiOccupancy.Params`.
    """

    category = "per-frame"
    name = "roi-occupancy"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    relates: RelatesLevel = "none"
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ("arenas",)
    emits: EmitsLevel = "individual"
    follows_ids: bool = False
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset({"image_px"})
    requires: ClassVar[tuple[Requirement, ...]] = (
        ParamColumn("x_col"),
        ParamColumn("y_col"),
        Column(C.group_col),
        Column(C.seq_col),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(Prefix("in_"), Prefix("dist_"), Column(ROI_COLUMN)),
        conditional=(
            *META_KEPT,
            *kept(ARENA_COLUMN),
            Conditional(Column(CENTER_DIST), when_param="center_distance"),
            Conditional(Column(RADIUS_NORM), when_param="center_distance"),
            Conditional(Prefix(FIT_PREFIX)),
        ),
    )

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        rois: Annotated[list[RoiName | Roi], Declared(_ROIS_DESCRIPTION)] = Field(
            min_length=1
        )
        priority: Annotated[list[RoiName] | None, Declared(_PRIORITY_DESCRIPTION)] = (
            None
        )
        boundary: Annotated[BoundaryMode, Declared(_BOUNDARY_DESCRIPTION)] = "arena"
        estimate_shape: Annotated[
            EstimateShape, Declared(_ESTIMATE_SHAPE_DESCRIPTION)
        ] = "circle"
        estimate_quantile: Annotated[
            float, Declared(_ESTIMATE_QUANTILE_DESCRIPTION)
        ] = Field(default=0.0, ge=0.0, lt=0.5)
        center_distance: Annotated[bool, Declared(_CENTER_DISTANCE_DESCRIPTION)] = False
        x_col: Annotated[str, Declared(_X_COL_DESCRIPTION)] = C.x_col
        y_col: Annotated[str, Declared(_Y_COL_DESCRIPTION)] = C.y_col

        @model_validator(mode="after")
        def _unique_names(self) -> Self:
            """Refuse a name that two regions share."""
            if not has_fields(self, "rois"):
                return self
            names = [_name_of(roi) for roi in self.rois]
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                message = (
                    f"each region of rois needs its own name, and {repeated} "
                    "name several"
                )
                raise ValueError(message)
            return self

        @model_validator(mode="after")
        def _priority_lists_the_regions(self) -> Self:
            """Refuse a priority that is not an order of the regions' names."""
            if not has_fields(self, "rois", "priority") or self.priority is None:
                return self
            names = [_name_of(roi) for roi in self.rois]
            if sorted(self.priority) != sorted(names):
                message = (
                    f"priority lists exactly the names of rois, {names}, in some "
                    f"order, and got {self.priority}"
                )
                raise ValueError(message)
            return self

        @model_validator(mode="after")
        def _estimate_settings_need_the_estimate(self) -> Self:
            """Refuse an estimate setting that the boundary mode does not read."""
            if not has_fields(self, "boundary", "estimate_shape", "estimate_quantile"):
                return self
            if self.boundary == "estimate":
                return self
            if self.estimate_shape != "circle" or self.estimate_quantile != 0.0:
                message = (
                    "estimate_shape and estimate_quantile apply only with "
                    f"boundary='estimate', and boundary is {self.boundary!r}"
                )
                raise ValueError(message)
            return self

    def __init__(
        self,
        inputs: RoiOccupancy.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        _refuse_scaled_inputs(self.inputs.feature_inputs)
        if not reads_arena_data(self.params):
            self.consumed_roots = ()
        self._ds: Dataset | None = None
        self._lookup: ArenaLookup | None = None

    def bind_dataset(self, ds: Dataset) -> None:
        """Take the dataset's arena lookup, read once per process."""
        self._ds = ds
        self._lookup = cached_arena_lookup(ds) if self.consumed_roots else None

    @staticmethod
    def temporal_reach(params: RoiOccupancy.Params) -> int | None:
        """Frames on each side that one row reads.

        A row reads its own position, so 0. An estimated boundary is fit to
        every frame of the arena, which no number of frames bounds.
        """
        return None if params.boundary == "estimate" else 0

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
        params = self.params
        ensure_columns(df, [params.x_col, params.y_col])
        x = df[params.x_col].to_numpy(dtype=np.float64)
        y = df[params.y_col].to_numpy(dtype=np.float64)
        columns = _Columns(len(df), params)
        if params.boundary == "estimate":
            self._fill_estimated(df, x, y, columns)
        elif not self.consumed_roots:
            shapes = {
                roi.name: roi.shape for roi in params.rois if isinstance(roi, Roi)
            }
            columns.fill(np.arange(len(df)), x, y, shapes, None)
        else:
            for arena, rows in self._arenas_of_rows(df):
                boundary = arena.boundary or region_boundary(arena.region)
                shapes = self._shapes(boundary, arena)
                columns.fill(rows, x[rows], y[rows], shapes, boundary)
        meta = [name for name in df.columns if name in C.meta_set() | {ARENA_COLUMN}]
        out = df[meta].reset_index(drop=True)
        return pd.concat([out, columns.frame()], axis=1)

    def _shapes(self, boundary: Boundary, arena: Arena | None) -> dict[str, Shape]:
        """Each region's absolute shape in the arena that *boundary* encloses.

        *arena* is ``None`` only when the params name no stored region, because a
        named region makes the feature read the entry's layout.
        """
        shapes: dict[str, Shape] = {}
        for named in self.params.rois:
            if isinstance(named, str):
                assert arena is not None, "a named region reads the entry's layout"
                roi = _stored_roi(arena, named)
            else:
                roi = named
            shapes[roi.name] = resolve_roi(roi, boundary)
        return shapes

    def _fill_estimated(
        self, df: pd.DataFrame, x: FloatArray, y: FloatArray, columns: _Columns
    ) -> None:
        """Fit each arena's boundary to its rows and fill the columns against it."""
        for label, arena, rows in self._estimate_groups(df):
            try:
                estimate = estimate_boundary(
                    x[rows],
                    y[rows],
                    shape=self.params.estimate_shape,
                    quantile=self.params.estimate_quantile,
                )
            except ValueError as error:
                group, sequence = entry_identity(df)
                where = "its rows" if label is None else f"the arena {label!r}"
                message = (
                    f"{self.name}: cannot fit a boundary to {where} of "
                    f"({group!r}, {sequence!r}): {error}"
                )
                raise ValueError(message) from error
            boundary = estimate.boundary
            shapes = self._shapes(boundary, arena)
            columns.fill(rows, x[rows], y[rows], shapes, boundary)
            columns.record_fit(rows, estimate)

    def _estimate_groups(
        self, df: pd.DataFrame
    ) -> list[tuple[str | None, Arena | None, npt.NDArray[np.intp]]]:
        """Each arena the estimate fits: its name, its stored arena and its rows.

        The layout is read only when the params name a stored region.
        """
        if self.consumed_roots:
            return [
                (arena.name, arena, rows) for arena, rows in self._arenas_of_rows(df)
            ]
        if ARENA_COLUMN in df.columns:
            return [(label, None, rows) for label, rows in _rows_by_arena(df)]
        return [(None, None, np.arange(len(df)))]

    def _arenas_of_rows(
        self, df: pd.DataFrame
    ) -> list[tuple[Arena, npt.NDArray[np.intp]]]:
        """Each arena of the entry's layout with the positions of its rows."""
        layout = self._layout(df)
        if ARENA_COLUMN not in df.columns:
            if len(layout.arenas) != 1:
                group, sequence = entry_identity(df)
                message = (
                    f"{self.name}: the layout of ({group!r}, {sequence!r}) names "
                    f"{len(layout.arenas)} arenas and its rows name none. Run "
                    "assign-arenas first, and read its output."
                )
                raise ValueError(message)
            return [(layout.arenas[0], np.arange(len(df)))]
        by_name = {arena.name: arena for arena in layout.arenas}
        found: list[tuple[Arena, npt.NDArray[np.intp]]] = []
        for label, rows in _rows_by_arena(df):
            arena = by_name.get(label)
            if arena is None:
                group, sequence = entry_identity(df)
                message = (
                    f"{self.name}: rows of ({group!r}, {sequence!r}) name the "
                    f"arena {label!r}, which its layout lacks. It names "
                    f"{sorted(by_name)}."
                )
                raise ValueError(message)
            found.append((arena, rows))
        return found

    def _layout(self, df: pd.DataFrame) -> ArenaLayout:
        """The layout of the entry *df* covers."""
        if self._ds is None or self._lookup is None:
            message = (
                f"{self.name} reads arena layouts and was not bound to a dataset. "
                "Run it through run_feature."
            )
            raise ValueError(message)
        entry = entry_identity(df)
        row = self._lookup.row(self._ds, entry)
        if row is None:
            message = (
                f"{self.name}: {entry!r} has no arena layout. Set one with "
                "`mosaic arenas set`."
            )
            raise ValueError(message)
        return read_arena_layout(self._ds, row)


def reads_arena_data(params: RoiOccupancy.Params) -> bool:
    """Whether *params* read the arenas of the entry's layout."""
    if any(isinstance(roi, str) for roi in params.rois):
        return True
    if params.boundary != "arena":
        return False
    return params.center_distance or any(
        isinstance(roi, Roi) and roi.placement == "arena" for roi in params.rois
    )


def _refuse_scaled_inputs(inputs: tuple[Result, ...]) -> None:
    """Refuse an input whose producer scales the pixel tracks into other units."""
    for item in inputs:
        slug = feature_slug_of(item.feature)
        if slug in _SCALING_FEATURES:
            message = (
                f"roi-occupancy computes in the pixels of the arena layouts, and "
                f"its input {item.feature!r} comes from {slug}, which changes the "
                "unit. Read the pixel tracks, then scale the dist_* columns with "
                "scale-to-cm and columns=[...]."
            )
            raise ValueError(message)


def _stored_roi(arena: Arena, name: str) -> Roi:
    """The region *name* stored on *arena*."""
    for roi in arena.rois:
        if roi.name == name:
            return roi
    message = (
        f"roi-occupancy: the arena {arena.name!r} stores no region {name!r}. It "
        f"stores {sorted(roi.name for roi in arena.rois)}."
    )
    raise ValueError(message)


def _rows_by_arena(df: pd.DataFrame) -> list[tuple[str, npt.NDArray[np.intp]]]:
    """The positions of each arena's rows, in name order. A row of no arena is left out."""
    codes, labels = pd.factorize(df[ARENA_COLUMN].astype("string"), sort=True)
    return [
        (str(label), np.flatnonzero(codes == code)) for code, label in enumerate(labels)
    ]


class _Columns:
    """The output columns of one entry, filled one arena's rows at a time.

    A row no :meth:`fill` reaches keeps NaN distances, as a missing position.
    """

    def __init__(self, length: int, params: RoiOccupancy.Params) -> None:
        names = [_name_of(roi) for roi in params.rois]
        self._names: list[str] = names
        self._order: list[str] = (
            list(params.priority) if params.priority is not None else names
        )
        self._center: bool = params.center_distance
        self._dist: dict[str, FloatArray] = {
            name: np.full(length, np.nan) for name in names
        }
        self._center_dist: FloatArray = np.full(length, np.nan)
        self._radius_norm: FloatArray = np.full(length, np.nan)
        self._fit: dict[str, FloatArray] = (
            {
                name: np.full(length, np.nan)
                for name in (
                    CIRCLE_FIT if params.estimate_shape == "circle" else RECT_FIT
                )
            }
            if params.boundary == "estimate"
            else {}
        )
        self._fit_rows: FloatArray = np.full(length if self._fit else 0, np.nan)

    def fill(
        self,
        rows: npt.NDArray[np.intp],
        x: FloatArray,
        y: FloatArray,
        shapes: Mapping[str, Shape],
        boundary: Boundary | None,
    ) -> None:
        """Write the distances of the positions *x*, *y* at *rows*."""
        for name, shape in shapes.items():
            self._dist[name][rows] = signed_distance(shape, x, y)
        if self._center and boundary is not None:
            cx, cy = boundary_center(boundary)
            self._center_dist[rows] = np.hypot(x - cx, y - cy)
            self._radius_norm[rows] = radius_norm(boundary, x, y)

    def record_fit(
        self, rows: npt.NDArray[np.intp], estimate: BoundaryEstimate
    ) -> None:
        """Write the fitted boundary and the positions it read at *rows*."""
        boundary = estimate.boundary
        match boundary:
            case CircleShape():
                values = (boundary.cx, boundary.cy, boundary.r)
            case RectShape():
                values = (boundary.x0, boundary.y0, boundary.x1, boundary.y1)
        for name, value in zip(self._fit, values, strict=True):
            self._fit[name][rows] = value
        self._fit_rows[rows] = estimate.rows

    def frame(self) -> pd.DataFrame:
        """The columns: ``in_`` and ``dist_`` in region order, ``roi``, the center."""
        located = ~np.isnan(self._dist[self._names[0]])
        member = {name: inside(self._dist[name]) for name in self._names}
        roi = np.full(len(located), NO_REGION_NAME, dtype=object)
        for name in reversed(self._order):
            roi[member[name]] = name
        roi[~located] = None
        columns: dict[str, object] = {}
        for name in self._names:
            flags = pd.array(member[name], dtype="boolean")
            flags[np.isnan(self._dist[name])] = pd.NA
            columns[f"in_{name}"] = flags
        for name in self._names:
            columns[f"dist_{name}"] = self._dist[name]
        columns[ROI_COLUMN] = pd.array(roi, dtype="string")
        if self._center:
            columns[CENTER_DIST] = self._center_dist
            columns[RADIUS_NORM] = self._radius_norm
        for name, values in self._fit.items():
            columns[name] = values
        if self._fit:
            fitted = ~np.isnan(self._fit_rows)
            counts = pd.array(
                np.where(fitted, self._fit_rows, 0).astype(np.int64), dtype="Int64"
            )
            counts[~fitted] = pd.NA
            columns[FIT_ROWS] = counts
        return pd.DataFrame(columns)
