"""Tests for the RoiOccupancy feature: each row's regions of interest, in pixels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.roi_occupancy import RoiOccupancy
from mosaic.behavior.feature_library.scale_to_cm import ScaleToCm
from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.arenas import Arena, ArenaLayout, Rect
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.run import AllEntriesFailed, run_feature
from mosaic.core.pipeline.track_space import TrackSpaceRefused
from mosaic.core.pipeline.types import Result
from mosaic.core.regions import CircleShape, PolygonShape, RectShape, RingShape, Roi
from tests.helpers import (
    Counted,
    MediaClip,
    add_tracks_table,
    entry_error_lines,
    make_dataset,
    refuse_index,
    write_media_index,
)

VARIANT = "convert-trex.0.2-1111111111"
"""The one tracks variant of every dataset here, in ``mosaic_v1``."""

type Position = tuple[float, float]


def _table(
    entry: Entry,
    positions: Sequence[Position],
    *,
    ids: Sequence[int] | None = None,
    arenas: Sequence[str | None] | None = None,
) -> pd.DataFrame:
    """One entry's ``mosaic_v1`` rows, one frame per position."""
    count = len(positions)
    table = pd.DataFrame(
        {
            "frame": np.arange(count, dtype=np.int64),
            "time": np.arange(count, dtype=float) / 30.0,
            "id": np.zeros(count, dtype=np.int64) if ids is None else list(ids),
            "group": entry[0],
            "sequence": entry[1],
            "X": [x for x, _ in positions],
            "Y": [y for _, y in positions],
        }
    )
    if arenas is not None:
        table["arena"] = list(arenas)
    return table


def _roi_dataset(
    tmp_path: Path,
    layouts: Mapping[Entry, ArenaLayout],
    tables: Sequence[pd.DataFrame],
) -> Dataset:
    """A dataset of *tables* in one ``mosaic_v1`` variant, with *layouts* set."""
    ds = make_dataset(tmp_path / "ds")
    for table in tables:
        add_tracks_table(ds, VARIANT, table, std_format="mosaic_v1")
    for (group, sequence), layout in layouts.items():
        _ = ds.set_arena_layout(layout, group=group, sequence=sequence)
    return ds


def _run(
    ds: Dataset,
    params: dict[str, object],
    *,
    parallel_workers: int | None = None,
    parallel_mode: str = "thread",
    execution_id: str | None = None,
) -> tuple[Result, dict[str, pd.DataFrame]]:
    """Run the feature over *ds*, returning its result and each entry's output."""
    result = run_feature(
        ds,
        RoiOccupancy(params=params),
        parallel_workers=parallel_workers,
        parallel_mode=parallel_mode,
        execution_id=execution_id,
    )
    assert result.run_id is not None
    root = ds.get_root("features") / result.feature / result.run_id
    return result, {path.stem: pd.read_parquet(path) for path in root.glob("*.parquet")}


def _circle_arena(
    name: str, cx: float, cy: float, r: float, rois: tuple[Roi, ...] = ()
) -> Arena:
    return Arena(
        name=name,
        region=Rect(x0=0, y0=0, x1=400, y1=300),
        boundary=CircleShape(cx=cx, cy=cy, r=r),
        rois=rois,
    )


def _single(arena: Arena) -> ArenaLayout:
    return ArenaLayout(width=400, height=300, arenas=(arena,))


CENTER_DISC = Roi(
    name="center",
    shape=CircleShape(cx=0.0, cy=0.0, r=0.5),
    placement="arena",
)
"""A disc of half the arena's radius at its center."""


# --- overlapping regions --------------------------------------------------------

DISC = Roi(name="disc", shape=CircleShape(cx=50.0, cy=50.0, r=10.0))
RING = Roi(name="ring", shape=RingShape(cx=50.0, cy=50.0, r_inner=10.0, r_outer=20.0))
BOX = Roi(name="box", shape=RectShape(x0=45.0, y0=40.0, x1=75.0, y1=60.0))
ELL = Roi(
    name="ell",
    shape=PolygonShape(
        vertices=(
            (100.0, 100.0),
            (140.0, 100.0),
            (140.0, 110.0),
            (110.0, 110.0),
            (110.0, 140.0),
            (100.0, 140.0),
        )
    ),
)
"""An L whose notch, ``110 < x < 140`` and ``110 < y < 140``, lies outside it."""

VISITS: tuple[tuple[Position, int], ...] = (
    ((50.0, 50.0), 3),
    ((65.0, 50.0), 2),
    ((50.0, 75.0), 4),
    ((105.0, 130.0), 2),
    ((125.0, 125.0), 1),
    ((60.0, 50.0), 1),
    ((math.nan, 50.0), 1),
)
"""Each position the one id visits and its frame count, in visit order."""


def _overlapping(priority: list[str] | None = None) -> pd.DataFrame:
    entry: Entry = ("g", "s")
    positions = [position for position, count in VISITS for _ in range(count)]
    feature = RoiOccupancy(
        params={
            "rois": [roi.model_dump(mode="json") for roi in (DISC, RING, BOX, ELL)],
            "priority": priority,
        }
    )
    return feature.apply(_table(entry, positions))


class TestOverlappingRegions:
    def test_each_region_counts_its_frames(self) -> None:
        out = _overlapping()

        counts = {
            name: int(out[f"in_{name}"].sum())
            for name in ("disc", "ring", "box", "ell")
        }
        nulls = {
            name: int(out[f"in_{name}"].isna().sum())
            for name in ("disc", "ring", "box", "ell")
        }

        assert counts == {"disc": 4, "ring": 3, "box": 6, "ell": 2}
        assert nulls == {"disc": 1, "ring": 1, "box": 1, "ell": 1}

    def test_roi_names_the_first_region_in_rois_order(self) -> None:
        out = _overlapping()

        assert out["roi"].fillna("<null>").value_counts().to_dict() == {
            "none": 5,
            "disc": 4,
            "ring": 2,
            "ell": 2,
            "<null>": 1,
        }

    def test_roi_follows_the_priority_order(self) -> None:
        out = _overlapping(priority=["box", "ell", "ring", "disc"])

        assert out["roi"].fillna("<null>").value_counts().to_dict() == {
            "box": 6,
            "none": 5,
            "ell": 2,
            "<null>": 1,
        }

    def test_distances_match_hand_computed_values(self) -> None:
        out = _overlapping().set_index("frame")

        # (50, 50): frame 0. (60, 50): frame 12. (50, 75): frame 5.
        # (105, 130): frame 9, in the L's bar. (125, 125): frame 11, in its notch.
        assert out.loc[0, "dist_disc"] == pytest.approx(-10.0)
        assert out.loc[0, "dist_ring"] == pytest.approx(10.0)
        assert out.loc[0, "dist_box"] == pytest.approx(-5.0)
        assert out.loc[12, "dist_disc"] == pytest.approx(0.0)
        assert bool(out.loc[12, "in_disc"])
        assert out.loc[5, "dist_box"] == pytest.approx(15.0)
        assert out.loc[9, "dist_ell"] == pytest.approx(-5.0)
        assert out.loc[11, "dist_ell"] == pytest.approx(15.0)

    def test_a_missing_position_reads_null(self) -> None:
        out = _overlapping().set_index("frame")

        missing = out[out.index == 13]
        names = ("disc", "ring", "box", "ell")
        assert missing[[f"in_{name}" for name in names]].isna().all(axis=None)
        assert missing[[f"dist_{name}" for name in names]].isna().all(axis=None)
        assert missing["roi"].isna().all()

    def test_columns_follow_rois_order(self) -> None:
        out = _overlapping()

        payload = [name for name in out.columns if name.startswith(("in_", "dist_"))]
        assert payload == [
            "in_disc",
            "in_ring",
            "in_box",
            "in_ell",
            "dist_disc",
            "dist_ring",
            "dist_box",
            "dist_ell",
        ]
        assert out["in_disc"].dtype == "boolean"
        assert out["dist_disc"].dtype == np.float64


# --- arena-relative regions -----------------------------------------------------

RELATIVE: tuple[Position, ...] = (
    (0.0, 0.0),
    (0.3, 0.0),
    (0.6, 0.0),
    (0.0, -0.9),
    (1.2, 0.0),
)
"""Positions in units of the arena's radius from its center."""


def _relative_dataset(tmp_path: Path, *, stored: bool) -> Dataset:
    rois = (CENTER_DISC,) if stored else ()
    arenas = {
        ("", "seq_a"): (100.0, 100.0, 50.0),
        ("", "seq_b"): (250.0, 150.0, 100.0),
    }
    layouts = {
        entry: _single(_circle_arena("tank", cx, cy, r, rois))
        for entry, (cx, cy, r) in arenas.items()
    }
    tables = [
        _table(entry, [(cx + r * u, cy + r * v) for u, v in RELATIVE])
        for entry, (cx, cy, r) in arenas.items()
    ]
    return _roi_dataset(tmp_path, layouts, tables)


class TestArenaRelativeRegions:
    @pytest.mark.parametrize("stored", [False, True], ids=["inline", "stored"])
    def test_two_arenas_read_one_relative_disc_alike(
        self, tmp_path: Path, stored: bool
    ) -> None:
        ds = _relative_dataset(tmp_path, stored=stored)
        rois: list[object] = (
            ["center"] if stored else [CENTER_DISC.model_dump(mode="json")]
        )

        _, outputs = _run(ds, {"rois": rois, "center_distance": True})

        a, b = outputs["seq_a"], outputs["seq_b"]
        assert a["in_center"].tolist() == [True, True, False, False, False]
        assert b["in_center"].tolist() == a["in_center"].tolist()
        assert a["arena_radius_norm"].to_numpy() == pytest.approx(
            [0.0, 0.3, 0.6, 0.9, 1.2]
        )
        assert b["arena_radius_norm"].to_numpy() == pytest.approx(
            a["arena_radius_norm"].to_numpy()
        )


# --- a polygon boundary ---------------------------------------------------------

NOTCHED = PolygonShape(
    vertices=(
        (0.0, 0.0),
        (120.0, 0.0),
        (120.0, 40.0),
        (80.0, 40.0),
        (80.0, 80.0),
        (120.0, 80.0),
        (120.0, 120.0),
        (0.0, 120.0),
    )
)
"""A square with a notch cut into its right side. Its centroid is ``(55, 60)``."""


def test_a_relative_disc_is_centered_where_the_center_distance_is_zero(
    tmp_path: Path,
) -> None:
    arena = Arena(
        name="tank", region=Rect(x0=0, y0=0, x1=130, y1=130), boundary=NOTCHED
    )
    entry: Entry = ("", "seq_a")
    ds = _roi_dataset(
        tmp_path,
        {entry: ArenaLayout(width=200, height=200, arenas=(arena,))},
        [_table(entry, [(55.0, 60.0), (115.0, 30.0)])],
    )
    middle = Roi(
        name="middle", shape=CircleShape(cx=0.0, cy=0.0, r=0.1), placement="arena"
    )

    _, outputs = _run(
        ds, {"rois": [middle.model_dump(mode="json")], "center_distance": True}
    )

    out = outputs["seq_a"]
    # The radius unit is half the box's shorter side, 60, so the disc's radius is 6.
    assert out["arena_center_dist"].iloc[0] == pytest.approx(0.0)
    assert out["dist_middle"].iloc[0] == pytest.approx(-6.0)
    # The ray from (55, 60) through (115, 30) first crosses the boundary at the
    # notch's wall, x = 80: 25 of the 60 pixels of x it covers.
    assert out["arena_radius_norm"].iloc[1] == pytest.approx(60.0 / 25.0)


# --- several arenas in one entry ------------------------------------------------


def _feeder(x0: float, y0: float) -> Roi:
    return Roi(name="feeder", shape=RectShape(x0=x0, y0=y0, x1=x0 + 20.0, y1=y0 + 20.0))


def _grid(*, right_feeder: bool = True) -> ArenaLayout:
    return ArenaLayout(
        width=200,
        height=100,
        arenas=(
            Arena(
                name="left",
                region=Rect(x0=0, y0=0, x1=100, y1=100),
                rois=(_feeder(10.0, 10.0),),
            ),
            Arena(
                name="right",
                region=Rect(x0=100, y0=0, x1=200, y1=100),
                rois=(_feeder(170.0, 70.0),) if right_feeder else (),
            ),
        ),
    )


GRID_POSITIONS: tuple[Position, ...] = (
    (20.0, 20.0),
    (180.0, 80.0),
    (180.0, 80.0),
    (20.0, 20.0),
)
GRID_IDS = (0, 0, 1, 1)
GRID_ARENAS = ("left", "left", "right", "right")


class TestSeveralArenas:
    def test_each_row_reads_its_own_arena_s_feeder(self, tmp_path: Path) -> None:
        entry: Entry = ("", "seq_a")
        ds = _roi_dataset(
            tmp_path,
            {entry: _grid()},
            [_table(entry, GRID_POSITIONS, ids=GRID_IDS, arenas=GRID_ARENAS)],
        )

        _, outputs = _run(ds, {"rois": ["feeder"]})

        out = outputs["seq_a"]
        assert out["in_feeder"].tolist() == [True, False, True, False]
        assert out["arena"].tolist() == list(GRID_ARENAS)

    def test_a_row_of_no_arena_reads_as_a_missing_position(
        self, tmp_path: Path
    ) -> None:
        entry: Entry = ("", "seq_a")
        arenas = ("left", None, "right", "right")
        ds = _roi_dataset(
            tmp_path,
            {entry: _grid()},
            [_table(entry, GRID_POSITIONS, ids=GRID_IDS, arenas=arenas)],
        )

        _, outputs = _run(ds, {"rois": ["feeder"]})

        out = outputs["seq_a"]
        assert out["in_feeder"].tolist() == [True, pd.NA, True, False]
        assert out["dist_feeder"].isna().tolist() == [False, True, False, False]
        assert out["roi"].isna().tolist() == [False, True, False, False]

    def test_a_missing_arena_or_region_fails_only_that_entry(
        self, tmp_path: Path
    ) -> None:
        good: Entry = ("", "seq_good")
        ghost: Entry = ("", "seq_ghost")
        bare: Entry = ("", "seq_bare")
        ds = _roi_dataset(
            tmp_path,
            {good: _grid(), ghost: _grid(), bare: _grid(right_feeder=False)},
            [
                _table(good, GRID_POSITIONS, ids=GRID_IDS, arenas=GRID_ARENAS),
                _table(
                    ghost,
                    GRID_POSITIONS,
                    ids=GRID_IDS,
                    arenas=("left", "left", "attic", "attic"),
                ),
                _table(bare, GRID_POSITIONS, ids=GRID_IDS, arenas=GRID_ARENAS),
            ],
        )

        result, outputs = _run(ds, {"rois": ["feeder"]}, execution_id="TESTROIFAIL")

        assert set(outputs) == {"seq_good"}
        assert sorted(result.failed_entries) == sorted(
            [make_entry_key(*ghost), make_entry_key(*bare)]
        )
        errors = entry_error_lines(ds, "TESTROIFAIL")
        assert any("attic" in line for line in errors)
        assert any("feeder" in line and "right" in line for line in errors)

    def test_an_entry_without_a_layout_fails_only_itself(self, tmp_path: Path) -> None:
        good: Entry = ("", "seq_good")
        nameless: Entry = ("", "seq_nameless")
        ds = _roi_dataset(
            tmp_path,
            {good: _grid()},
            [
                _table(good, GRID_POSITIONS, ids=GRID_IDS, arenas=GRID_ARENAS),
                _table(nameless, GRID_POSITIONS, ids=GRID_IDS, arenas=GRID_ARENAS),
            ],
        )

        result, outputs = _run(ds, {"rois": ["feeder"]}, execution_id="TESTROINOLAYOUT")

        assert set(outputs) == {"seq_good"}
        assert result.failed_entries == (make_entry_key(*nameless),)
        errors = entry_error_lines(ds, "TESTROINOLAYOUT")
        assert len(errors) == 1
        assert "mosaic arenas set" in errors[0]
        assert "seq_nameless" in errors[0]

    def test_a_layout_of_several_arenas_needs_an_arena_column(
        self, tmp_path: Path
    ) -> None:
        entry: Entry = ("", "seq_a")
        ds = _roi_dataset(
            tmp_path, {entry: _grid()}, [_table(entry, GRID_POSITIONS, ids=GRID_IDS)]
        )
        feature = RoiOccupancy(params={"rois": ["feeder"]})
        feature.bind_dataset(ds)

        with pytest.raises(ValueError, match="assign-arenas"):
            _ = feature.apply(_table(entry, GRID_POSITIONS, ids=GRID_IDS))


# --- params ---------------------------------------------------------------------


class TestParams:
    def test_a_name_repeated_across_rois_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="feeder"):
            _ = RoiOccupancy.Params.model_validate(
                {"rois": ["feeder", _feeder(0.0, 0.0).model_dump(mode="json")]}
            )

    @pytest.mark.parametrize(
        "priority",
        [["disc"], ["disc", "ring", "ring"], ["disc", "box"]],
        ids=["short", "repeated", "foreign"],
    )
    def test_a_priority_not_matching_rois_is_refused(self, priority: list[str]) -> None:
        with pytest.raises(ValidationError, match="priority"):
            _ = RoiOccupancy.Params.model_validate(
                {"rois": ["disc", "ring"], "priority": priority}
            )

    @pytest.mark.parametrize(
        "override",
        [{"estimate_shape": "rect"}, {"estimate_quantile": 0.1}],
        ids=["shape", "quantile"],
    )
    def test_an_estimate_setting_is_refused_under_the_arena_boundary(
        self, override: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError, match="estimate"):
            _ = RoiOccupancy.Params.model_validate({"rois": ["disc"], **override})

    def test_an_estimate_quantile_of_one_half_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _ = RoiOccupancy.Params.model_validate(
                {"rois": ["disc"], "boundary": "estimate", "estimate_quantile": 0.5}
            )

    def test_no_region_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            _ = RoiOccupancy.Params.model_validate({"rois": []})


# --- declarations ---------------------------------------------------------------


class TestDeclarations:
    def test_a_stored_region_reads_the_arenas_root(self) -> None:
        assert RoiOccupancy(params={"rois": ["feeder"]}).consumed_roots == ("arenas",)

    def test_a_relative_region_reads_the_arenas_root(self) -> None:
        feature = RoiOccupancy(params={"rois": [CENTER_DISC.model_dump(mode="json")]})
        assert feature.consumed_roots == ("arenas",)

    def test_the_center_distance_reads_the_arenas_root(self) -> None:
        feature = RoiOccupancy(
            params={"rois": [DISC.model_dump(mode="json")], "center_distance": True}
        )
        assert feature.consumed_roots == ("arenas",)

    def test_absolute_inline_regions_read_no_root(self) -> None:
        feature = RoiOccupancy(params={"rois": [DISC.model_dump(mode="json")]})
        assert feature.consumed_roots == ()


# --- units ----------------------------------------------------------------------


class TestUnits:
    @pytest.mark.parametrize(
        "producer",
        ["scale-to-cm__from__tracks", "scale-to-bl__from__tracks"],
        ids=["cm", "bl"],
    )
    def test_a_scaled_input_is_refused_naming_the_remedy(self, producer: str) -> None:
        inputs = RoiOccupancy.Inputs((Result(feature=producer),))

        with pytest.raises(ValueError, match="dist_") as refused:
            _ = RoiOccupancy(inputs=inputs, params={"rois": ["feeder"]})

        assert "pixel" in str(refused.value)

    def test_a_centimeter_tracks_table_is_refused(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path / "ds")
        add_tracks_table(
            ds,
            VARIANT,
            _table(("", "seq_a"), [(50.0, 50.0)]),
            std_format="mosaic_cm_v1",
        )

        with pytest.raises(TrackSpaceRefused) as refused:
            _ = _run(ds, {"rois": [DISC.model_dump(mode="json")]})

        assert refused.value.reason == "schema_family_mismatch"
        assert refused.value.detail["found"] == ["plane_cm"]

    def test_a_distance_scaled_by_scale_to_cm_is_the_pixel_value_times_the_scale(
        self, tmp_path: Path
    ) -> None:
        entry: Entry = ("", "seq_a")
        ds = _roi_dataset(
            tmp_path, {}, [_table(entry, [(50.0, 50.0), (65.0, 50.0), (90.0, 10.0)])]
        )
        pixels, outputs = _run(ds, {"rois": [DISC.model_dump(mode="json")]})
        scale = ScaleToCm(
            inputs=ScaleToCm.Inputs(
                (Result(feature=pixels.feature, run_id=pixels.run_id),)
            ),
            params={"cm_per_pixel": 0.25, "columns": ["dist_disc"]},
        )

        scaled = run_feature(ds, scale)

        assert scaled.run_id is not None
        root = ds.get_root("features") / scaled.feature / scaled.run_id
        out = pd.read_parquet(root / "seq_a.parquet")
        assert out["dist_disc_cm"].to_numpy() == pytest.approx(
            outputs["seq_a"]["dist_disc"].to_numpy() * 0.25
        )


# --- reading the layout ---------------------------------------------------------


FOUR = tuple(("", f"seq_{index}") for index in range(4))


def _four_arenas(tmp_path: Path) -> Dataset:
    layouts = {
        entry: _single(_circle_arena("tank", 100.0 + 20 * k, 100.0, 50.0))
        for k, entry in enumerate(FOUR)
    }
    tables = [
        _table(entry, [(100.0 + 20 * k + 10.0 * step, 100.0) for step in range(6)])
        for k, entry in enumerate(FOUR)
    ]
    return _roi_dataset(tmp_path, layouts, tables)


PARAMS_READING_ARENAS: dict[str, object] = {
    "rois": [CENTER_DISC.model_dump(mode="json")],
    "center_distance": True,
}


class TestLayoutReading:
    def test_a_thread_run_reads_each_index_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ds = _four_arenas(tmp_path)
        counted = Counted(monkeypatch)

        _, outputs = _run(
            ds, PARAMS_READING_ARENAS, parallel_workers=2, parallel_mode="thread"
        )

        assert len(outputs) == 4
        assert (counted.rows, counted.cameras) == (1, 1)

    def test_a_process_run_writes_what_a_thread_run_writes(
        self, tmp_path: Path
    ) -> None:
        threaded = _four_arenas(tmp_path / "threaded")
        processes = _four_arenas(tmp_path / "processes")

        _, by_thread = _run(
            threaded, PARAMS_READING_ARENAS, parallel_workers=2, parallel_mode="thread"
        )
        _, by_process = _run(
            processes,
            PARAMS_READING_ARENAS,
            parallel_workers=2,
            parallel_mode="process",
        )

        assert by_thread.keys() == by_process.keys()
        for name, table in by_thread.items():
            pd.testing.assert_frame_equal(table, by_process[name])


# --- the estimated boundary -----------------------------------------------------


def _filled_circle(cx: float, cy: float, r: float) -> list[Position]:
    """A rim of 360 positions, one per degree, and 16 inside it at 0.2 to 0.8 of r."""
    rim = [
        (cx + r * math.cos(math.radians(k)), cy + r * math.sin(math.radians(k)))
        for k in range(360)
    ]
    inner = [
        (cx + f * r * math.cos(math.radians(a)), cy + f * r * math.sin(math.radians(a)))
        for f in (0.2, 0.45, 0.55, 0.8)
        for a in (45, 135, 225, 315)
    ]
    return rim + inner


RIM_ROWS = 360
FILLED_ROWS = 376
"""The positions of ``_filled_circle``: the rim and sixteen inside it."""

ESTIMATED: dict[str, object] = {
    "rois": [CENTER_DISC.model_dump(mode="json")],
    "boundary": "estimate",
    "center_distance": True,
}


class TestEstimate:
    def test_a_trajectory_filling_a_circle_records_its_center_and_radius(
        self, tmp_path: Path
    ) -> None:
        entry: Entry = ("", "seq_a")
        positions = [*_filled_circle(200.0, 150.0, 100.0), (math.nan, math.nan)]
        ds = _roi_dataset(tmp_path, {}, [_table(entry, positions)])

        _, outputs = _run(ds, ESTIMATED)

        out = outputs["seq_a"]
        assert out["boundary_cx"].to_numpy() == pytest.approx(200.0, abs=0.5)
        assert out["boundary_cy"].to_numpy() == pytest.approx(150.0, abs=0.5)
        assert out["boundary_r"].to_numpy() == pytest.approx(100.0, abs=0.5)
        assert out["boundary_rows"].tolist() == [FILLED_ROWS] * len(positions)
        assert not [name for name in out.columns if name.startswith("boundary_x")]
        assert out["arena_radius_norm"].iloc[0] == pytest.approx(1.0, abs=0.01)

    def test_a_rect_records_its_box(self, tmp_path: Path) -> None:
        entry: Entry = ("", "seq_a")
        corners = [(100.0, 50.0), (300.0, 50.0), (300.0, 250.0), (100.0, 250.0)]
        ds = _roi_dataset(tmp_path, {}, [_table(entry, [*corners, (200.0, 150.0)])])

        _, outputs = _run(
            ds,
            {
                "rois": [CENTER_DISC.model_dump(mode="json")],
                "boundary": "estimate",
                "estimate_shape": "rect",
            },
        )

        out = outputs["seq_a"]
        assert out[["boundary_x0", "boundary_y0", "boundary_x1", "boundary_y1"]].iloc[
            0
        ].tolist() == [100.0, 50.0, 300.0, 250.0]
        assert out["boundary_rows"].iloc[0] == 5
        assert "boundary_cx" not in out.columns
        # The disc has half the arena's shorter half-side, 100, as its radius.
        assert out["in_center"].tolist() == [False] * 4 + [True]

    def test_the_arena_boundary_writes_no_fit_columns(self, tmp_path: Path) -> None:
        entry: Entry = ("", "seq_a")
        ds = _roi_dataset(
            tmp_path,
            {entry: _single(_circle_arena("tank", 200.0, 150.0, 100.0))},
            [_table(entry, _filled_circle(200.0, 150.0, 100.0))],
        )

        _, outputs = _run(ds, {**ESTIMATED, "boundary": "arena"})

        assert not [c for c in outputs["seq_a"].columns if c.startswith("boundary_")]

    def test_a_quantile_trims_an_outlier(self, tmp_path: Path) -> None:
        entry: Entry = ("", "seq_a")
        positions = [
            *_filled_circle(200.0, 150.0, 100.0)[:RIM_ROWS],
            (1000.0, 150.0),
        ]
        ds = _roi_dataset(tmp_path, {}, [_table(entry, positions)])

        _, untrimmed = _run(ds, ESTIMATED)
        _, trimmed = _run(ds, {**ESTIMATED, "estimate_quantile": 0.01})

        assert untrimmed["seq_a"]["boundary_r"].iloc[0] > 250.0
        assert trimmed["seq_a"]["boundary_r"].iloc[0] == pytest.approx(100.0, abs=0.5)
        assert trimmed["seq_a"]["boundary_cx"].iloc[0] == pytest.approx(200.0, abs=0.5)

    def test_a_relative_disc_over_the_estimate_matches_the_true_boundary(
        self, tmp_path: Path
    ) -> None:
        entry: Entry = ("", "seq_a")
        ds = _roi_dataset(
            tmp_path,
            {entry: _single(_circle_arena("tank", 200.0, 150.0, 100.0))},
            [_table(entry, _filled_circle(200.0, 150.0, 100.0))],
        )

        _, estimated = _run(ds, ESTIMATED)
        _, true = _run(ds, {**ESTIMATED, "boundary": "arena"})

        found, expected = estimated["seq_a"], true["seq_a"]
        assert found["in_center"].tolist() == expected["in_center"].tolist()
        assert int(found["in_center"].sum()) == 8
        assert found["arena_radius_norm"].to_numpy() == pytest.approx(
            expected["arena_radius_norm"].to_numpy(), abs=0.01
        )

    def test_a_stored_relative_region_resolves_against_the_estimate(
        self, tmp_path: Path
    ) -> None:
        entry: Entry = ("", "seq_a")
        positions = _filled_circle(200.0, 150.0, 100.0)
        # The layout's own boundary is far smaller and elsewhere.
        stored = _single(_circle_arena("tank", 50.0, 50.0, 10.0, rois=(CENTER_DISC,)))
        ds = _roi_dataset(tmp_path, {entry: stored}, [_table(entry, positions)])

        _, outputs = _run(ds, {"rois": ["center"], "boundary": "estimate"})

        expected = [math.hypot(x - 200.0, y - 150.0) <= 50.0 for x, y in positions]
        assert outputs["seq_a"]["in_center"].tolist() == expected

    def test_each_arena_of_an_entry_is_fit_to_its_own_rows(
        self, tmp_path: Path
    ) -> None:
        entry: Entry = ("", "seq_a")
        left = _filled_circle(100.0, 100.0, 50.0)
        right = _filled_circle(300.0, 150.0, 80.0)
        ds = _roi_dataset(
            tmp_path,
            {},
            [
                _table(
                    entry,
                    [*left, *right],
                    ids=[0] * len(left) + [1] * len(right),
                    arenas=["A"] * len(left) + ["B"] * len(right),
                )
            ],
        )

        _, outputs = _run(
            ds, {"rois": [DISC.model_dump(mode="json")], "boundary": "estimate"}
        )

        out = outputs["seq_a"]
        by_arena = out.groupby("arena")[["boundary_cx", "boundary_cy", "boundary_r"]]
        assert by_arena.nunique().to_numpy().max() == 1
        assert by_arena.first().to_numpy() == pytest.approx(
            np.array([[100.0, 100.0, 50.0], [300.0, 150.0, 80.0]]), abs=0.5
        )

    def test_an_arena_without_a_position_fails_its_entry(self, tmp_path: Path) -> None:
        good: Entry = ("", "seq_good")
        blank: Entry = ("", "seq_blank")
        half: Entry = ("", "seq_half")
        circle = _filled_circle(200.0, 150.0, 100.0)
        nowhere = [(math.nan, math.nan)] * 3
        ds = _roi_dataset(
            tmp_path,
            {},
            [
                _table(good, circle),
                _table(blank, nowhere),
                _table(
                    half,
                    [*circle, *nowhere],
                    arenas=["A"] * len(circle) + ["B"] * len(nowhere),
                ),
            ],
        )

        result, outputs = _run(
            ds,
            {"rois": [DISC.model_dump(mode="json")], "boundary": "estimate"},
            execution_id="TESTROIESTIMATEFAIL",
        )

        assert set(outputs) == {"seq_good"}
        assert sorted(result.failed_entries) == sorted(
            [make_entry_key(*blank), make_entry_key(*half)]
        )
        errors = entry_error_lines(ds, "TESTROIESTIMATEFAIL")
        assert len(errors) == 2
        assert all("no finite position" in line for line in errors)
        assert any("'B'" in line for line in errors)


# --- composition and drift ------------------------------------------------------


class _Applied:
    """The sequences ``RoiOccupancy.apply`` computed, in call order."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sequences: list[str] = []
        original = RoiOccupancy.apply

        def apply(feature: RoiOccupancy, df: pd.DataFrame) -> pd.DataFrame:
            if not df.empty:
                self.sequences.append(str(df["sequence"].iloc[0]))
            return original(feature, df)

        monkeypatch.setattr(RoiOccupancy, "apply", apply)

    def take(self) -> list[str]:
        taken, self.sequences = sorted(self.sequences), []
        return taken


@pytest.fixture
def applied(monkeypatch: pytest.MonkeyPatch) -> _Applied:
    return _Applied(monkeypatch)


class TestCompositionAndDrift:
    def test_a_run_serves_its_cache(self, tmp_path: Path, applied: _Applied) -> None:
        ds = _four_arenas(tmp_path)

        first, _ = _run(ds, PARAMS_READING_ARENAS)
        assert applied.take() == [f"seq_{k}" for k in range(4)]
        second, _ = _run(ds, PARAMS_READING_ARENAS)

        assert applied.take() == []
        assert first.run_id == second.run_id

    def test_a_layout_edit_recomputes_that_entry_under_the_same_run_id(
        self, tmp_path: Path, applied: _Applied
    ) -> None:
        ds = _four_arenas(tmp_path)
        first, before = _run(ds, PARAMS_READING_ARENAS)
        _ = applied.take()
        wider = _single(_circle_arena("tank", 120.0, 100.0, 100.0))
        _ = ds.set_arena_layout(wider, group="", sequence="seq_1")

        second, after = _run(ds, PARAMS_READING_ARENAS)

        assert applied.take() == ["seq_1"]
        assert first.run_id == second.run_id
        assert after["seq_1"]["arena_radius_norm"].to_numpy() == pytest.approx(
            before["seq_1"]["arena_radius_norm"].to_numpy() / 2.0
        )
        pd.testing.assert_frame_equal(after["seq_2"], before["seq_2"])

    def test_absolute_inline_regions_declare_no_root_and_serve_after_an_edit(
        self, tmp_path: Path, applied: _Applied
    ) -> None:
        ds = _four_arenas(tmp_path)
        params: dict[str, object] = {"rois": [DISC.model_dump(mode="json")]}
        assert RoiOccupancy(params=params).consumed_roots == ()
        _ = _run(ds, params)
        _ = applied.take()
        _ = ds.set_arena_layout(
            _single(_circle_arena("tank", 120.0, 100.0, 100.0)),
            group="",
            sequence="seq_1",
        )

        _ = _run(ds, params)

        assert applied.take() == []

    def test_a_refused_index_fails_roi_occupancy_and_no_other_feature(
        self, tmp_path: Path
    ) -> None:
        ds = _four_arenas(tmp_path)
        refuse_index(ds)

        assert run_feature(ds, SpeedAngvel()).failed_entries == ()
        with pytest.raises(AllEntriesFailed):
            _ = run_feature(
                ds,
                RoiOccupancy(params=PARAMS_READING_ARENAS),
                execution_id="TESTROIREFUSED",
            )

        errors = entry_error_lines(ds, "TESTROIREFUSED")
        assert len(errors) == len(FOUR)
        assert all("cannot be read" in line for line in errors)

    def test_an_entry_of_two_cameras_with_different_layouts_fails_only_itself(
        self, tmp_path: Path
    ) -> None:
        ds = _four_arenas(tmp_path)
        write_media_index(
            ds,
            [
                MediaClip(sequence="seq_3", filename="top.mp4", camera="top"),
                MediaClip(sequence="seq_3", filename="side.mp4", camera="side"),
            ],
        )
        for camera, radius in (("top", 50.0), ("side", 60.0)):
            _ = ds.set_arena_layout(
                _single(_circle_arena("tank", 160.0, 100.0, radius)),
                group="",
                sequence="seq_3",
                camera=camera,
            )

        assert run_feature(ds, SpeedAngvel()).failed_entries == ()
        result, outputs = _run(ds, PARAMS_READING_ARENAS, execution_id="TESTROICAMERAS")

        assert set(outputs) == {"seq_0", "seq_1", "seq_2"}
        assert result.failed_entries == (make_entry_key(*FOUR[3]),)
        errors = entry_error_lines(ds, "TESTROICAMERAS")
        assert len(errors) == 1
        assert "seq_3" in errors[0]
