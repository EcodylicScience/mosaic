"""Arena layouts: their geometry, their key and the arenas index that stores them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic.json_schema import JsonSchemaMode

from mosaic.core.arenas import (
    Arena,
    ArenaKey,
    ArenaLayout,
    Rect,
    layout_lookup_order,
    region_boundary,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.arena_index import (
    ArenaIndexError,
    ArenaLayoutRow,
    arena_index,
    arena_index_path,
)
from mosaic.core.recording import RecordingKey
from mosaic.core.regions import CircleShape, PolygonShape, RectShape, Roi
from mosaic.tracking.ops.assign_arenas import AssignArenasParams, assign_arenas_run_id
from tests.helpers import GOLDEN_DIR, add_track_sequences, make_dataset

PRINCETON_XS = (0, 506, 1012)
PRINCETON_YS = (0, 170, 346, 524, 702, 878, 1068)
TWO_ARENAS_FILE = GOLDEN_DIR / "arenas" / "two_arenas.json"

TWO_ARENAS_DIGEST = "de511afdfccdc88cebe112053afc690e9bef213ea3ce9297d5b4d6cda1465ba2"
FITTED_LAYOUT_DIGEST = (
    "1f8c5b43a3cbe09a589f8d42176aa56661f1050ca1af938ae28cd8da35986190"
)
FITTED_LAYOUT_RUN_ID = "assign-arenas.0.1-afb0b50bcd"


def _two_arenas(width: int = 200) -> ArenaLayout:
    """A layout of a left and a right arena over a 200x100 frame."""
    return ArenaLayout(
        width=width,
        height=100,
        arenas=(
            Arena(name="left", region=Rect(x0=0, y0=0, x1=100, y1=100)),
            Arena(name="right", region=Rect(x0=100, y0=0, x1=200, y1=100)),
        ),
    )


def _one_arena() -> ArenaLayout:
    return ArenaLayout(
        width=200,
        height=100,
        arenas=(Arena(name="all", region=Rect(x0=0, y0=0, x1=200, y1=100)),),
    )


def _fitted_layout() -> ArenaLayout:
    """A two-arena layout with an excluded rect and capacities."""
    return ArenaLayout(
        width=200,
        height=100,
        arenas=(
            Arena(
                name="left",
                region=Rect(x0=0, y0=0, x1=100, y1=100),
                exclude=(Rect(x0=40, y0=0, x1=60, y1=100),),
                capacity=2,
            ),
            Arena(
                name="right",
                region=Rect(x0=100, y0=0, x1=200, y1=100),
                capacity=1,
            ),
        ),
    )


def _nest() -> Roi:
    return Roi(name="nest", shape=RectShape(x0=0.0, y0=0.0, x1=10.0, y1=10.0))


def _with_boundary(
    boundary: CircleShape | PolygonShape | None,
    *,
    rois: tuple[Roi, ...] = (),
    region: Rect = Rect(x0=0, y0=0, x1=100, y1=100),
    width: int = 200,
) -> ArenaLayout:
    """A layout of one arena over *region* of a *width* by 100 frame."""
    return ArenaLayout(
        width=width,
        height=100,
        arenas=(Arena(name="left", region=region, boundary=boundary, rois=rois),),
    )


class TestGeometry:
    def test_the_princeton_grid_names_twelve_tanks_column_by_column(self) -> None:
        layout = ArenaLayout.grid(xs=PRINCETON_XS, ys=PRINCETON_YS)

        assert layout.names == (
            *(f"A{row}" for row in range(1, 7)),
            *(f"B{row}" for row in range(1, 7)),
        )
        assert (layout.width, layout.height) == (1012, 1068)
        assert layout.arenas[0].region == Rect(x0=0, y0=0, x1=506, y1=170)
        assert layout.arenas[11].region == Rect(x0=506, y0=878, x1=1012, y1=1068)
        assert {arena.capacity for arena in layout.arenas} == {1}

    def test_the_grid_excludes_the_rects_it_names(self) -> None:
        strip = Rect(x0=95, y0=0, x1=105, y1=170)
        layout = ArenaLayout.grid(
            xs=PRINCETON_XS, ys=PRINCETON_YS, exclude={"A1": [strip]}
        )

        assert layout.arenas[0].exclude == (strip,)
        with pytest.raises(ValueError, match="lacks"):
            _ = ArenaLayout.grid(xs=PRINCETON_XS, ys=PRINCETON_YS, exclude={"C1": []})

    def test_the_grid_gives_every_arena_the_rois_it_is_given(self) -> None:
        rois = (
            Roi(name="nest", shape=RectShape(x0=2.0, y0=2.0, x1=8.0, y1=8.0)),
            Roi(name="drain", shape=CircleShape(cx=5.0, cy=5.0, r=1.0)),
        )

        layout = ArenaLayout.grid(xs=PRINCETON_XS, ys=PRINCETON_YS, rois=rois)

        assert [arena.rois for arena in layout.arenas] == [rois] * 12
        assert layout.assignment_digest() == (
            ArenaLayout.grid(xs=PRINCETON_XS, ys=PRINCETON_YS).digest()
        )

    def test_a_grid_without_rois_dumps_without_the_field(self) -> None:
        layout = ArenaLayout.grid(xs=PRINCETON_XS, ys=PRINCETON_YS, rois=())

        assert all("rois" not in arena for arena in layout.model_dump()["arenas"])
        assert (
            layout.digest()
            == ArenaLayout.grid(xs=PRINCETON_XS, ys=PRINCETON_YS).digest()
        )

    def test_overlapping_arenas_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="overlap"):
            _ = ArenaLayout(
                width=200,
                height=100,
                arenas=(
                    Arena(name="a", region=Rect(x0=0, y0=0, x1=101, y1=100)),
                    Arena(name="b", region=Rect(x0=100, y0=0, x1=200, y1=100)),
                ),
            )

    def test_touching_arenas_share_no_pixel(self) -> None:
        """Rects are half-open, so the edge at x 100 belongs to the right arena."""
        layout = _two_arenas()

        assert not layout.arenas[0].region.overlaps(layout.arenas[1].region)

    def test_an_arena_outside_the_frame_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="leaves the 150x100 frame"):
            _ = _two_arenas(width=150)

    def test_an_exclude_outside_its_region_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="outside its region"):
            _ = Arena(
                name="a",
                region=Rect(x0=0, y0=0, x1=10, y1=10),
                exclude=(Rect(x0=5, y0=5, x1=20, y1=10),),
            )

    @pytest.mark.parametrize("name", ["", " a", "a__b", "a:b", "a/b"])
    def test_an_arena_name_is_an_entry_name_token(self, name: str) -> None:
        with pytest.raises(ValidationError):
            _ = Arena(name=name, region=Rect(x0=0, y0=0, x1=10, y1=10))

    def test_repeated_names_and_empty_layouts_are_refused(self) -> None:
        region = Rect(x0=0, y0=0, x1=10, y1=10)
        with pytest.raises(ValidationError, match="repeat"):
            _ = ArenaLayout(
                width=30,
                height=10,
                arenas=(
                    Arena(name="a", region=region),
                    Arena(name="a", region=Rect(x0=10, y0=0, x1=20, y1=10)),
                ),
            )
        with pytest.raises(ValidationError, match="at least one arena"):
            _ = ArenaLayout(width=30, height=10, arenas=())

    def test_a_bad_arena_reports_its_own_error_and_no_empty_layout(self) -> None:
        wire = {
            "width": 100,
            "height": 100,
            "arenas": [
                {
                    "name": "A1",
                    "region": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
                    "boundary": {"kind": "circle", "cx": 50, "cy": 50, "r": -1},
                }
            ],
        }

        with pytest.raises(ValidationError) as refused:
            _ = ArenaLayout.model_validate(wire)

        assert [error["loc"] for error in refused.value.errors()] == [
            ("arenas", 0, "boundary", "circle", "r")
        ]

    def test_the_schema_still_requires_an_arena(self) -> None:
        schema = ArenaLayout.model_json_schema()

        assert schema["properties"]["arenas"]["minItems"] == 1

    def test_a_rect_without_area_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="x0 < x1"):
            _ = Rect(x0=10, y0=0, x1=10, y1=5)


class TestDigest:
    def test_the_digest_is_pinned(self) -> None:
        """A layout's digest names it in the tracks it assigns, so it cannot move."""
        layout = ArenaLayout.grid(xs=PRINCETON_XS, ys=PRINCETON_YS)

        assert layout.digest() == (
            "9f955c44e22706564636823bb3a81fe3495be93c73e79afbde5d14f0f889d2c2"
        )

    def test_key_order_does_not_change_the_digest(self) -> None:
        layout = _two_arenas()
        dumped = layout.model_dump(mode="json")
        reordered = dict(reversed(list(dumped.items())))

        assert ArenaLayout.model_validate(reordered).digest() == layout.digest()

    def test_a_moved_edge_changes_the_digest(self) -> None:
        moved = ArenaLayout(
            width=200,
            height=100,
            arenas=(
                Arena(name="left", region=Rect(x0=0, y0=0, x1=99, y1=100)),
                Arena(name="right", region=Rect(x0=100, y0=0, x1=200, y1=100)),
            ),
        )

        assert moved.digest() != _two_arenas().digest()


class TestDigestStability:
    """A layout without a boundary or ROIs keeps its digest and its file bytes."""

    def test_a_layout_with_an_excluded_rect_and_a_capacity_keeps_its_digest(
        self,
    ) -> None:
        assert _fitted_layout().digest() == FITTED_LAYOUT_DIGEST

    def test_a_layout_without_either_keeps_its_digest(self) -> None:
        assert _two_arenas().digest() == TWO_ARENAS_DIGEST

    def test_a_layout_file_written_before_the_change_dumps_to_the_same_bytes(
        self,
    ) -> None:
        text = TWO_ARENAS_FILE.read_text(encoding="utf-8")

        layout = ArenaLayout.model_validate_json(text)

        assert layout == _fitted_layout()
        assert layout.model_dump_json(indent=2) + "\n" == text

    def test_a_layout_without_them_dumps_the_keys_it_always_did(self) -> None:
        dumped = _fitted_layout().model_dump(mode="json")

        assert [set(arena) for arena in dumped["arenas"]] == [
            {"name", "region", "exclude", "capacity"}
        ] * 2

    def test_the_two_fields_leave_a_default_out_of_a_dump(self) -> None:
        """Without ``exclude_if`` a dump carries ``boundary: null`` and ``rois: []``."""
        assert Arena.model_fields["boundary"].exclude_if is not None
        assert Arena.model_fields["rois"].exclude_if is not None

    def test_the_assignment_run_id_over_the_fitted_layout_stays_put(self) -> None:
        layout = _fitted_layout()

        run_id = assign_arenas_run_id(
            "assign-arenas",
            "0.1",
            AssignArenasParams(),
            "upstream-tracks-run",
            {"rec__s1": layout.assignment_digest()},
        )

        assert layout.assignment_digest() == FITTED_LAYOUT_DIGEST
        assert run_id == FITTED_LAYOUT_RUN_ID

    def test_a_layout_with_a_boundary_and_a_roi_dumps_both(self) -> None:
        layout = _with_boundary(CircleShape(cx=50.0, cy=50.0, r=40.0), rois=(_nest(),))

        (arena,) = layout.model_dump(mode="json")["arenas"]

        assert arena["boundary"]["kind"] == "circle"
        assert [roi["name"] for roi in arena["rois"]] == ["nest"]
        assert ArenaLayout.model_validate_json(layout.model_dump_json()) == layout


class TestSchema:
    @pytest.mark.parametrize("mode", ["serialization", "validation"])
    def test_the_schema_names_the_boundary_and_the_rois_with_their_types(
        self, mode: JsonSchemaMode
    ) -> None:
        schema = ArenaLayout.model_json_schema(mode=mode)
        properties = schema["$defs"]["Arena"]["properties"]

        boundary = json.dumps(properties["boundary"])
        assert "CircleShape" in boundary
        assert "PolygonShape" in boundary
        assert "RingShape" not in boundary
        assert properties["rois"]["type"] == "array"
        assert properties["rois"]["items"] == {"$ref": "#/$defs/Roi"}
        assert {"CircleShape", "PolygonShape", "Roi"} <= set(schema["$defs"])


class TestBoundaryKind:
    def test_a_boundary_without_a_kind_is_refused(self) -> None:
        arena = {
            "name": "left",
            "region": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
            "boundary": {"cx": 50.0, "cy": 50.0, "r": 20.0},
        }

        with pytest.raises(ValidationError, match="kind"):
            _ = Arena.model_validate(arena)

    def test_a_boundary_with_a_kind_is_read_as_that_shape(self) -> None:
        arena = Arena.model_validate(
            {
                "name": "left",
                "region": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
                "boundary": {"kind": "circle", "cx": 50.0, "cy": 50.0, "r": 20.0},
            }
        )

        assert arena.boundary == CircleShape(cx=50.0, cy=50.0, r=20.0)

    def test_a_ring_is_not_a_boundary(self) -> None:
        arena = {
            "name": "left",
            "region": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
            "boundary": {
                "kind": "ring",
                "cx": 50.0,
                "cy": 50.0,
                "r_inner": 5.0,
                "r_outer": 20.0,
            },
        }

        with pytest.raises(ValidationError, match="boundary"):
            _ = Arena.model_validate(arena)


class TestAssignmentDigest:
    def test_it_equals_the_digest_without_a_boundary_or_rois(self) -> None:
        layout = _two_arenas()

        assert layout.assignment_digest() == layout.digest()

    def test_it_ignores_a_boundary_and_rois_that_the_digest_covers(self) -> None:
        plain = _with_boundary(None)
        fitted = _with_boundary(CircleShape(cx=50.0, cy=50.0, r=40.0), rois=(_nest(),))

        assert fitted.assignment_digest() == plain.digest()
        assert fitted.digest() != plain.digest()

    def test_a_roi_alone_changes_the_digest_and_not_the_assignment_digest(
        self,
    ) -> None:
        plain = _with_boundary(None)
        with_roi = _with_boundary(None, rois=(_nest(),))

        assert with_roi.assignment_digest() == plain.assignment_digest()
        assert with_roi.digest() != plain.digest()

    def test_it_follows_the_regions_and_exclusions(self) -> None:
        assert _fitted_layout().assignment_digest() != _two_arenas().assignment_digest()


class TestBoundaryValidation:
    def test_a_circle_past_the_region_inside_the_frame_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="boundary"):
            _ = _with_boundary(CircleShape(cx=80.0, cy=50.0, r=30.0))

    def test_a_circle_in_another_arena_s_region_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="left"):
            _ = _with_boundary(CircleShape(cx=150.0, cy=50.0, r=20.0))

    def test_a_circle_past_the_frame_s_left_edge_is_accepted_when_its_clip_fits(
        self,
    ) -> None:
        layout = _with_boundary(CircleShape(cx=10.0, cy=50.0, r=30.0))

        assert layout.arenas[0].boundary == CircleShape(cx=10.0, cy=50.0, r=30.0)

    def test_a_circle_past_the_frame_s_far_edge_is_accepted_when_its_clip_fits(
        self,
    ) -> None:
        layout = _with_boundary(
            CircleShape(cx=190.0, cy=50.0, r=30.0),
            region=Rect(x0=100, y0=0, x1=200, y1=100),
        )

        assert layout.arenas[0].boundary is not None

    def test_a_boundary_with_no_part_in_the_frame_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="frame"):
            _ = _with_boundary(CircleShape(cx=-50.0, cy=50.0, r=20.0))

    def test_a_polygon_inside_the_region_is_accepted(self) -> None:
        polygon = PolygonShape(vertices=((10.0, 10.0), (90.0, 10.0), (50.0, 90.0)))

        assert _with_boundary(polygon).arenas[0].boundary == polygon

    def test_a_polygon_past_the_region_is_refused(self) -> None:
        polygon = PolygonShape(vertices=((10.0, 10.0), (110.0, 10.0), (50.0, 90.0)))

        with pytest.raises(ValidationError, match="boundary"):
            _ = _with_boundary(polygon)

    def test_a_repeated_roi_name_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="nest"):
            _ = Arena(
                name="left",
                region=Rect(x0=0, y0=0, x1=100, y1=100),
                rois=(_nest(), _nest()),
            )


class TestSingle:
    def test_a_circle_is_boxed_outward_and_clipped_to_the_frame(self) -> None:
        circle = CircleShape(cx=50.0, cy=60.0, r=80.0)

        layout = ArenaLayout.single(circle, width=640)

        assert layout.arenas[0].region == Rect(x0=0, y0=0, x1=130, y1=140)
        assert (layout.width, layout.height) == (640, 140)
        assert layout.names == ("A1",)
        assert layout.arenas[0].boundary == circle
        assert layout.arenas[0].capacity == 1

    def test_a_given_height_clips_the_region(self) -> None:
        circle = CircleShape(cx=50.0, cy=60.0, r=80.0)

        layout = ArenaLayout.single(circle, width=640, height=100)

        assert layout.arenas[0].region == Rect(x0=0, y0=0, x1=130, y1=100)
        assert (layout.width, layout.height) == (640, 100)

    def test_the_box_rounds_outward(self) -> None:
        layout = ArenaLayout.single(CircleShape(cx=10.5, cy=10.5, r=5.0))

        assert layout.arenas[0].region == Rect(x0=5, y0=5, x1=16, y1=16)
        assert (layout.width, layout.height) == (16, 16)

    def test_the_name_capacity_and_rois_are_the_arena_s(self) -> None:
        layout = ArenaLayout.single(
            CircleShape(cx=50.0, cy=50.0, r=40.0),
            name="tank",
            capacity=None,
            rois=(_nest(),),
        )

        arena = layout.arenas[0]
        assert (arena.name, arena.capacity, arena.rois) == ("tank", None, (_nest(),))

    def test_a_boundary_wholly_at_negative_x_is_refused(self) -> None:
        with pytest.raises(ValueError, match="frame"):
            _ = ArenaLayout.single(CircleShape(cx=-100.0, cy=50.0, r=20.0))

    def test_a_boundary_wholly_past_the_given_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="frame"):
            _ = ArenaLayout.single(CircleShape(cx=700.0, cy=50.0, r=20.0), width=640)

    def test_a_polygon_is_boxed_by_its_vertices(self) -> None:
        polygon = PolygonShape(vertices=((2.5, 3.0), (20.0, 3.0), (10.0, 30.2)))

        layout = ArenaLayout.single(polygon)

        assert layout.arenas[0].region == Rect(x0=2, y0=3, x1=20, y1=31)


def test_a_rect_region_is_its_float_box() -> None:
    assert region_boundary(Rect(x0=1, y0=2, x1=30, y1=40)) == RectShape(
        x0=1.0, y0=2.0, x1=30.0, y1=40.0
    )


def test_a_lookup_tries_the_entry_then_its_recording_then_every_camera() -> None:
    order = layout_lookup_order(
        RecordingKey("rec", "s1"), RecordingKey("rec", ""), "cam1"
    )

    assert order == (
        ArenaKey("rec", "s1", "cam1"),
        ArenaKey("rec", "", "cam1"),
        ArenaKey("rec", "s1", ""),
        ArenaKey("rec", "", ""),
    )
    undivided = RecordingKey("ns", "a")
    assert layout_lookup_order(undivided, undivided, "") == (ArenaKey("ns", "a", ""),)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Tracks of the divided recording ``rec``, the empty-group ``solo`` and ``ns/a``."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("rec", "s1"), ("rec", "s2"), ("", "solo"), ("ns", "a"))
    ds.set_continuous_groups(["rec"])
    return ds


class TestDatasetLayouts:
    def test_a_divided_recording_s_layout_applies_to_every_sequence(
        self, dataset: Dataset
    ) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")

        assert path == dataset.get_root("arenas") / "rec.json"
        for sequence in ("s1", "s2"):
            found = dataset.arena_layout("rec", sequence)
            assert found is not None
            assert found.key == ArenaKey("rec", "", "")
            assert found.layout == _two_arenas()
        assert dataset.arena_layout("", "solo") is None

    def test_a_camera_s_layout_comes_before_every_camera_s(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        _ = dataset.set_arena_layout(_one_arena(), group="rec", camera="top")

        top = dataset.arena_layout("rec", "s1", "top")
        side = dataset.arena_layout("rec", "s1", "side")

        assert top is not None and top.layout == _one_arena()
        assert side is not None and side.layout == _two_arenas()
        assert (dataset.get_root("arenas") / "rec.top.json").is_file()

    def test_an_undivided_recording_is_keyed_by_its_entry(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="ns", sequence="a")

        found = dataset.arena_layout("ns", "a")

        assert found is not None and found.key == ArenaKey("ns", "a", "")
        assert (dataset.get_root("arenas") / "ns__a.json").is_file()

    def test_setting_a_key_again_replaces_its_layout(self, dataset: Dataset) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        _ = dataset.set_arena_layout(_one_arena(), group="rec")

        assert list(dataset.arena_layouts()) == [ArenaKey("rec", "", "")]
        found = dataset.arena_layout("rec", "s1")
        assert found is not None and found.layout == _one_arena()

    def test_a_namespace_group_names_no_recording(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="not a continuous group"):
            _ = dataset.set_arena_layout(_two_arenas(), group="ns")

    def test_a_row_of_a_namespace_group_is_skipped_at_lookup(
        self, dataset: Dataset
    ) -> None:
        """A hand-written ``(ns, "")`` row matches no entry of ``ns``."""
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        index = arena_index(arena_index_path(dataset))
        index.append(
            [
                ArenaLayoutRow(
                    abs_path=Path("arenas/rec.json"),
                    group="ns",
                    sequence="",
                    camera="",
                    digest=_two_arenas().digest(),
                    n_arenas=2,
                )
            ]
        )

        assert dataset.arena_layout("ns", "a") is None

    def test_a_divided_recording_s_sequence_is_reserved(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="once per recording"):
            _ = dataset.set_arena_layout(_two_arenas(), group="rec", sequence="s1")

    def test_a_recording_without_entries_is_refused(self, dataset: Dataset) -> None:
        with pytest.raises(ValueError, match="no entry"):
            _ = dataset.set_arena_layout(_two_arenas(), group="", sequence="ghost")

    def test_two_keys_cannot_share_a_file(self, tmp_path: Path) -> None:
        ds = make_dataset(tmp_path / "ds")
        add_track_sequences(ds, ("", "a.b"), ("", "a"))
        _ = ds.set_arena_layout(_two_arenas(), group="", sequence="a.b")

        with pytest.raises(ArenaIndexError, match="holds the layout of"):
            _ = ds.set_arena_layout(_two_arenas(), group="", sequence="a", camera="b")

    def test_two_rows_of_one_key_are_refused_at_the_read(
        self, dataset: Dataset
    ) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        path = arena_index_path(dataset)
        lines = path.read_text().splitlines()
        _ = path.write_text("\n".join([*lines, lines[-1]]) + "\n")

        with pytest.raises(ArenaIndexError, match="two layouts"):
            _ = dataset.arena_layouts()

    def test_a_changed_file_is_refused_at_the_read(self, dataset: Dataset) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")
        _ = path.write_text(_one_arena().model_dump_json())

        with pytest.raises(ArenaIndexError, match="digest"):
            _ = dataset.arena_layouts()

    def test_a_missing_file_is_refused_at_the_read(self, dataset: Dataset) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")
        path.unlink()

        with pytest.raises(ArenaIndexError, match="cannot be read as a layout"):
            _ = dataset.arena_layouts()

    def test_a_garbled_index_is_refused_at_the_read(self, dataset: Dataset) -> None:
        _ = dataset.set_arena_layout(_two_arenas(), group="rec")
        path = arena_index_path(dataset)
        _ = path.write_text(path.read_text() + '"rec,unterminated\n')

        with pytest.raises(ArenaIndexError, match="cannot be read"):
            _ = dataset.arena_layouts()

    @pytest.mark.parametrize(
        "text", ["{not json", '{"width": 200}'], ids=["garbled", "invalid"]
    )
    def test_an_unparsable_file_is_refused_at_the_read(
        self, dataset: Dataset, text: str
    ) -> None:
        path = dataset.set_arena_layout(_two_arenas(), group="rec")
        _ = path.write_text(text)

        with pytest.raises(ArenaIndexError, match="cannot be read as a layout"):
            _ = dataset.arena_layouts()


def test_a_read_creates_no_file(dataset: Dataset) -> None:
    root = dataset.get_root("arenas")
    before = sorted(root.rglob("*")) if root.exists() else []

    assert dataset.arena_layouts() == {}
    assert dataset.arena_layout("rec", "s1") is None
    assert (sorted(root.rglob("*")) if root.exists() else []) == before

    _ = dataset.set_arena_layout(_two_arenas(), group="rec")
    written = {path: path.stat().st_mtime_ns for path in root.rglob("*")}
    _ = dataset.arena_layouts()
    _ = dataset.arena_layout("rec", "s2", "cam")
    assert {path: path.stat().st_mtime_ns for path in root.rglob("*")} == written


def test_the_path_passes_visit_the_arenas_index(dataset: Dataset) -> None:
    """A layout written with an absolute path in the tree is made relative."""
    from mosaic.core.pipeline.dataset_indexes import (
        iter_dataset_indexes,
        reconcilable_index,
    )

    file = dataset.set_arena_layout(_two_arenas(), group="rec")
    path = arena_index_path(dataset)
    text = path.read_text()
    _ = path.write_text(text.replace("arenas/rec.json", str(file)))

    assert path in {index.path for index in iter_dataset_indexes(dataset)}
    assert reconcilable_index("arenas") is arena_index
    assert dataset.make_portable()[str(path)] == 1
    assert path.read_text() == text
    moved = dataset.rewrite_index_paths({"arenas": "moved"}, dry_run=True)
    assert moved.get(str(path)) == 1
