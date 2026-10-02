"""Test each media pre-processing step's effect on the placement and the pixels.

Every spatial parameter is a source-space pixel and every frame parameter a
source-space frame number. The tests therefore place each step after a placement
that a recipe produces (an identity, or an earlier crop), and read the pixels back
from a frame whose values record their coordinates.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest
from mosaic_media import probe_media
from pydantic import TypeAdapter, ValidationError

from mosaic.core.media.preprocess import (
    MEDIA_STEPS,
    MIN_CROP_SIDE,
    AdjustStep,
    ClaheStep,
    CropStep,
    DecimateStep,
    Frame,
    FrameMap,
    GrayscaleStep,
    MaskStep,
    MediaStep,
    MediaStepSpec,
    Placement,
    TrimStep,
    apply_clahe,
    make_clahe,
    register_media_step,
    to_gray,
)
from mosaic.core.media.video_io import FFmpegVideoWriter

_SOURCE = Placement.identity(64, 48, 100, 30.0)


def _coordinate_frame(width: int = 64, height: int = 48) -> Frame:
    """Return a BGR frame whose blue channel contains each pixel's x and green its y."""
    grid = np.indices((height, width))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = grid[1]
    frame[..., 1] = grid[0]
    frame[..., 2] = 7
    return frame


def _textured_frame(width: int = 64, height: int = 48) -> Frame:
    """Return a BGR frame with enough variation for an equalization to change it."""
    generator = np.random.default_rng(20260928)
    return generator.integers(0, 256, (height, width, 3), dtype=np.uint8)


# --- crop --------------------------------------------------------------------


def test_a_crop_places_its_rectangle_and_keeps_the_frames() -> None:
    placed = CropStep(x=8, y=4, width=16, height=10).place(_SOURCE)

    assert (placed.offset_x, placed.offset_y) == (8, 4)
    assert (placed.width, placed.height) == (16, 10)
    assert (placed.source_width, placed.source_height) == (64, 48)
    assert placed.frames == _SOURCE.frames
    assert placed.fps == _SOURCE.fps


def test_a_crop_slices_a_color_and_a_gray_frame() -> None:
    step = CropStep(x=8, y=4, width=16, height=10)
    crop = step.bind(_SOURCE)
    frame = _coordinate_frame()

    cropped = crop(frame)

    assert cropped.shape == (10, 16, 3)
    assert (int(cropped[0, 0, 0]), int(cropped[0, 0, 1])) == (8, 4)
    assert (int(cropped[-1, -1, 0]), int(cropped[-1, -1, 1])) == (23, 13)
    assert np.array_equal(crop(frame[..., 0]), frame[4:14, 8:24, 0])


def test_a_second_crop_is_given_in_source_coordinates() -> None:
    """The second rectangle names source pixels instead of the first crop's pixels."""
    first = CropStep(x=8, y=4, width=32, height=24)
    second = CropStep(x=12, y=10, width=8, height=6)
    after_first = first.place(_SOURCE)
    frame = _coordinate_frame()

    placed = second.place(after_first)
    cropped = second.bind(after_first)(first.bind(_SOURCE)(frame))

    assert (placed.offset_x, placed.offset_y, placed.width, placed.height) == (
        12,
        10,
        8,
        6,
    )
    assert np.array_equal(cropped, frame[10:16, 12:20])


@pytest.mark.parametrize(
    ("x", "y", "width", "height"),
    [(0, 0, 7, 8), (0, 0, 8, 7), (0, 0, 2, 8), (0, 0, 8, 2), (-2, 0, 8, 8)],
    ids=["odd-width", "odd-height", "narrow", "short", "negative-x"],
)
def test_a_crop_refuses_a_rectangle_it_cannot_encode(
    x: int, y: int, width: int, height: int
) -> None:
    with pytest.raises(ValidationError):
        _ = CropStep(x=x, y=y, width=width, height=height)


def test_the_smallest_crop_is_the_encoder_minimum() -> None:
    _ = CropStep(x=0, y=0, width=MIN_CROP_SIDE, height=MIN_CROP_SIDE)

    with pytest.raises(ValidationError):
        _ = CropStep(x=0, y=0, width=MIN_CROP_SIDE - 2, height=MIN_CROP_SIDE)


def test_a_crop_outside_the_image_is_refused_naming_both_rectangles() -> None:
    with pytest.raises(ValueError) as refusal:
        _ = CropStep(x=60, y=0, width=8, height=8).place(_SOURCE)

    assert "(60, 0, 8, 8)" in str(refusal.value)
    assert "(0, 0, 64, 48)" in str(refusal.value)


def test_a_crop_outside_an_earlier_crop_is_refused() -> None:
    """A rectangle inside the source but outside the earlier crop is refused."""
    after_first = CropStep(x=8, y=4, width=32, height=24).place(_SOURCE)

    with pytest.raises(ValueError, match=r"\(8, 4, 32, 24\)"):
        _ = CropStep(x=0, y=0, width=16, height=16).place(after_first)


@pytest.mark.media
def test_a_crop_at_the_minimum_size_encodes(tmp_path: Path) -> None:
    """The minimum was measured against SVT-AV1, and this test measures it again."""
    side = MIN_CROP_SIDE
    path = tmp_path / "minimum.mp4"
    writer = FFmpegVideoWriter(path, side, side, fps=30.0)
    for level in (0, 80, 160):
        writer.write(np.full((side, side, 3), level, dtype=np.uint8))
    writer.close()

    facts = probe_media(path)

    assert (facts.width, facts.height, facts.frame_count) == (side, side, 3)


# --- mask --------------------------------------------------------------------

_SQUARE = [(2, 2), (5, 2), (5, 5), (2, 5)]


def test_a_mask_leaves_the_placement_alone() -> None:
    assert MaskStep(polygon=_SQUARE).place(_SOURCE) is _SOURCE


def test_a_mask_blacks_out_the_outside_of_a_color_and_a_gray_frame() -> None:
    placement = Placement.identity(8, 8, 10, 30.0)
    mask = MaskStep(polygon=_SQUARE).bind(placement)
    frame = np.full((8, 8, 3), 200, dtype=np.uint8)

    masked = mask(frame)

    assert masked.shape == (8, 8, 3)
    assert int((masked[..., 0] > 0).sum()) == 16
    assert np.all(masked[2:6, 2:6] == 200)
    assert int(masked[0, 0, 2]) == 0
    assert np.array_equal(mask(frame[..., 0]), masked[..., 0])
    assert np.all(frame == 200), "the input frame must not be written"


def test_a_mask_that_does_not_keep_blacks_out_the_inside() -> None:
    placement = Placement.identity(8, 8, 10, 30.0)
    mask = MaskStep(polygon=_SQUARE, keep=False).bind(placement)

    masked = mask(np.full((8, 8), 200, dtype=np.uint8))

    assert int((masked > 0).sum()) == 64 - 16
    assert np.all(masked[2:6, 2:6] == 0)


def test_a_mask_after_a_crop_is_drawn_in_source_coordinates() -> None:
    after_crop = CropStep(x=8, y=4, width=8, height=8).place(_SOURCE)
    polygon = [(x + 8, y + 4) for x, y in _SQUARE]

    masked = MaskStep(polygon=polygon).bind(after_crop)(
        np.full((8, 8), 200, dtype=np.uint8)
    )

    assert np.all(masked[2:6, 2:6] == 200)
    assert int((masked > 0).sum()) == 16


def test_a_mask_partly_outside_the_image_is_drawn_clipped() -> None:
    placement = Placement.identity(8, 8, 10, 30.0)
    step = MaskStep(polygon=[(-5, -5), (3, -5), (3, 3), (-5, 3)])

    masked = step.bind(step.place(placement))(np.full((8, 8), 200, dtype=np.uint8))

    assert int((masked > 0).sum()) == 16
    assert np.all(masked[0:4, 0:4] == 200)


def test_a_mask_wholly_outside_the_image_is_refused() -> None:
    after_crop = CropStep(x=8, y=4, width=8, height=8).place(_SOURCE)

    with pytest.raises(ValueError, match="does not cover a pixel"):
        _ = MaskStep(polygon=_SQUARE).place(after_crop)


def _circle(cx: int, cy: int, radius: int, vertices: int = 64) -> list[tuple[int, int]]:
    """Return a polygon that approximates a circle, with vertices on whole pixels."""
    angles = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
    return [
        (round(cx + radius * np.cos(angle)), round(cy + radius * np.sin(angle)))
        for angle in angles
    ]


def test_a_mask_whose_bounding_box_alone_meets_the_image_is_refused() -> None:
    """A circle's bounding box covers the crop in its corner. The circle does not."""
    source = Placement.identity(640, 480, 10, 30.0)
    after_crop = CropStep(x=120, y=40, width=32, height=32).place(source)

    with pytest.raises(ValueError, match="does not cover a pixel") as refusal:
        _ = MaskStep(polygon=_circle(320, 240, 200)).place(after_crop)

    assert "x 120..520 and y 40..440" in str(refusal.value)
    assert "x 120..151 and y 40..71" in str(refusal.value)


def test_a_thin_band_past_the_image_corner_is_refused() -> None:
    """The band's bounding box covers the whole image. The band misses every pixel."""
    placement = Placement.identity(8, 8, 10, 30.0)
    band = MaskStep(polygon=[(21, -1), (22, -1), (-1, 22), (-1, 21)])

    with pytest.raises(ValueError, match="does not cover a pixel") as refusal:
        _ = band.place(placement)

    assert "x -1..22 and y -1..22" in str(refusal.value)
    assert "x 0..7 and y 0..7" in str(refusal.value)


def test_a_mask_needs_three_vertices() -> None:
    with pytest.raises(ValidationError):
        _ = MaskStep(polygon=[(0, 0), (4, 4)])


# --- trim and decimate -------------------------------------------------------


def test_a_trim_keeps_its_source_range_and_changes_no_pixel() -> None:
    step = TrimStep(start=10, stop=20)
    frame = _coordinate_frame()

    assert step.place(_SOURCE).frames == FrameMap(10, 1, 10)
    assert step.bind(_SOURCE)(frame) is frame


def test_a_trim_after_a_decimate_starts_at_the_next_kept_frame() -> None:
    decimated = DecimateStep(every=4).place(_SOURCE)

    assert TrimStep(start=5, stop=30).place(decimated).frames == FrameMap(8, 4, 6)


@pytest.mark.parametrize(("start", "stop"), [(20, 20), (30, 20)])
def test_a_trim_refuses_a_start_at_or_past_its_stop(start: int, stop: int) -> None:
    with pytest.raises(ValidationError):
        _ = TrimStep(start=start, stop=stop)


def test_a_trim_refuses_a_stop_past_the_source() -> None:
    with pytest.raises(ValueError) as refusal:
        _ = TrimStep(start=10, stop=101).place(_SOURCE)

    assert "[10, 101)" in str(refusal.value)
    assert "[0, 100)" in str(refusal.value)


def test_a_trim_refuses_the_decimated_span_past_the_source() -> None:
    """A decimated span can extend past the last source frame, and a trim may not."""
    source = Placement.identity(64, 48, 99, 30.0)
    decimated = DecimateStep(every=4).place(source)
    assert decimated.frames.end == 100

    with pytest.raises(ValueError, match="past the source"):
        _ = TrimStep(start=0, stop=100).place(decimated)


def test_a_decimate_multiplies_the_step() -> None:
    step = DecimateStep(every=3)
    frame = _coordinate_frame()

    assert step.place(_SOURCE).frames == FrameMap(0, 3, 34)
    assert step.bind(_SOURCE)(frame) is frame


@pytest.mark.parametrize("every", [1, 0, -2])
def test_a_decimate_refuses_a_factor_below_two(every: int) -> None:
    with pytest.raises(ValidationError):
        _ = DecimateStep(every=every)


# --- grayscale, adjust, clahe ------------------------------------------------


def test_grayscale_converts_color_and_passes_gray() -> None:
    step = GrayscaleStep()
    frame = _textured_frame()
    gray = frame[..., 1].copy()
    convert = step.bind(_SOURCE)

    assert step.place(_SOURCE) is _SOURCE
    assert np.array_equal(convert(frame), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert convert(gray) is gray


def test_adjust_with_its_defaults_changes_nothing() -> None:
    frame = _textured_frame()
    step = AdjustStep()

    assert step.place(_SOURCE) is _SOURCE
    assert np.array_equal(step.lookup_table(), np.arange(256, dtype=np.uint8))
    assert np.array_equal(step.bind(_SOURCE)(frame), frame)


@pytest.mark.parametrize(
    ("step", "level", "expected"),
    [
        (AdjustStep(brightness=10.0), 0, 10),
        (AdjustStep(brightness=10.0), 250, 255),
        (AdjustStep(brightness=-10.0), 5, 0),
        (AdjustStep(contrast=2.0), 100, 200),
        (AdjustStep(contrast=2.0), 200, 255),
        (AdjustStep(gamma=2.0), 64, 128),
        (AdjustStep(contrast=2.0, brightness=-50.0, gamma=0.5), 100, 88),
        (AdjustStep(brightness=0.5), 1, 2),
        (AdjustStep(brightness=0.5), 2, 2),
    ],
    ids=[
        "brightness",
        "brightness-clipped",
        "darkened-clipped",
        "contrast",
        "contrast-clipped",
        "gamma",
        "all-three",
        "tie-rounds-up-to-even",
        "tie-rounds-down-to-even",
    ],
)
def test_adjust_maps_a_level_through_the_formula(
    step: AdjustStep, level: int, expected: int
) -> None:
    """Each level becomes ``clip(c * v + b, 0, 255)``, then the gamma curve.

    The gamma curve is ``255 * (v / 255) ** (1 / gamma)``. The result is rounded
    half to even, and 1.5 and 2.5 both become 2.
    """
    adjust = step.bind(_SOURCE)

    assert int(adjust(np.full((2, 2), level, dtype=np.uint8))[0, 0]) == expected
    assert np.all(adjust(np.full((2, 2, 3), level, dtype=np.uint8)) == expected)


@pytest.mark.parametrize(
    "fields",
    [
        {"brightness": 256.0},
        {"brightness": -256.0},
        {"contrast": 0.0},
        {"contrast": -1.0},
        {"contrast": float("inf")},
        {"gamma": 0.0},
        {"gamma": float("nan")},
    ],
    ids=[
        "brightness-high",
        "brightness-low",
        "zero-contrast",
        "negative-contrast",
        "infinite-contrast",
        "zero-gamma",
        "nan-gamma",
    ],
)
def test_adjust_refuses_a_value_out_of_range(fields: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        _ = AdjustStep.model_validate(fields)


def test_clahe_equalizes_gray_directly_and_color_on_lightness() -> None:
    step = ClaheStep(clip_limit=3.0, tile_grid_size=4)
    equalize = step.bind(_SOURCE)
    frame = _textured_frame()
    gray = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    reference = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = reference.apply(lab[:, :, 0])

    assert step.place(_SOURCE) is _SOURCE
    assert np.array_equal(equalize(gray), reference.apply(gray))
    assert np.array_equal(equalize(frame), cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    assert not np.array_equal(equalize(gray), gray), "the fixture must be changed"


@pytest.mark.parametrize(
    "fields",
    [{"clip_limit": 0.0}, {"clip_limit": -1.0}, {"tile_grid_size": 0}],
    ids=["zero-clip-limit", "negative-clip-limit", "zero-tiles"],
)
def test_clahe_refuses_a_value_out_of_range(fields: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        _ = ClaheStep.model_validate(fields)


def test_grayscale_then_clahe_equalizes_the_gray_frame() -> None:
    frame = _textured_frame()
    gray = GrayscaleStep().bind(_SOURCE)
    equalize = ClaheStep().bind(_SOURCE)

    chained = equalize(gray(frame))

    assert chained.ndim == 2
    assert np.array_equal(chained, apply_clahe(to_gray(frame), make_clahe(2.0, 8)))


# --- the registry and the union ----------------------------------------------


def test_registering_a_second_step_under_a_taken_name_is_refused() -> None:
    class TwinCrop(MediaStep):
        name: ClassVar[str] = "crop"
        version: ClassVar[str] = "0.1"
        moves_pixels: ClassVar[bool] = True
        appearance: ClassVar[bool] = False

    with pytest.raises(ValueError) as refusal:
        _ = register_media_step(TwinCrop)

    assert "TwinCrop" in str(refusal.value)
    assert "mosaic.core.media.preprocess.crop.CropStep" in str(refusal.value)
    assert MEDIA_STEPS["crop"] is CropStep


def test_registering_a_step_without_a_name_is_refused() -> None:
    class Nameless(MediaStep):
        version: ClassVar[str] = "0.1"
        moves_pixels: ClassVar[bool] = False
        appearance: ClassVar[bool] = False

    class Blank(MediaStep):
        name: ClassVar[str] = ""
        version: ClassVar[str] = "0.1"
        moves_pixels: ClassVar[bool] = False
        appearance: ClassVar[bool] = False

    for step in (Nameless, Blank):
        with pytest.raises(ValueError, match="non-empty 'name'"):
            _ = register_media_step(step)
    assert "" not in MEDIA_STEPS


def test_registering_a_step_without_its_declarations_is_refused() -> None:
    class Undeclared(MediaStep):
        name: ClassVar[str] = "undeclared"

    with pytest.raises(TypeError, match="version"):
        _ = register_media_step(Undeclared)
    assert "undeclared" not in MEDIA_STEPS


def test_the_base_step_names_the_class_that_did_not_define_its_methods() -> None:
    class Bare(MediaStep):
        pass

    with pytest.raises(NotImplementedError, match="Bare"):
        _ = Bare().place(_SOURCE)
    with pytest.raises(NotImplementedError, match="Bare"):
        _ = Bare().bind(_SOURCE)


def _union_members() -> list[type[MediaStep]]:
    """The model classes that the ``MediaStepSpec`` union contains."""
    annotated = MediaStepSpec.__value__
    union = typing.get_args(annotated)[0]
    return [
        member for member in typing.get_args(union) if issubclass(member, MediaStep)
    ]


def test_the_union_and_the_registry_name_the_same_steps() -> None:
    members = _union_members()

    assert sorted(member.name for member in members) == sorted(MEDIA_STEPS)
    assert all(MEDIA_STEPS[member.name] is member for member in members)


_DECLARED_FLAGS = {
    "crop": (True, False),
    "mask": (False, False),
    "trim": (False, False),
    "decimate": (False, False),
    "grayscale": (False, True),
    "adjust": (False, True),
    "clahe": (False, True),
}
"""Each step's ``(moves_pixels, appearance)``."""


@pytest.mark.parametrize("name", sorted(MEDIA_STEPS))
def test_every_step_declares_whether_it_moves_pixels_and_changes_appearance(
    name: str,
) -> None:
    step = MEDIA_STEPS[name]

    assert (step.moves_pixels, step.appearance) == _DECLARED_FLAGS.get(name)


def test_every_step_is_named_by_its_discriminator() -> None:
    for name, step in MEDIA_STEPS.items():
        assert step.model_fields["step"].default == name


def test_a_list_of_steps_validates_from_json_into_step_objects() -> None:
    text = json.dumps(
        [
            {"step": "crop", "x": 8, "y": 4, "width": 16, "height": 10},
            {"step": "trim", "start": 0, "stop": 50},
            {"step": "grayscale"},
            {"step": "clahe", "clip_limit": 3.0},
        ]
    )

    steps = TypeAdapter(list[MediaStepSpec]).validate_json(text)

    assert [type(step) for step in steps] == [
        CropStep,
        TrimStep,
        GrayscaleStep,
        ClaheStep,
    ]
    assert steps[0] == CropStep(x=8, y=4, width=16, height=10)
    assert steps[3] == ClaheStep(clip_limit=3.0)


@pytest.mark.parametrize(
    "item",
    [
        {"step": "blur", "radius": 3},
        {"x": 0, "y": 0, "width": 8, "height": 8},
        {"step": "grayscale", "strength": 1},
    ],
    ids=["unknown-step", "no-step", "unknown-field"],
)
def test_a_list_of_steps_refuses_a_step_it_does_not_know(
    item: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        _ = TypeAdapter(list[MediaStepSpec]).validate_python([item])
