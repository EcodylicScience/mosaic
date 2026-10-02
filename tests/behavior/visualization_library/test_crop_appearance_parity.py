"""Pin the crop features' CLAHE and gray conversion.

``egocentric-crop`` and ``interaction-crop-pipeline`` equalize a crop with CLAHE
and convert it to gray. A change to either computation, including a change to code
that the two share, fails here instead of altering crops under an unchanged feature
version.

Gray conversion is fixed-point arithmetic, so its output is pinned as a SHA-256
digest. CLAHE blends its tile tables in floating point, which is not bit-exact
across CPU architectures: the same OpenCV release gives different bytes on arm64
and x86_64. Its output is compared with OpenCV's CLAHE called directly, with the
clip limit and tile grid spelled out here rather than read from the feature.

CLAHE is checked twice. With the features' defaults, 25 tiles on this crop contain
12 pixels each, and OpenCV raises the clip threshold to one count per bin. Every
clip limit up to about 21 therefore gives the same bytes. With 4 tiles the clip
limit takes effect, which shows that each feature passes its clip limit and tile
grid through.

The crops are 8-bit, as decoded video is, and 16-bit, as a raw imgstore of a
16-bit camera is. Both conversions keep a crop's depth, and CLAHE refuses a
16-bit color crop because OpenCV converts only an 8-bit or float image to LAB.

The crop is taken around the fixture's center at the fixture's size, unrotated.
It is the whole fixture, so the comparisons measure the appearance path alone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from mosaic.behavior.visualization_library.egocentric_crop import EgocentricCrop
from mosaic.behavior.visualization_library.interaction_crop import (
    InteractionCropPipeline,
)

type Image = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
type CropFn = Callable[[Image], Image]

_WIDTH, _HEIGHT = 96, 72
_CENTER = (_WIDTH / 2.0, _HEIGHT / 2.0)
_GEOMETRY: dict[str, object] = {
    "crop_size": (_WIDTH, _HEIGHT),
    "rotate_to_heading": False,
}


def _texture(weights: tuple[int, int, int], modulus: int) -> npt.NDArray[np.int64]:
    """Return values that vary within every CLAHE tile, from weighted positions."""
    grid = np.indices((_HEIGHT, _WIDTH), dtype=np.int64)
    rows, columns = grid[0], grid[1]
    column_weight, row_weight, product_weight = weights
    return (
        columns * column_weight
        + rows * row_weight
        + (columns * rows * product_weight) % 37
    ) % modulus


def _color_crop() -> Image:
    """Return an 8-bit BGR crop whose three channels differ."""
    channels = [
        _texture(weights, 256) for weights in ((5, 3, 1), (2, 9, 4), (11, 13, 7))
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def _gray_crop() -> Image:
    return _texture((3, 7, 1), 256).astype(np.uint8)


def _deep_color_crop() -> Image:
    """Return a 16-bit BGR crop whose three channels differ."""
    channels = [
        _texture(weights, 65536)
        for weights in ((677, 911, 13), (1301, 229, 5), (97, 3001, 29))
    ]
    return np.stack(channels, axis=-1).astype(np.uint16)


def _deep_gray_crop() -> Image:
    return _texture((677, 911, 13), 65536).astype(np.uint16)


_FIXTURES: dict[str, Callable[[], Image]] = {
    "color": _color_crop,
    "gray": _gray_crop,
    "16-bit-color": _deep_color_crop,
    "16-bit-gray": _deep_gray_crop,
}


def _egocentric_crop(flags: dict[str, object]) -> CropFn:
    feature = EgocentricCrop(params={**_GEOMETRY, **flags})

    def crop(image: Image) -> Image:
        return feature.extract_egocentric_crop(image, _CENTER, 0.0)

    return crop


def _interaction_crop(flags: dict[str, object]) -> CropFn:
    feature = InteractionCropPipeline(params={**_GEOMETRY, **flags})

    def crop(image: Image) -> Image:
        return feature.extract_crop(image, _CENTER, 0.0)

    return crop


_FEATURES: dict[str, Callable[[dict[str, object]], CropFn]] = {
    "egocentric-crop": _egocentric_crop,
    "interaction-crop-pipeline": _interaction_crop,
}


_CLIPPING_CLAHE: dict[str, object] = {
    "use_clahe": True,
    "clahe_clip_limit": 3.0,
    "clahe_tile_grid_size": 4,
}
"""CLAHE settings under which the clip limit changes the result on these crops."""


def _digest(image: Image) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


@pytest.mark.parametrize(
    ("fixture", "digest"),
    [
        (
            "color",
            "ec9e8b11d022d07d1c1766546b7e0e9946d6123e3e8b191b0d52b9fc2f5fe777",
        ),
        (
            "gray",
            "403905dc03f24af1e1fdb430f636328ee6a8298ed76585e26e48177d24d476c2",
        ),
        (
            "16-bit-color",
            "8089cff0a4deffb909876312422db94b9baa261c88229641dc2ece72f02372f6",
        ),
        (
            "16-bit-gray",
            "9a7065d00fbecfe07915e11804db9f8f033d4da59b9d35fe12ad2fb8e8ed08e0",
        ),
    ],
)
def test_the_fixtures_are_the_ones_the_digests_were_taken_from(
    fixture: str, digest: str
) -> None:
    assert _digest(_FIXTURES[fixture]()) == digest


@pytest.mark.parametrize("feature", sorted(_FEATURES))
@pytest.mark.parametrize("fixture", sorted(_FIXTURES))
def test_the_crop_around_the_center_is_the_whole_fixture(
    feature: str, fixture: str
) -> None:
    image = _FIXTURES[fixture]()

    assert np.array_equal(_FEATURES[feature]({})(image), image)


def _opencv_clahe(
    image: Image, clip_limit: float, tile_grid_size: int
) -> npt.NDArray[np.uint8 | np.uint16]:
    """Equalize *image* with OpenCV's CLAHE, independently of mosaic's own code.

    A gray image is equalized directly and a BGR one on L of LAB.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
    )
    if image.ndim == 2:
        return np.asarray(clahe.apply(image), dtype=image.dtype)
    lab = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2LAB), dtype=image.dtype)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return np.asarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), dtype=image.dtype)


@pytest.mark.parametrize("feature", sorted(_FEATURES))
@pytest.mark.parametrize(
    ("flags", "fixture", "clip_limit", "tile_grid_size"),
    [
        ({"use_clahe": True}, "color", 2.0, 25),
        ({"use_clahe": True}, "gray", 2.0, 25),
        ({"use_clahe": True}, "16-bit-gray", 2.0, 25),
        (_CLIPPING_CLAHE, "color", 3.0, 4),
        (_CLIPPING_CLAHE, "gray", 3.0, 4),
    ],
    ids=[
        "clahe-on-color",
        "clahe-on-gray",
        "clahe-on-16-bit-gray",
        "clipping-clahe-on-color",
        "clipping-clahe-on-gray",
    ],
)
def test_a_crop_feature_equalizes_as_opencvs_clahe(
    feature: str,
    flags: dict[str, object],
    fixture: str,
    clip_limit: float,
    tile_grid_size: int,
) -> None:
    """Both features equalize exactly as OpenCV does with the stated settings."""
    image = _FIXTURES[fixture]()

    result = _FEATURES[feature](flags)(image)

    assert result.shape[:2] == (_HEIGHT, _WIDTH)
    assert result.dtype == image.dtype
    assert not np.array_equal(result, image), "the fixture must be changed"
    assert np.array_equal(result, _opencv_clahe(image, clip_limit, tile_grid_size))


@pytest.mark.parametrize("fixture", ["color", "gray"])
def test_the_clipping_settings_differ_from_each_default_alone(fixture: str) -> None:
    """A feature dropping either clipping setting for its default would fail above.

    The clipping cases pass only if each feature forwards both settings, because
    keeping either default alone gives different bytes.
    """
    image = _FIXTURES[fixture]()
    clipping = _opencv_clahe(image, 3.0, 4)

    assert not np.array_equal(clipping, _opencv_clahe(image, 2.0, 4))
    assert not np.array_equal(clipping, _opencv_clahe(image, 3.0, 25))


@pytest.mark.parametrize("feature", sorted(_FEATURES))
@pytest.mark.parametrize(
    ("fixture", "digest"),
    [
        (
            "color",
            "1e223ecca321f9746ac0a7b6738528b136c9f75835b31ebf2f26c643acbd2618",
        ),
        (
            "16-bit-color",
            "ceb5a87fcb33ddbd904f13908b62bdfe72a678fe60df92f69075b998fe62a3fb",
        ),
    ],
    ids=["gray-conversion", "16-bit-gray-conversion"],
)
def test_a_crop_feature_converts_to_gray_to_the_pinned_bytes(
    feature: str, fixture: str, digest: str
) -> None:
    """Both features compute the same bytes, and each case has one digest."""
    image = _FIXTURES[fixture]()

    result = _FEATURES[feature]({"grayscale": True})(image)

    assert result.shape == (_HEIGHT, _WIDTH)
    assert result.dtype == image.dtype
    assert _digest(result) == digest


@pytest.mark.parametrize("feature", sorted(_FEATURES))
def test_a_crop_feature_refuses_clahe_on_a_16_bit_color_crop(feature: str) -> None:
    equalize = _FEATURES[feature]({"use_clahe": True})

    with pytest.raises(cv2.error):
        _ = equalize(_deep_color_crop())
