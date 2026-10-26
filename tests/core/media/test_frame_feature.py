"""``frame_feature``: the k-means feature vector of one decoded frame."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mosaic.core.media.video_io import frame_feature


def _halves(width: int = 8, height: int = 6) -> npt.NDArray[np.uint8]:
    """Return a BGR frame that is black on its left half and white on its right."""
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:, width // 2 :] = 255
    return frame


def test_a_color_feature_keeps_every_channel_scaled_to_one() -> None:
    frame = np.empty((6, 8, 3), np.uint8)
    frame[:] = (10, 200, 30)
    feature = frame_feature(frame, None, (4, 2), False)
    assert feature.dtype == np.float32
    assert feature.shape == (4 * 2 * 3,)
    expected = np.tile(np.array([10, 200, 30], np.float32) / 255.0, 4 * 2)
    np.testing.assert_allclose(feature, expected)


def test_a_gray_feature_has_one_value_per_pixel() -> None:
    frame = np.full((6, 8, 3), 51, np.uint8)
    feature = frame_feature(frame, None, (5, 3), True)
    assert feature.dtype == np.float32
    assert feature.shape == (5 * 3,)
    np.testing.assert_allclose(feature, np.full(15, 0.2, np.float32))


def test_the_crop_is_taken_before_the_resize() -> None:
    whole = frame_feature(_halves(), None, (2, 2), True)
    right = frame_feature(_halves(), (4, 0, 4, 6), (2, 2), True)
    left = frame_feature(_halves(), (0, 0, 4, 6), (2, 2), True)
    np.testing.assert_allclose(whole, [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(right, np.ones(4, np.float32))
    np.testing.assert_allclose(left, np.zeros(4, np.float32))


def test_the_resize_filter_is_read_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-pixel stripe averages under area resizing and is dropped by nearest."""
    import cv2

    from mosaic.core.media import video_io

    frame = np.zeros((1, 9, 3), np.uint8)
    frame[:, 4] = 255
    area = frame_feature(frame, None, (3, 1), True)
    monkeypatch.setattr(video_io, "RESIZE_INTERPOLATION", cv2.INTER_NEAREST)
    nearest = frame_feature(frame, None, (3, 1), True)
    np.testing.assert_allclose(area, [0.0, 1 / 3, 0.0], rtol=1e-6)
    np.testing.assert_allclose(nearest, [0.0, 0.0, 0.0])
