"""``visualize_inference`` shows no window unless asked, and refuses before any work if it cannot."""

from __future__ import annotations

import inspect

import pytest

from mosaic.core.media import window_support
from mosaic.core.media.window_support import WindowSupportError
from mosaic.tracking.pose_training import inference
from mosaic.tracking.pose_training.inference import visualize_inference


def test_the_window_is_off_by_default() -> None:
    parameter = inspect.signature(visualize_inference).parameters["show_window"]
    assert parameter.default is False


def test_a_window_without_gui_support_is_refused_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(window_support, "gui_backend", lambda: None)

    def metadata_must_not_be_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("the video was opened before the refusal")

    monkeypatch.setattr(inference, "get_video_metadata", metadata_must_not_be_read)

    with pytest.raises(WindowSupportError, match="show_window=False"):
        visualize_inference(
            "missing.mp4",
            [[]],
            result_type="localizer",
            show_window=True,
        )


def test_no_window_and_no_output_is_refused_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def metadata_must_not_be_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("the video was opened before the refusal")

    monkeypatch.setattr(inference, "get_video_metadata", metadata_must_not_be_read)

    with pytest.raises(ValueError, match="output_path"):
        visualize_inference("missing.mp4", [[]], result_type="localizer")
