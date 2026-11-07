"""``play_video`` shows no window unless asked, and refuses before any work if it cannot."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from mosaic.behavior.visualization_library import playback
from mosaic.behavior.visualization_library.playback import (
    play_video,
    play_video_with_spec,
)
from mosaic.core.media import window_support
from mosaic.core.media.window_support import WindowSupportError


class _TouchedDataset:
    """A dataset that fails the test on any use."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"play_video read the dataset ({name}) before refusing")


def test_the_window_is_off_by_default() -> None:
    parameter = inspect.signature(play_video).parameters["show_window"]
    assert parameter.default is False


def test_a_window_without_gui_support_is_refused_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(window_support, "gui_backend", lambda: None)

    def overlay_must_not_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("the overlay was built before the refusal")

    monkeypatch.setattr(playback, "build_overlay", overlay_must_not_build)

    with pytest.raises(WindowSupportError, match="show_window=False"):
        play_video(
            _TouchedDataset(),  # pyright: ignore[reportArgumentType]
            "group",
            "sequence",
            feature_runs={},
            show_window=True,
        )


def test_no_window_and_no_output_is_refused_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overlay_must_not_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("the overlay was built before the refusal")

    monkeypatch.setattr(playback, "build_overlay", overlay_must_not_build)

    with pytest.raises(ValueError, match="output_path"):
        play_video(
            _TouchedDataset(),  # pyright: ignore[reportArgumentType]
            "group",
            "sequence",
            feature_runs={},
        )


def test_the_spec_wrapper_refuses_a_window_without_gui_support_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(window_support, "gui_backend", lambda: None)

    def overlay_must_not_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("the overlay was built before the refusal")

    monkeypatch.setattr(playback, "build_overlay", overlay_must_not_build)

    with pytest.raises(WindowSupportError, match="show_window=False"):
        play_video_with_spec(
            _TouchedDataset(),  # pyright: ignore[reportArgumentType]
            "group",
            "sequence",
            {},
            {},
            show_window=True,
        )


def test_the_spec_wrapper_refuses_no_window_and_no_output_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def overlay_must_not_build(*args: object, **kwargs: object) -> None:
        raise AssertionError("the overlay was built before the refusal")

    monkeypatch.setattr(playback, "build_overlay", overlay_must_not_build)

    with pytest.raises(ValueError, match="output_path"):
        play_video_with_spec(
            _TouchedDataset(),  # pyright: ignore[reportArgumentType]
            "group",
            "sequence",
            {},
            {},
        )


def test_the_spec_wrapper_forwards_the_window_and_output_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_build_overlay(**_: Any) -> tuple[dict[str, Any], None, None]:
        return {}, None, None

    monkeypatch.setattr(playback, "build_overlay", fake_build_overlay)
    seen: dict[str, Any] = {}

    def fake_play_video(**kwargs: Any) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(playback, "play_video", fake_play_video)
    target = tmp_path / "out.mp4"

    play_video_with_spec(
        _TouchedDataset(),  # pyright: ignore[reportArgumentType]
        "group",
        "sequence",
        {},
        {},
        output_path=target,
    )

    assert seen["output_path"] == target
    assert seen["show_window"] is False
