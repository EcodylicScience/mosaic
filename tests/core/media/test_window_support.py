"""Whether the installed OpenCV can open a window, and the refusal that asks.

The headless wheel has no window functions, and the failure of ``cv2.imshow`` comes
after the work that produced the frame. The build information names the GUI backend
of the build without touching a display, which a probe call on a machine without one
may not survive.
"""

from __future__ import annotations

import pytest

from mosaic.core.media import window_support
from mosaic.core.media.window_support import (
    WindowSupportError,
    gui_backend_from_build_information,
    require_playback_target,
    require_window_support,
)

_HEADLESS_BUILD = """
General configuration for OpenCV 4.11.0 =====================================
  Video I/O:
    FFMPEG:                      YES

  GUI:                           NONE

  Media I/O:
    ZLib:                        build (ver 1.3.1)
"""

_QT_BUILD = """
  GUI:                           QT6
    QT:                          YES (ver 6.11.3 )
      QT OpenGL support:         NO
    Cocoa:                       YES
"""

_WINDOWS_BUILD = """
  GUI:                           WIN32UI
    Win32 UI:                    YES
"""

_GTK_BUILD = """
  GUI:                           GTK3
    GTK+:                        YES (ver 3.24.33)
"""


@pytest.mark.parametrize(
    ("build_information", "expected"),
    [
        (_HEADLESS_BUILD, None),
        (_QT_BUILD, "QT6"),
        (_WINDOWS_BUILD, "WIN32UI"),
        (_GTK_BUILD, "GTK3"),
    ],
)
def test_the_gui_line_of_the_build_names_the_backend(
    build_information: str, expected: str | None
) -> None:
    assert gui_backend_from_build_information(build_information) == expected


def test_a_build_without_a_gui_line_counts_as_having_none() -> None:
    assert gui_backend_from_build_information("Video I/O:\n  FFMPEG: YES\n") is None


def test_the_probe_reads_the_installed_build(monkeypatch: pytest.MonkeyPatch) -> None:
    window_support.gui_backend.cache_clear()
    monkeypatch.setattr(
        window_support.cv2, "getBuildInformation", lambda: _HEADLESS_BUILD
    )
    try:
        assert window_support.gui_backend() is None
    finally:
        window_support.gui_backend.cache_clear()


def test_the_probe_reads_the_build_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def build_information() -> str:
        calls.append(1)
        return _QT_BUILD

    window_support.gui_backend.cache_clear()
    monkeypatch.setattr(window_support.cv2, "getBuildInformation", build_information)
    try:
        assert window_support.gui_backend() == "QT6"
        assert window_support.gui_backend() == "QT6"
    finally:
        window_support.gui_backend.cache_clear()
    assert len(calls) == 1


def test_a_build_with_a_backend_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(window_support, "gui_backend", lambda: "QT6")
    require_window_support("play_video")


def test_a_build_without_a_backend_is_refused_with_the_way_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(window_support, "gui_backend", lambda: None)
    with pytest.raises(WindowSupportError) as raised:
        require_window_support("play_video")
    message = str(raised.value)
    assert "play_video" in message
    assert "py-opencv" in message
    assert "show_window=False" in message


def test_a_call_with_no_window_and_no_output_is_refused_as_doing_nothing() -> None:
    with pytest.raises(ValueError) as raised:
        require_playback_target("play_video", show_window=False, output_path=None)
    message = str(raised.value)
    assert "play_video" in message
    assert "output_path" in message
    assert "show_window=True" in message


def test_a_call_with_no_window_and_an_empty_output_is_refused() -> None:
    with pytest.raises(ValueError, match="output_path"):
        require_playback_target("play_video", show_window=False, output_path="")


def test_a_call_with_an_output_needs_no_window_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(window_support, "gui_backend", lambda: None)
    require_playback_target("play_video", show_window=False, output_path="out.mp4")


def test_a_window_is_checked_for_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(window_support, "gui_backend", lambda: None)
    with pytest.raises(WindowSupportError):
        require_playback_target("play_video", show_window=True, output_path=None)
    monkeypatch.setattr(window_support, "gui_backend", lambda: "QT6")
    require_playback_target("play_video", show_window=True, output_path=None)
