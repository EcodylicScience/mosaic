"""Whether the installed OpenCV can open a window.

Mosaic requires the headless OpenCV wheel, whose window functions (``cv2.imshow``,
``cv2.waitKey``, ``cv2.destroyWindow``) raise ``cv2.error``. A call that shows a
window asks :func:`require_playback_target` first, so that it refuses before it reads
a frame rather than after.

The answer comes from the ``GUI:`` line of ``cv2.getBuildInformation()``, which names
the backend the build was compiled with (``NONE`` for the pip headless wheel, ``QT5``
or ``QT6``, ``GTK3``, ``WIN32UI``, ``COCOA`` for conda-forge's ``py-opencv`` and the
GUI wheel). A probe call such as ``cv2.namedWindow`` is not used: on a GUI build
with no display, Qt can abort the process instead of raising, so a probe could kill
the process that asked it.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import cv2

_GUI_LINE = re.compile(r"^\s*GUI:\s*(?P<backend>\S.*?)\s*$", re.MULTILINE)


class WindowSupportError(RuntimeError):
    """The installed OpenCV was built without a GUI backend."""


def gui_backend_from_build_information(build_information: str) -> str | None:
    """Read the GUI backend out of the text of ``cv2.getBuildInformation()``.

    Args:
        build_information: The build report of an OpenCV build.

    Returns:
        The backend the report names (for example ``"QT6"``), or None when the
        report names ``NONE`` or has no ``GUI:`` line.
    """
    match = _GUI_LINE.search(build_information)
    if match is None:
        return None
    backend = match.group("backend")
    return None if backend.upper() == "NONE" else backend


@cache
def gui_backend() -> str | None:
    """The GUI backend of the installed OpenCV, or None when it has none.

    The build does not change within a process, so the report is read once.
    """
    return gui_backend_from_build_information(cv2.getBuildInformation())


def require_window_support(caller: str) -> None:
    """Raise unless the installed OpenCV can open a window.

    Args:
        caller: The name of the function that was asked for a window, for the
            message.

    Raises:
        WindowSupportError: The installed OpenCV has no GUI backend.
    """
    if gui_backend() is not None:
        return
    message = (
        f"{caller} was asked to show a window (show_window=True), but the installed "
        "OpenCV was built without a GUI backend, as the opencv-python-headless wheel "
        "is. Install a GUI-capable OpenCV (for example conda-forge's py-opencv, as "
        "docs/installation.md describes) or pass show_window=False."
    )
    raise WindowSupportError(message)


def require_playback_target(
    caller: str, *, show_window: bool, output_path: str | Path | None
) -> None:
    """Raise unless a playback call has a window or a file to write.

    A call with neither decodes the whole video and returns nothing, so it is
    refused before it reads a frame.

    Args:
        caller: The name of the function that was called, for the messages.
        show_window: Whether the call was asked to show a window.
        output_path: The file the call was asked to write, if any.

    Raises:
        ValueError: The call asks for neither a window nor an output file.
        WindowSupportError: The call asks for a window and the installed OpenCV
            has no GUI backend.
    """
    if not show_window and not output_path:
        message = (
            f"{caller} has nothing to do: it shows no window (show_window=False) "
            "and writes no file. Pass output_path to save the video or "
            "show_window=True to watch it."
        )
        raise ValueError(message)
    if show_window:
        require_window_support(caller)
