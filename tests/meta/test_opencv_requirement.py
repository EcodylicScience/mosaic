"""Mosaic requires the headless OpenCV wheel, and no GUI one.

``opencv-python`` and ``opencv-python-headless`` ship one import package, so an
environment holding both has two builds of ``cv2`` with two vendored ffmpeg
libraries, and the process crashes at a different place on each run. Extras such
as ``lightning-action`` and ``movement`` require the headless wheel, so a GUI
requirement in mosaic's own dependencies puts both into every environment that
installs one of them. The headless wheel also needs no libGL, which a container
image does not carry.
"""

from __future__ import annotations

import tomllib

from tests.helpers import REPO_ROOT, distribution_names, project_requirements

PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_mosaic_requires_the_headless_opencv_wheel_with_a_bounded_range() -> None:
    """The requirement is the headless wheel, under the range mosaic is tested with."""
    declared = project_requirements(PYPROJECT).dependencies

    assert "opencv-python-headless>=4.7,<6" in declared
    assert "opencv_python" not in distribution_names(declared)


def test_no_extra_requires_the_gui_opencv_wheel() -> None:
    """An extra requiring the GUI wheel would bring it back beside the headless one."""
    extras = project_requirements(PYPROJECT).extras

    gui = {
        name
        for name, specifiers in extras.items()
        if "opencv_python" in distribution_names(specifiers)
    }
    assert gui == set()


def test_the_lock_resolves_mosaic_to_the_headless_opencv_wheel() -> None:
    """``uv.lock`` records what a locked install gets: the headless wheel, only."""
    packages = tomllib.loads((REPO_ROOT / "uv.lock").read_text())["package"]
    mosaic = next(
        package for package in packages if package["name"] == "mosaic-behavior"
    )

    requires = {dependency["name"] for dependency in mosaic["dependencies"]}
    assert "opencv-python-headless" in requires
    assert "opencv-python" not in requires
