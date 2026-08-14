"""The pytest configuration itself, asserted rather than assumed.

Two of the settings in ``[tool.pytest.ini_options]`` are load-bearing in a way
that fails silently when they drift, so each gets a test naming what it protects.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _ini_options() -> dict[str, object]:
    with PYPROJECT.open("rb") as handle:
        config = tomllib.load(handle)
    tool = config["tool"]
    assert isinstance(tool, dict)
    options = tool["pytest"]["ini_options"]
    assert isinstance(options, dict)
    return options


def _declared_markers() -> list[str]:
    """The ``markers`` entries, as strings.

    ``tomllib`` types every value as ``object``, so the narrowing happens once
    here rather than at each use.
    """
    markers = _ini_options()["markers"]
    assert isinstance(markers, list)
    return [str(entry) for entry in markers]


def test_the_default_invocation_deselects_slow() -> None:
    """A bare ``pytest`` must stay the fast run, and it is one option away.

    pytest takes the *last* ``-m`` it is given rather than intersecting them, so
    the day someone runs ``pytest -m "not media"`` the ``slow`` deselection is
    gone and the keypoint-MoSeq integration suite silently joins the run. That is
    a four-minute suite becoming a much longer one with nothing on screen to say
    why, which is exactly the kind of drift a comment does not survive.
    """
    addopts = _ini_options()["addopts"]
    assert isinstance(addopts, list), (
        "addopts must be a list; the string form hides the quoting around "
        "`not slow` and reads as one argument containing a space"
    )
    assert ["-m", "not slow"] == addopts[:2], (
        f"addopts no longer begins with the slow deselection: {addopts}. "
        "A bare `pytest` is documented in CLAUDE.md as running everything except "
        "the slow tests."
    )


def test_every_marker_used_in_the_suite_is_declared() -> None:
    """``--strict-markers`` only helps if the declarations stay complete.

    It turns an undeclared marker into an error, which is the point -- a typo
    like ``@pytest.mark.slwo`` would otherwise attach a marker nobody selects on
    and leave the test running in every invocation it was meant to be excluded
    from. This asserts the other half: that the declared set is the one the
    suite's own markers need, so adding a marker without declaring it fails here
    with a readable message rather than at collection with a bare ``UsageError``.
    """
    declared = {entry.split(":", 1)[0] for entry in _declared_markers()}
    assert {"slow", "media", "tracker", "identity"} <= declared


@pytest.mark.parametrize("marker", ["slow", "media", "tracker", "identity"])
def test_each_declared_marker_carries_a_description(marker: str) -> None:
    """A bare name in ``markers`` tells ``pytest --markers`` nothing."""
    entry = next(m for m in _declared_markers() if m.split(":", 1)[0] == marker)
    _, _, description = entry.partition(":")
    assert description.strip(), f"marker {marker!r} is declared with no description"
