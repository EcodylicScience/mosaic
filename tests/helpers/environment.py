"""What the surrounding machine provides, asked the same way everywhere.

The ffmpeg toolchain is the one capability the suite reaches for that is neither
a Python import nor part of the repository, so "is it here" gets one answer with
one shape. Before this, the same missing binary produced a clean skip in the
files that happened to request a guarding fixture and a bare ``FileNotFoundError``
naming a codec in the ones that did not.

``require_ffmpeg`` is a plain function rather than a fixture because the helpers
that need it are plain functions too -- ``add_media_sequence`` writes videos and
then indexes them, and a fixture cannot guard a call a test makes directly.
``pytest.skip`` works from anywhere inside a running test, so the two forms give
the same outcome.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# Both binaries. Probing shells out to ``ffprobe``; the transcode op and the
# raw-H.264 packet scan shell out to ``ffmpeg``, and it is the one that goes
# missing first.
FFMPEG_TOOLCHAIN = ("ffmpeg", "ffprobe")


def missing_ffmpeg_tools() -> tuple[str, ...]:
    """Which of the ffmpeg toolchain's binaries are not on ``PATH``."""
    return tuple(name for name in FFMPEG_TOOLCHAIN if shutil.which(name) is None)


def require_ffmpeg() -> None:
    """Skip the running test when the ffmpeg toolchain is not on ``PATH``.

    Under CI ``pytest_configure`` has already refused to start over a missing
    binary, so this is a local-only path: there, absence is a broken environment
    rather than a reason to run less.
    """
    missing = missing_ffmpeg_tools()
    if missing:
        pytest.skip(f"not on PATH: {', '.join(missing)}")


def sandbox_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> Path:
    """Point ``~`` at *home* for the running test, and work from a clean cwd.

    Both variables are set because the two platforms read different ones:
    ``posixpath`` consults ``HOME`` and ``ntpath`` consults ``USERPROFILE``, so
    setting one leaves the assertion vacuous on the other platform rather than
    failing there. ``HOMEDRIVE`` and ``HOMEPATH`` are cleared for the same
    reason -- ``ntpath`` falls back to them, and an inherited pair would win.

    The chdir is what makes an unexpanded path visible: ``Path("~/x").resolve()``
    lands under the working directory, so a test that does not move first would
    scatter a literal ``~`` into the checkout instead of into *tmp_path*, where
    :func:`assert_no_literal_tilde` can see it.
    """
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    monkeypatch.chdir(home)
    return home


def assert_no_literal_tilde(root: Path) -> None:
    """Fail if any path under *root* has a component that is literally ``~``.

    This is the assertion the tilde handling actually turns on. An unexpanded
    ``~`` never raises -- ``Path("~/x").resolve()`` is a perfectly good path --
    so the only evidence of the bug is the directory it leaves behind, and every
    boundary test ends by looking for one.
    """
    offenders = sorted(
        str(p) for p in root.rglob("*") if any(part == "~" for part in p.parts)
    )
    assert not offenders, (
        "a literal '~' directory was created, so a path was resolved before it "
        f"was expanded: {offenders}"
    )
