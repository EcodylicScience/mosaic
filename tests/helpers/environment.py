"""What the surrounding machine provides, asked the same way everywhere.

The ffmpeg toolchain is the one capability the suite reaches for that is neither
a Python import nor part of the repository, so "is it here" gets one answer with
one shape. Before this, the same missing binary produced a clean skip in the
files that happened to request a guarding fixture and a bare ``FileNotFoundError``
naming a codec in the ones that did not.

The two predicates below are the same question pointed inwards, and they are two
questions rather than one. :func:`inside_a_virtualenv` asks whether a file is
installed third-party code -- two of mosaic's own directories are where a user
builds an environment for an external tool, so a walk of the package tree can
reach a whole site-packages. :func:`runs_in_an_external_environment` asks whether
a file is mosaic's own but runs on the far side of one of those boundaries, where
it may take no import from mosaic at all. A walk wants whichever it actually
means, and often both.

They live here rather than in the guards that use them because a second copy is
how two guards come to disagree, and because the substring spelling of the second
was written wrong in three of them before it was written right in one.

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


def inside_a_virtualenv(source: Path, root: Path) -> bool:
    """True when *source* is installed third-party code rather than mosaic's own.

    Two directories under ``src/mosaic/`` are where a user builds an environment
    for an external tool -- the keypoint-MoSeq runner's and the Ultralytics
    tracking runner's -- so a walk of the package tree can reach a whole
    site-packages. It finds hits that are not mosaic's code at all: ``pandas``'s
    own ``frame.py`` carries ``df.to_parquet("df.parquet.gzip", ...)`` in a
    docstring example, which reads exactly like a final path, and the Ultralytics
    environment's site-packages holds Ultralytics itself.

    Detected two ways because a virtualenv directory can be called anything: a
    ``site-packages`` component *below the package root*, or a ``pyvenv.cfg`` in a
    directory between *source* and *root*.

    Both tests are deliberately relative to *root*. Under a non-editable install
    the package root is itself ``.../site-packages/mosaic``, so an absolute
    ``"site-packages" in source.parts`` is true of every file in the walk --
    which excludes the whole of mosaic, leaves the caller asserting against an
    empty list, and turns the guard green having checked nothing. The property
    being tested is where a file sits inside the package, never where the
    package was installed.
    """
    if "site-packages" in source.relative_to(root).parts:
        return True
    for parent in source.parents:
        if parent == root:
            return False
        if (parent / "pyvenv.cfg").is_file():
            return True
    return False


EXTERNAL_ENVIRONMENT_TREES = (
    ("behavior", "feature_library", "external"),
    ("tracking", "external"),
)
"""Source that runs in an environment mosaic builds nothing of and imports
nothing from, given as path components under the package root.

Excluded from a structural guard for two reasons, and a guard should mean the
one it says. A rule about mosaic's own wiring is unreachable from these trees by
construction: a program under one of them may take no import from ``mosaic`` at
all -- that separation is what keeps mosaic and an AGPL-licensed or
non-commercial tool two programs -- so it cannot call the function the guard is
about. And each tree is where the user *builds* a virtualenv; that half is
:func:`inside_a_virtualenv`'s subject, and it holds wherever a virtualenv lands
rather than only in these two places.
"""


def runs_in_an_external_environment(source: Path, root: Path) -> bool:
    """True when *source* lies inside one of the external-environment trees.

    A path-*component* test, not a substring one. ``"tracking/external" in
    source.as_posix()`` also matches ``tracking/external_helpers.py`` and
    ``tracking/externals/``, and ``tracking/`` is exactly where the mosaic-side
    launcher for these programs lands -- so a substring match would silently
    exempt a mosaic module that must be held to the rule. The spelling that names
    one tree and misses the other has the same shape, and was the live version of
    this in three guards.
    """
    return any(
        root.joinpath(*tree) in source.parents for tree in EXTERNAL_ENVIRONMENT_TREES
    )


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
