"""Installed entry points for the tests of what a package declares to mosaic.

:func:`install_entry_points` makes a set of entry points the installed ones, in this
process, and records which groups the code under test asks for.
:func:`run_with_distribution` runs a fresh interpreter that sees one installed
distribution, ``x`` at version ``1.0``, which declares entry points and ships the
modules they name. :class:`CountingEntryPoint` counts its loads, and
:class:`FailingEntryPoint` raises when it loads.
"""

from __future__ import annotations

import importlib
import os
import sys
import textwrap
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import Distribution, EntryPoint
from pathlib import Path
from subprocess import CompletedProcess
from typing import ClassVar, override

import pytest

import mosaic.entry_points as entry_points_module
from mosaic.core.track_converter import (
    INSTALLED_STATE,
    TRACK_CONVERTERS,
    ensure_track_converters_registered,
)
from tests.helpers.processes import run_child

__all__ = [
    "AcmeEntryPoint",
    "CountingEntryPoint",
    "FailingEntryPoint",
    "install_entry_points",
    "installed_track_converters",
    "plugin_module",
    "run_with_distribution",
]


class CountingEntryPoint(EntryPoint):
    """An entry point that counts its loads in :attr:`loads`, across instances.

    A test sets ``loads`` to zero before it installs one.
    """

    loads: ClassVar[int] = 0

    @override
    def load(self) -> object:
        type(self).loads += 1
        return super().load()


class FailingEntryPoint(EntryPoint):
    """An entry point whose load raises ``ImportError("the module is broken")``."""

    @override
    def load(self) -> object:
        msg = "the module is broken"
        raise ImportError(msg)


class AcmeDistribution(Distribution):
    """The installed distribution ``acme-tracks`` at version 1.2.0."""

    @override
    def read_text(self, filename: str) -> str | None:
        if filename == "METADATA":
            return "Metadata-Version: 2.1\nName: acme-tracks\nVersion: 1.2.0\n"
        return None

    @override
    def locate_file(self, path: str | os.PathLike[str]) -> Path:
        return Path(path)


class AcmeEntryPoint(EntryPoint):
    """An entry point that the distribution ``acme-tracks`` 1.2.0 declares."""

    def __init__(self, name: str, value: str, group: str) -> None:
        super().__init__(name, value, group)
        vars(self).update(dist=AcmeDistribution())


def install_entry_points(
    monkeypatch: pytest.MonkeyPatch, *found: EntryPoint
) -> list[str]:
    """Make *found* the installed entry points, and record the groups asked for.

    Args:
        monkeypatch: The test's monkeypatch, which restores the lookup afterwards.
        *found: The entry points that exist, whatever their group.

    Returns:
        The list that each lookup appends its group to, in order.
    """
    groups: list[str] = []

    def lookup(*, group: str) -> tuple[EntryPoint, ...]:
        groups.append(group)
        return tuple(each for each in found if each.group == group)

    monkeypatch.setattr(entry_points_module, "entry_points", lookup)
    return groups


def run_with_distribution(
    tmp_path: Path,
    entry_points_txt: str,
    modules: Mapping[str, str],
    argv: Sequence[str],
) -> CompletedProcess[str]:
    """Run *argv* in a child process that sees a distribution ``x`` of version 1.0.

    Args:
        tmp_path: A directory that the distribution's metadata and modules are
            written into and that the child finds on ``PYTHONPATH``.
        entry_points_txt: The content of the distribution's ``entry_points.txt``.
        modules: The source of each module the entry points name, by module name.
        argv: The command to run, its first item usually ``sys.executable``.
    """
    dist_info = tmp_path / "x.dist-info"
    dist_info.mkdir()
    _ = (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: x\nVersion: 1.0\n", encoding="utf-8"
    )
    _ = (dist_info / "entry_points.txt").write_text(entry_points_txt, encoding="utf-8")
    for name, source in modules.items():
        _ = (tmp_path / f"{name}.py").write_text(
            textwrap.dedent(source), encoding="utf-8"
        )
    inherited = os.environ.get("PYTHONPATH")
    path = os.pathsep.join([str(tmp_path), inherited] if inherited else [str(tmp_path)])
    return run_child(argv, env={**os.environ, "PYTHONPATH": path})


@contextmanager
def installed_track_converters(
    monkeypatch: pytest.MonkeyPatch, *found: EntryPoint
) -> Generator[list[str]]:
    """Make *found* the installed entry points, to be loaded by the next ``ensure``.

    The group of the installed packages loads before the block, and the load of
    *found* runs inside it. On exit the registered converters, their origins and the
    recorded refusals are what they were on entry, and no further load runs.

    Args:
        monkeypatch: The test's monkeypatch, which restores the lookup afterwards.
        *found: The entry points that exist, whatever their group.

    Yields:
        The groups that each lookup asked for, in order.
    """
    ensure_track_converters_registered()
    groups = install_entry_points(monkeypatch, *found)
    registered = dict(TRACK_CONVERTERS)
    origins = dict(INSTALLED_STATE.origins)
    refusals = dict(INSTALLED_STATE.refusals)
    monkeypatch.setattr(INSTALLED_STATE, "loaded", False)
    try:
        yield groups
    finally:
        TRACK_CONVERTERS.clear()
        TRACK_CONVERTERS.update(registered)
        INSTALLED_STATE.origins.clear()
        INSTALLED_STATE.origins.update(origins)
        INSTALLED_STATE.refusals.clear()
        INSTALLED_STATE.refusals.update(refusals)
        INSTALLED_STATE.loaded = True


def plugin_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> str:
    """Write *source* as a module of a name unique to the call and return its name.

    The module lies under *tmp_path*, which the test's imports find first, so its
    first import is the one that runs *source*.
    """
    name = f"acme_plugin_{uuid.uuid4().hex}"
    _ = (tmp_path / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    importlib.invalidate_caches()
    return name
