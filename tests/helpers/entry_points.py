"""Installed entry points for the tests of what a package declares to mosaic.

:func:`install_entry_points` makes a set of entry points the installed ones, in this
process, and records which groups the code under test asks for.
:func:`run_with_distribution` runs a fresh interpreter that sees one installed
distribution, ``x`` at version ``1.0``, which declares entry points and ships the
modules they name. :class:`CountingEntryPoint` counts its loads, and
:class:`FailingEntryPoint` raises when it loads.
"""

from __future__ import annotations

import os
import textwrap
from collections.abc import Mapping, Sequence
from importlib.metadata import EntryPoint
from pathlib import Path
from subprocess import CompletedProcess
from typing import ClassVar, override

import pytest

import mosaic.entry_points as entry_points_module
from tests.helpers.processes import run_child

__all__ = [
    "CountingEntryPoint",
    "FailingEntryPoint",
    "install_entry_points",
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
