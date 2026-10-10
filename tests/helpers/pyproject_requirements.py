"""Reads the requirements in a ``pyproject.toml``, and the distributions they name.

Requirements are parsed with a regular expression. This project does not declare
``packaging``, and importing it would make the suite depend on a transitive
dependency.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

_DISTRIBUTION: Final = re.compile(r"[A-Za-z0-9._-]+")
"""The leading run of name characters, which names the distribution.

An extras bracket, a version specifier, a direct reference and an environment
marker all follow it.
"""

SELF_REFERENCE: Final = re.compile(r"mosaic-behavior\[([a-z0-9,\-]+)\]")
"""``mosaic-behavior[x,y]``, a requirement on mosaic with extras.

The pattern also finds an install hint inside a message. It includes the
distribution, because a hint naming ``mosaic[...]`` names the wrong distribution
and is a defect apart from its extras.
"""

_REQUIREMENTS: Final = TypeAdapter(list[str])
_EXTRAS: Final = TypeAdapter(dict[str, list[str]])
_GROUPS: Final = TypeAdapter(dict[str, list[str | dict[str, str]]])
"""Validate a manifest's tables where they are read. ``tomllib`` returns an untyped
document. A dependency group lists requirements and ``include-group`` tables."""


@dataclass(frozen=True, slots=True)
class ProjectRequirements:
    """The requirements that a ``pyproject.toml`` declares.

    Attributes:
        dependencies: ``[project]`` ``dependencies``, the requirements of every
            install.
        extras: ``[project]`` ``optional-dependencies``, by extra. Empty when the
            table is absent.
        dependency_groups: ``[dependency-groups]``, by group. A group's
            ``include-group`` tables are left out. Empty when the table is
            absent.
    """

    dependencies: list[str]
    extras: dict[str, list[str]]
    dependency_groups: dict[str, list[str]]


def project_requirements(manifest: Path) -> ProjectRequirements:
    """Reads the requirements of the ``pyproject.toml`` at *manifest*."""
    document = tomllib.loads(manifest.read_text())
    project = document["project"]
    groups = _GROUPS.validate_python(document.get("dependency-groups", {}))
    return ProjectRequirements(
        dependencies=_REQUIREMENTS.validate_python(project["dependencies"]),
        extras=_EXTRAS.validate_python(project.get("optional-dependencies", {})),
        dependency_groups={
            name: [entry for entry in entries if isinstance(entry, str)]
            for name, entries in groups.items()
        },
    )


def distribution_names(specifiers: Iterable[str]) -> set[str]:
    """Returns the distribution that each requirement names, in one spelling.

    The spelling is lowercase, with ``_`` for ``-``. ``opencv-python`` and
    ``opencv_python`` name one distribution, and installed metadata and a
    ``pyproject.toml`` do not agree on which to write. A bare distribution name
    is a requirement too.
    """
    names: set[str] = set()
    for specifier in specifiers:
        matched = _DISTRIBUTION.match(specifier.strip())
        if matched is not None:
            names.add(matched.group(0).lower().replace("-", "_"))
    return names


def self_referenced_extras(specifier: str) -> list[str]:
    """Returns the extras that a ``mosaic-behavior[...]`` requirement names.

    Any other requirement names none.
    """
    matched = SELF_REFERENCE.fullmatch(specifier.strip())
    if matched is None:
        return []
    return [extra.strip() for extra in matched.group(1).split(",")]


def requirements_reaching(extras: Mapping[str, Sequence[str]], name: str) -> list[str]:
    """Returns every requirement the extra *name* installs, through self-references.

    Bundles are declared by self-reference (``all =
    ["mosaic-behavior[deep-learning,faiss]"]``) so that they cannot drift from
    their parts, and a scan of one extra's list therefore misses the requirements
    of its parts. A referenced extra that is not declared installs nothing.
    """
    seen: set[str] = set()
    reached: list[str] = []

    def walk(extra: str) -> None:
        if extra in seen:
            return
        seen.add(extra)
        for specifier in extras.get(extra, ()):
            referenced = self_referenced_extras(specifier)
            if not referenced:
                reached.append(specifier)
            for inner in referenced:
                walk(inner)

    walk(name)
    return reached
