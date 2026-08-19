"""Mosaic's own process imports no Ultralytics, and the runner imports no mosaic.

Ultralytics is AGPL-3.0. A mosaic that imports it is one work with it, and mosaic
could not then be offered under its own terms -- so Ultralytics runs in an
environment the user builds, and mosaic reaches it as a subprocess exchanging
JSON files. ``NOTICE`` says so, the runner program's own docstring says so, and
the installation documentation says so. Nothing checked it.

A convention cannot carry this, because the one environment where the breach
cannot be felt is a developer's: a machine with Ultralytics installed runs a
re-introduced ``from ultralytics import YOLO`` perfectly happily, every other
test passes, and the committed documents go on asserting a separation that has
quietly stopped being true. There is no failing run to notice, which is exactly
why the check has to be structural.

Structural, and **parsed rather than searched**. ``ultralytics_track/version.py``
explains at length why that directory is not named after the library, and writes
``from ultralytics import YOLO`` while explaining it. A text scan reports that
docstring as an import, and a guard that cries wolf on its own prose is a guard
somebody deletes.

The reverse direction has the same stake and is held the same way. The runner
program runs *inside* the Ultralytics environment, so a ``mosaic`` import there
would put mosaic's code into the same process as Ultralytics and recreate the
single combined work from the other side. ``mosaic_media`` is deliberately not
covered by that rule -- the runner reads video through it -- and the reason is
its Apache-2.0 license rather than its being separately packaged; see
:data:`MOSAIC_DISTRIBUTIONS_THE_ENVIRONMENT_MAY_DECLARE`.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from mosaic.tracking.common.toolenv import ToolNotFoundError, tool_invocation
from mosaic.tracking.ultralytics_track.run import ULTRALYTICS_ENV

from tests.helpers import inside_a_virtualenv

# Selected by CI's `tracking` job with `-m tracker` rather than by a filename
# list in the workflow, so a new file here is covered the day it lands.
pytestmark = pytest.mark.tracker

_ULTRALYTICS: Final = "ultralytics"
_MOSAIC: Final = "mosaic"

_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
_ENVIRONMENT_DIRECTORY: Final = (
    _REPO_ROOT / "src" / "mosaic" / "tracking" / "external" / "ultralytics-env"
)

RUNNER_PROGRAM: Final = "tracking/external/runner/ultralytics_runner.py"
"""The program that runs *inside* the Ultralytics environment.

It sits under ``src/mosaic/`` because that is where mosaic ships it from, not
because mosaic imports it: mosaic spawns it by path and never imports the module.
Every Ultralytics import in it is deferred into a function body, which is what
lets ``tests/test_ultralytics_wire_contract.py`` import the file in an
environment that has no Ultralytics at all.
"""

POSE_TRAINING_RESIDUAL: Final = (
    "tracking/pose_training/train.py",
    "tracking/pose_training/inference.py",
)
"""YOLO and POLO training, and single-model inference, still run in mosaic's own process.

Named here rather than tolerated by a broad exclusion, so the allowance shrinks
when they move out instead of quietly outliving the reason for it: the test below
fails on an allowed file that has *stopped* importing Ultralytics, which makes
the second half of that work impossible to leave half-done.
"""

ALLOWED_TO_IMPORT_ULTRALYTICS: Final = frozenset(
    {RUNNER_PROGRAM, *POSE_TRAINING_RESIDUAL}
)

_MENTIONS_WITHOUT_IMPORTING: Final = "tracking/ultralytics_track/version.py"
"""A file whose prose contains ``from ultralytics import YOLO`` and whose code
does not import it. The witness that this guard reads syntax rather than text."""

_REQUIREMENTS: Final = TypeAdapter(list[str])
_EXTRAS: Final = TypeAdapter(dict[str, list[str]])
"""Read a ``pyproject.toml`` table as the shape it is, validated where it is read
rather than narrowed by hand: ``tomllib`` returns an untyped document.
"""

EXTRAS_DECLARING_ULTRALYTICS: Final = frozenset({"pose", "polo"})
"""The extras that name an Ultralytics distribution in their own requirement list."""

EXTRAS_REACHING_ULTRALYTICS: Final = frozenset({"pose", "polo", "all", "recommended"})
"""Those two, plus the bundles that reach them by self-reference."""

MOSAIC_DISTRIBUTIONS_THE_ENVIRONMENT_MAY_DECLARE: Final = frozenset({"mosaic_media"})
"""``mosaic-media`` may sit beside Ultralytics because of its **license**.

Not because it is a separate distribution. Being separately packaged decides
nothing here -- ``mosaic-behavior`` is a separate distribution too, and it is the
one that must never appear, being the package whose separation from Ultralytics
this whole file is about. What makes ``mosaic-media`` safe is that it is
**Apache-2.0**, which is one-way compatible with AGPL-3.0: Apache-2.0 code may be
taken into an AGPL-covered work, and the combining is done by the user who builds
this environment, not by anyone shipping mosaic.

So the question a second entry has to answer is not "is it its own package" but
"are its terms ones the AGPL absorbs, and is the user still the only party
combining them". A mosaic-authored package under mosaic's own terms answers no
twice: mosaic is AGPL-3.0-or-later and its ``src/`` must stay commercially
relicensable, and neither survives being installed into an environment that
exists to keep AGPL-3.0 code out of mosaic's process.
"""


def _names_module(package: str, imported: str) -> bool:
    """Whether *imported* is *package* itself or something inside it.

    The dot matters in both directions. ``ultralytics_track`` and
    ``ultralytics_protocol`` are mosaic's own modules, and ``mosaic_media`` is a
    different distribution, so a prefix test without it would report all three.
    """
    return imported == package or imported.startswith(f"{package}.")


def _imports_of(source: Path, package: str) -> set[str]:
    """Every module of *package* that *source* imports, in every import form.

    Covers ``import ultralytics``, ``import ultralytics.cfg``, ``from ultralytics
    import YOLO`` and ``from ultralytics.trackers.track import TRACKER_MAP``
    alike, wherever they sit -- module scope or deferred inside a function body,
    which is where every one of the runner's live.

    A relative import is passed over: ``level`` above zero names a module inside
    the importing package, so ``from .ultralytics import x`` would be mosaic's
    own and not the library.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
        if isinstance(node, ast.Import):
            found.update(
                alias.name for alias in node.names if _names_module(package, alias.name)
            )
        elif isinstance(node, ast.ImportFrom):
            if (
                node.level == 0
                and node.module is not None
                and _names_module(package, node.module)
            ):
                found.add(node.module)
    return found


@dataclass(frozen=True, slots=True)
class ImportScan:
    """What one walk of the package tree read, and what it found."""

    scanned: frozenset[str]
    """Every file the walk parsed, as a path relative to the package root."""

    importers: dict[str, list[str]]
    """The files that import the package, mapped to the modules they name."""


def _scan(package: str) -> ImportScan:
    """Parse every mosaic source file and report which of them import *package*.

    Installed third-party code inside the tree is skipped: two directories under
    ``src/mosaic/`` are where a user builds an environment for an external tool,
    and the Ultralytics one holds Ultralytics itself. The predicate is
    ``tests.helpers.inside_a_virtualenv``, shared with the parquet-writer walk so
    the two cannot come to disagree about what mosaic's own source is -- and it
    is relative to the package root, because under a non-editable install an
    absolute ``site-packages`` test excludes the whole of mosaic and leaves the
    walk reading nothing.
    """
    import mosaic

    root = Path(mosaic.__file__).parent
    scanned: set[str] = set()
    importers: dict[str, list[str]] = {}
    for source in sorted(root.rglob("*.py")):
        if inside_a_virtualenv(source, root):
            continue
        relative = source.relative_to(root).as_posix()
        scanned.add(relative)
        imported = _imports_of(source, package)
        if imported:
            importers[relative] = sorted(imported)
    return ImportScan(scanned=frozenset(scanned), importers=importers)


@pytest.fixture(scope="module")
def ultralytics_scan() -> ImportScan:
    """One walk, shared: parsing the whole package tree twice buys nothing."""
    return _scan(_ULTRALYTICS)


def test_only_the_declared_files_import_ultralytics(
    ultralytics_scan: ImportScan,
) -> None:
    """Both directions, against a set written out by name.

    A file that imports Ultralytics and is not listed is the breach this whole
    arrangement exists to prevent. An allowed file that has *stopped* importing
    it is the other half: the allowance would otherwise outlive its reason, and
    the two ``pose_training`` modules are on their way out, so the day the second
    one moves is the day this list must shrink.
    """
    scanned = ultralytics_scan.scanned
    importers = ultralytics_scan.importers

    # A structural guard that can pass by scanning nothing is worse than no guard
    # at all: it reports a green invariant it never checked, and the virtualenv
    # exclusion is one over-broad predicate away from that. So the walk proves it
    # happened before its verdict is believed -- that it read a file that does
    # import Ultralytics, and one that does not.
    assert RUNNER_PROGRAM in scanned, (
        f"the walk never reached {RUNNER_PROGRAM}, so it scanned no part of the "
        f"code this guard is about; it read {len(scanned)} files"
    )
    assert _MENTIONS_WITHOUT_IMPORTING in scanned, (
        f"the walk never reached {_MENTIONS_WITHOUT_IMPORTING}, so it is not "
        f"covering mosaic's own tree; it read {len(scanned)} files"
    )
    import mosaic

    prose = (Path(mosaic.__file__).parent / _MENTIONS_WITHOUT_IMPORTING).read_text()
    assert f"from {_ULTRALYTICS} import" in prose, (
        f"{_MENTIONS_WITHOUT_IMPORTING} no longer names an Ultralytics import in "
        "its prose, so it no longer witnesses that this guard parses rather than "
        "greps. Point the witness at another file that does."
    )
    assert _MENTIONS_WITHOUT_IMPORTING not in importers

    forbidden = {
        name: modules
        for name, modules in importers.items()
        if name not in ALLOWED_TO_IMPORT_ULTRALYTICS
    }
    assert not forbidden, (
        f"these import Ultralytics into mosaic's own process: {forbidden}. "
        "Ultralytics is AGPL-3.0, so an import here makes mosaic one work with "
        "it. It runs in an environment of its own, reached through "
        "mosaic.tracking.ultralytics_track.run."
    )

    stopped = sorted(ALLOWED_TO_IMPORT_ULTRALYTICS - set(importers))
    assert not stopped, (
        f"{stopped} are allowed to import Ultralytics but no longer do. Remove "
        "them from ALLOWED_TO_IMPORT_ULTRALYTICS: an allowance nothing uses is "
        "one that will be reused by accident."
    )


def test_the_runner_program_takes_no_import_from_mosaic() -> None:
    """The reverse direction, and the one that needs nothing built.

    The runner and the protocol beside it run inside the Ultralytics environment,
    so a ``mosaic`` import in either would put mosaic's code in the same process
    as Ultralytics -- the same single combined work, arrived at from the other
    side. What they share with mosaic is a JSON contract and a command line, and
    that is what keeps them two programs.

    ``mosaic_media`` is what the environment declares and the runner reads video
    through, and is not what this refuses:
    :data:`MOSAIC_DISTRIBUTIONS_THE_ENVIRONMENT_MAY_DECLARE` says on what
    grounds.
    """
    directory = _REPO_ROOT / "src" / "mosaic" / "tracking" / "external" / "runner"
    sources = sorted(directory.glob("*.py"))
    assert [source.name for source in sources] == [
        "__init__.py",
        "ultralytics_protocol.py",
        "ultralytics_runner.py",
    ], f"the runner directory holds {[s.name for s in sources]}"

    importers = {
        source.name: sorted(imported)
        for source in sources
        if (imported := _imports_of(source, _MOSAIC))
    }
    assert not importers, (
        f"the runner program imports mosaic: {importers}. It runs inside the "
        "Ultralytics environment, so mosaic's code would be loaded in the same "
        "process as Ultralytics. The shared vocabulary belongs in "
        "ultralytics_protocol.py, which imports neither side."
    )


def _distributions(specifiers: list[str]) -> set[str]:
    """The distribution named by each requirement, canonically spelled.

    The name is the leading run of name characters; an extras bracket, a version
    specifier, a direct reference and an environment marker all follow it. Parsed
    here rather than with ``packaging``, which this project does not declare.
    """
    names: set[str] = set()
    for specifier in specifiers:
        matched = re.match(r"[A-Za-z0-9._-]+", specifier.strip())
        if matched is not None:
            names.add(matched.group(0).lower().replace("-", "_"))
    return names


def _requirements_reaching(extras: dict[str, list[str]], name: str) -> list[str]:
    """Every requirement *name* pulls in, following ``mosaic-behavior[...]`` edges.

    Bundles are declared by self-reference (``all = ["mosaic-behavior[pose,faiss]"]``)
    so they cannot drift from their parts. That is what defeats a scan of one
    extra's own list: ``all`` reaches Ultralytics and names nothing but itself.
    """
    seen: set[str] = set()
    reached: list[str] = []

    def walk(extra: str) -> None:
        if extra in seen:
            return
        seen.add(extra)
        for specifier in extras.get(extra, []):
            inner = re.fullmatch(
                r"mosaic-behavior\[([a-z0-9,\-]+)\]", specifier.strip()
            )
            if inner is None:
                reached.append(specifier)
                continue
            for referenced in inner.group(1).split(","):
                walk(referenced.strip())

    walk(name)
    return reached


def test_no_mosaic_install_declares_ultralytics_outside_the_pose_extras() -> None:
    """A plain install brings no Ultralytics, and exactly two extras still do.

    Declaring a distribution is not importing it, so this is a weaker property
    than the walk above -- but it is the surface an import becomes possible from,
    and it decides what a user gets without asking. A base dependency would put
    Ultralytics in every environment mosaic is installed into.

    ``pose`` and ``polo`` still declare it, for the training paths that have not
    moved yet, and the bundles that reach them still reach it. Pinned exactly
    rather than aspirationally, so retiring those extras is a change to this
    list and not a silent widening of it.
    """
    document = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    base = _REQUIREMENTS.validate_python(document["project"]["dependencies"])
    extras = _EXTRAS.validate_python(document["project"]["optional-dependencies"])

    assert _ULTRALYTICS not in _distributions(base), (
        "[project] dependencies declares Ultralytics, so every plain install of "
        "mosaic would carry it"
    )

    declaring = {
        name
        for name, specifiers in extras.items()
        if _ULTRALYTICS in _distributions(specifiers)
    }
    reaching = {
        name
        for name in extras
        if _ULTRALYTICS in _distributions(_requirements_reaching(extras, name))
    }
    assert declaring == EXTRAS_DECLARING_ULTRALYTICS
    assert reaching == EXTRAS_REACHING_ULTRALYTICS


def test_the_ultralytics_environment_declares_no_mosaic_distribution() -> None:
    """The environment that runs Ultralytics must not install mosaic beside it.

    Declaring ``mosaic-behavior`` there would make the import the runner is
    forbidden to write merely a matter of somebody writing it, and would put
    mosaic's code in a resolver's reach of the AGPL library on purpose.
    """
    document = tomllib.loads((_ENVIRONMENT_DIRECTORY / "pyproject.toml").read_text())
    declared = _distributions(
        _REQUIREMENTS.validate_python(document["project"]["dependencies"])
    )

    assert _ULTRALYTICS in declared, (
        f"{_ENVIRONMENT_DIRECTORY / 'pyproject.toml'} declares no Ultralytics; "
        "this test has lost its subject"
    )
    from_mosaic = {name for name in declared if name.startswith(_MOSAIC)}
    assert from_mosaic == MOSAIC_DISTRIBUTIONS_THE_ENVIRONMENT_MAY_DECLARE


def test_mosaic_is_not_importable_in_the_built_ultralytics_environment(
    tmp_path: Path,
) -> None:
    """The declaration above, confirmed against the environment a user built.

    A declaration is what the environment was asked for; this is what it holds. A
    transitive requirement, or a stray ``pip install`` into it, would put mosaic
    there without any file in the repository saying so.

    Run from an empty directory, because ``mosaic`` would otherwise be importable
    from whatever the working directory happens to hold.
    """
    try:
        invocation = tool_invocation(ULTRALYTICS_ENV, executable="python")
    except ToolNotFoundError as absent:
        pytest.skip(f"no Ultralytics environment resolves: {absent}")

    report = (
        "import importlib.util, sys;"
        "found = importlib.util.find_spec('mosaic');"
        "sys.stdout.write('present' if found else 'absent')"
    )
    completed = subprocess.run(
        [*invocation, "-c", report],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "absent", (
        "the Ultralytics environment can import mosaic. It runs an AGPL-3.0 "
        "library, and mosaic's separation from it is what lets mosaic be offered "
        "under its own terms."
    )
