"""Mosaic does not import a separated tool, and a tool's programs do not import mosaic.

Ultralytics, keypoint-MoSeq and VAME each run in an environment that the user
builds, and mosaic starts each as a subprocess. Ultralytics is AGPL-3.0, and a
mosaic that imports it is one work with it. keypoint-MoSeq is licensed for
non-commercial use only, a restriction that AGPL section 7 forbids adding to a
covered work. VAME is GPL-3.0, and an install of mosaic does not convey it.
``NOTICE`` states the terms. :data:`tests.helpers.SEPARATED_TOOLS` names each
tool's libraries, distribution, environments, programs and allowed importers, and
every check below runs over it.

A machine with a tool installed runs a reintroduced import without an error, and
every other test passes. The checks are therefore structural. They parse the
source instead of searching its text. ``ultralytics_track/version.py`` writes
``from ultralytics import YOLO`` in its prose, and a text search reports that
docstring as an import.

The reverse direction has the same stake. A program runs inside its tool's
environment, and a ``mosaic`` import there loads mosaic's code into the tool's
process. An environment declares a mosaic-authored distribution only where that
distribution's license permits the combination, as
:data:`tests.helpers.ULTRALYTICS_TOOL` records for ``mosaic-media``.

Mosaic and the Ultralytics runner share their vocabulary through the protocol
module beside the runner. The protocol module imports neither side, and declares
the request and the response of each subcommand that mosaic launches.

The checks that read files are unmarked and run in CI's ``test`` job. The check of
a built environment is marked with the job that builds it.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Final, TypedDict, get_args, get_type_hints

import pytest
from pydantic import TypeAdapter

import mosaic
from mosaic.core.pipeline.toolenv import (
    ToolNotFoundError,
    subprocess_env,
    tool_invocation,
)
from mosaic.tracking.common.ultralytics_env import RunnerSubcommand

from tests.helpers import (
    REPO_ROOT,
    SEPARATED_TOOLS,
    SOURCE_ROOT,
    ULTRALYTICS_PROTOCOL,
    EnvironmentMarker,
    ImportedName,
    SeparatedEnvironment,
    SeparatedTool,
    distribution_names,
    imports_of,
    inside_a_virtualenv,
    project_requirements,
    requirements_reaching,
    runs_in_an_external_environment,
)

_MOSAIC: Final = "mosaic"

_PACKAGE_ROOT: Final = Path(mosaic.__file__).parent
"""The root of the imported ``mosaic`` package, wherever it is installed."""

_MENTIONS_WITHOUT_IMPORTING: Final = "tracking/ultralytics_track/version.py"
"""A file that writes ``from ultralytics import YOLO`` in its prose.

Its code does not import Ultralytics. It witnesses that the walk reads syntax."""

_JOB_MARKS: Final[dict[EnvironmentMarker, pytest.MarkDecorator]] = {
    "tracker": pytest.mark.tracker,
    "slow": pytest.mark.slow,
}


class _LockedPackage(TypedDict):
    """The one field of a ``uv.lock`` package table that the checks read."""

    name: str


_LOCKED_PACKAGES: Final = TypeAdapter(list[_LockedPackage])


def _tool_id(tool: SeparatedTool) -> str:
    return tool.name


def _environment_parameters(*, built: bool) -> list[object]:
    """Each environment of each tool.

    A check of the *built* environment is marked with the CI job that builds it.
    """
    return [
        pytest.param(
            tool,
            environment,
            marks=(
                _JOB_MARKS[environment.built_by.marker]
                if built and environment.built_by is not None
                else ()
            ),
            id=environment.tool_env.rpartition(":")[2],
        )
        for tool in SEPARATED_TOOLS
        for environment in tool.environments
    ]


def _manifest(environment: SeparatedEnvironment) -> Path:
    return environment.source_directory / "pyproject.toml"


@pytest.fixture(scope="module")
def mosaic_imports() -> dict[str, list[ImportedName]]:
    """Returns the imports of each mosaic source file, by path under the package root.

    Installed third-party code in the tree is skipped. The external trees are
    where a user builds an environment, and one of them contains Ultralytics. The
    predicate is relative to the package root, because under a non-editable
    install an absolute ``site-packages`` test matches every file of mosaic, and
    the walk then reads none.
    """
    return {
        source.relative_to(_PACKAGE_ROOT).as_posix(): imports_of(
            source, scope="anywhere", relative=False
        )
        for source in sorted(_PACKAGE_ROOT.rglob("*.py"))
        if not inside_a_virtualenv(source, _PACKAGE_ROOT)
    }


@pytest.mark.parametrize("tool", SEPARATED_TOOLS, ids=_tool_id)
def test_only_the_declared_files_import_the_tool(
    tool: SeparatedTool, mosaic_imports: dict[str, list[ImportedName]]
) -> None:
    """Checks the importers of the tool against its allowed importers, both ways.

    A file that imports the tool and is not allowed is the breach. An allowed file
    can stop importing the tool. Its allowance then remains for a later file to
    reuse by accident.
    """
    unread = sorted(tool.allowed_importers - set(mosaic_imports))
    assert not unread, (
        f"the walk did not read {unread}, the files that import {tool.name}. It "
        f"read {len(mosaic_imports)} files."
    )

    importers = {
        name: sorted(
            {
                imported.module
                for imported in imports
                if any(imported.in_package(library) for library in tool.libraries)
            }
        )
        for name, imports in mosaic_imports.items()
    }
    importers = {name: modules for name, modules in importers.items() if modules}

    forbidden = {
        name: modules
        for name, modules in importers.items()
        if name not in tool.allowed_importers
    }
    assert not forbidden, (
        f"these files import {tool.name} ({tool.license}) into mosaic's process: "
        f"{forbidden}. {tool.name} runs in a separate environment, which mosaic "
        "starts as a subprocess."
    )

    stopped = sorted(tool.allowed_importers - set(importers))
    assert not stopped, (
        f"{stopped} may import {tool.name} and do not import it. Remove them from "
        "the tool's allowed importers. A later file can reuse an unused allowance "
        "by accident."
    )


def test_the_walk_parses_the_source(
    mosaic_imports: dict[str, list[ImportedName]],
) -> None:
    """A file that writes an import in its prose is not an importer."""
    assert _MENTIONS_WITHOUT_IMPORTING in mosaic_imports, (
        f"the walk did not read {_MENTIONS_WITHOUT_IMPORTING}, a file of mosaic's "
        f"tree. It read {len(mosaic_imports)} files."
    )
    prose = (SOURCE_ROOT / _MENTIONS_WITHOUT_IMPORTING).read_text()
    assert "from ultralytics import" in prose, (
        f"{_MENTIONS_WITHOUT_IMPORTING} does not write an Ultralytics import in "
        "its prose. It then fails to witness that the walk parses the source. "
        "Point the witness at another file that writes one."
    )
    assert not any(
        imported.in_package("ultralytics")
        for imported in mosaic_imports[_MENTIONS_WITHOUT_IMPORTING]
    )


@pytest.mark.parametrize("tool", SEPARATED_TOOLS, ids=_tool_id)
def test_a_program_runs_in_an_external_tree_and_imports_no_mosaic(
    tool: SeparatedTool,
) -> None:
    """Checks that each program lies in an external tree and does not import mosaic.

    A program shares a wire protocol and a command line with mosaic. The structural
    guards about mosaic's wiring skip the external-environment trees, and each
    program lies in one of them. A file allowed to import the tool is one of its
    programs, and it does not import mosaic either.
    """
    outside = sorted(
        program
        for program in tool.programs
        if not runs_in_an_external_environment(SOURCE_ROOT / program, SOURCE_ROOT)
    )
    assert not outside, (
        f"{outside} run in the {tool.name} environment and lie outside every "
        "external-environment tree"
    )
    assert tool.allowed_importers <= tool.programs

    importers = {
        program: sorted(
            imported.module
            for imported in imports_of(
                SOURCE_ROOT / program, scope="anywhere", relative=False
            )
            if imported.in_package(_MOSAIC)
        )
        for program in sorted(tool.programs)
    }
    importers = {program: modules for program, modules in importers.items() if modules}
    assert not importers, (
        f"these programs import mosaic: {importers}. They run in the {tool.name} "
        "environment, and the import loads mosaic's code into the process of "
        f"{tool.name}. Shared vocabulary belongs in the tool's protocol module, "
        "which imports neither side."
    )


def test_every_python_file_beside_a_program_is_a_program() -> None:
    """Every Python file beside a program is declared, for the check above to read.

    A program launched by its path does not run its package's ``__init__.py``.
    """
    programs = {program for tool in SEPARATED_TOOLS for program in tool.programs}
    directories = {PurePosixPath(program).parent for program in programs}
    present = {
        (directory / source.name).as_posix()
        for directory in directories
        for source in (SOURCE_ROOT / directory).glob("*.py")
        if source.name != "__init__.py"
    }

    assert present == programs, (
        f"undeclared: {sorted(present - programs)}; missing: "
        f"{sorted(programs - present)}"
    )


def test_each_ultralytics_subcommand_exchanges_models_of_the_protocol_module(
    runner_module: ModuleType,
) -> None:
    """Checks the runner's subcommands against mosaic's, and the models each exchanges.

    The runner's parser accepts the subcommands that ``RunnerSubcommand`` names,
    and no others. Each subcommand's request and response are models of the
    protocol module. Mosaic then builds each request and reads each response
    without importing the runner, and the runner validates each request without
    importing mosaic.
    """
    usage = re.search(r"\{([^}]*)\}", runner_module._parser().format_usage())
    assert usage is not None, "the runner's usage omits its subcommands"
    launched = set(get_args(RunnerSubcommand))
    assert set(usage.group(1).split(",")) == launched

    protocol = PurePosixPath(ULTRALYTICS_PROTOCOL).stem
    outside: dict[str, list[str]] = {}
    for subcommand in sorted(launched):
        answer = getattr(runner_module, f"run_{subcommand.replace('-', '_')}")
        hints = get_type_hints(answer)
        foreign = [
            model.__qualname__
            for model in (hints["request"], hints["return"])
            if model.__module__ != protocol
        ]
        if foreign:
            outside[subcommand] = foreign
    assert not outside, (
        f"these subcommands exchange models that {protocol} does not declare: "
        f"{outside}. A model declared in the runner is unreachable from mosaic "
        "without importing the runner, and one declared in mosaic is unreachable "
        "from the runner without importing mosaic."
    )


def _mosaic_declarations(distribution: str) -> list[str]:
    """Returns where mosaic's ``pyproject.toml`` and ``uv.lock`` name *distribution*.

    An extra is followed through its self-references, since ``all`` names only
    its parts. The lock lists every extra and group, transitively.
    """
    declared = project_requirements(REPO_ROOT / "pyproject.toml")
    places: list[str] = []
    if distribution in distribution_names(declared.dependencies):
        places.append("[project] dependencies")
    places.extend(
        f"the {name} extra"
        for name in declared.extras
        if distribution
        in distribution_names(requirements_reaching(declared.extras, name))
    )
    places.extend(
        f"the {name} dependency group"
        for name, requirements in declared.dependency_groups.items()
        if distribution in distribution_names(requirements)
    )
    locked = _LOCKED_PACKAGES.validate_python(
        tomllib.loads((REPO_ROOT / "uv.lock").read_text())["package"]
    )
    if distribution in distribution_names(package["name"] for package in locked):
        places.append("uv.lock")
    return places


@pytest.mark.parametrize("tool", SEPARATED_TOOLS, ids=_tool_id)
def test_no_mosaic_install_declares_the_tool(tool: SeparatedTool) -> None:
    """Mosaic's manifest and lock do not name the tool's distribution.

    An install of mosaic therefore conveys none of the separated tools, through
    its base dependencies, an extra, a dependency group or the lock.
    ``tool.license`` names each tool's terms, and they apply only to the
    environment that the user builds. A declaration does not import the tool,
    and this check is weaker than the walk above. A declaration decides what a
    user installs without asking.

    The detector first finds the distribution in each of the tool's environments,
    before its empty answer about mosaic is believed.
    """
    for environment in tool.environments:
        declared = distribution_names(
            project_requirements(_manifest(environment)).dependencies
        )
        assert tool.distribution in declared, (
            f"the detector did not find {tool.distribution} in "
            f"{_manifest(environment)}. Its empty answer about mosaic is then not "
            "evidence."
        )

    places = _mosaic_declarations(tool.distribution)
    assert not places, (
        f"mosaic declares {tool.distribution} in {places}. An install of mosaic "
        f"then contains {tool.name} ({tool.license})."
    )


@pytest.mark.parametrize(("tool", "environment"), _environment_parameters(built=False))
def test_an_environment_declares_only_its_permitted_mosaic_distributions(
    tool: SeparatedTool, environment: SeparatedEnvironment
) -> None:
    """An environment declares only the mosaic distributions that its tool permits.

    With ``mosaic-behavior`` declared there, one import line in a program combines
    mosaic with the tool.
    """
    declared = distribution_names(
        project_requirements(_manifest(environment)).dependencies
    )

    assert tool.distribution in declared, (
        f"{_manifest(environment)} does not declare {tool.distribution}. The test "
        "has lost its subject."
    )
    from_mosaic = {name for name in declared if name.startswith(_MOSAIC)}
    assert from_mosaic == tool.mosaic_distributions, (
        f"{_manifest(environment)} declares {sorted(from_mosaic)}, and the "
        f"{tool.name} environment permits {sorted(tool.mosaic_distributions)}"
    )


@pytest.mark.parametrize(("tool", "environment"), _environment_parameters(built=False))
def test_an_environment_names_the_tool_env_that_locates_it(
    tool: SeparatedTool, environment: SeparatedEnvironment
) -> None:
    """Resolves the environment's ``ToolEnv``, and compares its ``repo_venv``.

    The check of a built environment resolves the name too, and runs only in the
    job that builds the environment. This check runs on every push. The
    ``ToolEnv`` declares ``repo_venv`` if the registry records the environment
    as located in its directory, and only then. The check then compares
    ``repo_venv`` with that directory under the imported ``mosaic`` package.
    """
    located = environment.located_by()

    assert (located.repo_venv is not None) == environment.located_in_directory, (
        f"{environment.tool_env} declares repo_venv={located.repo_venv}, and the "
        f"registry records located_in_directory={environment.located_in_directory}"
    )
    if located.repo_venv is not None:
        expected = _PACKAGE_ROOT / environment.directory
        assert located.repo_venv.resolve() == expected.resolve(), (
            f"{environment.tool_env} locates the {tool.name} environment in "
            f"{located.repo_venv}, and the registry names {environment.directory}"
        )


@pytest.mark.parametrize(("tool", "environment"), _environment_parameters(built=True))
def test_mosaic_is_not_importable_in_a_built_environment(
    tool: SeparatedTool,
    environment: SeparatedEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks the declaration above against the environment that a user built.

    A transitive requirement, or a stray ``pip install`` into the environment,
    puts mosaic there without a file of the repository recording it.

    The interpreter runs from an empty directory, in the environment that every
    external tool receives. This process's ``PYTHONPATH`` names the directory
    that contains mosaic, as a checkout run with ``PYTHONPATH=src`` does, and
    that environment drops it. The check then reads only what the environment
    contains.
    """
    monkeypatch.setenv("PYTHONPATH", str(_PACKAGE_ROOT.parent))
    try:
        invocation = tool_invocation(environment.located_by(), executable="python")
    except ToolNotFoundError as absent:
        pytest.skip(f"{environment.directory} does not resolve: {absent}")

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
        env=subprocess_env(),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == "absent", (
        f"the {tool.name} environment can import mosaic, which loads mosaic's "
        f"code into the process of {tool.name} ({tool.license})"
    )
