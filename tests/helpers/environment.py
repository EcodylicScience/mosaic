"""What the surrounding machine provides, asked the same way everywhere.

The ffmpeg toolchain is the one capability the suite reaches for that is neither
a Python import nor part of the repository, so "is it here" gets one answer with
one shape. Before this, the same missing binary produced a clean skip in the
files that happened to request a guarding fixture and a bare ``FileNotFoundError``
naming a codec in the ones that did not.

The two predicates below are the same question pointed inwards, and they are two
questions rather than one. :func:`inside_a_virtualenv` asks whether a file is
installed third-party code. A user builds each external tool's environment
under one of the :data:`EXTERNAL_ENVIRONMENT_TREES`, and a walk of the package
tree then includes a whole site-packages.
:func:`runs_in_an_external_environment` asks whether a file is part of mosaic
but runs in one of those environments, where it may not import mosaic. A walk
wants whichever it means, and often both.

They live here rather than in the guards that use them because a second copy is
how two guards come to disagree, and because the substring spelling of the second
was written wrong in three of them before it was written right in one.

:data:`SEPARATED_TOOLS` names the tools that run in those environments, with each
tool's libraries, distribution, environments, programs and allowed importers. It
contains data only. An environment names its ``ToolEnv`` as a
``"module:attribute"`` string, which a check imports when it runs. This module
therefore does not import the feature library, which declares the kpms and VAME
environments.

:func:`isolate_tool_location` and :func:`fake_path_lookup` stand in for the
machine an external tool is located on: the variables its location ladder reads,
and ``$PATH``. They patch the ladder's seams: ``shutil.which``, the root of the
in-repository venv, and the user's ``~/.conda/envs``. A test then declares the
installed tools without patching the lookup itself.

``require_ffmpeg`` is a plain function rather than a fixture because the helpers
that need it are plain functions too -- ``add_media_sequence`` writes videos and
then indexes them, and a fixture cannot guard a call a test makes directly.
``pytest.skip`` works from anywhere inside a running test, so the two forms give
the same outcome.
"""

from __future__ import annotations

import importlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pytest

from mosaic.core.pipeline import toolenv
from mosaic.core.pipeline.toolenv import ToolEnv
from tests.helpers.paths import (
    FEATURE_RUNNERS,
    KPMS_PROTOCOL,
    KPMS_SERVER,
    RUNNER_WIRE,
    SOURCE_ROOT,
    ULTRALYTICS_PROTOCOL,
    ULTRALYTICS_RUNNER,
    VAME_PROTOCOL,
    VAME_RUNNER,
)

# Both binaries. Probing shells out to ``ffprobe``; the transcode op and the
# raw-H.264 packet scan shell out to ``ffmpeg``, and it is the one that goes
# missing first.
FFMPEG_TOOLCHAIN = ("ffmpeg", "ffprobe")

# Modules every CI job must have, installed through the `test` dependency group.
# `imgstore` gates tests behind ``pytest.importorskip``, so its absence
# presents as a skip rather than a failure -- a green CI that ran less than the
# workflow installed for. That is not hypothetical: the test step used to invoke
# `uv run pytest`, which re-synced the environment from `uv.lock` and pruned
# both extras before the first test ran.
#
# **A new optional dependency joins the install line and this tuple in the same
# change**, or its tests stop being evidence: adding it to the install alone
# leaves nothing to notice when it next vanishes.
#
# `pywt`, `h5py` and `tables` used to be here. They are base dependencies now, so
# requiring them of CI would assert something `pip install -e .` already
# guarantees -- and guarding them in a test is an error that
# ``test_optional_dependency_coverage`` reports, because a guard that can never
# fire masks a broken install.
#
# That rule is no longer only prose. ``test_optional_dependency_coverage``
# reads the suite's own ``importorskip`` calls out of its AST and fails when one
# names a module no tuple here requires -- which is how ``timm`` and ``tables``
# were found, both guarded and neither installed by any job, and how ``yaml`` was
# found being guarded despite being a *core* dependency, a guard that could never
# fire and would have masked a broken install.
CI_REQUIRED_MODULES = ("imgstore",)

# The same rule, scoped to one job. `torch` (via the `deep-learning` extra) is a
# ~200 MB wheel, so requiring it of every CI run would slow all of them down for
# tests only one job runs. It gets its own job instead, which sets
# MOSAIC_CI_IDENTITY=1 -- and inside that job the absence of torch is an error
# for exactly the reason above: `pytest.importorskip("torch")` would otherwise
# skip the T-Rex checkpoint tests green, and those are the only thing standing
# between a refactor and a silently randomly-initialised network inside T-Rex.
CI_IDENTITY_MODULES = ("torch", "timm")

# The same argument again for the tracking job, with one difference that changes
# its shape entirely: what that job installs is not a module. Ultralytics is
# AGPL-3.0 and mosaic never imports it -- it runs in an environment the user
# builds, reached as a subprocess -- so there is no import name to demand of
# *this* environment and deliberately no tuple below. Folding one in would claim
# a CI job installs Ultralytics here, and would then excuse an
# `importorskip("ultralytics")` that guards the wrong environment entirely.
#
# The rule itself is unchanged, pointed at the environment instead: under
# MOSAIC_CI_TRACKING an Ultralytics environment that does not resolve is a broken
# environment rather than a reason to skip. `test_ultralytics_preflight.py` skips
# its drift check when it cannot find one, and a job that builds the environment
# and then skips that check has proved nothing about the tracker tables it exists
# to compare.

# And again for the job that installs `feral`, which became possible at all only
# when FERAL started publishing to PyPI. `feral` is probed with
# ``importlib.util.find_spec`` rather than ``pytest.importorskip``, because two of
# the tests assert the *absence* path -- the ImportError naming the extra -- which
# ``importorskip`` cannot express. That probe is audited the same way; see
# ``test_optional_dependency_coverage``.
#
# Its own job for the usual reason and one more: FERAL pins its dependencies
# exactly, so the environment it produces is not the one the other jobs install.
CI_FERAL_MODULES = ("feral",)


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

    A user builds each external tool's environment under one of the
    :data:`EXTERNAL_ENVIRONMENT_TREES`, and a walk of the package tree then
    includes a whole site-packages. It finds hits outside mosaic's code:
    ``pandas``'s ``frame.py`` contains ``df.to_parquet("df.parquet.gzip", ...)``
    in a docstring example, which reads like a final path, and the Ultralytics
    environment's site-packages contains Ultralytics itself.

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


EnvironmentMarker = Literal["tracker", "slow"]
"""The marker that selects the CI job building an external environment."""


@dataclass(frozen=True, slots=True)
class CiJob:
    """A CI job that builds external environments.

    Attributes:
        marker: The marker that selects the job's tests.
        variable: The variable that the job sets. With it set,
            ``tests/conftest.py`` refuses to start when an environment that the
            job builds does not resolve.
    """

    marker: EnvironmentMarker
    variable: str


TRACKING_JOB: Final = CiJob("tracker", "MOSAIC_CI_TRACKING")
"""The ``tracking`` job of ``ci.yml``, which builds the Ultralytics and POLO environments."""

VAME_JOB: Final = CiJob("slow", "MOSAIC_CI_VAME")
"""The ``slow.yml`` workflow, which builds the VAME environment."""


@dataclass(frozen=True, slots=True)
class SeparatedEnvironment:
    """One environment that a separated tool runs in.

    Attributes:
        directory: The directory of its ``pyproject.toml``, relative to the
            package root.
        tool_env: The :class:`ToolEnv` that locates it, as ``"module:attribute"``,
            imported by :meth:`located_by`.
        built_by: The CI job that builds it, or ``None`` for an environment
            that only its user builds.
        located_in_directory: Whether the ``ToolEnv`` declares
            :attr:`directory` as its ``repo_venv``. A tool that is only an
            interpreter is located there. A tool with a console script is
            located on ``$PATH``.
    """

    directory: str
    tool_env: str
    built_by: CiJob | None
    located_in_directory: bool

    @property
    def source_directory(self) -> Path:
        """Returns :attr:`directory` in the repository checkout."""
        return SOURCE_ROOT / self.directory

    def located_by(self) -> ToolEnv:
        """Imports and returns the :class:`ToolEnv` that :attr:`tool_env` names."""
        module_name, _, attribute = self.tool_env.partition(":")
        found: object = getattr(importlib.import_module(module_name), attribute)
        if not isinstance(found, ToolEnv):
            kind = type(found).__name__
            message = f"{self.tool_env} names a {kind}. A ToolEnv is expected."
            raise TypeError(message)
        return found


@dataclass(frozen=True, slots=True)
class SeparatedTool:
    """A tool that runs in environments the user builds, outside mosaic's process.

    Paths are relative to the package root, and distributions are spelled as
    :func:`~tests.helpers.distribution_names` spells them.

    Attributes:
        name: The tool's name, for messages.
        license: Its license terms, for messages.
        libraries: The import names that only :attr:`allowed_importers` import.
        distribution: The distribution that installs them. An install of mosaic
            does not declare it.
        environments: The environments it runs in.
        programs: The files that run in its environments. They do not import
            mosaic.
        allowed_importers: The programs that import its libraries.
        mosaic_distributions: The mosaic-authored distributions that its
            environments may declare.
    """

    name: str
    license: str
    libraries: tuple[str, ...]
    distribution: str
    environments: tuple[SeparatedEnvironment, ...]
    programs: frozenset[str]
    allowed_importers: frozenset[str]
    mosaic_distributions: frozenset[str]


ULTRALYTICS_TOOL: Final = SeparatedTool(
    name="Ultralytics",
    license="AGPL-3.0",
    libraries=("ultralytics",),
    distribution="ultralytics",
    # Two environments, because POLO ships under the distribution name
    # `ultralytics` and so cannot share one with upstream. Both run the same
    # runner program.
    environments=(
        SeparatedEnvironment(
            "tracking/external/ultralytics-env",
            "mosaic.tracking.common.ultralytics_env:ULTRALYTICS_ENV",
            TRACKING_JOB,
            located_in_directory=False,
        ),
        SeparatedEnvironment(
            "tracking/external/polo-env",
            "mosaic.tracking.common.ultralytics_env:POLO_ENV",
            TRACKING_JOB,
            located_in_directory=False,
        ),
    ),
    programs=frozenset({ULTRALYTICS_RUNNER, ULTRALYTICS_PROTOCOL}),
    allowed_importers=frozenset({ULTRALYTICS_RUNNER}),
    # The environment may declare mosaic-media because it is Apache-2.0, a license
    # that AGPL-3.0 absorbs, and the user who builds the environment is the only
    # party that combines the two. A separate distribution is not enough.
    # mosaic-behavior is one too, and its `src/` must stay commercially
    # relicensable.
    mosaic_distributions=frozenset({"mosaic_media"}),
)
"""Ultralytics and the POLO fork: tracking, pose and point inference, and training."""

KPMS_TOOL: Final = SeparatedTool(
    name="keypoint-MoSeq",
    license="non-commercial research and academic use only",
    libraries=("keypoint_moseq", "jax_moseq"),
    distribution="keypoint_moseq",
    # A CI job does not build it. Its built-environment check skips in CI.
    environments=(
        SeparatedEnvironment(
            FEATURE_RUNNERS,
            "mosaic.behavior.feature_library.kpms:KPMS_ENV",
            None,
            located_in_directory=True,
        ),
    ),
    programs=frozenset({KPMS_SERVER, KPMS_PROTOCOL, RUNNER_WIRE}),
    allowed_importers=frozenset({KPMS_SERVER}),
    mosaic_distributions=frozenset(),
)
"""keypoint-MoSeq, behind the kpms feature."""

VAME_TOOL: Final = SeparatedTool(
    name="VAME",
    license="GPL-3.0",
    libraries=("vame",),
    distribution="vame_py",
    environments=(
        SeparatedEnvironment(
            f"{FEATURE_RUNNERS}/vame-env",
            "mosaic.behavior.feature_library.vame_feature:VAME_ENV",
            VAME_JOB,
            located_in_directory=True,
        ),
    ),
    programs=frozenset({VAME_RUNNER, VAME_PROTOCOL, RUNNER_WIRE}),
    allowed_importers=frozenset({VAME_RUNNER}),
    mosaic_distributions=frozenset(),
)
"""VAME, behind the vame feature."""

SEPARATED_TOOLS: Final = (ULTRALYTICS_TOOL, KPMS_TOOL, VAME_TOOL)
"""Every tool that runs in an environment the user builds, outside mosaic's process.

A separation check reads only the tools listed here.
"""


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


FAKE_CONDA = "/p/bin/conda"
"""A conda executable for a fake ``$PATH``, two levels deep for a reason.

``conda_invocation`` takes the conda installation as ``parent.parent`` of the
conda binary and probes the installation's ``envs`` and, for the env named
``base``, its ``bin``, on the real filesystem. A one-level fake such as
``/p/conda`` makes that installation ``/``, and the probe then reads the host's
``/bin``. ``/p`` exists nowhere. Every candidate misses, and the bare name is
used.
"""


def fake_path_lookup(monkeypatch: pytest.MonkeyPatch, found: Mapping[str, str]) -> None:
    """Answer every ``$PATH`` lookup a tool's location ladder makes from *found*.

    A name *found* does not hold is not on the fake ``$PATH``, so ``{}`` is a
    machine with nothing installed.
    """

    def which(name: str) -> str | None:
        return found.get(name)

    monkeypatch.setattr(toolenv.shutil, "which", which)


_NO_USER_ENVS = Path("/p/no-home/.conda/envs")
"""A ``~/.conda/envs`` for a test that isolates a tool's location.

It exists nowhere, like :data:`FAKE_CONDA`. A developer's ``~/.conda/envs`` may
contain an env with the name that a test places a tool in.
"""

_NO_REPO_VENV = Path("/p/no-venv")
"""A root of the in-repository venv for a test that isolates a tool's location.

It exists nowhere, like :data:`FAKE_CONDA`. That rung then refuses each tool
declaring :attr:`~mosaic.core.pipeline.toolenv.ToolEnv.repo_venv` by default,
whether or not the venv is built on the suite's machine.
"""


def isolate_tool_location(
    monkeypatch: pytest.MonkeyPatch,
    *envs: ToolEnv,
    found: Mapping[str, str],
    repo_venv: Path = _NO_REPO_VENV,
    user_envs: Path = _NO_USER_ENVS,
) -> None:
    """Unset what the location ladder reads for *envs*, and fake ``$PATH``.

    Each tool's conda-environment, executable and display variables are unset,
    and conda's own ``CONDA_EXE`` and ``CONDA_ENVS_DIRS``. A developer may export
    any of them, and a test of which rung wins would then report the machine
    rather than the code. ``$PATH`` lookups answer from *found*, as
    :func:`fake_path_lookup` does. The conda rung reads *user_envs* in place of
    ``~/.conda/envs``, and the default directory exists nowhere.

    The redirect is keyed on the declared directory. Every ``ToolEnv`` whose
    ``repo_venv`` equals one that *envs* declare reads *repo_venv* in place of
    that directory, a ``.placed`` variant and a ``ToolEnv`` outside *envs*
    included. Every other ``ToolEnv`` reads the ``repo_venv`` that it declares,
    or ``$PATH`` when it declares none. A test simulates a built venv by passing
    a directory that contains one. The default simulates an unbuilt venv.
    """
    for env in envs:
        for variable in (env.conda_env_var, env.bin_var, env.display_var):
            if variable:
                monkeypatch.delenv(variable, raising=False)
    for variable in ("CONDA_EXE", "CONDA_ENVS_DIRS"):
        monkeypatch.delenv(variable, raising=False)
    fake_path_lookup(monkeypatch, found)

    def resolve_user_envs() -> Path:
        return user_envs

    monkeypatch.setattr(toolenv, "_user_envs_dir", resolve_user_envs)
    declared = {env.repo_venv for env in envs if env.repo_venv is not None}
    if declared:

        def resolve_repo_venv(env: ToolEnv) -> Path | None:
            return repo_venv if env.repo_venv in declared else env.repo_venv

        monkeypatch.setattr(toolenv, "_repo_venv_root", resolve_repo_venv)


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
