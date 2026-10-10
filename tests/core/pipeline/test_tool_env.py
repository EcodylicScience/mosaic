"""The shared tool-placement primitives, on their own terms.

``tests/tracking/trex/test_trex_invocation.py``,
``tests/tracking/sleap/test_sleap_invocation.py`` and
``tests/tracking/litpose/test_litpose_invocation.py`` cover the five-step ladder
once per tool, and prove that each tool still resolves the way it did.
This covers the parts of the shared machinery that no single tool exercises
fully: the environment overlay, an explicitly chosen matplotlib backend, the
in-repository interpreter rung, the interpreter prerequisite, the conda env
search, and the two exception shapes.

The argv-summary test logs through a real tracker and is in
``tests/tracking/common/test_running_argv_summary.py``.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias

import pytest

from mosaic.core.pipeline.toolenv import (
    ToolEnv,
    ToolExitError,
    ToolInvocation,
    ToolNotFoundError,
    locate_tool,
    missing_interpreter,
    missing_output_error,
    subprocess_env,
    tool_invocation,
)
from tests.helpers import FAKE_CONDA, fake_path_lookup, isolate_tool_location


class _FakeNotFound(ToolNotFoundError):
    default_message = "install the fake tool"


class _FakeExit(ToolExitError):
    tool_name = "Fake"


class _TerseExit(ToolExitError):
    tool_name = "Terse"
    head = 2


_DIRECT = ToolEnv(
    tool="Fake",
    conda_env_var="MOSAIC_FAKE_CONDA_ENV",
    bin_var="MOSAIC_FAKE_BIN",
    bin_mode="direct",
    not_found=_FakeNotFound,
)
_SIBLING = ToolEnv(
    tool="Fake",
    conda_env_var="MOSAIC_FAKE_CONDA_ENV",
    bin_var="MOSAIC_FAKE_BIN",
    bin_mode="sibling",
    not_found=_FakeNotFound,
)
_LOCATED = ToolEnv(
    tool="Fake",
    conda_env_var="MOSAIC_FAKE_CONDA_ENV",
    bin_var="MOSAIC_FAKE_BIN",
    bin_mode="sibling",
    not_found=_FakeNotFound,
    locator="finder",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_tool_location(
        monkeypatch,
        _DIRECT,
        found={
            "runme": "/p/bin/runme",
            "finder": "/p/bin/finder",
            "conda": FAKE_CONDA,
        },
    )


# --- the two bin modes, which is the whole of what differs between tools ----


def test_direct_mode_treats_an_explicit_bin_as_the_executable() -> None:
    got = tool_invocation(
        _DIRECT.placed(bin_path="/x/somewhere-else"), executable="runme"
    )

    assert got == ["/x/somewhere-else"]


def test_sibling_mode_resolves_the_executable_beside_an_explicit_bin() -> None:
    got = tool_invocation(_SIBLING.placed(bin_path="/x/bin/other"), executable="runme")

    assert got == ["/x/bin/runme"]


def test_a_bare_sibling_name_is_looked_up_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MOSAIC_SLEAP_BIN=sleap-convert`` names the environment ``$PATH`` finds it in.

    Read as a path, a bare name has no directory, and the executable beside it
    would be a bare ``python`` that ``$PATH`` resolves to another interpreter.
    """
    monkeypatch.setenv("MOSAIC_FAKE_BIN", "finder")

    assert tool_invocation(_SIBLING, executable="python") == ["/p/bin/python"]


def test_a_bare_sibling_name_is_resolved_through_its_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare name that ``$PATH`` finds as a link names its target's environment."""
    env_bin = tmp_path.resolve() / "tools" / "fake" / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / "finder").touch()
    linked = tmp_path / "local-bin" / "finder"
    linked.parent.mkdir()
    linked.symlink_to(env_bin / "finder")
    fake_path_lookup(monkeypatch, {"finder": str(linked)})

    got = tool_invocation(_SIBLING.placed(bin_path="finder"), executable="python")

    assert got == [str(env_bin / "python")]


def test_a_bare_sibling_name_not_on_path_is_refused_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_path_lookup(monkeypatch, {})
    monkeypatch.setenv("MOSAIC_FAKE_BIN", "finder")

    with pytest.raises(
        _FakeNotFound, match="'finder', the value of MOSAIC_FAKE_BIN, is not on"
    ):
        tool_invocation(_SIBLING, executable="python")


def test_a_bare_sibling_name_not_on_path_is_refused_naming_the_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller's argument overrides the variable, so the variable is not named."""
    fake_path_lookup(monkeypatch, {})
    monkeypatch.setenv("MOSAIC_FAKE_BIN", "/elsewhere/finder")

    with pytest.raises(_FakeNotFound) as refused:
        tool_invocation(_SIBLING.placed(bin_path="finder"), executable="python")

    message = str(refused.value)
    assert "'finder', the binary this run was placed with, is not on" in message
    assert "MOSAIC_FAKE_BIN" not in message


def test_a_bare_direct_name_is_left_for_path_to_find() -> None:
    """``MOSAIC_TREX_BIN=trex`` is the executable, which ``$PATH`` finds at launch."""
    got = tool_invocation(_DIRECT.placed(bin_path="trex"), executable="trex")

    assert got == ["trex"]


def test_without_a_locator_the_executable_is_looked_up_directly() -> None:
    assert tool_invocation(_SIBLING, executable="runme") == ["/p/bin/runme"]


def test_a_locator_is_looked_up_and_the_executable_resolved_beside_it() -> None:
    """Lightning Pose's case: find ``litpose`` in order to find its ``python``.

    Looking up the executable itself would find the caller's own interpreter.
    """
    assert tool_invocation(_LOCATED, executable="python") == ["/p/bin/python"]


def test_a_linked_locator_is_resolved_to_the_environment_it_belongs_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``uv tool install`` links its scripts into ``~/.local/bin``.

    What sits beside the link is the user's other tools, not the environment,
    and the scripts a tool install does not link -- ``sleap-nn``, the
    environment's ``python`` -- are only beside the target.
    """
    env_bin = tmp_path.resolve() / "tools" / "fake" / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / "finder").touch()
    linked = tmp_path / "local-bin" / "finder"
    linked.parent.mkdir()
    linked.symlink_to(env_bin / "finder")
    fake_path_lookup(monkeypatch, {"finder": str(linked)})

    assert tool_invocation(_LOCATED, executable="python") == [str(env_bin / "python")]


# --- the in-repository interpreter rung ------------------------------------


def _with_repo_venv(root: Path) -> ToolEnv:
    """``_DIRECT``, but declaring an in-repository venv under *root*.

    Shares ``_DIRECT``'s variable names, so ``clean_env`` already unsets them
    for every test below.
    """
    return replace(_DIRECT, repo_venv=root)


def _build_in_repo_python(root: Path, *, win32: bool = False) -> Path:
    """Create the file the in-repository rung resolves to under *root*."""
    venv = root / ".venv"
    interpreter = venv / "Scripts" / "python.exe" if win32 else venv / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    return interpreter


def test_a_tool_without_the_in_repo_field_still_reaches_path() -> None:
    """``_DIRECT`` declares ``repo_venv=None`` and resolves from ``$PATH``."""
    assert tool_invocation(_DIRECT, executable="runme") == ["/p/bin/runme"]


def test_the_in_repo_rung_is_reached_once_every_other_rung_is_unset(
    tmp_path: Path,
) -> None:
    env = _with_repo_venv(tmp_path)
    interpreter = _build_in_repo_python(tmp_path)

    assert tool_invocation(env, executable="python") == [str(interpreter)]


def _conda_env_with(tmp_path: Path, env_name: str, executable: str) -> str:
    """Build a conda installation under *tmp_path* with an env that has *executable*.

    The installation is ``<tmp_path>/root``. Returns the path of its ``conda``.
    """
    conda = tmp_path / "root" / "bin" / "conda"
    conda.parent.mkdir(parents=True)
    conda.touch()
    env_bin = tmp_path / "root" / "envs" / env_name / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / executable).touch()
    return str(conda)


@dataclass(frozen=True)
class _Machine:
    """The interpreters that rungs 1 to 4 find for a tool declaring ``repo_venv``."""

    explicit: Path
    conda: str
    conda_python: Path


def _machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Machine:
    """Build an explicit interpreter and a conda env ``toolenv`` with a ``python``."""
    explicit = tmp_path / "explicit-python"
    explicit.touch()
    conda = _conda_env_with(tmp_path, "toolenv", "python")
    fake_path_lookup(monkeypatch, {"conda": conda})
    conda_python = tmp_path / "root" / "envs" / "toolenv" / "bin" / "python"
    return _Machine(explicit=explicit, conda=conda, conda_python=conda_python)


def _conda_argv(machine: _Machine) -> list[str]:
    return [
        machine.conda,
        "run",
        "--no-capture-output",
        "-n",
        "toolenv",
        str(machine.conda_python),
    ]


def _via_placed_conda_env(
    env: ToolEnv, machine: _Machine, monkeypatch: pytest.MonkeyPatch
) -> tuple[ToolEnv, list[str]]:
    return env.placed(conda_env="toolenv"), _conda_argv(machine)


def _via_placed_bin_path(
    env: ToolEnv, machine: _Machine, monkeypatch: pytest.MonkeyPatch
) -> tuple[ToolEnv, list[str]]:
    return env.placed(bin_path=machine.explicit), [str(machine.explicit)]


def _via_conda_env_var(
    env: ToolEnv, machine: _Machine, monkeypatch: pytest.MonkeyPatch
) -> tuple[ToolEnv, list[str]]:
    monkeypatch.setenv("MOSAIC_FAKE_CONDA_ENV", "toolenv")
    return env, _conda_argv(machine)


def _via_bin_var(
    env: ToolEnv, machine: _Machine, monkeypatch: pytest.MonkeyPatch
) -> tuple[ToolEnv, list[str]]:
    monkeypatch.setenv("MOSAIC_FAKE_BIN", str(machine.explicit))
    return env, [str(machine.explicit)]


_Place: TypeAlias = Callable[
    [ToolEnv, _Machine, pytest.MonkeyPatch], tuple[ToolEnv, list[str]]
]


@pytest.mark.parametrize(
    "place",
    [_via_placed_conda_env, _via_placed_bin_path, _via_conda_env_var, _via_bin_var],
    ids=["placed-conda-env", "placed-bin-path", "conda-env-var", "bin-var"],
)
def test_every_earlier_rung_wins_over_a_built_in_repo_venv(
    place: _Place,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rungs 1 to 4 all take precedence over a venv that the in-repo rung would find."""
    _build_in_repo_python(tmp_path)
    env = _with_repo_venv(tmp_path)
    machine = _machine(tmp_path, monkeypatch)

    placed, expected = place(env, machine, monkeypatch)

    assert tool_invocation(placed, executable="python") == expected


@pytest.mark.parametrize(
    ("place", "source"),
    [
        (
            _via_placed_conda_env,
            "the conda environment 'toolenv' that this run was placed in",
        ),
        (_via_placed_bin_path, "the binary this run was placed with"),
        (
            _via_conda_env_var,
            "the conda environment 'toolenv' named by MOSAIC_FAKE_CONDA_ENV",
        ),
        (_via_bin_var, "the value of MOSAIC_FAKE_BIN"),
    ],
    ids=["placed-conda-env", "placed-bin-path", "conda-env-var", "bin-var"],
)
def test_an_invocation_names_the_rung_that_it_came_from(
    place: _Place,
    source: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A message about a tool that cannot be started names where it was located."""
    env = _with_repo_venv(tmp_path)
    machine = _machine(tmp_path, monkeypatch)

    placed, expected = place(env, machine, monkeypatch)

    assert locate_tool(placed, executable="python") == ToolInvocation(
        argv=expected, source=source
    )
    assert tool_invocation(placed, executable="python") == expected


def test_the_in_repo_rung_names_its_directory(tmp_path: Path) -> None:
    env = _with_repo_venv(tmp_path)
    interpreter = _build_in_repo_python(tmp_path)

    assert locate_tool(env, executable="python") == ToolInvocation(
        argv=[str(interpreter)],
        source=f"the in-repository environment under {tmp_path}",
    )


def test_the_path_rung_names_the_name_that_it_looked_up() -> None:
    """A locator is the name looked up, whatever the executable is."""
    assert locate_tool(_DIRECT, executable="runme") == ToolInvocation(
        argv=["/p/bin/runme"], source="the $PATH lookup of 'runme'"
    )
    assert locate_tool(_LOCATED, executable="python") == ToolInvocation(
        argv=["/p/bin/python"], source="the $PATH lookup of 'finder'"
    )


def test_the_in_repo_rung_resolves_the_windows_layout_on_win32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    env = _with_repo_venv(tmp_path)
    interpreter = _build_in_repo_python(tmp_path, win32=True)

    assert tool_invocation(env, executable="python") == [str(interpreter)]


def test_a_missing_placed_bin_path_is_refused_naming_the_placement(
    tmp_path: Path,
) -> None:
    env = _with_repo_venv(tmp_path)
    missing = tmp_path / "missing-python"

    with pytest.raises(_FakeNotFound) as refused:
        tool_invocation(env.placed(bin_path=missing), executable="python")

    message = str(refused.value)
    assert str(missing) in message
    assert "the binary this run was placed with" in message


def test_a_missing_bin_var_is_refused_naming_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _with_repo_venv(tmp_path)
    missing = tmp_path / "missing-python"
    monkeypatch.setenv("MOSAIC_FAKE_BIN", str(missing))

    with pytest.raises(_FakeNotFound) as refused:
        tool_invocation(env, executable="python")

    message = str(refused.value)
    assert str(missing) in message
    assert "MOSAIC_FAKE_BIN" in message


def test_a_missing_sibling_executable_is_refused_in_a_repo_venv_tool(
    tmp_path: Path,
) -> None:
    """``_from_bin`` applies the same file check in sibling mode.

    Every tool that declares ``repo_venv`` has a single interpreter and is placed
    in direct mode. The check is in shared code and applies to sibling mode too.
    """
    env = replace(_SIBLING, repo_venv=tmp_path)
    placed_bin = tmp_path / "placed-bin" / "other"
    placed_bin.parent.mkdir()
    placed_bin.touch()

    with pytest.raises(_FakeNotFound) as refused:
        tool_invocation(env.placed(bin_path=str(placed_bin)), executable="python")

    message = str(refused.value)
    assert str(placed_bin.parent / "python") in message
    assert "the binary this run was placed with" in message


def test_a_missing_in_repo_venv_is_refused_naming_the_resolved_path(
    tmp_path: Path,
) -> None:
    env = _with_repo_venv(tmp_path)
    expected = tmp_path / ".venv" / "bin" / "python"

    with pytest.raises(_FakeNotFound) as refused:
        tool_invocation(env, executable="python")

    message = str(refused.value)
    assert str(expected) in message
    assert f"the in-repository environment under {tmp_path}" in message


def test_a_missing_in_repo_venv_also_carries_the_tools_own_install_help(
    tmp_path: Path,
) -> None:
    """The refusal ends with the tool's install hint as well as the path."""
    env = _with_repo_venv(tmp_path)

    with pytest.raises(_FakeNotFound) as refused:
        tool_invocation(env, executable="python")

    assert _FakeNotFound.default_message in str(refused.value)


def test_isolating_an_in_repo_venv_tool_does_not_redirect_one_with_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolating a repo_venv tool leaves one that declares none on ``$PATH``."""
    with_venv = _with_repo_venv(tmp_path)
    isolate_tool_location(
        monkeypatch, with_venv, _DIRECT, found={"runme": "/p/bin/runme"}
    )

    assert tool_invocation(_DIRECT, executable="runme") == ["/p/bin/runme"]


# --- the interpreter prerequisite -------------------------------------------


def test_an_unlocated_interpreter_is_answered_with_the_tools_install_help(
    tmp_path: Path,
) -> None:
    """No rung locates ``python``, and the in-repository environment is not built."""
    missing = missing_interpreter(_with_repo_venv(tmp_path))

    assert missing == _FakeNotFound.default_message


def test_a_located_interpreter_is_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[object] = []

    def refuse_to_start(*args: object, **kwargs: object) -> None:
        started.append(args)
        msg = "locating the interpreter started a process"
        raise AssertionError(msg)

    monkeypatch.setattr(subprocess, "Popen", refuse_to_start)
    _ = _build_in_repo_python(tmp_path)

    assert missing_interpreter(_with_repo_venv(tmp_path)) is None
    assert started == []


# --- the environment ------------------------------------------------------


def test_an_inherited_notebook_backend_is_neutralized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")

    assert subprocess_env()["MPLBACKEND"] == "Agg"


def test_an_explicitly_chosen_backend_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the ``module://`` form is a notebook artifact worth overriding."""
    monkeypatch.setenv("MPLBACKEND", "TkAgg")

    assert subprocess_env()["MPLBACKEND"] == "TkAgg"


def test_no_backend_stays_no_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MPLBACKEND", raising=False)

    assert "MPLBACKEND" not in subprocess_env()


def test_the_overlay_wins_over_the_inherited_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TREx's DISPLAY: a per-call value must beat whatever the caller exported."""
    monkeypatch.setenv("DISPLAY", ":0")

    assert subprocess_env({"DISPLAY": ":99"})["DISPLAY"] == ":99"


# --- which install a named conda env actually runs -------------------------


def test_a_named_conda_env_runs_its_own_executable_not_one_earlier_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``conda run`` resolves a bare name against ``$PATH``, and does not put the
    environment's ``bin`` first. A same-named script inherited from somewhere
    earlier -- a ``uv tool install`` in ``~/.local/bin`` is the common one --
    would answer instead, so the pinned environment would silently not be the one
    that ran. Naming the file settles it.
    """
    conda = _conda_env_with(tmp_path, "toolenv", "runme")
    fake_path_lookup(monkeypatch, {"conda": conda})

    got = tool_invocation(_DIRECT.placed(conda_env="toolenv"), executable="runme")

    assert got == [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        "toolenv",
        str(tmp_path / "root" / "envs" / "toolenv" / "bin" / "runme"),
    ]


def test_an_executable_absent_from_the_env_falls_back_to_the_bare_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No candidate holds it, so there is nothing to name and conda resolves it."""
    conda = _conda_env_with(tmp_path, "toolenv", "runme")
    fake_path_lookup(monkeypatch, {"conda": conda})

    got = tool_invocation(_DIRECT.placed(conda_env="toolenv"), executable="absent-here")

    assert got[-1] == "absent-here"


def test_conda_envs_dirs_locates_an_env_outside_the_base_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conda = _conda_env_with(tmp_path, "unused", "runme")
    elsewhere = tmp_path / "elsewhere" / "toolenv" / "bin"
    elsewhere.mkdir(parents=True)
    (elsewhere / "runme").touch()
    fake_path_lookup(monkeypatch, {"conda": conda})
    monkeypatch.setenv("CONDA_ENVS_DIRS", str(tmp_path / "elsewhere"))

    got = tool_invocation(_DIRECT.placed(conda_env="toolenv"), executable="runme")

    assert got[-1] == str(elsewhere / "runme")


def test_the_users_env_directory_locates_an_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conda keeps a user's envs in ``~/.conda/envs``, after the installation's."""
    conda = _conda_env_with(tmp_path, "unused", "runme")
    user_envs = tmp_path / "home" / ".conda" / "envs"
    runme = user_envs / "toolenv" / "bin" / "runme"
    runme.parent.mkdir(parents=True)
    runme.touch()
    isolate_tool_location(
        monkeypatch, _DIRECT, found={"conda": conda}, user_envs=user_envs
    )

    got = tool_invocation(_DIRECT.placed(conda_env="toolenv"), executable="runme")

    assert got[-1] == str(runme)


def test_the_env_named_base_is_the_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conda = _conda_env_with(tmp_path, "unused", "runme")
    runme = tmp_path / "root" / "bin" / "runme"
    runme.touch()
    fake_path_lookup(monkeypatch, {"conda": conda})

    got = tool_invocation(_DIRECT.placed(conda_env="base"), executable="runme")

    assert got[-1] == str(runme)


def test_an_absent_env_does_not_run_the_installations_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installation's ``bin/python`` belongs to the env named ``base`` alone."""
    conda = _conda_env_with(tmp_path, "unused", "python")
    (tmp_path / "root" / "bin" / "python").touch()
    fake_path_lookup(monkeypatch, {"conda": conda})

    got = tool_invocation(_DIRECT.placed(conda_env="no-such-env"), executable="python")

    assert got[-1] == "python"


def test_the_conda_rung_resolves_the_windows_interpreter_on_win32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conda on Windows installs an env's interpreter in the env's prefix."""
    monkeypatch.setattr(sys, "platform", "win32")
    conda = _conda_env_with(tmp_path, "unused", "python")
    interpreter = tmp_path / "root" / "envs" / "toolenv" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    fake_path_lookup(monkeypatch, {"conda": conda})
    monkeypatch.setenv("MOSAIC_FAKE_CONDA_ENV", "toolenv")
    env = _with_repo_venv(tmp_path / "repo")

    assert tool_invocation(env, executable="python")[-1] == str(interpreter)
    assert missing_interpreter(env) is None


def test_the_conda_rung_resolves_a_windows_console_script_on_win32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conda on Windows installs console scripts in the env's ``Scripts``."""
    monkeypatch.setattr(sys, "platform", "win32")
    conda = _conda_env_with(tmp_path, "unused", "runme")
    script = tmp_path / "root" / "envs" / "toolenv" / "Scripts" / "runme.exe"
    script.parent.mkdir(parents=True)
    script.touch()
    fake_path_lookup(monkeypatch, {"conda": conda})

    got = tool_invocation(_DIRECT.placed(conda_env="toolenv"), executable="runme")

    assert got[-1] == str(script)


def test_an_absent_env_refuses_a_tool_that_is_only_an_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal names the conda rung and the directories that it searched."""
    conda = _conda_env_with(tmp_path, "unused", "python")
    (tmp_path / "root" / "bin" / "python").touch()
    fake_path_lookup(monkeypatch, {"conda": conda})
    monkeypatch.setenv("MOSAIC_FAKE_CONDA_ENV", "no-such-env")
    env = _with_repo_venv(tmp_path / "repo")

    with pytest.raises(_FakeNotFound) as refused:
        tool_invocation(env, executable="python")

    message = str(refused.value)
    rung = "the conda environment 'no-such-env' named by MOSAIC_FAKE_CONDA_ENV"
    assert rung in message
    assert str(tmp_path / "root" / "envs" / "no-such-env" / "bin") in message
    assert _FakeNotFound.default_message in message
    assert missing_interpreter(env) == _FakeNotFound.default_message


# --- the exceptions -------------------------------------------------------


def test_a_missing_tool_raises_its_own_subclass_with_its_own_hint() -> None:
    with pytest.raises(_FakeNotFound, match="install the fake tool"):
        tool_invocation(
            ToolEnv(
                tool="Fake",
                conda_env_var="MOSAIC_FAKE_CONDA_ENV",
                bin_var="MOSAIC_FAKE_BIN",
                bin_mode="direct",
                not_found=_FakeNotFound,
            ),
            executable="absent",
        )


def test_a_missing_conda_names_the_bin_variable_to_set_instead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message has to name *this* tool's variable, not a generic one."""
    fake_path_lookup(monkeypatch, {})

    with pytest.raises(_FakeNotFound, match="MOSAIC_FAKE_BIN"):
        tool_invocation(_DIRECT.placed(conda_env="nope"), executable="runme")


def test_the_exit_error_carries_the_streams_and_elides_a_long_command() -> None:
    error = _FakeExit(["a", "b", "c", "d", "e", "f", "g"], 3, "out", "boom")

    assert error.returncode == 3
    assert error.stdout == "out"
    assert error.stderr == "boom"
    assert "Fake exited with code 3" in str(error)
    assert "a b c d e f ..." in str(error)


def test_a_short_command_is_not_elided() -> None:
    assert "..." not in str(_FakeExit(["a", "b"], 1, "", ""))


def test_a_tool_may_shorten_how_much_of_its_command_is_echoed() -> None:
    error = _TerseExit(["tool", "--flag", "value"], 1, "", "")

    assert "tool --flag ..." in str(error)
    assert "value" not in str(error)


_PROGRAM = "import sys\nprint(sys.argv)"


@pytest.mark.parametrize(
    ("argv", "echoed"),
    [
        (
            ["/envs/lp/bin/python", "-c", _PROGRAM, "config.yaml", "out.csv"],
            "/envs/lp/bin/python -c <program> config.yaml out.csv",
        ),
        (
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "lp",
                "python",
                "-c",
                _PROGRAM,
            ],
            "conda run --no-capture-output -n lp python ...",
        ),
    ],
    ids=["bin", "conda"],
)
def test_a_python_program_is_named_and_never_quoted(
    argv: list[str], echoed: str
) -> None:
    """``python -c <program>`` runs Lightning Pose and SLEAP from either placement.

    The program is one argv token however many lines it holds, so no count of
    tokens elides it: from a bin placement it is the third.
    """
    error = _FakeExit(argv, 1, "", "")

    assert echoed in str(error)
    assert "import sys" not in str(error)
    assert error.cmd == argv


# --- what the tool said, when the tool said it went fine ---------------------


def test_a_missing_output_names_what_was_expected_and_quotes_both_streams() -> None:
    """The exit code already said "fine", so this message is the whole diagnosis.

    Both streams, because which one carries the reason is the tool's choice:
    Lightning Pose writes its traceback to stdout and leaves stderr holding the
    launcher's "See above for error".
    """
    error = missing_output_error(
        "SLEAP", Path("/w/vid.predictions.slp"), "sleap-nn is not installed", "quiet"
    )

    text = str(error)
    assert isinstance(error, FileNotFoundError)
    assert "/w/vid.predictions.slp" in text
    assert "sleap-nn is not installed" in text, "the reason the tool gave was dropped"
    assert "quiet" in text


def test_a_silent_tool_says_so_rather_than_printing_empty_labels() -> None:
    """Two empty labelled sections read as a formatting bug, not as a fact."""
    text = str(missing_output_error("T-Rex", Path("/w/v.pv"), "", "   \n "))

    assert "The tool printed nothing." in text
    assert "Stdout" not in text
    assert "Stderr" not in text


def test_the_exit_error_also_quotes_stdout() -> None:
    """The case that motivated this: the traceback was on stdout all along."""
    text = str(_FakeExit(["tool"], 1, "Traceback: the real reason", "See above"))

    assert "Traceback: the real reason" in text
    assert "See above" in text


def test_the_launchers_own_epilogue_does_not_crowd_out_the_tools_error() -> None:
    """``conda run`` appends its report, echoing the whole command, after the child.

    Tailing the raw stream lands inside that echo and never reaches the
    traceback, which is how a real "nvidia-dali is required for video inference"
    was delivered as a screenful of the snippet mosaic had just sent.
    """
    stderr = (
        "ImportError: nvidia-dali is required for video inference\n"
        "ERROR conda.cli.main_run:execute(125): `conda run python -c\n"
        + "from lightning_pose.api import Model\n" * 40
        + "` failed. (See above for error)"
    )

    text = str(_FakeExit(["tool"], 1, "", stderr))

    assert "nvidia-dali is required" in text
    assert "lightning_pose.api" not in text
