"""The shared tool-placement primitives, on their own terms.

``tests/test_{trex,sleap,litpose}_invocation.py`` cover the five-step ladder once
per tool, which is what proves each tool still resolves the way it did. This
covers what those three cannot: the parts of the shared machinery that no single
tool exercises fully -- the environment overlay, an explicitly chosen matplotlib
backend, and the two exception shapes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.tracking.common import toolenv
from mosaic.tracking.common.toolenv import (
    ToolEnv,
    ToolExitError,
    ToolNotFoundError,
    missing_output_error,
    subprocess_env,
    tool_invocation,
)


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


_ON_PATH: dict[str, str] = {
    "runme": "/p/bin/runme",
    "finder": "/p/bin/finder",
    "conda": "/p/bin/conda",
}


def _fake_which(name: str) -> str | None:
    return _ON_PATH.get(name)


def _nothing_on_path(_name: str) -> str | None:
    return None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "MOSAIC_FAKE_CONDA_ENV",
        "MOSAIC_FAKE_BIN",
        "CONDA_EXE",
        "CONDA_ENVS_DIRS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(toolenv.shutil, "which", _fake_which)


# --- the two bin modes, which is the whole of what differs between tools ----


def test_direct_mode_treats_an_explicit_bin_as_the_executable() -> None:
    got = tool_invocation(
        _DIRECT.placed(bin_path="/x/somewhere-else"), executable="runme"
    )

    assert got == ["/x/somewhere-else"]


def test_sibling_mode_resolves_the_executable_beside_an_explicit_bin() -> None:
    got = tool_invocation(_SIBLING.placed(bin_path="/x/bin/other"), executable="runme")

    assert got == ["/x/bin/runme"]


def test_without_a_locator_the_executable_is_looked_up_directly() -> None:
    assert tool_invocation(_SIBLING, executable="runme") == ["/p/bin/runme"]


def test_a_locator_is_looked_up_and_the_executable_resolved_beside_it() -> None:
    """Lightning Pose's case: find ``litpose`` in order to find its ``python``.

    Looking up the executable itself would find the caller's own interpreter.
    """
    assert tool_invocation(_LOCATED, executable="python") == ["/p/bin/python"]


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


def _conda_env_with(tmp_path: Path, env_name: str, executable: str) -> str:
    """A conda layout under *tmp_path*: ``<root>/bin/conda`` and one env holding
    *executable*. Returns the path to the fake ``conda``."""
    conda = tmp_path / "root" / "bin" / "conda"
    conda.parent.mkdir(parents=True)
    conda.touch()
    env_bin = tmp_path / "root" / "envs" / env_name / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / executable).touch()
    return str(conda)


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
    monkeypatch.setattr(
        toolenv.shutil, "which", lambda name: conda if name == "conda" else None
    )

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
    monkeypatch.setattr(
        toolenv.shutil, "which", lambda name: conda if name == "conda" else None
    )

    got = tool_invocation(_DIRECT.placed(conda_env="toolenv"), executable="absent-here")

    assert got[-1] == "absent-here"


def test_conda_envs_dirs_locates_an_env_outside_the_base_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conda = _conda_env_with(tmp_path, "unused", "runme")
    elsewhere = tmp_path / "elsewhere" / "toolenv" / "bin"
    elsewhere.mkdir(parents=True)
    (elsewhere / "runme").touch()
    monkeypatch.setattr(
        toolenv.shutil, "which", lambda name: conda if name == "conda" else None
    )
    monkeypatch.setenv("CONDA_ENVS_DIRS", str(tmp_path / "elsewhere"))

    got = tool_invocation(_DIRECT.placed(conda_env="toolenv"), executable="runme")

    assert got[-1] == str(elsewhere / "runme")


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
    monkeypatch.setattr(toolenv.shutil, "which", _nothing_on_path)

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
    """Lightning Pose's argv is ``python -c <a whole program>``."""
    error = _TerseExit(["python", "-c", "import sys; ..."], 1, "", "")

    assert "python -c ..." in str(error)
    assert "import sys" not in str(error)


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
