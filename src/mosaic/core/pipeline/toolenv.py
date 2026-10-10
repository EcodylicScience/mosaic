"""Where an external tool lives, and how it is launched.

Mosaic drives several tools too heavy to share mosaic's environment. TRex pins
``python=3.11``, SLEAP brings PyTorch and Qt, and Lightning Pose brings PyTorch
and DALI. Each tool runs in a separate environment and is located the same five
ways, in the same order:

1. ``ToolEnv.conda_env``,
2. ``ToolEnv.bin_path``,
3. ``MOSAIC_<TOOL>_CONDA_ENV``,
4. ``MOSAIC_<TOOL>_BIN``,
5. ``$PATH``.

The first two are what a caller chooses for one run, through
:meth:`ToolEnv.placed`; the next two are what the machine says. That ladder was
written three times, along with ``conda run`` construction, the two exception
shapes, and the ``MPLBACKEND`` fix below. What differs between the tools is the
values :class:`ToolEnv` holds.

A tool that is only an interpreter declares :attr:`ToolEnv.repo_venv` and
takes the fifth rung from that directory instead of ``$PATH``. Keypoint-MoSeq
and VAME do not install a console script, and a ``$PATH`` lookup of ``python``
finds mosaic's interpreter. For such a tool, every rung must find an
interpreter file. A rung that finds none raises the tool's
:class:`ToolNotFoundError`, naming the rung and the paths it read.

**The ladder resolves placement, never identity.** Which environment a tool ran
in is a property of the machine, so none of these values reaches a ``run_id``:
two machines with the tool installed differently must agree on what a run is
called.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Final, Literal

from mosaic.core.pipeline.subprocess_util import command_summary, foreign_environment
from mosaic.user_paths import user_path

__all__ = [
    "BinMode",
    "ToolEnv",
    "ToolExitError",
    "ToolInvocation",
    "ToolNotFoundError",
    "captured_output",
    "conda_invocation",
    "display_overlay",
    "PROBE_DEADLINE_FLOOR_SECONDS",
    "interpreter_refusal",
    "locate_tool",
    "missing_interpreter",
    "missing_output_error",
    "subprocess_env",
    "tool_invocation",
]

PROBE_DEADLINE_FLOOR_SECONDS: Final = 900.0
"""The least time a silent subcommand gets to answer, whatever the caller's bound is.

``idle_timeout`` bounds *silence*, which is the right unit once work is under
way: a tool that prints a line per decoded batch is hung if it goes quiet. A
probe prints nothing at all between spawn and answer, so the same number would
be a deadline on a cold torch import and a checkpoint load off a network mount
-- work proceeding exactly as intended. A user who shortens the window so a hung
tool dies quickly must not thereby put a stopwatch on loading a model, so a
probe gets the caller's value or this floor, whichever is longer.

A property of probes rather than of any one tool, which is why it lives here:
the Ultralytics probe, ``tracker-defaults`` and the SLEAP probe are silent for
the same stretch and take the same floor. No such floor belongs on tracking or
inference, which report per batch, and raising their bound would blunt the one
thing supervising them.
"""

_TAIL: Final = 500
"""How much of each captured stream a failure message echoes."""

_CONDA_EPILOGUE: Final = "ERROR conda.cli.main_run:"
"""Where ``conda run``'s own report of the failure begins.

``conda run`` appends this *after* the child's output, and it echoes the entire
command -- for Lightning Pose, an inline program of a dozen lines. Tailing the
raw stream therefore lands squarely inside conda's echo and never reaches the
child's traceback, which is how a real ``nvidia-dali is required for video
inference`` arrived as a screenful of the snippet mosaic had just sent.
"""


def captured_output(stdout: str, stderr: str) -> str:
    """The tail of both streams, labelled, for a message about a failed run.

    **Both** streams, because which one carries the reason is the tool's choice
    and not a stable one, and the tail is taken from the child's own output
    rather than from the launcher's epilogue after it.
    """
    parts: list[str] = []
    for name, stream in (("Stdout", stdout), ("Stderr", stderr)):
        own, _, _ = stream.partition(_CONDA_EPILOGUE)
        tail = own.strip()[-_TAIL:]
        if tail:
            parts.append(f"  {name} (last {_TAIL} chars): {tail}")
    return "\n".join(parts) if parts else "  The tool printed nothing."


def missing_output_error(
    tool: str, expected: Path, stdout: str, stderr: str
) -> FileNotFoundError:
    """The tool exited *successfully* and its expected output is not there.

    Worth its own helper because the exit code has already said "fine" and this
    check is the only thing left that disagrees -- so its message is the whole
    diagnosis the operator gets. A tool that exits 0 having produced nothing has
    usually explained itself on one of its streams first: SLEAP 1.6 prints
    ``sleap-nn is not installed`` and returns 0, which read, without the streams,
    as the unactionable ``Expected .slp file not found after inference``.
    """
    return FileNotFoundError(
        f"{tool} exited successfully but did not write its expected output:\n"
        f"  Expected: {expected}\n"
        f"{captured_output(stdout, stderr)}"
    )


BinMode = Literal["direct", "sibling"]
"""How an explicit ``MOSAIC_<TOOL>_BIN`` names the executable to run.

``direct`` -- the value *is* the executable, which is TRex's single binary.
``sibling`` -- the value names some entry in the environment's ``bin``, and the
executable is resolved beside it. SLEAP installs several console scripts
together, so pointing at any one of them names the directory the others live in;
Lightning Pose is driven through its environment's ``python`` rather than a
console verb, so the ``litpose`` script names where that ``python`` is.
"""


class ToolNotFoundError(FileNotFoundError):
    """An external tool, or ``conda``, could not be located.

    Subclassed per tool so a caller can still catch one tool specifically, with
    the install hint as a class attribute rather than a re-implemented
    ``__init__``.
    """

    default_message: ClassVar[str] = "The external tool was not found on $PATH."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or type(self).default_message)


class ToolExitError(RuntimeError):
    """An external tool exited with a non-zero return code.

    ``head`` is how many argv tokens the message echoes before eliding. It is a
    readability knob, not a limit: six covers a console script and its first
    flags. A program passed with ``-c`` is never quoted
    (:func:`~mosaic.core.pipeline.subprocess_util.command_summary`).
    """

    tool_name: ClassVar[str] = "The external tool"
    head: ClassVar[int] = 6

    def __init__(
        self,
        cmd: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.cmd: list[str] = cmd
        self.returncode: int = returncode
        self.stdout: str = stdout
        self.stderr: str = stderr
        super().__init__(
            f"{type(self).tool_name} exited with code {returncode}.\n"
            f"  Command: {command_summary(cmd, type(self).head)}\n"
            f"{captured_output(stdout, stderr)}"
        )


@dataclass(frozen=True, slots=True)
class ToolEnv:
    """One tool's placement: the names it answers to, and how to reach it.

    The first eight fields describe the tool and are declared once beside it.
    The last three are what one caller chooses for one run, and
    :meth:`placed` is how they are set -- so a caller states where a tool lives
    as one value instead of an argument per aspect beside every call. All of it
    is placement, so none of it reaches a ``run_id``.

    Attributes:
        tool: Human name, used only in messages.
        conda_env_var: The ``MOSAIC_<TOOL>_CONDA_ENV`` variable name.
        bin_var: The ``MOSAIC_<TOOL>_BIN`` variable name.
        bin_mode: How ``bin_var`` names the executable.
        not_found: This tool's :class:`ToolNotFoundError` subclass, raised for
            both a missing tool and a missing ``conda``.
        locator: What to look up on ``$PATH`` when it is not the executable
            itself. Empty means the executable is its own locator. A locator
            names the environment rather than the thing run: Lightning Pose and
            Ultralytics look up a script in order to find the ``python`` beside
            it, which is not on ``$PATH`` under a distinguishing name, and SLEAP
            looks up ``sleap-convert`` in order to run ``sleap-nn``, which a
            ``uv tool install`` does not link. The locator found is resolved
            through any symlink before its directory is read.
        display_var: The variable naming this tool's ``DISPLAY``, for a tool
            that needs one. Empty means the tool runs without a display, which
            is every tool but TRex.
        repo_venv: For an interpreter-only tool, the directory that contains
            its ``.venv``. ``None``, the default, keeps ``$PATH`` as step 5 of
            the ladder for a tool with a console script.
        conda_env: Step 1 of the ladder. Empty defers to ``conda_env_var``.
        bin_path: Step 2 of the ladder. Empty defers to ``bin_var``.
        display: ``DISPLAY`` for the tool's subprocess. Empty defers to
            ``display_var`` and then to the caller's own ``DISPLAY``.
    """

    tool: str
    conda_env_var: str
    bin_var: str
    bin_mode: BinMode
    not_found: type[ToolNotFoundError]
    locator: str = ""
    display_var: str = ""
    repo_venv: Path | None = None
    conda_env: str = ""
    bin_path: str = ""
    display: str = ""

    def placed(
        self,
        *,
        conda_env: str | None = None,
        bin_path: str | Path | None = None,
        display: str | None = None,
    ) -> ToolEnv:
        """This tool, reached through the environment, binary or display given.

        Each argument left unset keeps what this placement already says, so
        naming one aspect states nothing about the others.

        Args:
            conda_env: Conda environment to run the tool in.
            bin_path: Explicit binary, read according to :attr:`bin_mode`.
            display: ``DISPLAY`` for the tool's subprocess.

        Returns:
            A placement differing from this one in the fields given.
        """
        return replace(
            self,
            conda_env=conda_env or self.conda_env,
            bin_path=str(bin_path) if bin_path else self.bin_path,
            display=display or self.display,
        )


def _user_envs_dir() -> Path:
    """Return ``~/.conda/envs``, the env directory that conda adds for the user.

    The conda rung reads it only here. A test substitutes this function to search
    a directory that does not depend on the machine the suite runs on.
    """
    return user_path("~/.conda/envs")


def _conda_env_prefixes(conda: str, env_name: str) -> list[Path]:
    """Return the prefixes that conda env *env_name* may be installed at, in order.

    They derive from the location of the ``conda`` executable, which costs no
    subprocess on a launch. The env named ``base`` is the installation itself.
    Any other name is looked up in each directory that ``CONDA_ENVS_DIRS`` lists,
    then in the installation's ``envs``, then in ``~/.conda/envs``, the user's
    env directory. The ``envs_dirs`` of a ``.condarc`` file are not read.
    """
    base = user_path(conda).resolve().parent.parent
    if env_name == "base":
        return [base]
    roots = [
        user_path(entry)
        for entry in os.environ.get("CONDA_ENVS_DIRS", "").split(os.pathsep)
        if entry
    ]
    roots.extend((base / "envs", _user_envs_dir()))
    return [root / env_name for root in roots]


def conda_invocation(
    env: ToolEnv, env_name: str, executable: str, *, source: str
) -> list[str]:
    """An argv prefix running *executable* inside conda env *env_name*.

    ``conda run`` rather than a bare path into the environment's ``bin``, so the
    target environment is fully activated: each of these tools resolves native
    libraries against its own ``CONDA_PREFIX``, which a bare path does not set.
    ``--no-capture-output`` leaves the child's streams attached, which is what
    the inactivity watchdog and the progress callback read.

    **The executable is passed by absolute path when one can be found.** ``conda
    run`` resolves a bare name against ``$PATH``, and it does not place the
    environment's ``bin`` first: an entry the caller inherited -- ``~/.local/bin``
    is the common one, holding a ``uv tool install`` of the same tool -- can sit
    ahead of it and answer instead. The named environment is then not the one
    that runs, silently, so the version recorded as provenance describes a
    different install and two machines configured identically execute different
    code. Naming the file directly settles it while keeping the activation that
    made ``conda run`` the right wrapper: ``CONDA_PREFIX`` is still set, because
    ``conda run`` sets it for the environment it was asked for, not for the path
    it ends up executing.

    A tool with a console script is run by its bare name when the search does not
    find it in a prefix of *env_name*. A tool declaring :attr:`ToolEnv.repo_venv`
    is refused instead.

    Args:
        env: The tool's placement.
        env_name: The conda environment to run *executable* in.
        executable: The program to run.
        source: The rung that named *env_name*, as :attr:`ToolInvocation.source`
            words it.

    Raises:
        ToolNotFoundError: The subclass named by ``env.not_found``, when
            ``conda`` is not found, or when *env* declares ``repo_venv`` and the
            search does not find *executable*.
    """
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if conda is None:
        raise env.not_found(
            f"'conda' was not found on $PATH; cannot run {executable} in env "
            f"'{env_name}'. Set {env.bin_var} to an explicit path instead, or "
            f"make conda available."
        )
    # `shutil.which` already returns an absolute path; the `CONDA_EXE` fallback
    # does not, and this value becomes argv[0] below.
    conda = str(user_path(conda))
    candidates = [
        _installed_executable(prefix, executable, "conda")
        for prefix in _conda_env_prefixes(conda, env_name)
    ]
    resolved = next((path for path in candidates if path.is_file()), None)
    if resolved is None and env.repo_venv is not None:
        searched = ", ".join(str(path.parent) for path in candidates)
        claim = f"is not a file in any of {searched}"
        raise env.not_found(interpreter_refusal(env, executable, source, claim))
    target = executable if resolved is None else str(resolved)
    return [conda, "run", "--no-capture-output", "-n", env_name, target]


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """How a tool is launched, and which rung of the ladder located it.

    Attributes:
        argv: The argv prefix that launches the executable.
        source: The rung that located the executable, worded for a message:
            a conda environment and what chose it, the binary placed for this
            run, a ``MOSAIC_<TOOL>_BIN`` value, the in-repository environment
            or the ``$PATH`` lookup.
    """

    argv: list[str]
    source: str


def tool_invocation(env: ToolEnv, *, executable: str) -> list[str]:
    """Resolve how to launch *executable*, as an argv prefix.

    It is the argv of :func:`locate_tool`, which also names the rung.

    Raises:
        ToolNotFoundError: The subclass named by ``env.not_found``, as
            :func:`locate_tool` raises it.
    """
    return locate_tool(env, executable=executable).argv


def locate_tool(env: ToolEnv, *, executable: str) -> ToolInvocation:
    """Resolve how to launch *executable*, and name the rung that located it.

    Precedence, first match wins: ``env.conda_env``, ``env.bin_path``,
    ``env.conda_env_var``, ``env.bin_var``, then ``env.repo_venv`` if *env*
    declares one, else ``$PATH``. What a caller chose beats what the machine
    says. One run can then be placed elsewhere without disturbing the others,
    and :meth:`ToolEnv.placed` is how a caller chooses.

    Raises:
        ToolNotFoundError: The subclass named by ``env.not_found``, when
            neither the tool nor ``conda`` can be located, or when a rung finds
            no interpreter file for a tool declaring ``repo_venv``.
    """
    if env.conda_env:
        source = f"the conda environment '{env.conda_env}' that this run was placed in"
        return ToolInvocation(
            conda_invocation(env, env.conda_env, executable, source=source), source
        )
    if env.bin_path:
        source = "the binary this run was placed with"
        return ToolInvocation(
            [_from_bin(env, env.bin_path, executable, source=source)], source
        )

    env_conda = os.environ.get(env.conda_env_var)
    if env_conda:
        source = f"the conda environment '{env_conda}' named by {env.conda_env_var}"
        return ToolInvocation(
            conda_invocation(env, env_conda, executable, source=source), source
        )
    env_bin = os.environ.get(env.bin_var)
    if env_bin:
        source = f"the value of {env.bin_var}"
        return ToolInvocation(
            [_from_bin(env, env_bin, executable, source=source)], source
        )

    root = _repo_venv_root(env)
    if root is not None:
        source = f"the in-repository environment under {root}"
        return ToolInvocation([_repo_interpreter(env, root, source)], source)

    # A locator is looked up in order to find something beside it; without one
    # the executable is what is looked up, and what is found is what runs.
    looked_up = env.locator or executable
    source = f"the $PATH lookup of '{looked_up}'"
    found = shutil.which(looked_up)
    if found is None:
        raise env.not_found()
    if not env.locator:
        return ToolInvocation([found], source)
    # Resolved, because a uv or pipx tool install links its scripts into
    # ~/.local/bin, and what sits beside the link is not the environment.
    return ToolInvocation([str(Path(found).resolve().parent / executable)], source)


def _repo_venv_root(env: ToolEnv) -> Path | None:
    """Return the directory that rung 5 reads *env*'s in-repository venv from.

    Rung 5 reads :attr:`ToolEnv.repo_venv` only here. A test substitutes this
    function to place the venv independently of the machine the suite runs on.
    """
    return env.repo_venv


def _repo_interpreter(env: ToolEnv, root: Path, source: str) -> str:
    """Return the interpreter in the ``.venv`` under *root*, read in place of ``$PATH``.

    Raises:
        ToolNotFoundError: The subclass named by ``env.not_found``, naming
            *source*, when the venv under *root* has not been built.
    """
    resolved = _installed_executable(root / ".venv", "python", "venv")
    _require_interpreter(env, resolved, source)
    return str(resolved)


_EnvironmentKind = Literal["venv", "conda"]
"""The kinds of environment that the ladder reads an executable from."""


def _installed_executable(
    prefix: Path, executable: str, kind: _EnvironmentKind
) -> Path:
    """Return the path at which an environment of *kind* installs *executable*.

    On POSIX, both kinds install every executable in ``<prefix>/bin``. On Windows,
    each executable has an ``.exe`` suffix. A venv installs every one in
    ``<prefix>/Scripts``. Conda installs its interpreter in *prefix* itself and
    console scripts in ``<prefix>/Scripts``.
    """
    if sys.platform != "win32":
        return prefix / "bin" / executable
    if kind == "conda" and executable == "python":
        return prefix / "python.exe"
    return prefix / "Scripts" / f"{executable}.exe"


def _require_interpreter(env: ToolEnv, resolved: Path, source: str) -> None:
    """Refuse a *resolved* that is not a file, for a tool declaring ``repo_venv``.

    Such a tool does not have a locator that confirms an install. This check
    confirms the interpreter instead, and a missing one raises the tool's typed
    refusal before any subprocess starts. A tool with a console script is not
    checked.

    The refusal names *resolved* and *source* and ends with
    ``env.not_found.default_message``, the install help that each tool declares.
    """
    if env.repo_venv is not None and not resolved.is_file():
        raise env.not_found(interpreter_refusal(env, resolved, source, "is not a file"))


def missing_interpreter(env: ToolEnv) -> str | None:
    """Return the tool's install help, or ``None`` when its ladder locates ``python``.

    The interpreter is located as starting the tool locates it, without starting
    it. A feature's catalog prerequisite answers with this for a tool that is only
    an interpreter.
    """
    try:
        _ = tool_invocation(env, executable="python")
    except ToolNotFoundError:
        return env.not_found.default_message
    return None


def interpreter_refusal(
    env: ToolEnv, interpreter: str | Path, source: str, claim: str
) -> str:
    """Return a refusal of a tool's interpreter, ending with the tool's install help.

    Args:
        env: The tool's placement.
        interpreter: The interpreter's path.
        source: The rung that located it, as :attr:`ToolInvocation.source`
            words it.
        claim: What is wrong with the interpreter, a predicate such as
            ``is not a file``.
    """
    return (
        f"The {env.tool} interpreter '{interpreter}', {source}, {claim}. "
        f"{env.not_found.default_message}"
    )


def _from_bin(
    env: ToolEnv, bin_path: str | Path, executable: str, *, source: str
) -> str:
    """Read an explicit ``bin`` value according to the tool's :class:`BinMode`.

    Expanded but never resolved. These values arrive from ``MOSAIC_<TOOL>_BIN``,
    which no shell expands -- it is the value of a variable, not a word being
    expanded -- so an unexpanded ``~/bin/trex`` becomes ``argv[0]`` verbatim and
    fails with ENOENT. Resolving would break the other supported spelling: a bare
    ``MOSAIC_TREX_BIN=trex`` is a name for ``$PATH`` to find, and resolving it
    would turn it into ``$CWD/trex``.

    A bare name in sibling mode is looked up on ``$PATH`` and resolved through
    any link, as the ``$PATH`` rung resolves a locator. Read as a path, a bare
    name has no directory, and the executable beside it would be a bare name too,
    which ``$PATH`` resolves to whatever comes first: for
    ``MOSAIC_SLEAP_BIN=sleap-convert``, the first ``python`` on ``$PATH``.

    Args:
        env: The tool's placement.
        bin_path: The value to read.
        executable: What to run, beside the value in sibling mode.
        source: Where *bin_path* came from, named in a refusal.

    Raises:
        ToolNotFoundError: The subclass named by ``env.not_found``, for a bare
            sibling name that is not on ``$PATH``, or, when *env* declares
            ``repo_venv``, for a resolved path that is not a file.
    """
    if env.bin_mode == "direct":
        resolved = user_path(bin_path)
        _require_interpreter(env, resolved, source)
        return str(resolved)
    given = str(bin_path).strip()
    if os.path.dirname(given):
        resolved = user_path(given).parent / executable
        _require_interpreter(env, resolved, source)
        return str(resolved)
    found = shutil.which(given)
    if found is None:
        raise env.not_found(
            f"'{given}', {source}, is not on $PATH, so the {env.tool} environment "
            f"it names cannot be located. Name the file by its path."
        )
    resolved = Path(found).resolve().parent / executable
    _require_interpreter(env, resolved, source)
    return str(resolved)


# Jupyter exports ``MPLBACKEND=module://matplotlib_inline.backend_inline`` into
# the kernel environment. Inherited by a tool subprocess it makes matplotlib fail
# at import -- and TRex's pybind11 glue turns that into terminate() and a SIGABRT,
# so the failure does not even name matplotlib. No external tool needs an
# interactive backend, so an inherited IPython ``module://`` backend is replaced
# with a headless-safe one. An explicitly chosen backend is left alone.
_INLINE_BACKEND_PREFIX: Final = "module://"
_HEADLESS_BACKEND: Final = "Agg"


def display_overlay(env: ToolEnv) -> dict[str, str] | None:
    """Return the ``DISPLAY`` overlay *env* asks for, or ``None`` for none.

    A tool that initializes an OpenGL context needs a display even when it draws
    no window. On a headless host that is a virtual framebuffer (``Xvfb :99
    ...``), named either on the placement or in the tool's own ``display_var``.
    ``None`` leaves whatever ``DISPLAY`` the caller already exported.
    """
    chosen = env.display or os.environ.get(env.display_var, "")
    return {"DISPLAY": chosen} if chosen else None


def subprocess_env(overlay: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment that an external tool runs in.

    It starts from :func:`foreign_environment`, this process's environment
    without this interpreter's Python search path and library directories. Any
    *overlay* is applied to it, and an inherited notebook matplotlib backend is
    replaced with ``Agg``.
    """
    run_env = {**foreign_environment(), **(overlay or {})}
    if run_env.get("MPLBACKEND", "").startswith(_INLINE_BACKEND_PREFIX):
        run_env["MPLBACKEND"] = _HEADLESS_BACKEND
    return run_env
