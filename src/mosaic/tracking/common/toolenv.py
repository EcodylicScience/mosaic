"""Where an external tracking tool lives, and how it is launched.

Every integrated tracker runs a tool that is too heavy to share the mosaic
environment -- TRex pins ``python=3.11``, SLEAP brings PyTorch and Qt, Lightning
Pose brings PyTorch and DALI -- so each is reached through its own environment
and located the same five ways, in the same order:

1. the per-call ``<tool>_conda_env`` argument,
2. the per-call ``<tool>_bin`` argument,
3. ``MOSAIC_<TOOL>_CONDA_ENV``,
4. ``MOSAIC_<TOOL>_BIN``,
5. ``$PATH``.

That ladder was written three times, along with ``conda run`` construction, the
two exception shapes, and the ``MPLBACKEND`` fix below. What actually differs
between the three is four values, which is what :class:`ToolEnv` carries.

**The ladder resolves placement, never identity.** Which environment a tool ran
in is a property of the machine, so none of these values reaches a ``run_id``:
two machines with the tool installed differently must agree on what a run is
called.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal

__all__ = [
    "BinMode",
    "ToolEnv",
    "ToolExitError",
    "ToolNotFoundError",
    "conda_invocation",
    "subprocess_env",
    "tool_invocation",
]

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
    """An external tracking tool, or ``conda``, could not be located.

    Subclassed per tool so a caller can still catch one tool specifically, with
    the install hint as a class attribute rather than a re-implemented
    ``__init__``.
    """

    default_message: ClassVar[str] = "The tracking tool was not found on $PATH."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or type(self).default_message)


class ToolExitError(RuntimeError):
    """An external tracking tool exited with a non-zero return code.

    ``head`` is how many argv tokens the message echoes before eliding. It is a
    readability knob, not a limit: six covers a console script and its first
    flags, and Lightning Pose lowers it to four because its argv is
    ``python -c <an entire program>`` and printing the program helps nobody.
    """

    tool_name: ClassVar[str] = "The tracking tool"
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
        head = type(self).head
        cmd_str = " ".join(cmd[:head]) + (" ..." if len(cmd) > head else "")
        super().__init__(
            f"{type(self).tool_name} exited with code {returncode}.\n"
            f"  Command: {cmd_str}\n"
            f"  Stderr (last 500 chars): {stderr[-500:]}"
        )


@dataclass(frozen=True, slots=True)
class ToolEnv:
    """One tool's placement: the names it answers to, and how to reach it.

    Attributes:
        tool: Human name, used only in messages.
        conda_env_var: The ``MOSAIC_<TOOL>_CONDA_ENV`` variable name.
        bin_var: The ``MOSAIC_<TOOL>_BIN`` variable name.
        bin_mode: How ``bin_var`` names the executable.
        not_found: This tool's :class:`ToolNotFoundError` subclass, raised for
            both a missing tool and a missing ``conda``.
        locator: What to look up on ``$PATH`` when it is not the executable
            itself. Empty means the executable is its own locator. Only
            Lightning Pose differs: it looks up ``litpose`` in order to find the
            ``python`` beside it, because the interpreter is not on ``$PATH``
            under a distinguishing name.
    """

    tool: str
    conda_env_var: str
    bin_var: str
    bin_mode: BinMode
    not_found: type[ToolNotFoundError]
    locator: str = ""


def _conda_env_executable(conda: str, env_name: str, executable: str) -> Path | None:
    """``<prefix>/bin/<executable>`` for env *env_name*, when it is there.

    Derived from the ``conda`` executable's own location rather than by asking
    conda, which would cost a subprocess on every launch. Both layouts a conda
    installation uses are tried -- ``<root>/envs/<name>`` for a named env under
    the base installation, and ``<root>`` itself for the base env -- plus every
    directory ``CONDA_ENVS_DIRS`` names, which is how an env outside the base
    installation is found. ``None`` when no candidate holds the executable, which
    leaves the caller with the bare name and today's behaviour.
    """
    roots: list[Path] = []
    for entry in os.environ.get("CONDA_ENVS_DIRS", "").split(os.pathsep):
        if entry:
            roots.append(Path(entry) / env_name)
    base = Path(conda).resolve().parent.parent
    roots.append(base / "envs" / env_name)
    roots.append(base)
    for prefix in roots:
        candidate = prefix / "bin" / executable
        if candidate.is_file():
            return candidate
    return None


def conda_invocation(env: ToolEnv, env_name: str, executable: str) -> list[str]:
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
    """
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if conda is None:
        raise env.not_found(
            f"'conda' was not found on $PATH; cannot run {executable} in env "
            f"'{env_name}'. Set {env.bin_var} to an explicit path instead, or "
            f"make conda available."
        )
    resolved = _conda_env_executable(conda, env_name, executable)
    target = str(resolved) if resolved is not None else executable
    return [conda, "run", "--no-capture-output", "-n", env_name, target]


def tool_invocation(
    env: ToolEnv,
    *,
    executable: str,
    conda_env: str | None = None,
    bin_path: str | Path | None = None,
) -> list[str]:
    """Resolve how to launch *executable*, as an argv prefix.

    Precedence, first match wins: the *conda_env* argument, the *bin_path*
    argument, ``env.conda_env_var``, ``env.bin_var``, then ``$PATH``. Arguments
    beat environment variables so one call can override a machine's default
    without disturbing the others.

    Raises:
        ToolNotFoundError: The subclass named by ``env.not_found``, when neither
            the tool nor ``conda`` can be located.
    """
    if conda_env:
        return conda_invocation(env, conda_env, executable)
    if bin_path:
        return [_from_bin(env, bin_path, executable)]

    env_conda = os.environ.get(env.conda_env_var)
    if env_conda:
        return conda_invocation(env, env_conda, executable)
    env_bin = os.environ.get(env.bin_var)
    if env_bin:
        return [_from_bin(env, env_bin, executable)]

    # A locator is looked up in order to find something beside it; without one
    # the executable is what is looked up, and what is found is what runs.
    found = shutil.which(env.locator or executable)
    if found is None:
        raise env.not_found()
    return [str(Path(found).parent / executable) if env.locator else found]


def _from_bin(env: ToolEnv, bin_path: str | Path, executable: str) -> str:
    """Read an explicit ``bin`` value according to the tool's :class:`BinMode`."""
    if env.bin_mode == "direct":
        return str(bin_path)
    return str(Path(bin_path).parent / executable)


# Jupyter exports ``MPLBACKEND=module://matplotlib_inline.backend_inline`` into
# the kernel environment. Inherited by a tool subprocess it makes matplotlib fail
# at import -- and TRex's pybind11 glue turns that into terminate() and a SIGABRT,
# so the failure does not even name matplotlib. No tracking tool needs an
# interactive backend, so an inherited IPython ``module://`` backend is replaced
# with a headless-safe one. An explicitly chosen backend is left alone.
_INLINE_BACKEND_PREFIX: Final = "module://"
_HEADLESS_BACKEND: Final = "Agg"


def subprocess_env(overlay: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment an external tracking tool runs in.

    The caller's environment, with any *overlay* applied and an inherited
    notebook matplotlib backend neutralized.
    """
    run_env = {**os.environ, **(overlay or {})}
    if run_env.get("MPLBACKEND", "").startswith(_INLINE_BACKEND_PREFIX):
        run_env["MPLBACKEND"] = _HEADLESS_BACKEND
    return run_env
