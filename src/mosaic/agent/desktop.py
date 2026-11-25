"""Register mosaic's server in Claude Desktop's configuration.

Claude Desktop starts a server without the shell's environment, and a step the
server submits runs with the server's own. The entry therefore carries a fixed
``PATH`` of this environment's directories and the variables the invoking shell
set, and :func:`desktop_server_entry` builds it, with what the user should know
about it. :func:`desktop_config_path` finds the file that Desktop reads and
:func:`install_desktop_server` merges the entry into it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic.agent import (
    MCP_SERVER_NAME,
    SITE_URL,
    Installation,
    WriteAction,
    server_command,
    with_mcp_server,
)
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.toolenv import conda_executable
from mosaic.env_settings import flag_env

if TYPE_CHECKING:
    from mosaic.core.pipeline.toolenv import ToolEnv

__all__ = [
    "CONFIG_NAME",
    "DesktopConfigRefused",
    "DesktopEntry",
    "DesktopInstall",
    "desktop_config_path",
    "desktop_server_entry",
    "install_desktop_server",
    "path_checked_tools",
]

CONFIG_NAME: Final = "claude_desktop_config.json"
"""The name of Claude Desktop's configuration file."""

FFMPEG_PAGE: Final = f"{SITE_URL}installation/#one-ffmpeg-build-per-environment"
"""The installation page's section on the ffmpeg build of an environment."""

_TOOL_VARIABLE: Final = re.compile(r"^MOSAIC_[A-Z0-9]+_(CONDA_ENV|BIN|PYTHON)$")
"""The variables that place an external tool (``core/pipeline/toolenv.py``)."""

_COPIED_VARIABLES: Final = (
    "MOSAIC_KPMS_LICENSE_ACCEPTED",
    "CONDA_ENVS_DIRS",
    "MOSAIC_ALLOW_TOOL_CODECS",
    "MOSAIC_HIDRA_MODELS_ROOT",
    "MOSAIC_TREX_DISPLAY",
)
"""The variables copied by name: a step the shell runs reads each of them."""

_TRUST_WARNING: Final = (
    "MOSAIC_TRUST_PICKLED_FILES=1 lets this server unpickle raw files in every "
    "dataset it serves, and unpickling can run code in the file. Use it only "
    "for datasets from sources you trust."
)

_SYSTEM_DIRECTORIES: Final = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")
"""The system directories on macOS and Linux."""

_WINDOWS_ENVIRONMENT_DIRECTORIES: Final = (
    "",
    "Library/mingw-w64/bin",
    "Library/usr/bin",
    "Library/bin",
    "Scripts",
    "bin",
)
"""The directories ``conda activate`` adds on Windows, relative to the prefix."""

_CONDA_SCRIPT_DIRECTORIES: Final = ("condabin", "bin", "Scripts")
"""The directories of a conda installation that can hold the ``conda`` program."""


class DesktopConfigRefused(Exception):
    """The configuration file cannot be found or written, and nothing was written.

    The message names the path and what to do about it.
    """


@dataclass(frozen=True)
class DesktopInstall:
    """The result of :func:`install_desktop_server`.

    Attributes:
        path: The configuration file.
        action: Whether it was created, rewritten, or already current.
        dropped_env_keys: The keys of the earlier entry's ``env`` that the new
            entry does not have, sorted.
    """

    path: Path
    action: WriteAction
    dropped_env_keys: list[str]


@dataclass(frozen=True)
class DesktopEntry:
    """The server entry for Claude Desktop and what to tell the user about it.

    Attributes:
        entry: The ``mcpServers`` entry: ``command``, ``args`` and ``env``.
        warnings: Findings that leave the entry weaker than the shell's setup,
            to print on standard error.
        notes: Changes made to a copied value, to print.
    """

    entry: dict[str, JsonValue]
    warnings: list[str]
    notes: list[str]


def path_checked_tools() -> tuple[tuple[ToolEnv, str], ...]:
    """The tools whose executable the shell may reach only through its ``PATH``.

    Each is a placement and the executable name its runner searches with. The
    fifth rung of a tool's ladder searches ``$PATH``, which a Desktop server
    does not have. The interpreter-only tools (keypoint-MoSeq, VAME) do not
    search it, and Lightning Pose looks up ``python``, which cannot tell
    whether the tool is installed.

    Returns:
        The placement and executable of TRex, SLEAP, Ultralytics, POLO and HiDRA.
    """
    from mosaic.behavior.feature_library.hidra_feature import (
        HIDRA_ENV,
        HIDRA_PREDICT_SCRIPT,
    )
    from mosaic.tracking.common.ultralytics_env import (
        POLO_ENV,
        ULTRALYTICS_ENV,
        YOLO_SCRIPT,
    )
    from mosaic.tracking.sleap.run import SLEAP_ENV
    from mosaic.tracking.sleap.training import SLEAP_NN_EXECUTABLE
    from mosaic.tracking.trex.run import TREX_ENV, TREX_EXECUTABLE

    return (
        (TREX_ENV, TREX_EXECUTABLE),
        (SLEAP_ENV, SLEAP_NN_EXECUTABLE),
        (ULTRALYTICS_ENV, YOLO_SCRIPT),
        (POLO_ENV, YOLO_SCRIPT),
        (HIDRA_ENV, HIDRA_PREDICT_SCRIPT),
    )


def desktop_server_entry(
    installation: Installation,
    datasets: Sequence[Path],
    *,
    trust_pickled_files: bool,
    environ: Mapping[str, str] | None = None,
    prefix: Path | None = None,
    platform: str | None = None,
    conda: Callable[[], Path | None] = conda_executable,
) -> DesktopEntry:
    """The Claude Desktop entry that starts ``mosaic agent serve`` over *datasets*.

    ``env.PATH`` lists this environment's directories, then the conda
    installation's ``condabin``, then the system directories, each once and only
    when it exists. It never holds the shell's ``PATH`` or the installation's
    ``bin``, so the server finds this environment's ffmpeg and no other build.
    The other ``env`` keys are the tool placement variables, the settings that a
    step reads, and pickle trust, copied from *environ* when set.

    Args:
        installation: The mosaic whose server the entry starts.
        datasets: The manifests, or the directories that hold them, to serve.
            Each is written as given, so pass absolute paths.
        trust_pickled_files: Whether to write ``MOSAIC_TRUST_PICKLED_FILES=1``
            whatever the shell holds.
        environ: The invoking shell's environment. ``None`` reads the process's.
        prefix: The environment's prefix. ``None`` is ``sys.prefix``.
        platform: The platform's name as ``sys.platform`` spells it. ``None``
            is the process's.
        conda: Finds the ``conda`` executable. The default searches the
            process's ``PATH`` and ``CONDA_EXE``.

    Returns:
        The entry, its warnings and its notes.
    """
    shell = os.environ if environ is None else environ
    windows = (sys.platform if platform is None else platform) == "win32"
    root = Path(sys.prefix if prefix is None else prefix)
    warnings: list[str] = []
    notes: list[str] = []

    environment = _existing(_environment_directories(root, windows=windows))
    found = conda()
    scripts = _conda_scripts(found, warnings)
    directories = _unique(
        [*environment, *scripts, *_existing(_system_directories(shell, windows))]
    )
    separator = ";" if windows else ":"
    env: dict[str, JsonValue] = {"PATH": separator.join(directories)}

    _warn_without_ffmpeg(environment, warnings)
    env.update(_copied(shell, notes, warnings))
    _warn_about_path_only_tools(shell, directories, warnings)
    _trust_pickled_files(shell, trust_pickled_files, env, warnings)

    command, arguments = server_command(installation, datasets)
    entry: dict[str, JsonValue] = {"command": command, "args": arguments, "env": env}
    return DesktopEntry(entry, warnings, notes)


def _environment_directories(prefix: Path, *, windows: bool) -> list[Path]:
    if windows:
        return [
            prefix / name if name else prefix
            for name in _WINDOWS_ENVIRONMENT_DIRECTORIES
        ]
    return [prefix / "bin"]


def _system_directories(shell: Mapping[str, str], windows: bool) -> list[Path]:
    if not windows:
        return [Path(name) for name in _SYSTEM_DIRECTORIES]
    system_root = shell.get("SystemRoot") or shell.get("SYSTEMROOT")
    if not system_root:
        return []
    base = Path(system_root)
    return [base / "System32", base, base / "System32" / "Wbem"]


def _existing(candidates: Sequence[Path]) -> list[str]:
    return [str(path) for path in candidates if path.is_dir()]


def _unique(directories: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(directories))


def _conda_scripts(conda: Path | None, warnings: list[str]) -> list[str]:
    """The ``condabin`` of the installation holding *conda*, when it has one.

    A symlink to ``conda`` is resolved first: Homebrew links Miniforge's ``conda``
    into its own ``bin``.
    """
    if conda is None:
        return []
    real = conda.resolve()
    if real.parent.name not in _CONDA_SCRIPT_DIRECTORIES:
        warnings.append(
            f"conda is at {real}, in a directory that is not an installation's "
            "condabin, bin or Scripts, so the entry's PATH leaves conda out and a "
            "tool placed by a conda environment is not found."
        )
        return []
    condabin = real.parent.parent / "condabin"
    if condabin.is_dir():
        return [str(condabin)]
    warnings.append(
        f"conda is at {real}, but {condabin} does not exist, so the entry's PATH "
        "leaves conda out and a tool placed by a conda environment is not found."
    )
    return []


def _warn_without_ffmpeg(environment: Sequence[str], warnings: list[str]) -> None:
    search = os.pathsep.join(environment)
    missing = [
        name
        for name in ("ffmpeg", "ffprobe")
        if shutil.which(name, path=search) is None
    ]
    if missing:
        warnings.append(
            f"{' and '.join(missing)} not found in this environment's directories, "
            "so the server cannot read or write video. A server that only reads "
            f"works without them. See {FFMPEG_PAGE}"
        )


def _copied(
    shell: Mapping[str, str], notes: list[str], warnings: list[str]
) -> dict[str, JsonValue]:
    """The copied variables of *shell*, with a bare ``_BIN`` resolved."""
    names = sorted(
        name
        for name in shell
        if _TOOL_VARIABLE.match(name) is not None or name in _COPIED_VARIABLES
    )
    copied: dict[str, JsonValue] = {}
    for name in names:
        value = shell[name]
        if not value:
            continue
        if name.endswith("_BIN") and "/" not in value and "\\" not in value:
            value = _resolved_bin(name, value, shell, notes, warnings)
        copied[name] = value
    return copied


def _resolved_bin(
    name: str,
    value: str,
    shell: Mapping[str, str],
    notes: list[str],
    warnings: list[str],
) -> str:
    """*value*, a program name, as an absolute path on the shell's ``PATH``."""
    found = shutil.which(value, path=shell.get("PATH", ""))
    if found is None:
        warnings.append(
            f"{name}={value} names a program that is not on your shell's PATH, "
            "and the server's PATH is fixed. Set it to an absolute path and run again."
        )
        return value
    notes.append(
        f"{name}={value} is written as {found}, where your shell's PATH finds it."
    )
    return found


def _warn_about_path_only_tools(
    shell: Mapping[str, str], directories: Sequence[str], warnings: list[str]
) -> None:
    for env, executable in path_checked_tools():
        if shell.get(env.conda_env_var) or shell.get(env.bin_var):
            continue
        found = shutil.which(executable, path=shell.get("PATH", ""))
        if found is None or str(Path(found).parent) in directories:
            continue
        warnings.append(
            f"{env.tool} is on your shell's PATH at {found}, which Claude Desktop's "
            f"server does not search; set {env.bin_var}={found} or "
            f"{env.conda_env_var} and run again."
        )


def _trust_pickled_files(
    shell: Mapping[str, str],
    flag: bool,
    env: dict[str, JsonValue],
    warnings: list[str],
) -> None:
    from mosaic.core.track_library.safe_load import TRUST_VARIABLE

    if flag or flag_env(TRUST_VARIABLE, shell):
        env[TRUST_VARIABLE] = "1"
        warnings.append(_TRUST_WARNING)


def desktop_config_path(
    override: Path | None = None,
    *,
    platform: str | None = None,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """The configuration file that Claude Desktop reads.

    On macOS and Linux it is the file in Desktop's directory under the home
    directory. On Windows the documented ``%APPDATA%`` file and the two places
    the MSIX package reads instead are searched: one existing file is the
    answer, several are refused, and with none the documented path is the answer
    when its directory exists. The file itself need not exist, but its directory
    must.

    Args:
        override: A file to use in place of the search.
        platform: The platform's name as ``sys.platform`` spells it. ``None``
            is the process's.
        home: The home directory. ``None`` is the user's.
        environ: The environment that names the Windows directories. ``None``
            is the process's.

    Returns:
        The file to write.

    Raises:
        DesktopConfigRefused: The directory does not exist, or no file or
            several files were found on Windows.
    """
    if override is not None:
        _require_directory(
            override,
            f"The directory {override.parent} of --desktop-config does not exist. "
            "Create it or name another file.",
        )
        return override
    name = sys.platform if platform is None else platform
    if name == "win32":
        return _windows_config_path(os.environ if environ is None else environ)
    base = Path.home() if home is None else home
    directory = (
        base / "Library" / "Application Support" / "Claude"
        if name == "darwin"
        else base / ".config" / "Claude"
    )
    path = directory / CONFIG_NAME
    _require_directory(
        path,
        f"Claude Desktop's configuration directory {directory} does not exist. "
        "Install Claude Desktop and open it once, or pass --desktop-config.",
    )
    return path


def _require_directory(path: Path, message: str) -> None:
    if not path.parent.is_dir():
        raise DesktopConfigRefused(message)


def _windows_config_path(shell: Mapping[str, str]) -> Path:
    roaming = shell.get("APPDATA")
    local = shell.get("LOCALAPPDATA")
    documented = Path(roaming) / "Claude" / CONFIG_NAME if roaming else None
    patterns: list[Path] = []
    if local:
        patterns = [
            Path(local)
            / "Packages"
            / "Claude_*"
            / "LocalCache"
            / "Roaming"
            / "Claude"
            / CONFIG_NAME,
            Path(local) / "Claude-Data" / "Claude_*" / CONFIG_NAME,
        ]
    existing = [documented] if documented is not None and documented.is_file() else []
    for pattern in patterns:
        existing += sorted(
            path
            for path in Path(pattern.anchor).glob(
                str(pattern.relative_to(pattern.anchor))
            )
            if path.is_file()
        )
    if len(existing) == 1:
        return existing[0]
    if existing:
        listed = ", ".join(str(path) for path in existing)
        msg = (
            f"Several Claude Desktop configuration files exist: {listed}. "
            "Guessing could write a file the app does not read. Pass "
            "--desktop-config to choose one."
        )
        raise DesktopConfigRefused(msg)
    if documented is not None and documented.parent.is_dir():
        return documented
    looked = ", ".join(str(path) for path in [documented, *patterns] if path)
    msg = (
        "Claude Desktop's configuration file was not found"
        + (f"; looked for {looked}" if looked else "")
        + ". Install Claude Desktop and open it once, or pass --desktop-config."
    )
    raise DesktopConfigRefused(msg)


def install_desktop_server(path: Path, entry: dict[str, JsonValue]) -> DesktopInstall:
    """Set mosaic's server in the configuration file *path* to *entry*.

    Every other server and top-level key is kept, and the earlier mosaic entry
    is replaced whole. The write is atomic, keeps the file's mode and follows a
    symlink to its target. A result with the same content as the file, however
    it is formatted, writes nothing.

    Args:
        path: The configuration file. Its directory must exist.
        entry: The server entry, as :func:`desktop_server_entry` builds it.

    Returns:
        The action taken and the keys that the earlier entry's ``env`` held and
        the new one does not.

    Raises:
        DesktopConfigRefused: The file cannot be read, is not JSON, is not an
            object or has no object of servers, or its directory does not
            exist or cannot be written. Nothing is written.
    """
    existing = path.exists()
    try:
        before = path.read_text(encoding="utf-8-sig") if existing else ""
        after = with_mcp_server(before, entry, config_name=CONFIG_NAME)
    except json.JSONDecodeError as exc:
        msg = (
            f"{path} is not valid JSON: {exc}. Fix or delete the file and run "
            "again; nothing was written."
        )
        raise DesktopConfigRefused(msg) from exc
    except (OSError, ValueError) as exc:
        msg = (
            f"{path} cannot be used: {exc}. Fix it and run again; nothing was written."
        )
        raise DesktopConfigRefused(msg) from exc
    dropped = _dropped_env_keys(before, entry)
    earlier: JsonValue = json.loads(before) if before.strip() else None
    if earlier is not None and earlier == json.loads(after):
        return DesktopInstall(path, "unchanged", dropped)
    target = path.resolve()
    mode = stat.S_IMODE(target.stat().st_mode) if existing else None

    def fill(temporary: Path) -> None:
        _ = temporary.write_text(after, encoding="utf-8")
        if mode is not None:
            temporary.chmod(mode)

    try:
        atomic_write(target, fill, create_parent=False)
    except FileNotFoundError as exc:
        msg = (
            f"The directory {target.parent} of {path} does not exist. Create it "
            "or name another file with --desktop-config; nothing was written."
        )
        raise DesktopConfigRefused(msg) from exc
    except OSError as exc:
        msg = f"{path} could not be written: {exc}. Nothing was changed."
        raise DesktopConfigRefused(msg) from exc
    return DesktopInstall(path, "updated" if existing else "created", dropped)


def _dropped_env_keys(before: str, entry: dict[str, JsonValue]) -> list[str]:
    """The ``env`` keys of the mosaic entry in *before* that *entry* lacks."""
    if not before.strip():
        return []
    configured: JsonValue = json.loads(before)
    servers = configured.get("mcpServers") if isinstance(configured, dict) else None
    earlier = servers.get(MCP_SERVER_NAME) if isinstance(servers, dict) else None
    old = earlier.get("env") if isinstance(earlier, dict) else None
    if not isinstance(old, dict):
        return []
    new = entry.get("env")
    kept = new if isinstance(new, dict) else {}
    return sorted(key for key in old if key not in kept)
