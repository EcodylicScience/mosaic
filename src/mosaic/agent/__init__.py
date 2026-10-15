"""The guide for an AI coding agent that uses mosaic in a user's project.

The guide is ``SKILL.md`` in this package, shipped as package data. An agent
reads it from the installed mosaic, and nothing copies it into a project.
:func:`install_agent_files` writes a stub skill and a short pointer that both
tell the agent to run ``mosaic agent guide``, which prints :func:`guide_text`
and :func:`find_installation`'s paths. Given the datasets to serve, it also
registers ``mosaic agent serve`` as an MCP server in a project's ``.mcp.json``,
the project configuration that Claude Code reads.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
import sysconfig
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Final, Literal

from mosaic.core.json_value import JsonValue

__all__ = [
    "AGENT_FILES",
    "BLOCK_BEGIN",
    "BLOCK_END",
    "MCP_CONFIG",
    "MCP_SERVER_NAME",
    "SITE_URL",
    "SKILL_DIRECTORY",
    "AgentFileWrite",
    "Installation",
    "WriteAction",
    "find_installation",
    "guide_text",
    "install_agent_files",
    "mcp_server_entry",
    "pointer_block",
    "skill_stub",
    "with_mcp_server",
    "with_pointer_block",
]

GUIDE_RESOURCE: Final = "SKILL.md"
"""The packaged guide, beside this module."""

SITE_URL: Final = "https://ecodylicscience.github.io/mosaic/"
"""The documentation site, named when no local ``docs/`` directory exists."""

SKILL_DIRECTORY: Final = Path(".claude") / "skills" / "mosaic"
"""Where the stub skill goes, relative to the project directory."""

AGENT_FILES: Final = ("CLAUDE.md", "AGENTS.md")
"""The agent memory files that receive the pointer block, when they exist."""

BLOCK_BEGIN: Final = "<!-- mosaic-agent-guide:begin -->"
"""The line opening the pointer block in an agent memory file."""

BLOCK_END: Final = "<!-- mosaic-agent-guide:end -->"
"""The line closing the pointer block in an agent memory file."""

MCP_CONFIG: Final = ".mcp.json"
"""The project's MCP server configuration, relative to the project directory."""

MCP_SERVER_NAME: Final = "mosaic"
"""The name under which ``.mcp.json`` lists mosaic's server."""

_BLOCK_PATTERN: Final = re.compile(
    re.escape(BLOCK_BEGIN) + r".*?" + re.escape(BLOCK_END), re.DOTALL
)

WriteAction = Literal["created", "updated", "unchanged"]
"""What writing one agent file did to it."""


def guide_text() -> str:
    """The packaged guide, read from the installed mosaic.

    Returns:
        The contents of ``mosaic/agent/SKILL.md``, frontmatter included.
    """
    return files(__name__).joinpath(GUIDE_RESOURCE).read_text(encoding="utf-8")


@dataclass(frozen=True)
class Installation:
    """Where the running mosaic lives, as an agent needs to reach it.

    Attributes:
        executable: The ``mosaic`` console script of the running interpreter, or
            ``None`` when it cannot be found.
        interpreter: The running Python interpreter.
        docs: The checkout's ``docs/`` directory, or ``None`` for an install
            without one.
        manifest: The manifest of the dataset enclosing the working directory,
            or ``None`` outside a dataset.
    """

    executable: Path | None
    interpreter: Path
    docs: Path | None
    manifest: Path | None

    @property
    def mosaic_command(self) -> tuple[str, ...]:
        """The arguments that run this installation's ``mosaic``.

        The executable when it was found, else :attr:`interpreter_command`.
        """
        if self.executable is not None:
            return (str(self.executable),)
        return self.interpreter_command

    @property
    def interpreter_command(self) -> tuple[str, ...]:
        """The arguments that run ``mosaic`` through the interpreter, without a script."""
        return (str(self.interpreter), "-m", "mosaic")

    @property
    def guide_command(self) -> str:
        """The shell command that prints the guide for this installation."""
        return shlex.join([*self.mosaic_command, "agent", "guide"])

    @property
    def fallback_command(self) -> str:
        """The same command through the interpreter, which needs no script."""
        return shlex.join([*self.interpreter_command, "agent", "guide"])


def find_installation(working_directory: Path) -> Installation:
    """Locate the running mosaic, its documentation and the enclosing dataset.

    The executable is the script the process was started as when that is
    ``mosaic``, else the ``mosaic`` script in the running interpreter's scripts
    directory. A ``mosaic`` found on ``$PATH`` may belong to another
    environment, so ``$PATH`` is not searched.

    Args:
        working_directory: The directory whose enclosing dataset is reported.

    Returns:
        The resolved paths.
    """
    return Installation(
        executable=_mosaic_executable(),
        interpreter=Path(sys.executable).absolute(),
        docs=_docs_directory(),
        manifest=_enclosing_manifest(working_directory.resolve()),
    )


def _mosaic_executable() -> Path | None:
    started_as = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if started_as is not None and started_as.stem == "mosaic" and started_as.is_file():
        return started_as.resolve()
    scripts = Path(sysconfig.get_path("scripts"))
    for name in ("mosaic", "mosaic.exe"):
        candidate = scripts / name
        if candidate.is_file():
            return candidate.absolute()
    return None


def _docs_directory() -> Path | None:
    # src/mosaic/agent/__init__.py -> the checkout root. In a wheel install the
    # same walk lands inside site-packages' parent, which holds no docs/index.md.
    docs = Path(__file__).resolve().parents[3] / "docs"
    return docs if (docs / "index.md").is_file() else None


def _enclosing_manifest(directory: Path) -> Path | None:
    from mosaic.core.manifest import resolve_manifest_path

    for candidate in (directory, *directory.parents):
        try:
            return resolve_manifest_path(candidate)
        except FileNotFoundError:
            continue
    return None


def skill_stub(installation: Installation) -> str:
    """The stub ``SKILL.md`` written into a project.

    It holds no guidance of its own. It tells the agent to print the guide from
    the installed mosaic, which keeps the guidance matched to the installation.

    Args:
        installation: The mosaic the stub points at.

    Returns:
        The file's contents in Agent Skills format.
    """
    description = (
        "Use whenever code imports mosaic, the mosaic CLI is run, or a "
        "dataset.yaml, tracks, features or pipelines are read or changed. "
        "Prints the guide for the installed version of mosaic."
    )
    lines = [
        "---",
        "name: mosaic",
        f"description: {description}",
        "---",
        "",
        "# mosaic",
        "",
        "Before importing mosaic or running the mosaic CLI, run this command and",
        "follow the guide it prints:",
        "",
        "```bash",
        installation.guide_command,
        "```",
        "",
    ]
    if installation.executable is not None:
        lines += [
            "If that command is not found, run this one instead:",
            "",
            "```bash",
            installation.fallback_command,
            "```",
            "",
        ]
    lines += [
        "The installed mosaic prints the guide, which therefore always matches",
        "the installed version. Never edit the installed mosaic package,",
        "including an editable checkout of it.",
        "",
    ]
    return "\n".join(lines)


def pointer_block(installation: Installation) -> str:
    """The marked block added to an agent memory file, markers included.

    Args:
        installation: The mosaic the block points at.

    Returns:
        The block, without a trailing newline.
    """
    return "\n".join(
        [
            BLOCK_BEGIN,
            "This project uses mosaic. Before importing mosaic or running its CLI, run",
            f"`{installation.guide_command}` and follow the guide it prints.",
            BLOCK_END,
        ]
    )


def with_pointer_block(text: str, block: str) -> str:
    """*text* holding *block* once: in place of a marked block, else appended.

    Args:
        text: The current contents of an agent memory file.
        block: The marked block, as :func:`pointer_block` returns it.

    Returns:
        The new contents.

    Raises:
        ValueError: If *text* opens a block that it never closes.
    """
    if _BLOCK_PATTERN.search(text) is not None:
        return _BLOCK_PATTERN.sub(lambda _match: block, text, count=1)
    if BLOCK_BEGIN in text:
        msg = f"{BLOCK_BEGIN} has no {BLOCK_END} after it"
        raise ValueError(msg)
    if not text.strip():
        return f"{block}\n"
    return f"{text.rstrip()}\n\n{block}\n"


@dataclass(frozen=True)
class AgentFileWrite:
    """One file :func:`install_agent_files` wrote or left as it was.

    Attributes:
        path: The file.
        action: Whether it was created, rewritten, or already current.
    """

    path: Path
    action: WriteAction


def install_agent_files(
    directory: Path,
    installation: Installation,
    *,
    memory: bool,
    mcp_datasets: Sequence[Path] | None = None,
) -> list[AgentFileWrite]:
    """Point the agents working in *directory* at the mosaic guide.

    Writes the stub skill to ``<directory>/.claude/skills/mosaic/SKILL.md``. With
    *memory*, puts the pointer block in every agent memory file in *directory*,
    creating ``CLAUDE.md`` when none exists. With *mcp_datasets*, sets mosaic's
    server entry of ``.mcp.json`` (:func:`with_mcp_server`). A second call
    replaces what the first wrote and adds nothing beside it. Every file is read
    and checked before any is written, so a refusal writes nothing.

    Args:
        directory: The project directory. It must exist.
        installation: The mosaic the files point at.
        memory: Whether to write the pointer block.
        mcp_datasets: The manifests, or the directories that hold them, that the
            MCP server serves. ``None`` leaves ``.mcp.json`` alone.

    Returns:
        Every file written or checked, the skill first and ``.mcp.json`` last.

    Raises:
        NotADirectoryError: If *directory* is not an existing directory.
        ValueError: If an agent memory file opens a block that it never closes,
            or ``.mcp.json`` is not JSON or not an object of servers.
    """
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    skill_path = directory / SKILL_DIRECTORY / "SKILL.md"
    contents = [(skill_path, skill_stub(installation))]
    if memory:
        present = [
            directory / name for name in AGENT_FILES if (directory / name).is_file()
        ]
        block = pointer_block(installation)
        contents += [
            (target, with_pointer_block(_read(target), block))
            for target in present or [directory / AGENT_FILES[0]]
        ]
    if mcp_datasets is not None:
        config = directory / MCP_CONFIG
        entry = mcp_server_entry(installation, mcp_datasets)
        contents.append((config, with_mcp_server(_read(config), entry)))
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    return [_write(path, text) for path, text in contents]


def mcp_server_entry(
    installation: Installation, datasets: Sequence[Path]
) -> dict[str, JsonValue]:
    """The ``.mcp.json`` entry that starts ``mosaic agent serve`` over *datasets*.

    Args:
        installation: The mosaic whose server the entry starts.
        datasets: The manifests, or the directories that hold them, to serve.
            Each is written as given, so pass absolute paths.

    Returns:
        A stdio server entry: its ``type``, ``command`` and ``args``.
    """
    command, *leading = installation.mosaic_command
    arguments: list[JsonValue] = [*leading, "agent", "serve"]
    for dataset in datasets:
        arguments += ["--dataset", str(dataset)]
    return {"type": "stdio", "command": command, "args": arguments}


def with_mcp_server(text: str, entry: dict[str, JsonValue]) -> str:
    """*text*, a ``.mcp.json`` document, with mosaic's server set to *entry*.

    Every other key and server is kept. An empty *text* starts a new document.

    Args:
        text: The current contents of ``.mcp.json``.
        entry: The server entry, as :func:`mcp_server_entry` returns it.

    Returns:
        The new contents, indented JSON ending in a newline.

    Raises:
        ValueError: *text* is not a JSON object, or its ``mcpServers`` is not one.
    """
    configured: JsonValue = json.loads(text) if text.strip() else {}
    if not isinstance(configured, dict):
        msg = f"{MCP_CONFIG} does not hold a JSON object"
        raise ValueError(msg)
    servers = configured.get("mcpServers", {})
    if not isinstance(servers, dict):
        msg = f"mcpServers in {MCP_CONFIG} is not a JSON object"
        raise ValueError(msg)
    configured["mcpServers"] = {**servers, MCP_SERVER_NAME: entry}
    return json.dumps(configured, indent=2) + "\n"


def _read(path: Path) -> str:
    """The text of *path*, empty when it does not exist."""
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _write(path: Path, text: str) -> AgentFileWrite:
    if not path.exists():
        _ = path.write_text(text, encoding="utf-8")
        return AgentFileWrite(path, "created")
    if path.read_text(encoding="utf-8") == text:
        return AgentFileWrite(path, "unchanged")
    _ = path.write_text(text, encoding="utf-8")
    return AgentFileWrite(path, "updated")
