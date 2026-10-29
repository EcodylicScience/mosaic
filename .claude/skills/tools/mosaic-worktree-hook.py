#!/usr/bin/env -S python3 -B
"""Make a Python run inside a mosaic worktree import that worktree's source,
as a PreToolUse hook.

A worktree under `.claude/worktrees/<name>/` shares the main checkout's
`.venv` through a symlink, and that environment's editable install points at
the main checkout's `src/`. Without `PYTHONPATH=<worktree>/src` a test or
script run in the worktree silently imports and tests the main checkout.

When the command runs in a worktree (its working directory, or the target of a
leading `cd`) and a step runs python or pytest, the hook prefixes the command
with an export of `PYTHONPATH=<worktree>/src`. A command that sets PYTHONPATH
itself passes untouched, and so does a step that `remote` launches, which runs
in the task machine's own checkout. When the step names a relative `.venv/`
path and the worktree has no `.venv`, the hook denies it and names the symlink
to create.

The hook also applies chores-hook.py's rewrite and answers with one
`updatedInput` carrying both prefixes. Claude Code defines no merge of two
hooks that rewrite one call, so .claude/settings.json runs this hook in place
of chores-hook.py.
"""

import shlex
from collections.abc import Callable
from pathlib import Path

from hook_payload import read_payload
from json_value import Json
from mosaic_commands import PYTHON, Step, load_tool_module, steps, target_directory
from pre_tool_use_hook import bash_command, deny, tool_input, update_input

CHORES_HOOK = load_tool_module("chores-hook.py")
WORKTREES = (".claude", "worktrees")


def worktree_root(directory: Path) -> Path | None:
    """Return the worktree that contains `directory`, None outside one."""
    parts = directory.parts
    for index in range(len(parts) - 2):
        if parts[index : index + 2] == WORKTREES:
            return Path(*parts[: index + 3])
    return None


def runs_python(step: Step) -> bool:
    return not step.launched_by("remote") and (
        step.program == "pytest" or PYTHON.fullmatch(step.program) is not None
    )


def chores_rewrite(payload: dict[str, Json]) -> dict[str, Json] | None:
    """Return chores-hook.py's rewritten tool input, None when it passes the
    call untouched."""
    rewrite: Callable[[dict[str, Json]], dict[str, Json] | None] = CHORES_HOOK.rewrite
    return rewrite(payload)


def worktree_prefix(payload: dict[str, Json], command: str) -> str | None:
    """Return the PYTHONPATH export for a python run in a worktree, an empty
    string to leave the command as it is, or None after denying it."""
    cwd = payload.get("cwd")
    if not isinstance(cwd, str) or "PYTHONPATH=" in command:
        return ""
    command_steps = steps(command)
    python_steps = [step for step in command_steps if runs_python(step)]
    if not python_steps:
        return ""
    root = worktree_root(target_directory(Path(cwd), command_steps))
    if root is None or not (root / "src" / "mosaic").is_dir():
        return ""
    if not (root / ".venv").exists() and any(
        ".venv/" in step.program_token and not step.program_token.startswith("/")
        for step in python_steps
    ):
        main_checkout = root.parent.parent.parent
        deny(
            f"The worktree {root} has no .venv. Link the main checkout's environment "
            f"first: ln -sfn {shlex.quote(str(main_checkout / '.venv'))} "
            f"{shlex.quote(str(root / '.venv'))}"
        )
        return None
    source = shlex.quote(str(root / "src"))
    return f'export PYTHONPATH={source}"${{PYTHONPATH:+:$PYTHONPATH}}"; '


def main() -> None:
    payload = read_payload()
    if payload is None:
        return
    command = bash_command(payload)
    if command is None:
        return
    prefix = worktree_prefix(payload, command)
    if prefix is None:
        return
    updated = chores_rewrite(payload) or tool_input(payload)
    rewritten = updated.get("command")
    if not isinstance(rewritten, str):
        return
    if prefix or rewritten != command:
        update_input({**updated, "command": prefix + rewritten})


if __name__ == "__main__":
    main()
