#!/usr/bin/env -S python3 -B
"""Give every agent a separate chores state file, from a PreToolUse hook.

A Bash command that invokes `chores` is rewritten to export `CHORES_FILE`
first. The path is keyed by session and agent, under the system temp
directory. Parallel subagents never share a file, and agents do not have to
know the file exists. A call outside any subagent is keyed as `main`. The
same prefix appends this script's directory to PATH, so `chores` resolves
without an install step.

Every other command, every other tool, and any payload that this hook cannot
read pass untouched. The hook exits 0 without printing.
"""
import re
import shlex
from pathlib import Path

from hook_payload import read_payload
from hook_state import hook_state_directory, safe_id
from json_value import Json
from lexical_paths import lexically_normal
from pre_tool_use_hook import bash_command, tool_input, update_input

# Matches `chores` as a whole word in command position or after a shell
# operator. A file name such as test_chores.sh does not trigger the rewrite.
CHORES_INVOCATION = re.compile(r"(?<![\w./-])chores(?![\w.-])")


def state_file(session_id: str, agent_id: str) -> Path:
    """Return the per-agent state file path for one session."""
    return hook_state_directory("chores") / session_id / agent_id


def rewrite(payload: dict[str, Json]) -> dict[str, Json] | None:
    """Return the updated tool input, or None to pass the command untouched."""
    command = bash_command(payload)
    if command is None or not CHORES_INVOCATION.search(command):
        return None
    session_id = payload.get("session_id")
    agent_id = payload.get("agent_id") or "main"
    if not safe_id(session_id) or not safe_id(agent_id):
        return None
    path = shlex.quote(str(state_file(session_id, agent_id)))
    tools_dir = shlex.quote(str(lexically_normal(Path(__file__).absolute().parent)))
    prefix = f'export CHORES_FILE={path}; export PATH="$PATH":{tools_dir}; '
    return {**tool_input(payload), "command": prefix + command}


def main() -> None:
    payload = read_payload()
    if payload is None:
        return
    updated = rewrite(payload)
    if updated is not None:
        update_input(updated)


if __name__ == "__main__":
    main()
