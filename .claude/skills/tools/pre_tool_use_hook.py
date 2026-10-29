"""Read the tool call of a PreToolUse payload and print the hook's answers."""

from hook_payload import print_answer
from json_value import Json

EVENT = "PreToolUse"


def tool_input(payload: dict[str, Json]) -> dict[str, Json]:
    """Return the payload's tool input, empty when it is not an object."""
    value = payload.get("tool_input")
    return value if isinstance(value, dict) else {}


def bash_command(payload: dict[str, Json]) -> str | None:
    """Return the command of a Bash tool call, None for any other call."""
    if payload.get("tool_name") != "Bash":
        return None
    command = tool_input(payload).get("command")
    return command if isinstance(command, str) else None


def deny(reason: str) -> None:
    """Block the tool call, telling the session why."""
    print_answer(
        EVENT,
        {"permissionDecision": "deny", "permissionDecisionReason": reason},
    )


def add_context(text: str) -> None:
    """Let the tool call run, and add `text` to the session's context."""
    print_answer(EVENT, {"additionalContext": text})


def update_input(updated: dict[str, Json]) -> None:
    """Let the tool call run with `updated` as its input."""
    print_answer(EVENT, {"updatedInput": updated})
