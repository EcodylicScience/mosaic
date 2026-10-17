"""Answer the MCP server's confirmations from a Claude Code ``Elicitation`` hook.

An operator who runs an agent unattended approves conditions in advance:
``mosaic agent confirm-hook --approve gpu_lane --approve training_op`` reads the
hook's input on standard input and accepts a confirmation whose conditions are
all approved. Claude Code shows a hook only the question's ``message``, so the
conditions are read from the tag that ends it
(:func:`~mosaic.agent.mcp.spec.read_confirmation_tag`).

The module imports the tool contract alone, and neither the MCP SDK nor the
analysis stack.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Final, Literal

from mosaic.agent import MCP_SERVER_NAME
from mosaic.agent.mcp.spec import (
    CONFIRM_FIELD,
    ConfirmationCondition,
    read_confirmation_tag,
)
from mosaic.core.json_value import JsonValue
from mosaic.core.narrowing import is_str_mapping

__all__ = ["HOOK_EVENT", "HookFallback", "answer_elicitation"]

HOOK_EVENT: Final = "Elicitation"
"""The Claude Code hook event that an MCP server's elicitation raises."""

type HookFallback = Literal["ask", "cancel", "decline"]
"""What to answer a confirmation whose conditions are not all approved.

``ask`` prints nothing and leaves the question to the client's dialog.
``decline`` refuses it, and a decline from any hook overrides another hook's
accept, so a hook added later cannot approve more than the operator did.
``cancel`` answers that nobody saw the question.
"""


def answer_elicitation(
    hook_input: Mapping[str, object],
    approved: Collection[ConfirmationCondition],
    *,
    server: str = MCP_SERVER_NAME,
    otherwise: HookFallback = "ask",
) -> dict[str, JsonValue] | None:
    """Return the hook's answer to one elicitation, ``None`` to print nothing.

    Only a form that the server *server* asks for is answered. Another server's
    elicitation, a URL elicitation, and a form other than the one-box
    confirmation are left to the dialog. An input that names no ``mode`` is a
    form, and one that names no ``requested_schema`` is read by its tag alone.

    The server asks for nothing but confirmations, so a confirmation whose last
    line is no readable tag counts as one whose conditions are not all
    approved: under ``otherwise="decline"`` nothing from the server is left
    for another hook to accept.

    Args:
        hook_input: The JSON object that Claude Code passes the hook.
        approved: The conditions the operator approves.
        server: The name under which the client lists mosaic's server.
        otherwise: What to answer a confirmation with a condition outside
            *approved*, or without a readable tag.

    Returns:
        The hook output accepting the confirmation with its box ticked, or
        declining or cancelling it, or ``None``.
    """
    if hook_input.get("hook_event_name") != HOOK_EVENT:
        return None
    if hook_input.get("mcp_server_name") != server:
        return None
    if hook_input.get("mode") not in (None, "form"):
        return None
    schema = hook_input.get("requested_schema")
    if schema is not None:
        properties = schema.get("properties") if is_str_mapping(schema) else None
        if not is_str_mapping(properties) or set(properties) != {CONFIRM_FIELD}:
            return None
    message = hook_input.get("message")
    tagged = read_confirmation_tag(message) if isinstance(message, str) else None
    if tagged is not None and tagged[1] <= set(approved):
        return _output("accept", {CONFIRM_FIELD: True})
    if otherwise == "ask":
        return None
    return _output(otherwise, None)


def _output(
    action: Literal["accept", "cancel", "decline"], content: dict[str, JsonValue] | None
) -> dict[str, JsonValue]:
    """Return the hook output that answers the elicitation with *action*."""
    answer: dict[str, JsonValue] = {"hookEventName": HOOK_EVENT, "action": action}
    if content is not None:
        answer["content"] = content
    return {"hookSpecificOutput": answer}
