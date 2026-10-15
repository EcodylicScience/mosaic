"""The shared MCP tool set stays small: each tool costs every client's context."""

from __future__ import annotations

from mosaic.agent.mcp.spec import SHARED_TOOL_LIMIT, TOOLS


def test_the_shared_set_declares_at_most_fifteen_tools() -> None:
    assert SHARED_TOOL_LIMIT == 15
    assert len(TOOLS) <= SHARED_TOOL_LIMIT, (
        f"{len(TOOLS)} shared tools are declared, above {SHARED_TOOL_LIMIT}. Fold "
        "the new capability into an existing tool or declare it in the server "
        "that alone offers it."
    )
