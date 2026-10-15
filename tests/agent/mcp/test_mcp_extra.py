"""The MCP SDK behind the ``mcp`` extra, loaded by the server only when it is used."""

from __future__ import annotations

import sys

import pytest

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.server import build_server
from mosaic.optional_dependency import MissingOptionalDependency
from tests.helpers import modules_loaded_by


def test_importing_the_server_imports_no_sdk() -> None:
    loaded = modules_loaded_by("import mosaic.agent.mcp.server", ["mcp", "mcp_types"])

    assert loaded == []


def test_building_a_server_without_the_sdk_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mcp", None)

    with pytest.raises(MissingOptionalDependency) as raised:
        _ = build_server(LocalBackend([], warm=False))

    assert raised.value.extra == "mcp"
    assert 'pip install "mosaic-behavior[mcp]"' in str(raised.value)
