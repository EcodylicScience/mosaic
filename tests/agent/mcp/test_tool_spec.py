"""The shared tool contract: what it declares, what it imports and how it dispatches."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.spec import (
    MAX_PREVIEW_ROWS,
    TOOLS,
    ToolBackend,
    ToolSpec,
    call_tool,
    tool_spec,
)
from mosaic.core.pipeline import preview
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from tests.helpers import modules_loaded_by

BUDGET_MODULES = (
    "mcp",
    "mosaic.tracking",
    "mosaic.behavior",
    "mosaic.behavior.feature_library",
    "pandas",
    "cv2",
    "av",
)
"""The modules that importing the contract must leave unloaded."""


def test_importing_the_contract_loads_neither_an_sdk_nor_the_analysis_stack() -> None:
    statement = (
        "from mosaic.agent.mcp.spec import TOOLS\n"
        "for spec in TOOLS:\n"
        "    _ = spec.input_model.model_json_schema()"
    )

    assert modules_loaded_by(statement, BUDGET_MODULES) == []


def test_building_a_local_backend_loads_neither_an_sdk_nor_the_analysis_stack() -> None:
    statement = (
        "from mosaic.agent.mcp.local import LocalBackend\n"
        "_ = LocalBackend([], warm=False)"
    )

    assert modules_loaded_by(statement, BUDGET_MODULES) == []


def test_every_tool_has_a_unique_name_and_a_backend_method_of_that_name() -> None:
    names = [spec.name for spec in TOOLS]

    assert len(set(names)) == len(names)
    assert all(callable(getattr(ToolBackend, name, None)) for name in names)
    assert all(callable(getattr(LocalBackend, name, None)) for name in names)


@pytest.mark.parametrize("spec", TOOLS, ids=lambda spec: spec.name)
def test_every_output_reference_resolves_to_a_model_with_a_schema(
    spec: ToolSpec,
) -> None:
    model = spec.output_model()

    assert issubclass(model, BaseModel)
    assert json.dumps(model.model_json_schema())
    assert json.dumps(spec.input_model.model_json_schema())
    assert tool_spec(spec.name) is spec


def test_no_tool_reaches_beyond_the_datasets_it_serves() -> None:
    assert not any(spec.annotations.open_world for spec in TOOLS)


def test_only_submission_and_cancel_change_a_dataset_and_both_ask_to_confirm() -> None:
    changing = {spec.name for spec in TOOLS if not spec.annotations.read_only}
    confirming = {spec.name for spec in TOOLS if spec.needs_confirmation}

    assert changing == confirming == {"submit_pipeline", "cancel_request"}
    cancel = tool_spec("cancel_request")
    assert cancel is not None
    assert cancel.annotations.destructive


def test_the_preview_bound_is_the_one_the_preview_reader_enforces() -> None:
    assert MAX_PREVIEW_ROWS == preview.MAX_PREVIEW_ROWS


def test_a_page_larger_than_one_hundred_is_refused_before_the_backend() -> None:
    backend = LocalBackend([], warm=False)

    refused = call_tool(backend, "list_datasets", {"limit": 101})

    assert isinstance(refused, ErrorEnvelope)
    assert (refused.kind, refused.reason) == ("invalid", "invalid_arguments")
    assert refused.detail is not None
    assert refused.detail["errors"] == [
        {"field": "limit", "message": "Input should be less than or equal to 100"}
    ]


def test_an_unknown_tool_is_not_found() -> None:
    refused = call_tool(LocalBackend([], warm=False), "delete_set", {})

    assert isinstance(refused, ErrorEnvelope)
    assert (refused.kind, refused.reason) == ("not_found", "unknown_tool")


@pytest.mark.parametrize("merged", ["search_step_types", "describe_step_type"])
def test_a_merged_tool_name_points_at_its_replacement(merged: str) -> None:
    refused = call_tool(LocalBackend([], warm=False), merged, {"name": "trex"})

    assert isinstance(refused, ErrorEnvelope)
    assert (refused.kind, refused.reason) == ("not_found", "unknown_tool")
    assert refused.detail == {"tool": merged, "replacement": "describe_step_types"}
    assert "describe_step_types" in refused.message


@pytest.mark.parametrize(
    "arguments",
    [
        {"names": ["trex"], "query": "track"},
        {"names": ["trex"], "type": "op"},
        {"names": ["trex"], "available_only": True},
        {"names": ["trex"], "cursor": "[]"},
        {"detail": "fields"},
        {"query": "track", "detail": "schema"},
        {"names": []},
        {"names": [f"step-{index}" for index in range(9)]},
    ],
)
def test_names_with_a_filter_or_fields_without_names_are_invalid(
    arguments: dict[str, object],
) -> None:
    refused = call_tool(LocalBackend([], warm=False), "describe_step_types", arguments)

    assert isinstance(refused, ErrorEnvelope)
    assert (refused.kind, refused.reason) == ("invalid", "invalid_arguments")


def test_no_tool_takes_an_overwrite_argument() -> None:
    offered = {name for spec in TOOLS for name in spec.input_model.model_fields}

    assert "overwrite" not in offered


def test_the_modules_beside_the_server_load_no_sdk() -> None:
    statement = (
        "import mosaic.agent.mcp.config, mosaic.agent.mcp.prompts, "
        "mosaic.agent.mcp.resources, mosaic.agent.mcp.summaries"
    )

    assert modules_loaded_by(statement, ("mcp", "mcp_types")) == []
