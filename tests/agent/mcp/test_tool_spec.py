"""The shared tool contract: what it declares, what it imports and how it dispatches."""

from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import BaseModel, ValidationError

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.spec import (
    CONFIRMATION_CONDITIONS,
    MAX_PREVIEW_ROWS,
    MAX_RESULT_CHARS,
    MAX_WAIT_SECONDS,
    RECIPE_EXAMPLE,
    RESULT_TOO_LARGE,
    TOOLS,
    DatasetList,
    DatasetListItem,
    ScopeSelector,
    ListDatasetsInput,
    PlanRecipeInput,
    SubmitPipelineInput,
    SubmittedPipeline,
    SubmittedStep,
    ToolBackend,
    ToolSpec,
    call_tool,
    confirmation_tag,
    read_confirmation_tag,
    tool_spec,
)
from mosaic.core.pipeline import preview
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import Recipe
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


def test_submission_and_cancel_change_a_dataset_and_both_ask_to_confirm() -> None:
    changing = {spec.name for spec in TOOLS if not spec.annotations.read_only}
    confirming = {spec.name for spec in TOOLS if spec.needs_confirmation}

    assert confirming == {"submit_pipeline", "cancel_request", "manage_dataset"}
    # patch_recipe saves a recipe file and changes no data, so it asks nothing.
    assert changing == confirming | {"patch_recipe"}
    cancel = tool_spec("cancel_request")
    assert cancel is not None
    assert cancel.annotations.destructive


def test_patching_saves_a_recipe_and_neither_destroys_nor_asks() -> None:
    patch = tool_spec("patch_recipe")

    assert patch is not None
    assert not patch.annotations.read_only
    assert not patch.annotations.destructive
    assert patch.annotations.idempotent
    assert not patch.needs_confirmation


def test_submission_is_destructive() -> None:
    """With cleanup 'superseded' a submission removes feature runs."""
    submit = tool_spec("submit_pipeline")

    assert submit is not None
    assert submit.annotations.destructive
    assert "cleanup" in submit.confirm_when
    assert "cleanup" in submit.input_model.model_fields


@pytest.mark.parametrize("name", ["plan_recipe", "submit_pipeline"])
def test_retire_sits_beside_cleanup_and_takes_only_digests(name: str) -> None:
    spec = tool_spec(name)

    assert spec is not None
    fields = spec.input_model.model_fields
    assert {"cleanup", "retire"} <= set(fields)
    assert fields["retire"].default_factory is list
    assert "this call only" in (fields["retire"].description or "")
    model = spec.input_model
    assert model.model_validate(_arguments(name, retire=["0123456789abcdef"]))
    for bad in (
        ["short"],
        ["0123456789ABCDEF"],
        ["../escape"],
        ["0123456789abcdef"] * 9,
    ):
        with pytest.raises(ValidationError):
            _ = model.model_validate(_arguments(name, retire=bad))


def _arguments(name: str, **extra: object) -> dict[str, object]:
    base: dict[str, object] = {"dataset": "uuid", "recipe": {}}
    if name == "submit_pipeline":
        base["plan_fingerprint"] = "x"
    return {**base, **extra}


def test_submission_asks_only_for_gpu_training_tracking_partial_bind_duplicate_cleanup_and_sealed_scoring() -> (
    None
):
    submit = tool_spec("submit_pipeline")
    cancel = tool_spec("cancel_request")

    assert submit is not None
    assert cancel is not None
    assert submit.confirm_when == (
        "gpu_lane",
        "training_op",
        "tracking_op",
        "allow_partial",
        "bind",
        "open_duplicate",
        "cleanup",
        "score_sealed",
    )
    assert cancel.confirm_when == ("hard_cancel",)


def test_every_confirmation_condition_belongs_to_a_tool() -> None:
    asked = {condition for spec in TOOLS for condition in spec.confirm_when}

    assert asked == set(CONFIRMATION_CONDITIONS)


def test_the_confirmation_tag_round_trips_and_refuses_unknown_codes_and_earlier_lines() -> (
    None
):
    tag = confirmation_tag("submit_pipeline", ["gpu_lane", "bind", "gpu_lane"])
    question = f"mosaic asks you to confirm submit_pipeline: a; b. Proceed?\n\n{tag}"

    assert tag == "[mosaic-confirm tool=submit_pipeline conditions=bind,gpu_lane]"
    assert read_confirmation_tag(question) == (
        "submit_pipeline",
        frozenset({"bind", "gpu_lane"}),
    )
    assert read_confirmation_tag(f"{tag}\nProceed?") is None
    assert read_confirmation_tag(f"note {tag}") is None
    assert read_confirmation_tag(tag.replace("bind", "anything")) is None
    assert read_confirmation_tag(tag.replace("submit_pipeline", "delete_all")) is None
    assert read_confirmation_tag(tag.replace("bind,", "")) == (
        "submit_pipeline",
        frozenset({"gpu_lane"}),
    )


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


def test_a_wait_longer_than_the_cap_is_refused_before_the_backend() -> None:
    backend = LocalBackend([], warm=False)

    refused = call_tool(
        backend,
        "get_request_status",
        {"dataset": "d", "request_id": "R0", "wait_seconds": MAX_WAIT_SECONDS + 1},
    )

    assert isinstance(refused, ErrorEnvelope)
    assert (refused.kind, refused.reason) == ("invalid", "invalid_arguments")
    assert refused.detail is not None
    assert refused.detail["errors"] == [
        {
            "field": "wait_seconds",
            "message": f"Input should be less than or equal to {MAX_WAIT_SECONDS}",
        }
    ]


def test_a_status_read_waits_and_shrinks_only_when_asked() -> None:
    spec = tool_spec("get_request_status")
    assert spec is not None
    fields = spec.input_model.model_fields

    assert (fields["wait_seconds"].default, fields["brief"].default) == (0, False)
    # Only brief shrinks a result, so only it is offered when one is too large.
    assert spec.narrowing == ("brief",)
    assert "wait_seconds" in spec.description


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


class _Oversized:
    """A backend whose listing and submission each answer with *n* items."""

    def __init__(self, n: int) -> None:
        self.n: int = n

    def list_datasets(self, args: ListDatasetsInput) -> DatasetList:
        items = tuple(
            DatasetListItem(
                uuid=f"{index:032x}",
                name=f"dataset {index}",
                n_sequences=index,
                api_managed=False,
            )
            for index in range(self.n)
        )
        return DatasetList(items=items, total=self.n, next_cursor=None)

    def submit_pipeline(self, args: SubmitPipelineInput) -> SubmittedPipeline:
        steps = tuple(
            SubmittedStep(
                step=f"step_{index}",
                runs="speed-angvel",
                execution_id=f"{index:026d}",
                lane="feature-compute",
                run_id=None,
                status="absent",
            )
            for index in range(self.n)
        )
        return SubmittedPipeline(
            request_id="r",
            recipe_digest="d",
            plan_fingerprint=args.plan_fingerprint,
            steps=steps,
            duplicates=(),
            driver_pid=1,
            driver_log="log",
            poll_after_seconds=5,
        )


def test_a_result_above_the_budget_is_refused_with_how_to_narrow_it() -> None:
    backend = cast("ToolBackend", _Oversized(1000))

    refused = call_tool(backend, "list_datasets", {})
    small = call_tool(cast("ToolBackend", _Oversized(3)), "list_datasets", {})

    assert isinstance(refused, ErrorEnvelope)
    assert (refused.kind, refused.reason) == ("invalid", RESULT_TOO_LARGE)
    assert refused.detail is not None
    assert refused.detail["narrow_with"] == ["limit"]
    assert refused.detail["budget"] == MAX_RESULT_CHARS
    chars = refused.detail["chars"]
    assert isinstance(chars, int) and chars > MAX_RESULT_CHARS
    assert "limit" in refused.message
    assert isinstance(small, DatasetList)


def test_a_write_tool_result_is_never_replaced_by_the_backstop() -> None:
    backend = cast("ToolBackend", _Oversized(1000))

    submitted = call_tool(
        backend,
        "submit_pipeline",
        {"dataset": "d", "recipe": RECIPE_EXAMPLE, "plan_fingerprint": "f"},
    )

    assert isinstance(submitted, SubmittedPipeline)
    assert len(submitted.model_dump_json()) > MAX_RESULT_CHARS


def test_every_tool_with_a_limit_names_it_among_its_narrowing() -> None:
    for spec in TOOLS:
        fields = set(spec.input_model.model_fields)
        assert set(spec.narrowing) <= fields, spec.name
        if "limit" in fields:
            assert "limit" in spec.narrowing, spec.name


def test_the_recipe_example_is_a_recipe_document() -> None:
    plan = tool_spec("plan_recipe")
    assert plan is not None
    recipe = plan.input_schema()["properties"]
    assert isinstance(recipe, dict)
    described = recipe["recipe"]
    assert isinstance(described, dict)

    assert described["examples"] == [RECIPE_EXAMPLE]
    assert Recipe.model_validate(RECIPE_EXAMPLE).steps


@pytest.mark.parametrize(
    ("tool", "path"),
    [
        ("list_sequences", ("SequenceRow", "recording")),
        ("describe_dataset", ("DatasetCounts", "recordings")),
        ("describe_dataset", ("DatasetCounts", "divided_recordings")),
        ("describe_dataset", ("DividedRecordingSummary", "first_sequence")),
        ("describe_dataset", ("DividedRecordingSummary", "last_sequence")),
        ("describe_dataset", ("UndividedRecordingSummary", "kind")),
        ("describe_dataset", ("DividedRecordingSummary", "division")),
        ("describe_dataset", ("RecordingDivisionSummary", "frames")),
        ("describe_dataset", ("RecordingDivisionSummary", "chunksize")),
        ("describe_dataset", ("RecordingDivisionSummary", "identity")),
        ("describe_dataset", ("RecordingDivisionSummary", "camera_serial")),
        ("describe_dataset", ("RecordingDivisionSummary", "first_start_utc")),
        ("describe_dataset", ("RecordingDivisionSummary", "last_start_utc")),
        ("describe_dataset", ("RecordingDivisionSummary", "n_breaks")),
        ("list_sequences", ("SequenceRow", "first_frame")),
        ("list_sequences", ("SequenceRow", "frame_count")),
        ("list_sequences", ("SequenceRow", "start_utc")),
        ("list_sequences", ("SequenceRow", "follows_gap")),
    ],
)
def test_the_listing_and_the_summary_name_the_recordings(
    tool: str, path: tuple[str, str]
) -> None:
    spec = tool_spec(tool)
    assert spec is not None
    definitions = spec.output_schema()["$defs"]
    assert isinstance(definitions, dict)
    model, field = path
    described = definitions[model]
    assert isinstance(described, dict)
    properties = described["properties"]
    assert isinstance(properties, dict)

    assert field in properties


def test_no_tool_takes_an_overwrite_argument() -> None:
    offered = {name for spec in TOOLS for name in spec.input_model.model_fields}

    assert "overwrite" not in offered


def test_the_modules_beside_the_server_load_no_sdk() -> None:
    statement = (
        "import mosaic.agent.mcp.config, mosaic.agent.mcp.prompts, "
        "mosaic.agent.mcp.resources, mosaic.agent.mcp.summaries"
    )

    assert modules_loaded_by(statement, ("mcp", "mcp_types")) == []


def test_entries_accept_a_groups_selector() -> None:
    """A plan names entries, whole groups, or every entry, and so does each step."""
    planned = PlanRecipeInput.model_validate(
        {
            "dataset": "d",
            "recipe": {},
            "entries": {"groups": ["train", "test"]},
            "step_entries": {
                "fit": {"groups": ["train"]},
                "score": [["test", "s3"]],
            },
        }
    )

    assert planned.entries == ScopeSelector(groups=["train", "test"])
    assert planned.step_entries == {
        "fit": ScopeSelector(groups=["train"]),
        "score": [("test", "s3")],
    }
    with pytest.raises(ValueError, match="groups"):
        _ = PlanRecipeInput.model_validate(
            {"dataset": "d", "recipe": {}, "entries": {"groups": []}}
        )
