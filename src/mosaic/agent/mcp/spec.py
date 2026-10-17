"""The shared tools of mosaic's MCP servers, declared without an MCP SDK.

Each tool is one :class:`ToolSpec`: its name, a description written for a model,
an input model, an output model, its annotations and the conditions under which
it refuses without the user's confirmation. :data:`TOOLS` lists the shared set in
the order a server registers it. A server answers each tool through a
:class:`ToolBackend`, and :func:`call_tool` validates a call's arguments and
dispatches it to the backend's method of the same name.

A failure is returned as an
:class:`~mosaic.core.pipeline.error_envelope.ErrorEnvelope`, never raised, so a
server reports every tool's failures in one shape.

The module imports the standard library, pydantic and the status, cancel and
error modules of the pipeline, and loads neither pandas, cv2, av, the feature
library nor ``mosaic.tracking`` (``tests/agent/mcp/test_tool_spec.py``). An
output model whose module loads them is named by a :class:`ModelRef` and
imported when :meth:`ToolSpec.output_model` is called.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.artifact_kinds import ArtifactKind, ArtifactStatus
from mosaic.core.pipeline.cancel import RequestCancel
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import (
    SCHEMA_VERSION,
    BoundRef,
    EntryErrorsWire,
    RecordedErrorWire,
    RequestStatusWire,
    RequestSummaryWire,
    RequestVerdict,
    strip_schema_keywords,
)
from mosaic.core.strict_model import StrictModel
from mosaic.runlog import AttemptLiveness, AttemptOutcome

if TYPE_CHECKING:
    from mosaic.core.dataset_summary import DatasetSummary
    from mosaic.core.pipeline.graph import PlanWire, ValidationReport
    from mosaic.core.pipeline.inventory import InventoryWire
    from mosaic.core.pipeline.preview import ArtifactPreview

    from .outputs import StepSuggestions, StepTypes

__all__ = [
    "CONFIRMATION_CONDITIONS",
    "CONFIRMATION_DECLINED",
    "CONFIRMATION_REQUIRED",
    "DATASET_MANAGED",
    "DEFAULT_PAGE_SIZE",
    "INVALID_ARGUMENTS",
    "INVALID_CURSOR",
    "LARGE_SCOPE_ENTRIES",
    "LISTED_STATUS_NOT",
    "LISTING_MISSING_SAMPLE",
    "MAX_DESCRIBED_STEP_TYPES",
    "MAX_PAGE_SIZE",
    "MAX_PREVIEW_ROWS",
    "MAX_RESULT_CHARS",
    "MAX_SCHEMA_BYTES",
    "MAX_STEP_TYPE_DETAIL_CHARS",
    "MERGED_TOOLS",
    "PLAN_MISSING_SAMPLE",
    "PREVIEW_DEFAULT_ROWS",
    "PREVIEW_MAX_COLUMNS",
    "RECIPE_EXAMPLE",
    "RESULT_TOO_LARGE",
    "SHARED_TOOL_LIMIT",
    "TOOLS",
    "UNKNOWN_DATASET",
    "UNKNOWN_TOOL",
    "ActivityPage",
    "ActivityView",
    "AttemptDiagnosis",
    "CancelRequestInput",
    "ConfirmationCondition",
    "ConfirmationFinding",
    "DatasetList",
    "DatasetListItem",
    "DescribeDatasetInput",
    "DescribeStepTypesInput",
    "DiagnoseFailuresInput",
    "FailureDiagnosis",
    "GetInventoryInput",
    "GetRequestStatusInput",
    "ListActivityInput",
    "ListDatasetsInput",
    "ListSequencesInput",
    "ModelRef",
    "PlanRecipeInput",
    "PreviewOutputInput",
    "RunSummary",
    "SequencePage",
    "SequenceRow",
    "SubmitPipelineInput",
    "SubmittedPipeline",
    "SubmittedStep",
    "StepTypeDetailLevel",
    "SuggestMode",
    "SuggestStepsInput",
    "ToolAnnotations",
    "ToolBackend",
    "ToolInput",
    "ToolName",
    "ToolResult",
    "ToolSpec",
    "ValidateRecipeInput",
    "call_tool",
    "confirmation_declined",
    "confirmation_required",
    "dataset_managed",
    "invalid_cursor",
    "result_too_large",
    "tool_spec",
    "unknown_dataset",
    "unknown_tool",
]

SHARED_TOOL_LIMIT: Final = 15
"""The most tools the shared set may declare. Each tool costs a model context."""

MAX_SCHEMA_BYTES: Final = 8192
"""The most bytes of compact JSON that one tool's input or output schema takes."""

MAX_PAGE_SIZE: Final = 100
"""The most items one page of a listing returns."""

DEFAULT_PAGE_SIZE: Final = 20
"""The items a listing returns when its call names no limit."""

MAX_PREVIEW_ROWS: Final = 50
"""The most rows a preview returns, as ``preview_artifact`` allows."""

PREVIEW_DEFAULT_ROWS: Final = 5
"""The rows a preview returns when its call names no limit."""

PREVIEW_MAX_COLUMNS: Final = 40
"""The columns a preview reads when its call names no ``max_columns``."""

MAX_RESULT_CHARS: Final = 40_000
"""The most characters of JSON that a read tool's result takes.

A larger result is refused with :data:`RESULT_TOO_LARGE`, naming the arguments
that narrow it. A tool that changes a dataset is never refused after the fact.
"""

PLAN_MISSING_SAMPLE: Final = 3
"""The missing entries that ``plan_recipe`` names per step."""

LISTING_MISSING_SAMPLE: Final = 5
"""The missing keys that a ``get_inventory`` listing names per artifact."""

MAX_DESCRIBED_STEP_TYPES: Final = 8
"""The most step types one ``describe_step_types`` call describes by name."""

MAX_STEP_TYPE_DETAIL_CHARS: Final = 16_000
"""The most characters of JSON that one step type's detail takes.

Not enforced per call. A test measures every step type of the catalog, and the
largest, ``vame``, takes about 15,700.
"""

LARGE_SCOPE_ENTRIES: Final = 100
"""The most entries a step may be asked for before a submission asks to confirm."""

LISTED_STATUS_NOT: Final[tuple[ArtifactStatus, ...]] = ("complete",)
"""The statuses that an inventory listing leaves out when its call names none."""

CONFIRMATION_REQUIRED: Final = "confirmation_required"
"""The ``reason`` of a refusal that a call with ``confirm: true`` answers."""

CONFIRMATION_DECLINED: Final = "confirmation_declined"
"""The ``reason`` of a call that the user declined when the server asked."""

DATASET_MANAGED: Final = "dataset_managed"
"""The ``reason`` of a refusal to change a dataset that a service manages."""

UNKNOWN_DATASET: Final = "unknown_dataset"
"""The ``reason`` of a call naming a dataset that the server does not serve."""

UNKNOWN_TOOL: Final = "unknown_tool"
"""The ``reason`` of a call naming a tool that the shared set lacks."""

INVALID_ARGUMENTS: Final = "invalid_arguments"
"""The ``reason`` of a call whose arguments do not fit the tool's input model."""

INVALID_CURSOR: Final = "invalid_cursor"
"""The ``reason`` of a listing given a cursor that no page of it returned."""

RESULT_TOO_LARGE: Final = "result_too_large"
"""The ``reason`` of a read whose result is above :data:`MAX_RESULT_CHARS`."""

type ToolName = Literal[
    "list_datasets",
    "describe_dataset",
    "list_sequences",
    "get_inventory",
    "describe_step_types",
    "validate_recipe",
    "suggest_steps",
    "plan_recipe",
    "submit_pipeline",
    "get_request_status",
    "cancel_request",
    "list_activity",
    "diagnose_failures",
    "preview_output",
]
"""The name of each shared tool."""

MERGED_TOOLS: Final[Mapping[str, ToolName]] = {
    "search_step_types": "describe_step_types",
    "describe_step_type": "describe_step_types",
}
"""Each tool name that an earlier release served, by the tool that replaced it."""

type ConfirmationCondition = Literal[
    "gpu_lane",
    "training_op",
    "tracking_op",
    "allow_partial",
    "bind",
    "large_scope",
    "whole_dataset",
    "open_duplicate",
    "hard_cancel",
]
"""A condition under which a tool refuses until the user confirms the call."""

CONFIRMATION_CONDITIONS: Final[Mapping[ConfirmationCondition, str]] = {
    "gpu_lane": "a step runs on a GPU lane",
    "training_op": "a step trains a model",
    "tracking_op": (
        "a step runs a tracker or model inference, and TRex requires a paid "
        "license for use by a company"
    ),
    "allow_partial": "allow_partial lets a step run over fewer entries than planned",
    "bind": "bind pins artifacts that the recipe did not produce",
    "large_scope": (f"a step is asked for more than {LARGE_SCOPE_ENTRIES} entries"),
    "whole_dataset": "the submission covers the whole dataset",
    "open_duplicate": (
        "a request of this recipe over an overlapping scope has not ended, and "
        "both compute the same runs"
    ),
    "hard_cancel": (
        "a running step loses its work when cancelled. A keypoint-MoSeq or VAME "
        "fit stops at once, and training loses the epoch in progress"
    ),
}
"""What each confirmation condition means, as a person reads it."""

type SuggestMode = Literal["attach", "insert", "input"]
"""Which question ``suggest_steps`` answers about a recipe."""

type ActivityView = Literal["requests", "runs"]
"""Which records ``list_activity`` lists."""

PageSize = Annotated[
    int,
    Field(
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"The most items to return, 1 to {MAX_PAGE_SIZE}.",
    ),
]
"""The ``limit`` of a listing."""

_CURSOR_DESCRIPTION: Final = (
    "The next_cursor of the previous page, to read the next one."
)

Cursor = Annotated[str | None, Field(description=_CURSOR_DESCRIPTION)]
"""The ``cursor`` of a listing."""

_RecordId = Annotated[str, Field(pattern=r"^[0-9A-Za-z_-]+$")]
"""A request or attempt id: one file name stem of letters, digits, ``_`` and ``-``."""

DatasetId = Annotated[
    str, Field(description="The dataset's uuid, as list_datasets returns it.")
]
"""The ``dataset`` that a dataset tool reads."""

RECIPE_EXAMPLE: Final[dict[str, JsonValue]] = {
    "schema_version": SCHEMA_VERSION,
    "name": "speed clusters",
    "steps": [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
        },
        {
            "id": "templates",
            "type": "feature",
            "feature": "extract-templates",
            "inputs": [{"step": "speed"}],
            "params": {"n_templates": 2000},
        },
        {
            "id": "clusters",
            "type": "feature",
            "feature": "global-kmeans",
            "inputs": [{"step": "speed"}],
            "params": {"templates": {"step": "templates"}, "k": 25},
        },
    ],
}
"""A recipe that every installation runs: speed, its templates, and k-means on them.

Its step types need nothing beyond the base install, so the example validates
wherever the server runs.
"""

RecipeDocument = Annotated[
    dict[str, JsonValue],
    Field(
        description=(
            "The recipe as a JSON object: schema_version, name and steps. A step "
            "is {id, type: 'feature', feature, inputs, params} or {id, type: 'op', "
            "kind, params}. An input is 'tracks' or {step: <id>}."
        ),
        examples=[RECIPE_EXAMPLE],
    ),
]
"""A recipe document, parsed by the backend so that a malformed one is reported."""

Entries = Annotated[
    list[tuple[str, str]] | None,
    Field(
        description=(
            "The [group, sequence] entries to cover, or null for every entry of "
            "the dataset."
        )
    ),
]
"""The entries a plan or a submission covers."""


class ToolInput(StrictModel):
    """The arguments of one tool call. Each tool's input model subclasses it."""


class ListDatasetsInput(ToolInput):
    """The arguments of ``list_datasets``."""

    limit: PageSize = DEFAULT_PAGE_SIZE
    cursor: Cursor = None


class DescribeDatasetInput(ToolInput):
    """The arguments of ``describe_dataset``."""

    dataset: DatasetId


class ListSequencesInput(ToolInput):
    """The arguments of ``list_sequences``."""

    dataset: DatasetId
    group: str | None = Field(
        default=None, description="List the sequences of this group only."
    )
    tracks_variant: str | None = Field(
        default=None,
        description=(
            "List this tracks variant's tables. Null lists the variant each entry "
            "resolves to. '' names the unlabeled tables."
        ),
    )
    limit: PageSize = DEFAULT_PAGE_SIZE
    cursor: Cursor = None


class GetInventoryInput(ToolInput):
    """The arguments of ``get_inventory``."""

    dataset: DatasetId
    kinds: list[ArtifactKind] | None = Field(
        default=None, description="Report these artifact kinds only."
    )
    entries: Entries = None
    tracks_variant: str | None = Field(
        default=None,
        description="Keep the feature runs that read this tracks variant.",
    )
    status_not: list[ArtifactStatus] | None = Field(
        default=None,
        description=(
            "Leave out artifacts with these statuses. Null leaves out the complete "
            "ones from a listing and none from a ref. [] reports every artifact."
        ),
    )
    ref: str | None = Field(
        default=None,
        description=(
            "Report only this artifact, in depth, as a listing's ref names it."
        ),
    )
    limit: PageSize = DEFAULT_PAGE_SIZE
    cursor: Cursor = None


type StepTypeDetailLevel = Literal["summary", "fields", "schema"]
"""How much ``describe_step_types`` returns of each step type it names."""


class DescribeStepTypesInput(ToolInput):
    """The arguments of ``describe_step_types``: a search, or names to describe.

    ``names`` describes those step types and takes no search argument. A
    search returns index rows, which ``detail`` cannot deepen.
    """

    query: str = Field(
        default="",
        description="Words to find in a step type's name, summary or category.",
    )
    type: Literal["feature", "op", "tracks"] | None = Field(
        default=None, description="Return step types of this type only."
    )
    category: str | None = Field(
        default=None, description="Return step types of this category only."
    )
    available_only: bool = Field(
        default=False,
        description="Leave out step types this installation cannot run.",
    )
    names: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_DESCRIBED_STEP_TYPES,
        description=(
            "Describe these step types, feature slugs or op kinds, in place of "
            "a search."
        ),
    )
    detail: StepTypeDetailLevel | None = Field(
        default=None,
        description=(
            "With names: 'summary' gives index rows, 'fields' (the default) "
            "adds the params table, 'schema' adds the params JSON Schema."
        ),
    )
    limit: PageSize = DEFAULT_PAGE_SIZE
    cursor: Cursor = None

    @model_validator(mode="after")
    def _names_or_a_search(self) -> Self:
        if self.names is not None:
            searched = [
                name
                for name, given in (
                    ("query", bool(self.query)),
                    ("type", self.type is not None),
                    ("category", self.category is not None),
                    ("available_only", self.available_only),
                    ("cursor", self.cursor is not None),
                )
                if given
            ]
            if searched:
                msg = f"names describes step types and takes no {', '.join(searched)}"
                raise ValueError(msg)
        elif self.detail in ("fields", "schema"):
            msg = f"detail {self.detail!r} describes the step types names lists"
            raise ValueError(msg)
        return self

    @property
    def level(self) -> StepTypeDetailLevel:
        """The detail asked for: ``fields`` by default with names, else ``summary``."""
        if self.detail is not None:
            return self.detail
        return "fields" if self.names is not None else "summary"


class ValidateRecipeInput(ToolInput):
    """The arguments of ``validate_recipe``."""

    recipe: RecipeDocument


class SuggestStepsInput(ToolInput):
    """The arguments of ``suggest_steps``."""

    recipe: RecipeDocument
    mode: SuggestMode = Field(
        description=(
            "'attach': step types to add below step. 'insert': step types to "
            "insert between step and the steps that read it. 'input': inputs to "
            "add to step."
        )
    )
    step: str | None = Field(
        default=None,
        description=(
            "The step asked about. Null asks about the top of the recipe under "
            "'attach' and 'insert'. Required under 'input'."
        ),
    )


class PlanRecipeInput(ToolInput):
    """The arguments of ``plan_recipe``."""

    dataset: DatasetId
    recipe: RecipeDocument
    entries: Entries = None


class SubmitPipelineInput(ToolInput):
    """The arguments of ``submit_pipeline``."""

    dataset: DatasetId
    recipe: RecipeDocument
    plan_fingerprint: str = Field(
        description=(
            "The fingerprint of the plan_recipe result that the user reviewed. A "
            "plan that has changed since refuses the submission."
        )
    )
    entries: Entries = None
    allow_partial: bool = Field(
        default=False,
        description=(
            "Let a step proceed when it would run over fewer entries than planned."
        ),
    )
    bind: dict[str, BoundRef] | None = Field(
        default=None,
        description=(
            "Artifacts the recipe did not produce, pinned by step id as "
            "{feature, run_id}."
        ),
    )
    owner: str = Field(default="", description="Recorded on the request.")
    confirm: bool = Field(
        default=False,
        description=(
            "Set only after the user agreed to the conditions that a refusal "
            "with reason confirmation_required listed."
        ),
    )


class GetRequestStatusInput(ToolInput):
    """The arguments of ``get_request_status``."""

    dataset: DatasetId
    request_id: _RecordId = Field(
        description="The request, as submit_pipeline named it."
    )


class CancelRequestInput(ToolInput):
    """The arguments of ``cancel_request``."""

    dataset: DatasetId
    request_id: _RecordId = Field(description="The request to cancel.")
    confirm: bool = Field(
        default=False,
        description=(
            "Set only after the user agreed to the conditions that a refusal "
            "with reason confirmation_required listed."
        ),
    )


class ListActivityInput(ToolInput):
    """The arguments of ``list_activity``."""

    dataset: DatasetId
    view: ActivityView = Field(
        default="requests",
        description=(
            "'requests': pipeline submissions with their verdicts. 'runs': every "
            "attempt with its outcome."
        ),
    )
    limit: PageSize = DEFAULT_PAGE_SIZE
    cursor: _RecordId | None = Field(default=None, description=_CURSOR_DESCRIPTION)


class DiagnoseFailuresInput(ToolInput):
    """The arguments of ``diagnose_failures``. Exactly one of the two ids is given."""

    dataset: DatasetId
    request_id: _RecordId | None = Field(
        default=None, description="Diagnose every failed step of this request."
    )
    execution_id: _RecordId | None = Field(
        default=None, description="Diagnose this one attempt."
    )


class PreviewOutputInput(ToolInput):
    """The arguments of ``preview_output``."""

    dataset: DatasetId
    ref: str = Field(
        description=(
            "'feature/<storage>/<run_id>', 'tracks-variant/<run_id>' or "
            "'labels-variant/<kind>/<run_id>', as get_inventory names an artifact."
        )
    )
    group: str = Field(description="The entry's group, '' when it has none.")
    sequence: str = Field(description="The entry's sequence.")
    columns: list[str] | None = Field(
        default=None,
        description=(
            "The columns to read, by name or fnmatch pattern ('W_*_f0'). Null "
            "reads the identity and bookkeeping columns first."
        ),
    )
    max_columns: int = Field(
        default=PREVIEW_MAX_COLUMNS,
        ge=1,
        le=200,
        description="The most columns to read. The rest are named as omitted.",
    )
    listed: Literal["read", "all"] = Field(
        default="read",
        description="List only the columns read, or every column of the table.",
    )
    limit: int = Field(
        default=PREVIEW_DEFAULT_ROWS,
        ge=0,
        le=MAX_PREVIEW_ROWS,
        description=f"The rows of the head to return, 0 to {MAX_PREVIEW_ROWS}.",
    )


class DatasetListItem(StrictModel):
    """One dataset that the server serves.

    Attributes:
        uuid: The manifest's uuid, which every dataset tool takes.
        name: The manifest's name.
        n_sequences: The ``(group, sequence)`` entries whose tracks table exists,
            as ``describe_dataset`` counts them.
        api_managed: Whether a service manages the dataset, which refuses the
            tools that change it.
    """

    uuid: str
    name: str
    n_sequences: int
    api_managed: bool


class DatasetList(StrictModel):
    """One page of the datasets that the server serves, ordered by name.

    Attributes:
        items: The page.
        total: How many datasets the server serves.
        next_cursor: The cursor of the next page, ``None`` on the last.
    """

    items: tuple[DatasetListItem, ...]
    total: int
    next_cursor: str | None


class SequenceRow(StrictModel):
    """One entry's tracks table, as the tracks index records it.

    Attributes:
        group: The entry's group, ``""`` when it has none.
        sequence: The entry's sequence.
        tracks_variant: The tracks variant of the table, ``""`` for an
            unlabeled table.
        producer: What wrote the table: a conversion, a tracker or inference.
        std_format: The track schema of the table.
        n_rows: The table's rows, ``None`` when unrecorded.
        n_keypoints: The keypoints of the table, ``None`` when unrecorded.
        frame_min: The table's first frame, ``None`` when unrecorded.
        frame_max: The table's last frame, ``None`` when unrecorded.
    """

    group: str
    sequence: str
    tracks_variant: str
    producer: str
    std_format: str
    n_rows: int | None
    n_keypoints: int | None
    frame_min: int | None
    frame_max: int | None


class SequencePage(StrictModel):
    """One page of a dataset's entries, ordered by group and sequence.

    Attributes:
        items: The page.
        total: How many entries the query matches.
        next_cursor: The cursor of the next page, ``None`` on the last.
    """

    items: tuple[SequenceRow, ...]
    total: int
    next_cursor: str | None


class SubmittedStep(StrictModel):
    """One step of a submitted request.

    Attributes:
        step: The step's id in the recipe.
        runs: The feature slug or op kind it runs.
        execution_id: The attempt the request assigned it.
        lane: The queue lane of its work.
        run_id: The run the plan resolved, ``None`` when it could not.
        status: The plan's status for the step when it was submitted.
    """

    step: str
    runs: str
    execution_id: str
    lane: str
    run_id: str | None
    status: ArtifactStatus


class SubmittedPipeline(StrictModel):
    """A submitted request and the process running it.

    Attributes:
        request_id: The request, which ``get_request_status`` takes.
        recipe_digest: The recipe the request names.
        plan_fingerprint: The fingerprint of the plan it was submitted under.
        steps: Each step, in topological order.
        duplicates: The open requests of the same recipe over an overlapping
            scope.
        driver_pid: The process driving the request.
        driver_log: The log that the driver and its steps append to.
        poll_after_seconds: How long to wait before polling the status.
    """

    request_id: str
    recipe_digest: str
    plan_fingerprint: str
    steps: tuple[SubmittedStep, ...]
    duplicates: tuple[str, ...]
    driver_pid: int
    driver_log: str
    poll_after_seconds: int


class RunSummary(StrictModel):
    """One attempt in a listing, as its run-log reports it.

    Attributes:
        execution_id: The attempt.
        kind: Its job kind.
        target: Its run-log target, the storage name for a feature run.
        run_id: The run it recorded producing, ``""`` before it named one.
        status: Its run-log status.
        outcome: ``status``, with ``partial`` for a finished attempt that lost
            entries.
        owner: Who started it, ``""`` when unrecorded.
        created_at: When it was created.
        started_at: When its process started, ``""`` before.
        finished_at: When it ended, ``""`` before.
        entries_failed: How many entities it lost.
        entries_written: How many entries hold its output.
        cache_hit: Whether it found its work already done.
        error: The failure it recorded, ``None`` when none.
        host: The host its process recorded, ``""`` when unrecorded.
        heartbeat_at: When it last wrote its run-log.
        phase: The last phase it declared, such as ``fit``, ``""`` before one.
        liveness: Whether its process may still be running: ``alive``,
            ``dead``, ``ended``, or ``unknown`` for a process on another host
            whose run-log declares no heartbeat.
    """

    execution_id: str
    kind: str
    target: str
    run_id: str
    status: str
    outcome: AttemptOutcome
    owner: str
    created_at: str
    started_at: str
    finished_at: str
    entries_failed: int
    entries_written: int
    cache_hit: bool
    error: RecordedErrorWire | None
    host: str = ""
    heartbeat_at: str = ""
    phase: str = ""
    liveness: AttemptLiveness | None = None


class ActivityPage(StrictModel):
    """One page of a dataset's requests or attempts, newest first.

    Attributes:
        view: Which records the page lists.
        requests: The requests, under the ``requests`` view.
        runs: The attempts, under the ``runs`` view.
        next_cursor: The cursor of the next page, ``None`` on the last.
    """

    view: ActivityView
    requests: tuple[RequestSummaryWire, ...] = ()
    runs: tuple[RunSummary, ...] = ()
    next_cursor: str | None


class AttemptDiagnosis(StrictModel):
    """One failed or partial attempt, with what it lost and what answers it.

    Attributes:
        step: The attempt's step in its request, ``""`` outside a request.
        execution_id: The attempt.
        outcome: How it stands.
        error: The failure it recorded, ``None`` when none.
        entry_errors: The entities it lost with their failure records, ``None``
            when it lost none.
        allow_partial_applies: Whether resubmitting with ``allow_partial``
            answers the failure: a coverage shortfall, or a quarantined entity.
    """

    step: str
    execution_id: str
    outcome: AttemptOutcome
    error: RecordedErrorWire | None
    entry_errors: EntryErrorsWire | None
    allow_partial_applies: bool


class FailureDiagnosis(StrictModel):
    """The failures of one request or one attempt.

    Attributes:
        request_id: The request diagnosed, ``None`` for one attempt.
        verdict: The request's verdict, ``None`` for one attempt.
        attempts: Each attempt that failed or lost entries, in step order.
    """

    request_id: str | None
    verdict: RequestVerdict | None
    attempts: tuple[AttemptDiagnosis, ...]


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A pydantic model named by its module and name, imported on use.

    Attributes:
        module: The module that defines the model.
        name: The model's name in the module.
    """

    module: str
    name: str

    def resolve(self) -> type[BaseModel]:
        """Import the module and return the model.

        Raises:
            TypeError: The name is bound to something other than a pydantic model.
        """
        found: object = getattr(importlib.import_module(self.module), self.name)
        if isinstance(found, type) and issubclass(found, BaseModel):
            return found
        msg = f"{self.module}.{self.name} is not a pydantic model"
        raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class ToolAnnotations:
    """The hints that a client reads about a tool's effects.

    Attributes:
        read_only: The tool changes nothing.
        destructive: A change it makes can lose work.
        idempotent: A second identical call changes nothing more.
        open_world: The tool reaches beyond the datasets it serves. False for
            every shared tool.
    """

    read_only: bool
    destructive: bool
    idempotent: bool
    open_world: bool = False


READ: Final = ToolAnnotations(read_only=True, destructive=False, idempotent=True)
"""The annotations of a tool that only reads."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One shared tool, declared once for every server and backend.

    Attributes:
        name: The tool's name.
        title: A short name a client displays.
        description: What the tool does, written for a model.
        input_model: The model its arguments validate against.
        output: Its output model, or a reference to one whose module is outside
            this module's import budget.
        annotations: Its effects, as hints.
        confirm_when: The conditions under which it refuses until called with
            ``confirm: true``. Empty when it never asks.
        narrowing: The arguments that make its result smaller, which a refusal
            for a result above :data:`MAX_RESULT_CHARS` names.
        always_load: Whether a client should load the tool up front rather
            than on search, for the tools of the pipeline workflow.
    """

    name: ToolName
    title: str
    description: str
    input_model: type[ToolInput]
    output: type[BaseModel] | ModelRef
    annotations: ToolAnnotations
    confirm_when: tuple[ConfirmationCondition, ...] = ()
    narrowing: tuple[str, ...] = ()
    always_load: bool = False

    @property
    def needs_confirmation(self) -> bool:
        """Whether a call may be refused until the user confirms it."""
        return bool(self.confirm_when)

    def output_model(self) -> type[BaseModel]:
        """Return the output model, importing its module when it is referenced."""
        output = self.output
        return output.resolve() if isinstance(output, ModelRef) else output

    def input_schema(self) -> dict[str, JsonValue]:
        """Return the JSON Schema that a client is given for the arguments.

        The input model's validation schema without its ``title`` keywords,
        which repeat the field names. The descriptions stay, because they tell a
        model what each argument means.
        """
        schema: dict[str, JsonValue] = self.input_model.model_json_schema()
        return strip_schema_keywords(schema, _INPUT_DROPPED_KEYWORDS)

    def output_schema(self) -> dict[str, JsonValue]:
        """Return the JSON Schema that the structured result of a call validates against.

        The output model's serialization schema without its ``title``,
        ``description`` and ``default`` keywords. The tool's description says
        what the result reports, and a result contains every field. The schema
        of the largest output then stays under :data:`MAX_SCHEMA_BYTES`.
        """
        schema: dict[str, JsonValue] = self.output_model().model_json_schema(
            mode="serialization"
        )
        return strip_schema_keywords(schema, _OUTPUT_DROPPED_KEYWORDS)


_INPUT_DROPPED_KEYWORDS: Final = frozenset({"title"})
"""The schema keywords left out of an input schema."""

_OUTPUT_DROPPED_KEYWORDS: Final = frozenset({"title", "description", "default"})
"""The schema keywords left out of an output schema."""

_PAGING = (
    " Returns one page. Pass next_cursor back as cursor to read the next, until "
    "it is null."
)

TOOLS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="list_datasets",
        title="List datasets",
        description=(
            "List the mosaic datasets this server serves, each with the uuid that "
            "every other dataset tool takes, its name and its number of sequences."
            + _PAGING
        ),
        input_model=ListDatasetsInput,
        output=DatasetList,
        annotations=READ,
        narrowing=("limit",),
    ),
    ToolSpec(
        name="describe_dataset",
        title="Describe a dataset",
        description=(
            "Describe one dataset: its manifest fields, notes and tags, the counts "
            "of sequences, videos, tracks variants, feature runs and trained models, "
            "the track schemas of its tables and the frame rates of its videos."
        ),
        input_model=DescribeDatasetInput,
        output=ModelRef("mosaic.core.dataset_summary", "DatasetSummary"),
        annotations=READ,
    ),
    ToolSpec(
        name="list_sequences",
        title="List sequences",
        description=(
            "List a dataset's entries, (group, sequence), with the tracks table "
            "each resolves to: its variant, producer, schema, rows, keypoints and "
            "frame extent." + _PAGING
        ),
        input_model=ListSequencesInput,
        output=SequencePage,
        annotations=READ,
        narrowing=("group", "tracks_variant", "limit"),
    ),
    ToolSpec(
        name="get_inventory",
        title="Get the inventory",
        description=(
            "Report what a dataset holds: each computed artifact with its kind, "
            "status, coverage and recorded params. By default only artifacts that "
            "are not complete are listed, without params. Pass a listed ref to "
            "report one artifact in depth, with its params." + _PAGING
        ),
        input_model=GetInventoryInput,
        output=ModelRef("mosaic.core.pipeline.inventory", "InventoryWire"),
        annotations=READ,
        narrowing=("kinds", "entries", "status_not", "ref", "limit"),
    ),
    ToolSpec(
        name="describe_step_types",
        title="Describe step types",
        description=(
            "Find and describe the features and ops a recipe may run. A search "
            "by words in their name, summary or category returns them best "
            "match first, each with its lane, the columns it reads and writes, "
            "what this installation lacks to run it and any license "
            f"restriction. names (up to {MAX_DESCRIBED_STEP_TYPES}) returns each "
            "one's params as a flat table: dotted paths for nested fields, "
            "type, default, constraints and meaning. A field marked reference "
            '\'result\' takes {"step": "<id>"} naming another step, with an '
            'optional pattern. Any string field may take {"step": "<id>"}, '
            "which substitutes that step's run or model id. detail 'schema' "
            "adds the params JSON Schema." + _PAGING
        ),
        input_model=DescribeStepTypesInput,
        output=ModelRef("mosaic.agent.mcp.outputs", "StepTypes"),
        annotations=READ,
        narrowing=("names", "detail", "limit"),
        always_load=True,
    ),
    ToolSpec(
        name="validate_recipe",
        title="Validate a recipe",
        description=(
            "Check a recipe against the step types, without a dataset. Returns "
            "every problem and caution with the step and field it is at. A recipe "
            "that is not valid is reported, not refused."
        ),
        input_model=ValidateRecipeInput,
        output=ModelRef("mosaic.core.pipeline.graph", "ValidationReport"),
        annotations=READ,
    ),
    ToolSpec(
        name="suggest_steps",
        title="Suggest steps",
        description=(
            "Suggest what may be added to a recipe: step types to attach below a "
            "step, step types to insert above a step's readers, or inputs to add "
            "to a step. Each allowed step type comes with the step to add, "
            "complete but for its id. Refused ones are only counted."
        ),
        input_model=SuggestStepsInput,
        output=ModelRef("mosaic.agent.mcp.outputs", "StepSuggestions"),
        annotations=READ,
        narrowing=("step", "mode"),
    ),
    ToolSpec(
        name="plan_recipe",
        title="Plan a recipe",
        description=(
            "Resolve a recipe against a dataset without running it: each step's "
            "run id, status, coverage, entries asked, lane and why it waits, the "
            "problems that refuse it, and the plan fingerprint that submit_pipeline "
            "requires. Show the user the plan before submitting."
        ),
        input_model=PlanRecipeInput,
        output=ModelRef("mosaic.core.pipeline.graph", "PlanWire"),
        annotations=READ,
        narrowing=("entries",),
        always_load=True,
    ),
    ToolSpec(
        name="submit_pipeline",
        title="Submit a pipeline",
        description=(
            "Submit a planned recipe and start running it in the background. "
            "Requires the fingerprint of a current plan_recipe result. Returns the "
            "request id and how long to wait before polling get_request_status. "
            "Refused with reason confirmation_required when the work is costly or "
            "a choice is the user's. Ask the user, then call again with confirm "
            "true. Do not set confirm without the user's agreement."
        ),
        input_model=SubmitPipelineInput,
        output=SubmittedPipeline,
        annotations=ToolAnnotations(
            read_only=False, destructive=False, idempotent=False
        ),
        confirm_when=(
            "gpu_lane",
            "training_op",
            "tracking_op",
            "allow_partial",
            "bind",
            "large_scope",
            "whole_dataset",
            "open_duplicate",
        ),
        always_load=True,
    ),
    ToolSpec(
        name="get_request_status",
        title="Get a request's status",
        description=(
            "Report how far a submitted request got: its verdict, poll_after_seconds "
            "(wait that long before polling again), and per step the outcome, "
            "progress, phase, liveness, recorded error and a sample of lost "
            "entries. A request reads 'orphaned' when its driver stopped or its "
            "step's process is provably gone (pid exited, host rebooted, or no "
            "heartbeat for 10 minutes). The user resumes it with mosaic pipeline "
            "run --manifest <manifest> --request <request_id>, where <manifest> "
            "is the path describe_dataset reports."
        ),
        input_model=GetRequestStatusInput,
        output=RequestStatusWire,
        annotations=READ,
        always_load=True,
    ),
    ToolSpec(
        name="cancel_request",
        title="Cancel a request",
        description=(
            "Cancel a request that has not ended. Its running step is stopped and "
            "its unstarted steps are recorded cancelled. Refused with reason "
            "confirmation_required when a running step loses its work. Ask the "
            "user, then call again with confirm true."
        ),
        input_model=CancelRequestInput,
        output=RequestCancel,
        annotations=ToolAnnotations(read_only=False, destructive=True, idempotent=True),
        confirm_when=("hard_cancel",),
        always_load=True,
    ),
    ToolSpec(
        name="list_activity",
        title="List activity",
        description=(
            "List a dataset's pipeline requests with their verdicts, or every "
            "attempt with its outcome, newest first." + _PAGING
        ),
        input_model=ListActivityInput,
        output=ActivityPage,
        annotations=READ,
        narrowing=("limit",),
    ),
    ToolSpec(
        name="diagnose_failures",
        title="Diagnose failures",
        description=(
            "Explain why a request or an attempt failed: each failed attempt's "
            "error and refusal, the entities it lost with their attempts, backoff "
            "and quarantine, and whether resubmitting with allow_partial answers "
            "it. The tool changes nothing. Propose the remedies to the user."
        ),
        input_model=DiagnoseFailuresInput,
        output=FailureDiagnosis,
        annotations=READ,
        narrowing=("execution_id",),
    ),
    ToolSpec(
        name="preview_output",
        title="Preview an output",
        description=(
            "Preview one entry's table of a feature run, a tracks variant or a "
            "labels variant: the columns read with their statistics, its row "
            "count and frame extent, the first rows, and the path of the whole "
            "file. A wide table reads its identity and bookkeeping columns first "
            "and names the rest as omitted. A labels variant also counts each "
            "class's events and the frames labeled."
        ),
        input_model=PreviewOutputInput,
        output=ModelRef("mosaic.core.pipeline.preview", "ArtifactPreview"),
        annotations=READ,
        narrowing=("columns", "max_columns", "limit"),
    ),
)
"""The shared tools, in the order a server lists them."""

_BY_NAME: Final[Mapping[str, ToolSpec]] = {spec.name: spec for spec in TOOLS}


def tool_spec(name: str) -> ToolSpec | None:
    """Return the shared tool named *name*, ``None`` when the set has none."""
    return _BY_NAME.get(name)


type ToolResult[T: BaseModel] = T | ErrorEnvelope
"""What a backend returns for a call: the tool's output, or why it failed."""


class ToolBackend(Protocol):
    """What answers the shared tools: one method per tool, named for it.

    Each method takes the tool's validated input and returns its output model or
    an :class:`ErrorEnvelope`. A method raises only for a fault of the backend.
    """

    def list_datasets(self, args: ListDatasetsInput) -> ToolResult[DatasetList]:
        """Answer ``list_datasets``."""
        ...

    def describe_dataset(
        self, args: DescribeDatasetInput
    ) -> ToolResult[DatasetSummary]:
        """Answer ``describe_dataset``."""
        ...

    def list_sequences(self, args: ListSequencesInput) -> ToolResult[SequencePage]:
        """Answer ``list_sequences``."""
        ...

    def get_inventory(self, args: GetInventoryInput) -> ToolResult[InventoryWire]:
        """Answer ``get_inventory``."""
        ...

    def describe_step_types(
        self, args: DescribeStepTypesInput
    ) -> ToolResult[StepTypes]:
        """Answer ``describe_step_types``."""
        ...

    def validate_recipe(
        self, args: ValidateRecipeInput
    ) -> ToolResult[ValidationReport]:
        """Answer ``validate_recipe``."""
        ...

    def suggest_steps(self, args: SuggestStepsInput) -> ToolResult[StepSuggestions]:
        """Answer ``suggest_steps``."""
        ...

    def plan_recipe(self, args: PlanRecipeInput) -> ToolResult[PlanWire]:
        """Answer ``plan_recipe``."""
        ...

    def submit_pipeline(
        self, args: SubmitPipelineInput
    ) -> ToolResult[SubmittedPipeline]:
        """Answer ``submit_pipeline``."""
        ...

    def get_request_status(
        self, args: GetRequestStatusInput
    ) -> ToolResult[RequestStatusWire]:
        """Answer ``get_request_status``."""
        ...

    def cancel_request(self, args: CancelRequestInput) -> ToolResult[RequestCancel]:
        """Answer ``cancel_request``."""
        ...

    def list_activity(self, args: ListActivityInput) -> ToolResult[ActivityPage]:
        """Answer ``list_activity``."""
        ...

    def diagnose_failures(
        self, args: DiagnoseFailuresInput
    ) -> ToolResult[FailureDiagnosis]:
        """Answer ``diagnose_failures``."""
        ...

    def preview_output(self, args: PreviewOutputInput) -> ToolResult[ArtifactPreview]:
        """Answer ``preview_output``."""
        ...


def call_tool(
    backend: ToolBackend, name: str, arguments: Mapping[str, object]
) -> BaseModel:
    """Validate *arguments* for the tool *name* and answer it through *backend*.

    Args:
        backend: What answers the tool.
        name: The tool's name.
        arguments: The call's arguments, as a client sent them.

    Returns:
        The tool's output, or an :class:`ErrorEnvelope`: ``not_found`` for an
        unknown tool, ``invalid`` for arguments the input model refuses, or the
        backend's own.
    """
    spec = tool_spec(name)
    if spec is None:
        return unknown_tool(name)
    try:
        request = spec.input_model.model_validate(dict(arguments))
    except ValidationError as error:
        return _invalid_arguments(spec.name, error)
    return _bounded(spec, _dispatch(backend, request))


def _bounded(spec: ToolSpec, output: BaseModel) -> BaseModel:
    """Return *output*, or :func:`result_too_large` for a read above the budget.

    A tool that changes a dataset has done so by the time it returns, so its
    result is returned whatever its size.
    """
    if isinstance(output, ErrorEnvelope) or not spec.annotations.read_only:
        return output
    chars = len(output.model_dump_json())
    if chars <= MAX_RESULT_CHARS:
        return output
    return result_too_large(spec, chars)


def result_too_large(spec: ToolSpec, chars: int) -> ErrorEnvelope:
    """Refuse a result of *chars* characters, naming the arguments that narrow it."""
    narrow_with: list[JsonValue] = list(spec.narrowing)
    advice = (
        f"Call again narrowed by {', '.join(spec.narrowing)}."
        if spec.narrowing
        else "No argument narrows it. The mosaic CLI prints it whole."
    )
    return ErrorEnvelope(
        kind="invalid",
        reason=RESULT_TOO_LARGE,
        message=(
            f"{spec.name} would return {chars} characters, above the "
            f"{MAX_RESULT_CHARS} a call returns. {advice}"
        ),
        detail={
            "tool": spec.name,
            "chars": chars,
            "budget": MAX_RESULT_CHARS,
            "narrow_with": narrow_with,
        },
    )


def _dispatch(backend: ToolBackend, request: ToolInput) -> BaseModel:
    """Call the method of *backend* that answers *request*'s tool."""
    match request:
        case ListDatasetsInput():
            return backend.list_datasets(request)
        case DescribeDatasetInput():
            return backend.describe_dataset(request)
        case ListSequencesInput():
            return backend.list_sequences(request)
        case GetInventoryInput():
            return backend.get_inventory(request)
        case DescribeStepTypesInput():
            return backend.describe_step_types(request)
        case ValidateRecipeInput():
            return backend.validate_recipe(request)
        case SuggestStepsInput():
            return backend.suggest_steps(request)
        case PlanRecipeInput():
            return backend.plan_recipe(request)
        case SubmitPipelineInput():
            return backend.submit_pipeline(request)
        case GetRequestStatusInput():
            return backend.get_request_status(request)
        case CancelRequestInput():
            return backend.cancel_request(request)
        case ListActivityInput():
            return backend.list_activity(request)
        case DiagnoseFailuresInput():
            return backend.diagnose_failures(request)
        case PreviewOutputInput():
            return backend.preview_output(request)
        case _:
            msg = f"no tool takes {type(request).__name__}"
            raise TypeError(msg)


def unknown_tool(name: str) -> ErrorEnvelope:
    """Report a call naming no shared tool, and the tool that replaced a merged one."""
    replacement = MERGED_TOOLS.get(name)
    if replacement is None:
        return ErrorEnvelope(
            kind="not_found",
            reason=UNKNOWN_TOOL,
            message=f"no tool named {name!r}",
            detail={"tool": name},
        )
    return ErrorEnvelope(
        kind="not_found",
        reason=UNKNOWN_TOOL,
        message=f"no tool named {name!r}; {replacement} replaced it",
        detail={"tool": name, "replacement": replacement},
    )


def _invalid_arguments(name: ToolName, error: ValidationError) -> ErrorEnvelope:
    """Report the arguments of a call to *name* that its input model refuses."""
    errors: list[JsonValue] = [
        {
            "field": ".".join(str(part) for part in item["loc"]),
            "message": item["msg"],
        }
        for item in error.errors(include_url=False)
    ]
    message = (
        f"the arguments of {name} do not fit its input: "
        f"{error.error_count()} problem(s)"
    )
    return ErrorEnvelope(
        kind="invalid",
        reason=INVALID_ARGUMENTS,
        message=message,
        detail={"tool": name, "errors": errors},
    )


@dataclass(frozen=True, slots=True)
class ConfirmationFinding:
    """One confirmation condition that a call meets, and where.

    Attributes:
        condition: The condition.
        message: What in the call meets it, naming the step or the count.
    """

    condition: ConfirmationCondition
    message: str


def confirmation_required(
    name: ToolName, findings: Sequence[ConfirmationFinding]
) -> ErrorEnvelope:
    """Refuse a call to *name* until the user confirms *findings*.

    Args:
        name: The tool refused.
        findings: Each condition the call meets, at least one.

    Returns:
        An ``invalid`` envelope with reason :data:`CONFIRMATION_REQUIRED`, whose
        detail lists each condition and what meets it.
    """
    conditions: list[JsonValue] = [finding.condition for finding in findings]
    listed: list[JsonValue] = [
        {
            "condition": finding.condition,
            "meaning": CONFIRMATION_CONDITIONS[finding.condition],
            "message": finding.message,
        }
        for finding in findings
    ]
    sentences = "; ".join(finding.message for finding in findings)
    message = (
        f"{name} needs the user's confirmation: {sentences}. Ask the user, and "
        "call again with confirm true if they agree."
    )
    return ErrorEnvelope(
        kind="invalid",
        reason=CONFIRMATION_REQUIRED,
        message=message,
        detail={"tool": name, "conditions": conditions, "findings": listed},
    )


def confirmation_declined(name: ToolName) -> ErrorEnvelope:
    """Report a call to *name* that the user declined when asked to confirm it."""
    message = (
        f"the user declined to confirm {name}, which did not run. Do not call it "
        "again unless the user asks for it."
    )
    return ErrorEnvelope(
        kind="invalid",
        reason=CONFIRMATION_DECLINED,
        message=message,
        detail={"tool": name},
    )


def dataset_managed(
    name: ToolName, dataset: str, managed_by: JsonValue
) -> ErrorEnvelope:
    """Refuse a call to *name* that would change a dataset a service manages."""
    message = (
        f"dataset {dataset} is managed by {managed_by!r}, and {name} would change "
        "it. Make the change through the managing service."
    )
    return ErrorEnvelope(
        kind="unavailable",
        reason=DATASET_MANAGED,
        message=message,
        detail={"tool": name, "dataset": dataset, "managed_by": managed_by},
    )


def unknown_dataset(dataset: str) -> ErrorEnvelope:
    """Report a call naming a dataset that the server does not serve."""
    message = (
        f"this server serves no dataset {dataset!r}; list_datasets names the "
        "datasets it serves"
    )
    return ErrorEnvelope(
        kind="not_found",
        reason=UNKNOWN_DATASET,
        message=message,
        detail={"dataset": dataset},
    )


def invalid_cursor(cursor: str) -> ErrorEnvelope:
    """Report a listing given a cursor that no page of it returned."""
    return ErrorEnvelope(
        kind="invalid",
        reason=INVALID_CURSOR,
        message=f"cursor {cursor!r} was not returned by this listing",
        detail={"field": "cursor"},
    )
