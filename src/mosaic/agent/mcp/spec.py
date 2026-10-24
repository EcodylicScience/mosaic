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
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.artifact_kinds import ArtifactKind, ArtifactStatus
from mosaic.core.pipeline.cancel import RequestCancel
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import (
    MAX_WAIT_SECONDS,
    SCHEMA_VERSION,
    BoundRef,
    CleanupMode,
    EntryErrorsWire,
    RecordedErrorWire,
    RequestStatusWire,
    RequestSummaryWire,
    RequestVerdict,
    strip_schema_keywords,
)
from mosaic.core.pipeline.graph.digest import DIGEST_LENGTH
from mosaic.core.pipeline.graph.patch import PatchOp
from mosaic.core.pipeline.parallelism import MAX_WORKERS_CEILING, ParallelMode
from mosaic.core.strict_model import StrictModel
from mosaic.runlog import AttemptLiveness, AttemptOutcome

if TYPE_CHECKING:
    from mosaic.core.dataset_summary import DatasetSummary
    from mosaic.core.pipeline.graph import ValidationReport
    from mosaic.core.pipeline.inventory import InventoryWire
    from mosaic.core.pipeline.preview import ArtifactPreview

    from .outputs import PatchedRecipe, PlanPage, StepSuggestions, StepTypes

__all__ = [
    "CLEANUP_NOT_ALLOWED",
    "CONFIRMATION_CONDITIONS",
    "CONFIRMATION_DECLINED",
    "CONFIRMATION_REQUIRED",
    "CONFIRMATION_TAG",
    "CONFIRMATION_UNANSWERED",
    "CONFIRMATION_UNAVAILABLE",
    "CONFIRM_FIELD",
    "DATASET_MANAGED",
    "DEFAULT_PAGE_SIZE",
    "INVALID_ARGUMENTS",
    "INVALID_CURSOR",
    "LISTED_STATUS_NOT",
    "LISTING_MISSING_SAMPLE",
    "MAX_DESCRIBED_STEP_TYPES",
    "MAX_PAGE_SIZE",
    "MAX_PATCH_OPS",
    "MAX_PREVIEW_ROWS",
    "MAX_RESULT_CHARS",
    "MAX_ASSUMED_GROUPS",
    "MAX_RETIRED_RECIPES",
    "MAX_SCHEMA_BYTES",
    "MAX_STEP_TYPE_DETAIL_CHARS",
    "MAX_WAIT_SECONDS",
    "MERGED_TOOLS",
    "PLAN_MISSING_SAMPLE",
    "PLAN_RETIRABLE_SAMPLE",
    "PLAN_RETIRED_SAMPLE",
    "PLAN_SUPERSEDED_SAMPLE",
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
    "DatasetAction",
    "DatasetChange",
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
    "ManageDatasetInput",
    "ModelRef",
    "PatchRecipeInput",
    "PlanRecipeInput",
    "PreviewOutputInput",
    "RunSummary",
    "ScopeSelector",
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
    "cleanup_not_allowed",
    "confirmation_declined",
    "confirmation_required",
    "confirmation_tag",
    "confirmation_unanswered",
    "confirmation_unavailable",
    "dataset_managed",
    "invalid_cursor",
    "read_confirmation_tag",
    "result_too_large",
    "tool_spec",
    "unknown_dataset",
    "unknown_tool",
]

SHARED_TOOL_LIMIT: Final = 16
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

PLAN_SUPERSEDED_SAMPLE: Final = 5
"""The superseded runs that ``plan_recipe`` lists per step, removable ones first.

Each takes about 400 characters, and a session that tunes a chain leaves dozens
in every storage of it, so the core default of 50 would put a plan over
:data:`MAX_RESULT_CHARS`, which no argument narrows. The totals count them all.
"""

PLAN_RETIRED_SAMPLE: Final = 5
"""The retired runs of storages no step names that ``plan_recipe`` lists.

They take about 400 characters each and no argument narrows the list, as for
:data:`PLAN_SUPERSEDED_SAMPLE`. The totals count them all.
"""

PLAN_RETIRABLE_SAMPLE: Final = 8
"""The saved recipes that ``plan_recipe`` lists as retirable, newest activity first.

Each takes about 300 characters with its request ids. A session that iterates a
recipe saves dozens.
"""

MAX_RETIRED_RECIPES: Final = 8
"""The most recipes one call retires."""

MAX_PATCH_OPS: Final = 64
"""The most edits one ``patch_recipe`` call applies."""

MAX_ASSUMED_GROUPS: Final = 16
"""The most groups one ``plan_recipe`` call assumes added."""

LISTING_MISSING_SAMPLE: Final = 5
"""The missing keys that a ``get_inventory`` listing names per artifact."""

MAX_DESCRIBED_STEP_TYPES: Final = 8
"""The most step types one ``describe_step_types`` call describes by name."""

MAX_STEP_TYPE_DETAIL_CHARS: Final = 16_000
"""The most characters of JSON that one step type's detail takes.

Not enforced per call. A test measures every step type of the catalog, and the
largest, ``vame``, takes about 15,700.
"""

LISTED_STATUS_NOT: Final[tuple[ArtifactStatus, ...]] = ("complete",)
"""The statuses that an inventory listing leaves out when its call names none."""

CONFIRMATION_REQUIRED: Final = "confirmation_required"
"""The ``reason`` of a refusal that a call with ``confirm: true`` answers."""

CONFIRMATION_DECLINED: Final = "confirmation_declined"
"""The ``reason`` of a call whose confirmation was declined when the server asked."""

CONFIRMATION_UNANSWERED: Final = "confirmation_unanswered"
"""The ``reason`` of a call whose confirmation nobody answered when the server asked.

A client that cannot show the question, such as one run without a person,
cancels it, and the call does not run.
"""

CONFIRMATION_UNAVAILABLE: Final = "confirmation_unavailable"
"""The ``reason`` of a call that needs a confirmation this client cannot ask for.

A server that does not take ``confirm`` from a model, such as one served over
HTTP without ``--allow-model-confirm``, refuses the call when its client cannot
ask the user.
"""

CONFIRMATION_TAG: Final = "mosaic-confirm"
"""The name of the tag that ends a confirmation question.

The question's last line is ``[mosaic-confirm tool=<tool> conditions=<a>,<b>]``,
naming the tool and its conditions sorted, which a client's hook reads to answer
by condition (:func:`read_confirmation_tag`). The grammar is a contract with
those hooks.
"""

CONFIRM_FIELD: Final = "confirm"
"""The one boolean field of the form that asks to confirm a call."""

_TAG_LINE: Final = re.compile(
    rf"\[{CONFIRMATION_TAG} tool=([a-z_]+) conditions=([a-z_]+(?:,[a-z_]+)*)\]\s*"
)
"""The last line of a confirmation question: the tag and nothing else."""

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

CLEANUP_NOT_ALLOWED: Final = "cleanup_not_allowed"
"""The ``reason`` of a cleanup the server was not started to allow."""

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
    "patch_recipe",
    "plan_recipe",
    "submit_pipeline",
    "get_request_status",
    "cancel_request",
    "list_activity",
    "diagnose_failures",
    "preview_output",
    "manage_dataset",
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
    "open_duplicate",
    "cleanup",
    "score_sealed",
    "hard_cancel",
    "create_dataset",
    "add_source",
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
    "open_duplicate": (
        "a request of this recipe over an overlapping scope has not ended, and "
        "both compute the same runs"
    ),
    "cleanup": (
        "the submission deletes superseded runs of its steps' storage after its "
        "steps finish, and with retire the runs that retired recipes wrote in any "
        "storage"
    ),
    "score_sealed": (
        "a step scores predictions against labels over a group the dataset's "
        "splits seal, which spends the one look a sealed group allows"
    ),
    "hard_cancel": (
        "a running step loses its work when cancelled. A keypoint-MoSeq or VAME "
        "fit stops at once, and training loses the epoch in progress"
    ),
    "create_dataset": "the call creates a dataset: a directory and its manifest",
    "add_source": (
        "the call declares or removes a source, which changes the files the "
        "dataset reads"
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
            "kind, params}. An input is 'tracks' or {step: <id>}. A step or an "
            "input with repeat, as {id: 'xgb_s{seed}', repeat: {seed: [1, 2]}}, "
            "stands for one copy per value with {seed} replaced; '{seed}' alone "
            "takes the value. Or name a recipe the dataset saved (patch_recipe "
            "and submit_pipeline save one): {digest: <digest>} or {saved: <name>}."
        ),
        examples=[RECIPE_EXAMPLE],
    ),
]
"""A recipe document, parsed by the backend so that a malformed one is reported."""


class ScopeSelector(StrictModel):
    """Entries named as ``mosaic.core.scope.Scope`` and ``--step-scope`` name them.

    ``entries`` alone, or ``groups`` and ``sequences``, which cover their cross
    product.
    """

    entries: list[tuple[str, str]] | None = None
    groups: list[str] | None = Field(default=None, min_length=1)
    sequences: list[str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _names_entries(self) -> Self:
        if self.entries is None and self.groups is None and self.sequences is None:
            msg = "a scope names entries, groups or sequences"
            raise ValueError(msg)
        if self.entries is not None and (
            self.groups is not None or self.sequences is not None
        ):
            msg = "a scope names entries, or groups and sequences, not both"
            raise ValueError(msg)
        return self


_SCOPE_FORMS: Final = (
    "[[group, sequence], ...], or {entries: [[group, sequence], ...]}, or "
    "{groups: [...], sequences: [...]} with either left out (their cross product)"
)

Entries = Annotated[
    list[tuple[str, str]] | ScopeSelector | None,
    Field(
        description=(
            f"The entries to cover: {_SCOPE_FORMS}. Null covers every entry of "
            "the dataset, and is what a request with steps scoped to different "
            "groups takes, since a step covers only what entries also covers."
        )
    ),
]
"""The entries a plan or a submission covers."""

StepEntries = Annotated[
    dict[str, list[tuple[str, str]] | ScopeSelector] | None,
    Field(
        description=(
            f"Narrow single steps, keyed by step id, each as {_SCOPE_FORMS}: a "
            "fit on the training groups and its scoring on the test groups in "
            "one request. Each is intersected with entries, so leave entries "
            "null to cover them all. A step reading another through inputs or "
            "tracks must narrow to a subset of it."
        )
    ),
]
"""The scopes narrowing single steps of a plan or a submission."""


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


_REFERENCE_DATASET: Final = (
    "The dataset holding a recipe named by digest or saved name. Not needed for a "
    "recipe sent whole."
)


class ValidateRecipeInput(ToolInput):
    """The arguments of ``validate_recipe``."""

    recipe: RecipeDocument
    dataset: str | None = Field(default=None, description=_REFERENCE_DATASET)


class SuggestStepsInput(ToolInput):
    """The arguments of ``suggest_steps``."""

    recipe: RecipeDocument
    dataset: str | None = Field(default=None, description=_REFERENCE_DATASET)
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


_CLEANUP_DESCRIPTION: Final = (
    "'superseded' also judges the other runs of the recipe's feature storages, "
    "each removable or kept with its reason, under the server's cleanup policy."
)

RetireDigests = Annotated[
    list[Annotated[str, Field(pattern=rf"^[0-9a-f]{{{DIGEST_LENGTH}}}$")]],
    Field(
        max_length=MAX_RETIRED_RECIPES,
        description=(
            "Digests of saved recipes to retire, from the retirable list of a "
            "plan_recipe result with cleanup 'superseded'. The runs their requests "
            "wrote are then judged in any storage, and stop being kept for "
            "belonging to another recipe. It needs cleanup 'superseded' and "
            "applies to this call only: pass it to plan_recipe and again to "
            "submit_pipeline. A recipe with a request that has not ended is "
            "refused."
        ),
    ),
]
"""The recipes a plan or a submission retires, by digest."""


AssumeAdded = Annotated[
    list[str],
    Field(
        max_length=MAX_ASSUMED_GROUPS,
        description=(
            "Groups to assume join the dataset. The cautions then name the steps "
            "that would refit, and those that would only re-score, as the same "
            "request covers them. Changes nothing else."
        ),
    ),
]
"""Groups a plan is asked to assume added, which only adds cautions."""


class PatchRecipeInput(ToolInput):
    """The arguments of ``patch_recipe``."""

    dataset: DatasetId
    base: RecipeDocument
    ops: list[PatchOp] = Field(
        default_factory=list,
        max_length=MAX_PATCH_OPS,
        description=(
            "The edits, applied in order to the expanded base: add_step "
            "{document}, remove_step {step}, set_param {step, path, value} (a "
            "dotted path into params; without value the param is removed) and "
            "replace_step {step, document}. No edits saves the base as it is."
        ),
    )


class PlanRecipeInput(ToolInput):
    """The arguments of ``plan_recipe``."""

    dataset: DatasetId
    recipe: RecipeDocument
    entries: Entries = None
    step_entries: StepEntries = None
    cleanup: CleanupMode = Field(
        default="none",
        description=_CLEANUP_DESCRIPTION
        + " The fingerprint then covers the removable runs.",
    )
    retire: RetireDigests = Field(default_factory=list)
    assume_added: AssumeAdded = Field(default_factory=list)
    limit: PageSize | None = Field(
        default=None,
        description=(
            f"The most steps to return, 1 to {MAX_PAGE_SIZE}. Omitted: every step. "
            "A plan of about 40 steps or more is refused unless paged. The "
            "fingerprint is the whole plan's on every page."
        ),
    )
    cursor: Cursor = None


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
    step_entries: StepEntries = None
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
    cleanup: CleanupMode = Field(
        default="none",
        description=(
            "'superseded' removes the runs the plan_recipe result with cleanup "
            "'superseded' found removable, after every step finished. The server "
            "must allow it."
        ),
    )
    retire: RetireDigests = Field(default_factory=list)
    workers: int | None = Field(
        default=None,
        ge=1,
        le=MAX_WORKERS_CEILING,
        description=(
            "How many entries a parallelizable feature step computes at once. "
            "Omitted: the server's default, which plan_recipe reports in its "
            "parallel block. More than the server's max_workers is refused. Memory "
            "grows with the count, so lower it when a step ends with a dead "
            "worker, a memory floor, or a peak_rss_bytes near the host's memory."
        ),
    )
    parallel_mode: ParallelMode | None = Field(
        default=None,
        description=(
            "'thread' (default) shares one copy of the feature and speeds up "
            "numpy and pandas work. 'process' speeds up only work that holds the "
            "interpreter lock, such as a Python loop, and uses far more memory, "
            "since each worker is an interpreter with a copy of every entry's "
            "table. The server must allow it."
        ),
    )
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
    wait_seconds: int = Field(
        default=0,
        ge=0,
        le=MAX_WAIT_SECONDS,
        description=(
            f"Wait up to this many seconds (at most {MAX_WAIT_SECONDS}) and return "
            "as soon as a step of the request ends, or the request does. 0 returns "
            "at once. Wait with this instead of watching the driver's process."
        ),
    )
    brief: bool = Field(
        default=False,
        description=(
            "List only the steps that are running, failed, cancelled or partial, "
            "and those that ended during the wait. Steps that finished cleanly "
            "and steps not started are counted in omitted_finished and "
            "omitted_pending."
        ),
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


type DatasetAction = Literal[
    "create", "add_source", "remove_source", "scan", "set_tags"
]
"""What ``manage_dataset`` does to a dataset."""

type SourceKindName = Literal["media", "tracks", "labels"]
"""Which raw files a source declares."""

TagValue = str | int | float | bool | None
"""A dataset tag's value."""


class ManageDatasetInput(ToolInput):
    """The arguments of ``manage_dataset``. Each action reads the fields it names."""

    action: DatasetAction = Field(
        description=(
            "create: a new dataset at path, named name. add_source: declare the "
            "kind's files at path. remove_source: undeclare kind's source_id "
            "(its rows stay). scan: index the declared sources of kind, or all; "
            "source_id narrows it to one. set_tags: set tags, remove remove_tags."
        )
    )
    dataset: str | None = Field(
        default=None, description="The dataset's uuid. Every action but create."
    )
    path: str | None = Field(
        default=None,
        description=(
            "create: the new dataset's directory, inside the server's "
            "dataset_roots. add_source: where the files are, inside its "
            "source_roots."
        ),
    )
    name: str | None = Field(default=None, description="create: the dataset's name.")
    kind: SourceKindName | None = None
    source_id: str | None = Field(
        default=None,
        description="The source's id. add_source: default the directory's name.",
    )
    files: list[str] | None = Field(
        default=None,
        description="add_source: only these files, relative to path.",
    )
    patterns: list[str] | None = Field(
        default=None, description="add_source, tracks or labels: file globs."
    )
    src_format: str | None = Field(
        default=None,
        description=(
            "add_source, tracks or labels: the converter that reads the files, "
            "such as calms21_npy."
        ),
    )
    extensions: list[str] | None = Field(
        default=None, description="add_source, media: file suffixes."
    )
    recursive: bool = True
    multi_sequences_per_file: bool = False
    group_from: Literal["filename", "parent"] | None = None
    group_pattern: str | None = None
    tags: dict[str, TagValue] | None = Field(
        default=None,
        description=(
            "set_tags: values by tag name. A tag not defined yet is defined, "
            "typed by its value."
        ),
    )
    remove_tags: list[str] = Field(default_factory=list)
    confirm: bool = Field(
        default=False,
        description=(
            "Set only after the user agreed to the conditions that a refusal "
            "with reason confirmation_required listed."
        ),
    )


class DatasetChange(StrictModel):
    """What ``manage_dataset`` changed.

    Attributes:
        action: The action done.
        dataset: The dataset's uuid.
        name: The dataset's name.
        manifest: The manifest file.
        message: What changed, in a sentence.
        scanned: Under ``scan``, the raw index rows of each kind scanned.
        orphaned_rows: Under ``remove_source``, the index rows the source
            claimed, which stay.
        next_steps: What the dataset needs before a pipeline runs on it.
    """

    action: DatasetAction
    dataset: str
    name: str
    manifest: str
    message: str
    scanned: dict[str, int] = Field(default_factory=dict)
    orphaned_rows: int | None = None
    next_steps: tuple[str, ...] = ()


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
        driver_pid: The process driving the request, for the user. A caller waits
            with ``get_request_status`` and ``wait_seconds``, never on this pid.
        driver_log: The log that the driver and its steps append to: their
            output, which explains a failure and does not report progress.
        poll_after_seconds: How long to wait before polling the status, for a
            caller that does not pass ``wait_seconds``.
        cleanup_runs: How many superseded runs the request removes after its
            steps, 0 without a cleanup.
        cleanup_bytes: The bytes they hold.
        workers: How many entries a parallelizable step computes at once.
        parallel_mode: How it computes them. A feature that defines
            ``set_scope`` runs in threads whatever this is.
    """

    request_id: str
    recipe_digest: str
    plan_fingerprint: str
    steps: tuple[SubmittedStep, ...]
    duplicates: tuple[str, ...]
    driver_pid: int
    driver_log: str
    poll_after_seconds: int
    cleanup_runs: int = 0
    cleanup_bytes: int = 0
    workers: int = 1
    parallel_mode: ParallelMode = "thread"


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
        name="patch_recipe",
        title="Patch a recipe",
        description=(
            "Change a recipe by edits and save the result in the dataset, so "
            "that the other recipe tools take it as {digest: <digest>} instead "
            "of the whole document. Returns its digest, its number of steps "
            "after repeat is expanded, and its validation report. A recipe that "
            "parses is saved even when it has problems, so it can be patched "
            "again. Saving changes no data and runs nothing."
        ),
        input_model=PatchRecipeInput,
        output=ModelRef("mosaic.agent.mcp.outputs", "PatchedRecipe"),
        # It writes one content-addressed file, which the same call writes again
        # with the same bytes.
        annotations=ToolAnnotations(
            read_only=False, destructive=False, idempotent=True
        ),
    ),
    ToolSpec(
        name="plan_recipe",
        title="Plan a recipe",
        description=(
            "Resolve a recipe against a dataset without running it: each step's "
            "run id, status, coverage, entries asked, lane and why it waits, the "
            "problems that refuse it, the cautions to show the user, and the plan "
            "fingerprint that submit_pipeline requires. Show the user the plan "
            "and its cautions before submitting. With cleanup 'superseded' it also "
            "lists the saved recipes that retire can reclaim runs from. A plan of "
            "about 40 steps or more is refused as too large: page it with limit "
            "(about 25) and cursor, and show the user every page."
        ),
        input_model=PlanRecipeInput,
        output=ModelRef("mosaic.agent.mcp.outputs", "PlanPage"),
        annotations=READ,
        narrowing=("limit",),
        always_load=True,
    ),
    ToolSpec(
        name="submit_pipeline",
        title="Submit a pipeline",
        description=(
            "Submit a planned recipe and start running it in the background. "
            "Requires the fingerprint of a current plan_recipe result, planned "
            "with the same cleanup and retire. Returns the request id. Wait for "
            "the request with get_request_status and wait_seconds. "
            "When the work is costly or a choice is the user's, a client that can "
            "ask the user asks. Otherwise the call is refused with reason "
            "confirmation_required: ask the user, then call again with confirm "
            "true. Do not set confirm without the user's agreement. After "
            "confirmation_unanswered, confirmation_declined or "
            "confirmation_unavailable, stop and report."
        ),
        input_model=SubmitPipelineInput,
        output=SubmittedPipeline,
        # Destructive: with cleanup 'superseded' it removes feature runs.
        annotations=ToolAnnotations(
            read_only=False, destructive=True, idempotent=False
        ),
        confirm_when=(
            "gpu_lane",
            "training_op",
            "tracking_op",
            "allow_partial",
            "bind",
            "open_duplicate",
            "cleanup",
            "score_sealed",
        ),
        always_load=True,
    ),
    ToolSpec(
        name="get_request_status",
        title="Get a request's status",
        description=(
            "Report how far a submitted request got: its verdict, and per step the "
            "outcome, progress, phase, liveness, recorded error and a sample of "
            "lost entries. To wait for it, pass wait_seconds: the call returns as "
            "soon as a step ends or the request does, or after that many seconds. "
            "Do not poll the driver's process. brief lists only the steps that "
            "need attention. Without wait_seconds, wait poll_after_seconds before "
            "reading again. A request reads 'orphaned' when its driver stopped or "
            "its step's process is provably gone (pid exited, host rebooted, or no "
            "heartbeat for 10 minutes). The user resumes it with mosaic pipeline "
            "run --manifest <manifest> --request <request_id>, where <manifest> "
            "is the path describe_dataset reports."
        ),
        input_model=GetRequestStatusInput,
        output=RequestStatusWire,
        annotations=READ,
        narrowing=("brief",),
        always_load=True,
    ),
    ToolSpec(
        name="cancel_request",
        title="Cancel a request",
        description=(
            "Cancel a request that has not ended. Its running step is stopped and "
            "its unstarted steps are recorded cancelled. When a running step "
            "loses its work, a client that can ask the user asks. Otherwise the "
            "call is refused with reason confirmation_required: ask the user, "
            "then call again with confirm true. After confirmation_unanswered, "
            "confirmation_declined or confirmation_unavailable, stop and report."
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
    ToolSpec(
        name="manage_dataset",
        title="Manage a dataset",
        description=(
            "Create a dataset, declare or remove a source of raw files, scan the "
            "sources into the raw indexes, or set the dataset's tags. Paths must "
            "lie inside the roots the server's operator allows. Converting is a "
            "recipe step (convert-tracks, convert-labels), and next_steps says "
            "what a dataset still needs. Creating a dataset or changing its "
            "sources asks the user to confirm: a client that can ask the user "
            "asks; otherwise the call is refused with reason "
            "confirmation_required: ask the user, then call again with confirm "
            "true. After confirmation_unanswered, confirmation_declined or "
            "confirmation_unavailable, stop and report. A scan reads and "
            "checksums every file and can take minutes."
        ),
        input_model=ManageDatasetInput,
        output=DatasetChange,
        annotations=ToolAnnotations(
            read_only=False, destructive=False, idempotent=False
        ),
        confirm_when=("create_dataset", "add_source"),
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

    def patch_recipe(self, args: PatchRecipeInput) -> ToolResult[PatchedRecipe]:
        """Answer ``patch_recipe``."""
        ...

    def plan_recipe(self, args: PlanRecipeInput) -> ToolResult[PlanPage]:
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

    def manage_dataset(self, args: ManageDatasetInput) -> ToolResult[DatasetChange]:
        """Answer ``manage_dataset``."""
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
        case PatchRecipeInput():
            return backend.patch_recipe(request)
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
        case ManageDatasetInput():
            return backend.manage_dataset(request)
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


def confirmation_unavailable(refusal: ErrorEnvelope) -> ErrorEnvelope:
    """Refuse a call that needs a confirmation that its client cannot ask for.

    Args:
        refusal: The call's :func:`confirmation_required` refusal, whose
            detail names the tool, the conditions and their findings.

    Returns:
        An ``invalid`` envelope with reason :data:`CONFIRMATION_UNAVAILABLE` and
        the refusal's detail.
    """
    detail = refusal.detail or {}
    tool = detail.get("tool")
    findings = detail.get("findings")
    sentences = (
        "; ".join(
            str(finding.get("message"))
            for finding in findings
            if isinstance(finding, dict)
        )
        if isinstance(findings, list)
        else ""
    )
    message = (
        f"{tool} needs the user's confirmation ({sentences}), and this client "
        "cannot ask the user, so this server does not accept confirm from the "
        "model. Stop and report the conditions to the user. The work can run "
        "from a client that asks the user (MCP elicitation), from the mosaic CLI, "
        "or here once the operator serves with --allow-model-confirm."
    )
    return ErrorEnvelope(
        kind="invalid", reason=CONFIRMATION_UNAVAILABLE, message=message, detail=detail
    )


def confirmation_declined(
    name: ToolName, conditions: Sequence[ConfirmationCondition] = ()
) -> ErrorEnvelope:
    """Report a call to *name* whose confirmation was declined when the server asked.

    The decline may come from the user or from a client's hook, so the message
    does not say who declined.

    Args:
        name: The tool that did not run.
        conditions: The conditions the question asked about.
    """
    listed: list[JsonValue] = list(conditions)
    message = (
        f"the confirmation of {name} was declined, so it did not run. Do not call "
        "it again unless the user asks for it."
    )
    return ErrorEnvelope(
        kind="invalid",
        reason=CONFIRMATION_DECLINED,
        message=message,
        detail={"tool": name, "conditions": listed},
    )


def confirmation_unanswered(
    name: ToolName, conditions: Sequence[ConfirmationCondition]
) -> ErrorEnvelope:
    """Report a call to *name* whose confirmation nobody answered.

    A client that cannot show the question cancels it. No one can confirm in
    that session, so the message tells the model to report and stop rather
    than run the same work another way.

    Args:
        name: The tool that did not run.
        conditions: The conditions the question asked about.
    """
    listed: list[JsonValue] = list(conditions)
    asked = ", ".join(conditions) or "its conditions"
    message = (
        f"nobody answered the confirmation that {name} asked for ({asked}), so it "
        "did not run. No one can confirm in this session: report the plan and "
        "the conditions that need approval, and stop. Do not call it again, or "
        "run the same work through the mosaic CLI, unless the user asks."
    )
    return ErrorEnvelope(
        kind="invalid",
        reason=CONFIRMATION_UNANSWERED,
        message=message,
        detail={"tool": name, "conditions": listed},
    )


def confirmation_tag(
    name: ToolName, conditions: Iterable[ConfirmationCondition]
) -> str:
    """Return the tag that ends the question confirming *name* under *conditions*.

    The conditions are sorted and each is named once, so one question has one
    tag: ``[mosaic-confirm tool=submit_pipeline conditions=bind,gpu_lane]``.
    """
    listed = ",".join(sorted(set(conditions)))
    return f"[{CONFIRMATION_TAG} tool={name} conditions={listed}]"


def read_confirmation_tag(
    message: str,
) -> tuple[ToolName, frozenset[ConfirmationCondition]] | None:
    """Return the tool and conditions that a confirmation question asks about.

    Only the message's last line is read, and it must be the tag alone: the
    findings above it name dataset content such as step ids, which must not be
    able to pass for a tag.

    Args:
        message: The question, as a client's hook receives it.

    Returns:
        The tool and its conditions, or ``None`` when the last line is no tag,
        names a tool outside :data:`TOOLS` or a condition outside
        :data:`CONFIRMATION_CONDITIONS`.
    """
    matched = _TAG_LINE.fullmatch(message.rsplit("\n", 1)[-1])
    if matched is None:
        return None
    spec = tool_spec(matched.group(1))
    named = set(matched.group(2).split(","))
    conditions: frozenset[ConfirmationCondition] = frozenset(
        condition for condition in CONFIRMATION_CONDITIONS if condition in named
    )
    if spec is None or len(conditions) != len(named):
        return None
    return spec.name, conditions


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


def cleanup_not_allowed(name: ToolName) -> ErrorEnvelope:
    """Refuse a cleanup that the server was not started to allow."""
    message = (
        f"{name} with cleanup 'superseded' removes feature runs, and this server "
        "does not allow it. The operator starts the server with --allow-cleanup "
        "feature-runs, or lists feature-runs under allow_cleanup in its "
        "configuration. Submit without cleanup, or report this to the user."
    )
    return ErrorEnvelope(
        kind="unavailable",
        reason=CLEANUP_NOT_ALLOWED,
        message=message,
        detail={"tool": name, "allow_cleanup": "feature-runs"},
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
