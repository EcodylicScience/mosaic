"""The prompts that a server offers: a workflow over the shared tools, filled in.

Each :class:`PromptSpec` declares a prompt's name, description and arguments,
and :func:`render_prompt` writes its text from the arguments that a client
sends. The text names the shared tools and the order to call them in. An
argument is copied into the text only as an identifier the user supplied: a
dataset uuid, a request id, a feature name, a run id or a goal. The module
imports no MCP SDK.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

__all__ = [
    "PROMPTS",
    "PromptArgumentSpec",
    "PromptArgumentsMissing",
    "PromptSpec",
    "prompt_spec",
    "render_prompt",
]


@dataclass(frozen=True, slots=True)
class PromptArgumentSpec:
    """One argument of a prompt.

    Attributes:
        name: The argument's name.
        description: What it names, written for the person choosing it.
    """

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One prompt, declared for every server.

    Attributes:
        name: The prompt's name.
        title: A short name a client displays.
        description: What the prompt does.
        arguments: Its arguments, each required.
        template: Writes the text from the arguments, by name.
    """

    name: str
    title: str
    description: str
    arguments: tuple[PromptArgumentSpec, ...]
    template: Callable[[Mapping[str, str]], str]


class PromptArgumentsMissing(ValueError):
    """A prompt asked for without one of its arguments."""


_DATASET: Final = PromptArgumentSpec(
    "dataset", "The dataset's uuid, as list_datasets names it."
)


def _build_pipeline(arguments: Mapping[str, str]) -> str:
    return f"""\
Build a mosaic pipeline on dataset {arguments["dataset"]} for this goal:

{arguments["goal"]}

1. Call describe_dataset and list_sequences to learn what the dataset holds.
2. Call describe_step_types with a query for the features and ops that serve
   the goal, then with names=[...] for the ones you choose.
3. Draft a recipe. Call validate_recipe and fix every problem, and use
   suggest_steps where a step needs an input or a following step.
4. Call plan_recipe. Show the user each step's status, run id, entries asked
   and lane, and which steps are already complete.
5. Stop and ask the user to approve the plan. Do not call submit_pipeline until
   they agree. Set confirm only when the user agreed to the conditions that a
   confirmation_required refusal lists.
6. After submitting, poll get_request_status as poll_after_seconds says, and
   preview the outputs with preview_output when the request has finished."""


def _diagnose_dataset(arguments: Mapping[str, str]) -> str:
    return f"""\
Diagnose dataset {arguments["dataset"]}. Change nothing.

1. Call get_inventory for the artifacts that are not complete, and read each
   damaged or drifted one in depth with its ref. Note frame_axis_mismatch and
   frame_tail_short in an artifact's notes.
2. Call list_activity for the requests, and diagnose_failures for each one that
   failed, is partial or reads orphaned. A request reads orphaned when nothing
   drives it any more: its driver stopped, or its step's process is provably
   gone. A step whose liveness reads unknown runs on another host that wrote no
   heartbeat; ask the user whether that process still runs.
3. Report each finding with the remedy you propose, such as a mosaic CLI
   command for the user to run (mosaic measure-tracks, mosaic reconcile, a
   re-run of a pipeline request, mosaic reindex). Apply none of them."""


def _explain_request(arguments: Mapping[str, str]) -> str:
    return f"""\
Explain request {arguments["rid"]} of dataset {arguments["dataset"]}.

1. Call get_request_status for its verdict and each step's outcome and progress.
2. Call diagnose_failures for the request if a step failed or lost entries.
3. Tell the user what each step did, why a refused or failed step stopped, and
   what answers it: allow_partial for a coverage shortfall or a quarantined
   entry, a changed recipe for a problem in it, or a CLI command for the user to
   run. A request reading orphaned, because its driver stopped or its step's
   process is provably gone, is resumed with mosaic pipeline run --manifest
   <manifest> --request <request id>. A step whose liveness reads unknown runs on
   another host that wrote no heartbeat: if the user confirms that its process
   has exited, they record it with mosaic cancel --manifest <manifest> --request
   <request id> --assume-exited and then submit the recipe again."""


def _compare_runs(arguments: Mapping[str, str]) -> str:
    feature = arguments["feature"]
    return f"""\
Compare two runs of feature {feature} on dataset {arguments["dataset"]}: run
{arguments["a"]} and run {arguments["b"]}.

1. Call get_inventory with kinds ["feature"] and status_not [], and find both
   runs. Read each in depth with its ref, which reports its recorded params,
   coverage and upstream runs.
2. Report how their params, tracks variants, upstream runs and coverage differ.
3. Call preview_output for one entry that both runs cover, and compare the
   columns and their statistics."""


def _summarize_dataset(arguments: Mapping[str, str]) -> str:
    return f"""\
Summarize dataset {arguments["dataset"]} for the user.

1. Call describe_dataset for its counts, track schemas and frame rates.
2. Call get_inventory with status_not [] for what has been computed.
3. Call list_activity for its recent requests.
4. Write a short summary: the recordings, the tracks, the features computed and
   their completeness, and the recent pipeline activity."""


PROMPTS: Final[tuple[PromptSpec, ...]] = (
    PromptSpec(
        name="build_pipeline",
        title="Build a pipeline",
        description=(
            "Draft, validate and plan a recipe for a goal, then stop for the "
            "user's approval before submitting it."
        ),
        arguments=(
            _DATASET,
            PromptArgumentSpec("goal", "What the pipeline should compute."),
        ),
        template=_build_pipeline,
    ),
    PromptSpec(
        name="diagnose_dataset",
        title="Diagnose a dataset",
        description=(
            "Find incomplete and damaged artifacts and failed requests, and "
            "propose remedies without applying any."
        ),
        arguments=(_DATASET,),
        template=_diagnose_dataset,
    ),
    PromptSpec(
        name="explain_request",
        title="Explain a request",
        description="Explain what a submitted request did and why it stopped.",
        arguments=(
            _DATASET,
            PromptArgumentSpec("rid", "The request id, as submit_pipeline named it."),
        ),
        template=_explain_request,
    ),
    PromptSpec(
        name="compare_runs",
        title="Compare two runs",
        description="Compare the params, inputs and outputs of two runs of a feature.",
        arguments=(
            _DATASET,
            PromptArgumentSpec("feature", "The feature's storage name or slug."),
            PromptArgumentSpec("a", "The first run id."),
            PromptArgumentSpec("b", "The second run id."),
        ),
        template=_compare_runs,
    ),
    PromptSpec(
        name="summarize_dataset",
        title="Summarize a dataset",
        description="Summarize what a dataset holds and what has been computed.",
        arguments=(_DATASET,),
        template=_summarize_dataset,
    ),
)
"""The prompts, in the order a server lists them."""

_BY_NAME: Final[Mapping[str, PromptSpec]] = {spec.name: spec for spec in PROMPTS}


def prompt_spec(name: str) -> PromptSpec | None:
    """Return the prompt named *name*, ``None`` when none is declared."""
    return _BY_NAME.get(name)


def render_prompt(spec: PromptSpec, arguments: Mapping[str, str]) -> str:
    """Write *spec*'s text from *arguments*.

    Raises:
        PromptArgumentsMissing: An argument of *spec* is absent or empty.
    """
    missing = [
        argument.name for argument in spec.arguments if not arguments.get(argument.name)
    ]
    if missing:
        msg = f"prompt {spec.name} needs {', '.join(missing)}"
        raise PromptArgumentsMissing(msg)
    return spec.template(arguments)
