"""The paragraph that a server returns beside each tool's structured result.

:func:`summarize` writes one paragraph per output model. A paragraph names the
counts, statuses and identifiers that mosaic generates: uuids, request and
attempt ids, run ids, fingerprints, and the step ids of the caller's recipe. It
leaves out the text that a dataset holds, such as names, notes, tags and paths,
which the structured result returns in typed fields only. A failure's paragraph
is its envelope's message, the only text that a failed call returns.

The module imports no MCP SDK. It imports the output models, and through them
pandas.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel

from mosaic.core.dataset_summary import DatasetSummary
from mosaic.core.pipeline.cancel import RequestCancel
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import (
    RequestStatusWire,
    ValidationReport,
)
from mosaic.core.pipeline.inventory import InventoryWire
from mosaic.core.pipeline.preview import ArtifactPreview

from .outputs import PatchedRecipe, PlanPage, StepSuggestions, StepTypes
from .spec import (
    ActivityPage,
    DatasetChange,
    DatasetList,
    FailureDiagnosis,
    SequencePage,
    SubmittedPipeline,
)

__all__ = ["summarize"]

_LISTED: Final = 10
"""The most identifiers one paragraph names before it counts the rest."""

_STRUCTURED: Final = "The JSON result has the details."


def summarize(output: BaseModel) -> str:
    """Return one paragraph that summarizes a tool's result.

    Args:
        output: A shared tool's output model, or an :class:`ErrorEnvelope`.

    Returns:
        The paragraph, without a trailing newline.

    Raises:
        TypeError: *output* is no shared tool's output.
    """
    match output:
        case ErrorEnvelope():
            return _error(output)
        case DatasetList():
            return _dataset_list(output)
        case DatasetSummary():
            return _dataset_summary(output)
        case SequencePage():
            return _page(output.total, len(output.items), output.next_cursor, "entry")
        case InventoryWire():
            return _inventory(output)
        case StepTypes():
            return _step_types(output)
        case PatchedRecipe():
            return _patched(output)
        case ValidationReport():
            return _validation(output)
        case StepSuggestions():
            return _suggestions(output)
        case PlanPage():
            return _plan(output)
        case SubmittedPipeline():
            return _submitted(output)
        case RequestStatusWire():
            return _request_status(output)
        case RequestCancel():
            return _cancel(output)
        case ActivityPage():
            return _activity(output)
        case FailureDiagnosis():
            return _diagnosis(output)
        case ArtifactPreview():
            return _preview(output)
        case DatasetChange():
            return _dataset_change(output)
        case _:
            msg = f"no shared tool returns {type(output).__name__}"
            raise TypeError(msg)


def _error(envelope: ErrorEnvelope) -> str:
    reason = f", {envelope.reason}" if envelope.reason else ""
    step = f" at step {envelope.step!r}" if envelope.step else ""
    return f"Failed ({envelope.kind}{reason}){step}: {envelope.message}"


def _names(identifiers: Sequence[str]) -> str:
    """Join up to :data:`_LISTED` identifiers, counting the rest."""
    shown = ", ".join(identifiers[:_LISTED])
    rest = len(identifiers) - _LISTED
    return f"{shown} and {rest} more" if rest > 0 else shown


def _page(
    total: int, shown: int, next_cursor: str | None, noun: str, named: str = ""
) -> str:
    """Return the paragraph of one page, with *named* after its counts."""
    more = (
        " Pass next_cursor as cursor for the next page."
        if next_cursor is not None
        else ""
    )
    return f"{total} {noun}(s) match, {shown} on this page.{named}{more} {_STRUCTURED}"


def _dataset_list(listed: DatasetList) -> str:
    uuids = [item.uuid for item in listed.items]
    named = f" Their uuids: {_names(uuids)}." if uuids else ""
    return _page(listed.total, len(listed.items), listed.next_cursor, "dataset", named)


def _dataset_summary(summary: DatasetSummary) -> str:
    counts = summary.counts
    known = [
        f"{counts.sequences} sequence(s)",
        f"{counts.recordings} recording(s) ({counts.divided_recordings} divided)",
        f"{counts.videos} video(s)",
        f"{len(summary.arena_layouts)} arena layout(s)",
        *(
            f"{count} {noun}"
            for count, noun in (
                (counts.tracks_variants, "tracks variant(s)"),
                (counts.feature_runs, "feature run(s)"),
                (counts.models, "trained model(s)"),
            )
            if count is not None
        ),
    ]
    managed = (
        " A service manages it, so submissions and cancels are refused."
        if summary.api_managed
        else ""
    )
    return f"Dataset {summary.uuid} holds {', '.join(known)}.{managed} {_STRUCTURED}"


def _inventory(found: InventoryWire) -> str:
    statuses: dict[str, int] = {}
    for artifact in found.artifacts:
        statuses[artifact.status] = statuses.get(artifact.status, 0) + 1
    counted = ", ".join(f"{count} {status}" for status, count in statuses.items())
    on_page = f" On this page: {counted}." if counted else ""
    unavailable = (
        f" Kinds not reported: {', '.join(found.unavailable_kinds)}."
        if found.unavailable_kinds
        else ""
    )
    return (
        _page(found.total, len(found.artifacts), found.next_cursor, "artifact")
        + on_page
        + unavailable
    )


def _step_types(found: StepTypes) -> str:
    """Name the step types described, or count the page a search returned."""
    unknown = (
        f" No step type is named {_names(list(found.unknown))}."
        if found.unknown
        else ""
    )
    if found.details:
        described = [
            f"{detail.name} ({len(detail.params)} params field(s))"
            for detail in found.details
        ]
        return f"Described {_names(described)}.{unknown} {_STRUCTURED}"
    if found.next_cursor is None and found.unknown:
        named = [entry.name for entry in found.entries]
        return f"Listed {_names(named)}.{unknown} {_STRUCTURED}"
    return _page(found.total, len(found.entries), found.next_cursor, "step type")


def _validation(report: ValidationReport) -> str:
    verdict = "valid" if report.valid else "not valid"
    digest = f" Its digest is {report.digest}." if report.digest else ""
    return (
        f"The recipe is {verdict}: {len(report.problems)} problem(s), "
        f"{len(report.cautions)} caution(s), {len(report.unchecked)} step(s) whose "
        f"columns are unchecked.{digest} {_STRUCTURED}"
    )


def _patched(patched: PatchedRecipe) -> str:
    verdict = "valid" if patched.valid else "not valid"
    return (
        f"Saved the recipe as digest {patched.digest}: {patched.steps_total} "
        f"step(s), {verdict}, {len(patched.problems)} problem(s). Pass "
        f"{{digest: {patched.digest}}} as the recipe. {_STRUCTURED}"
    )


def _dataset_change(change: DatasetChange) -> str:
    scanned = (
        " Raw index rows: "
        + ", ".join(f"{kind} {rows}" for kind, rows in change.scanned.items())
        + "."
        if change.scanned
        else ""
    )
    steps = (
        f" {len(change.next_steps)} next step(s) before a pipeline reads it."
        if change.next_steps
        else ""
    )
    return (
        f"{change.action} done on dataset {change.dataset}.{scanned}{steps} "
        f"{_STRUCTURED}"
    )


def _suggestions(found: StepSuggestions) -> str:
    where = f"step {found.step!r}" if found.step is not None else "the top"
    if found.mode == "input":
        allowed = f"{len(found.inputs)} input(s) may be added"
    else:
        allowed = f"{len(found.steps)} step type(s) may be added"
    return (
        f"At {where} ({found.mode}), {allowed}, and {found.refused} were refused. "
        f"{_STRUCTURED}"
    )


def _plan(plan: PlanPage) -> str:
    steps = [
        f"{step.step} ({step.status}, {step.asked} asked, lane {step.lane})"
        for step in plan.steps
    ]
    problems = (
        f" {len(plan.problems)} problem(s) refuse a submission."
        if plan.problems
        else ""
    )
    done = " Every step is already complete." if plan.complete else ""
    cautions = (
        f" {len(plan.cautions)} caution(s) to show the user: "
        f"{_names([str(caution) for caution in plan.cautions])}."
        if plan.cautions
        else ""
    )
    narrowed = [
        f"{step.step} to {step.coverage.target}"
        for step in plan.steps
        if step.step_scope is not None
    ]
    narrowing = (
        f" Narrowed by step_entries: {_names(narrowed)} entries." if narrowed else ""
    )
    waiting = [
        f"{step.step} on {step.reason.step}"
        for step in plan.steps
        if step.reason is not None
        and step.reason.kind == "identity_unresolved"
        and step.reason.step != step.step
    ]
    deferred = (
        f" Named only once the step above runs: {_names(waiting)}." if waiting else ""
    )
    orphaned = sum(1 for item in plan.open_requests if item.verdict == "orphaned")
    unended = (
        f" {len(plan.open_requests)} request(s) of this recipe have not ended, "
        f"{orphaned} of them orphaned: "
        f"{_names([f'{item.request_id} {item.verdict}' for item in plan.open_requests])}."
        if plan.open_requests
        else ""
    )
    cleanup = (
        f" {plan.cleanup.removable_runs} superseded run(s) "
        f"({plan.cleanup.removable_bytes / 1e9:.1f} GB) are removable and "
        f"{plan.cleanup.kept_runs} kept; each step lists a sample of its storage's "
        f"under superseded."
        if plan.cleanup is not None
        else ""
    )
    if plan.cleanup is not None and plan.cleanup.retired:
        unplaced = len(plan.cleanup.retired_runs) + plan.cleanup.retired_omitted
        cleanup += (
            f" It retires {len(plan.cleanup.retired)} recipe(s); {unplaced} of "
            f"their runs sit in storages no step names, under cleanup.retired_runs."
        )
    elif plan.cleanup is not None and plan.retirable:
        cleanup += (
            f" {len(plan.retirable) + plan.retirable_omitted} saved recipe(s) wrote "
            f"runs that retire can reclaim, under retirable."
        )
    shown = (
        f"{len(plan.steps)} of its {plan.steps_total} step(s)"
        if len(plan.steps) < plan.steps_total
        else f"{plan.steps_total} step(s)"
    )
    paging = (
        f" More steps follow: call again with cursor {plan.next_cursor} to show "
        f"the user every step before submitting."
        if plan.next_cursor is not None
        else ""
    )
    return (
        f"The plan covers {plan.scope_size} entries in {shown}: "
        f"{_names(steps)}.{paging}{done}{narrowing}{deferred}{problems}{cautions}"
        f"{unended}"
        f"{cleanup} "
        f"Its fingerprint is {plan.fingerprint}, which submit_pipeline requires. "
        f"Show the user the plan and its cautions before submitting. {_STRUCTURED}"
    )


def _submitted(submitted: SubmittedPipeline) -> str:
    duplicates = (
        f" Open requests of the same recipe: {_names(list(submitted.duplicates))}."
        if submitted.duplicates
        else ""
    )
    cleanup = (
        f" Once every step finished it removes {submitted.cleanup_runs} superseded "
        f"run(s), {submitted.cleanup_bytes / 1e9:.1f} GB."
        if submitted.cleanup_runs
        else ""
    )
    return (
        f"Submitted request {submitted.request_id} with "
        f"{len(submitted.steps)} step(s), running in the background.{duplicates}"
        f"{cleanup} Wait for it with get_request_status and wait_seconds, which "
        "returns when a step or the request ends. Do not watch the driver's "
        "process. Its log, driver_log, explains a failure."
    )


def _request_status(status: RequestStatusWire) -> str:
    steps = [
        f"{step.step} {step.outcome}"
        + (f" ({step.liveness})" if step.liveness in ("dead", "unknown") else "")
        for step in status.steps
    ]
    poll = (
        f"; poll again in {status.poll_after_seconds} s"
        if status.poll_after_seconds is not None
        else ""
    )
    left_out = [
        f"{count} {kind}"
        for count, kind in (
            (status.omitted_finished, "finished"),
            (status.omitted_pending, "not started"),
        )
        if count
    ]
    omitted = f" Not listed: {_names(left_out)}." if left_out else ""
    removal = status.cleanup
    cleanup = (
        f" Cleanup {removal.status}: removed {removal.removed_runs} of "
        f"{removal.approved_runs} approved run(s), kept {removal.kept_runs}, failed "
        f"{removal.failed_runs}."
        if removal is not None
        else ""
    )
    return (
        f"Request {status.request_id} is {status.verdict}{poll}. "
        f"Steps: {_names(steps) or 'none listed'}.{omitted}{cleanup} {_STRUCTURED}"
    )


def _cancel(cancelled: RequestCancel) -> str:
    return (
        f"Cancel of request {cancelled.request_id}: {cancelled.outcome}, now "
        f"{cancelled.status}, {len(cancelled.steps)} step(s) acted on. "
        f"{_STRUCTURED}"
    )


def _activity(page: ActivityPage) -> str:
    if page.view == "requests":
        listed = [f"{item.request_id} {item.verdict}" for item in page.requests]
    else:
        listed = [
            f"{run.execution_id} {run.outcome}"
            + (f" ({run.liveness})" if run.liveness and run.liveness != "ended" else "")
            for run in page.runs
        ]
    more = (
        " Pass next_cursor as cursor for older ones."
        if page.next_cursor is not None
        else ""
    )
    return f"{len(listed)} {page.view}, newest first: {_names(listed) or 'none'}.{more}"


def _diagnosis(diagnosis: FailureDiagnosis) -> str:
    subject = (
        f"Request {diagnosis.request_id} ({diagnosis.verdict})"
        if diagnosis.request_id is not None
        else "The attempt"
    )
    if not diagnosis.attempts:
        return f"{subject} has no failed or partial attempt."
    attempts = [
        f"{attempt.step or attempt.execution_id} {attempt.outcome}"
        + (", allow_partial applies" if attempt.allow_partial_applies else "")
        for attempt in diagnosis.attempts
    ]
    return (
        f"{subject}: {_names(attempts)}. Propose the remedies to the user. "
        f"{_STRUCTURED}"
    )


def _preview(preview: ArtifactPreview) -> str:
    extent = (
        f", frames {preview.frame_min} to {preview.frame_max}"
        if preview.frame_min is not None and preview.frame_max is not None
        else ""
    )
    more = ", and more rows follow" if preview.has_more else ""
    return (
        f"The table has {preview.n_rows} row(s) and {len(preview.columns)} "
        f"column(s){extent}. The preview returns {len(preview.rows)} row(s){more}. "
        f"{_STRUCTURED}"
    )
