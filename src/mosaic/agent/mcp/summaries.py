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

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel

from mosaic.core.dataset_summary import DatasetSummary
from mosaic.core.narrowing import is_str_mapping
from mosaic.core.pipeline.cancel import RequestCancel
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import (
    CatalogEntry,
    PlanWire,
    RequestStatusWire,
    ValidationReport,
)
from mosaic.core.pipeline.inventory import InventoryWire
from mosaic.core.pipeline.preview import ArtifactPreview

from .outputs import StepSuggestions, StepTypePage
from .spec import (
    ActivityPage,
    DatasetList,
    FailureDiagnosis,
    SequencePage,
    SubmittedPipeline,
)

__all__ = ["summarize"]

_LISTED: Final = 10
"""The most identifiers one paragraph names before it counts the rest."""

_STRUCTURED: Final = "The structured result has the details."


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
        case StepTypePage():
            return _page(
                output.total, len(output.entries), output.next_cursor, "step type"
            )
        case CatalogEntry():
            return _step_type(output)
        case ValidationReport():
            return _validation(output)
        case StepSuggestions():
            return _suggestions(output)
        case PlanWire():
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
        f"{counts.videos} video(s)",
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


def _step_type(entry: CatalogEntry) -> str:
    state = (
        " This installation cannot run it."
        if entry.unavailable
        else " This installation can run it."
    )
    license_note = " Its license restricts some uses." if entry.license_note else ""
    fields = _field_count(entry.params_schema)
    return (
        f"Step type {entry.name} ({entry.type}, version {entry.version or 'none'}) "
        f"takes {fields} params field(s).{state}{license_note} {_STRUCTURED}"
    )


def _field_count(schema: Mapping[str, object] | None) -> int:
    """Count the top-level properties of a params schema, 0 without one."""
    properties = None if schema is None else schema.get("properties")
    return len(properties) if is_str_mapping(properties) else 0


def _validation(report: ValidationReport) -> str:
    verdict = "valid" if report.valid else "not valid"
    digest = f" Its digest is {report.digest}." if report.digest else ""
    return (
        f"The recipe is {verdict}: {len(report.problems)} problem(s), "
        f"{len(report.cautions)} caution(s), {len(report.unchecked)} step(s) whose "
        f"columns are unchecked.{digest} {_STRUCTURED}"
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


def _plan(plan: PlanWire) -> str:
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
    return (
        f"The plan covers {plan.scope_size} entries in {len(plan.steps)} step(s): "
        f"{_names(steps)}.{done}{problems} Its fingerprint is {plan.fingerprint}, "
        "which submit_pipeline requires. Show the user the plan before "
        f"submitting. {_STRUCTURED}"
    )


def _submitted(submitted: SubmittedPipeline) -> str:
    duplicates = (
        f" Open requests of the same recipe: {_names(list(submitted.duplicates))}."
        if submitted.duplicates
        else ""
    )
    return (
        f"Submitted request {submitted.request_id} with "
        f"{len(submitted.steps)} step(s), driven by process "
        f"{submitted.driver_pid}.{duplicates} Poll get_request_status after "
        f"{submitted.poll_after_seconds} seconds."
    )


def _request_status(status: RequestStatusWire) -> str:
    steps = [f"{step.step} {step.outcome}" for step in status.steps]
    poll = "" if status.terminal else "; poll again later"
    return (
        f"Request {status.request_id} is {status.verdict}{poll}. "
        f"Steps: {_names(steps)}. {_STRUCTURED}"
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
        listed = [f"{run.execution_id} {run.outcome}" for run in page.runs]
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
