"""The divisions a media write made, as ``mosaic scan --json`` and the MCP print them.

:func:`division_report_wire` projects the plans of a media scan or a division
``MediaIndexScope`` (``MediaScanResult.divisions``,
``MediaIndexResult.divisions``) onto JSON-ready models. A held-back and a
refused division each carry their ``reason``, a stable literal.

Imports pydantic, :mod:`mosaic.core.strict_model` and the standard-library leaf
:mod:`~mosaic.core.media.store_division` only, so the agent tool contract can
declare a field over these models.
"""

from __future__ import annotations

from collections.abc import Iterable

from mosaic.core.strict_model import StrictModel

from .store_division import (
    DivisionPlan,
    HeldBack,
    HoldBackReason,
    RefusedSpan,
    SpanRefusalReason,
    StartSource,
)

__all__ = [
    "DivisionReportWire",
    "DivisionSequenceWire",
    "DivisionSpanWire",
    "division_report_wire",
    "division_span_wire",
]


class DivisionSequenceWire(StrictModel):
    """One sequence that a division wrote.

    Attributes:
        group: The recording's name.
        sequence: The sequence's name, its UTC start ``YYYYMMDDTHHMMSSZ``.
        store: The store's stored path.
        chunk_first: The sequence's first chunk.
        chunk_last: The sequence's last chunk.
        first_frame: The sequence's first frame on the recording's frame axis.
        frame_count: The sequence's frames.
        follows_gap: Whether a gap or a store boundary precedes the sequence.
        start_utc: The UTC time of the sequence's first frame, ISO 8601.
        start_source: The clock source that named the sequence.
    """

    group: str
    sequence: str
    store: str
    chunk_first: int
    chunk_last: int
    first_frame: int
    frame_count: int
    follows_gap: bool
    start_utc: str
    start_source: StartSource


class DivisionSpanWire(StrictModel):
    """A sequence held back, or a division or store refused, with its reason.

    Attributes:
        group: The recording's name.
        store: The store's stored path.
        chunk_first: The span's first chunk.
        chunk_last: The span's last chunk.
        reason: Why the span is not written.
        message: The reason, for a person.
    """

    group: str
    store: str
    chunk_first: int
    chunk_last: int
    reason: HoldBackReason | SpanRefusalReason
    message: str


class DivisionReportWire(StrictModel):
    """Every sequence that a write divided, held back or refused.

    Attributes:
        divisions: The sequences written, in recording order.
        held_back: The trailing sequences still recording.
        refused: The divisions and stores not written.
    """

    divisions: tuple[DivisionSequenceWire, ...] = ()
    held_back: tuple[DivisionSpanWire, ...] = ()
    refused: tuple[DivisionSpanWire, ...] = ()


def division_span_wire(group: str, span: HeldBack | RefusedSpan) -> DivisionSpanWire:
    """Return a held-back or refused *span* of the recording *group* as its model."""
    return DivisionSpanWire(
        group=group,
        store=span.abs_path,
        chunk_first=span.chunk_first,
        chunk_last=span.chunk_last,
        reason=span.reason,
        message=span.message,
    )


def division_report_wire(plans: Iterable[DivisionPlan]) -> DivisionReportWire:
    """Return what *plans* divided, held back and refused."""
    listed = list(plans)
    return DivisionReportWire(
        divisions=tuple(
            DivisionSequenceWire(
                group=plan.recording,
                sequence=division.row.sequence,
                store=division.row.abs_path,
                chunk_first=division.row.chunk_first,
                chunk_last=division.row.chunk_last,
                first_frame=division.row.first_frame,
                frame_count=division.row.frame_count,
                follows_gap=division.row.follows_gap,
                start_utc=division.row.start_utc.isoformat(timespec="microseconds"),
                start_source=division.row.start_source,
            )
            for plan in listed
            for division in plan.divisions
        ),
        held_back=tuple(
            division_span_wire(plan.recording, held)
            for plan in listed
            for held in plan.held_back
        ),
        refused=tuple(
            division_span_wire(plan.recording, refused)
            for plan in listed
            for refused in plan.refused
        ),
    )
