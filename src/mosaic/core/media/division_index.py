"""The media index rows of the imgstore recordings that a write divides.

:func:`plan_recording` reads the stores of one recording, chooses its rule and
applies :func:`~mosaic.core.media.store_division.plan_division` to them, and
:func:`recording_rows` turns the plan into one media index row per division.
A media scan of a dividing source and a division ``MediaIndexScope`` both
divide through these two functions, so the two paths write the same rows.

A recording divided before keeps its rule, its names and its frames: the
committed divisions are passed back as frozen. A plan that cannot reproduce
them, or a declaration that changes the rule or the name, re-divides the
recording. :class:`PlannedRecording` says why, and the caller refuses the
re-division while the recording's entries have outputs. The caller refuses the
division of a store indexed whole on that store's entries alone, because the
store's undivided row is replaced whether or not the recording's other stores
are divided.

The module reads frames through :mod:`~mosaic.core.media.imgstore_io`, so a
dataset read imports it only inside the function that divides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .imgstore_io import StoreProbe, probe_store, store_plan_input, window_probe
from .store_division import (
    CommittedDivision,
    DivisionPlan,
    DivisionRefused,
    DivisionRule,
    RecordingIdentity,
    StoreClock,
    StoreWindow,
    build_division_rows,
    division_display_name,
    division_rule,
    plan_division,
)

__all__ = [
    "PlannedRecording",
    "StoreToDivide",
    "nominal_interval",
    "plan_recording",
    "recording_rows",
]


@dataclass(frozen=True)
class StoreToDivide:
    """One store of a recording, with the row that indexes it whole.

    Attributes:
        path: The store's directory, resolved.
        store_row: The store's undivided media index row as the write builds
            it. Its ``abs_path`` is the stored path that the division rows
            record.
        sync_uuid: The store's synchronization uuid, ``""`` without one.
    """

    path: Path
    store_row: Mapping[str, object]
    sync_uuid: str = ""

    @property
    def abs_path(self) -> str:
        """The store's stored path."""
        return str(self.store_row.get("abs_path", ""))


@dataclass(frozen=True)
class PlannedRecording:
    """The plan of one recording, and the stores it read.

    Attributes:
        plan: The plan.
        probes: Each store's one read, by stored path.
        redivision: Why the plan re-divides the recording, ``None`` when it
            repeats or extends the committed divisions.
        clocks: Each store's clock, by stored path.
    """

    plan: DivisionPlan
    probes: Mapping[str, StoreProbe]
    redivision: str | None
    clocks: Mapping[str, StoreClock]

    def display_names(self) -> dict[str, str]:
        """Return the wall-clock label of each division, by sequence.

        Each label is the division's start in its store's ``timezone_local``
        (:func:`~mosaic.core.media.store_division.division_display_name`).
        """
        return {
            division.row.sequence: division_display_name(
                division.row.start_utc,
                self.clocks[division.row.abs_path].timezone_local,
            )
            for division in self.plan.divisions
        }


def nominal_interval(probes: Sequence[StoreProbe]) -> float:
    """Return the median step between the frame times of *probes*' chunks.

    Raises:
        ValueError: If the stores record fewer than two frame times.
    """
    steps = [
        np.diff(probe.chunk_times[chunk_id])
        for probe in probes
        for chunk_id in sorted(probe.chunk_times)
    ]
    joined = np.concatenate(steps) if steps else np.empty(0)
    if not joined.size:
        message = "the stores record fewer than two frame times to measure a rate by"
        raise ValueError(message)
    return float(np.median(joined))


def _refuse_synchronized(stores: Sequence[StoreToDivide]) -> None:
    """Refuse a store of a synchronized set, whose cameras are one recording."""
    for store in stores:
        if store.sync_uuid:
            message = (
                f"the store {store.abs_path!r} belongs to the synchronized "
                f"recording {store.sync_uuid!r}. Dividing a synchronized set is "
                "not supported. Index its stores whole."
            )
            raise DivisionRefused("synchronized_store", message)


def _rule(
    probes: Sequence[StoreProbe],
    frames: int,
    identity: RecordingIdentity,
    committed: Sequence[CommittedDivision],
) -> tuple[DivisionRule, bool]:
    """Return the recording's rule, and whether it is the committed one.

    A committed rule of the declared frames, chunksize and identity is kept,
    with the gap threshold frozen when the recording was first divided.
    """
    chunksize = probes[0].chunksize
    for division in committed:
        rule = division.rule
        if (rule.frames, rule.chunksize, rule.identity) == (
            frames,
            chunksize,
            identity,
        ):
            return rule, True
    return division_rule(
        frames,
        chunksize=chunksize,
        nominal_interval=nominal_interval(probes),
        identity=identity,
    ), False


def plan_recording(
    stores: Sequence[StoreToDivide],
    *,
    frames: int,
    identity: RecordingIdentity,
    name: str | None,
    include_partial: bool,
    committed: Sequence[CommittedDivision],
) -> PlannedRecording:
    """Read the stores of one recording and plan its divisions.

    Args:
        stores: The recording's one store, or every store of its camera.
        frames: The declared frames of a full division.
        identity: The declared recording identity.
        name: The declared name, ``None`` to derive it.
        include_partial: Whether to write the trailing sequence still
            recording.
        committed: The recording's division rows in the media index.

    Raises:
        DivisionRefused: If a store belongs to a synchronized set, or the
            planner refuses the recording.
        MediaProbeError: If a store does not open.
    """
    _refuse_synchronized(stores)
    probes = {store.abs_path: probe_store(store.path) for store in stores}
    inputs = [
        store_plan_input(probes[store.abs_path], store.abs_path) for store in stores
    ]
    clocks = {plan_input.abs_path: plan_input.clock for plan_input in inputs}
    rule, kept = _rule(list(probes.values()), frames, identity, committed)
    redivision: str | None = None
    if committed and not kept:
        redivision = "its declared rule changed"
    if redivision is None and committed:
        try:
            plan = plan_division(
                inputs,
                rule,
                name=name,
                include_partial=include_partial,
                frozen=[division.row for division in committed],
            )
        except DivisionRefused as refused:
            if refused.reason != "frozen_mismatch":
                raise
            redivision = f"a division it holds moved: {refused}"
        else:
            names = {division.recording for division in committed}
            if names == {plan.recording}:
                return PlannedRecording(plan, probes, None, clocks)
            redivision = f"its name changed from {sorted(names)} to {plan.recording!r}"
    plan = plan_division(inputs, rule, name=name, include_partial=include_partial)
    return PlannedRecording(plan, probes, redivision, clocks)


def recording_rows(
    planned: PlannedRecording, stores: Sequence[StoreToDivide]
) -> list[dict[str, object]]:
    """Return the division rows of *planned*, in recording order.

    Each division's row is its store's row with the measured cells of the
    division's window (:func:`~mosaic.core.media.store_division.build_division_rows`).

    Raises:
        MediaProbeError: If a division's window does not measure.
    """
    rows: list[dict[str, object]] = []
    for store in stores:
        probe = planned.probes[store.abs_path]
        cells: dict[StoreWindow, Mapping[str, object]] = {
            division.row.window: window_probe(probe, division.row.window).row_cells()
            for division in planned.plan.divisions
            if division.row.abs_path == store.abs_path
        }
        rows.extend(build_division_rows(store.store_row, planned.plan, cells))
    order = {
        (division.row.abs_path, division.row.chunk_first): position
        for position, division in enumerate(planned.plan.divisions)
    }
    rows.sort(
        key=lambda row: order[(str(row["abs_path"]), int(str(row["chunk_first"])))]
    )
    return rows
