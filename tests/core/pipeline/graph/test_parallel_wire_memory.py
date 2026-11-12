"""The plan's memory figures are the limit that applies and what is left under it."""

from __future__ import annotations

import pytest

from mosaic.core.pipeline.graph import serialize as serialize_module
from mosaic.core.pipeline.graph.plan import Plan
from mosaic.core.pipeline.graph.serialize import plan_wire
from mosaic.core.pipeline.graph.wire import ParallelWire
from mosaic.core.pipeline.parallelism import ParallelLimits

GIB = 1024**3


def _parallel_wire() -> ParallelWire:
    plan = Plan(recipe_digest="digest", scope=frozenset())
    wire = plan_wire(plan, parallel=ParallelLimits(max_workers=4)).parallel
    assert wire is not None
    return wire


def test_the_figures_are_the_limit_and_its_available_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        serialize_module, "usable_memory_bytes", lambda: (8 * GIB, 5 * GIB)
    )

    wire = _parallel_wire()

    assert (wire.memory_total_bytes, wire.memory_available_bytes) == (
        8 * GIB,
        5 * GIB,
    )


def test_a_host_without_memory_figures_reports_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(serialize_module, "usable_memory_bytes", lambda: (0, 0))

    wire = _parallel_wire()

    assert (wire.memory_total_bytes, wire.memory_available_bytes) == (0, 0)
