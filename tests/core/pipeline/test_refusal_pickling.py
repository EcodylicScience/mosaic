"""A refusal survives a trip through a process worker with its reason and detail.

A process pool pickles the exception a worker raised and rebuilds it in the parent.
Default exception pickling calls the class with its ``args``, which no refusal
subclass with its own ``__init__`` signature accepts.
"""

from __future__ import annotations

import pickle

from mosaic.core.pipeline.entry_claim import RunRootHeld
from mosaic.core.pipeline.graph.step_preflight import CoverageShortfall, StepRefused
from mosaic.core.pipeline.markers import InflightMarker
from mosaic.core.pipeline.refusal import Refusal, ToolUnavailable


def _round_trip[E: BaseException](error: E) -> E:
    return pickle.loads(pickle.dumps(error))  # noqa: S301


def test_a_tool_unavailable_survives_pickle_with_its_tool_cause_and_reason() -> None:
    error = ToolUnavailable("vame", "environment", "no interpreter was found")

    restored = _round_trip(error)

    assert type(restored) is ToolUnavailable
    assert (restored.reason, restored.detail) == (
        "tool_unavailable",
        {"tool": "vame", "cause": "environment"},
    )
    assert str(restored) == "no interpreter was found"
    assert restored.error_json("fit") == error.error_json("fit")


def test_a_step_refusal_survives_pickle_with_its_step_and_detail() -> None:
    error = StepRefused("version_moved", "fit", "moved", {"runs": "kpms"})

    restored = _round_trip(error)

    assert type(restored) is StepRefused
    assert (restored.reason, restored.step_id, restored.detail) == (
        "version_moved",
        "fit",
        {"runs": "kpms"},
    )
    assert str(restored) == "moved"


def test_a_coverage_shortfall_survives_pickle_with_its_counts() -> None:
    error = CoverageShortfall("fit", covered=2, target=3, detail="one entry failed")

    restored = _round_trip(error)

    assert type(restored) is CoverageShortfall
    assert (restored.covered, restored.target) == (2, 3)
    assert restored.step_id == "fit"
    assert restored.detail == error.detail
    assert str(restored) == str(error)


def test_a_plain_refusal_survives_pickle() -> None:
    error = Refusal("clip0.mp4 is av1", reason="undecodable_codec")

    restored = _round_trip(error)

    assert (type(restored), restored.reason, restored.step_id) == (
        Refusal,
        "undecodable_codec",
        "",
    )


def test_a_run_root_refusal_survives_pickle_with_its_holder() -> None:
    held = InflightMarker(
        execution_id="01HOLDER", host="otherhost", pid=4242, expires_at="2026-11-06"
    )
    error = RunRootHeld("held by 01HOLDER", held=held)

    restored = _round_trip(error)

    assert type(restored) is RunRootHeld
    assert restored.held == held
    assert (restored.reason, restored.detail) == (
        "run_root_held",
        {
            "holder_execution_id": "01HOLDER",
            "holder_host": "otherhost",
            "holder_expires_at": "2026-11-06",
            "holder_pid": 4242,
        },
    )
    assert str(restored) == "held by 01HOLDER"
    assert restored.error_json("train") == error.error_json("train")


def test_an_unreadable_holder_names_no_one() -> None:
    error = RunRootHeld("held by another host", held=None)

    restored = _round_trip(error)

    assert restored.held is None
    assert restored.detail == {
        "holder_execution_id": "",
        "holder_host": "",
        "holder_expires_at": "",
        "holder_pid": None,
    }
