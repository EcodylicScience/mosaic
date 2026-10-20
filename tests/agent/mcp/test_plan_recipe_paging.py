"""A plan of many steps is read a page at a time, and every page names one plan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.outputs import PlanPage
from mosaic.agent.mcp.spec import MAX_RESULT_CHARS, call_tool
from mosaic.agent.mcp.summaries import summarize
from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from tests.helpers import RecordingStarter, add_tracks_variant, make_dataset

VARIANT = "convert-trex.0.2-1111111111"

STEPS = 90
"""Enough heading steps to put an unpaged plan over :data:`MAX_RESULT_CHARS`."""

PAGE = 25


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "tracked", name="tracked")
    add_tracks_variant(ds, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    return ds


@pytest.fixture
def backend(dataset: Dataset) -> LocalBackend:
    return LocalBackend(
        [dataset.manifest_path], warm=False, start_driver=RecordingStarter()
    )


def _recipe(steps: int = STEPS) -> dict[str, JsonValue]:
    return {
        "schema_version": 1,
        "name": "many",
        "steps": [
            {
                "id": f"heading{index}",
                "type": "feature",
                "feature": "heading",
                "inputs": ["tracks"],
                "params": {"output_col": f"heading_{index}"},
            }
            for index in range(steps)
        ],
    }


def _plan(
    backend: LocalBackend, dataset: Dataset, **arguments: object
) -> PlanPage | ErrorEnvelope:
    uuid = dataset.manifest.uuid
    assert uuid is not None
    answered = call_tool(
        backend,
        "plan_recipe",
        {"dataset": uuid, "recipe": _recipe(), **arguments},
    )
    assert isinstance(answered, PlanPage | ErrorEnvelope), answered
    return answered


def _pages(backend: LocalBackend, dataset: Dataset, limit: int) -> list[PlanPage]:
    pages: list[PlanPage] = []
    cursor: str | None = None
    while True:
        arguments: dict[str, object] = {"limit": limit}
        if cursor is not None:
            arguments["cursor"] = cursor
        page = _plan(backend, dataset, **arguments)
        assert isinstance(page, PlanPage), page
        pages.append(page)
        cursor = page.next_cursor
        if cursor is None:
            return pages


def test_an_oversized_plan_is_refused_naming_limit(
    backend: LocalBackend, dataset: Dataset
) -> None:
    refused = _plan(backend, dataset)

    assert isinstance(refused, ErrorEnvelope)
    assert refused.reason == "result_too_large"
    assert refused.detail is not None
    assert refused.detail["narrow_with"] == ["limit"]


def test_the_pages_cover_the_plan_once_and_each_fits(
    backend: LocalBackend, dataset: Dataset
) -> None:
    pages = _pages(backend, dataset, PAGE)

    ids = [step.step for page in pages for step in page.steps]
    assert ids == [f"heading{index}" for index in range(STEPS)]
    assert len(pages) == -(-STEPS // PAGE)
    assert all(page.steps_total == STEPS for page in pages)
    assert all(len(page.model_dump_json()) < MAX_RESULT_CHARS for page in pages)
    assert pages[-1].next_cursor is None


def test_every_page_carries_the_one_fingerprint_of_the_whole_plan(
    backend: LocalBackend, dataset: Dataset
) -> None:
    pages = _pages(backend, dataset, PAGE)
    whole = _pages(backend, dataset, 10)

    assert {page.fingerprint for page in pages + whole} == {pages[0].fingerprint}
    assert {page.recipe_digest for page in pages} == {pages[0].recipe_digest}


def test_the_plan_level_blocks_are_on_the_first_page_alone(
    backend: LocalBackend, dataset: Dataset
) -> None:
    pages = _pages(backend, dataset, PAGE)

    assert [caution.where for caution in pages[0].cautions] == ["scope"]
    for later in pages[1:]:
        assert later.problems == ()
        assert later.cautions == ()
        assert later.open_requests == ()
        assert later.cleanup is None
        assert later.parallel is None


def test_a_plan_without_a_limit_returns_every_step_and_no_cursor(
    backend: LocalBackend, dataset: Dataset
) -> None:
    uuid = dataset.manifest.uuid
    assert uuid is not None
    small = call_tool(backend, "plan_recipe", {"dataset": uuid, "recipe": _recipe(3)})

    assert isinstance(small, PlanPage)
    assert [step.step for step in small.steps] == ["heading0", "heading1", "heading2"]
    assert small.steps_total == 3
    assert small.next_cursor is None


@pytest.mark.parametrize(
    "cursor",
    [
        "not json",
        '"heading3"',
        json.dumps(["no-such-step", "x"]),
        json.dumps(["heading3"]),
        json.dumps(["heading3", "another-fingerprint"]),
    ],
)
def test_a_cursor_that_this_plan_did_not_write_is_invalid(
    backend: LocalBackend, dataset: Dataset, cursor: str
) -> None:
    refused = _plan(backend, dataset, limit=PAGE, cursor=cursor)

    assert isinstance(refused, ErrorEnvelope)
    assert refused.reason == "invalid_cursor"


def test_the_summary_of_a_page_points_at_the_next_one(
    backend: LocalBackend, dataset: Dataset
) -> None:
    first, *_ = _pages(backend, dataset, PAGE)

    summary = summarize(first)

    assert f"{PAGE} of its {STEPS} step(s)" in summary
    assert first.next_cursor is not None
    assert "call again with cursor" in summary
