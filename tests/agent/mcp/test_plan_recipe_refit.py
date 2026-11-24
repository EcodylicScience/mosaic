"""A plan_recipe step names the refit its fit is due as one line."""

from __future__ import annotations

from pathlib import Path

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.outputs import PlanPage, PlanPageStep
from mosaic.agent.mcp.spec import call_tool
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import plan_pipeline, plan_wire, run_pipeline
from tests.helpers import (
    TEMPLATES_RECIPE,
    RecordingStarter,
    add_tracks_variant,
    copy_of,
    drifted_templates,
    make_dataset,
    templates_recipe,
)

VARIANT = "convert-trex.0.2-1111111111"


def _refits(dataset: Dataset) -> dict[str, str | None]:
    """The ``refit`` of each step of ``plan_recipe`` over the templates recipe."""
    uuid = dataset.manifest.uuid
    assert uuid is not None
    backend = LocalBackend(
        [dataset.manifest_path], warm=False, start_driver=RecordingStarter()
    )
    answered = call_tool(
        backend, "plan_recipe", {"dataset": uuid, "recipe": TEMPLATES_RECIPE}
    )
    assert isinstance(answered, PlanPage), answered
    return {step.step: step.refit for step in answered.steps}


def test_a_drifted_fit_names_its_refit(tmp_path: Path) -> None:
    dataset = drifted_templates(tmp_path / "fitted", variant=VARIANT)

    assert _refits(dataset) == {"speed": None, "templates": "1 moved (seq_a)"}


def test_a_fit_a_link_holds_names_the_link(tmp_path: Path) -> None:
    source = drifted_templates(tmp_path / "source", variant=VARIANT)
    copy = copy_of(source, tmp_path / "copy")

    assert _refits(copy) == {
        "speed": None,
        "templates": f"1 moved (seq_a), fitted in link source-{source.uuid}",
    }


def test_an_undrifted_fit_names_no_refit(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "fitted")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    _ = run_pipeline(dataset, templates_recipe())

    assert _refits(dataset) == {"speed": None, "templates": None}


def _asked_notes(dataset: Dataset) -> dict[str, str | None]:
    """The ``refit`` of each plan page step, the templates step asked to fit again."""
    plan = plan_pipeline(dataset, templates_recipe(), refit=["templates"])
    return {step.step: PlanPageStep.of(step).refit for step in plan_wire(plan).steps}


def test_an_asked_refit_says_it_was_asked(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "fitted")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    _ = run_pipeline(dataset, templates_recipe())

    assert _asked_notes(dataset) == {"speed": None, "templates": "on request"}


def test_an_asked_refit_of_a_drifted_fit_says_both(tmp_path: Path) -> None:
    source = drifted_templates(tmp_path / "source", variant=VARIANT)
    copy = copy_of(source, tmp_path / "copy")

    assert _asked_notes(source)["templates"] == "on request, 1 moved (seq_a)"
    assert _asked_notes(copy)["templates"] == (
        f"on request, 1 moved (seq_a), fitted in link source-{source.uuid}"
    )
