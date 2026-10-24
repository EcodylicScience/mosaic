"""Recipes named by digest or saved name, patched in place, and scoped as the CLI scopes.

A recipe saved by ``patch_recipe`` is named by its digest in the four recipe tools,
so a long recipe is sent once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.outputs import PatchedRecipe, StepSuggestions
from mosaic.agent.mcp.spec import SubmittedPipeline, call_tool
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import (
    PlanWire,
    ValidationReport,
    load_request,
    recipe_path,
)
from mosaic.core.scope import Scope
from tests.helpers import RecordingStarter, calms21_dataset

RECIPE: dict[str, object] = {
    "schema_version": 1,
    "name": "split",
    "steps": [
        {"id": "speed", "type": "feature", "feature": "speed-angvel"},
        {
            "id": "templates_{n}",
            "type": "feature",
            "feature": "extract-templates",
            "repeat": {"n": [4, 6]},
            "inputs": [{"step": "speed"}],
            "params": {"n_templates": "{n}"},
        },
    ],
}
"""Speed and two templates steps, written with a repeat."""

TRAIN = {"groups": ["train"]}


@pytest.fixture
def split(tmp_path: Path) -> Dataset:
    """Converted tracks of a train group of two sequences and a test group of one."""
    ds = calms21_dataset(
        tmp_path / "split",
        {"train": {"a": {"s1": 20, "s2": 20}}, "test": {"a": {"s3": 20}}},
        seed=4,
    )
    _ = ds.convert_all_tracks()
    return ds


@pytest.fixture
def starter() -> RecordingStarter:
    return RecordingStarter()


@pytest.fixture
def backend(split: Dataset, starter: RecordingStarter) -> LocalBackend:
    return LocalBackend([split.manifest_path], warm=False, start_driver=starter)


def _uuid(dataset: Dataset) -> str:
    uuid = dataset.manifest.uuid
    assert uuid is not None
    return uuid


def _call[T](
    backend: LocalBackend, tool: str, expected: type[T], /, **arguments: object
) -> T:
    answered = call_tool(backend, tool, arguments)
    assert isinstance(answered, expected), answered
    return answered


def _saved(
    backend: LocalBackend, dataset: Dataset, **arguments: object
) -> PatchedRecipe:
    return _call(
        backend,
        "patch_recipe",
        PatchedRecipe,
        dataset=_uuid(dataset),
        **{"base": RECIPE, **arguments},
    )


def test_patching_with_no_edits_saves_the_expanded_recipe(
    backend: LocalBackend, split: Dataset
) -> None:
    saved = _saved(backend, split)
    validated = _call(backend, "validate_recipe", ValidationReport, recipe=RECIPE)

    assert saved.valid
    assert (saved.name, saved.steps_total) == ("split", 3)
    assert saved.digest == validated.digest
    assert saved.digest is not None
    assert recipe_path(split.base_dir, saved.digest).is_file()
    assert "repeat" not in recipe_path(split.base_dir, saved.digest).read_text()


def test_a_plan_by_digest_is_the_plan_of_the_document_and_its_call_is_small(
    backend: LocalBackend, split: Dataset
) -> None:
    saved = _saved(backend, split)
    by_reference: dict[str, object] = {
        "dataset": _uuid(split),
        "recipe": {"digest": saved.digest},
        "step_entries": {"templates_4": TRAIN, "templates_6": TRAIN},
    }

    referenced = _call(backend, "plan_recipe", PlanWire, **by_reference)
    sent = _call(backend, "plan_recipe", PlanWire, **{**by_reference, "recipe": RECIPE})
    named = _call(
        backend,
        "plan_recipe",
        PlanWire,
        **{**by_reference, "recipe": {"saved": "split"}},
    )

    assert referenced.fingerprint == sent.fingerprint == named.fingerprint
    assert referenced.recipe_digest == saved.digest
    assert len(json.dumps(by_reference)) < 5_000


def test_a_patched_recipe_is_submitted_by_its_digest(
    backend: LocalBackend, split: Dataset, starter: RecordingStarter
) -> None:
    base = _saved(backend, split)
    patched = _saved(
        backend,
        split,
        base={"digest": base.digest},
        ops=[
            {
                "op": "set_param",
                "step": "templates_6",
                "path": "n_templates",
                "value": 8,
            }
        ],
    )
    arguments: dict[str, object] = {
        "dataset": _uuid(split),
        "recipe": {"digest": patched.digest},
        "step_entries": {"templates_4": TRAIN, "templates_6": TRAIN},
    }
    plan = _call(backend, "plan_recipe", PlanWire, **arguments)

    submitted = _call(
        backend,
        "submit_pipeline",
        SubmittedPipeline,
        **arguments,
        plan_fingerprint=plan.fingerprint,
    )

    assert patched.digest != base.digest
    assert submitted.recipe_digest == patched.digest
    assert starter.started == [submitted.request_id]


def test_a_patch_with_a_problem_is_saved_and_reports_it(
    backend: LocalBackend, split: Dataset
) -> None:
    patched = _saved(
        backend,
        split,
        ops=[{"op": "set_param", "step": "speed", "path": "bogus", "value": 1}],
    )

    assert not patched.valid
    assert [problem.step for problem in patched.problems] == ["speed"]
    assert patched.digest is not None
    assert recipe_path(split.base_dir, patched.digest).is_file()


def test_a_patch_that_does_not_parse_saves_nothing(
    backend: LocalBackend, split: Dataset
) -> None:
    refused = _call(
        backend,
        "patch_recipe",
        ErrorEnvelope,
        dataset=_uuid(split),
        base=RECIPE,
        ops=[{"op": "remove_step", "step": "speed"}],
    )

    assert refused.kind == "invalid"
    assert not (split.base_dir / ".mosaic" / "pipelines").exists()


def test_a_reference_needs_a_dataset_and_a_saved_recipe(
    backend: LocalBackend, split: Dataset
) -> None:
    saved = _saved(backend, split)
    without = _call(
        backend, "validate_recipe", ErrorEnvelope, recipe={"digest": saved.digest}
    )
    unsaved = _call(
        backend,
        "validate_recipe",
        ErrorEnvelope,
        recipe={"digest": "0" * 16},
        dataset=_uuid(split),
    )
    validated = _call(
        backend,
        "validate_recipe",
        ValidationReport,
        recipe={"digest": saved.digest},
        dataset=_uuid(split),
    )
    suggested = _call(
        backend,
        "suggest_steps",
        StepSuggestions,
        recipe={"saved": "split"},
        dataset=_uuid(split),
        mode="attach",
        step="speed",
    )

    assert (without.kind, without.reason) == ("invalid", "dataset_required")
    assert (unsaved.kind, unsaved.reason) == ("not_found", "recipe_not_saved")
    assert validated.digest == saved.digest
    assert suggested.steps


def test_a_managed_dataset_saves_no_patch(
    backend: LocalBackend, split: Dataset
) -> None:
    with split.manifest_path.open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")

    refused = _call(
        backend, "patch_recipe", ErrorEnvelope, dataset=_uuid(split), base=RECIPE
    )

    assert refused.reason == "dataset_managed"


@pytest.mark.parametrize(
    ("entries", "step_entries", "scoped"),
    [
        (
            {"entries": [["train", "s1"], ["train", "s2"], ["test", "s3"]]},
            {"entries": [["train", "s1"], ["train", "s2"]]},
            Scope(entries=[("train", "s1"), ("train", "s2")]),
        ),
        (None, {"sequences": ["s1", "s2"]}, Scope(sequences=["s1", "s2"])),
        (
            {"groups": ["train", "test"]},
            {"groups": ["train"], "sequences": ["s1"]},
            Scope(groups=["train"], sequences=["s1"]),
        ),
        (None, [["train", "s1"]], Scope(entries=[("train", "s1")])),
    ],
)
def test_the_library_scope_forms_and_the_bare_list_are_accepted(
    split: Dataset,
    entries: object,
    step_entries: object,
    scoped: Scope,
) -> None:
    backend = LocalBackend(
        [split.manifest_path], warm=False, start_driver=RecordingStarter()
    )
    arguments: dict[str, object] = {
        "dataset": _uuid(split),
        "recipe": RECIPE,
        "entries": entries,
        "step_entries": {"templates_4": step_entries},
    }
    plan = _call(backend, "plan_recipe", PlanWire, **arguments)

    submitted = _call(
        backend,
        "submit_pipeline",
        SubmittedPipeline,
        **arguments,
        plan_fingerprint=plan.fingerprint,
    )

    request = load_request(split.base_dir, submitted.request_id)
    assert request.step_scopes == {"templates_4": scoped}


@pytest.mark.parametrize(
    "entries",
    [{}, {"entries": [["train", "s1"]], "groups": ["train"]}, {"groups": []}],
)
def test_a_scope_that_names_nothing_or_mixes_forms_is_invalid(
    backend: LocalBackend, split: Dataset, entries: object
) -> None:
    refused = _call(
        backend,
        "plan_recipe",
        ErrorEnvelope,
        dataset=_uuid(split),
        recipe=RECIPE,
        entries=entries,
    )

    assert refused.reason == "invalid_arguments"
