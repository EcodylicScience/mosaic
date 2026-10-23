"""A recipe changed by edits, and a saved recipe named by digest or by name."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    FeatureStepSpec,
    PatchInvalid,
    PatchOp,
    Recipe,
    RecipeInvalid,
    RecipeNameAmbiguous,
    RecipeNotSaved,
    StepRef,
    patch_recipe,
    recipe_digest,
    recipe_path,
    recipe_reference,
    resolve_saved_recipe,
    save_recipe,
)
from tests.helpers import chain_recipe, named_recipe, speed_recipe

_OPS: TypeAdapter[list[PatchOp]] = TypeAdapter(list[PatchOp])


def _patched(recipe: Recipe, *ops: dict[str, JsonValue]) -> Recipe:
    return patch_recipe(recipe, _OPS.validate_python(list(ops)))


def _templates(recipe: Recipe) -> FeatureStepSpec:
    step = recipe.step("templates")
    assert isinstance(step, FeatureStepSpec)
    return step


def test_set_param_sets_a_nested_path_and_removes_a_param_without_a_value() -> None:
    patched = _patched(
        chain_recipe(),
        {"op": "set_param", "step": "templates", "path": "n_templates", "value": 8},
        {"op": "set_param", "step": "templates", "path": "cv.folds", "value": 3},
    )
    removed = _patched(
        patched, {"op": "set_param", "step": "templates", "path": "cv.folds"}
    )

    assert _templates(patched).params == {"n_templates": 8, "cv": {"folds": 3}}
    assert _templates(removed).params == {"n_templates": 8, "cv": {}}


def test_set_param_to_null_sets_null() -> None:
    patched = _patched(
        chain_recipe(),
        {"op": "set_param", "step": "templates", "path": "n_templates", "value": None},
    )

    assert _templates(patched).params == {"n_templates": None}


def test_add_step_appends_and_expands_its_repeat() -> None:
    patched = _patched(
        chain_recipe(),
        {
            "op": "add_step",
            "document": {
                "id": "more_{n}",
                "type": "feature",
                "feature": "extract-templates",
                "repeat": {"n": [2, 3]},
                "inputs": [{"step": "speed"}],
                "params": {"n_templates": "{n}"},
            },
        },
    )

    assert patched.ids == ("speed", "templates", "more_2", "more_3")


def test_replace_step_keeps_the_id() -> None:
    patched = _patched(
        chain_recipe(),
        {
            "op": "replace_step",
            "step": "templates",
            "document": {
                "type": "feature",
                "feature": "extract-templates",
                "inputs": [{"step": "speed"}],
                "params": {"n_templates": 16},
            },
        },
    )

    assert _templates(patched).params == {"n_templates": 16}
    assert _templates(patched).inputs == [StepRef(step="speed")]


def test_a_step_still_read_cannot_be_removed() -> None:
    with pytest.raises(RecipeInvalid, match="references unknown step 'speed'"):
        _ = _patched(chain_recipe(), {"op": "remove_step", "step": "speed"})


def test_removing_a_step_and_its_reader_leaves_the_rest() -> None:
    patched = _patched(
        chain_recipe(),
        {"op": "remove_step", "step": "speed"},
        {"op": "remove_step", "step": "templates"},
    )

    assert patched.ids == ()


@pytest.mark.parametrize(
    ("op", "fragment"),
    [
        ({"op": "remove_step", "step": "nope"}, "no step 'nope'"),
        (
            {
                "op": "set_param",
                "step": "templates",
                "path": "n_templates.x",
                "value": 1,
            },
            "is not a mapping",
        ),
        ({"op": "set_param", "step": "templates", "path": "absent"}, "is not set"),
        ({"op": "set_param", "step": "templates", "path": "a..b", "value": 1}, "empty"),
        (
            {"op": "replace_step", "step": "templates", "document": {"id": "other"}},
            "is not 'templates'",
        ),
    ],
)
def test_an_edit_that_cannot_apply_names_its_position(
    op: dict[str, JsonValue], fragment: str
) -> None:
    with pytest.raises(PatchInvalid) as raised:
        _ = _patched(
            chain_recipe(),
            {"op": "set_param", "step": "speed", "path": "a", "value": 1},
            op,
        )

    assert raised.value.index == 1
    assert fragment in raised.value.problem


def test_a_reference_is_an_object_of_one_known_key() -> None:
    assert recipe_reference({"digest": "abc"}) == ("digest", "abc")
    assert recipe_reference({"saved": "speed"}) == ("saved", "speed")
    assert recipe_reference({"name": "speed"}) is None
    assert recipe_reference({"digest": "abc", "steps": []}) is None
    assert recipe_reference({"digest": 3}) is None


def test_a_saved_recipe_is_found_by_digest_and_by_name(tmp_path: Path) -> None:
    recipe = chain_recipe()
    _ = save_recipe(tmp_path, recipe)
    digest = recipe_digest(recipe)

    assert resolve_saved_recipe(tmp_path, ("digest", digest)) == recipe
    assert resolve_saved_recipe(tmp_path, ("saved", recipe.name)) == recipe


def test_a_recipe_saved_in_an_earlier_canonical_form_is_found_by_its_digest_now(
    tmp_path: Path,
) -> None:
    recipe = chain_recipe()
    path = recipe_path(tmp_path, "0" * 16)
    path.parent.mkdir(parents=True)
    _ = path.write_text(recipe.model_dump_json())

    assert resolve_saved_recipe(tmp_path, ("digest", recipe_digest(recipe))) == recipe


def test_an_unsaved_digest_or_name_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(RecipeNotSaved) as raised:
        _ = resolve_saved_recipe(tmp_path, ("digest", "0" * 16))
    with pytest.raises(RecipeNotSaved, match="lowercase hex"):
        _ = resolve_saved_recipe(tmp_path, ("digest", "XYZ"))
    with pytest.raises(RecipeNotSaved):
        _ = resolve_saved_recipe(tmp_path, ("saved", "speed"))

    assert raised.value.error_reason() == "recipe_not_saved"
    assert "patch_recipe" in str(raised.value)


def test_a_name_two_saved_recipes_share_is_refused_newest_first(tmp_path: Path) -> None:
    older = speed_recipe()
    newer = _patched(
        older,
        {"op": "set_param", "step": "speed", "path": "step_size", "value": 2},
    )
    old_path = save_recipe(tmp_path, older)
    _ = save_recipe(tmp_path, newer)
    os.utime(old_path, (1, 1))
    _ = save_recipe(tmp_path, named_recipe(older, "other"))

    with pytest.raises(RecipeNameAmbiguous) as raised:
        _ = resolve_saved_recipe(tmp_path, ("saved", "speed"))

    assert raised.value.digests == (recipe_digest(newer), recipe_digest(older))
