"""A recipe's ``repeat`` items expand at parse, and the expanded form is what is named."""

from __future__ import annotations

import json

import pytest

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    MAX_EXPANDED_ITEMS,
    FeatureStepSpec,
    Recipe,
    RecipeInvalid,
    StepRef,
    canonical_recipe,
    expand_repeats,
    read_recipe,
    recipe_digest,
    validation_report,
)
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    REPO_ROOT,
    chain_recipe,
)

BEHAVIORS: list[JsonValue] = list(range(7))
SEEDS: list[JsonValue] = [1, 2, 3]


def _speed(**extra: JsonValue) -> dict[str, JsonValue]:
    return {"type": "feature", "feature": "speed-angvel", "inputs": ["tracks"], **extra}


def test_a_repeated_step_stands_for_one_step_per_value_with_the_value_typed() -> None:
    recipe = read_recipe(
        {
            "steps": [
                _speed(
                    id="speed_{step}",
                    repeat={"step": [1, 2]},
                    params={"step_size": "{step}"},
                )
            ]
        }
    )

    assert recipe.ids == ("speed_1", "speed_2")
    assert [step.params for step in recipe.steps] == [
        {"step_size": 1},
        {"step_size": 2},
    ]


def test_several_names_expand_to_their_product_the_first_varying_slowest() -> None:
    recipe = read_recipe(
        {
            "steps": [
                _speed(
                    id="s_{a}_{b}",
                    repeat={"a": ["x", "y"], "b": [1, 2]},
                    params={"note": "{a}-{b}"},
                )
            ]
        }
    )

    assert recipe.ids == ("s_x_1", "s_x_2", "s_y_1", "s_y_2")
    assert recipe.step("s_y_1").params == {"note": "y-1"}


def test_a_list_item_expands_in_place_and_sees_the_outer_values() -> None:
    recipe = read_recipe(
        {
            "steps": [
                _speed(id="a_{s}", repeat={"s": [1, 2]}),
                {
                    "id": "joined_{t}",
                    "type": "feature",
                    "feature": "temporal-stack",
                    "repeat": {"t": ["p"]},
                    "inputs": [
                        "tracks",
                        {"step": "a_{s}", "repeat": {"s": [1, 2]}},
                    ],
                    "params": {"tag": "{t}"},
                },
            ]
        }
    )

    joined = recipe.step("joined_p")
    assert isinstance(joined, FeatureStepSpec)
    assert joined.inputs == ["tracks", StepRef(step="a_1"), StepRef(step="a_2")]
    assert joined.params == {"tag": "p"}


def test_braces_around_another_word_are_left_as_written() -> None:
    recipe = read_recipe(
        {"steps": [_speed(id="s{k}", repeat={"k": [1]}, params={"fmt": "{other}/{k}"})]}
    )

    assert recipe.step("s1").params == {"fmt": "{other}/1"}


def test_the_expanded_recipe_is_named_as_the_one_written_out() -> None:
    written = {
        "name": "seeds",
        "steps": [
            _speed(id="speed"),
            {
                "id": "t_{seed}",
                "type": "feature",
                "feature": "extract-templates",
                "repeat": {"seed": [1, 2]},
                "inputs": [{"step": "speed"}],
                "params": {"n_templates": 4, "random_state": "{seed}"},
            },
        ],
    }
    spelled = {
        "name": "seeds",
        "steps": [
            _speed(id="speed"),
            *(
                {
                    "id": f"t_{seed}",
                    "type": "feature",
                    "feature": "extract-templates",
                    "inputs": [{"step": "speed"}],
                    "params": {"n_templates": 4, "random_state": seed},
                }
                for seed in (1, 2)
            ),
        ],
    }

    repeated = read_recipe(written)

    assert recipe_digest(repeated) == recipe_digest(read_recipe(spelled))
    assert "repeat" not in json.dumps(canonical_recipe(repeated))
    assert recipe_digest(Recipe.model_validate(written)) == recipe_digest(repeated)


def test_a_recipe_without_repeat_is_returned_as_it_is() -> None:
    document = canonical_recipe(chain_recipe())

    assert expand_repeats(document) is document
    assert recipe_digest(read_recipe(document)) == recipe_digest(chain_recipe())


@pytest.mark.parametrize(
    ("step", "where", "fragment"),
    [
        (_speed(id="s", repeat={"k": [1, 2]}), "repeat", "must name {k}"),
        (_speed(id="s{k}", repeat={"k": []}), "repeat", "non-empty list"),
        (_speed(id="s{k}", repeat=["k"]), "repeat", "maps each name"),
        (_speed(id="s{k}", repeat={"k-1": [1]}), "repeat", "not an identifier"),
        (
            _speed(id="s{k}", repeat={"k": [[1]]}),
            "id",
            "must be a string, a number or a boolean",
        ),
        (
            _speed(id="s", inputs=[{"step": "a{v}", "repeat": {"v": [[1]]}}]),
            "inputs[0].step",
            "must be a string, a number or a boolean",
        ),
        (
            _speed(
                id="s{k}",
                repeat={"k": [1]},
                inputs=[{"step": "a", "repeat": {"k": [2]}}],
            ),
            "inputs[0].repeat",
            "already repeated",
        ),
        (
            _speed(id="s", inputs=[{"step": "a", "repeat": {"k": []}}]),
            "inputs[0].repeat",
            "non-empty list",
        ),
    ],
)
def test_a_repeat_that_cannot_expand_names_the_step_that_declared_it(
    step: dict[str, JsonValue], where: str, fragment: str
) -> None:
    with pytest.raises(RecipeInvalid) as raised:
        _ = read_recipe({"steps": [step]})

    (problem,) = raised.value.problems
    assert problem.step == step["id"]
    assert problem.where == where
    assert fragment in problem.message


def test_a_step_without_an_id_is_placed_by_its_position() -> None:
    with pytest.raises(RecipeInvalid) as raised:
        _ = read_recipe({"steps": [_speed(id="a"), _speed(repeat={"k": [1]})]})

    (problem,) = raised.value.problems
    assert (problem.step, problem.where) == ("", "steps[1].id")


def test_a_param_named_repeat_is_left_as_written() -> None:
    params: dict[str, JsonValue] = {
        "repeat": 3,
        "nested": {"repeat": {"k": [1]}},
        "items": [{"repeat": {"k": [1]}}],
    }
    document: dict[str, JsonValue] = {"steps": [_speed(id="s", params=params)]}

    assert expand_repeats(document) is document
    assert read_recipe(document).step("s").params == params


def test_an_id_that_is_a_placeholder_alone_stays_text() -> None:
    recipe = read_recipe({"steps": [_speed(id="{k}", repeat={"k": [1, 2]})]})

    assert recipe.ids == ("1", "2")


def test_the_limit_counts_the_inputs_of_every_step() -> None:
    with pytest.raises(RecipeInvalid) as raised:
        _ = read_recipe(
            {
                "steps": [
                    _speed(
                        id="s{a}",
                        repeat={"a": list(range(300))},
                        inputs=[{"step": "x{b}", "repeat": {"b": list(range(100))}}],
                    )
                ]
            }
        )

    (problem,) = raised.value.problems
    assert "more than" in problem.message


def test_a_repeat_past_the_limit_is_refused() -> None:
    side = int(MAX_EXPANDED_ITEMS**0.5) + 1
    with pytest.raises(RecipeInvalid, match="more than"):
        _ = read_recipe(
            {
                "steps": [
                    _speed(
                        id="s{a}_{b}",
                        repeat={"a": list(range(side)), "b": list(range(side))},
                    )
                ]
            }
        )


def test_a_fault_after_expansion_is_reported_at_the_expanded_step() -> None:
    with pytest.raises(RecipeInvalid) as raised:
        _ = read_recipe(
            {"steps": [_speed(id="s{k}", repeat={"k": [1, 2]}, run={"bogus": 1})]}
        )

    assert {problem.step for problem in raised.value.problems} == {"s1", "s2"}


def _task3_sized() -> dict[str, JsonValue]:
    """A per-behavior recipe the size of the CalMS21 Task 3 one, with repeats."""
    per_behavior: dict[str, JsonValue] = {"b": list(BEHAVIORS)}
    per_model: dict[str, JsonValue] = {"b": list(BEHAVIORS), "seed": list(SEEDS)}
    document: dict[str, JsonValue] = {
        "schema_version": 1,
        "name": "task 3 sized",
        "steps": [
            CONVERT_TRACKS_STEP,
            CONVERT_LABELS_STEP,
            {
                "id": "heading",
                "type": "feature",
                "feature": "heading",
                "inputs": ["tracks"],
                "tracks": {"step": "tracks"},
                "params": {"method": "two_point", "front_idx": 0, "rear_idx": 3},
            },
            {
                "id": "pairs",
                "type": "feature",
                "feature": "pair-position",
                "inputs": ["tracks", {"step": "heading"}],
                "tracks": {"step": "tracks"},
            },
            {
                "id": "labeled_{b}",
                "type": "feature",
                "feature": "extract-labeled-templates",
                "repeat": per_behavior,
                "inputs": [{"step": "pairs"}],
                "params": {"labels": {"step": "labels"}, "n_total": 300},
            },
            {
                "id": "xgb_{b}_s{seed}",
                "type": "feature",
                "feature": "xgboost",
                "repeat": per_model,
                "inputs": [{"step": "pairs"}],
                "params": {
                    "templates": {"step": "labeled_{b}"},
                    "default_class": 0,
                    "random_state": "{seed}",
                    "cv_folds": 3,
                },
            },
            {
                "id": "eval_{b}_s{seed}",
                "type": "feature",
                "feature": "evaluate-labels",
                "repeat": per_model,
                "inputs": [{"step": "xgb_{b}_s{seed}"}],
                "params": {"labels": {"step": "labels"}, "average_over": ["{b}"]},
            },
            {
                "id": "blend_{b}",
                "type": "feature",
                "feature": "probability-ensemble",
                "repeat": per_behavior,
                "inputs": [{"step": "xgb_{b}_s{seed}", "repeat": {"seed": SEEDS}}],
                "params": {"default_class": 0},
            },
            {
                "id": "tune_{b}",
                "type": "feature",
                "feature": "tune-threshold",
                "repeat": per_behavior,
                "inputs": [{"step": "blend_{b}"}],
                "params": {
                    "labels": {"step": "labels"},
                    "default_class": 0,
                    "average_over": ["{b}"],
                },
            },
            {
                "id": "apply_{b}",
                "type": "feature",
                "feature": "probability-ensemble",
                "repeat": per_behavior,
                "inputs": [{"step": "blend_{b}"}],
                "params": {
                    "default_class": 0,
                    "decision_threshold": {"step": "tune_{b}"},
                },
            },
            {
                "id": "score_{b}",
                "type": "feature",
                "feature": "evaluate-labels",
                "repeat": per_behavior,
                "inputs": [{"step": "apply_{b}"}],
                "params": {"labels": {"step": "labels"}, "average_over": ["{b}"]},
            },
        ],
    }
    return document


def test_a_task3_sized_recipe_is_under_twenty_declared_steps_and_validates() -> None:
    document = _task3_sized()
    steps = document["steps"]
    assert isinstance(steps, list)

    recipe = read_recipe(document)
    report = validation_report(document)

    assert len(steps) < 20
    assert len(recipe.steps) >= 71
    assert report.valid, report.problems
    assert len(json.dumps(document)) < 5_000
    blend = recipe.step("blend_6")
    assert isinstance(blend, FeatureStepSpec)
    assert [ref.step for ref in blend.inputs if isinstance(ref, StepRef)] == [
        "xgb_6_s1",
        "xgb_6_s2",
        "xgb_6_s3",
    ]


def test_the_chain_guide_repeat_example_has_the_steps_it_names() -> None:
    import re

    guide = REPO_ROOT / "docs" / "guides" / "pipelines" / "chain-steps.md"
    (block,) = [
        b
        for b in re.findall(r"```json\n(.*?)```", guide.read_text(), re.S)
        if '"repeat"' in b
    ]

    recipe = read_recipe(json.loads(block))

    assert recipe.ids == (
        "speed",
        "templates_s1",
        "templates_s2",
        "templates_s3",
        "clusters_s1",
        "clusters_s2",
        "clusters_s3",
    )
    assert recipe.step("templates_s2").params["random_state"] == 2
    assert validation_report(json.loads(block)).valid
