"""Pipeline validation reads each feature's column contract, with no dataset.

All of a step's inputs are merged into one frame, and a feature reads columns
from it by name. Each feature declares the columns it reads (``requires``) and
the columns of its output (``provides``). Validation walks the recipe and
propagates, per step, the columns guaranteed and the columns possible. A
required column absent from every input is a problem. A column that an input may
contain is a caution.

The recipe at the centre is a real analysis graph of 20 steps, copied from the
guppies paper into ``tests/data``. It validates. Three copies of it, each wired
wrong in a way that once failed only at run time, do not.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    TRACKS_INPUT,
    FeatureStepSpec,
    Problem,
    Recipe,
    RecipeInvalid,
    Step,
    StepRef,
    attach_candidates,
    can_connect,
    check_recipe,
    declaration_catalog,
    editor_catalog,
    input_candidates,
    read_recipe,
    resolved_columns,
    validation_report,
)
from tests.helpers import (
    GUPPIES_RECIPE,
    catalog_without_contract,
    golden_path,
    guppies_recipe,
    read_golden,
)

DELTA_STEPS = (
    "delta_focal_n10",
    "delta_focal_bf_n10",
    "delta_model_n10",
    "delta_focal_n30",
    "delta_focal_bf_n30",
    "delta_model_n30",
)
"""The guppies recipe's nn-delta-response steps, each reading ``ANGLE``."""

runner = CliRunner()


def _rewired(recipe: Recipe, step_id: str, inputs: list[str]) -> Recipe:
    """*recipe* with *step_id* reading *inputs*, each a step id or ``tracks``."""
    items: list[str | StepRef] = [
        TRACKS_INPUT if name == TRACKS_INPUT else StepRef(step=name) for name in inputs
    ]
    steps: list[Step] = [
        step.model_copy(update={"inputs": items}) if step.id == step_id else step
        for step in recipe.steps
    ]
    return recipe.model_copy(update={"steps": steps})


def _with_params(recipe: Recipe, step_id: str, params: dict[str, JsonValue]) -> Recipe:
    """*recipe* with *params* set on *step_id*, over the params it has."""
    steps: list[Step] = [
        step.model_copy(update={"params": {**step.params, **params}})
        if step.id == step_id
        else step
        for step in recipe.steps
    ]
    return recipe.model_copy(update={"steps": steps})


def _inputs_of(recipe: Recipe, step_id: str) -> list[str]:
    """The ids *step_id* reads, ``tracks`` included."""
    step = recipe.step(step_id)
    assert isinstance(step, FeatureStepSpec)
    return [item if isinstance(item, str) else item.step for item in step.inputs]


# --- the guppies recipe ---------------------------------------------------------


def test_the_guppies_recipe_validates_with_cautions_only() -> None:
    """Its one unconfirmed column is ``ANGLE``, possible in its tracks by schema."""
    report = validation_report(read_golden(GUPPIES_RECIPE))

    assert report.valid, report.problems
    assert report.problems == ()
    assert [caution.step for caution in report.cautions] == list(DELTA_STEPS)
    assert all("'ANGLE'" in caution.message for caution in report.cautions)


def test_nearest_neighbor_reading_speed_is_refused_at_its_inputs() -> None:
    """The wire that used to fail at run time with "not a track-shaped table"."""
    recipe = _rewired(guppies_recipe(), "nn", ["speed"])
    expected = (
        "nearest-neighbor reads a track-shaped table, and speed-angvel neither "
        "passes its input through nor writes X and Y"
    )

    assert check_recipe(recipe) == (Problem("nn", "inputs", expected),)


def test_a_delta_step_without_its_neighbor_input_is_refused() -> None:
    """The wire that used to fail with "needs a neighbor id column"."""
    guppies = guppies_recipe()
    without_nn = [
        name for name in _inputs_of(guppies, "delta_focal_n10") if name != "nn"
    ]
    recipe = _rewired(guppies, "delta_focal_n10", without_nn)
    expected = (
        "needs 'nn_id' or 'nn_fishID', and its merged input from smooth, speed, "
        "focal_tags lacks it. nearest-neighbor provides it"
    )

    assert check_recipe(recipe) == (Problem("delta_focal_n10", "inputs", expected),)


def test_a_bins_step_reading_speed_names_every_response_column() -> None:
    """The wire that used to raise a ``KeyError`` inside ``nn-delta-bins``."""
    recipe = _rewired(guppies_recipe(), "bins_focal_n10", ["speed", "focal_tags"])

    problems = check_recipe(recipe)

    assert [(problem.step, problem.where) for problem in problems] == [
        ("bins_focal_n10", "inputs")
    ] * 4
    for column, problem in zip(
        ("neighbor_x", "neighbor_y", "dangle", "dspeed"), problems, strict=True
    ):
        assert problem.message == (
            f"needs '{column}', and its merged input from speed, focal_tags "
            "lacks it. nn-delta-response provides it"
        )


@pytest.mark.parametrize(
    "category_specs",
    [[], [{"source_col": "speed_smooth", "new_col": "fast"}]],
    ids=["no-spec", "a-spec-writing-another-column"],
)
def test_a_bins_step_without_a_category_column_is_a_caution(
    category_specs: JsonValue,
) -> None:
    """``nn-delta-bins`` writes rows per focal or neighbor category, and none without.

    Without ``focal_tags`` no step writes ``Focal_fish``, and so ``nn-delta-response``
    writes no ``neighbor_focal``. The tracks may contain ``Focal_fish``. A spec
    writing a third column leaves both category columns to the input.
    """
    guppies = guppies_recipe()
    kept = {"smooth", "speed", "nn", "delta_focal_n10", "bins_focal_n10"}
    recipe = guppies.model_copy(
        update={"steps": [step for step in guppies.steps if step.id in kept]}
    )
    for step_id in ("delta_focal_n10", "bins_focal_n10"):
        without_tags = [
            name for name in _inputs_of(recipe, step_id) if name != "focal_tags"
        ]
        recipe = _rewired(recipe, step_id, without_tags)
    recipe = _with_params(recipe, "bins_focal_n10", {"category_specs": category_specs})

    report = validation_report(recipe.model_dump(mode="json"))

    assert report.valid, report.problems
    (caution,) = [c for c in report.cautions if c.step == "bins_focal_n10"]
    assert "'Focal_fish' or 'neighbor_focal'" in caution.message


_CATEGORY_FROM_SPEED: dict[str, JsonValue] = {
    "focal_category_col": "fast",
    "neighbor_category_col": "neighbor_fast",
    "category_specs": [
        {"source_col": "speed_smooth", "new_col": "fast", "quantile": 0.5}
    ],
}
"""Bins whose focal category column a spec writes from the smoothed speed."""


def _bins_below_delta(speed_params: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """``nn-delta-response`` over the tracks, then bins under :data:`_CATEGORY_FROM_SPEED`."""

    def feature(
        step_id: str, slug: str, inputs: list[JsonValue], params: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return {
            "id": step_id,
            "type": "feature",
            "feature": slug,
            "inputs": inputs,
            "params": params,
        }

    return {
        "steps": [
            feature("heading", "heading", [TRACKS_INPUT], {}),
            feature("speed", "speed-angvel", [TRACKS_INPUT], speed_params),
            feature("nn", "nearest-neighbor", [TRACKS_INPUT], {}),
            feature(
                "delta",
                "nn-delta-response",
                [TRACKS_INPUT, {"step": "heading"}, {"step": "speed"}, {"step": "nn"}],
                {"speed_col": "speed"},
            ),
            feature("bins", "nn-delta-bins", [{"step": "delta"}], _CATEGORY_FROM_SPEED),
        ]
    }


def test_a_category_column_that_a_spec_writes_is_not_required_of_the_input() -> None:
    """The input lacks ``fast`` and ``neighbor_fast``, and has the spec's source."""
    report = validation_report(_bins_below_delta({"smooth_window": 5}))

    assert report.valid, report.problems
    assert [caution for caution in report.cautions if caution.step == "bins"] == []


def test_a_spec_whose_source_column_is_missing_is_a_problem() -> None:
    """``speed_smooth`` exists only when ``smooth_window`` is set."""
    report = validation_report(_bins_below_delta({}))
    expected = (
        "needs 'speed_smooth', and its merged input from delta lacks it. "
        "speed-angvel (when smooth_window is set) provides it"
    )

    assert report.problems == (Problem("bins", "inputs", expected),)


def test_the_fixture_is_the_recipe_the_paper_runs() -> None:
    """Twenty steps, joins of up to five inputs: the shape the checks are about."""
    recipe = guppies_recipe()

    assert len(recipe.steps) == 20
    assert max(len(_inputs_of(recipe, step.id)) for step in recipe.steps) == 5


# --- one wire ---------------------------------------------------------------------


def test_a_track_input_is_refused_below_a_feature_that_drops_the_track_frame() -> None:
    catalog = declaration_catalog()
    consumer = catalog.entries["nearest-neighbor"].consumes

    assert not can_connect(catalog.entries["speed-angvel"].produces, consumer)
    assert can_connect(catalog.entries["trajectory-smooth"].produces, consumer)
    assert can_connect(catalog.entries[TRACKS_INPUT].produces, consumer)


def test_attaching_nearest_neighbor_follows_the_same_rule() -> None:
    recipe = guppies_recipe()

    below_speed = attach_candidates(recipe, "speed").candidates["nearest-neighbor"]
    below_smooth = attach_candidates(recipe, "smooth").candidates["nearest-neighbor"]

    assert not below_speed.allowed
    assert "neither passes its input through" in below_speed.reason
    assert below_smooth.allowed
    assert below_smooth.template == {
        "type": "feature",
        "feature": "nearest-neighbor",
        "inputs": [{"step": "smooth"}],
    }


# --- a step still being wired -------------------------------------------------------


def _partial_delta() -> Recipe:
    """The guppies recipe's upstream, and a delta step reading only ``smooth``."""
    guppies = guppies_recipe()
    kept = [
        step
        for step in guppies.steps
        if step.id in {"smooth", "ff10", "speed", "nn", "focal_tags"}
    ]
    return Recipe.model_validate(
        {
            "steps": [
                *(step.model_dump(mode="json") for step in kept),
                {
                    "id": "delta",
                    "type": "feature",
                    "feature": "nn-delta-response",
                    "inputs": [{"step": "smooth"}],
                    "params": {"speed_col": "speed_smooth"},
                },
                {
                    "id": "bins",
                    "type": "feature",
                    "feature": "nn-delta-bins",
                    "inputs": [{"step": "delta"}],
                },
            ]
        }
    )


def test_input_candidates_offer_the_steps_that_supply_what_is_still_needed() -> None:
    answer = input_candidates(_partial_delta(), "delta")

    assert answer.step == "delta"
    still = {still.need.label: still for still in answer.still_needs}
    assert set(still) == {
        "'ANGLE'",
        "'speed_smooth' or 'speed'",
        "'nn_id' or 'nn_fishID'",
    }
    assert still["'nn_id' or 'nn_fishID'"].status == "absent"
    assert [p.feature for p in still["'nn_id' or 'nn_fishID'"].providers] == [
        "nearest-neighbor"
    ]
    assert still["'ANGLE'"].status == "possible"

    nn = answer.candidates["nn"]
    assert nn.allowed
    assert nn.input == {"step": "nn"}
    assert "nn_id" in nn.supplies
    assert nn.meets == ("'nn_id' or 'nn_fishID'",)
    speed = answer.candidates["speed"]
    assert speed.allowed
    assert "speed_smooth" in speed.supplies
    assert speed.meets == ("'speed_smooth' or 'speed'",)


def test_input_candidates_never_offer_the_step_its_inputs_or_its_descendants() -> None:
    answer = input_candidates(_partial_delta(), "delta")

    assert not {"delta", "bins", "smooth"} & set(answer.candidates)
    assert answer.candidates[TRACKS_INPUT].input == TRACKS_INPUT


def _speed(step_id: str, params: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """A ``speed-angvel`` step over ``smooth``."""
    return {
        "id": step_id,
        "type": "feature",
        "feature": "speed-angvel",
        "inputs": [{"step": "smooth"}],
        "params": params,
    }


def _speed_twice(
    params_a: dict[str, JsonValue], params_b: dict[str, JsonValue]
) -> tuple[Recipe, Recipe]:
    """An aggregate reading ``speed_a``, and one reading ``speed_a`` and ``speed_b``."""
    upstream: list[JsonValue] = [
        _SMOOTH,
        _speed("speed_a", params_a),
        _speed("speed_b", params_b),
    ]
    partial = read_recipe(
        {"steps": [*upstream, _aggregate("speed", [{"step": "speed_a"}])]}
    )
    joined = read_recipe(
        {
            "steps": [
                *upstream,
                _aggregate("speed", [{"step": "speed_a"}, {"step": "speed_b"}]),
            ]
        }
    )
    return partial, joined


def test_two_runs_of_one_feature_may_both_be_inputs() -> None:
    """They share a storage name, and their params give them two identifiers.

    The candidate offered for a step and the validation of the wire it makes agree.
    """
    partial, joined = _speed_twice({"smooth_window": 5}, {"step_size": 4})

    assert input_candidates(partial, "aggregate").candidates["speed_b"].allowed
    assert check_recipe(joined) == ()


@pytest.mark.parametrize(
    ("params_a", "params_b"),
    [({"step_size": 4}, {"step_size": 4}), ({}, {"step_size": None})],
    ids=["the-same-params", "a-default-written-out"],
)
def test_two_steps_that_resolve_to_one_run_are_not_two_inputs(
    params_a: dict[str, JsonValue], params_b: dict[str, JsonValue]
) -> None:
    """A plan gives them one run id, which a step's inputs list once.

    Validation reports the wire, and the candidate for it is refused with the same
    reason.
    """
    partial, joined = _speed_twice(params_a, params_b)
    expected = (
        "'speed_a' and 'speed_b' run speed-angvel with the same params, inputs "
        "and run arguments. They resolve to one run, and a step reads each run once"
    )

    assert check_recipe(joined) == (Problem("aggregate", "inputs", expected),)
    candidate = input_candidates(partial, "aggregate").candidates["speed_b"]
    assert not candidate.allowed
    assert candidate.reason == expected


def test_one_step_listed_twice_among_the_inputs_is_a_problem() -> None:
    recipe = read_recipe(
        {
            "steps": [
                _SMOOTH,
                _speed("speed_a", {}),
                _aggregate("speed", [{"step": "speed_a"}, {"step": "speed_a"}]),
            ]
        }
    )
    expected = (
        "'speed_a' appears 2 times among the inputs, and a step reads each input once"
    )

    assert check_recipe(recipe) == (Problem("aggregate", "inputs", expected),)


def test_a_still_needed_column_is_a_problem_once_the_recipe_is_finished() -> None:
    """The partial answer and validation agree on what is missing."""
    problems = check_recipe(_partial_delta())

    assert [(problem.step, problem.where) for problem in problems] == [
        ("delta", "inputs"),
        ("delta", "inputs"),
    ]
    assert "'speed_smooth' or 'speed'" in problems[0].message
    assert "'nn_id' or 'nn_fishID'" in problems[1].message


def test_input_candidates_refuse_a_step_that_reads_no_inputs() -> None:
    recipe = Recipe.model_validate(
        {"steps": [{"id": "trex", "type": "op", "kind": "trex"}]}
    )
    with pytest.raises(RecipeInvalid) as raised:
        _ = input_candidates(recipe, "trex")
    assert raised.value.problems[0].where == "kind"
    with pytest.raises(RecipeInvalid):
        _ = input_candidates(recipe, "no-such-step")


# --- what decides a column ---------------------------------------------------------------


def _speed_then_summary(speed_params: dict[str, object]) -> Recipe:
    return Recipe.model_validate(
        {
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "params": speed_params,
                },
                {
                    "id": "summary",
                    "type": "feature",
                    "feature": "social-motion-summary",
                    "inputs": [{"step": "speed"}],
                    "params": {"speed_col": "speed_smooth"},
                },
            ]
        }
    )


def test_a_column_a_param_decides_is_guaranteed_or_absent_from_the_recipe() -> None:
    """``speed_smooth`` exists when, and only when, ``smooth_window`` is set."""
    assert check_recipe(_speed_then_summary({"smooth_window": 5})) == ()

    problems = check_recipe(_speed_then_summary({}))
    assert len(problems) == 1
    assert problems[0].message.endswith(
        "speed-angvel (when smooth_window is set) provides it"
    )


def test_a_column_an_upstream_column_decides_is_only_possible() -> None:
    """``angvel`` needs ``ANGLE``, which the tracks may contain by schema."""
    columns = resolved_columns(_speed_then_summary({"smooth_window": 5}))["speed"]

    assert {"speed", "vx", "vy", "speed_smooth", "id", "frame"} <= columns.guaranteed
    assert "angvel" in columns.possible - columns.guaranteed
    assert "speed_step" not in columns.possible


def test_a_passthrough_keeps_its_input_and_adds_its_own() -> None:
    columns = resolved_columns(guppies_recipe())["smooth"]

    assert {"X", "Y", "id", "frame", "bad_frame"} <= columns.guaranteed
    assert "ANGLE" in columns.possible
    assert "poseX" in columns.possible_prefixes


def _aggregate(column: str, inputs: list[JsonValue]) -> dict[str, JsonValue]:
    """A ``frame-aggregate`` step over *column*, reading *inputs*."""
    return {
        "id": "aggregate",
        "type": "feature",
        "feature": "frame-aggregate",
        "inputs": inputs,
        "params": {"column": column},
    }


_SMOOTH: dict[str, JsonValue] = {
    "id": "smooth",
    "type": "feature",
    "feature": "trajectory-smooth",
}


@pytest.mark.parametrize("column", ["poseP0", "X#head"])
@pytest.mark.parametrize("via", [TRACKS_INPUT, "smooth"])
def test_a_tracker_column_no_declaration_names_is_a_caution(
    column: str, via: str
) -> None:
    """SLEAP writes ``poseP0`` and TRex writes ``X#head``, two columns outside every schema."""
    reads: JsonValue = via if via == TRACKS_INPUT else {"step": via}
    report = validation_report({"steps": [_SMOOTH, _aggregate(column, [reads])]})

    assert report.valid, report.problems
    (caution,) = [caution for caution in report.cautions if caution.step == "aggregate"]
    assert f"'{column}'" in caution.message


def test_a_column_a_feature_writes_is_absent_from_the_tracks() -> None:
    """The tracks contain a column that a declaration names only where a schema does."""
    report = validation_report({"steps": [_aggregate("nn_id", [TRACKS_INPUT])]})
    expected = (
        "needs 'nn_id', and its merged input from tracks lacks it. "
        "nearest-neighbor; nn-delta-response provide it"
    )

    assert report.problems == (Problem("aggregate", "inputs", expected),)


def test_a_missing_track_column_names_the_tracks_input() -> None:
    speed: dict[str, JsonValue] = {
        "id": "speed",
        "type": "feature",
        "feature": "speed-angvel",
    }
    report = validation_report({"steps": [speed, _aggregate("X", [{"step": "speed"}])]})
    expected = (
        "needs 'X', and its merged input from speed lacks it. "
        "The tracks input provides it"
    )

    assert report.problems == (Problem("aggregate", "inputs", expected),)

    delta: dict[str, JsonValue] = {
        "id": "delta",
        "type": "feature",
        "feature": "nn-delta-response",
        "inputs": [{"step": "speed"}],
    }
    answer = input_candidates(read_recipe({"steps": [speed, delta]}), "delta")
    still = {still.need.label: still for still in answer.still_needs}
    assert [provider.feature for provider in still["'X'"].providers] == [TRACKS_INPUT]
    assert "'X'" in answer.candidates[TRACKS_INPUT].meets


def test_an_undeclared_producer_never_causes_a_refusal() -> None:
    """Below an undeclared step, a requirement is a caution."""
    document: dict[str, object] = {
        "steps": [
            {"id": "subsample", "type": "feature", "feature": "track-subsample"},
            {
                "id": "bins",
                "type": "feature",
                "feature": "nn-delta-bins",
                "inputs": [{"step": "subsample"}],
            },
        ]
    }
    report = validation_report(document)

    assert report.valid, report.problems
    unchecked = [caution for caution in report.cautions if caution.step == "bins"]
    assert len(unchecked) == 1
    assert "cannot be checked" in unchecked[0].message
    assert "'neighbor_x'" in unchecked[0].message


def test_an_undeclared_consumer_is_listed_unchecked_and_not_cautioned() -> None:
    report = validation_report(
        {
            "steps": [
                {"id": "smooth", "type": "feature", "feature": "trajectory-smooth"},
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": [{"step": "smooth"}],
                },
            ]
        },
        catalog_without_contract("trajectory-smooth"),
    )

    assert report.valid
    assert report.unchecked == ("smooth",)
    assert all(caution.step != "smooth" for caution in report.cautions)


def test_a_document_that_does_not_parse_lists_no_unchecked_step() -> None:
    report = validation_report({"steps": [{"id": "a", "type": "feature"}]})

    assert not report.valid
    assert report.unchecked == ()


# --- the wire ---------------------------------------------------------------------------


def test_the_catalog_states_each_feature_s_columns_under_its_defaults() -> None:
    entries = {entry.name: entry for entry in editor_catalog().entries}
    speed = entries["speed-angvel"]

    assert speed.requires is not None
    assert [need.label for need in speed.requires] == [
        "'id'",
        "'X'",
        "'Y'",
        "'frame' or 'time'",
    ]
    assert speed.provides is not None
    assert not speed.provides.passthrough
    assert [names.label for names in speed.provides.adds] == ["'speed'", "'vx'", "'vy'"]
    smoothed = next(
        conditional
        for conditional in speed.provides.conditional
        if conditional.names.columns == ("speed_smooth",)
    )
    assert smoothed.when_param == "smooth_window"

    bins = entries["nn-delta-bins"]
    assert bins.provides is not None
    assert "'Exp'" in [names.label for names in bins.provides.adds]

    undeclared = entries["global-tsne"]
    assert undeclared.requires is None
    assert undeclared.provides is None


def test_the_cli_prints_cautions_beside_a_valid_verdict() -> None:
    result = runner.invoke(
        app, ["pipeline", "validate", "--recipe", f"@{golden_path(GUPPIES_RECIPE)}"]
    )

    assert result.exit_code == 0, result.output
    assert "This recipe is valid." in result.stdout
    assert result.stderr.count("caution:") == len(DELTA_STEPS)
