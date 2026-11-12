"""Pipeline validation reads each feature's column contract, with no dataset.

All of a step's inputs are merged into one frame, and a feature reads columns
from it by name. Each feature declares the columns it reads (``requires``) and
the columns of its output (``provides``). Validation walks the recipe and
propagates, per step, the columns guaranteed and the columns possible. A
required column absent from every input is a problem. A column that an input may
contain is a caution.

The recipe at the center is a real analysis graph of 20 steps, copied from the
guppies paper into ``tests/data``. It validates. Three copies of it, each wired
wrong in a way that once failed only at run time, do not.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.column_contract import (
    META_KEPT,
    Column,
    PAYLOAD_LABEL,
    NamedColumns,
    Payload,
    Prefix,
    Provides,
)
from mosaic.core.pipeline.graph import (
    TRACKS_INPUT,
    UNKNOWN_COLUMNS,
    ColumnSet,
    Declaration,
    DeclarationCatalog,
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
from mosaic.core.pipeline.graph.validate import pair_rules
from mosaic.core.pipeline.graph.columns import (
    merged_input,
    need_status,
    needs_of,
    output_columns,
    provided_wire,
    providers_of,
    requires_wire,
    tracks_columns,
    tracks_provides,
)
from mosaic.core.pipeline.graph.validate import merge_cautions
from mosaic.core.pipeline.graph.wire import NeedStatus
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
    (merged,) = merge_cautions(report.cautions)
    assert merged.step == DELTA_STEPS[0]
    assert all(f"'{step}'" in merged.message for step in DELTA_STEPS[1:])


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
            "lacks it. neighbor-delta-response; nn-delta-response provide it"
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


# --- providers in a track space -----------------------------------------------------

_METRICS: dict[str, JsonValue] = {
    "id": "metrics",
    "type": "feature",
    "feature": "ffgroups-metrics",
    "inputs": ["tracks"],
}
"""A planar tracks reader whose ``speed`` its tracks lack.

``speed-angvel`` (planar) and ``pair-world`` (world frame) both write ``speed``.
"""


def _speed_need() -> NamedColumns:
    """The ``speed`` requirement of ``ffgroups-metrics``."""
    declared = declaration_catalog().get("ffgroups-metrics")
    assert declared is not None and declared.requires is not None
    [speed] = [
        need
        for need in needs_of(declared.requires, declared.param_defaults)
        if need.label == "'speed'"
    ]
    return speed


def _speed_providers(consumer: Declaration, besides: str = "") -> list[str]:
    """The providers of ``speed`` offered to a step declared as *consumer*."""
    catalog = declaration_catalog()
    providers = providers_of(
        _speed_need(),
        catalog,
        tracks_columns(catalog),
        besides=besides,
        consumer=consumer,
    )
    return [provider.feature for provider in providers]


def _declared(slug: str) -> Declaration:
    declared = declaration_catalog().get(slug)
    assert declared is not None
    return declared


def test_a_planar_tracks_reader_is_offered_no_world_frame_provider() -> None:
    answer = input_candidates(read_recipe({"steps": [_METRICS]}), "metrics")

    still = {still.need.label: still for still in answer.still_needs}
    assert [provider.feature for provider in still["'speed'"].providers] == [
        "speed-angvel"
    ]


def test_a_world_frame_tracks_reader_is_offered_no_planar_provider() -> None:
    """``pose-egocentric`` as a reader that also takes inputs of any row shape.

    As registered it requires the track shape, which neither provider's output
    keeps, so the track space would never be the reason one is left out.
    """
    world = _declared("pose-egocentric")
    any_shape = replace(
        world, consumes=replace(world.consumes, requires_track_shape=False)
    )

    assert _speed_providers(any_shape) == ["pair-world"]


def test_a_reader_of_features_alone_is_offered_providers_in_every_space() -> None:
    """It reads no tracks, so the space of its input's tracks is its input's to check."""
    providers = _speed_providers(_declared("global-kmeans"), besides="global-kmeans")

    assert {"pair-world", "speed-angvel"} <= set(providers)


def test_the_problem_for_an_absent_column_names_providers_in_the_step_space() -> None:
    report = validation_report({"steps": [_METRICS]})

    [message] = [
        problem.message
        for problem in report.problems
        if problem.step == "metrics" and "'speed'" in problem.message
    ]
    assert message.endswith("speed-angvel provides it"), message
    assert "pair-world" not in message


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


_SMOOTH_MARKS = frozenset(
    {"bad_frame", "interpolated", "interp_run_frames", "unbridgeable", "long_interp"}
)


@pytest.mark.parametrize(
    "inputs", [("smooth", "ff20", "speed"), ("nn", "speed", "ff10", "smooth")]
)
def test_the_smoothing_marks_reach_a_summary_under_their_own_names(
    inputs: tuple[str, ...],
) -> None:
    """A summary's ``exclude_cols`` names the marks, wherever ``smooth`` merges in."""
    columns = resolved_columns(guppies_recipe())

    merged = merged_input([columns[step] for step in inputs])

    assert merged.guaranteed >= _SMOOTH_MARKS


def _aggregate(column: str, inputs: list[JsonValue]) -> dict[str, JsonValue]:
    """A ``frame-aggregate`` step over *column*, reading *inputs*."""
    return {
        "id": "aggregate",
        "type": "feature",
        "feature": "frame-aggregate",
        "inputs": inputs,
        "params": {"columns": [column]},
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


def test_the_tracks_may_hold_a_third_axis() -> None:
    """``mosaic_world_v1`` requires ``Z`` and its keypoints carry ``poseZ``."""
    possible = tracks_provides()
    columns = {
        c.column.name for c in possible.conditional if isinstance(c.column, Column)
    }
    prefixes = {
        c.column.prefix for c in possible.conditional if isinstance(c.column, Prefix)
    }

    assert "Z" in columns
    assert {"poseX", "poseY", "poseZ"} <= prefixes


def test_the_tracks_still_feed_keypoint_kinematics() -> None:
    steps: list[JsonValue] = [
        {
            "id": "kin",
            "type": "feature",
            "feature": "keypoint-kinematics",
            "inputs": [TRACKS_INPUT],
        }
    ]

    report = validation_report({"steps": steps})

    assert report.valid, report.problems


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
    report = validation_report(document, catalog_without_contract("track-subsample"))

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


@pytest.mark.parametrize(
    ("available", "status"),
    [
        (ColumnSet(guaranteed=frozenset({"frame", "speed"})), "guaranteed"),
        (ColumnSet(guaranteed_prefixes=frozenset({"poseX"})), "guaranteed"),
        (
            ColumnSet(guaranteed=frozenset({"frame"}), possible=frozenset({"speed"})),
            "possible",
        ),
        (ColumnSet(possible_prefixes=frozenset({"poseX"})), "possible"),
        (ColumnSet(guaranteed=frozenset({"frame", "id"}), unlisted=True), "possible"),
        (UNKNOWN_COLUMNS, "unknown"),
        (ColumnSet(guaranteed=frozenset({"frame", "id", "fps", "label"})), "absent"),
    ],
)
def test_a_payload_requirement_reads_any_column_outside_the_bookkeeping(
    available: ColumnSet, status: NeedStatus
) -> None:
    assert need_status(NamedColumns(payload=True), available) == status


def _payload_catalog() -> DeclarationCatalog:
    """The registries' catalog, with two declarations a payload test needs.

    ``extract-templates`` requires a numeric payload and passes its input
    through. The output of ``heading`` lacks a payload column, because it is
    declared to write the identity columns alone.
    """
    catalog = declaration_catalog()
    entries = dict(catalog.entries)
    entries["extract-templates"] = replace(
        entries["extract-templates"],
        requires=(Payload(),),
        provides=Provides(passthrough=True),
    )
    entries["heading"] = replace(
        entries["heading"], provides=Provides(conditional=META_KEPT)
    )
    return DeclarationCatalog(entries=entries)


def _templates_over(feature: str) -> dict[str, JsonValue]:
    return {
        "steps": [
            {"id": "upstream", "type": "feature", "feature": feature},
            {
                "id": "templates",
                "type": "feature",
                "feature": "extract-templates",
                "inputs": [{"step": "upstream"}],
                "params": {"n_templates": 10},
            },
        ]
    }


def test_validation_meets_a_payload_requirement_with_any_measurement() -> None:
    report = validation_report(_templates_over("speed-angvel"), _payload_catalog())

    assert report.valid, report.problems
    assert all(caution.step != "templates" for caution in report.cautions)
    assert report.unchecked == ()


def test_validation_refuses_a_payload_requirement_on_identity_columns_alone() -> None:
    report = validation_report(_templates_over("heading"), _payload_catalog())

    problems = [problem for problem in report.problems if problem.step == "templates"]
    assert [problem.where for problem in problems] == ["inputs"]
    assert problems[0].message.startswith(f"needs {PAYLOAD_LABEL}")


def test_the_catalog_states_a_payload_requirement() -> None:
    (need,) = requires_wire((Payload(),), {})

    assert need.payload
    assert need.label == PAYLOAD_LABEL
    assert (need.columns, need.prefixes) == ((), ())


# --- columns an aggregate keeps -------------------------------------------------------

AGGREGATES = ("frame-aggregate", "sequence-aggregate", "individual-aggregate")
"""The aggregates that take ``keep_columns``."""


def _aggregate_step(
    step_id: str, feature: str, inputs: list[JsonValue], keep: list[str]
) -> dict[str, JsonValue]:
    """An aggregate of ``speed`` that keeps the columns *keep*."""
    return {
        "id": step_id,
        "type": "feature",
        "feature": feature,
        "inputs": inputs,
        "params": {"columns": ["speed"], "aggs": ["mean"], "keep_columns": list(keep)},
    }


_SPEED: dict[str, JsonValue] = {
    "id": "speed",
    "type": "feature",
    "feature": "speed-angvel",
}


@pytest.mark.parametrize("feature", AGGREGATES)
def test_a_kept_column_that_the_feature_input_cannot_provide_is_refused(
    feature: str,
) -> None:
    """``speed-angvel`` declares no ``strain``, and the merged input holds no tracks."""
    steps = [_SPEED, _aggregate_step("agg", feature, [{"step": "speed"}], ["strain"])]
    expected = (
        "needs 'strain', and its merged input from speed lacks it. "
        "The tracks input may contain it"
    )

    assert validation_report({"steps": steps}).problems == (
        Problem("agg", "inputs", expected),
    )


@pytest.mark.parametrize("feature", AGGREGATES)
def test_a_kept_column_is_a_caution_when_the_tracks_are_among_the_inputs(
    feature: str,
) -> None:
    """The tracks table's columns are not declared, so ``strain`` may be there."""
    inputs: list[JsonValue] = [{"step": "speed"}, TRACKS_INPUT]
    steps = [_SPEED, _aggregate_step("agg", feature, inputs, ["strain"])]

    report = validation_report({"steps": steps})

    assert report.valid, report.problems
    (caution,) = report.cautions
    assert caution.step == "agg"
    assert "'strain' may be missing from its merged input" in caution.message


@pytest.mark.parametrize("feature", AGGREGATES)
def test_a_kept_column_that_the_feature_input_declares_is_not_refused(
    feature: str,
) -> None:
    steps = [_SPEED, _aggregate_step("agg", feature, [{"step": "speed"}], ["vx"])]

    report = validation_report({"steps": steps})

    assert report.valid, report.problems
    assert report.cautions == ()


def test_a_frame_aggregate_provides_the_columns_it_keeps() -> None:
    """``sequence-aggregate`` below it keeps ``strain`` without a caution of its own."""
    tracks_too: list[JsonValue] = [{"step": "speed"}, TRACKS_INPUT]
    steps = [
        _SPEED,
        _aggregate_step("frame", "frame-aggregate", tracks_too, ["strain"]),
        {
            "id": "sequence",
            "type": "feature",
            "feature": "sequence-aggregate",
            "inputs": [{"step": "frame"}],
            "params": {"columns": ["speed_mean"], "keep_columns": ["strain"]},
        },
    ]

    report = validation_report({"steps": steps})

    assert report.valid, report.problems
    assert [caution.step for caution in report.cautions] == ["frame"]


# --- payload families -------------------------------------------------------------------

_FAMILIES = Provides(conditional=META_KEPT, payload_families=True)
"""An output named after its input's payload columns, as ``temporal-stack`` writes."""


def test_a_payload_family_is_named_after_each_payload_column_of_the_input() -> None:
    merged = ColumnSet(
        guaranteed=frozenset({"frame", "id", "fps", "speed"}),
        possible=frozenset({"frame", "id", "fps", "speed", "angvel"}),
        guaranteed_prefixes=frozenset({"poseX"}),
        possible_prefixes=frozenset({"poseX", "poseP"}),
    )

    output = output_columns(_FAMILIES, merged, {})

    assert output.guaranteed == {"frame", "id"}
    assert output.guaranteed_prefixes == {"speed", "poseX"}
    assert output.possible_prefixes == {"speed", "angvel", "poseX", "poseP"}
    assert output.admits("speed__t+05")
    assert not output.admits("fps")


@pytest.mark.parametrize(
    ("need", "status"),
    [
        (NamedColumns(payload=True), "guaranteed"),
        (NamedColumns(prefixes=("poseX",)), "guaranteed"),
        (NamedColumns(columns=("speed",)), "possible"),
        (NamedColumns(columns=("speed__pool_mean",)), "possible"),
        (NamedColumns(columns=("vx",)), "absent"),
    ],
)
def test_a_payload_family_meets_a_payload_need_and_guarantees_no_name(
    need: NamedColumns, status: NeedStatus
) -> None:
    merged = ColumnSet(
        guaranteed=frozenset({"frame", "id", "speed"}),
        guaranteed_prefixes=frozenset({"poseX"}),
    )

    assert need_status(need, output_columns(_FAMILIES, merged, {})) == status


def test_payload_families_keep_what_the_input_leaves_open() -> None:
    """Families of an unbounded input are unbounded, and of the tracks unlisted."""
    assert output_columns(_FAMILIES, UNKNOWN_COLUMNS, {}).unbounded

    tracks = tracks_columns(declaration_catalog())
    families = output_columns(_FAMILIES, tracks, {})
    assert families.unlisted
    assert families.vocabulary == tracks.vocabulary
    assert families.admits("poseP0__t+00")


def test_the_catalog_states_payload_families() -> None:
    assert provided_wire(_FAMILIES, {}).payload_families
    assert not provided_wire(Provides(passthrough=True), {}).payload_families


@pytest.mark.parametrize("producer", ["scale-to-cm", "temporal-stack"])
def test_an_output_named_after_its_input_may_feed_a_track_feature(
    producer: str,
) -> None:
    """Its payload families may include ``X`` and ``Y``."""
    catalog = declaration_catalog()

    assert can_connect(
        catalog.entries[producer].produces, catalog.entries["nearest-neighbor"].consumes
    )


def test_a_track_feature_below_a_conversion_in_place_is_not_refused() -> None:
    """``convert`` returns a track-shaped table, which ``speed-angvel`` reads."""
    report = validation_report(
        {
            "steps": [
                {
                    "id": "cm",
                    "type": "feature",
                    "feature": "scale-to-cm",
                    "params": {"mode": "convert"},
                },
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": [{"step": "cm"}],
                },
            ]
        }
    )

    assert report.valid, report.problems
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
        "'time'",
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

    tsne = entries["global-tsne"]
    assert tsne.requires is not None
    assert [need.payload for need in tsne.requires] == [True]
    assert tsne.provides is not None
    assert [names.label for names in tsne.provides.adds] == ["'tsne_x'", "'tsne_y'"]

    declarations = declaration_catalog()
    for name, entry in entries.items():
        declared = declarations.get(name)
        assert declared is not None, name
        assert (entry.requires is None) == (declared.requires is None), name
        assert (entry.provides is None) == (declared.provides is None), name


def test_the_cli_prints_cautions_beside_a_valid_verdict() -> None:
    result = runner.invoke(
        app, ["pipeline", "validate", "--recipe", f"@{golden_path(GUPPIES_RECIPE)}"]
    )

    assert result.exit_code == 0, result.output
    assert "This recipe is valid." in result.stdout
    assert result.stderr.count("caution:") == 1
    assert all(f"'{step}'" in result.stderr for step in DELTA_STEPS[1:])


def test_the_cli_json_keeps_one_caution_per_step() -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "validate",
            "--json",
            "--recipe",
            f"@{golden_path(GUPPIES_RECIPE)}",
        ],
    )

    assert result.exit_code == 0, result.output
    steps = [caution["step"] for caution in json.loads(result.stdout)["cautions"]]
    assert steps == list(DELTA_STEPS)


def test_stacked_inputs_guarantee_only_what_each_input_guarantees() -> None:
    """A stack renames nothing: a column is sure when every input has it."""
    sure = frozenset({"frame", "predicted_label"})
    first = ColumnSet(guaranteed=sure, possible=sure | {"extra"})
    second = ColumnSet(guaranteed=sure | {"other"}, possible=sure | {"other"})

    stacked = merged_input([first, second], stack=True)
    joined = merged_input([first, second])

    assert stacked.guaranteed == {"frame", "predicted_label"}
    assert stacked.possible == {"frame", "predicted_label", "extra", "other"}
    assert "predicted_label__1" in joined.guaranteed
    assert "predicted_label__1" not in stacked.possible


def test_one_stacked_input_merges_as_itself() -> None:
    only = ColumnSet(guaranteed=frozenset({"frame", "speed"}))
    assert merged_input([only], stack=True) == merged_input([only])


def test_a_widening_merge_names_each_individual_column_twice() -> None:
    """Validation predicts the ``__id1`` and ``__id2`` columns that the merge writes."""
    pairs = ColumnSet(guaranteed=frozenset({"frame", "id1", "id2", "dist"}))
    individuals = ColumnSet(
        guaranteed=frozenset({"frame", "id", "speed"}),
        possible=frozenset({"frame", "id", "speed", "fps"}),
    )

    widened = merged_input([pairs, individuals], widen=True)

    assert widened.guaranteed == {
        "frame",
        "id1",
        "id2",
        "dist",
        "speed__id1",
        "speed__id2",
    }
    assert "fps" not in widened.possible
    reversed_order = merged_input([individuals, pairs], widen=True)
    assert "speed__id1" not in reversed_order.possible


def _over_tracks(step_id: str, feature: str) -> dict[str, JsonValue]:
    return {
        "id": step_id,
        "type": "feature",
        "feature": feature,
        "inputs": [TRACKS_INPUT],
        "params": {},
    }


def _stack_of(first: str, second: str) -> Recipe:
    """A ``temporal-stack`` over a pair-level and an individual-level step."""
    steps: list[JsonValue] = [
        _over_tracks("pairs", "pair-position"),
        _over_tracks("speed", "speed-angvel"),
        {
            "id": "stack",
            "type": "feature",
            "feature": "temporal-stack",
            "inputs": [{"step": first}, {"step": second}],
            "params": {},
        },
    ]
    return read_recipe({"steps": steps})


def test_a_recipe_widens_an_individual_input_after_a_pair_input() -> None:
    assert check_recipe(_stack_of("pairs", "speed")) == ()


def test_a_recipe_with_the_individual_input_first_is_refused() -> None:
    problems = check_recipe(_stack_of("speed", "pairs"))

    assert [problem.step for problem in problems] == ["stack"]
    assert "pair-level input first" in problems[0].message


# --- one caution for one finding -------------------------------------------------


def _pose_readers(*, wavelet_columns: list[str] | None = None) -> dict[str, JsonValue]:
    """Converted CalMS21 tracks read by two keypoint features, and a wavelet on one."""
    steps: list[JsonValue] = [
        {
            "id": "tracks",
            "type": "op",
            "kind": "convert-tracks",
            "params": {"source_format": "calms21_npy"},
        },
        {
            "id": "kin",
            "type": "feature",
            "feature": "keypoint-kinematics",
            "inputs": [TRACKS_INPUT],
            "tracks": {"step": "tracks"},
        },
        {
            "id": "pk",
            "type": "feature",
            "feature": "pair-keypoints",
            "inputs": [TRACKS_INPUT],
            "tracks": {"step": "tracks"},
            "params": {"heading": [0, 1]},
        },
    ]
    if wavelet_columns is not None:
        steps.append(
            {
                "id": "wav",
                "type": "feature",
                "feature": "individual-wavelet",
                "inputs": [{"step": "kin"}],
                "params": {"cols": list(wavelet_columns)},
            }
        )
    return {"schema_version": 1, "steps": steps}


def test_each_reader_of_converted_tracks_keeps_its_pose_caution_until_merged() -> None:
    """The tracks may hold no keypoints, so the pose columns stay a caution.

    ``mosaic_v1`` makes keypoints optional ("a centroid-only tracker emits
    none"), and ``trex_v1`` alone requires ``poseX``, so a recipe checked with
    no dataset has no guarantee for converted tracks. A plan reads the tables
    and does. The report keeps one caution per step, which an editor marks at
    that step, and a listing merges them into one.
    """
    report = validation_report(_pose_readers())

    assert report.valid, report.problems
    assert [(c.step, c.where) for c in report.cautions] == [
        ("kin", "inputs"),
        ("pk", "inputs"),
    ]
    assert all(
        "'poseX*', 'poseY*' may be missing" in c.message for c in report.cautions
    )
    (merged,) = merge_cautions(report.cautions)
    assert (merged.step, merged.where) == ("kin", "inputs")
    assert merged.message.endswith("; the same holds for steps 'pk'")


def test_a_caution_for_one_of_several_columns_says_any_and_names_a_few() -> None:
    """The wavelet transforms the ``cols`` present and needs only one of them."""
    columns = [f"{kind}__kp{k}" for kind in ("speed", "vx", "vy") for k in range(7)]

    report = validation_report(_pose_readers(wavelet_columns=columns))

    (caution,) = [c for c in report.cautions if c.step == "wav"]
    assert " or " not in caution.message
    assert caution.message.startswith(
        "any of 'speed__kp0', 'speed__kp1', 'speed__kp2' and 18 more"
    )
    assert "speed__kp6" not in caution.message


def test_a_requirement_of_few_columns_is_still_named_whole() -> None:
    assert NamedColumns(columns=("a", "b", "c")).phrase == "'a' or 'b' or 'c'"
    assert NamedColumns(columns=("a", "b", "c", "d")).phrase == (
        "any of 'a', 'b', 'c' and 1 more"
    )
    assert NamedColumns(columns=("a", "b", "c", "d")).label == (
        "'a' or 'b' or 'c' or 'd'"
    )


def test_cautions_with_one_text_merge_into_one_naming_every_step() -> None:
    cautions = (
        Problem("a", "inputs", "'x' may be missing"),
        Problem("b", "inputs", "'y' may be missing"),
        Problem("c", "inputs", "'x' may be missing"),
        Problem("d", "inputs", "'x' may be missing"),
        Problem("d", "params", "'x' may be missing"),
    )

    assert merge_cautions(cautions) == (
        Problem("a", "inputs", "'x' may be missing; the same holds for steps 'c', 'd'"),
        Problem("b", "inputs", "'y' may be missing"),
        Problem("d", "params", "'x' may be missing"),
    )


# --- symmetric pair columns -----------------------------------------------------------

_PAIR_KEYS = frozenset({"frame", "id1", "id2", "perspective"})

_SYMMETRIC_PAIRS = Provides(
    adds=(
        *(Column(name) for name in sorted(_PAIR_KEYS)),
        Column("dist"),
        Column("heading"),
    ),
    symmetric=(Column("dist"),),
)
"""A pair feature whose ``dist`` holds one value on both rows of a pair."""


def _pairs(*, symmetric: frozenset[str] = frozenset(), mirrored: bool) -> ColumnSet:
    """Pair rows with a ``dist`` column, declared symmetric or not."""
    columns = _PAIR_KEYS | {"dist"}
    return ColumnSet(
        guaranteed=columns, possible=columns, symmetric=symmetric, mirrored=mirrored
    )


def _pair_recipe(*steps: dict[str, JsonValue]) -> Recipe:
    """``pair-position`` over the tracks, then *steps*."""
    return read_recipe({"steps": [_over_tracks("pairs", "pair-position"), *steps]})


def test_a_declared_symmetric_column_propagates_through_passthrough() -> None:
    pairs = output_columns(_SYMMETRIC_PAIRS, ColumnSet(), {})
    through = output_columns(
        Provides(passthrough=True, adds=(Column("speed"),)), pairs, {}
    )
    rewritten = output_columns(
        Provides(passthrough=True, adds=(Column("dist"),)), pairs, {}
    )

    assert pairs.is_symmetric("dist")
    assert not pairs.is_symmetric("heading")
    assert pairs.mirrored
    assert through.is_symmetric("dist")
    assert through.mirrored
    assert not rewritten.is_symmetric("dist")
    assert resolved_columns(_pair_recipe())["pairs"].is_symmetric("AB_dist")


def test_a_payload_family_of_a_symmetric_column_is_symmetric() -> None:
    stack: dict[str, JsonValue] = {
        "id": "stack",
        "type": "feature",
        "feature": "temporal-stack",
        "inputs": [{"step": "pairs"}],
        "params": {},
    }

    columns = resolved_columns(_pair_recipe(stack))["stack"]

    assert columns.is_symmetric("AB_dist__pool_mean")
    assert not columns.is_symmetric("A_speed__pool_mean")
    assert columns.mirrored


def test_a_widened_column_is_not_symmetric() -> None:
    individuals = ColumnSet(
        guaranteed=frozenset({"frame", "id", "speed"}),
        symmetric=frozenset({"speed"}),
    )

    widened = merged_input(
        [_pairs(symmetric=frozenset({"dist"}), mirrored=True), individuals],
        widen=True,
    )

    assert widened.is_symmetric("dist")
    assert not widened.is_symmetric("speed__id1")
    assert not widened.is_symmetric("speed__id2")
    assert widened.mirrored


def test_a_renamed_column_keeps_its_symmetry() -> None:
    directional = _pairs(mirrored=True)
    symmetric = _pairs(symmetric=frozenset({"dist"}), mirrored=True)

    later = merged_input([directional, symmetric])
    first = merged_input([symmetric, directional])

    assert later.is_symmetric("dist__1")
    assert not later.is_symmetric("dist")
    assert first.is_symmetric("dist")
    assert not first.is_symmetric("dist__1")


def test_an_undeclared_input_has_no_symmetric_column() -> None:
    pairs = output_columns(_SYMMETRIC_PAIRS, ColumnSet(), {})

    undeclared = output_columns(None, pairs, {})
    merged = merged_input([pairs, UNKNOWN_COLUMNS])

    assert not undeclared.is_symmetric("dist")
    assert not undeclared.mirrored
    assert merged.is_symmetric("dist")
    assert not merged.mirrored


def test_a_merge_is_mirrored_only_when_every_pair_input_is() -> None:
    mirrored = _pairs(mirrored=True)
    unmirrored = _pairs(mirrored=False)
    individuals = ColumnSet(guaranteed=frozenset({"frame", "id", "speed"}))

    assert merged_input([mirrored, mirrored]).mirrored
    assert not merged_input([mirrored, unmirrored]).mirrored
    assert not merged_input([unmirrored, mirrored]).mirrored
    assert merged_input([mirrored, individuals]).mirrored
    assert merged_input([mirrored, mirrored], stack=True).mirrored
    assert not merged_input([mirrored, unmirrored], stack=True).mirrored


def test_a_family_writer_keeps_its_input_mirrored() -> None:
    assert output_columns(_FAMILIES, _pairs(mirrored=True), {}).mirrored
    assert not output_columns(_FAMILIES, _pairs(mirrored=False), {}).mirrored
    assert not output_columns(
        Provides(adds=(Column("frame"),)), _pairs(mirrored=True), {}
    ).mirrored


def test_a_family_of_a_symmetric_column_names_only_its_own_members() -> None:
    """``distance`` is symmetric, and ``distance_from_centroid`` is another column.

    A family is the column itself and the names under ``<column>__``, so a
    widened individual measurement whose name begins with ``distance`` is no
    member of it.
    """
    steps: list[JsonValue] = [
        _over_tracks("facing", "pair-facing"),
        {
            "id": "first_stack",
            "type": "feature",
            "feature": "temporal-stack",
            "inputs": [{"step": "facing"}],
            "params": {},
        },
        _over_tracks("speed", "speed-angvel"),
        {
            "id": "groups",
            "type": "feature",
            "feature": "ffgroups-metrics",
            "inputs": [TRACKS_INPUT, {"step": "speed"}],
            "params": {},
        },
        {
            "id": "second_stack",
            "type": "feature",
            "feature": "temporal-stack",
            "inputs": [{"step": "first_stack"}, {"step": "groups"}],
            "params": {},
        },
        {
            "id": "total",
            "type": "feature",
            "feature": "frame-aggregate",
            "inputs": [{"step": "second_stack"}],
            "params": {
                "columns": ["distance_from_centroid__id1__pool_mean"],
                "aggs": ["sum"],
            },
        },
    ]
    recipe = read_recipe({"steps": steps})

    stacked = resolved_columns(recipe)["second_stack"]

    assert stacked.is_symmetric("distance__pool_mean")
    assert not stacked.is_symmetric("distance_from_centroid__id1__pool_mean")
    assert pair_rules(recipe) == {"total": "each_perspective"}


# --- a time filter reads the time column -----------------------------------------


def _filtered_aggregate(run: dict[str, JsonValue]) -> Recipe:
    """``sequence-aggregate`` over ``social-motion-summary``, whose rows hold no time."""
    return Recipe.model_validate(
        {
            "steps": [
                {"id": "speed", "type": "feature", "feature": "speed-angvel"},
                {
                    "id": "summary",
                    "type": "feature",
                    "feature": "social-motion-summary",
                    "inputs": [{"step": "speed"}],
                },
                {
                    "id": "late",
                    "type": "feature",
                    "feature": "sequence-aggregate",
                    "inputs": [{"step": "summary"}],
                    "params": {"columns": ["speed_cv"]},
                    "run": run,
                },
            ]
        }
    )


@pytest.mark.parametrize("boundary", ["filter_start_time", "filter_end_time"])
def test_a_time_filter_over_an_input_without_time_is_a_problem_at_the_filter(
    boundary: str,
) -> None:
    problems = check_recipe(_filtered_aggregate({boundary: 1.0}))

    (problem,) = problems
    assert (problem.step, problem.where) == ("late", f"run.{boundary}")
    assert "'time'" in problem.message
    assert "filter_start_frame" in problem.message


def test_a_time_filter_over_the_tracks_is_not_a_problem() -> None:
    recipe = Recipe.model_validate(
        {
            "steps": [
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "run": {"filter_start_time": 1.0, "filter_end_time": 5.0},
                }
            ]
        }
    )

    assert check_recipe(recipe) == ()


def test_a_frame_filter_over_an_input_without_time_is_not_a_problem() -> None:
    assert check_recipe(_filtered_aggregate({"filter_start_frame": 3})) == ()
