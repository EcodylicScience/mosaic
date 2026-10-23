"""A fit may not include a held-out group, and a new group's refits are named.

A dataset's ``splits`` give a group the role ``train``, ``test`` or ``sealed``. The
planner refuses a step that fits on the entries it covers (``scope_dependent`` and
``fits``, read off the built feature) when those entries hold a ``test`` or
``sealed`` group, and ``refit_set`` says which steps a joining group would refit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    BoundRef,
    Request,
    Plan,
    Recipe,
    RecipeRefused,
    StepFacts,
    assume_added_cautions,
    plan_pipeline,
    plan_wire,
    run_pipeline,
    refit_set,
    step_facts,
    submit_request,
)
from mosaic.core.scope import Scope
from tests.helpers import CONVERT_LABELS_STEP, CONVERT_TRACKS_STEP, calms21_dataset

TRAIN = Scope(groups=["train"])
TEST = Scope(groups=["test"])
BOTH = Scope(groups=["train", "test"])

SPEED: dict[str, JsonValue] = {
    "id": "speed",
    "type": "feature",
    "feature": "speed-angvel",
    "inputs": ["tracks"],
}
TEMPLATES: dict[str, JsonValue] = {
    "id": "templates",
    "type": "feature",
    "feature": "extract-templates",
    "inputs": [{"step": "speed"}],
    "params": {"n_templates": 4},
}
KMEANS: dict[str, JsonValue] = {
    "id": "kmeans",
    "type": "feature",
    "feature": "global-kmeans",
    "inputs": [{"step": "speed"}],
    "params": {"templates": {"step": "templates"}, "k": 2},
}
PCA_FIT: dict[str, JsonValue] = {
    "id": "fit",
    "type": "feature",
    "feature": "pair-posedistance-pca",
    "inputs": ["tracks"],
    "params": {"n_components": 2, "pose": {"pose_n": 3}},
}
PCA_APPLY: dict[str, JsonValue] = {
    "id": "apply",
    "type": "feature",
    "feature": "pair-posedistance-pca",
    "inputs": ["tracks"],
    "params": {"n_components": 2, "pose": {"pose_n": 3}, "model": {"step": "fit"}},
}


def _recipe(*steps: dict[str, JsonValue]) -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": "held-out", "steps": list(steps)}
    )


@pytest.fixture
def split(tmp_path: Path) -> Dataset:
    """Converted tracks of a train group of two sequences and a test group of one."""
    ds = calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 40, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=5,
    )
    _ = ds.convert_all_tracks()
    return ds


@pytest.fixture
def roled(split: Dataset) -> Dataset:
    """The same dataset, with its groups given the roles their names say."""
    split.set_splits("train", ["train"])
    split.set_splits("test", ["test"])
    return split


def _problems(plan: Plan) -> list[tuple[str, str]]:
    return [(problem.step, problem.where) for problem in plan.problems]


# --- the guard ----------------------------------------------------------------


def test_a_fitting_step_over_a_test_group_is_a_problem(roled: Dataset) -> None:
    plan = plan_pipeline(roled, _recipe(SPEED, TEMPLATES), refuse=False)

    assert _problems(plan) == [("templates", "scope")]
    message = plan.problems[0].message
    assert "'test' (test)" in message
    assert "'train'" in message
    assert "step scope" in message
    assert '{"step": "templates"}' in message


def test_a_step_scope_that_includes_the_test_group_points_at_itself(
    roled: Dataset,
) -> None:
    plan = plan_pipeline(
        roled,
        _recipe(SPEED, TEMPLATES),
        step_scopes={"templates": BOTH},
        refuse=False,
    )

    assert _problems(plan) == [("templates", "step_scopes.templates")]


def test_a_fitting_step_scoped_to_the_train_groups_passes(roled: Dataset) -> None:
    plan = plan_pipeline(
        roled, _recipe(SPEED, TEMPLATES, KMEANS), step_scopes={"templates": TRAIN}
    )

    assert plan.problems == ()
    assert plan.step("kmeans").coverage.target == plan.scope


def test_a_sealed_group_is_held_out_as_a_test_group_is(roled: Dataset) -> None:
    roled.set_splits("sealed", ["test"])

    plan = plan_pipeline(roled, _recipe(SPEED, TEMPLATES), refuse=False)

    assert "'test' (sealed)" in plan.problems[0].message


def test_the_plan_refuses_and_a_submission_refuses_with_it(roled: Dataset) -> None:
    recipe = _recipe(SPEED, TEMPLATES)

    with pytest.raises(RecipeRefused) as refused:
        _ = plan_pipeline(roled, recipe)
    with pytest.raises(RecipeRefused):
        _ = submit_request(roled, recipe)

    assert _problems(refused.value.plan) == [("templates", "scope")]


def test_a_scope_free_fitter_is_not_flagged_where_its_templates_are_not(
    roled: Dataset,
) -> None:
    plan = plan_pipeline(
        roled,
        _recipe(SPEED, TEMPLATES, KMEANS),
        step_scopes={"templates": TRAIN},
        refuse=False,
    )

    assert plan.step("kmeans").fits
    assert not plan.step("kmeans").scope_dependent
    assert plan.problems == ()


def test_a_dataset_without_splits_plans_as_it_did(split: Dataset) -> None:
    plan = plan_pipeline(split, _recipe(SPEED, TEMPLATES))

    assert plan.problems == ()
    assert plan.splits == {}
    assert [caution.message for caution in plan_wire(plan).cautions] == [
        "the plan covers all 3 entries of the dataset"
    ]


def test_a_group_with_no_role_is_not_held_out(split: Dataset) -> None:
    split.set_splits("train", ["train"])

    assert plan_pipeline(split, _recipe(SPEED, TEMPLATES)).problems == ()


def test_applying_a_fitted_basis_to_the_test_group_passes(roled: Dataset) -> None:
    recipe = _recipe(PCA_FIT, PCA_APPLY)

    plan = plan_pipeline(roled, recipe, step_scopes={"fit": TRAIN}, refuse=False)

    assert plan.problems == ()
    assert plan.step("fit").fits and plan.step("fit").scope_dependent
    assert not plan.step("apply").fits
    assert not plan.step("apply").scope_dependent
    assert plan.step("apply").coverage.target == plan.scope


def test_the_unscoped_basis_fit_is_a_problem_beside_its_apply(roled: Dataset) -> None:
    plan = plan_pipeline(roled, _recipe(PCA_FIT, PCA_APPLY), refuse=False)

    assert _problems(plan) == [("fit", "scope")]


ARHMM_FIT: dict[str, JsonValue] = {
    "id": "fit",
    "type": "feature",
    "feature": "arhmm",
    "inputs": [{"step": "speed"}],
    "params": {"n_states": 2, "n_iter": 2},
}
ARHMM_APPLY: dict[str, JsonValue] = {
    "id": "apply",
    "type": "feature",
    "feature": "arhmm",
    "inputs": [{"step": "speed"}],
    "params": {"n_states": 2, "n_iter": 2, "model": {"step": "fit"}},
}


def test_applying_a_fitted_arhmm_to_the_test_group_passes(roled: Dataset) -> None:
    recipe = _recipe(SPEED, ARHMM_FIT, ARHMM_APPLY)

    plan = plan_pipeline(roled, recipe, step_scopes={"fit": TRAIN}, refuse=False)
    unscoped = plan_pipeline(roled, recipe, refuse=False)

    assert plan.problems == ()
    assert plan.step("fit").fits
    assert not plan.step("apply").fits
    # It applies the fit, so the entries it covers are not part of its identity.
    assert not plan.step("apply").scope_dependent
    assert _problems(unscoped) == [("fit", "scope")]


def test_a_fit_bound_to_a_run_is_not_a_fit_over_the_test_group(
    roled: Dataset,
) -> None:
    recipe = _recipe(SPEED, TEMPLATES)
    _ = run_pipeline(roled, recipe, step_scopes={"templates": TRAIN})
    done = plan_pipeline(roled, recipe, step_scopes={"templates": TRAIN})
    templates = done.step("templates")
    assert templates.run_id is not None
    request = Request(
        request_id="r",
        recipe_digest="d",
        bind={
            "templates": BoundRef(
                feature=templates.storage_name, run_id=templates.run_id
            )
        },
    )

    bound = plan_pipeline(roled, recipe, request=request, refuse=False)
    unbound = plan_pipeline(roled, recipe, refuse=False)

    assert ("templates", "scope") not in _problems(bound)
    assert _problems(unbound) == [("templates", "scope")]


def test_a_graph_that_converts_first_cautions_the_fit_it_cannot_yet_check(
    tmp_path: Path,
) -> None:
    ds = calms21_dataset(
        tmp_path / "raw",
        {"train": {"a": {"s1": 40, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=5,
    )
    ds.set_splits("train", ["train"])
    ds.set_splits("test", ["test"])
    recipe = _recipe(
        CONVERT_TRACKS_STEP, {**SPEED, "tracks": {"step": "tracks"}}, TEMPLATES
    )

    plan = plan_pipeline(ds, recipe, refuse=False)
    cautions = plan_wire(plan).cautions

    assert plan.deferred_by == "tracks"
    assert plan.problems == ()
    assert cautions[0].message == (
        "the dataset holds no tracks yet, so the plan covers every entry that step "
        "'tracks' converts. The manifest gives the groups 'test' (test), "
        "'train' (train) a role"
    )
    assert [(c.step, c.where) for c in cautions[1:]] == [("templates", "scope")]
    assert "'test' (test)" in cautions[1].message


def test_a_scoped_fit_over_a_graph_that_converts_first_has_no_such_caution(
    tmp_path: Path,
) -> None:
    ds = calms21_dataset(
        tmp_path / "raw",
        {"train": {"a": {"s1": 40, "s2": 30}}, "test": {"a": {"s3": 30}}},
        seed=5,
    )
    ds.set_splits("test", ["test"])
    recipe = _recipe(
        CONVERT_TRACKS_STEP, {**SPEED, "tracks": {"step": "tracks"}}, TEMPLATES
    )

    plan = plan_pipeline(ds, recipe, step_scopes={"templates": TRAIN}, refuse=False)

    assert [c.step for c in plan_wire(plan).cautions[1:]] == []


def _labelled(base: Path) -> Dataset:
    ds = calms21_dataset(
        base,
        {
            "train": {"a": {f"s{i}": 240 for i in range(1, 4)}},
            "test": {"a": {"t1": 240}},
        },
        seed=11,
        n_classes=3,
    )
    _ = ds.convert_all_tracks()
    ds.set_splits("train", ["train"])
    ds.set_splits("test", ["test"])
    return ds


def _classifier_recipe() -> Recipe:
    return _recipe(
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
            "id": "labeled",
            "type": "feature",
            "feature": "extract-labeled-templates",
            "inputs": [{"step": "pairs"}],
            "params": {"labels": {"step": "labels"}, "n_total": 300},
        },
        {
            "id": "classifier",
            "type": "feature",
            "feature": "xgboost",
            "inputs": [{"step": "pairs"}],
            "params": {"templates": {"step": "labeled"}, "default_class": 0},
        },
        {
            "id": "evaluate",
            "type": "feature",
            "feature": "evaluate-labels",
            "inputs": [{"step": "classifier"}],
            "params": {
                "labels": {"step": "labels"},
                "perspective": 0,
                "average_over": [0, 1, 2],
            },
        },
    )


def test_scoring_the_test_group_passes_and_the_labeled_templates_are_the_fit(
    tmp_path: Path,
) -> None:
    ds = _labelled(tmp_path / "ds")
    recipe = _classifier_recipe()
    final = {"labeled": TRAIN, "classifier": BOTH, "evaluate": TEST}

    passing = plan_pipeline(ds, recipe, step_scopes=final)
    leaking = plan_pipeline(
        ds, recipe, step_scopes={"classifier": BOTH, "evaluate": TEST}, refuse=False
    )

    assert passing.problems == ()
    assert not passing.step("evaluate").fits
    assert passing.step("evaluate").scope_dependent
    assert passing.step("classifier").fits
    assert not passing.step("classifier").scope_dependent
    assert _problems(leaking) == [("labeled", "scope")]


# --- the whole-dataset caution -------------------------------------------------


def test_the_whole_dataset_caution_names_the_groups_and_their_roles(
    roled: Dataset,
) -> None:
    plan = plan_pipeline(roled, _recipe(SPEED), refuse=False)

    caution = plan_wire(plan).cautions[0]

    assert (caution.step, caution.where) == ("", "scope")
    assert caution.message == (
        "the plan covers all 3 entries of the dataset, in the groups "
        "'test' (test), 'train' (train)"
    )


def test_the_caution_counts_groups_that_have_no_role(split: Dataset) -> None:
    split.set_splits("test", ["test"])

    plan = plan_pipeline(split, _recipe(SPEED))

    assert (
        plan_wire(plan)
        .cautions[0]
        .message.endswith("in the groups 'test' (test) and 1 with no role")
    )


def test_a_narrowed_plan_has_no_whole_dataset_caution(roled: Dataset) -> None:
    plan = plan_pipeline(roled, _recipe(SPEED), scope=TRAIN)

    assert plan_wire(plan).cautions == ()


# --- the refit set --------------------------------------------------------------

CHAIN = _recipe(
    SPEED,
    TEMPLATES,
    KMEANS,
    {
        "id": "score",
        "type": "feature",
        "feature": "evaluate-labels",
        "inputs": [{"step": "kmeans"}],
    },
    {
        "id": "report",
        "type": "feature",
        "feature": "speed-angvel",
        "inputs": [{"step": "score"}],
    },
    {
        "id": "later",
        "type": "feature",
        "feature": "speed-angvel",
        "inputs": ["tracks"],
        "after": ["templates"],
    },
)
FACTS = {
    "speed": StepFacts("speed"),
    "templates": StepFacts("templates", scope_dependent=True, fits=True),
    "kmeans": StepFacts("kmeans", fits=True),
    "score": StepFacts("score", scope_dependent=True),
    "report": StepFacts("report"),
    "later": StepFacts("later"),
}


def _moves(
    groups: list[str],
    scope: Scope | None = None,
    facts: dict[str, StepFacts] = FACTS,
) -> dict[str, str]:
    return {
        moved.step_id: moved.action
        for moved in refit_set(CHAIN, facts, groups, scope=scope)
    }


def test_a_joining_group_refits_the_fit_and_what_hashes_it() -> None:
    moved = refit_set(CHAIN, FACTS, ["new"])

    assert [(step.step_id, step.action) for step in moved] == [
        ("templates", "refit"),
        ("kmeans", "refit"),
        ("score", "re-score"),
        ("report", "re-score"),
    ]
    assert moved[0].joins == ("new",)
    assert moved[0].via == ()
    assert moved[1].joins == ()
    assert moved[1].via == ("templates",)
    assert moved[3].via == ("score",)
    assert not any(step.uncertain for step in moved)


def test_a_step_scope_that_names_other_groups_keeps_the_fit_where_it_is() -> None:
    narrowed = dict(FACTS, templates=StepFacts("templates", True, True, TRAIN))

    assert _moves(["new"], facts=narrowed) == {
        "score": "re-score",
        "report": "re-score",
    }
    assert _moves(["train"], facts=narrowed)["templates"] == "refit"


def test_a_step_scope_that_names_the_group_takes_it() -> None:
    named = dict(
        FACTS, templates=StepFacts("templates", True, True, Scope(groups=["new"]))
    )

    assert _moves(["new"], facts=named)["templates"] == "refit"


def test_a_request_that_names_groups_not_including_it_moves_nothing() -> None:
    assert _moves(["new"], scope=TRAIN) == {}
    assert "templates" in _moves(["new"], scope=Scope(groups=["new"]))


def test_a_request_that_names_entries_moves_nothing() -> None:
    assert _moves(["new"], scope=Scope(entries=[("train", "s1")])) == {}


def test_a_sequences_selector_is_uncertain_and_errs_towards_moving() -> None:
    moved = refit_set(CHAIN, FACTS, ["new"], scope=Scope(sequences=["s1"]))

    assert [step.step_id for step in moved][:2] == ["templates", "kmeans"]
    assert all(step.uncertain for step in moved)
    groups_and_sequences = Scope(groups=["train"], sequences=["s1"])
    assert refit_set(CHAIN, FACTS, ["new"], scope=groups_and_sequences) == ()


def test_an_after_edge_does_not_carry_a_move() -> None:
    assert "later" not in _moves(["new"])


def test_a_step_missing_from_the_facts_moves_only_with_a_parent() -> None:
    assert _moves(["new"], facts={}) == {}
    assert _moves(["new"], facts={"templates": FACTS["templates"]}) == {
        "templates": "refit",
        "kmeans": "re-score",
        "score": "re-score",
        "report": "re-score",
    }


def test_no_assumed_group_moves_nothing() -> None:
    assert refit_set(CHAIN, FACTS, []) == ()


def test_the_facts_of_a_plan_read_its_flags_and_step_scopes(roled: Dataset) -> None:
    plan = plan_pipeline(
        roled, _recipe(SPEED, TEMPLATES, KMEANS), step_scopes={"templates": TRAIN}
    )

    facts = step_facts(plan)

    assert facts["templates"] == StepFacts("templates", True, True, TRAIN)
    assert facts["kmeans"] == StepFacts("kmeans", False, True, None)
    assert facts["speed"] == StepFacts("speed")


# --- assume_added -----------------------------------------------------------------


def test_assuming_a_group_added_lists_the_refits_and_the_rescores(
    roled: Dataset,
) -> None:
    recipe = _recipe(SPEED, TEMPLATES, KMEANS)
    plan = plan_pipeline(roled, recipe, refuse=False)

    cautions = assume_added_cautions(recipe, plan, ["new"])

    assert [(caution.step, caution.where) for caution in cautions] == [
        ("", "assume_added")
    ]
    assert cautions[0].message == (
        "if group 'new' joined, these steps would refit: templates, kmeans"
    )


def test_a_train_scoped_fit_is_unmoved_by_a_group_outside_its_scope(
    roled: Dataset,
) -> None:
    recipe = _recipe(SPEED, TEMPLATES, KMEANS)
    plan = plan_pipeline(roled, recipe, step_scopes={"templates": TRAIN})

    cautions = assume_added_cautions(recipe, plan, ["new"])

    assert cautions[0].message == "if group 'new' joined, no step's run would change"


def test_the_role_of_an_assumed_group_is_named_when_it_moves_nothing(
    roled: Dataset,
) -> None:
    roled.set_splits("sealed", ["vault"])
    recipe = _recipe(SPEED, TEMPLATES)
    plan = plan_pipeline(roled, recipe, step_scopes={"templates": TRAIN})
    wider = plan_pipeline(roled, recipe, scope=TRAIN)

    held_by_scope = assume_added_cautions(recipe, plan, ["vault"])
    held_by_request = assume_added_cautions(recipe, wider, ["vault"], scope=TRAIN)

    assert held_by_scope[0].message == (
        "if group 'vault' joined ('vault' is sealed), no step's run would change"
    )
    assert held_by_request[0].message == held_by_scope[0].message


def test_a_whole_dataset_request_marks_a_refit_that_would_take_a_sealed_group(
    roled: Dataset,
) -> None:
    roled.set_splits("sealed", ["vault"])
    recipe = _recipe(SPEED, TEMPLATES)
    plan = plan_pipeline(roled, recipe, refuse=False)

    cautions = assume_added_cautions(recipe, plan, ["vault"])

    assert cautions[0].message == (
        "if group 'vault' joined ('vault' is sealed), these steps would refit: "
        "templates (refused: its fit would include 'vault')"
    )


def test_a_group_already_in_the_plan_is_reported_and_left_out(roled: Dataset) -> None:
    recipe = _recipe(SPEED, TEMPLATES)
    plan = plan_pipeline(roled, recipe, step_scopes={"templates": TRAIN})

    cautions = assume_added_cautions(recipe, plan, ["test", "new"])

    assert [caution.message.split(",")[0] for caution in cautions] == [
        "group 'test' is already in this plan's scope",
        "if group 'new' joined",
    ]


def test_assuming_nothing_adds_no_caution(roled: Dataset) -> None:
    recipe = _recipe(SPEED)

    plan = plan_pipeline(roled, recipe)

    assert assume_added_cautions(recipe, plan, []) == ()


def test_the_cautions_follow_the_plans_own_in_the_wire(roled: Dataset) -> None:
    recipe = _recipe(SPEED, TEMPLATES)
    plan = plan_pipeline(roled, recipe, step_scopes={"templates": TRAIN})
    extra = assume_added_cautions(recipe, plan, ["new"])

    wire = plan_wire(plan, extra_cautions=extra)

    assert wire.cautions[-1] == extra[0]
    assert wire.cautions[0].where == "scope"
    assert plan_wire(plan).fingerprint == wire.fingerprint
