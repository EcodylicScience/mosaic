"""A block cross-validation gap must cover what a classifier's inputs read.

A feature may declare ``temporal_reach``, the frames on each side that one of its
output rows depends on. The planner sums the declarations along the recipe's data
edges above a classifier planned for block cross-validation and compares the longest
path with ``cv_gap_frames``: a problem when every step declares a reach, a caution
when some do not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import Plan, Recipe, plan_pipeline, plan_wire
from mosaic.core.pipeline.graph.reach import UNDECLARED, UNKNOWN, class_reach
from mosaic.core.pipeline.graph.resolve import (
    StepTypeDetail,
    feature_class_for_slug,
    step_type_detail,
)
from tests.helpers import CONVERT_LABELS_STEP, CONVERT_TRACKS_STEP, calms21_dataset

SPEED: dict[str, JsonValue] = {
    "id": "speed",
    "type": "feature",
    "feature": "speed-angvel",
    "inputs": ["tracks"],
    "tracks": {"step": "tracks"},
}
STACK: dict[str, JsonValue] = {
    "id": "stack",
    "type": "feature",
    "feature": "temporal-stack",
    "inputs": [{"step": "speed"}],
    "params": {"half": 10, "skip": 5, "sigma_stack": 0, "add_pool": False},
}
TEMPLATES = {
    "feature": "extract-labeled-templates",
    "pattern": "templates.parquet",
    "load": {"kind": "parquet"},
}
HEADING: dict[str, JsonValue] = {
    "id": "heading",
    "type": "feature",
    "feature": "heading",
    "inputs": ["tracks"],
    "tracks": {"step": "tracks"},
    "params": {"method": "two_point", "front_idx": 0, "rear_idx": 3},
}
POSITION: dict[str, JsonValue] = {
    "id": "position",
    "type": "feature",
    "feature": "pair-position",
    "inputs": ["tracks", {"step": "heading"}],
    "tracks": {"step": "tracks"},
}


def _recipe(*steps: dict[str, JsonValue]) -> Recipe:
    return Recipe.model_validate(
        {"schema_version": 1, "name": "reach", "steps": list(steps)}
    )


def _classifier(source: str, **params: JsonValue) -> list[dict[str, JsonValue]]:
    """The labeled templates drawn from *source* and an xgboost over it, in blocks."""
    return [
        {
            "id": "labeled",
            "type": "feature",
            "feature": "extract-labeled-templates",
            "inputs": [{"step": source}],
            "params": {"labels": {"step": "labels"}, "n_total": 300},
        },
        {
            "id": "classifier",
            "type": "feature",
            "feature": "xgboost",
            "inputs": [{"step": source}],
            "params": {
                "templates": {"step": "labeled"},
                "default_class": 0,
                "cv_folds": 3,
                "cv_unit": "block",
                **params,
            },
        },
    ]


@pytest.fixture
def raw_ds(tmp_path: Path) -> Dataset:
    """A dataset whose tracks are not converted yet: the first plan of a recipe."""
    return calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {f"s{i}": 240 for i in range(1, 4)}}},
        seed=11,
        n_classes=3,
    )


@pytest.fixture
def ds(raw_ds: Dataset) -> Dataset:
    _ = raw_ds.convert_all_tracks()
    return raw_ds


def _plan(ds: Dataset, *steps: dict[str, JsonValue]) -> Plan:
    return plan_pipeline(
        ds, _recipe(CONVERT_TRACKS_STEP, CONVERT_LABELS_STEP, *steps), refuse=False
    )


def _gap_problems(plan: Plan) -> list[str]:
    return [p.message for p in plan.problems if p.where == "params.cv_gap_frames"]


# --- the declarations -----------------------------------------------------------


def test_a_step_type_reports_its_reach_at_default_params() -> None:
    assert step_type_detail("pair-posedistance-pca").temporal_reach == 0
    assert step_type_detail("xgboost").temporal_reach == 0
    assert step_type_detail("speed-angvel").temporal_reach == 1
    # The default stack: half 60, sigma_stack 30 read int(4 * 30 + 0.5) = 120 more.
    assert step_type_detail("temporal-stack").temporal_reach == 180
    # Its default head is a GRU, which no option bounds.
    assert step_type_detail("sequence-classifier").temporal_reach == UNKNOWN


def test_a_feature_that_declares_nothing_is_not_reported_as_zero() -> None:
    assert step_type_detail("pair-egocentric").temporal_reach == UNDECLARED
    assert step_type_detail("tracks").temporal_reach == UNDECLARED
    assert step_type_detail("convert-tracks").temporal_reach == UNDECLARED


def test_a_detail_served_without_a_reach_reads_as_undeclared() -> None:
    served = json.loads(step_type_detail("speed-angvel").model_dump_json())
    del served["temporal_reach"]

    assert StepTypeDetail.model_validate(served).temporal_reach == UNDECLARED


def _reach(slug: str, **params: object) -> object:
    cls = feature_class_for_slug(slug)
    assert cls is not None
    reach: object = cls.temporal_reach(cls.Params.from_overrides(params))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    return reach  # pyright: ignore[reportUnknownVariableType]


def test_a_stack_reads_its_window_and_its_smoothing() -> None:
    # Stack: half plus the Gaussian radius; pooling: its own Gaussian radius.
    assert _reach("temporal-stack", half=10, sigma_stack=0, add_pool=False) == 10
    assert _reach("temporal-stack", half=10, sigma_stack=2, add_pool=False) == 18
    assert _reach("temporal-stack", half=0, sigma_stack=0, add_pool=False) == 0
    assert _reach("temporal-stack", use_temporal_stack=False, add_pool=False) == 0
    assert (
        _reach(
            "temporal-stack",
            half=5,
            sigma_stack=0,
            sigma_pool=10,
            fps=30,
            win_sec=2,
            pool_stats=["mean"],
        )
        == 30
    )


def test_a_trajectory_smooth_without_interpolation_reads_its_filter() -> None:
    off = {"interpolate_centroid": False, "interpolate_pose": False}
    assert _reach("trajectory-smooth", **off) == 0
    assert _reach("trajectory-smooth", savgol_window=11, **off) == 5
    assert _reach("trajectory-smooth", speed_threshold=3.0, expand_frames=2, **off) == 3


def test_a_trajectory_smooth_that_interpolates_reads_an_unbounded_span() -> None:
    assert _reach("trajectory-smooth") is None
    assert (
        _reach("trajectory-smooth", interpolate_centroid=False, interpolate_pose=True)
        is None
    )


def test_a_speed_reads_back_one_step_and_its_smoothing_window() -> None:
    assert _reach("speed-angvel") == 1
    assert _reach("speed-angvel", step_size=4) == 4
    assert _reach("speed-angvel", smooth_window=9) == 5
    assert _reach("speed-angvel", step_size=8, smooth_window=9) == 8


def test_a_wavelet_reads_half_its_longest_support() -> None:
    # The longest scale is fps / (f_min * central frequency); cmor1.5-1.0 spans 16.
    reach = _reach("pair-wavelet")
    assert isinstance(reach, int)
    assert reach == 1200
    assert _reach("pair-wavelet", f_min=0.4) == 600


def test_the_tcn_reach_is_its_receptive_field_less_the_frame() -> None:
    # blocks=6, kernel 3 reads 127 frames in all, 63 on each side.
    assert (
        _reach(
            "sequence-classifier",
            head="tcn",
            default_class=0,
            templates=TEMPLATES,
            labels={"kind": "behavior"},
        )
        == 63
    )
    assert (
        _reach(
            "sequence-classifier",
            head="tcn",
            head_config={"blocks": 5},
            default_class=0,
            templates=TEMPLATES,
            labels={"kind": "behavior"},
        )
        == 31
    )
    assert (
        _reach(
            "sequence-classifier",
            head="tcn",
            head_config={"blocks": 3, "kernel_size": 5},
            default_class=0,
            templates=TEMPLATES,
            labels={"kind": "behavior"},
        )
        == 14
    )
    assert (
        _reach(
            "sequence-classifier",
            head="mlp",
            default_class=0,
            templates=TEMPLATES,
            labels={"kind": "behavior"},
        )
        == 0
    )
    assert (
        _reach(
            "sequence-classifier",
            head="gru",
            default_class=0,
            templates=TEMPLATES,
            labels={"kind": "behavior"},
        )
        is None
    )


# --- the check ------------------------------------------------------------------


def test_a_gap_shorter_than_the_reach_above_the_classifier_is_a_problem(
    ds: Dataset,
) -> None:
    plan = _plan(ds, SPEED, STACK, *_classifier("stack", cv_gap_frames=10))

    (message,) = _gap_problems(plan)
    assert [p.step for p in plan.problems] == ["classifier"]
    # speed 1 + stack 10 = 11 frames on each side.
    assert "reads 11 frames" in message
    assert "'speed' 1, 'stack' 10" in message
    assert "cv_gap_frames to 11 or more" in message
    assert plan.reach_cautions == ()


def test_a_gap_that_covers_the_reach_passes(ds: Dataset) -> None:
    plan = _plan(ds, SPEED, STACK, *_classifier("stack", cv_gap_frames=11))

    assert plan.problems == ()
    assert plan.reach_cautions == ()


def test_the_sequence_unit_is_not_checked(ds: Dataset) -> None:
    plan = _plan(ds, SPEED, STACK, *_classifier("stack", cv_unit="sequence"))

    assert plan.problems == ()
    assert plan.reach_cautions == ()


def test_a_classifier_without_cross_validation_is_not_checked(ds: Dataset) -> None:
    plan = _plan(
        ds,
        SPEED,
        STACK,
        {
            "id": "labeled",
            "type": "feature",
            "feature": "extract-labeled-templates",
            "inputs": [{"step": "stack"}],
            "params": {"labels": {"step": "labels"}, "n_total": 300},
        },
        {
            "id": "classifier",
            "type": "feature",
            "feature": "xgboost",
            "inputs": [{"step": "stack"}],
            "params": {"templates": {"step": "labeled"}, "default_class": 0},
        },
    )

    assert plan.problems == ()
    assert plan.reach_cautions == ()


def test_the_longest_path_decides_and_a_shorter_branch_adds_nothing(
    ds: Dataset,
) -> None:
    # Two reads of the same classifier input: the long path through the stack
    # (1 + 10) and a second, shorter one through the speed alone (1).
    plan = _plan(
        ds,
        SPEED,
        STACK,
        {
            "id": "labeled",
            "type": "feature",
            "feature": "extract-labeled-templates",
            "inputs": [{"step": "speed"}],
            "params": {"labels": {"step": "labels"}, "n_total": 300},
        },
        {
            "id": "classifier",
            "type": "feature",
            "feature": "xgboost",
            "inputs": [{"step": "stack"}],
            "params": {
                "templates": {"step": "labeled"},
                "default_class": 0,
                "cv_folds": 3,
                "cv_unit": "block",
                "cv_gap_frames": 11,
            },
        },
    )

    assert plan.problems == ()
    assert plan.reach_cautions == ()


def test_a_step_that_declares_no_reach_cautions_instead(ds: Dataset) -> None:
    plan = _plan(ds, HEADING, POSITION, *_classifier("position", cv_gap_frames=0))

    assert plan.problems == ()
    (caution,) = plan.reach_cautions
    assert (caution.step, caution.where) == ("classifier", "params.cv_gap_frames")
    assert "'position' declare no temporal reach" in caution.message
    assert caution in plan_wire(plan).cautions


def test_a_gap_the_user_set_answers_a_step_that_declares_no_reach(ds: Dataset) -> None:
    # The caution says to set the gap to the longest window above the classifier,
    # so setting one has to clear it.
    plan = _plan(ds, HEADING, POSITION, *_classifier("position", cv_gap_frames=30))

    assert plan.problems == ()
    assert plan.reach_cautions == ()


def test_a_gap_below_the_known_reach_still_cautions_beside_an_unknown(
    ds: Dataset,
) -> None:
    # Position reads the stack, which declares 10 frames, so a gap of 5 is short
    # of what is known whatever position reads. speed 1 + stack 10 = 11.
    plan = _plan(
        ds,
        SPEED,
        STACK,
        {**POSITION, "inputs": ["tracks", {"step": "stack"}]},
        *_classifier("position", cv_gap_frames=5),
    )

    assert _gap_problems(plan) == []
    (caution,) = plan.reach_cautions
    assert "at least 11 frames" in caution.message


def test_a_tuned_threshold_step_reads_no_frames_around_its_row(ds: Dataset) -> None:
    assert step_type_detail("tune-threshold").temporal_reach == 0


def test_a_reach_no_params_bound_cautions_and_names_what_is_known(ds: Dataset) -> None:
    plan = _plan(
        ds,
        {
            "id": "smooth",
            "type": "feature",
            "feature": "trajectory-smooth",
            "inputs": ["tracks"],
            "tracks": {"step": "tracks"},
        },
        {**SPEED, "inputs": [{"step": "smooth"}]},
        STACK,
        *_classifier("stack", cv_gap_frames=0),
    )

    assert _gap_problems(plan) == []
    (caution,) = plan.reach_cautions
    assert "'smooth' read a span no params bound" in caution.message
    assert "at least 11 frames" in caution.message


def _sequence_classifier(
    source: str, **params: JsonValue
) -> list[dict[str, JsonValue]]:
    """The labeled templates drawn from *source* and a sequence-classifier over it."""
    return [
        {
            "id": "labeled",
            "type": "feature",
            "feature": "extract-labeled-templates",
            "inputs": [{"step": source}],
            "params": {"labels": {"step": "labels"}, "n_total": 300},
        },
        {
            "id": "classifier",
            "type": "feature",
            "feature": "sequence-classifier",
            "inputs": [{"step": source}],
            "params": {
                "templates": {"step": "labeled"},
                "labels": {"step": "labels"},
                "default_class": 0,
                "cv_folds": 3,
                "cv_unit": "block",
                **params,
            },
        },
    ]


def test_the_classifiers_own_reach_is_not_counted_against_its_gap(ds: Dataset) -> None:
    # A fold model trains on pieces cut out of the runs, so its head never reads a
    # held-out frame. Only the speed above it (1 frame) can leak one.
    tcn = _plan(
        ds,
        SPEED,
        *_sequence_classifier("speed", head="tcn", cv_gap_frames=1),
    )
    mlp = _plan(
        ds,
        SPEED,
        *_sequence_classifier("speed", head="mlp", cv_gap_frames=1),
    )

    assert _gap_problems(tcn) == []
    assert tcn.reach_cautions == ()
    assert _gap_problems(mlp) == []
    assert mlp.reach_cautions == ()


def test_a_recurrent_head_over_declared_steps_plans_clean_at_the_gap_above_it(
    ds: Dataset,
) -> None:
    plan = _plan(
        ds,
        SPEED,
        *_sequence_classifier("speed", head="gru", cv_gap_frames=1),
    )

    assert _gap_problems(plan) == []
    assert plan.reach_cautions == ()


def test_a_gap_short_of_the_steps_above_a_sequence_classifier_is_a_problem(
    ds: Dataset,
) -> None:
    plan = _plan(
        ds,
        SPEED,
        STACK,
        *_sequence_classifier("stack", head="tcn", cv_gap_frames=10),
    )

    (message,) = _gap_problems(plan)
    assert "reads 11 frames" in message
    assert "'classifier' " not in message


def test_a_gap_is_checked_before_the_tracks_are_converted(raw_ds: Dataset) -> None:
    # The first plan of a recipe: nothing above the classifier has run, so its
    # identity is unresolved, and the check must not wait for it.
    plan = _plan(raw_ds, SPEED, STACK, *_classifier("stack", cv_gap_frames=2))

    classifier = next(step for step in plan.steps if step.step_id == "classifier")
    assert classifier.run_id is None
    assert classifier.fits
    (message,) = _gap_problems(plan)
    # The stack's own window (10), not the default one, and speed's 1.
    assert "reads 11 frames" in message
    assert plan_wire(plan).problems == plan.problems


def test_a_gap_that_covers_the_reach_passes_before_the_tracks_are_converted(
    raw_ds: Dataset,
) -> None:
    plan = _plan(raw_ds, SPEED, STACK, *_classifier("stack", cv_gap_frames=11))

    assert plan.problems == ()
    assert plan.reach_cautions == ()


def test_an_undeclared_step_cautions_before_the_tracks_are_converted(
    raw_ds: Dataset,
) -> None:
    plan = _plan(raw_ds, HEADING, POSITION, *_classifier("position", cv_gap_frames=0))

    (caution,) = plan.reach_cautions
    assert "'position' declare no temporal reach" in caution.message


def test_a_classifier_given_a_model_before_the_tracks_exist_fits_nothing(
    raw_ds: Dataset,
) -> None:
    fitted = _sequence_classifier("speed", cv_folds=None, cv_unit="sequence")
    plan = _plan(
        raw_ds,
        SPEED,
        *fitted,
        {
            "id": "applied",
            "type": "feature",
            "feature": "sequence-classifier",
            "inputs": [{"step": "speed"}],
            "params": {"model": {"step": "classifier"}, "default_class": 0},
        },
    )

    by_id = {step.step_id: step for step in plan.steps}
    assert by_id["classifier"].fits
    assert not by_id["applied"].fits


def test_a_step_bound_to_a_run_is_not_checked(ds: Dataset) -> None:
    from mosaic.core.pipeline.graph import BoundRef, Request

    recipe = _recipe(
        CONVERT_TRACKS_STEP,
        CONVERT_LABELS_STEP,
        SPEED,
        STACK,
        *_classifier("stack", cv_gap_frames=0),
    )

    checked = plan_pipeline(ds, recipe, refuse=False)
    bound = plan_pipeline(
        ds,
        recipe,
        request=Request(
            request_id="r",
            recipe_digest="d",
            bind={"classifier": BoundRef(feature="xgboost", run_id="0.4-aaaaaaaaaa")},
        ),
        refuse=False,
    )

    assert _gap_problems(checked)
    assert _gap_problems(bound) == []


def test_the_catalog_reach_of_a_class_without_default_params_is_unknown() -> None:
    class Windowed:
        class Params:
            @classmethod
            def from_overrides(cls, params: object) -> object:
                raise ValueError("needs a window")

        @staticmethod
        def temporal_reach(params: object) -> int:
            return 3

    class Fixed:
        temporal_reach = 7

    class Silent:
        pass

    assert class_reach(Windowed) == UNKNOWN
    assert class_reach(Fixed) == 7
    assert class_reach(Silent) == UNDECLARED
