"""What a recipe is refused for, decided against the real registries.

The semantic half of recipe validation, which needs ``FEATURES`` and ``OPS`` and
so lives apart from ``test_pipeline_recipe.py``'s structural half. Every case
here is one a graph would otherwise discover at run time -- after the expensive
steps above it had already run -- or would not discover at all, the sharpest
being a multi-input join of mismatched entity granularity, which produces a
per-frame cartesian product and raises nothing.

No dataset is built anywhere in this file, and that is the property being
measured as much as any single refusal: a canvas asks these questions with
nothing selected.
"""

from __future__ import annotations

from typing import NoReturn

import pytest

from mosaic.core.pipeline.graph import (
    Problem,
    Recipe,
    RecipeInvalid,
    check_recipe,
    reject_unless_valid,
)
from mosaic.core.pipeline.graph import validate as validate_module

type Document = dict[str, object]


def _problems(document: Document) -> tuple[Problem, ...]:
    """Every fault in *document*, which must parse as a recipe first."""
    return check_recipe(Recipe.model_validate(document))


def _at(document: Document, where: str) -> Problem:
    """The one problem reported at *where*, or a failure naming what was found."""
    found = [problem for problem in _problems(document) if problem.where == where]
    assert len(found) == 1, (
        f"expected one problem at {where}, got {_problems(document)}"
    )
    return found[0]


# A chain that validates, for the cases that need a well-formed neighbour.
SPEED_STEP: Document = {
    "id": "speed",
    "type": "feature",
    "feature": "speed-angvel",
    "inputs": ["tracks"],
}

WORKED_EXAMPLE: Document = {
    "schema_version": 1,
    "name": "trex to global tsne",
    "steps": [
        {
            "id": "transcode",
            "type": "op",
            "kind": "transcode",
            "params": {"target": "analysis"},
        },
        {
            "id": "trex",
            "type": "op",
            "kind": "trex",
            "params": {"track_max_individuals": 4, "cm_per_pixel": 0.05},
            "after": ["transcode"],
        },
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
            "tracks": {"step": "trex"},
        },
        {
            "id": "templates",
            "type": "feature",
            "feature": "extract-templates",
            "inputs": [{"step": "speed"}],
            "params": {"n_templates": 8},
        },
        {
            "id": "tsne",
            "type": "feature",
            "feature": "global-tsne",
            "inputs": [{"step": "speed"}],
            "params": {
                "templates": {"step": "templates", "pattern": "templates.parquet"},
                "perplexity": 5,
            },
        },
    ],
}


def test_the_worked_example_validates() -> None:
    """The five-step graph the phase is written around must be admissible.

    Its transcode step names no entries, deliberately: an op step's scope comes
    from the plan, because a recipe naming ``(group, sequence)`` pairs is about
    one dataset and a recipe is meant to travel.
    """
    reject_unless_valid(Recipe.model_validate(WORKED_EXAMPLE))


def test_an_unknown_feature_slug_is_refused_by_name() -> None:
    problem = _at(
        {"steps": [{"id": "a", "type": "feature", "feature": "no-such-feature"}]},
        "feature",
    )
    assert "no-such-feature" in problem.message


def test_an_unknown_op_kind_is_refused_by_name() -> None:
    problem = _at({"steps": [{"id": "a", "type": "op", "kind": "no-such-op"}]}, "kind")
    assert "no-such-op" in problem.message


def test_params_are_validated_against_the_feature_s_own_model() -> None:
    """Not against a schema copy: the model is what the run will validate with."""
    problem = _at(
        {
            "steps": [
                {
                    "id": "a",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "params": {"not_a_field": 1},
                }
            ]
        },
        "params.not_a_field",
    )
    assert "not permitted" in problem.message


def test_a_cycle_is_reported_once_and_stops_the_walk() -> None:
    """Every other question is asked of a step's upstreams, and a cycle has none."""
    problems = _problems(
        {
            "steps": [
                {
                    "id": "a",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": [{"step": "b"}],
                },
                {
                    "id": "b",
                    "type": "feature",
                    "feature": "temporal-stack",
                    "inputs": [{"step": "a"}],
                },
            ]
        }
    )
    assert len(problems) == 1
    assert "cycle" in problems[0].message
    assert problems[0].step == ""


def test_a_reference_in_a_field_that_holds_a_plain_value_is_refused() -> None:
    """A reference substituted into a number is a type error two layers down."""
    problem = _at(
        {
            "steps": [
                SPEED_STEP,
                {
                    "id": "tsne",
                    "type": "feature",
                    "feature": "global-tsne",
                    "inputs": [{"step": "speed"}],
                    "params": {"perplexity": {"step": "speed"}},
                },
            ]
        },
        "params.perplexity",
    )
    assert "run reference" in problem.message


def test_a_reference_in_a_field_the_model_does_not_declare_is_refused() -> None:
    problem = _at(
        {
            "steps": [
                SPEED_STEP,
                {
                    "id": "nn",
                    "type": "feature",
                    "feature": "nearest-neighbor",
                    "inputs": ["tracks"],
                    "params": {"no_such_field": {"step": "speed"}},
                },
            ]
        },
        "params.no_such_field",
    )
    assert "no_such_field" in problem.message


def test_an_artifact_reference_that_would_resolve_by_glob_is_refused() -> None:
    """A wildcard names a run root's contents, not one artifact.

    The producer's run directory holds one per-entry output parquet per sequence
    beside its named artifacts, so a glob resolves whichever sorts first and the
    step fits on a per-entry table while reporting success. Refused here because
    it needs no dataset: the author hears it before the producing step has run.
    """
    problem = _at(
        {
            "steps": [
                SPEED_STEP,
                {
                    "id": "templates",
                    "type": "feature",
                    "feature": "extract-templates",
                    "inputs": [{"step": "speed"}],
                    "params": {"n_templates": 8},
                },
                {
                    "id": "tsne",
                    "type": "feature",
                    "feature": "global-tsne",
                    "inputs": [{"step": "speed"}],
                    "params": {
                        "templates": {"step": "templates", "pattern": "*.parquet"}
                    },
                },
            ]
        },
        "params.templates",
    )
    assert "*.parquet" in problem.message
    assert "sorts first" in problem.message


def test_an_artifact_reference_with_no_pattern_is_admissible() -> None:
    """The declared type names the file, so the recipe does not have to.

    This is the shape that used to validate, plan, run, and read a per-entry
    output. It stays admissible -- what changed is where it resolves to.
    """
    reject_unless_valid(
        Recipe.model_validate(
            {
                "steps": [
                    SPEED_STEP,
                    {
                        "id": "templates",
                        "type": "feature",
                        "feature": "extract-templates",
                        "inputs": [{"step": "speed"}],
                        "params": {"n_templates": 8},
                    },
                    {
                        "id": "tsne",
                        "type": "feature",
                        "feature": "global-tsne",
                        "inputs": [{"step": "speed"}],
                        "params": {"templates": {"step": "templates"}},
                    },
                ]
            }
        )
    )


def test_an_artifact_reference_to_a_step_of_another_feature_is_refused() -> None:
    """A typed artifact names the feature that writes it.

    Several features write a ``model.joblib``, so a reference to the wrong step
    resolves to a file that loads and is read as the wrong model. The author
    hears it before any step has run.
    """
    problem = _at(
        {
            "steps": [
                SPEED_STEP,
                {
                    "id": "kmeans",
                    "type": "feature",
                    "feature": "global-kmeans",
                    "inputs": [{"step": "speed"}],
                    "params": {"model": {"step": "speed", "pattern": "model.joblib"}},
                },
            ]
        },
        "params.model",
    )
    assert problem.step == "kmeans"
    assert "global-kmeans" in problem.message
    assert "speed-angvel" in problem.message


def test_an_artifact_reference_to_a_step_of_its_own_feature_is_admissible() -> None:
    reject_unless_valid(
        Recipe.model_validate(
            {
                "steps": [
                    SPEED_STEP,
                    {
                        "id": "templates",
                        "type": "feature",
                        "feature": "extract-templates",
                        "inputs": [{"step": "speed"}],
                        "params": {"n_templates": 8},
                    },
                    {
                        "id": "fit",
                        "type": "feature",
                        "feature": "global-kmeans",
                        "inputs": [{"step": "speed"}],
                        "params": {"templates": {"step": "templates"}, "k": 2},
                    },
                    {
                        "id": "apply",
                        "type": "feature",
                        "feature": "global-kmeans",
                        "inputs": [{"step": "speed"}],
                        "params": {"model": {"step": "fit"}, "k": 2},
                    },
                ]
            }
        )
    )


def test_templates_and_model_together_are_refused() -> None:
    """``GlobalModelParams`` allows exactly one source, and validation asks it."""
    problem = _at(
        {
            "steps": [
                SPEED_STEP,
                {
                    "id": "templates",
                    "type": "feature",
                    "feature": "extract-templates",
                    "inputs": [{"step": "speed"}],
                    "params": {"n_templates": 4},
                },
                {
                    "id": "tsne",
                    "type": "feature",
                    "feature": "global-tsne",
                    "inputs": [{"step": "speed"}],
                    "params": {
                        "templates": {
                            "step": "templates",
                            "pattern": "templates.parquet",
                        },
                        "model": {
                            "feature": "global-tsne",
                            "run_id": "0.1-aaaaaaaaaa",
                            "load": {"kind": "joblib"},
                        },
                    },
                },
            ]
        },
        "params",
    )
    assert "Exactly one" in problem.message


@pytest.mark.parametrize(
    ("step", "reported"),
    [
        (
            {
                "id": "a",
                "type": "feature",
                "feature": "speed-angvel",
                "params": {"overwrite": True},
            },
            "a graph step may not overwrite",
        ),
        (
            {
                "id": "a",
                "type": "op",
                "kind": "trex",
                "params": {"overwrite": True},
            },
            "a graph step may not overwrite",
        ),
    ],
    ids=["feature", "op"],
)
def test_overwrite_in_params_is_refused(step: Document, reported: str) -> None:
    """Refused on presence, and refused once.

    It mutates content under a stable address, so a downstream reader gets a
    mixed read its own ``run_id`` records nothing about. ``overwrite: false`` in
    a file is an author expecting the key to mean something, so the refusal does
    not read the value -- and the params model is asked about everything else
    with the key removed, so the answer is not doubled by "extra inputs are not
    permitted" on a feature that has no such field.
    """
    problems = _problems({"steps": [step]})
    assert [problem.where for problem in problems] == ["params.overwrite"]
    assert reported in problems[0].message


def test_extract_frames_is_excluded_by_ownership() -> None:
    """Not a capability gap: mosaic-api owns its lifecycle and its identifier."""
    problem = _at(
        {
            "steps": [
                {
                    "id": "a",
                    "type": "op",
                    "kind": "extract-frames",
                    "params": {"method": "uniform", "n_frames": 3},
                }
            ]
        },
        "kind",
    )
    assert "mosaic-api" in problem.message


def test_an_op_may_not_be_wired_into_a_feature_s_inputs() -> None:
    """What a feature reads from a tracker is its tracks variant, not its run."""
    problem = _at(
        {
            "steps": [
                {"id": "trex", "type": "op", "kind": "trex"},
                {
                    "id": "speed",
                    "type": "feature",
                    "feature": "speed-angvel",
                    "inputs": [{"step": "trex"}],
                },
            ]
        },
        "inputs",
    )
    assert "tracks field" in problem.message


def test_a_tracks_reference_must_name_a_step_that_writes_tracks() -> None:
    problem = _at(
        {
            "steps": [
                SPEED_STEP,
                {
                    "id": "stack",
                    "type": "feature",
                    "feature": "temporal-stack",
                    "inputs": [{"step": "speed"}],
                    "tracks": {"step": "speed"},
                },
            ]
        },
        "tracks",
    )
    assert "writes no tracks variant" in problem.message


def test_a_join_of_mismatched_entity_granularity_is_refused() -> None:
    """The refusal that matters, and the one nothing else would ever report.

    Two inputs at different entity levels share no identity column, so merging
    them on ``frame`` alone pairs every row of one with every row of the other.
    It raises nothing and produces a plausible table.
    """
    problem = _at(
        {
            "steps": [
                SPEED_STEP,
                {
                    "id": "pairs",
                    "type": "feature",
                    "feature": "pair-egocentric",
                    "inputs": ["tracks"],
                },
                {
                    "id": "scaler",
                    "type": "feature",
                    "feature": "global-scaler",
                    "inputs": [{"step": "speed"}, {"step": "pairs"}],
                    "params": {
                        "templates": {"feature": "extract-templates", "run_id": None}
                    },
                },
            ]
        },
        "inputs",
    )
    assert "different entity levels" in problem.message


def test_a_fault_is_reported_once_rather_than_cascading() -> None:
    """A step below a broken one restates nothing: fix the upstream first."""
    problems = _problems(
        {
            "steps": [
                {"id": "bad", "type": "feature", "feature": "no-such-feature"},
                {
                    "id": "below",
                    "type": "feature",
                    "feature": "temporal-stack",
                    "inputs": [{"step": "bad"}],
                },
            ]
        }
    )
    assert [problem.step for problem in problems] == ["bad"]


def test_every_problem_is_reported_rather_than_the_first() -> None:
    """An author fixing a recipe wants the list, not one round trip per fault."""
    problems = _problems(
        {
            "steps": [
                {"id": "one", "type": "feature", "feature": "no-such-feature"},
                {"id": "two", "type": "op", "kind": "no-such-op"},
            ]
        }
    )
    assert {problem.step for problem in problems} == {"one", "two"}


def test_reject_unless_valid_carries_every_problem_on_the_exception() -> None:
    recipe = Recipe.model_validate(
        {
            "steps": [
                {"id": "one", "type": "feature", "feature": "no-such-feature"},
                {"id": "two", "type": "op", "kind": "no-such-op"},
            ]
        }
    )
    with pytest.raises(RecipeInvalid) as raised:
        reject_unless_valid(recipe)
    assert len(raised.value.problems) == 2
    assert "no-such-feature" in str(raised.value)
    assert "no-such-op" in str(raised.value)


def test_an_ordering_edge_from_a_media_writer_is_permitted() -> None:
    """``transcode -> trex`` is the ordinary shape, and it is admissible now.

    It was the one hazard here that produced a wrong answer rather than waste:
    a tracker's identity has no term for the media it read, so re-transcoding
    left every run below it reading as complete over different pixels. A bridged
    tracks row now records what media it consumed and the inventory compares it,
    so such a graph reads as drifted rather than as done.
    """
    reject_unless_valid(
        Recipe.model_validate(
            {
                "steps": [
                    {
                        "id": "transcode",
                        "type": "op",
                        "kind": "transcode",
                        "params": {"target": "analysis"},
                    },
                    {
                        "id": "trex",
                        "type": "op",
                        "kind": "trex",
                        "after": ["transcode"],
                    },
                ]
            }
        )
    )


_CROP_STEP: Document = {
    "id": "pre",
    "type": "op",
    "kind": "preprocess",
    "params": {"steps": [{"step": "crop", "x": 0, "y": 0, "width": 32, "height": 24}]},
}


def test_a_media_reference_to_a_preprocess_step_validates() -> None:
    """A tracker names a variant by the step that writes it, with no ``after``."""
    assert (
        _problems(
            {
                "steps": [
                    _CROP_STEP,
                    {
                        "id": "sleap",
                        "type": "op",
                        "kind": "sleap",
                        "params": {
                            "model_paths": ["models/sleap/run"],
                            "media": {"step": "pre"},
                        },
                    },
                ]
            }
        )
        == ()
    )


@pytest.mark.parametrize("writer", ["transcode", "export-joined", "trex"])
def test_a_media_reference_must_name_a_preprocess_step(writer: str) -> None:
    """Only a preprocess run id names a variant that a consumer can read.

    A transcode writes media too, but its run id does not address a file that a
    ``media`` field can resolve. The reference is refused at the step and field
    that contain it.
    """
    problem = _at(
        {
            "steps": [
                {"id": "upstream", "type": "op", "kind": writer},
                {
                    "id": "sleap",
                    "type": "op",
                    "kind": "sleap",
                    "params": {
                        "model_paths": ["models/sleap/run"],
                        "media": {"step": "upstream"},
                    },
                },
            ]
        },
        "params.media",
    )

    assert problem.step == "sleap"
    assert f"runs {writer!r}" in problem.message
    assert "preprocess" in problem.message


def test_a_preprocess_step_may_read_another_preprocess_step() -> None:
    """A chained variant names its upstream variant the same way a tracker does."""
    assert (
        _problems(
            {
                "steps": [
                    _CROP_STEP,
                    {
                        "id": "gray",
                        "type": "op",
                        "kind": "preprocess",
                        "params": {
                            "steps": [{"step": "grayscale"}],
                            "media": {"step": "pre"},
                        },
                    },
                ]
            }
        )
        == ()
    )


def test_a_field_the_params_model_refuses_is_reported_at_that_field() -> None:
    problem = _at(
        {
            "steps": [
                {
                    "id": "tsne",
                    "type": "feature",
                    "feature": "global-tsne",
                    "inputs": [{"step": "speed"}],
                    "params": {"perplexity": "x"},
                },
                SPEED_STEP,
            ]
        },
        "params.perplexity",
    )
    assert problem.step == "tsne"


def test_a_missing_optional_dependency_is_a_problem_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction raises ImportError for a package outside the prerequisites."""

    def _unavailable(_spec: object) -> NoReturn:
        raise ImportError("lightning-action is required")

    monkeypatch.setattr(validate_module, "build_step_feature", _unavailable)
    problem = _at({"steps": [SPEED_STEP]}, "feature")

    assert problem.step == "speed"
    assert "not installed" in problem.message


# --- labels references ----------------------------------------------------------


def _labels_recipe(*steps: Document) -> Document:
    """Two label conversions, the speed step, and *steps*."""
    return {
        "schema_version": 1,
        "name": "labels",
        "steps": [
            {
                "id": "labels",
                "type": "op",
                "kind": "convert-labels",
                "params": {"source_format": "calms21_npy"},
            },
            {
                "id": "swapped",
                "type": "op",
                "kind": "convert-labels",
                "params": {
                    "source_format": "calms21_npy",
                    "params": {"resident_id": 1, "intruder_id": 0},
                },
            },
            SPEED_STEP,
            *steps,
        ],
    }


def _labeled(step_id: str, labels: str) -> Document:
    return {
        "id": step_id,
        "type": "feature",
        "feature": "extract-labeled-templates",
        "inputs": [{"step": "speed"}],
        "params": {"labels": {"step": labels}, "n_total": 10},
    }


def test_a_labels_reference_must_name_a_labels_writer() -> None:
    problem = _at(_labels_recipe(_labeled("labeled", "speed")), "params.labels")

    assert problem.step == "labeled"
    assert "writes no labels variant" in problem.message


def test_a_labels_reference_to_a_convert_labels_step_validates() -> None:
    assert _problems(_labels_recipe(_labeled("labeled", "labels"))) == ()


TEMPLATE_STEPS: dict[str, Document] = {
    "extract-templates": {
        "id": "templates",
        "type": "feature",
        "feature": "extract-templates",
        "inputs": [{"step": "speed"}],
        "params": {"n_templates": 8},
    },
    "extract-labeled-templates": _labeled("templates", "labels"),
}


@pytest.mark.parametrize("template", sorted(TEMPLATE_STEPS))
@pytest.mark.parametrize(
    "inputs",
    [[{"step": "templates"}], [{"step": "templates"}, {"step": "speed"}]],
    ids=["alone", "beside-another-input"],
)
def test_reading_a_template_step_through_inputs_is_refused(
    template: str, inputs: list[Document]
) -> None:
    """A template step writes only its sample, and its table for each entry is empty.

    The problem names the inputs of the template step, and is raised even when
    another input meets the reader's requirements. It is the reader's only
    problem, because the columns that an empty input lacks would restate it.
    """
    kmeans: Document = {
        "id": "kmeans",
        "type": "feature",
        "feature": "global-kmeans",
        "inputs": inputs,
        "params": {"templates": {"step": "templates"}, "k": 2},
    }
    recipe = _labels_recipe(TEMPLATE_STEPS[template], kmeans)

    refused = [problem for problem in _problems(recipe) if problem.step == "kmeans"]

    assert len(refused) == 1
    assert "writes only its sample" in refused[0].message
    assert refused[0].where == "inputs"
    assert f"'templates' runs {template}" in refused[0].message
    assert "Read its inputs (speed) here" in refused[0].message


def test_a_reader_of_the_template_step_s_inputs_validates() -> None:
    kmeans: Document = {
        "id": "kmeans",
        "type": "feature",
        "feature": "global-kmeans",
        "inputs": [{"step": "speed"}],
        "params": {"templates": {"step": "templates"}, "k": 2},
    }

    assert _problems(_labels_recipe(TEMPLATE_STEPS["extract-templates"], kmeans)) == ()


def test_one_step_may_reference_one_labels_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run reads one labels variant, so two labels fields must name one step.

    No registered feature declares two labels fields, so the site rule is
    stood in for: every field of the step reads as a labels site.
    """

    def every_field_is_a_labels_site(*_: str) -> str:
        return "labels"

    monkeypatch.setattr(
        validate_module, "params_reference_site", every_field_is_a_labels_site
    )
    step = _labeled("labeled", "labels")
    params = step["params"]
    assert isinstance(params, dict)
    params["other_labels"] = {"step": "swapped"}
    recipe = Recipe.model_validate(_labels_recipe())
    walk = validate_module._walked(recipe, None)  # pyright: ignore[reportPrivateUsage]
    checked = Recipe.model_validate(_labels_recipe(step)).step("labeled")

    found = validate_module._labels_reference_problems(  # pyright: ignore[reportPrivateUsage]
        checked, "extract-labeled-templates", "feature", walk
    )

    assert [problem.where for problem in found] == ["params.other_labels"]
    assert "one run reads one labels variant" in found[0].message


def test_two_steps_differing_only_in_their_labels_step_are_two_runs() -> None:
    recipe = Recipe.model_validate(
        _labels_recipe(_labeled("first", "labels"), _labeled("second", "swapped"))
    )
    walk = validate_module._walked(recipe, None)  # pyright: ignore[reportPrivateUsage]

    assert walk.resolved["labels"].labels_kind == "behavior"
    assert walk.resolved["labels"].labels_variant != (
        walk.resolved["swapped"].labels_variant
    )
    assert walk.resolved["first"].run_id != walk.resolved["second"].run_id
