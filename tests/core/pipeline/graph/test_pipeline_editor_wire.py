"""The answers a pipeline editor reads, as the JSON it reads them in.

An editor lists the step types on offer, reports what is wrong with the recipe
it edits, offers what may follow a step, and offers what may be inserted between
a step and its children. Each answer is a wire model. The
CLI's ``--json`` output is the same model dumped. These tests check the models
and the functions that fill them. Every answer here is computed from the
registries alone, without a dataset. The plan preview's wire needs a dataset and
is tested in ``test_pipeline_plan.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    EXCLUDED_KINDS,
    TRACKS_INPUT,
    FeatureStepSpec,
    Problem,
    Recipe,
    RecipeInvalid,
    Step,
    StepRef,
    attach_candidates,
    check_recipe,
    declaration_catalog,
    editor_catalog,
    input_candidates,
    insert_candidates,
    params_reference_site,
    read_recipe,
    render_location,
    resolved_levels,
    step_type_detail,
    validation_report,
)

runner = CliRunner()

type Document = dict[str, object]

BRANCH_STEPS: list[Document] = [
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
        "id": "smooth",
        "type": "feature",
        "feature": "trajectory-smooth",
        "inputs": ["tracks"],
        "tracks": {"step": "trex"},
    },
    {
        "id": "speed",
        "type": "feature",
        "feature": "speed-angvel",
        "inputs": [{"step": "smooth"}],
    },
    {
        "id": "nn",
        "type": "feature",
        "feature": "nearest-neighbor",
        "inputs": [{"step": "smooth"}],
    },
]

BRANCH: Document = {
    "schema_version": 1,
    "name": "track, smooth, then two branches",
    "steps": BRANCH_STEPS,
}


# --- the catalog ------------------------------------------------------------------


def test_the_catalog_lists_every_runnable_step_type_and_serializes() -> None:
    catalog = editor_catalog()
    listed = {entry.name for entry in catalog.entries}

    assert listed == set(declaration_catalog().names()) - set(EXCLUDED_KINDS)
    assert [entry.name for entry in catalog.entries] == sorted(listed)
    _ = json.loads(catalog.model_dump_json())


def test_reference_fields_agree_with_how_validation_substitutes() -> None:
    """The catalog states each reference field, which a JSON Schema cannot declare.

    Each entry's ``reference_fields`` must name the same fields, with the same
    substitution, as the rule that validation applies.
    """
    for entry in editor_catalog().entries:
        if entry.type == "tracks":
            continue
        schema = entry.params_schema or {}
        properties = schema.get("properties", {})
        assert isinstance(properties, dict)
        for field_name in properties:
            site = params_reference_site(entry.type, entry.name, field_name)
            expected = site if site in ("result", "identifier") else None
            assert entry.reference_fields.get(field_name) == expected, (
                entry.name,
                field_name,
            )


def test_a_catalog_entry_carries_its_schema_version_and_summary() -> None:
    entries = {entry.name: entry for entry in editor_catalog().entries}
    tsne = entries["global-tsne"]

    assert tsne.reference_fields["templates"] == "result"
    assert tsne.params_schema is not None
    assert "perplexity" in str(tsne.params_schema)
    assert tsne.version
    assert tsne.summary
    assert entries[TRACKS_INPUT].params_schema is None


def test_a_catalog_summary_is_a_whole_sentence_below_the_name() -> None:
    """The editor shows the summary under the name. It ends its one sentence."""
    for entry in editor_catalog().entries:
        assert not entry.summary.startswith(f"'{entry.name}'"), entry.name
        assert entry.summary.endswith("."), entry.name


# --- reading and validating -------------------------------------------------------


def test_a_structural_fault_is_reported_at_the_step_s_id() -> None:
    report = validation_report(
        {"schema_version": 1, "steps": [{"id": "tsne", "type": "feature"}]}
    )

    assert not report.valid
    assert report.digest is None
    assert report.problems == (Problem("tsne", "feature", "Field required"),)


def test_a_step_without_an_id_is_reported_by_its_position() -> None:
    report = validation_report({"steps": [{"type": "feature"}]})

    assert {problem.where for problem in report.problems} == {
        "steps[0].id",
        "steps[0].feature",
    }
    assert all(problem.step == "" for problem in report.problems)


def test_a_step_with_an_unknown_type_is_reported_at_type() -> None:
    report = validation_report({"steps": [{"id": "a", "type": "widget"}]})

    assert [(problem.step, problem.where) for problem in report.problems] == [
        ("a", "type")
    ]


def test_a_document_that_is_not_a_recipe_is_one_whole_graph_problem() -> None:
    report = validation_report([])

    assert [(problem.step, problem.where) for problem in report.problems] == [
        ("", "recipe")
    ]


def test_read_recipe_raises_what_the_report_lists() -> None:
    document: Document = {"steps": [{"id": "tsne", "type": "feature"}]}

    with pytest.raises(RecipeInvalid) as raised:
        _ = read_recipe(document)
    assert raised.value.problems == validation_report(document).problems


def test_a_valid_recipe_reports_its_digest() -> None:
    report = validation_report(BRANCH)

    assert report.valid, report.problems
    assert report.digest is not None


@pytest.mark.parametrize(
    ("location", "rendered"),
    [
        ((), ""),
        (("perplexity",), "perplexity"),
        (("keypoint_names", 0), "keypoint_names[0]"),
        (("steps", 2, "inputs", 1), "steps[2].inputs[1]"),
    ],
)
def test_render_location(location: tuple[int | str, ...], rendered: str) -> None:
    assert render_location(location) == rendered


# --- levels and attach candidates -------------------------------------------------


def test_a_passthrough_resolves_to_the_level_of_its_input() -> None:
    recipe = Recipe.model_validate(
        {
            "steps": [
                {"id": "pair", "type": "feature", "feature": "pair-egocentric"},
                {
                    "id": "smooth",
                    "type": "feature",
                    "feature": "movement-smooth",
                    "inputs": [{"step": "pair"}],
                },
            ]
        }
    )

    assert resolved_levels(recipe, declaration_catalog()) == {
        "pair": "pair",
        "smooth": "pair",
    }


def test_both_branches_may_follow_the_smoothing_step() -> None:
    report = attach_candidates(read_recipe(BRANCH), "smooth")

    for name in ("speed-angvel", "nearest-neighbor"):
        attachment = report.candidates[name]
        assert attachment.allowed, attachment.reason
        assert attachment.template == {
            "type": "feature",
            "feature": name,
            "inputs": [{"step": "smooth"}],
        }


def test_an_op_may_not_follow_a_feature() -> None:
    attachment = attach_candidates(read_recipe(BRANCH), "smooth").candidates["trex"]

    assert not attachment.allowed
    assert "takes no pipeline inputs" in attachment.reason
    assert attachment.template is None


def test_a_top_level_feature_reads_tracks_or_is_refused_with_the_reason() -> None:
    report = attach_candidates(read_recipe(BRANCH), None)

    assert report.candidates["speed-angvel"].template == {
        "type": "feature",
        "feature": "speed-angvel",
        "inputs": ["tracks"],
    }
    refused = report.candidates["global-tsne"]
    assert not refused.allowed
    assert "does not read tracks directly" in refused.reason
    assert report.candidates["trex"].template == {"type": "op", "kind": "trex"}


def test_a_feature_below_a_tracker_reads_its_tracks_variant() -> None:
    attachment = attach_candidates(read_recipe(BRANCH), "trex").candidates[
        "speed-angvel"
    ]

    assert attachment.template == {
        "type": "feature",
        "feature": "speed-angvel",
        "inputs": ["tracks"],
        "tracks": {"step": "trex"},
    }


def test_an_op_below_an_op_is_ordered_after_it() -> None:
    report = attach_candidates(read_recipe(BRANCH), "transcode")

    assert report.candidates["trex"].template == {
        "type": "op",
        "kind": "trex",
        "after": ["transcode"],
    }
    refused = report.candidates["speed-angvel"]
    assert not refused.allowed
    assert "writes no tracks variant" in refused.reason


def test_excluded_kinds_and_tracks_are_never_candidates() -> None:
    names = set(attach_candidates(read_recipe(BRANCH), None).candidates)

    assert not names & set(EXCLUDED_KINDS)
    assert TRACKS_INPUT not in names


def test_attaching_below_an_unknown_step_is_refused() -> None:
    with pytest.raises(RecipeInvalid) as raised:
        _ = attach_candidates(read_recipe(BRANCH), "no-such-step")
    assert raised.value.problems[0].where == "after"


# --- inserting between a step and its children -------------------------------------


def _reading_inserted(step: Step, after: str) -> Step:
    """*step* reading the step ``inserted`` first, where it read *after* first."""
    if not isinstance(step, FeatureStepSpec) or not step.inputs:
        return step
    first, *rest = step.inputs
    if not isinstance(first, StepRef) or first.step != after:
        return step
    return step.model_copy(update={"inputs": [StepRef(step="inserted"), *rest]})


def _inserted(recipe: Recipe, after: str, template: dict[str, JsonValue]) -> Recipe:
    """*recipe* with a feature *template* inserted below *after*, read by its children."""
    added = FeatureStepSpec.model_validate({"id": "inserted", **template})
    steps = [_reading_inserted(step, after) for step in recipe.steps]
    return Recipe(name=recipe.name, steps=[*steps, added])


def test_inserting_below_the_smoothing_step_reaches_both_branches() -> None:
    report = insert_candidates(read_recipe(BRANCH), "smooth")

    assert report.children == ("speed", "nn")
    again = report.candidates["trajectory-smooth"]
    assert again.allowed, again.reason
    assert again.template == {
        "type": "feature",
        "feature": "trajectory-smooth",
        "inputs": [{"step": "smooth"}],
    }
    assert again.template is not None
    grown = _inserted(read_recipe(BRANCH), "smooth", again.template)
    assert check_recipe(grown) == ()


def test_a_candidate_that_a_child_cannot_read_is_refused_naming_the_child() -> None:
    recipe = read_recipe(BRANCH)

    refused = insert_candidates(recipe, "smooth").candidates["speed-angvel"]

    assert attach_candidates(recipe, "smooth").candidates["speed-angvel"].allowed
    assert not refused.allowed
    assert refused.template is None
    assert refused.reason.startswith("'speed' cannot read speed-angvel")


def test_a_candidate_whose_output_lacks_a_child_s_column_is_refused() -> None:
    document: Document = {
        "steps": [
            {"id": "speed", "type": "feature", "feature": "speed-angvel"},
            {
                "id": "mean_speed",
                "type": "feature",
                "feature": "frame-aggregate",
                "inputs": [{"step": "speed"}],
                "params": {"column": "speed"},
            },
        ]
    }

    refused = insert_candidates(read_recipe(document), "speed").candidates[
        "ffgroups-metrics"
    ]

    assert not refused.allowed
    assert refused.reason.startswith(
        "'mean_speed' would read ffgroups-metrics in place of 'speed'"
    )
    assert "lack 'speed'" in refused.reason


def test_a_candidate_that_breaks_a_step_below_the_children_is_refused() -> None:
    """``temporal-stack`` takes its input's entity level, and passes it down.

    ``attention-target`` writes individuals. Read by ``stack`` in place of the
    pair-level ``facing``, it leaves ``both`` joining individuals from ``stack``
    with pairs from ``pos``.
    """
    document: Document = {
        "steps": [
            {"id": "smooth", "type": "feature", "feature": "trajectory-smooth"},
            {
                "id": "facing",
                "type": "feature",
                "feature": "pair-facing",
                "inputs": [{"step": "smooth"}],
            },
            {
                "id": "pos",
                "type": "feature",
                "feature": "pair-egocentric",
                "inputs": [{"step": "smooth"}],
            },
            {
                "id": "stack",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": [{"step": "facing"}],
            },
            {
                "id": "both",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": [{"step": "stack"}, {"step": "pos"}],
            },
        ]
    }
    recipe = read_recipe(document)

    report = insert_candidates(recipe, "facing")
    refused = report.candidates["attention-target"]

    assert check_recipe(recipe) == ()
    assert report.children == ("stack",)
    assert attach_candidates(recipe, "facing").candidates["attention-target"].allowed
    assert not refused.allowed
    assert refused.reason.startswith("'both' would then be invalid:")
    assert "different entity levels" in refused.reason


def test_a_join_reading_the_step_through_a_later_input_is_not_a_child() -> None:
    """``both`` reads ``facing`` through its second input, and keeps reading it.

    ``attention-target`` writes individuals, and ``both`` cannot join them with
    the pairs from ``pos``. It may stand between ``facing`` and ``stack``, the
    one step that reads ``facing`` first.
    """
    document: Document = {
        "steps": [
            {"id": "smooth", "type": "feature", "feature": "trajectory-smooth"},
            {
                "id": "facing",
                "type": "feature",
                "feature": "pair-facing",
                "inputs": [{"step": "smooth"}],
            },
            {
                "id": "pos",
                "type": "feature",
                "feature": "pair-egocentric",
                "inputs": [{"step": "smooth"}],
            },
            {
                "id": "stack",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": [{"step": "facing"}],
            },
            {
                "id": "both",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": [{"step": "pos"}, {"step": "facing"}],
            },
        ]
    }
    recipe = read_recipe(document)

    report = insert_candidates(recipe, "facing")
    between = report.candidates["attention-target"]

    assert check_recipe(recipe) == ()
    assert report.children == ("stack",)
    assert between.allowed, between.reason
    assert insert_candidates(recipe, "pos").children == ("both",)


def test_below_a_tracker_the_children_read_its_variant() -> None:
    report = insert_candidates(read_recipe(BRANCH), "trex")

    assert report.children == ("smooth",)
    again = report.candidates["trajectory-smooth"]
    assert again.allowed, again.reason
    assert again.template == {
        "type": "feature",
        "feature": "trajectory-smooth",
        "inputs": ["tracks"],
        "tracks": {"step": "trex"},
    }
    assert report.candidates["speed-angvel"].reason.startswith("'smooth' cannot read")


def test_inserting_below_a_leaf_is_attaching() -> None:
    recipe = read_recipe(BRANCH)

    report = insert_candidates(recipe, "nn")

    assert report.children == ()
    assert report.candidates == attach_candidates(recipe, "nn").candidates


def test_inserting_below_an_unknown_step_is_refused() -> None:
    with pytest.raises(RecipeInvalid) as raised:
        _ = insert_candidates(read_recipe(BRANCH), "no-such-step")
    assert raised.value.problems[0].where == "after"


def test_a_params_error_elsewhere_does_not_block_the_picker() -> None:
    speed = {**BRANCH_STEPS[3], "params": {"not_a_field": 1}}
    document: Document = {
        **BRANCH,
        "steps": [*BRANCH_STEPS[:3], speed, *BRANCH_STEPS[4:]],
    }

    report = attach_candidates(read_recipe(document), "smooth")
    assert report.candidates["speed-angvel"].allowed


def test_every_template_is_wired_the_way_validation_accepts() -> None:
    """A template plus an id validates, apart from the parts the author still adds.

    Those are params the author must fill, and required columns that its one
    input lacks. ``input_candidates`` reports each such column as still needed,
    and validation reports the same column as a problem. A step type that this
    installation cannot run is refused without a template. A feature whose
    construction fails for want of an undeclared package therefore fails here.
    """
    recipe = read_recipe(BRANCH)
    for above in (None, *(step.id for step in recipe.steps)):
        report = attach_candidates(recipe, above)
        for name, attachment in report.candidates.items():
            if attachment.template is None:
                continue
            grown = Recipe.model_validate(
                {"steps": [*BRANCH_STEPS, {"id": "added", **attachment.template}]}
            )
            needed = (
                {
                    still.need.label
                    for still in input_candidates(grown, "added").still_needs
                    if still.status == "absent"
                }
                if attachment.template.get("type") == "feature"
                else set[str]()
            )
            wiring = [
                problem
                for problem in check_recipe(grown)
                if problem.step == "added"
                and not problem.where.startswith("params")
                and not any(
                    problem.message.startswith(f"needs {label},") for label in needed
                )
            ]
            assert not wiring, (above, name, wiring)


# --- the CLI ----------------------------------------------------------------------


def test_catalog_json_is_the_editor_catalog() -> None:
    result = runner.invoke(app, ["pipeline", "catalog", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == editor_catalog().model_dump(mode="json")


def test_catalog_describes_a_named_op_as_its_step_type_detail() -> None:
    result = runner.invoke(app, ["pipeline", "catalog", "trex", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload == step_type_detail("trex").model_dump(mode="json")
    assert payload["type"] == "op"
    assert payload["params"]
    assert payload["params_schema"] is None


def test_catalog_gives_the_schema_of_a_named_step_type_on_request() -> None:
    result = runner.invoke(
        app, ["pipeline", "catalog", "tracks", "--detail", "schema", "--json"]
    )
    schema = runner.invoke(
        app, ["pipeline", "catalog", "speed-angvel", "--detail", "schema", "--json"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["params"] == []
    assert "step_size" in json.loads(schema.stdout)["params_schema"]["properties"]


def test_catalog_refuses_a_name_no_step_type_has() -> None:
    result = runner.invoke(app, ["pipeline", "catalog", "no-such-type", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["kind"] == "not_found"


def test_validate_answers_an_unparseable_recipe_in_json(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    _ = path.write_text(json.dumps({"steps": [{"id": "tsne", "type": "feature"}]}))

    result = runner.invoke(
        app, ["pipeline", "validate", "--recipe", f"@{path}", "--json"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert payload["problems"] == [
        {"step": "tsne", "where": "feature", "message": "Field required"}
    ]
