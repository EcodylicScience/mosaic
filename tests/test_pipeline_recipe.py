"""The recipe file: what it accepts, what it refuses, and how it is ordered.

Structural refusals only -- the semantic ones (an unknown slug, params that do
not validate, an inadmissible edge) need the registries and are tested beside
``validate``. What is here is what a recipe means as a document: ids are unique
and named, every reference resolves, the digest is a property of the graph rather
than of its formatting, and the order is deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from mosaic.core.pipeline.graph import (
    BoundRef,
    Recipe,
    RecipeCycle,
    Request,
    ancestors_of,
    canonical_json,
    children_of,
    descendants_of,
    edges,
    load_recipe,
    load_request,
    params_step_refs,
    parents_of,
    recipe_digest,
    recipe_path,
    save_recipe,
    save_request,
    storage_name_of,
    topological_order,
)

type Document = dict[str, object]

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
            "params": {},
        },
        {
            "id": "templates",
            "type": "feature",
            "feature": "extract-templates",
            "inputs": [{"step": "speed"}],
            "params": {},
        },
        {
            "id": "tsne",
            "type": "feature",
            "feature": "global-tsne",
            "inputs": [{"step": "speed"}],
            "params": {
                "templates": {"step": "templates", "pattern": "templates.parquet"},
                "perplexity": 50,
            },
        },
    ],
}


def _copy_of_worked() -> Document:
    """A fresh mutable copy, so one test cannot edit another's fixture."""
    return cast("Document", json.loads(json.dumps(WORKED_EXAMPLE)))


def _steps(document: Document) -> list[Document]:
    """The step list of a raw document, typed."""
    return cast("list[Document]", document["steps"])


@pytest.fixture
def worked() -> Recipe:
    return Recipe.model_validate(WORKED_EXAMPLE)


# --- structure ---------------------------------------------------------------


def test_the_worked_example_parses(worked: Recipe) -> None:
    assert worked.ids == ("transcode", "trex", "speed", "templates", "tsne")


def test_a_duplicate_step_id_is_refused() -> None:
    doc = {
        "steps": [
            {"id": "a", "type": "feature", "feature": "speed-angvel"},
            {"id": "a", "type": "feature", "feature": "heading"},
        ]
    }
    with pytest.raises(ValidationError, match="duplicate step id"):
        _ = Recipe.model_validate(doc)


def test_an_empty_step_id_is_refused() -> None:
    doc = {"steps": [{"id": "  ", "type": "feature", "feature": "speed-angvel"}]}
    with pytest.raises(ValidationError, match="non-empty id"):
        _ = Recipe.model_validate(doc)


def test_a_reference_to_an_undeclared_step_is_refused() -> None:
    doc = {
        "steps": [
            {
                "id": "b",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": [{"step": "nope"}],
            }
        ]
    }
    with pytest.raises(ValidationError, match="unknown step 'nope'"):
        _ = Recipe.model_validate(doc)


def test_a_self_reference_is_refused() -> None:
    doc = {
        "steps": [
            {
                "id": "a",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": [{"step": "a"}],
            }
        ]
    }
    with pytest.raises(ValidationError, match="references itself"):
        _ = Recipe.model_validate(doc)


def test_an_unknown_field_is_refused_rather_than_ignored() -> None:
    """A typo silently dropped is a step that does not do what its author read."""
    doc = {
        "steps": [
            {
                "id": "a",
                "type": "feature",
                "feature": "speed-angvel",
                "paramz": {"window": 5},
            }
        ]
    }
    with pytest.raises(ValidationError):
        _ = Recipe.model_validate(doc)


def test_a_newer_schema_version_is_refused_rather_than_read() -> None:
    doc: dict[str, object] = {"schema_version": 99, "steps": []}
    with pytest.raises(ValidationError, match="newer than this mosaic understands"):
        _ = Recipe.model_validate(doc)


def test_overwrite_has_no_field_in_the_run_block() -> None:
    """It mutates content under a stable address; the format must not express it."""
    doc = {
        "steps": [
            {
                "id": "a",
                "type": "feature",
                "feature": "speed-angvel",
                "run": {"overwrite": True},
            }
        ]
    }
    with pytest.raises(ValidationError):
        _ = Recipe.model_validate(doc)


# --- params references -------------------------------------------------------


def test_a_params_step_reference_is_found_by_shape(worked: Recipe) -> None:
    refs = params_step_refs(worked.step("tsne").params)
    assert set(refs) == {"templates"}
    assert refs["templates"].step == "templates"
    assert refs["templates"].pattern == "templates.parquet"


def test_a_nested_params_reference_is_not_found() -> None:
    """Only top-level fields are substituted, so only those may be found."""
    assert params_step_refs({"outer": {"inner": {"step": "a"}}}) == {}


def test_a_malformed_params_reference_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="not a valid one"):
        _ = params_step_refs({"templates": {"step": "a", "patttern": "x.parquet"}})


# --- topology ----------------------------------------------------------------


def test_topological_order_puts_every_step_after_its_references(
    worked: Recipe,
) -> None:
    ordered = [step.id for step in topological_order(worked)]
    assert ordered.index("transcode") < ordered.index("trex")
    assert ordered.index("trex") < ordered.index("speed")
    assert ordered.index("speed") < ordered.index("templates")
    assert ordered.index("templates") < ordered.index("tsne")


def test_topological_order_is_declaration_order_where_the_graph_allows() -> None:
    """Two independent branches keep the order their author wrote them in."""
    doc = {
        "steps": [
            {"id": "root", "type": "feature", "feature": "speed-angvel"},
            {
                "id": "zulu",
                "type": "feature",
                "feature": "temporal-stack",
                "inputs": [{"step": "root"}],
            },
            {
                "id": "alpha",
                "type": "feature",
                "feature": "frame-aggregate",
                "inputs": [{"step": "root"}],
            },
        ]
    }
    recipe = Recipe.model_validate(doc)
    assert [step.id for step in topological_order(recipe)] == ["root", "zulu", "alpha"]


def test_a_cycle_is_named_rather_than_looped_on() -> None:
    doc = {
        "steps": [
            {
                "id": "a",
                "type": "feature",
                "feature": "temporal-stack",
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
    recipe = Recipe.model_validate(doc)
    with pytest.raises(RecipeCycle, match="a, b"):
        _ = topological_order(recipe)


def test_an_after_edge_counts_as_a_parent(worked: Recipe) -> None:
    """Ordering-only is still ordering: a step held on parents is held on that one."""
    assert parents_of(worked, "trex") == ("transcode",)


def test_parents_are_deduplicated_across_reference_sites() -> None:
    """One upstream named twice is one parent, not two."""
    doc = {
        "steps": [
            {"id": "speed", "type": "feature", "feature": "speed-angvel"},
            {
                "id": "down",
                "type": "feature",
                "feature": "global-tsne",
                "inputs": [{"step": "speed"}],
                "params": {"templates": {"step": "speed"}},
                "after": ["speed"],
            },
        ]
    }
    assert parents_of(Recipe.model_validate(doc), "down") == ("speed",)


def test_children_descendants_and_ancestors(worked: Recipe) -> None:
    assert children_of(worked, "speed") == ("templates", "tsne")
    assert descendants_of(worked, "speed") == {"speed", "templates", "tsne"}
    assert ancestors_of(worked, "tsne") == {
        "tsne",
        "speed",
        "templates",
        "trex",
        "transcode",
    }


def test_edges_are_derived_from_the_bodies_with_their_sites(worked: Recipe) -> None:
    """There is no edges array to drift; this reads the substitution sites back."""
    found = {(e.producer, e.consumer, e.site, e.where) for e in edges(worked)}
    assert ("transcode", "trex", "after", "after[0]") in found
    assert ("trex", "speed", "tracks", "tracks") in found
    assert ("speed", "templates", "inputs", "inputs[0]") in found
    assert ("templates", "tsne", "params", "params.templates") in found


# --- digest ------------------------------------------------------------------


def test_the_digest_ignores_key_order(worked: Recipe) -> None:
    """Two spellings of one graph are one pipeline, not two."""
    doc = _copy_of_worked()
    doc["steps"] = [dict(reversed(list(step.items()))) for step in _steps(doc)]
    shuffled: Document = dict(reversed(list(doc.items())))
    assert list(_steps(shuffled)[0]) != list(_steps(_copy_of_worked())[0])
    assert recipe_digest(Recipe.model_validate(shuffled)) == recipe_digest(worked)


def test_the_digest_ignores_an_explicitly_written_default(worked: Recipe) -> None:
    """An omitted default and a written one are the same recipe."""
    doc = _copy_of_worked()
    for step in _steps(doc):
        if step["type"] == "feature":
            step["run"] = {"overlap_frames": 0}
    assert recipe_digest(Recipe.model_validate(doc)) == recipe_digest(worked)


def test_the_digest_moves_when_a_param_moves(worked: Recipe) -> None:
    doc = _copy_of_worked()
    params = cast("Document", _steps(doc)[-1]["params"])
    params["perplexity"] = 30
    assert recipe_digest(Recipe.model_validate(doc)) != recipe_digest(worked)


def test_the_canonical_form_is_compact_and_sorted(worked: Recipe) -> None:
    text = canonical_json(worked)
    assert ", " not in text and '": ' not in text
    assert text.startswith('{"name":')


# --- store -------------------------------------------------------------------


def test_saving_a_recipe_is_idempotent_and_addressed_by_digest(
    tmp_path: Path, worked: Recipe
) -> None:
    first = save_recipe(tmp_path, worked)
    second = save_recipe(tmp_path, worked)
    assert first == second == recipe_path(tmp_path, recipe_digest(worked))
    assert load_recipe(first).ids == worked.ids


def test_a_request_round_trips(tmp_path: Path, worked: Recipe) -> None:
    request = Request(
        request_id="req-1",
        recipe_digest=recipe_digest(worked),
        owner="jacob",
        entries=[("", "seq_a")],
        allow_partial=True,
        step_executions={"speed": "01ABC"},
        step_versions={"speed": "0.3", "trex": "0.2"},
    )
    _ = save_request(tmp_path, request)
    read_back = load_request(tmp_path, "req-1")
    assert read_back == request
    assert read_back.entry_set() == frozenset({("", "seq_a")})
    assert read_back.execution_of("speed") == "01ABC"


def test_a_request_names_the_step_it_assigned_no_attempt(worked: Recipe) -> None:
    """An unassigned step is a malformed request, not a step yet to start.

    Returning an empty identifier would be read as "no attempt yet" by a step
    pinning its parent, which is the one reading that must not be guessable.
    """
    request = Request(request_id="r", recipe_digest=recipe_digest(worked))
    with pytest.raises(KeyError, match="assigns no execution id to step 'speed'"):
        _ = request.execution_of("speed")


def test_a_recipe_never_carries_a_resolved_run_id(worked: Recipe) -> None:
    """G5: identifiers are dataset state, so the portable file must hold none."""
    assert "run_id" not in canonical_json(worked)


def test_a_bind_must_actually_pin(worked: Recipe) -> None:
    with pytest.raises(ValidationError, match="names no run_id"):
        _ = Request(
            request_id="r",
            recipe_digest=recipe_digest(worked),
            bind={"speed": BoundRef(feature="speed-angvel__from__tracks", run_id="")},
        )


# --- storage names -----------------------------------------------------------


def test_storage_name_matches_the_run_feature_rule() -> None:
    assert storage_name_of("speed-angvel", ["tracks"]) == "speed-angvel__from__tracks"
    assert storage_name_of("global-tsne", []) == "global-tsne"
    assert (
        storage_name_of("temporal-stack", ["speed-angvel__from__tracks", "tracks"])
        == "temporal-stack__from__speed-angvel__from__tracks+tracks"
    )
