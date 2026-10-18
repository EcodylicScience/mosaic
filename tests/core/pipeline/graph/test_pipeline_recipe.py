"""The recipe file: what it accepts, what it refuses, and how it is ordered.

Structural refusals only -- the semantic ones (an unknown slug, params that do
not validate, an inadmissible edge) need the registries and are tested beside
``validate``. What is here is what a recipe means as a document: ids are unique
and named, every reference resolves, the digest is a property of the graph rather
than of its formatting, and the order is deterministic.
"""

from __future__ import annotations

import json
import math
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
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline._utils import (
    NAME_MAX_BYTES,
    SHORTENED_MARKER,
    SHORTENED_NAME_BYTES,
    derive_storage_name,
    hash_params,
)
from mosaic.core.pipeline.graph.digest import document_digest
from mosaic.core.pipeline.graph.model import canonical_params
from mosaic.core.scope import Scope
from tests.helpers import recipe_saved_by_0_17

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


PINNED_DOCUMENT: Document = {
    "name": "p3",
    "steps": [
        {
            "id": "trex",
            "type": "op",
            "kind": "trex",
            "params": {"track_threshold": 15.0, "cm_per_pixel": 0.05},
        },
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "tracks": {"step": "trex"},
            "params": {
                "f_max": 5.0,
                "log_floor": -3.0,
                "zero": -0.0,
                "flag": True,
                "sampling": {"fps_default": 30.0},
            },
            "run": {"filter_start_time": 5.0},
        },
    ],
}
"""A recipe with whole-number floats in params and in a typed field."""


def test_an_integral_float_and_its_integer_are_one_recipe() -> None:
    """A hand-written ``5.0`` and a JavaScript client's ``5`` name one recipe."""
    as_floats: Document = {
        "name": "n",
        "steps": [
            {
                "id": "speed",
                "type": "feature",
                "feature": "speed-angvel",
                "params": {"fps": 30.0, "nested": {"floor": -3.0, "each": [1.0]}},
            }
        ],
    }
    as_ints: Document = {
        "name": "n",
        "steps": [
            {
                "id": "speed",
                "type": "feature",
                "feature": "speed-angvel",
                "params": {"fps": 30, "nested": {"floor": -3, "each": [1]}},
            }
        ],
    }
    assert recipe_digest(Recipe.model_validate(as_floats)) == recipe_digest(
        Recipe.model_validate(as_ints)
    )


def test_a_whole_number_in_params_parses_as_an_integer() -> None:
    params: dict[str, JsonValue] = {
        "whole": 5.0,
        "negative_zero": -0.0,
        "flag": True,
        "half": 0.5,
        "nan": float("nan"),
        "inf": float("inf"),
        "huge": 2.0**53,
        "nested": [{"x": 2.0}, [3.0]],
    }
    found = canonical_params(params)
    assert found["whole"] == 5 and type(found["whole"]) is int
    assert found["negative_zero"] == 0 and type(found["negative_zero"]) is int
    assert found["flag"] is True
    assert found["half"] == 0.5
    assert math.isnan(cast("float", found["nan"]))
    assert found["inf"] == float("inf")
    assert type(found["huge"]) is float
    assert found["nested"] == [{"x": 2}, [3]]
    assert type(cast("list[dict[str, int]]", found["nested"])[0]["x"]) is int


def test_a_typed_float_field_digests_alike_however_written() -> None:
    written_int = json.loads(json.dumps(PINNED_DOCUMENT))
    _steps(written_int)[1]["run"] = {"filter_start_time": 5}
    assert recipe_digest(Recipe.model_validate(written_int)) == recipe_digest(
        Recipe.model_validate(PINNED_DOCUMENT)
    )


def test_the_digest_of_integral_floats_is_pinned() -> None:
    """Never regenerate these: a recipe's name must not move by accident.

    ``a382f3f695561521`` is what mosaic 0.17 named this recipe, before whole
    numbers in params were normalized; ``tests/data/recipe_saved_by_0_17.json``
    is the file it saved.
    """
    assert recipe_digest(Recipe.model_validate(PINNED_DOCUMENT)) == "51f069d7b95bc6a4"
    assert document_digest(recipe_saved_by_0_17()) == "a382f3f695561521"


def test_saving_writes_the_document_its_digest_names(tmp_path: Path) -> None:
    recipe = Recipe.model_validate(PINNED_DOCUMENT)
    path = save_recipe(tmp_path, recipe)
    text = path.read_text()
    assert document_digest(json.loads(text)) == recipe_digest(recipe)
    assert '"fps_default": 30' in text and "30.0" not in text


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
        scope=Scope(entries=[("", "seq_a")]),
        allow_partial=True,
        step_executions={"speed": "01ABC"},
        step_versions={"speed": "0.3", "trex": "0.2"},
    )
    _ = save_request(tmp_path, request)
    read_back = load_request(tmp_path, "req-1")
    assert read_back == request
    assert read_back.scope == Scope(entries=[("", "seq_a")])
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


_EGO = "pair-egocentric__from__tracks"
_PCA = "pair-posedistance-pca__from__tracks"
_WAVELET_PCA = f"pair-wavelet__from__{_PCA}"
_WAVELET_EGO = f"pair-wavelet__from__{_EGO}"
_SCALER = storage_name_of("global-scaler", [_EGO, _PCA, _WAVELET_PCA, _WAVELET_EGO])
_STACK = storage_name_of("temporal-stack", [_SCALER])


def _bytes(name: str) -> int:
    return len(name.encode("utf-8"))


def test_a_storage_name_that_fits_is_unchanged() -> None:
    """The longest name a CalMS21 run wrote, four levels deep, keeps its lineage."""
    stack = storage_name_of("temporal-stack", [_EGO, _PCA])
    classifier = storage_name_of("xgboost", [stack, _WAVELET_PCA, _WAVELET_EGO])
    name = storage_name_of("evaluate-labels", [classifier])

    assert _bytes(name) == 231
    assert name == f"evaluate-labels__from__{classifier}"
    assert SHORTENED_MARKER not in name


def test_a_storage_name_too_long_for_a_directory_is_abbreviated() -> None:
    """Two multi-input parents give a 440-byte lineage, which cannot be created."""
    full = f"extract-labeled-templates__from__{_SCALER}+{_STACK}"
    assert _bytes(full) == 440

    name = storage_name_of("extract-labeled-templates", [_SCALER, _STACK])

    assert name == (
        "extract-labeled-templates__from__global-scaler__from__"
        "pair-egocentric__from__tracks+pair-posedistance-pca__from__tracks"
        "+pair-wavelet~ac28332f35"
    )
    assert _bytes(name) <= SHORTENED_NAME_BYTES
    assert name == storage_name_of("extract-labeled-templates", [_SCALER, _STACK])


def test_a_single_input_lineage_past_the_limit_is_abbreviated() -> None:
    parent = storage_name_of("temporal-stack", [_STACK])
    name = storage_name_of("extract-labeled-templates", [parent])

    assert _bytes(f"extract-labeled-templates__from__{parent}") > NAME_MAX_BYTES
    assert name.startswith("extract-labeled-templates__from__temporal-stack__from__")
    assert SHORTENED_MARKER in name
    assert _bytes(name) <= SHORTENED_NAME_BYTES


def test_an_input_longer_than_the_budget_leaves_an_empty_head() -> None:
    """A single token past the budget is cut at its start."""
    name = storage_name_of("temporal-stack", ["x" * 300])

    assert name == f"temporal-stack__from__{SHORTENED_MARKER}{hash_params('x' * 300)}"


def test_lineages_that_share_a_head_get_different_names() -> None:
    other_stack = storage_name_of("temporal-stack", [_SCALER.replace("ego", "allo")])
    name = storage_name_of("extract-labeled-templates", [_SCALER, _STACK])
    other = storage_name_of("extract-labeled-templates", [_SCALER, other_stack])

    head = name.split(SHORTENED_MARKER)[0]
    assert other.split(SHORTENED_MARKER)[0] == head
    assert other != name


@pytest.mark.parametrize(
    "inputs",
    [["tracks"], [_EGO, _PCA], [_SCALER, _STACK], ["x" * 300]],
    ids=["one", "two", "abbreviated", "empty-head"],
)
def test_the_planner_and_run_feature_derive_one_name(inputs: list[str]) -> None:
    assert storage_name_of("temporal-stack", inputs) == derive_storage_name(
        "temporal-stack", "+".join(inputs)
    )
