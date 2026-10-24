"""One step type in detail, its params as a flat table.

``step_type_detail`` extends a step type's index row with what it consumes, its
resource class, whether its output depends on its scope, and one row per params
field. Nested models read as dotted paths, and a field that takes a run reference
is typed as a recipe writes one. ``detail="schema"`` adds the params JSON Schema
without titles or class docstrings.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    StepTypeDetail,
    StepTypeNotFound,
    catalog_index,
    compact_params_schema,
    param_rows,
    step_type_detail,
)


def _keys_outside_property_maps(value: JsonValue) -> set[str]:
    """Every schema keyword in *value*, leaving out names in a properties map."""
    found: set[str] = set()
    if isinstance(value, list):
        for item in value:
            found |= _keys_outside_property_maps(item)
    elif isinstance(value, dict):
        for key, inner in value.items():
            found.add(key)
            if key in {"properties", "$defs"} and isinstance(inner, dict):
                for schema in inner.values():
                    found |= _keys_outside_property_maps(schema)
            elif key not in {"default", "const", "enum", "examples"}:
                found |= _keys_outside_property_maps(inner)
    return found


def test_each_step_type_detail_matches_its_index_row() -> None:
    for row in catalog_index().entries:
        detail = step_type_detail(row.name)

        assert detail.model_dump(include=set(type(row).model_fields)) == (
            row.model_dump()
        )
        assert detail.params_schema is None


def test_nested_config_fields_read_as_dotted_paths() -> None:
    rows = {row.path: row for row in step_type_detail("global-tsne").params}

    assert rows["fit"].type == "TSNEFitConfig"
    assert rows["fit"].constructed
    assert rows["fit.exaggeration_iters"].type == "integer"
    assert rows["fit.exaggeration_iters"].default == 250
    assert rows["fit.exaggeration_iters"].constraints == ">= 1"
    assert rows["mapping.k"].description


def test_a_result_field_is_shown_in_recipe_form() -> None:
    rows = {row.path: row for row in step_type_detail("global-kmeans").params}

    templates = rows["templates"]
    assert templates.reference == "result"
    assert templates.type == '{"step": "<id>", "pattern"?: "<glob>"}'
    # A reference is not expanded into the fields of the artifact it names.
    assert not any(path.startswith("templates.") for path in rows)
    # A plain Result takes no pattern.
    assert rows["pair_filter"].type == '{"step": "<id>"}'


def test_identifier_fields_carry_no_reference_flag() -> None:
    rows = {row.path: row for row in step_type_detail("global-kmeans").params}

    assert rows["device"].reference is None
    assert rows["device"].type == "string"


def test_a_feature_reports_its_class_scope_dependence() -> None:
    assert step_type_detail("extract-templates").scope_dependent
    assert not step_type_detail("speed-angvel").scope_dependent


def test_fits_is_not_scope_dependence() -> None:
    # A scope-free fitter learns without naming its run by its scope, and a
    # scoring step names its run by its scope and learns nothing.
    assert step_type_detail("xgboost").fits
    assert not step_type_detail("xgboost").scope_dependent
    assert step_type_detail("evaluate-labels").scope_dependent
    assert not step_type_detail("evaluate-labels").fits
    assert not step_type_detail("speed-angvel").fits


def test_a_detail_served_without_fits_reads_as_not_fitting() -> None:
    # An API older than the field serves no ``fits``; its consumers still parse.
    served = json.loads(step_type_detail("speed-angvel").model_dump_json())
    del served["fits"]

    assert not StepTypeDetail.model_validate(served).fits


def test_a_feature_reports_whether_it_computes_entries_in_parallel() -> None:
    assert step_type_detail("speed-angvel").parallelizable
    assert not step_type_detail("extract-templates").parallelizable
    assert not step_type_detail("tracks").parallelizable


def test_an_op_is_not_parallelizable() -> None:
    assert not step_type_detail("transcode").parallelizable


def test_the_index_row_of_a_feature_reports_parallelizable() -> None:
    rows = {entry.name: entry for entry in catalog_index().entries}

    assert rows["speed-angvel"].parallelizable
    assert not rows["global-kmeans"].parallelizable


def test_a_step_type_names_what_it_consumes() -> None:
    assert step_type_detail("speed-angvel").consumes == ("tracks", "features")
    assert step_type_detail("tracks").consumes == ("nothing",)


def test_the_schema_detail_drops_titles_and_class_docstrings() -> None:
    detail = step_type_detail("global-kmeans", detail="schema")
    schema = detail.params_schema

    assert schema is not None
    assert "title" not in _keys_outside_property_maps(schema)
    assert "save_state" not in json.dumps(schema)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    k = properties["k"]
    assert isinstance(k, dict)
    assert k["description"]


def test_a_property_named_title_survives_compaction() -> None:
    schema: dict[str, JsonValue] = {
        "title": "Params",
        "properties": {"title": {"title": "Title", "type": "string"}},
    }

    assert compact_params_schema(schema) == {
        "properties": {"title": {"type": "string"}}
    }


class _Shapes(BaseModel):
    needed: int
    built: list[int] = Field(default_factory=list)
    plain: int = 3


def test_a_required_field_and_a_factory_default_are_told_apart() -> None:
    rows = {row.path: row for row in param_rows(_Shapes.model_json_schema())}

    assert (rows["needed"].required, rows["needed"].constructed) == (True, False)
    assert (rows["built"].required, rows["built"].constructed) == (False, True)
    assert (rows["plain"].required, rows["plain"].constructed) == (False, False)
    assert rows["plain"].default == 3


def test_an_unknown_name_is_not_found() -> None:
    with pytest.raises(StepTypeNotFound):
        _ = step_type_detail("no-such-step-type")
