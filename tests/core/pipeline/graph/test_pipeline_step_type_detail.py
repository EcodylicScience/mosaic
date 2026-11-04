"""One step type in detail, its params as a flat table.

``step_type_detail`` extends a step type's index row with what it consumes, its
resource class, whether its output depends on its scope, and one row per params
field. Nested models read as dotted paths, and a field that takes a run reference
is typed as a recipe writes one. ``detail="schema"`` adds the params JSON Schema
without titles or class docstrings.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from mosaic.behavior.feature_library import FEATURES
from mosaic.core.json_value import JsonValue
from mosaic.core.params import Declared, DeclaredModel, Params
from mosaic.core.pipeline.graph import (
    StepTypeDetail,
    StepTypeNotFound,
    catalog_index,
    compact_params_schema,
    declaration_catalog,
    editor_catalog,
    param_rows,
    step_type_detail,
)
from mosaic.core.pipeline.graph.param_table import description_text
from mosaic.core.pipeline.types.feature import (
    DependencyLookup,
    EmitsLevel,
    InputStream,
)
from mosaic.core.pipeline.types.inputs import InputsLike


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


def test_the_detail_states_follows_ids() -> None:
    assert step_type_detail("speed-angvel").follows_ids
    assert not step_type_detail("heading").follows_ids
    assert not step_type_detail("tracks").follows_ids
    assert not step_type_detail("infer-pose").follows_ids


class _DeclaresNothing:
    """A feature class with every member of the protocol and no optional declaration."""

    name = "declares-nothing"
    version = "0"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()

    @property
    def inputs(self) -> InputsLike:
        raise NotImplementedError

    @property
    def params(self) -> Params:
        raise NotImplementedError

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        raise NotImplementedError

    def fit(self, inputs: InputStream) -> None:
        raise NotImplementedError

    def save_state(self, run_root: Path) -> None:
        raise NotImplementedError

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@pytest.fixture
def declares_nothing_registered() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(FEATURES, "DeclaresNothing", _DeclaresNothing)
        declaration_catalog.cache_clear()
        editor_catalog.cache_clear()
        yield
    declaration_catalog.cache_clear()
    editor_catalog.cache_clear()


def test_a_feature_that_declares_nothing_follows_ids(
    declares_nothing_registered: None,
) -> None:
    declared = declaration_catalog().get("declares-nothing")

    assert declared is not None
    assert declared.follows_ids is True


def test_a_detail_served_without_follows_ids_reads_as_not_following() -> None:
    served = json.loads(step_type_detail("speed-angvel").model_dump_json())
    del served["follows_ids"]

    assert not StepTypeDetail.model_validate(served).follows_ids


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


class _Inner(DeclaredModel):
    depth: Annotated[int, Declared("how deep the inner model reaches")] = 1


class _WithUnwired(Params):
    used: Annotated[int, Declared("a field some code reads")] = 1
    inert: Annotated[
        _Inner, Declared("a field no code reads", unwired="nothing reads it")
    ] = _Inner()
    kept: Annotated[_Inner, Declared("a model some code reads")] = _Inner()


def test_the_rows_leave_out_an_unwired_field_and_the_rows_of_its_model() -> None:
    schema = _WithUnwired.model_json_schema()

    paths = [row.path for row in param_rows(schema)]

    assert paths == ["used", "kept", "kept.depth"]
    properties = schema["properties"]
    assert properties["inert"]["x-mosaic-unwired"] == "nothing reads it"


def test_the_reference_text_of_an_unwired_field_keeps_its_note() -> None:
    schema = _WithUnwired.model_json_schema()

    text = description_text(schema["properties"]["inert"])

    assert text == "a field no code reads Unwired: nothing reads it."


def test_an_unknown_name_is_not_found() -> None:
    with pytest.raises(StepTypeNotFound):
        _ = step_type_detail("no-such-step-type")
