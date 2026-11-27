"""A feature's column contract: how its references resolve, and the check before apply.

The contract is two class attributes, ``requires`` and ``provides``. Pipeline
validation reads them from the declaration catalog, and ``run_feature`` checks
``requires`` against each entry's merged frame before ``apply``. These tests
cover the vocabulary itself and the run-time half. The validation half is tested
beside the rest of the pipeline graph.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from mosaic.core.helpers import make_entry_key
from mosaic.core.json_value import JsonValue
from mosaic.core.params import Params
from mosaic.core.pipeline.column_contract import (
    AnyOf,
    Column,
    Conditional,
    Derived,
    Expressions,
    Keypoints,
    MissingInputColumns,
    PAYLOAD_LABEL,
    NamedColumns,
    ParamColumn,
    ParamIs,
    Payload,
    Prefix,
    Provides,
    Requirement,
    Template,
    When,
    check_input_columns,
    contract_fields,
    declared_provides,
    declared_requires,
    is_set,
    needs_of,
    resolve_names,
)
from mosaic.core.pipeline.loading import load_entry_data, merge_naming
from mosaic.core.pipeline.run import AllEntriesFailed, run_feature
from mosaic.core.pipeline.types import (
    Inputs,
    NON_PAYLOAD_COLS,
    InputStream,
    LoadSpec,
    ParquetLoadSpec,
    TrackInput,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.runlog import run_log_path
from tests.helpers import add_track_sequences, entry_error_lines, make_dataset

# --- resolving references ---------------------------------------------------------


def test_a_column_and_a_prefix_name_themselves() -> None:
    assert resolve_names(Column("X"), {}) == NamedColumns(columns=("X",))
    assert resolve_names(Prefix("poseX"), {}) == NamedColumns(prefixes=("poseX",))


@pytest.mark.parametrize(
    ("value", "columns"),
    [
        ("speed_smooth", ("speed_smooth",)),
        (["bad_frame", "Focal_fish"], ("bad_frame", "Focal_fish")),
        (None, ()),
        ("", ()),
        ([], ()),
    ],
    ids=["one", "several", "none", "empty", "empty-list"],
)
def test_a_params_field_names_the_columns_its_value_holds(
    value: JsonValue, columns: tuple[str, ...]
) -> None:
    named = resolve_names(ParamColumn("speed_col"), {"speed_col": value})
    assert named.columns == columns
    assert named.params == ("speed_col",)
    assert not named.unknown


@pytest.mark.parametrize(
    "values",
    [{}, {"speed_col": 3}, {"speed_col": ["a", 1]}, {"speed_col": {"step": "x"}}],
    ids=["missing", "number", "mixed-list", "reference"],
)
def test_a_value_that_names_no_column_is_unknown(values: dict[str, JsonValue]) -> None:
    """Unknown and not absent: a recipe mid-edit is not refused for it."""
    assert resolve_names(ParamColumn("speed_col"), values).unknown


def test_an_unset_field_falls_back_to_a_name_or_a_template() -> None:
    literal = ParamColumn("group_size_col", fallback="group_size")
    assert resolve_names(literal, {"group_size_col": ""}).columns == ("group_size",)

    template = ParamColumn("output_column", fallback="{column}_{agg}")
    values: dict[str, JsonValue] = {"output_column": None, "column": "X", "agg": "mean"}
    assert resolve_names(template, values).columns == ("X_mean",)
    assert resolve_names(template, {**values, "output_column": "out"}).columns == (
        "out",
    )
    assert resolve_names(template, {"output_column": None, "agg": "mean"}).unknown


def test_names_are_renamed_then_prefixed() -> None:
    renamed = ParamColumn("fields", renamed_by="field_renames")
    values: dict[str, JsonValue] = {
        "fields": ["focal", "sex"],
        "field_renames": {"focal": "Focal_fish"},
    }
    assert resolve_names(renamed, values).columns == ("Focal_fish", "sex")

    prefixed = ParamColumn("tag_cols", prefix="neighbor_")
    assert resolve_names(prefixed, {"tag_cols": ["bad_frame"]}).columns == (
        "neighbor_bad_frame",
    )


def test_suffixes_name_each_column_once_per_item() -> None:
    ref = ParamColumn("columns", suffixes="aggs")
    values: dict[str, JsonValue] = {
        "columns": ["speed", "angvel"],
        "aggs": ["mean", "max"],
    }

    named = resolve_names(ref, values)

    assert named.columns == ("speed_mean", "speed_max", "angvel_mean", "angvel_max")
    assert named.params == ("columns", "aggs")
    assert resolve_names(ref, {"columns": ["speed"], "aggs": None}).unknown


_SHARES = ParamColumn(
    "columns", suffixes="aggs", families=("share",), members="categories"
)


def test_a_family_suffix_names_one_column_per_member() -> None:
    values: dict[str, JsonValue] = {
        "columns": ["a"],
        "aggs": ["mean", "share"],
        "categories": ["x", "y"],
    }

    named = resolve_names(_SHARES, values)

    assert named.columns == ("a_mean", "a_share_x", "a_share_y")
    assert named.possible_prefixes == ()
    assert named.params == ("columns", "aggs", "categories")
    assert not named.unknown


def test_a_family_expands_in_place_within_each_column() -> None:
    values: dict[str, JsonValue] = {
        "columns": ["a", "b"],
        "aggs": ["share", "mean"],
        "categories": ["x", "y"],
    }

    assert resolve_names(_SHARES, values).columns == (
        "a_share_x",
        "a_share_y",
        "a_mean",
        "b_share_x",
        "b_share_y",
        "b_mean",
    )


def test_a_family_without_members_is_a_possible_prefix() -> None:
    values: dict[str, JsonValue] = {
        "columns": ["a", "b"],
        "aggs": ["mean", "share"],
        "categories": None,
    }

    named = resolve_names(_SHARES, values)

    assert named.columns == ("a_mean", "b_mean")
    assert named.possible_prefixes == ("a_share_", "b_share_")
    assert not named.unknown


@pytest.mark.parametrize(
    "extra",
    [{}, {"categories": [1]}, {"categories": "x"}, {"categories": {"x": 1}}],
    ids=["absent", "not-strings", "string", "mapping"],
)
def test_members_that_cannot_be_read_leave_a_family_unknown(
    extra: dict[str, JsonValue],
) -> None:
    values: dict[str, JsonValue] = {
        "columns": ["a"],
        "aggs": ["mean", "share"],
        **extra,
    }

    assert resolve_names(_SHARES, values).unknown


def test_the_members_are_read_only_when_a_family_suffix_is_listed() -> None:
    values: dict[str, JsonValue] = {"columns": ["a"], "aggs": ["mean"]}

    named = resolve_names(_SHARES, values)

    assert named.columns == ("a_mean",)
    assert named.possible_prefixes == ()
    assert not named.unknown


def test_a_reference_without_families_names_no_possible_prefix() -> None:
    values: dict[str, JsonValue] = {
        "columns": ["a"],
        "aggs": ["mean", "share"],
        "categories": ["x"],
    }

    named = resolve_names(ParamColumn("columns", suffixes="aggs"), values)

    assert named.columns == ("a_mean", "a_share")
    assert named.possible_prefixes == ()


_BINS = ParamColumn("bins", key="column")
"""A field spelled as ``value_bins`` spells its list of mappings."""


def test_a_key_names_the_columns_that_each_mapping_holds() -> None:
    values: dict[str, JsonValue] = {
        "bins": [
            {"column": "speed", "edges": [0, 1]},
            {"column": ["angvel", "accel"], "edges": [2]},
            {"column": "", "edges": []},
            {"column": None, "edges": []},
        ]
    }

    named = resolve_names(_BINS, values)

    assert named.columns == ("speed", "angvel", "accel")
    assert named.params == ("bins",)
    assert not named.unknown


@pytest.mark.parametrize("bins", [None, []], ids=["none", "empty"])
def test_an_unset_keyed_field_names_none_or_its_fallback(bins: JsonValue) -> None:
    values: dict[str, JsonValue] = {"bins": bins}

    assert resolve_names(_BINS, values) == NamedColumns(params=("bins",))
    fallback = ParamColumn("bins", key="column", fallback="speed")
    assert resolve_names(fallback, values).columns == ("speed",)


@pytest.mark.parametrize(
    "bins",
    [
        "speed",
        ["speed"],
        [{"edges": [0]}],
        [{"column": 3}],
        [{"column": ["a", 3]}],
        [{"column": {"a": 1}}],
    ],
    ids=["string", "not-mappings", "key-absent", "number", "mixed-list", "mapping"],
)
def test_a_keyed_field_of_another_shape_is_unknown(bins: JsonValue) -> None:
    assert resolve_names(_BINS, {"bins": bins}).unknown


def test_a_missing_keyed_field_is_unknown() -> None:
    assert resolve_names(_BINS, {}).unknown


def test_a_key_is_renamed_then_prefixed_then_suffixed() -> None:
    ref = ParamColumn(
        "bins", key="column", prefix="p_", renamed_by="renames", suffix="_bin"
    )
    values: dict[str, JsonValue] = {
        "bins": [{"column": ["a", "b"]}],
        "renames": {"a": "z"},
    }

    assert resolve_names(ref, values).columns == ("p_z_bin", "p_b_bin")


def test_a_constant_suffix_ends_each_name() -> None:
    ref = ParamColumn("columns", suffix="_bin")

    named = resolve_names(ref, {"columns": ["speed", "angvel"]})

    assert named.columns == ("speed_bin", "angvel_bin")
    assert named.possible_prefixes == ()


def test_a_constant_suffix_precedes_the_suffixes_field() -> None:
    ref = ParamColumn("columns", suffix="_bin", suffixes="aggs")
    values: dict[str, JsonValue] = {"columns": ["a"], "aggs": ["mean", "max"]}

    assert resolve_names(ref, values).columns == ("a_bin_mean", "a_bin_max")


def test_a_constant_suffix_does_not_name_an_unset_field() -> None:
    ref = ParamColumn("columns", suffix="_bin")

    assert resolve_names(ref, {"columns": []}) == NamedColumns(params=("columns",))


def test_an_open_family_is_a_possible_prefix_beside_a_closed_one() -> None:
    ref = ParamColumn("columns", suffixes="aggs", open_families=("q",))
    values: dict[str, JsonValue] = {"columns": ["a", "b"], "aggs": ["mean", "q", "max"]}

    named = resolve_names(ref, values)

    assert named.columns == ("a_mean", "a_max", "b_mean", "b_max")
    assert named.possible_prefixes == ("a_q_", "b_q_")
    assert named.params == ("columns", "aggs")
    assert not named.unknown


def test_an_open_family_does_not_read_its_members() -> None:
    """The member texts of ``quantile`` are formatted floats, so no list names them."""
    ref = ParamColumn(
        "columns",
        suffixes="aggs",
        families=("share",),
        members="categories",
        open_families=("q",),
    )
    values: dict[str, JsonValue] = {"columns": ["a"], "aggs": ["q"]}

    named = resolve_names(ref, values)

    assert named.possible_prefixes == ("a_q_",)
    assert named.columns == ()
    assert not named.unknown


def test_an_open_family_and_a_member_family_name_their_own_columns() -> None:
    ref = ParamColumn(
        "columns",
        suffixes="aggs",
        families=("share",),
        members="categories",
        open_families=("q",),
    )
    values: dict[str, JsonValue] = {
        "columns": ["a"],
        "aggs": ["share", "q"],
        "categories": ["x"],
    }

    named = resolve_names(ref, values)

    assert named.columns == ("a_share_x",)
    assert named.possible_prefixes == ("a_q_",)


def test_a_family_listed_as_open_ignores_its_closed_declaration() -> None:
    ref = ParamColumn(
        "columns",
        suffixes="aggs",
        families=("share",),
        members="categories",
        open_families=("share",),
    )
    values: dict[str, JsonValue] = {"columns": ["a"], "aggs": ["share"]}

    named = resolve_names(ref, values)

    assert named.possible_prefixes == ("a_share_",)
    assert not named.unknown


def test_a_constant_suffix_in_open_families_is_a_possible_prefix_of_each_name() -> None:
    ref = ParamColumn("columns", suffix="_q", open_families=("_q",))

    named = resolve_names(ref, {"columns": ["a", "b"]})

    assert named.columns == ()
    assert named.possible_prefixes == ("a_q", "b_q")
    assert named.met_by({"a_q0.5"})
    assert not named.met_by({"a"})


def test_a_key_with_a_constant_suffix_in_open_families() -> None:
    ref = ParamColumn("rules", key="columns", suffix="_q", open_families=("_q",))
    values: dict[str, JsonValue] = {"rules": [{"columns": ["a"]}, {"columns": "b"}]}

    assert resolve_names(ref, values).possible_prefixes == ("a_q", "b_q")


def test_possible_prefixes_meet_a_requirement_and_survive_an_any_of_group() -> None:
    assert NamedColumns(possible_prefixes=("a_share_",)).met_by({"a_share_x"})
    assert not NamedColumns(possible_prefixes=("a_share_",)).met_by({"a_mean"})

    values: dict[str, JsonValue] = {
        "columns": ["a"],
        "aggs": ["share"],
        "categories": None,
    }
    (need,) = needs_of((AnyOf((_SHARES, Column("other"))),), values)

    assert need.possible_prefixes == ("a_share_",)
    assert need.columns == ("other",)
    assert need.met_by({"a_share_x"})


def test_an_unset_field_may_mean_the_data_decides() -> None:
    """``id-tag-columns`` attaches every label field for ``None``, and none for ``[]``."""
    ref = ParamColumn("fields", unset="unknown")
    assert resolve_names(ref, {"fields": None}).unknown
    assert resolve_names(ref, {"fields": []}) == NamedColumns(params=("fields",))
    assert resolve_names(ref, {"fields": ["focal"]}).columns == ("focal",)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), (False, False), ("", False), ([], False), (0, True), (5, True)],
)
def test_what_counts_as_a_set_param(value: JsonValue, expected: bool) -> None:
    assert is_set(value) is expected


# --- requirements -------------------------------------------------------------------


def test_any_of_resolves_to_the_union_of_its_members() -> None:
    requires: tuple[Requirement, ...] = (
        AnyOf((ParamColumn("speed_col"), Column("speed"))),
        Prefix("poseX"),
    )
    needs = needs_of(requires, {"speed_col": "speed_smooth"})

    assert [need.columns for need in needs] == [("speed_smooth", "speed"), ()]
    assert needs[0].label == "'speed_smooth' or 'speed'"
    assert needs[1].label == "'poseX*'"
    assert needs[0].met_by({"speed"})
    assert needs[1].met_by({"poseX0", "poseY0"})
    assert not needs[1].met_by({"X"})


def test_a_requirement_on_an_unset_field_is_left_out() -> None:
    """An unset ``subgroup_col`` does not name a column, and is not required."""
    assert needs_of((ParamColumn("subgroup_col"),), {"subgroup_col": None}) == ()


def test_a_list_param_requirement_needs_every_column_it_names() -> None:
    """One column of ``["strain", "tag"]`` does not meet a requirement for both."""
    needs = needs_of(
        (ParamColumn("subgroup_col"),), {"subgroup_col": ["strain", "tag"]}
    )

    assert needs == (
        NamedColumns(columns=("strain",), params=("subgroup_col",)),
        NamedColumns(columns=("tag",), params=("subgroup_col",)),
    )
    assert [need.met_by({"strain"}) for need in needs] == [True, False]
    assert all(need.met_by({"strain", "tag"}) for need in needs)


def test_a_string_param_requirement_needs_its_one_column() -> None:
    needs = needs_of((ParamColumn("speed_col"),), {"speed_col": "speed"})

    assert needs == (NamedColumns(columns=("speed",), params=("speed_col",)),)


def test_a_repeated_name_in_a_list_param_is_one_requirement() -> None:
    needs = needs_of((ParamColumn("columns"),), {"columns": ["a", "a", "b"]})

    assert [need.columns for need in needs] == [("a",), ("b",)]


def test_an_unreadable_list_param_stays_one_unknown_requirement() -> None:
    needs = needs_of((ParamColumn("columns"),), {"columns": ["a", 3]})

    assert needs == (NamedColumns(params=("columns",), unknown=True),)


def test_a_conditional_list_param_requirement_needs_every_column() -> None:
    gated = When(ParamColumn("columns"), (ParamIs("aggs", equals="mean"),))
    values: dict[str, JsonValue] = {"columns": ["a", "b"], "aggs": ["mean"]}

    assert [need.columns for need in needs_of((gated,), values)] == [("a",), ("b",)]
    assert needs_of((gated,), {**values, "aggs": ["max"]}) == ()
    (unknown_a, unknown_b) = needs_of((gated,), {"columns": ["a", "b"]})
    assert unknown_a.unknown
    assert unknown_b.unknown


def test_an_any_of_group_of_a_list_param_is_met_by_any_one_column() -> None:
    """``AnyOf`` is the one way to ask for any one of the columns a list names."""
    (need,) = needs_of((AnyOf((ParamColumn("cols"),)),), {"cols": ["a", "b"]})

    assert need.columns == ("a", "b")
    assert need.met_by({"b"})
    assert not need.met_by({"c"})


def test_a_derivation_that_writes_one_column_of_a_list_leaves_the_rest_required() -> (
    None
):
    requires: tuple[Requirement, ...] = (_SPECS, ParamColumn("columns"))
    values: dict[str, JsonValue] = {
        "specs": [{"new_col": "a", "source_col": "x"}],
        "columns": ["a", "b"],
    }

    needs = needs_of(requires, values)

    assert [need.columns for need in needs] == [("x",), ("b",)]


def test_a_list_param_requirement_leaves_out_its_possible_prefixes() -> None:
    """A family that a non-empty output may lack is never a hard requirement."""
    ref = ParamColumn("columns", suffixes="aggs", families=("q",), members="members")
    values: dict[str, JsonValue] = {
        "columns": ["a", "b"],
        "aggs": ["mean", "q"],
        "members": None,
    }

    needs = needs_of((ref,), values)

    assert [need.columns for need in needs] == [("a_mean",), ("b_mean",)]
    assert all(not need.possible_prefixes for need in needs)


def test_a_prefix_and_a_family_template_stay_one_requirement_each() -> None:
    requires: tuple[Requirement, ...] = (
        Prefix("poseX"),
        Template("{head}_", family=True),
    )

    needs = needs_of(requires, {"head": "nose"})

    assert needs == (
        NamedColumns(prefixes=("poseX",)),
        NamedColumns(prefixes=("nose_",), params=("head",)),
    )


_SPECS = Derived("specs", name_key="new_col", source_key="source_col")
"""A derivation spelled as ``nn-delta-bins`` spells its ``category_specs``."""


def test_a_derivation_requires_what_it_reads_and_meets_what_follows_it() -> None:
    """``apply`` reads a requirement listed before the derivation, and then derives."""
    requires: tuple[Requirement, ...] = (
        Column("fast"),
        _SPECS,
        AnyOf((ParamColumn("focal_col"), Column("neighbor_fast"))),
    )
    values: dict[str, JsonValue] = {
        "focal_col": "fast",
        "specs": [
            {"source_col": "speed", "new_col": "fast"},
            {"source_col": "fast", "new_col": "faster"},
            {"source_col": "", "new_col": "skipped"},
        ],
    }

    needs = needs_of(requires, values)

    assert needs == (
        NamedColumns(columns=("fast",)),
        NamedColumns(columns=("speed",), params=("specs",)),
    )


@pytest.mark.parametrize(
    "specs",
    [None, [], [{"new_col": "fast"}]],
    ids=["none", "empty", "no-source"],
)
def test_a_derivation_without_mappings_leaves_what_follows_it_required(
    specs: JsonValue,
) -> None:
    """``apply`` skips a mapping without a source, and writes nothing for it."""
    requires: tuple[Requirement, ...] = (_SPECS, Column("fast"))

    assert needs_of(requires, {"specs": specs}) == (NamedColumns(columns=("fast",)),)


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"specs": "speed"},
        {"specs": ["speed"]},
        {"specs": [{"source_col": 3, "new_col": "fast"}]},
    ],
    ids=["missing", "not-a-list", "not-a-mapping", "not-a-name"],
)
def test_a_derivation_that_cannot_be_read_leaves_what_follows_it_unknown(
    values: dict[str, JsonValue],
) -> None:
    """It may write any column. The check passes what follows it, and validation cautions."""
    requires: tuple[Requirement, ...] = (Column("X"), _SPECS, Column("fast"))

    needs = needs_of(requires, values)

    assert needs == (
        NamedColumns(columns=("X",)),
        NamedColumns(params=("specs",), unknown=True),
        NamedColumns(columns=("fast",), unknown=True),
    )


_EXPRESSIONS = Expressions("exprs", name_key="name", expr_key="expr")
"""A list of expressions spelled as ``derive-columns`` spells its ``columns``."""


def test_an_expression_requires_what_it_reads_unless_an_earlier_one_writes_it() -> None:
    values: dict[str, JsonValue] = {
        "exprs": [
            {"name": "fast", "expr": "speed * 2 + offset"},
            {"name": "faster", "expr": "fast + speed + sqrt(accel)"},
        ]
    }

    needs = needs_of((_EXPRESSIONS,), values)

    assert needs == (
        NamedColumns(columns=("speed",), params=("exprs",)),
        NamedColumns(columns=("offset",), params=("exprs",)),
        NamedColumns(columns=("accel",), params=("exprs",)),
    )


def test_an_expression_reading_a_later_column_requires_it() -> None:
    """``apply`` writes in list order, so a forward reference reads the input."""
    values: dict[str, JsonValue] = {
        "exprs": [
            {"name": "faster", "expr": "fast * 2"},
            {"name": "fast", "expr": "speed"},
        ]
    }

    needs = needs_of((_EXPRESSIONS,), values)

    assert needs == (
        NamedColumns(columns=("fast",), params=("exprs",)),
        NamedColumns(columns=("speed",), params=("exprs",)),
    )


def test_an_expression_reading_its_own_name_requires_the_input_column() -> None:
    values: dict[str, JsonValue] = {"exprs": [{"name": "x", "expr": "x * 2"}]}

    assert needs_of((_EXPRESSIONS,), values) == (
        NamedColumns(columns=("x",), params=("exprs",)),
    )


def test_an_expression_meets_what_follows_it_and_not_what_precedes_it() -> None:
    requires: tuple[Requirement, ...] = (Column("fast"), _EXPRESSIONS, Column("slow"))
    values: dict[str, JsonValue] = {
        "exprs": [{"name": "slow", "expr": "speed"}, {"name": "fast", "expr": "speed"}]
    }

    needs = needs_of(requires, values)

    assert needs == (
        NamedColumns(columns=("fast",)),
        NamedColumns(columns=("speed",), params=("exprs",)),
    )


def test_a_mapping_without_a_name_or_an_expression_writes_nothing() -> None:
    requires: tuple[Requirement, ...] = (_EXPRESSIONS, Column("fast"))
    values: dict[str, JsonValue] = {
        "exprs": [{"name": "fast", "expr": ""}, {"name": "", "expr": "speed"}]
    }

    assert needs_of(requires, values) == (NamedColumns(columns=("fast",)),)
    assert needs_of(requires, {"exprs": None}) == (NamedColumns(columns=("fast",)),)


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"exprs": "speed"},
        {"exprs": ["speed"]},
        {"exprs": [{"name": 3, "expr": "speed"}]},
        {"exprs": [{"name": "fast", "expr": 3}]},
        {"exprs": [{"name": "fast", "expr": "speed +"}]},
        {"exprs": [{"name": "fast", "expr": "speed.real"}]},
    ],
    ids=[
        "missing",
        "not-a-list",
        "not-a-mapping",
        "name-not-text",
        "expression-not-text",
        "does-not-parse",
        "refused-node",
    ],
)
def test_an_expression_list_that_cannot_be_read_leaves_what_follows_it_unknown(
    values: dict[str, JsonValue],
) -> None:
    requires: tuple[Requirement, ...] = (Column("X"), _EXPRESSIONS, Column("fast"))

    needs = needs_of(requires, values)

    assert needs == (
        NamedColumns(columns=("X",)),
        NamedColumns(params=("exprs",), unknown=True),
        NamedColumns(columns=("fast",), unknown=True),
    )


def test_contract_fields_list_the_field_of_an_expression_list() -> None:
    assert contract_fields((_EXPRESSIONS,), None) == {"exprs"}
    assert contract_fields((_BINS,), None) == {"bins"}


def test_the_expression_kind_joins_the_requirement_union() -> None:
    adapter: TypeAdapter[tuple[Requirement, ...]] = TypeAdapter(tuple[Requirement, ...])

    dumped = adapter.dump_python((_EXPRESSIONS,), mode="json")

    assert dumped == [
        {
            "field": "exprs",
            "name_key": "name",
            "expr_key": "expr",
            "kind": "expressions",
        }
    ]
    assert adapter.validate_python(dumped) == (_EXPRESSIONS,)


def test_a_stored_contract_without_the_new_fields_validates() -> None:
    """A sidecar written before ``key``, ``suffix`` and ``open_families`` still loads."""
    adapter: TypeAdapter[tuple[Requirement, ...]] = TypeAdapter(tuple[Requirement, ...])
    stored: list[JsonValue] = [
        {
            "field": "columns",
            "fallback": "",
            "prefix": "",
            "renamed_by": "",
            "unset": "absent",
            "suffixes": "aggs",
            "families": ["share"],
            "members": "categories",
            "kind": "param",
        },
        {"of": [{"name": "x", "kind": "column"}], "kind": "any_of"},
    ]
    provides = TypeAdapter(Provides)

    (requirement, _) = adapter.validate_python(stored)
    read = provides.validate_python(
        {"adds": [stored[0]], "conditional": [], "passthrough": False}
    )

    assert requirement == ParamColumn(
        "columns", suffixes="aggs", families=("share",), members="categories"
    )
    assert read.adds == (requirement,)


def test_a_template_names_a_column_from_the_params() -> None:
    """``pair-facing`` names its head keypoint from a prefix and an index."""
    values: dict[str, JsonValue] = {
        "x_prefix": "kpX",
        "pose_head_index": 0,
        "pose": {"x_prefix": "bodyX"},
    }

    assert resolve_names(Template("{x_prefix}{pose_head_index}"), values) == (
        NamedColumns(columns=("kpX0",), params=("x_prefix", "pose_head_index"))
    )
    assert resolve_names(Template("{pose.x_prefix}", family=True), values) == (
        NamedColumns(prefixes=("bodyX",), params=("pose",))
    )


@pytest.mark.parametrize(
    "values",
    [{"x_prefix": "kpX"}, {"x_prefix": "kpX", "pose_head_index": None}, {"pose": 3}],
    ids=["missing", "unset", "not-a-model"],
)
def test_a_template_over_a_value_that_is_not_there_is_unknown(
    values: dict[str, JsonValue],
) -> None:
    for template in ("{x_prefix}{pose_head_index}", "{pose.x_prefix}"):
        assert resolve_names(Template(template), values).unknown


def test_a_keypoint_selection_requires_each_keypoint_that_it_selects() -> None:
    selected: dict[str, JsonValue] = {
        "pose_indices": [3, 5],
        "x_prefix": "kpX",
        "y_prefix": "kpY",
    }

    needs = needs_of((Keypoints(),), {"pose": selected})
    defaults = needs_of((Keypoints(),), {"pose": {"pose_n": 2}})

    assert [need.columns for need in needs] == [
        ("kpX3",),
        ("kpY3",),
        ("kpX5",),
        ("kpY5",),
    ]
    assert [need.columns for need in defaults] == [
        ("poseX0",),
        ("poseY0",),
        ("poseX1",),
        ("poseY1",),
    ]


@pytest.mark.parametrize(
    "values", [{}, {"pose": None}, {"pose": {"pose_n": "many"}}], ids=str
)
def test_a_keypoint_selection_that_cannot_be_read_is_unknown(
    values: dict[str, JsonValue],
) -> None:
    assert needs_of((Keypoints(),), values) == (
        NamedColumns(params=("pose",), unknown=True),
    )


_GATED_ANGLE = When(Column("ANGLE"), (ParamIs("orientation_gate_cos"),))
"""``approach-avoidance`` reads ``ANGLE`` while its orientation gate is set."""


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"orientation_gate_cos": 0.866}, (NamedColumns(columns=("ANGLE",)),)),
        ({"orientation_gate_cos": None}, ()),
        ({}, (NamedColumns(columns=("ANGLE",), unknown=True),)),
    ],
    ids=["set", "unset", "missing"],
)
def test_a_conditional_requirement_applies_while_its_condition_holds(
    values: dict[str, JsonValue], expected: tuple[NamedColumns, ...]
) -> None:
    assert needs_of((_GATED_ANGLE,), values) == expected


def test_a_condition_tests_a_value_or_its_absence() -> None:
    by_value = When(Column("ANGLE"), (ParamIs("heading_source", equals="orientation"),))
    unless_scaled = When(Column("group"), (ParamIs("cm_per_pixel", negated=True),))

    assert needs_of((by_value,), {"heading_source": "orientation"})
    assert not needs_of((by_value,), {"heading_source": "auto"})
    assert needs_of((unless_scaled,), {"cm_per_pixel": None})
    assert not needs_of((unless_scaled,), {"cm_per_pixel": 0.1})


def test_a_condition_on_a_list_holds_while_the_list_contains_the_value() -> None:
    needs_time = When(Column("time"), (ParamIs("aggs", equals="duration"),))

    assert needs_of((needs_time,), {"aggs": ["mean", "duration"]}) == (
        NamedColumns(columns=("time",)),
    )
    assert not needs_of((needs_time,), {"aggs": ["mean"]})
    assert not needs_of((needs_time,), {"aggs": []})
    assert needs_of((needs_time,), {"aggs": "duration"})
    assert needs_of((needs_time,), {}) == (
        NamedColumns(columns=("time",), unknown=True),
    )


def test_a_negated_condition_on_a_list_holds_while_the_value_is_absent() -> None:
    unless_duration = When(
        Column("time"), (ParamIs("aggs", equals="duration", negated=True),)
    )

    assert not needs_of((unless_duration,), {"aggs": ["mean", "duration"]})
    assert needs_of((unless_duration,), {"aggs": ["mean"]})


def test_every_condition_must_hold() -> None:
    head = When(
        Template("poseX{pose_head_index}"),
        (ParamIs("use_pixel_coords"), ParamIs("pose_head_index")),
    )

    assert needs_of((head,), {"use_pixel_coords": True, "pose_head_index": 0}) == (
        NamedColumns(columns=("poseX0",), params=("pose_head_index",)),
    )
    assert not needs_of((head,), {"use_pixel_coords": False, "pose_head_index": 0})
    assert not needs_of((head,), {"use_pixel_coords": True, "pose_head_index": None})


def test_contract_fields_include_the_fields_that_conditions_and_selections_read() -> (
    None
):
    requires: tuple[Requirement, ...] = (
        _GATED_ANGLE,
        Keypoints(),
        Template("{x_prefix}{pose.pose_n}"),
    )
    assert contract_fields(requires, None) == {
        "orientation_gate_cos",
        "pose",
        "x_prefix",
    }


def test_contract_fields_name_the_members_field_and_not_the_family_suffixes() -> None:
    provides = Provides(adds=(_SHARES,))

    assert contract_fields((), provides) == {"columns", "aggs", "categories"}


def test_contract_fields_lists_every_params_field_the_declarations_read() -> None:
    requires: tuple[Requirement, ...] = (AnyOf((ParamColumn("speed_col"),)), _SPECS)
    provides = Provides(
        adds=(ParamColumn("output_column", fallback="{column}_{agg}"),),
        conditional=(
            Conditional(Column("dx_back"), when_param="emit_backward"),
            Conditional(
                ParamColumn("tags", renamed_by="renames"),
                when_present=ParamColumn("focal_col"),
            ),
        ),
    )
    assert contract_fields(requires, provides) == {
        "speed_col",
        "specs",
        "output_column",
        "column",
        "agg",
        "emit_backward",
        "tags",
        "renames",
        "focal_col",
    }


def test_an_absent_declaration_reads_as_undeclared_and_a_wrong_one_raises() -> None:
    class Undeclared:
        pass

    class Malformed:
        requires: ClassVar[object] = ("X",)
        provides: ClassVar[object] = ("X",)

    assert declared_requires(Undeclared) is None
    assert declared_provides(Undeclared) is None
    with pytest.raises(TypeError, match="Malformed.requires"):
        _ = declared_requires(Malformed)
    with pytest.raises(TypeError, match="Malformed.provides"):
        _ = declared_provides(Malformed)


# --- the merge naming rule ------------------------------------------------------------


def test_the_naming_rule_is_what_the_merge_does(tmp_path: Path) -> None:
    """Validation names a merged input's columns by the rule the merge applies."""
    frame = np.arange(4)
    first = pd.DataFrame(
        {"frame": frame, "id": 0, "sequence": "s", "X": 1.0, "speed": 2.0}
    )
    second = pd.DataFrame({"frame": frame, "id": 0, "sequence": "s", "speed": 3.0})
    third = pd.DataFrame({"frame": frame, "id": 0, "speed": 4.0, "nn_id": 1.0})
    specs: list[tuple[Path, LoadSpec]] = []
    for position, table in enumerate((first, second, third)):
        path = tmp_path / f"{position}.parquet"
        table.to_parquet(path)
        specs.append((path, ParquetLoadSpec()))

    merged = load_entry_data(specs)
    assert merged is not None

    columns: frozenset[str] = frozenset(first.columns)
    for position, table in ((1, second), (2, third)):
        columns = merge_naming(columns, list(table.columns), position).columns
    assert columns == set(merged.columns)
    assert {"speed", "speed__1", "speed__2"} <= columns
    assert "sequence__1" not in columns


# --- the check before apply -------------------------------------------------------------


class _Undeclared:
    """A stand-in feature without column requirements."""

    name = "test-reads-two"
    version = "0.1"
    category = "per-frame"
    emits: EmitsLevel = "individual"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        other_col: str = "feat_b"

    def __init__(
        self,
        inputs: _Undeclared.Inputs | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs: _Undeclared.Inputs = inputs or self.Inputs(("tracks",))
        self.params: _Undeclared.Params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        return None

    def save_state(self, run_root: Path) -> None:
        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"frame": df["frame"], "id": df["id"], "out": 1.0})


class _ReadsTwoColumns(_Undeclared):
    """The stand-in, reading one named column and one that its params name."""

    requires: ClassVar[tuple[Requirement, ...]] = (
        Column("feat_a"),
        AnyOf((ParamColumn("other_col"), Column("feat_c"))),
    )


def test_a_frame_meeting_every_requirement_passes() -> None:
    check_input_columns(_ReadsTwoColumns(), ["frame", "feat_a", "feat_b"])
    check_input_columns(_ReadsTwoColumns(), ["feat_a", "feat_c"])
    check_input_columns(
        _ReadsTwoColumns(params={"other_col": "elsewhere"}), ["feat_a", "elsewhere"]
    )


def test_a_missing_column_raises_one_message_naming_it_and_the_inputs() -> None:
    with pytest.raises(MissingInputColumns) as raised:
        check_input_columns(_ReadsTwoColumns(), ["frame", "id"])

    assert raised.value.feature == "test-reads-two"
    assert raised.value.inputs == ("tracks",)
    assert [need.label for need in raised.value.missing] == [
        "'feat_a'",
        "'feat_b' or 'feat_c'",
    ]
    assert str(raised.value) == (
        "test-reads-two cannot read its input. The frame merged from tracks lacks "
        "'feat_a'; 'feat_b' or 'feat_c'"
    )


def test_a_missing_column_survives_pickle_with_its_message_and_attributes() -> None:
    """A process worker raises it, and the executor unpickles it in the parent."""
    with pytest.raises(MissingInputColumns) as raised:
        check_input_columns(_ReadsTwoColumns(), ["frame", "id"])

    crossed = pickle.loads(pickle.dumps(raised.value))

    assert type(crossed) is MissingInputColumns
    assert str(crossed) == str(raised.value)
    assert crossed.feature == raised.value.feature
    assert crossed.missing == raised.value.missing
    assert crossed.inputs == raised.value.inputs


def test_an_undeclared_feature_is_not_checked() -> None:
    check_input_columns(_Undeclared(), [])


def test_run_feature_fails_the_entry_that_lacks_a_column(tmp_path: Path) -> None:
    """The check is the per-entry failure path: the entry is lost, the rest kept."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("g", "has"), ("g", "lacks"))
    path = ds.get_root("tracks") / "g__has.parquet"
    table = pd.read_parquet(path)
    table["feat_b"] = 1.0
    table.to_parquet(path)
    execution_id = "TESTCOLUMNS1"

    result = run_feature(ds, _ReadsTwoColumns(), execution_id=execution_id)

    assert result.failed_entries == (make_entry_key("g", "lacks"),)
    errors = entry_error_lines(ds, execution_id)
    assert len(errors) == 1
    assert "MissingInputColumns" in errors[0]
    assert "lacks 'feat_b' or 'feat_c'" in errors[0]


# --- a payload requirement --------------------------------------------------------------


class _ReadsPayload(_Undeclared):
    """The stand-in, reading every numeric payload column, as a global model does."""

    requires: ClassVar[tuple[Requirement, ...]] = (Column("frame"), Payload())


def test_a_payload_requirement_is_met_by_any_measurement() -> None:
    (need,) = needs_of((Payload(),), {})

    assert need.payload
    assert need.label == PAYLOAD_LABEL
    assert need.met_by({"frame", "id", "speed"})
    assert need.met_by({"poseX0"})
    assert not need.met_by(NON_PAYLOAD_COLS)


@pytest.mark.parametrize(
    "name", ["identity_score", "identity_margin", "identity_source_id"]
)
def test_an_identity_assignment_column_is_not_a_measurement(name: str) -> None:
    (need,) = needs_of((Payload(),), {})

    assert not need.met_by({"frame", "id", name})


def test_a_payload_requirement_is_tagged_in_the_closed_set() -> None:
    adapter = TypeAdapter(tuple[Requirement, ...])

    dumped = adapter.dump_python((Payload(), Column("frame")), mode="json")

    assert dumped == [{"kind": "payload"}, {"kind": "column", "name": "frame"}]
    assert adapter.validate_python(dumped) == (Payload(), Column("frame"))
    assert declared_requires(_ReadsPayload) == (Column("frame"), Payload())
    assert contract_fields((Payload(),), None) == frozenset()


def test_the_check_before_apply_reads_only_a_numeric_payload() -> None:
    check_input_columns(
        _ReadsPayload(), ["frame", "id", "speed"], numeric=["frame", "id", "speed"]
    )

    with pytest.raises(MissingInputColumns) as raised:
        check_input_columns(
            _ReadsPayload(), ["frame", "id", "trial"], numeric=["frame", "id"]
        )
    assert str(raised.value).endswith(f"lacks {PAYLOAD_LABEL}")
    with pytest.raises(MissingInputColumns):
        check_input_columns(_ReadsPayload(), ["frame", "id", "fps", "label"])


def test_run_feature_fails_the_entry_without_a_numeric_payload(tmp_path: Path) -> None:
    """A column of text is not a payload, and the check reads the frame's dtypes."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("g", "has"), ("g", "lacks"))
    path = ds.get_root("tracks") / "g__lacks.parquet"
    table = pd.read_parquet(path)
    identity = ["frame", "time", "id", "group", "sequence"]
    textual = table[identity].assign(trial="a")
    textual.to_parquet(path)
    execution_id = "TESTCOLUMNS3"

    result = run_feature(ds, _ReadsPayload(), execution_id=execution_id)

    assert result.failed_entries == (make_entry_key("g", "lacks"),)
    log = run_log_path(ds.base_dir, execution_id).read_text()
    assert f"lacks {PAYLOAD_LABEL}" in log


def test_a_library_feature_reports_its_missing_neighbor_column(tmp_path: Path) -> None:
    """``nn-delta-response`` over tracks alone, with nothing to name a neighbor."""
    from mosaic.core.pipeline.graph import build_feature

    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("g", "s"), n_ids=2)
    execution_id = "TESTCOLUMNS2"

    with pytest.raises(AllEntriesFailed):
        _ = run_feature(
            ds,
            build_feature("nn-delta-response", ["tracks"], None),
            execution_id=execution_id,
        )

    log = run_log_path(ds.base_dir, execution_id).read_text()
    assert "nn-delta-response cannot read its input" in log
    assert "'nn_id' or 'nn_fishID'" in log


def test_nn_delta_bins_reads_a_category_column_that_its_specs_write(
    tmp_path: Path,
) -> None:
    """A spec writes the focal category from ``speed_smooth``, which is then required.

    Neither category column is in the input. The entry with ``speed_smooth`` writes
    rows, and the entry without it fails the check.
    """
    from mosaic.core.pipeline.graph import build_feature

    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("g", "has"), ("g", "lacks"), n_ids=2)
    noise = np.random.default_rng(0)
    for sequence in ("has", "lacks"):
        path = ds.get_root("tracks") / f"g__{sequence}.parquet"
        table = pd.read_parquet(path)
        for column in ("neighbor_x", "neighbor_y", "dangle", "dspeed"):
            table[column] = noise.normal(0.0, 1.0, len(table))
        if sequence == "has":
            table["speed_smooth"] = noise.uniform(0.0, 1.0, len(table))
        table.to_parquet(path)
    params: dict[str, JsonValue] = {
        "focal_category_col": "fast",
        "neighbor_category_col": "neighbor_fast",
        "category_specs": [
            {"source_col": "speed_smooth", "new_col": "fast", "quantile": 0.5}
        ],
    }
    execution_id = "TESTCOLUMNS3"

    result = run_feature(
        ds,
        build_feature("nn-delta-bins", ["tracks"], params),
        execution_id=execution_id,
    )

    assert result.failed_entries == (make_entry_key("g", "lacks"),)
    assert result.run_id is not None
    run_root = ds.get_root("features") / "nn-delta-bins__from__tracks" / result.run_id
    assert not pd.read_parquet(run_root / "g__has.parquet").empty
    log = run_log_path(ds.base_dir, execution_id).read_text()
    assert "lacks 'speed_smooth'" in log
