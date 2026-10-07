"""A feature's column contract: how its references resolve, and the check before apply.

The contract is two class attributes, ``requires`` and ``provides``. Pipeline
validation reads them from the declaration catalog, and ``run_feature`` checks
``requires`` against each entry's merged frame before ``apply``. These tests
cover the vocabulary itself and the run-time half. The validation half is tested
beside the rest of the pipeline graph.
"""

from __future__ import annotations

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
    EmitsLevel,
    Inputs,
    NON_PAYLOAD_COLS,
    InputStream,
    LoadSpec,
    ParquetLoadSpec,
    TrackInput,
)
from mosaic.runlog import run_log_path
from tests.helpers import add_track_sequences, make_dataset

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
    errors = [
        line
        for line in run_log_path(ds.base_dir, execution_id).read_text().splitlines()
        if '"entry_error"' in line
    ]
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
