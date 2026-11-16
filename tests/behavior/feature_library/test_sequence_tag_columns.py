"""``sequence-tag-columns`` attaches the sequence tags of the dataset's tag table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.helpers import identity_columns
from mosaic.behavior.feature_library.sequence_tag_columns import SequenceTagColumns
from mosaic.core.annotations.tag_table import TagRead, TagTable
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import plan_pipeline, read_recipe
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.storage_names import derive_storage_name
from mosaic.core.pipeline.tag_reads import tag_read_problems
from mosaic.core.pipeline.types import DependencyLookup, TrackInputs
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.typed_attribute import TypedAttributeType, TypedAttributeValue
from tests.helpers import (
    add_track_sequences,
    add_tracks_variant,
    make_dataset,
    tag_table,
    write_tags,
)

PRESENT: Entry = ("g", "s1")
EMPTY: Entry = ("g", "s2")
ABSENT: Entry = ("g", "s3")

TYPES: dict[str, TypedAttributeType] = {
    "flag": "bool",
    "count": "int",
    "ratio": "float",
    "note": "text",
    "kind": "categorical",
    "marked": "label",
}

ASSIGNED: dict[str, TypedAttributeValue] = {
    "flag": True,
    "count": 3,
    "ratio": 1.5,
    "note": "windy",
    "kind": "treated",
    "marked": None,
}


def _feature(
    fields: list[str], renames: dict[str, str] | None = None
) -> SequenceTagColumns:
    params: dict[str, object] = {"fields": fields}
    if renames is not None:
        params["field_renames"] = renames
    return SequenceTagColumns(params=params)


def _table() -> TagTable:
    """Every type assigned on ``PRESENT``, none on ``EMPTY``, no row for ``ABSENT``."""
    return tag_table({PRESENT: ASSIGNED, EMPTY: {}}, TYPES)


def _tracks(entry: Entry, ids: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    """Two frames of each id, with a column the feature does not carry."""
    rows = [
        {
            "frame": frame,
            "time": frame / 30.0,
            "id": identity,
            "group": entry[0],
            "sequence": entry[1],
            "X": float(frame + identity),
        }
        for frame in range(2)
        for identity in ids
    ]
    return pd.DataFrame(rows)


def _loaded(ds: Dataset, table: TagTable, feature: SequenceTagColumns) -> None:
    """Save *table* and load *feature* from the revision, for every entry."""
    revision = write_tags(ds, table)
    lookup: DependencyLookup = {
        entry: revision.path for entry in (PRESENT, EMPTY, ABSENT)
    }
    assert feature.load_state(Path(), {}, {"_tags": lookup})


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    return make_dataset(tmp_path / "ds")


def _applied(
    ds: Dataset, entry: Entry, fields: list[str], renames: dict[str, str] | None = None
) -> pd.DataFrame:
    feature = _feature(fields, renames)
    _loaded(ds, _table(), feature)
    return feature.apply(_tracks(entry))


def test_the_feature_declares_what_it_reads_and_writes() -> None:
    feature = _feature(["flag", "kind"], {"kind": "Kind"})

    assert feature.tag_read() == TagRead(
        sequence_tags=("flag", "kind"),
        writes=frozenset({"group", "sequence", "id", "frame", "time"}),
        renames=(("kind", "Kind"),),
    )
    assert type(feature).follows_ids is False


def test_each_type_reads_as_its_column_dtype(ds: Dataset) -> None:
    out = _applied(ds, PRESENT, list(TYPES))

    assert str(out["flag"].dtype) == "boolean"
    assert str(out["count"].dtype) == "Int64"
    assert out["ratio"].dtype == np.float64
    assert out["note"].dtype == object
    assert out["kind"].dtype == object
    assert out["marked"].dtype == np.bool_
    assert out["flag"].tolist() == [True] * 6
    assert out["count"].tolist() == [3] * 6
    assert out["ratio"].tolist() == [1.5] * 6
    assert out["note"].tolist() == ["windy"] * 6
    assert out["kind"].tolist() == ["treated"] * 6
    assert out["marked"].tolist() == [True] * 6


@pytest.mark.parametrize("entry", [PRESENT, EMPTY, ABSENT])
def test_each_type_survives_parquet(ds: Dataset, tmp_path: Path, entry: Entry) -> None:
    out = _applied(ds, entry, list(TYPES))
    path = tmp_path / "out.parquet"

    _ = write_parquet_atomic(out, path)
    back = pd.read_parquet(path)

    assert str(back["flag"].dtype) == "boolean"
    assert str(back["count"].dtype) == "Int64"
    assert back["ratio"].dtype == np.float64
    assert back["marked"].dtype == np.bool_
    for column in ("note", "kind"):
        assert pd.api.types.is_string_dtype(back[column].dtype)
        assert [None if pd.isna(v) else v for v in back[column]] == out[column].tolist()
    typed = [
        "frame",
        "time",
        "group",
        "sequence",
        "id",
        "flag",
        "count",
        "ratio",
        "marked",
    ]
    pd.testing.assert_frame_equal(back[typed], out[typed])


@pytest.mark.parametrize("entry", [EMPTY, ABSENT])
def test_an_entry_without_a_value_gets_null_columns_and_a_false_label(
    ds: Dataset, entry: Entry
) -> None:
    tracks = _tracks(entry)
    feature = _feature(list(TYPES))
    _loaded(ds, _table(), feature)

    out = feature.apply(tracks)

    assert out[["frame", "time", "group", "sequence", "id"]].equals(
        tracks[["frame", "time", "group", "sequence", "id"]]
    )
    assert out["flag"].isna().all()
    assert out["count"].isna().all()
    assert out["ratio"].isna().all()
    assert out["note"].tolist() == [None] * 6
    assert out["kind"].tolist() == [None] * 6
    assert out["marked"].tolist() == [False] * 6
    assert str(out["flag"].dtype) == "boolean"
    assert str(out["count"].dtype) == "Int64"
    assert out["ratio"].dtype == np.float64
    assert out["marked"].dtype == np.bool_


def test_a_label_is_true_where_assigned_and_false_elsewhere(ds: Dataset) -> None:
    feature = _feature(["marked"])
    _loaded(ds, _table(), feature)

    assigned = feature.apply(_tracks(PRESENT))
    elsewhere = feature.apply(_tracks(EMPTY))

    assert assigned["marked"].tolist() == [True] * 6
    assert elsewhere["marked"].tolist() == [False] * 6


def test_field_renames_name_the_column(ds: Dataset) -> None:
    out = _applied(ds, PRESENT, ["kind", "count"], {"kind": "Condition"})

    assert list(out.columns) == [
        "frame",
        "time",
        "group",
        "sequence",
        "id",
        "Condition",
        "count",
    ]
    assert out["Condition"].tolist() == ["treated"] * 6


def test_the_output_has_the_identity_columns_and_the_fields_only(ds: Dataset) -> None:
    out = _applied(ds, PRESENT, ["kind"])

    assert set(out.columns) == {"frame", "time", "id", "group", "sequence", "kind"}


def test_every_row_of_an_entry_gets_its_value_whatever_its_id(ds: Dataset) -> None:
    feature = _feature(["kind"])
    _loaded(ds, _table(), feature)
    tracks = _tracks(PRESENT, ids=(7, 0, 12, 3))

    out = feature.apply(tracks)

    assert out["id"].tolist() == tracks["id"].tolist()
    assert out["kind"].tolist() == ["treated"] * len(tracks)


def test_a_feature_loaded_without_the_tag_table_refuses() -> None:
    with pytest.raises(ValueError, match="sequence-tag-columns"):
        _ = _feature(["kind"]).load_state(Path(), {}, {})


def test_loading_a_table_that_does_not_suit_the_fields_refuses_as_a_run_does(
    ds: Dataset,
) -> None:
    feature = _feature(["absent"])

    with pytest.raises(
        ValueError,
        match=(
            "sequence-tag-columns: params.fields: "
            "'absent' is not a sequence tag of the dataset"
        ),
    ):
        _loaded(ds, _table(), feature)


@pytest.mark.parametrize(
    ("fields", "renames", "where"),
    [
        ([], None, "params.fields"),
        (["absent"], None, "params.fields"),
        (["kind"], {"kind": "group"}, "params.field_renames"),
        (["kind", "note"], {"kind": "x", "note": "x"}, "params.field_renames"),
        (["kind", "note"], {"kind": "note"}, "params.field_renames"),
    ],
)
def test_the_table_refuses_a_step_it_does_not_suit(
    fields: list[str], renames: dict[str, str] | None, where: str
) -> None:
    feature = _feature(fields, renames)

    problems = tag_read_problems(_table(), feature.tag_read())

    assert [problem[0] for problem in problems] == [where]


def test_a_sequence_tag_named_like_a_written_column_is_refused() -> None:
    table = tag_table({PRESENT: {"group": "x"}}, {"group": "text"})

    problems = tag_read_problems(table, _feature(["group"]).tag_read())

    assert [problem[0] for problem in problems] == ["params.fields"]


def _outputs(ds: Dataset, run_id: str) -> dict[str, pd.DataFrame]:
    storage = derive_storage_name(
        "sequence-tag-columns", TrackInputs(("tracks",)).storage_suffix()
    )
    run_root = feature_run_root(ds, storage, run_id)
    return {
        path.name: pd.read_parquet(path) for path in sorted(run_root.glob("*.parquet"))
    }


def test_a_run_writes_the_same_output_in_two_processes_as_in_one(
    tmp_path: Path,
) -> None:
    entries = (PRESENT, EMPTY, ABSENT)
    runs: dict[str, tuple[str | None, dict[str, pd.DataFrame]]] = {}
    for name, workers, mode in (("alone", 1, "thread"), ("together", 2, "process")):
        dataset = make_dataset(tmp_path / name, uuid="fixed-uuid")
        add_track_sequences(dataset, *entries, n_ids=3)
        _ = write_tags(dataset, _table())
        result = run_feature(
            dataset,
            _feature(["kind", "count", "marked"], {"kind": "Condition"}),
            parallel_workers=workers,
            parallel_mode=mode,
        )
        runs[name] = (result.run_id, _outputs(dataset, result.run_id or ""))

    (alone_id, alone), (together_id, together) = runs["alone"], runs["together"]
    assert alone_id == together_id
    assert alone.keys() == together.keys()
    assert len(together) == 3
    for name, table in alone.items():
        pd.testing.assert_frame_equal(together[name], table)
    first = next(iter(alone.values()))
    assert {"Condition", "count", "marked"} <= set(first.columns)


def test_a_plan_over_tracks_without_tracklets_raises_no_ids_problem(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    add_tracks_variant(
        dataset,
        "convert-trex.0.2-1111111111",
        "seq_a",
        std_format="mosaic_v1",
        carries_tracklets=False,
    )
    _ = write_tags(dataset, tag_table({("", "seq_a"): {"kind": "treated"}}, TYPES))
    step: dict[str, JsonValue] = {
        "id": "step",
        "type": "feature",
        "feature": "sequence-tag-columns",
        "params": {"fields": ["kind"]},
    }
    recipe = read_recipe({"steps": [step]})

    plan = plan_pipeline(dataset, recipe)

    assert not [problem for problem in plan.problems if problem.where == "inputs"]


def test_the_identity_columns_copy_what_the_input_has_and_fill_the_rest() -> None:
    tracks = _tracks(PRESENT)

    kept = identity_columns(tracks)
    filled = identity_columns(tracks[["frame", "X"]])

    assert list(kept.columns) == ["frame", "time", "group", "sequence", "id"]
    pd.testing.assert_frame_equal(kept, tracks[list(kept.columns)])
    assert list(filled.columns) == ["frame", "group", "sequence", "id"]
    assert filled["group"].tolist() == [""] * 6
    assert filled["id"].isna().all()
