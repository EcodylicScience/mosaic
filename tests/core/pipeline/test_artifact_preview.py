"""``preview_artifact`` reads one entry's parquet table of a tracks variant or a feature run.

Each test compares the preview's schema, row count, frame extent, head or
statistics with the table read whole through pandas. Non-finite floats are null,
and the refusals are classified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import error_envelope
from mosaic.core.pipeline.graph import Recipe, run_pipeline
from mosaic.core.pipeline.inventory import FeatureRunRef, TracksVariantRef
from mosaic.core.pipeline.preview import (
    MAX_PREVIEW_ROWS,
    PreviewInvalid,
    PreviewNotFound,
    PreviewUnreadable,
    preview_artifact,
    preview_ref,
)
from tests.helpers import (
    add_track_sequences,
    add_tracks_table,
    make_dataset,
    track_table,
)

VARIANT = "convert-trex.0.2-1111111111"
ENTRY = ("", "seq_a")
FEATURE_STORAGE = "speed-angvel__from__tracks"

RECIPE: dict[str, object] = {
    "schema_version": 1,
    "name": "speed",
    "steps": [
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
        }
    ],
}


def _table() -> pd.DataFrame:
    """Two individuals over 30 frames, with NaN, infinite, boolean and text cells."""
    table = track_table("", "seq_a", n_rows=30, n_ids=2)
    x = table["X"].to_numpy(copy=True)
    x[0] = np.nan
    x[1] = np.inf
    table["X"] = x
    table["visible"] = table["frame"] % 2 == 0
    table["label"] = "fish"
    return table


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_table(ds, VARIANT, _table(), std_format="mosaic_v1")
    return ds


def _stored(ds: Dataset) -> pd.DataFrame:
    return pd.read_parquet(ds.get_root("tracks") / VARIANT / "seq_a.parquet")


def test_the_preview_describes_the_whole_table(dataset: Dataset) -> None:
    table = _stored(dataset)

    preview = preview_artifact(dataset, TracksVariantRef(VARIANT), ENTRY)

    assert preview.ref == f"tracks-variant/{VARIANT}"
    assert (preview.group, preview.sequence) == ENTRY
    assert [column.name for column in preview.columns] == list(table.columns)
    assert preview.n_rows == len(table)
    assert (preview.frame_min, preview.frame_max) == (0, 29)
    assert preview.read_columns == tuple(table.columns)
    assert len(preview.rows) == 20
    assert preview.has_more
    assert Path(preview.path).is_file()


def test_columns_are_typed_as_the_table_reads_name_them(dataset: Dataset) -> None:
    preview = preview_artifact(dataset, TracksVariantRef(VARIANT), ENTRY)

    types = {column.name: column.type for column in preview.columns}
    assert types["frame"] == "int"
    assert types["X"] == "float"
    assert types["visible"] == "int"
    assert types["label"] == "text"
    arrow = {column.name: column.arrow_type for column in preview.columns}
    assert arrow["frame"] == "int64"
    assert arrow["visible"] == "bool"


def test_the_head_is_json_safe(dataset: Dataset) -> None:
    preview = preview_artifact(
        dataset, TracksVariantRef(VARIANT), ENTRY, columns=["X", "visible", "label"]
    )

    assert preview.rows[0] == (None, 1, "fish")
    assert preview.rows[1] == (None, 1, "fish")
    assert preview.rows[2][1:] == (0, "fish")
    _ = json.dumps(preview.model_dump(mode="json"), allow_nan=False)


def test_statistics_cover_the_finite_values_of_every_row(dataset: Dataset) -> None:
    table = _stored(dataset)
    finite = table["X"][np.isfinite(table["X"])]

    preview = preview_artifact(
        dataset, TracksVariantRef(VARIANT), ENTRY, columns=["X", "frame"], limit=3
    )

    statistics = {column.name: column.statistics for column in preview.columns}
    x = statistics["X"]
    assert x is not None
    assert x.min == pytest.approx(finite.min())
    assert x.max == pytest.approx(finite.max())
    assert x.mean == pytest.approx(finite.mean())
    assert x.nan_fraction == pytest.approx(1 / len(table))
    frame = statistics["frame"]
    assert frame is not None
    assert (frame.min, frame.max, frame.nan_fraction) == (0, 29, 0.0)
    assert statistics["Y"] is None
    assert statistics["label"] is None


def test_statistics_are_reduced_across_row_groups(dataset: Dataset) -> None:
    path = dataset.get_root("tracks") / VARIANT / "seq_a.parquet"
    table = _stored(dataset)
    x = table["X"].to_numpy(copy=True)
    x[[9, 17, 25]] = [np.nan, -np.inf, np.nan]
    table["X"] = x
    table.to_parquet(path, index=False, row_group_size=7)

    preview = preview_artifact(
        dataset, TracksVariantRef(VARIANT), ENTRY, columns=["X", "visible"], limit=0
    )

    statistics = {column.name: column.statistics for column in preview.columns}
    finite = table["X"][np.isfinite(table["X"])]
    x_statistics = statistics["X"]
    assert x_statistics is not None
    assert x_statistics.min == pytest.approx(finite.min())
    assert x_statistics.max == pytest.approx(finite.max())
    assert x_statistics.mean == pytest.approx(finite.mean())
    assert x_statistics.nan_fraction == pytest.approx(3 / len(table))
    visible = statistics["visible"]
    assert visible is not None
    assert (visible.min, visible.max) == (0, 1)
    assert visible.mean == pytest.approx(table["visible"].mean())
    assert (preview.frame_min, preview.frame_max) == (0, 29)


def test_integer_extremes_are_exact_beyond_float_precision(dataset: Dataset) -> None:
    path = dataset.get_root("tracks") / VARIANT / "seq_a.parquet"
    table = _stored(dataset)
    table["count"] = np.arange(len(table), dtype=np.uint64) + np.uint64(2**63 + 5)
    table.to_parquet(path, index=False)

    preview = preview_artifact(
        dataset, TracksVariantRef(VARIANT), ENTRY, columns=["count"], limit=0
    )

    statistics = {column.name: column.statistics for column in preview.columns}
    count = statistics["count"]
    assert count is not None
    assert (count.min, count.max) == (2**63 + 5, 2**63 + 5 + len(table) - 1)


def test_the_unlabeled_tables_are_previewed(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, "seq_a", n_rows=12)

    preview = preview_artifact(ds, TracksVariantRef(""), ENTRY)

    assert preview.ref == "tracks-variant/"
    assert preview.n_rows == 12
    assert (preview.frame_min, preview.frame_max) == (0, 11)


def test_columns_narrow_the_head_in_the_order_asked(dataset: Dataset) -> None:
    table = _stored(dataset)

    preview = preview_artifact(
        dataset,
        TracksVariantRef(VARIANT),
        ENTRY,
        columns=["id", "frame", "id"],
        limit=3,
    )

    assert preview.read_columns == ("id", "frame")
    expected = [(int(i), int(f)) for i, f in table[["id", "frame"]].to_numpy()[:3]]
    assert list(preview.rows) == expected
    assert len(preview.columns) == len(table.columns)


def test_the_frame_extent_is_read_without_the_frame_column_asked(
    dataset: Dataset,
) -> None:
    preview = preview_artifact(
        dataset, TracksVariantRef(VARIANT), ENTRY, columns=["label"], limit=0
    )

    assert (preview.frame_min, preview.frame_max) == (0, 29)
    assert preview.rows == ()
    assert all(column.statistics is None for column in preview.columns)


def test_a_feature_run_is_previewed(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_table(
        ds, VARIANT, track_table("", "seq_a", n_ids=2), std_format="mosaic_v1"
    )
    done = run_pipeline(ds, Recipe.model_validate(RECIPE))
    run_id = done.outcomes[0].run_id
    assert run_id is not None

    preview = preview_artifact(ds, FeatureRunRef(FEATURE_STORAGE, run_id), ENTRY)

    table = pd.read_parquet(preview.path)
    assert preview.ref == f"feature/{FEATURE_STORAGE}/{run_id}"
    assert preview.n_rows == len(table)
    assert [column.name for column in preview.columns] == list(table.columns)
    assert (preview.frame_min, preview.frame_max) == (
        int(table["frame"].min()),
        int(table["frame"].max()),
    )
    expected = [
        None if isinstance(value, float) and not math.isfinite(value) else value
        for value in table.iloc[0].tolist()
    ]
    assert list(preview.rows[0]) == expected


@pytest.mark.parametrize("limit", [-1, MAX_PREVIEW_ROWS + 1])
def test_a_limit_out_of_range_is_invalid(dataset: Dataset, limit: int) -> None:
    with pytest.raises(PreviewInvalid) as raised:
        _ = preview_artifact(dataset, TracksVariantRef(VARIANT), ENTRY, limit=limit)

    assert error_envelope(raised.value).detail == {"field": "limit"}


def test_an_unknown_column_is_invalid(dataset: Dataset) -> None:
    with pytest.raises(PreviewInvalid, match="'nope'") as raised:
        _ = preview_artifact(
            dataset, TracksVariantRef(VARIANT), ENTRY, columns=["X", "nope"]
        )

    assert raised.value.error_kind == "invalid"


@pytest.mark.parametrize(
    "ref",
    [
        TracksVariantRef("convert-trex.0.2-0000000000"),
        TracksVariantRef(""),
        FeatureRunRef(FEATURE_STORAGE, "0.2-0000000000"),
    ],
)
def test_an_unindexed_table_is_not_found(
    dataset: Dataset, ref: TracksVariantRef | FeatureRunRef
) -> None:
    with pytest.raises(PreviewNotFound) as raised:
        _ = preview_artifact(dataset, ref, ENTRY)

    assert raised.value.error_kind == "not_found"


def test_an_unindexed_entry_is_not_found(dataset: Dataset) -> None:
    with pytest.raises(PreviewNotFound) as raised:
        _ = preview_artifact(dataset, TracksVariantRef(VARIANT), ("g", "seq_a"))

    assert error_envelope(raised.value).detail == {
        "ref": f"tracks-variant/{VARIANT}",
        "group": "g",
        "sequence": "seq_a",
    }


def test_a_missing_file_is_not_found(dataset: Dataset) -> None:
    (dataset.get_root("tracks") / VARIANT / "seq_a.parquet").unlink()

    with pytest.raises(PreviewNotFound, match="absent"):
        _ = preview_artifact(dataset, TracksVariantRef(VARIANT), ENTRY)


def test_an_unreadable_file_is_a_conflict(dataset: Dataset) -> None:
    path = dataset.get_root("tracks") / VARIANT / "seq_a.parquet"
    _ = path.write_bytes(b"PAR1 truncated garbage")

    with pytest.raises(PreviewUnreadable, match="not a readable parquet") as raised:
        _ = preview_artifact(dataset, TracksVariantRef(VARIANT), ENTRY)

    envelope = error_envelope(raised.value)
    assert envelope.kind == "conflict"
    assert envelope.detail == {
        "ref": f"tracks-variant/{VARIANT}",
        "group": "",
        "sequence": "seq_a",
    }


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        (
            "feature/speed__from__tracks/0.2-abc",
            FeatureRunRef("speed__from__tracks", "0.2-abc"),
        ),
        (f"tracks-variant/{VARIANT}", TracksVariantRef(VARIANT)),
        ("tracks-variant/", TracksVariantRef("")),
    ],
)
def test_an_address_names_its_ref(
    address: str, expected: TracksVariantRef | FeatureRunRef
) -> None:
    assert preview_ref(address) == expected


@pytest.mark.parametrize(
    "address",
    [
        "feature/speed__from__tracks",
        "feature//0.2-abc",
        "feature/a/b/c",
        "tracks-variant",
        "tracks-variant/a/b",
        "trained-model/train-pose/train-pose.0.1-0123456789",
    ],
)
def test_another_address_is_invalid(address: str) -> None:
    with pytest.raises(PreviewInvalid):
        _ = preview_ref(address)
