"""``preview_artifact`` reads one entry's parquet table of a tracks variant or a feature run.

Each test compares the preview's schema, row count, frame extent, head or
statistics with the table read whole through pandas. Non-finite floats are null,
and the refusals are classified.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import error_envelope
from mosaic.core.pipeline.graph import run_pipeline
from mosaic.core.pipeline.inventory import (
    FeatureRunRef,
    LabelsVariantRef,
    TracksVariantRef,
)
from mosaic.core.pipeline.labels_index import write_labels_row
from mosaic.core.pipeline.preview import (
    MAX_OMITTED_NAMES,
    MAX_PREVIEW_ROWS,
    PreviewInvalid,
    PreviewNotFound,
    PreviewUnreadable,
    preview_artifact,
    preview_ref,
)
from tests.helpers import (
    speed_recipe,
    add_track_sequences,
    add_tracks_table,
    assert_tree_unchanged,
    make_dataset,
    track_table,
    tree_snapshot,
)

VARIANT = "convert-trex.0.2-1111111111"
LABELS_VARIANT = "boris.0.1-2222222222"
ENTRY = ("", "seq_a")
FEATURE_STORAGE = "speed-angvel__from__tracks"


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
    done = run_pipeline(ds, speed_recipe())
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
        (
            f"labels-variant/behavior/{LABELS_VARIANT}",
            LabelsVariantRef("behavior", LABELS_VARIANT),
        ),
    ],
)
def test_an_address_names_its_ref(
    address: str, expected: TracksVariantRef | FeatureRunRef | LabelsVariantRef
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
        "labels-variant/behavior",
        f"labels-variant/../{LABELS_VARIANT}",
        f"labels-variant/behavior/a/{LABELS_VARIANT}",
    ],
)
def test_another_address_is_invalid(address: str) -> None:
    with pytest.raises(PreviewInvalid):
        _ = preview_ref(address)


# --- wide tables: max_columns, patterns and listed ------------------------------

WIDE_STORAGE = 400


def _wide(tmp_path: Path) -> Dataset:
    """A table of 400 payload columns before its identity and bookkeeping columns."""
    ds = make_dataset(tmp_path / "ds")
    base = track_table("", "seq_a", n_rows=4, n_ids=1)
    payload = pd.DataFrame(
        {f"W_{i:03d}_f{i % 2}": np.arange(4, dtype=float) for i in range(WIDE_STORAGE)}
    )
    meta = base[["frame", "id", "group", "sequence"]].assign(label="fish")
    table = pd.concat([payload, meta], axis=1)
    add_tracks_table(ds, VARIANT, table, std_format="mosaic_v1")
    return ds


def test_max_columns_reads_bookkeeping_columns_first(tmp_path: Path) -> None:
    ds = _wide(tmp_path)

    preview = preview_artifact(
        ds, TracksVariantRef(VARIANT), ENTRY, max_columns=7, listed="read"
    )

    assert set(preview.read_columns) >= {"frame", "id", "group", "sequence", "label"}
    assert preview.read_columns == (
        "W_000_f0",
        "W_001_f1",
        "frame",
        "id",
        "group",
        "sequence",
        "label",
    )


def test_a_400_column_table_is_bounded_by_max_columns(tmp_path: Path) -> None:
    ds = _wide(tmp_path)

    preview = preview_artifact(
        ds, TracksVariantRef(VARIANT), ENTRY, max_columns=40, listed="read", limit=5
    )

    assert len(preview.read_columns) == 40
    assert preview.n_columns == WIDE_STORAGE + 5
    assert len(preview.columns) == 40
    assert len(preview.model_dump_json()) < 15_000


def test_a_pattern_expands_in_table_order(tmp_path: Path) -> None:
    ds = _wide(tmp_path)

    preview = preview_artifact(
        ds, TracksVariantRef(VARIANT), ENTRY, columns=["frame", "W_00?_f1"]
    )

    assert preview.read_columns == (
        "frame",
        "W_001_f1",
        "W_003_f1",
        "W_005_f1",
        "W_007_f1",
        "W_009_f1",
    )


def test_an_exact_name_wins_over_a_pattern_reading(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    table = track_table("", "seq_a", n_rows=4, n_ids=1).assign(**{"a*": 1.0, "ab": 2.0})
    add_tracks_table(ds, VARIANT, table, std_format="mosaic_v1")

    preview = preview_artifact(ds, TracksVariantRef(VARIANT), ENTRY, columns=["a*"])

    assert preview.read_columns == ("a*",)


def test_a_pattern_matching_nothing_is_unknown(dataset: Dataset) -> None:
    with pytest.raises(PreviewInvalid, match="'Q_\\*'") as raised:
        _ = preview_artifact(
            dataset, TracksVariantRef(VARIANT), ENTRY, columns=["X", "Q_*"]
        )

    assert raised.value.error_detail()["unknown"] == ["Q_*"]


def test_listed_read_lists_only_the_columns_read(dataset: Dataset) -> None:
    table = _stored(dataset)

    preview = preview_artifact(
        dataset, TracksVariantRef(VARIANT), ENTRY, columns=["X", "id"], listed="read"
    )

    assert [column.name for column in preview.columns] == ["id", "X"]
    assert preview.n_columns == len(table.columns)
    assert preview.columns_omitted == ()


def test_omitted_columns_are_named_up_to_the_cap(tmp_path: Path) -> None:
    ds = _wide(tmp_path)

    preview = preview_artifact(ds, TracksVariantRef(VARIANT), ENTRY, max_columns=10)

    assert len(preview.columns_omitted) == MAX_OMITTED_NAMES
    assert preview.columns_omitted_has_more
    assert preview.columns_omitted[0] == "W_005_f1"
    assert not set(preview.columns_omitted) & set(preview.read_columns)


def test_an_unknown_column_error_is_bounded_and_suggests_close_names(
    tmp_path: Path,
) -> None:
    ds = _wide(tmp_path)
    asked = ["W_000_f2", *(f"nope_{i}" for i in range(30))]

    with pytest.raises(PreviewInvalid) as raised:
        _ = preview_artifact(ds, TracksVariantRef(VARIANT), ENTRY, columns=asked)

    message = str(raised.value)
    assert "and 11 more" in message
    assert "nope_29" not in message
    assert len(message) < 2_000
    detail = raised.value.error_detail()
    assert detail["field"] == "columns"
    assert detail["n_columns"] == WIDE_STORAGE + 5
    unknown = detail["unknown"]
    assert isinstance(unknown, list)
    assert len(unknown) == 10
    close = detail["close"]
    assert isinstance(close, dict)
    found = close["W_000_f2"]
    assert isinstance(found, list)
    assert "W_000_f0" in found


# --- labels variants ------------------------------------------------------------


def _add_labels(ds: Dataset, arrays: dict[str, np.ndarray]) -> Path:
    """Write one entry's labels file into the variant and index it."""
    out_path = ds.get_root("labels") / "behavior" / LABELS_VARIANT / "seq_a.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # numpy's stub types a splatted mapping as the allow_pickle keyword.
    savez: Callable[..., None] = np.savez_compressed
    savez(out_path, **arrays)
    write_labels_row(
        ds,
        run_id=LABELS_VARIANT,
        group="",
        sequence="seq_a",
        out_path=out_path,
        producer="convert-labels-boris_aggregated_csv",
        label_kind="behavior",
        label_format="",
        n_frames=100,
        label_ids="0,1,2",
        label_names="other,attack,mount",
    )
    return out_path


def _pair_events(*events: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    frames, labels, id1, id2 = zip(*events, strict=True)
    return {
        "label_format": np.array("individual_pair_v1"),
        "frames": np.array(frames),
        "labels": np.array(labels),
        "individual_ids": np.array(list(zip(id1, id2, strict=True))),
    }


def test_a_labels_variant_is_previewed_with_class_counts_and_coverage(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = _add_labels(
        ds, _pair_events((10, 1, 0, 1), (11, 1, 0, 1), (12, 2, 0, 1), (40, 1, 1, 2))
    )

    preview = preview_artifact(
        ds, LabelsVariantRef("behavior", LABELS_VARIANT), ENTRY, limit=2
    )

    assert preview.ref == f"labels-variant/behavior/{LABELS_VARIANT}"
    assert preview.read_columns == ("frame", "label", "id1", "id2")
    assert preview.rows == ((10, 1, 0, 1), (11, 1, 0, 1))
    assert preview.has_more
    assert (preview.frame_min, preview.frame_max) == (10, 40)
    labels = preview.labels
    assert labels is not None
    assert labels.label_format == "individual_pair_v1"
    assert (labels.n_events, labels.frames_labeled, labels.n_pairs) == (4, 4, 2)
    assert labels.coverage == pytest.approx(0.04)
    counts = {c.name: (c.id, c.events) for c in labels.classes}
    assert counts == {"other": (0, 0), "attack": (1, 3), "mount": (2, 1)}
    assert not labels.symmetric_deduplicated


def test_mirrored_pair_events_are_counted_once(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = _add_labels(
        ds, _pair_events((10, 1, 0, 1), (10, 1, 1, 0), (11, 2, 0, 1), (11, 2, 1, 0))
    )

    preview = preview_artifact(ds, LabelsVariantRef("behavior", LABELS_VARIANT), ENTRY)

    labels = preview.labels
    assert labels is not None
    assert preview.n_rows == 4
    assert labels.n_events == 2
    assert labels.symmetric_deduplicated
    assert {c.id: c.events for c in labels.classes} == {0: 0, 1: 1, 2: 1}
    assert labels.n_pairs == 1


def test_a_dense_labels_table_is_previewed(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = _add_labels(ds, {"labels": np.array([0, 0, 1, 2, 2, 2])})

    preview = preview_artifact(ds, LabelsVariantRef("behavior", LABELS_VARIANT), ENTRY)

    assert preview.read_columns == ("frame", "label")
    labels = preview.labels
    assert labels is not None
    assert labels.label_format == "dense"
    assert (labels.n_events, labels.frames_labeled, labels.n_pairs) == (6, 6, None)
    assert {c.id: c.events for c in labels.classes} == {0: 2, 1: 1, 2: 3}


def test_previewing_labels_writes_nothing_and_unpickles_nothing(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    events = _pair_events((10, 1, 0, 1))
    events["individual_ids"] = np.array([(0, 1)], dtype=object)
    _ = _add_labels(ds, events)
    before = tree_snapshot(ds.base_dir)

    with pytest.raises(PreviewUnreadable, match="unpickling"):
        _ = preview_artifact(ds, LabelsVariantRef("behavior", LABELS_VARIANT), ENTRY)

    assert_tree_unchanged(ds.base_dir, before)


def test_a_labels_preview_that_succeeds_writes_nothing(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = _add_labels(ds, _pair_events((10, 1, 0, 1), (10, 1, 1, 0)))
    before = tree_snapshot(ds.base_dir)

    preview = preview_artifact(ds, LabelsVariantRef("behavior", LABELS_VARIANT), ENTRY)

    assert preview.labels is not None
    assert_tree_unchanged(ds.base_dir, before)


def _truncated(path: Path) -> None:
    _ = path.write_bytes(b"PK\x03\x04 cut short")


def _ragged(path: Path) -> None:
    arrays = _pair_events((10, 1, 0, 1), (11, 1, 0, 1))
    arrays["labels"] = np.array([1])
    savez: Callable[..., None] = np.savez_compressed
    savez(path, **arrays)


@pytest.mark.parametrize("damage", [_truncated, _ragged])
def test_a_damaged_labels_file_is_unreadable(
    tmp_path: Path, damage: Callable[[Path], None]
) -> None:
    ds = make_dataset(tmp_path / "ds")
    damage(_add_labels(ds, _pair_events((10, 1, 0, 1))))

    with pytest.raises(PreviewUnreadable):
        _ = preview_artifact(ds, LabelsVariantRef("behavior", LABELS_VARIANT), ENTRY)
