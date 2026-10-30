"""The appearance table: per-instance vectors beside a tracks table.

Each invariant of :func:`validate_appearance` is refused with a message that names
the source and the first violation. The reader returns ``None`` for an entry whose
index cell is blank and raises for a recorded file that is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry, make_entry_key
from mosaic.core.pipeline.appearance import (
    APPEARANCE_PREFIX,
    AppearanceTable,
    AppearanceTableError,
    appearance_columns,
    appearance_path,
    read_appearance,
    validate_appearance,
    write_appearance,
)
from mosaic.core.pipeline.tracks_identity import tracks_variant_root

from tests.helpers import (
    add_tracks_table,
    appearance_table,
    make_dataset,
    track_table,
)

VARIANT = "convert-trex.0.2-1111111111"
ENTRY: Entry = ("", "seq_a")
SOURCE = "seq_a.parquet"
DIM = 4


def _tracks() -> pd.DataFrame:
    """Two individuals over three frames."""
    return track_table(*ENTRY, n_rows=3, n_ids=2)


def _variant_dataset(tmp_path: Path, *, with_appearance: bool) -> Dataset:
    """A dataset with one tracks row for ``ENTRY``, with or without a table."""
    dataset = make_dataset(tmp_path)
    add_tracks_table(
        dataset,
        VARIANT,
        _tracks(),
        consumed_source_roots=(),
        appearance_dim=DIM if with_appearance else None,
    )
    return dataset


def test_the_prefix_and_the_column_names() -> None:
    assert APPEARANCE_PREFIX == "emb_"
    assert appearance_columns(3) == ("emb_0", "emb_1", "emb_2")
    assert appearance_columns(0) == ()


def test_the_path_is_under_the_variant_directory() -> None:
    root = Path("tracks") / VARIANT

    assert appearance_path(root, "g__s") == root / "appearance" / "g__s.parquet"


def test_a_valid_table_returns_its_dimension() -> None:
    tracks = _tracks()

    assert (
        validate_appearance(appearance_table(tracks, DIM), tracks, source=SOURCE) == DIM
    )


def test_time_is_tolerated() -> None:
    tracks = _tracks()
    table = appearance_table(tracks, DIM)
    table["time"] = table["frame"] / 30.0

    assert validate_appearance(table, tracks, source=SOURCE) == DIM


def test_a_table_may_cover_fewer_rows_than_the_tracks_table() -> None:
    tracks = _tracks()

    assert (
        validate_appearance(
            appearance_table(tracks, DIM).iloc[:2], tracks, source=SOURCE
        )
        == DIM
    )


def test_an_empty_table_is_valid() -> None:
    tracks = _tracks()

    assert (
        validate_appearance(
            appearance_table(tracks, DIM).iloc[:0], tracks, source=SOURCE
        )
        == DIM
    )


def _repeat_a_row(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([table, table.iloc[[0]]], ignore_index=True)


def _absent_row(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table.loc[0, "frame"] = 999
    return table


def _off_norm(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["emb_0"] = table["emb_0"] * np.float32(1.01)
    return table


def _gap(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    return table.drop(columns=["emb_1"])


def _no_vectors(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    return table.drop(columns=list(appearance_columns(DIM)))


def _nan(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table.loc[1, "emb_2"] = np.float32("nan")
    return table


def _no_key(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    return table.drop(columns=["frame"])


def _float64_vectors(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    return table.astype({column: np.float64 for column in appearance_columns(DIM)})


def _float_frame(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    return table.astype({"frame": np.float64})


def _text_vectors(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    return table.astype({"emb_1": str})


def _no_sequence(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    return table.drop(columns=["sequence"])


def _tracks_without_id(table: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    tracks.drop(columns=["id"], inplace=True)
    return table


@pytest.mark.parametrize(
    ("break_it", "message"),
    [
        pytest.param(_repeat_a_row, r"\(frame, id\) is not unique", id="repeated-key"),
        pytest.param(
            _absent_row, "is not in the tracks table", id="absent-from-tracks"
        ),
        pytest.param(_off_norm, "more than 0.001 from 1", id="norm"),
        pytest.param(_gap, "are not contiguous from emb_0", id="gap"),
        pytest.param(_no_vectors, "the table has no emb_ columns", id="no-vectors"),
        pytest.param(_nan, "NaN in emb_2 at row 1", id="nan"),
        pytest.param(_no_key, "the table lacks the frame column", id="missing-key"),
        pytest.param(
            _no_sequence, "the table lacks the sequence column", id="missing-label"
        ),
        pytest.param(
            _tracks_without_id,
            "the tracks table lacks the id column",
            id="tracks-missing-key",
        ),
        pytest.param(
            _float64_vectors, "the emb_0 column is float64, not float32", id="float64"
        ),
        pytest.param(
            _float_frame, "the frame column is float64, not int64", id="float-frame"
        ),
        pytest.param(
            _text_vectors, "the emb_1 column is .*, not float32", id="text-vectors"
        ),
    ],
)
def test_each_invariant_is_refused_naming_the_source(
    break_it: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame], message: str
) -> None:
    tracks = _tracks()
    broken = break_it(appearance_table(tracks, DIM), tracks)

    with pytest.raises(AppearanceTableError, match=message) as raised:
        _ = validate_appearance(broken, tracks, source=SOURCE)

    assert str(raised.value).startswith(f"{SOURCE}: ")
    assert isinstance(raised.value, ValueError)


def test_a_norm_within_the_tolerance_passes() -> None:
    tracks = _tracks()
    table = appearance_table(tracks, DIM)
    table["emb_0"] = table["emb_0"] * np.float32(1.0005)

    assert validate_appearance(table, tracks, source=SOURCE) == DIM


def test_a_table_round_trips_through_the_index(tmp_path: Path) -> None:
    dataset = _variant_dataset(tmp_path, with_appearance=True)
    expected = appearance_table(_tracks(), DIM)

    table = read_appearance(dataset, VARIANT, ENTRY)

    assert isinstance(table, AppearanceTable)
    assert table.frame.dtype == np.int64
    assert table.id.dtype == np.int64
    assert table.vectors.dtype == np.float32
    assert table.vectors.flags.c_contiguous
    assert table.vectors.shape == (len(expected), DIM)
    assert table.frame.tolist() == expected["frame"].tolist()
    assert table.id.tolist() == expected["id"].tolist()
    np.testing.assert_array_equal(
        table.vectors, expected[list(appearance_columns(DIM))].to_numpy()
    )


def test_a_written_table_keeps_its_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "appearance" / "seq_a.parquet"
    expected = appearance_table(_tracks(), DIM)

    write_appearance(path, expected)

    stored = pd.read_parquet(path)
    assert stored.dtypes.to_dict() == expected.dtypes.to_dict()
    assert [p.name for p in path.parent.iterdir()] == ["seq_a.parquet"]


def test_a_blank_cell_reads_none(tmp_path: Path) -> None:
    dataset = _variant_dataset(tmp_path, with_appearance=False)

    assert read_appearance(dataset, VARIANT, ENTRY) is None


def test_an_entry_without_a_row_reads_none(tmp_path: Path) -> None:
    dataset = _variant_dataset(tmp_path, with_appearance=True)

    assert read_appearance(dataset, VARIANT, ("", "other")) is None
    assert read_appearance(dataset, "convert-trex.0.2-2222222222", ENTRY) is None


def test_a_recorded_file_that_is_missing_raises(tmp_path: Path) -> None:
    dataset = _variant_dataset(tmp_path, with_appearance=True)
    root = tracks_variant_root(dataset.get_root("tracks"), VARIANT)
    appearance_path(root, make_entry_key(*ENTRY)).unlink()

    with pytest.raises(FileNotFoundError, match="seq_a"):
        _ = read_appearance(dataset, VARIANT, ENTRY)


def test_a_file_without_a_key_column_is_refused(tmp_path: Path) -> None:
    dataset = _variant_dataset(tmp_path, with_appearance=True)
    root = tracks_variant_root(dataset.get_root("tracks"), VARIANT)
    path = appearance_path(root, make_entry_key(*ENTRY))
    pd.read_parquet(path).drop(columns=["id"]).to_parquet(path)

    with pytest.raises(AppearanceTableError, match="the table lacks the id column"):
        _ = read_appearance(dataset, VARIANT, ENTRY)


def test_reading_creates_nothing(tmp_path: Path) -> None:
    dataset = _variant_dataset(tmp_path, with_appearance=True)
    before = sorted(dataset.base_dir.rglob("*"))

    _ = read_appearance(dataset, VARIANT, ENTRY)

    assert sorted(dataset.base_dir.rglob("*")) == before
