"""``load_run_matrix`` reads one finished run as a row index and an array.

An analysis script that wants a model's columns used to choose between
``load_values`` and reading the parquet files itself. ``load_values`` takes one
``ResultColumn`` per column (546 for the wavelets), has no perspective filter and
returns one float64 frame of every entry. The script that went to the files
re-implemented the alignment.

The reader returns the identity of each row beside a ``(rows, columns)`` array of
the dtype asked for. The cases below hold it to what ``load_values`` returns on the
same run, and to what the reader promises on top of it: the same order whatever the
order of the index, a refusal where the run is partial or the read would not fit,
and no write.
"""

from __future__ import annotations

import hashlib
import tracemalloc
import zlib
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline import load_run_matrix as exported_load_run_matrix
from mosaic.core.pipeline import run_matrix as run_matrix_module
from mosaic.core.pipeline.index import feature_index_path
from mosaic.core.pipeline.parquet_reader import ParquetUnreadable, iter_column_batches
from mosaic.core.pipeline.run import load_values
from mosaic.core.pipeline.run_matrix import RunMatrix, load_run_matrix
from mosaic.core.pipeline.track_universe import current_run_id
from mosaic.core.pipeline.types import Result, ResultColumn
from mosaic.core.scope import Scope
from tests.helpers import (
    PairClusterFeature,
    PlainFeature,
    add_track_sequences,
    make_dataset,
    run_speed,
    tree_snapshot,
)

N_WIDE = 12
"""How many ``w<k>`` columns the wide feature writes."""

ENTRIES = (("g2", "s1"), ("g1", "s2"), ("g1", "s1"), ("", "solo"))
"""Added out of order, with one entry in the empty group."""

KEY = ["group", "sequence", "frame", "id"]


class WideFeature(PlainFeature):
    """Numeric columns of several types, and the columns a reader must leave out.

    ``w<k>`` alternate between float32 and float64, ``w0`` holds a NaN every fifth
    frame, ``count`` is an integer and ``flag`` a boolean. ``note`` is a string and
    ``fps`` is bookkeeping. Each entry's values are seeded from its key, so two
    entries never hold the same numbers.
    """

    name = "prov-wide"
    n_wide = N_WIDE

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        key = make_entry_key(str(df["group"].iloc[0]), str(df["sequence"].iloc[0]))
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        frames = df["frame"].to_numpy()
        columns: dict[str, object] = {
            "frame": frames,
            "id": df["id"].to_numpy(),
            "group": df["group"].to_numpy(),
            "sequence": df["sequence"].to_numpy(),
        }
        for k in range(self.n_wide):
            dtype = np.float32 if k % 2 else np.float64
            columns[f"w{k}"] = rng.random(len(df)).astype(dtype)
        columns["w0"] = np.where(frames % 5 == 0, np.nan, rng.random(len(df)))
        columns["count"] = frames.astype(np.int64) * 3
        columns["flag"] = frames % 2 == 0
        columns["note"] = np.full(len(df), "text")
        columns["fps"] = np.full(len(df), 30.0)
        return pd.DataFrame(columns)


class VeryWideFeature(WideFeature):
    """The same table with hundreds of columns, for the test of the memory a read takes."""

    name = "prov-very-wide"
    n_wide = 200


def wide_run(
    tmp_path: Path, *, n_rows: int = 10, n_ids: int = 2
) -> tuple[Dataset, Result]:
    """A dataset with ``ENTRIES`` and one run of :class:`WideFeature` over them."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *ENTRIES, n_rows=n_rows, n_ids=n_ids)
    return ds, ds.run_feature(WideFeature())


def pair_run(tmp_path: Path) -> tuple[Dataset, Result]:
    """A dataset with ``ENTRIES`` and one run of both perspectives of a pair."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *ENTRIES, n_rows=4, n_ids=2)
    return ds, ds.run_feature(PairClusterFeature())


def output_of(ds: Dataset, result: Result, group: str, sequence: str) -> Path:
    """The parquet that the index lists for one entry of *result*."""
    index = pd.read_csv(
        feature_index_path(ds, result.feature), dtype=str, keep_default_na=False
    )
    rows = index[
        (index["run_id"] == result.run_id)
        & (index["group"] == group)
        & (index["sequence"] == sequence)
    ]
    return ds.resolve_path(rows["abs_path"].iloc[0])


def rewrite_index(
    ds: Dataset, result: Result, edit: Callable[[pd.DataFrame], pd.DataFrame]
) -> None:
    """Rewrite the feature index of *result* through *edit*, a frame function."""
    path = feature_index_path(ds, result.feature)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    edit(frame).to_csv(path, index=False)


def tree_digest(root: Path) -> str:
    """A hash of every path and every byte under *root*."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        digest.update(path.relative_to(root).as_posix().encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


# --- the round trip ---------------------------------------------------------


def test_the_matrix_is_what_load_values_returns_for_the_same_run(
    tmp_path: Path,
) -> None:
    ds, result = wide_run(tmp_path)

    matrix = load_run_matrix(ds, result)

    sources = [
        ResultColumn(feature=result.feature, run_id=result.run_id, column=name)
        for name in matrix.columns
    ]
    expected = load_values(ds, sources)
    assert isinstance(matrix, RunMatrix)
    assert matrix.columns[0] == "w0"
    assert matrix.values.dtype == np.float32
    assert matrix.values.shape == (len(expected), len(matrix.columns))
    np.testing.assert_array_equal(
        matrix.values, expected[list(matrix.columns)].to_numpy(dtype=np.float32)
    )
    pd.testing.assert_frame_equal(
        matrix.index.reset_index(drop=True),
        expected[KEY].reset_index(drop=True),
        check_dtype=False,
    )
    assert matrix.feature == result.feature
    assert matrix.run_id == result.run_id


def test_the_default_columns_are_the_numeric_payload(tmp_path: Path) -> None:
    """Identity, bookkeeping and string columns are not values."""
    ds, result = wide_run(tmp_path)

    matrix = load_run_matrix(ds, result)

    expected = [f"w{k}" for k in range(N_WIDE)] + ["count", "flag"]
    assert list(matrix.columns) == expected
    assert list(matrix.index.columns) == KEY


def test_nan_cells_stay_nan_and_integers_and_booleans_become_numbers(
    tmp_path: Path,
) -> None:
    ds, result = wide_run(tmp_path)

    matrix = load_run_matrix(ds, result)

    w0 = matrix.values[:, matrix.columns.index("w0")]
    frames = matrix.index["frame"].to_numpy()
    assert np.isnan(w0[frames % 5 == 0]).all()
    assert not np.isnan(w0[frames % 5 != 0]).any()
    np.testing.assert_array_equal(
        matrix.values[:, matrix.columns.index("count")], frames * 3
    )
    np.testing.assert_array_equal(
        matrix.values[:, matrix.columns.index("flag")], (frames % 2 == 0) * 1.0
    )


def test_a_pair_run_keeps_the_pair_identity_in_the_index(tmp_path: Path) -> None:
    ds, result = pair_run(tmp_path)

    matrix = load_run_matrix(ds, result)

    assert list(matrix.index.columns) == [
        "group",
        "sequence",
        "frame",
        "id1",
        "id2",
        "perspective",
    ]
    assert matrix.columns == ("cluster",)
    expected = load_values(
        ds,
        [ResultColumn(feature=result.feature, run_id=result.run_id, column="cluster")],
    )
    np.testing.assert_array_equal(
        matrix.values[:, 0], expected["cluster"].to_numpy(dtype=np.float32)
    )
    assert matrix.index["perspective"].tolist() == expected["perspective"].tolist()


def test_the_export_is_the_module_function() -> None:
    assert exported_load_run_matrix is load_run_matrix


# --- narrowing the read -----------------------------------------------------


def test_a_perspective_keeps_only_that_perspectives_rows(tmp_path: Path) -> None:
    ds, result = pair_run(tmp_path)
    everything = load_run_matrix(ds, result)

    focal = load_run_matrix(ds, result, perspective=0)

    keep = (everything.index["perspective"] == 0).to_numpy()
    assert set(focal.index["perspective"]) == {0}
    assert len(focal.values) == int(keep.sum()) == len(everything.values) // 2
    np.testing.assert_array_equal(focal.values, everything.values[keep])
    pd.testing.assert_frame_equal(
        focal.index.reset_index(drop=True),
        everything.index[keep].reset_index(drop=True),
    )


def test_a_perspective_is_ignored_by_a_run_that_writes_none(tmp_path: Path) -> None:
    """As ``align_predictions_to_labels`` does, so one call suits both kinds of run."""
    ds, result = wide_run(tmp_path)

    filtered = load_run_matrix(ds, result, perspective=0)

    np.testing.assert_array_equal(filtered.values, load_run_matrix(ds, result).values)


def test_a_glob_selects_the_columns_it_matches(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)
    everything = load_run_matrix(ds, result)

    matrix = load_run_matrix(ds, result, columns="w1*")

    assert matrix.columns == ("w1", "w10", "w11")
    for name in matrix.columns:
        np.testing.assert_array_equal(
            matrix.values[:, matrix.columns.index(name)],
            everything.values[:, everything.columns.index(name)],
        )


def test_a_list_keeps_the_order_it_was_given_and_takes_a_column_once(
    tmp_path: Path,
) -> None:
    ds, result = wide_run(tmp_path)

    matrix = load_run_matrix(ds, result, columns=["count", "w3", "w*", "w3"])

    assert matrix.columns[:2] == ("count", "w3")
    assert matrix.columns[2:] == tuple(f"w{k}" for k in range(N_WIDE) if k != 3)


def test_an_explicit_name_can_select_a_bookkeeping_column(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)

    matrix = load_run_matrix(ds, result, columns="fps")

    assert matrix.columns == ("fps",)
    assert set(matrix.values[:, 0]) == {30.0}


@pytest.mark.parametrize("pattern", ["nope*", "note", "frame"])
def test_a_pattern_that_matches_no_numeric_column_is_refused(
    tmp_path: Path, pattern: str
) -> None:
    """A string column and an identity column are not values, and the index has frame."""
    ds, result = wide_run(tmp_path)

    with pytest.raises(ValueError, match="matches no numeric column"):
        _ = load_run_matrix(ds, result, columns=pattern)


def test_the_dtype_is_the_one_asked_for(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)

    wide = load_run_matrix(ds, result, dtype=np.float64)

    assert wide.values.dtype == np.float64
    sources = [
        ResultColumn(feature=result.feature, run_id=result.run_id, column=name)
        for name in wide.columns
    ]
    np.testing.assert_array_equal(
        wide.values, load_values(ds, sources)[list(wide.columns)].to_numpy(np.float64)
    )


@pytest.mark.parametrize("dtype", [np.int64, "int8", bool, str])
def test_a_dtype_that_cannot_hold_a_missing_value_is_refused(
    tmp_path: Path, dtype: object
) -> None:
    ds, result = wide_run(tmp_path)

    with pytest.raises(ValueError, match="floating"):
        _ = load_run_matrix(ds, result, dtype=dtype)  # pyright: ignore[reportArgumentType]


def test_a_scope_narrows_the_entries_that_are_read(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)
    everything = load_run_matrix(ds, result)

    by_group = load_run_matrix(ds, result, scope=Scope(groups=["g1"]))
    by_entry = load_run_matrix(
        ds, result, scope=Scope(entries=[("g2", "s1"), ("", "solo")])
    )

    assert by_group.entries == (("g1", "s1"), ("g1", "s2"))
    assert set(by_group.index["group"]) == {"g1"}
    assert by_entry.entries == (("", "solo"), ("g2", "s1"))
    keep = everything.index["group"].isin(["", "g2"]).to_numpy() & (
        everything.index["sequence"].isin(["solo", "s1"]).to_numpy()
    )
    np.testing.assert_array_equal(by_entry.values, everything.values[keep])


def test_the_rows_follow_group_and_sequence_whatever_the_index_order(
    tmp_path: Path,
) -> None:
    ds, result = wide_run(tmp_path)
    before = load_run_matrix(ds, result)
    assert before.entries == (("", "solo"), ("g1", "s1"), ("g1", "s2"), ("g2", "s1"))

    rewrite_index(ds, result, lambda frame: frame.iloc[::-1].reset_index(drop=True))
    after = load_run_matrix(ds, result)

    np.testing.assert_array_equal(after.values, before.values)
    pd.testing.assert_frame_equal(after.index, before.index)
    assert after.entries == before.entries


def test_small_batches_give_the_same_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry is read a batch at a time, and a batch edge is not a row edge."""
    ds, result = pair_run(tmp_path)
    whole = load_run_matrix(ds, result, perspective=1)

    monkeypatch.setattr(run_matrix_module, "BATCH_ROWS", 3)
    batched = load_run_matrix(ds, result, perspective=1)

    np.testing.assert_array_equal(batched.values, whole.values)
    pd.testing.assert_frame_equal(batched.index, whole.index)


# --- which run is read ------------------------------------------------------


def test_an_unpinned_result_reads_the_run_the_one_rule_names(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("g", "a"), ("g", "b"), n_rows=6)
    first = run_speed(ds, 1)
    second = run_speed(ds, 2)
    feature = "speed-angvel__from__tracks"
    assert first != second

    unpinned = load_run_matrix(ds, Result(feature=feature))
    pinned = load_run_matrix(ds, Result(feature=feature, run_id=first))

    assert unpinned.run_id == current_run_id(ds, feature)
    assert pinned.run_id == first
    assert len(unpinned.values) == len(pinned.values) > 0


def test_a_run_the_index_does_not_hold_is_refused(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)

    with pytest.raises(FileNotFoundError, match="0.1-0000000000"):
        _ = load_run_matrix(ds, Result(feature=result.feature, run_id="0.1-0000000000"))


def test_a_feature_that_never_ran_is_refused_with_the_features_that_did(
    tmp_path: Path,
) -> None:
    ds, result = wide_run(tmp_path)

    with pytest.raises(FileNotFoundError, match=result.feature):
        _ = load_run_matrix(ds, Result(feature="wide"))


# --- a run that is not whole ------------------------------------------------


def test_an_output_the_index_lists_and_disk_lacks_is_refused(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)
    output_of(ds, result, "g1", "s2").unlink()

    with pytest.raises(FileNotFoundError, match=r"g1/s2|g1__s2"):
        _ = load_run_matrix(ds, result)

    partial = load_run_matrix(ds, result, allow_partial=True)
    assert partial.missing == (("g1", "s2"),)
    assert ("g1", "s2") not in partial.entries
    assert len(partial.entries) == 3


def test_a_read_that_avoids_the_missing_output_succeeds(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)
    output_of(ds, result, "g1", "s2").unlink()

    matrix = load_run_matrix(ds, result, scope=Scope(entries=[("g1", "s1")]))

    assert matrix.entries == (("g1", "s1"),)
    assert matrix.missing == ()


def test_a_run_whose_outputs_are_all_gone_names_the_repair(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)
    for group, sequence in ENTRIES:
        output_of(ds, result, group, sequence).unlink()

    with pytest.raises(FileNotFoundError, match="make_portable"):
        _ = load_run_matrix(ds, result, allow_partial=True)


def test_a_run_that_is_not_finished_is_refused(tmp_path: Path) -> None:
    """A run that lost an entry, or is still going, never gets ``finished_at``."""
    ds, result = wide_run(tmp_path)
    rewrite_index(ds, result, lambda frame: frame.assign(finished_at=""))

    with pytest.raises(ValueError, match="not finished"):
        _ = load_run_matrix(ds, result)

    matrix = load_run_matrix(ds, result, allow_partial=True)
    assert matrix.finished is False
    assert matrix.missing == ()
    assert len(matrix.entries) == len(ENTRIES)


def test_a_finished_run_reports_itself_whole(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)

    matrix = load_run_matrix(ds, result)

    assert matrix.finished is True
    assert matrix.missing == ()


def test_a_scope_naming_an_entry_the_run_lacks_is_refused(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)
    scope = Scope(entries=[("g1", "s1"), ("g9", "never")])

    with pytest.raises(ValueError, match=r"g9/never"):
        _ = load_run_matrix(ds, result, scope=scope)

    partial = load_run_matrix(ds, result, scope=scope, allow_partial=True)
    assert partial.entries == (("g1", "s1"),)
    assert partial.missing == (("g9", "never"),)


def test_a_selection_of_nothing_is_refused_even_when_partial_is_allowed(
    tmp_path: Path,
) -> None:
    ds, result = wide_run(tmp_path)

    with pytest.raises(ValueError, match="selects no output"):
        _ = load_run_matrix(
            ds, result, scope=Scope(groups=["absent"]), allow_partial=True
        )


def test_entries_that_write_different_columns_are_refused(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)
    path = output_of(ds, result, "g1", "s2")
    frame = pd.read_parquet(path)
    frame.drop(columns=["w5"]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match=r"g1/s2.*w5|w5.*g1/s2"):
        _ = load_run_matrix(ds, result)


def test_entries_that_order_the_same_columns_differently_are_read_by_name(
    tmp_path: Path,
) -> None:
    ds, result = wide_run(tmp_path)
    before = load_run_matrix(ds, result)
    path = output_of(ds, result, "g1", "s2")
    frame = pd.read_parquet(path)
    frame[list(frame.columns[::-1])].to_parquet(path, index=False)

    after = load_run_matrix(ds, result)

    np.testing.assert_array_equal(after.values, before.values)


def test_an_output_that_cannot_be_read_is_named(tmp_path: Path) -> None:
    """An interrupted write leaves a file with no footer."""
    ds, result = wide_run(tmp_path)
    path = output_of(ds, result, "g1", "s1")
    path.write_bytes(path.read_bytes()[:64])

    with pytest.raises(ParquetUnreadable, match="g1__s1"):
        _ = load_run_matrix(ds, result)


# --- the batch reader -------------------------------------------------------


def test_the_batch_reader_yields_the_named_columns_in_file_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "table.parquet"
    pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [0.5, np.nan, 2.5, 3.5, 4.5],
            "c": [True, False, True, False, True],
            "unread": list("abcde"),
        }
    ).to_parquet(path, index=False)

    batches = list(iter_column_batches(path, ["c", "b"], 2))

    assert [len(batch["b"]) for batch in batches] == [2, 2, 1]
    assert all(set(batch) == {"b", "c"} for batch in batches)
    np.testing.assert_array_equal(
        np.concatenate([batch["b"] for batch in batches]),
        [0.5, np.nan, 2.5, 3.5, 4.5],
    )
    assert batches[0]["c"].dtype == np.bool_


def test_the_batch_reader_reads_nothing_for_no_columns(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(path, index=False)

    assert list(iter_column_batches(path, [], 2)) == []


# --- memory -----------------------------------------------------------------


def test_a_read_that_would_not_fit_is_refused_before_any_value_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, result = wide_run(tmp_path)
    reads: list[tuple[str, ...]] = []
    real = run_matrix_module.iter_column_batches

    def recording(
        path: Path, names: Sequence[str], batch_size: int
    ) -> Generator[dict[str, npt.NDArray[Any]]]:
        reads.append(tuple(names))
        return real(path, names, batch_size)

    monkeypatch.setattr(run_matrix_module, "iter_column_batches", recording)
    monkeypatch.setattr(run_matrix_module, "host_memory_bytes", lambda: (4096, 1024))

    with pytest.raises(MemoryError) as refusal:
        _ = load_run_matrix(ds, result)

    message = str(refusal.value)
    assert result.feature in message and str(result.run_id) in message
    assert "scope=" in message and "columns=" in message and "dtype=" in message
    assert reads == []


def test_the_refusal_counts_the_rows_that_survive_the_perspective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimate is of the matrix returned, not of the files read."""
    ds, result = pair_run(tmp_path)
    monkeypatch.setattr(run_matrix_module, "host_memory_bytes", lambda: (4096, 16))

    with pytest.raises(MemoryError, match=r"\b32 rows"):
        _ = load_run_matrix(ds, result)
    with pytest.raises(MemoryError, match=r"\b16 rows"):
        _ = load_run_matrix(ds, result, perspective=0)


def test_a_read_that_fits_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, result = wide_run(tmp_path)
    monkeypatch.setattr(run_matrix_module, "host_memory_bytes", lambda: (2**40, 2**40))

    assert len(load_run_matrix(ds, result).values) > 0


def test_a_host_that_reports_no_memory_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``psutil`` the host reports ``(0, 0)``, which is unknown and not empty."""
    ds, result = wide_run(tmp_path)
    monkeypatch.setattr(run_matrix_module, "host_memory_bytes", lambda: (0, 0))

    assert len(load_run_matrix(ds, result).values) > 0


def test_the_peak_is_the_matrix_and_not_a_copy_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entries are written into one array, and not concatenated into a second."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *[("g", f"s{k}") for k in range(6)], n_rows=800, n_ids=2)
    result = ds.run_feature(VeryWideFeature())
    monkeypatch.setattr(run_matrix_module, "BATCH_ROWS", 200)
    _ = load_run_matrix(ds, result)

    tracemalloc.start()
    try:
        matrix = load_run_matrix(ds, result)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    identity_bytes = int(matrix.index.memory_usage(deep=False).sum())
    assert matrix.values.shape == (9600, 202)
    assert peak < matrix.values.nbytes + 4 * identity_bytes + 2**20


# --- a read writes nothing --------------------------------------------------


def test_reading_a_run_changes_no_file_of_the_dataset(tmp_path: Path) -> None:
    ds, result = wide_run(tmp_path)
    pair = ds.run_feature(PairClusterFeature())
    base = tmp_path / "ds"
    reader = open_dataset(base, ensure_roots=False)
    before_digest = tree_digest(base)
    before_stats = tree_snapshot(base)

    _ = load_run_matrix(reader, result)
    _ = load_run_matrix(reader, Result(feature=result.feature), columns="w1*")
    _ = load_run_matrix(reader, pair, perspective=0, scope=Scope(groups=["g1"]))
    _ = load_run_matrix(reader, result, dtype=np.float64, allow_partial=True)

    assert tree_digest(base) == before_digest
    assert tree_snapshot(base) == before_stats
