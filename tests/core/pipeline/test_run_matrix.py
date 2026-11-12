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
import json
import tracemalloc
import zlib
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pytest

from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core import pipeline as pipeline_package
from mosaic.core.pipeline import load_run_matrix as exported_load_run_matrix
from mosaic.core.pipeline import run_matrix as run_matrix_module
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.index import feature_index_path, feature_run_root
from mosaic.core.pipeline.parquet_reader import ParquetUnreadable, iter_column_batches
from mosaic.core.pipeline.run import load_values
from mosaic.core.pipeline.run_matrix import (
    RunMatrix,
    RunRead,
    RunTooLarge,
    load_run_matrix,
    plan_run_read,
    read_run,
)
from mosaic.core.pipeline.track_universe import AmbiguousTrackLeaf, current_run_id
from mosaic.core.pipeline.types import Result, ResultColumn
from mosaic.core.scope import Scope
from tests.helpers import (
    PairClusterFeature,
    PlainFeature,
    add_track_sequences,
    labeled_run,
    make_dataset,
    rewrite_index,
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


class PairWideFeature(PairClusterFeature):
    """Both perspectives of a pair with six numeric columns beside ``cluster``."""

    name = "prov-pair-wide"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        table = super().apply(df)
        for k in range(6):
            table[f"w{k}"] = (
                table["frame"] * 10.0 + table["perspective"] * 0.5 + k / 100.0
            )
        return table


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
    exported = {
        name: getattr(pipeline_package, name) for name in pipeline_package.__all__
    }
    assert exported["RunRead"] is RunRead
    assert exported["plan_run_read"] is plan_run_read
    assert exported["read_run"] is read_run


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


def test_a_glob_selects_the_bookkeeping_columns_the_default_leaves_out(
    tmp_path: Path,
) -> None:
    """``None`` is the payload, and a pattern is what it matches among the numeric columns."""
    ds, result = wide_run(tmp_path)

    everything = load_run_matrix(ds, result, columns="*")

    assert "fps" in everything.columns
    assert set(everything.columns) - set(load_run_matrix(ds, result).columns) == {"fps"}
    assert "note" not in everything.columns


def test_a_name_with_glob_characters_selects_itself_and_not_what_it_would_match(
    tmp_path: Path,
) -> None:
    """``a[1]`` as a glob matches ``a1``, and as the name of a column it is that column."""
    ds, result = wide_run(tmp_path)
    for group, sequence in ENTRIES:
        path = output_of(ds, result, group, sequence)
        frame = pd.read_parquet(path)
        frame["a[1]"] = 1.0
        frame["a1"] = 2.0
        frame.to_parquet(path, index=False)

    literal = load_run_matrix(ds, result, columns="a[1]")
    globbed = load_run_matrix(ds, result, columns="a[0-9]")

    assert literal.columns == ("a[1]",)
    assert set(literal.values[:, 0]) == {1.0}
    assert globbed.columns == ("a1",)


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


def test_a_file_that_shrinks_between_the_count_and_the_read_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The array was sized from the footers, so a different file cannot fill it."""
    ds, result = wide_run(tmp_path)
    path = output_of(ds, result, "g1", "s1")
    real = run_matrix_module.usable_memory_bytes

    def shrinking() -> tuple[int, int]:
        frame = pd.read_parquet(path)
        frame.iloc[: len(frame) // 2].to_parquet(path, index=False)
        return real()

    monkeypatch.setattr(run_matrix_module, "usable_memory_bytes", shrinking)

    with pytest.raises(
        RuntimeError, match=r"g1/s1 changed while it was being read"
    ) as raised:
        _ = load_run_matrix(ds, result)

    assert not isinstance(raised.value, ClassifiedError)


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
    monkeypatch.setattr(run_matrix_module, "usable_memory_bytes", lambda: (4096, 1024))

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
    monkeypatch.setattr(run_matrix_module, "usable_memory_bytes", lambda: (4096, 16))

    with pytest.raises(MemoryError, match=r"\b32 rows"):
        _ = load_run_matrix(ds, result)
    with pytest.raises(MemoryError, match=r"\b16 rows"):
        _ = load_run_matrix(ds, result, perspective=0)


def test_a_read_that_fits_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, result = wide_run(tmp_path)
    monkeypatch.setattr(
        run_matrix_module, "usable_memory_bytes", lambda: (2**40, 2**40)
    )

    assert len(load_run_matrix(ds, result).values) > 0


def test_a_host_that_reports_no_memory_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``psutil`` the host reports ``(0, 0)``, which is unknown and not empty."""
    ds, result = wide_run(tmp_path)
    monkeypatch.setattr(run_matrix_module, "usable_memory_bytes", lambda: (0, 0))

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


def very_wide_run(tmp_path: Path) -> tuple[Dataset, Result]:
    """Three entries of 1,600 rows, each file holding 204 columns of numbers."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *[("g", f"s{k}") for k in range(3)], n_rows=800, n_ids=2)
    return ds, ds.run_feature(VeryWideFeature())


def arrow_held_bytes() -> int:
    """What Arrow's memory pool holds now, which pyarrow's missing stubs leave unknown."""
    return int(pa.total_allocated_bytes())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def watch_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[str, ...], int]]:
    """Record each read of a file as its column names and what Arrow holds during it."""
    seen: list[tuple[tuple[str, ...], int]] = []
    real = iter_column_batches

    def watching(
        path: Path, names: Sequence[str], batch_size: int
    ) -> Generator[dict[str, npt.NDArray[Any]]]:
        held = 0
        for batch in real(path, names, batch_size):
            held = max(held, arrow_held_bytes())
            yield batch
        seen.append((tuple(names), held))

    monkeypatch.setattr(run_matrix_module, "iter_column_batches", watching)
    return seen


def test_a_file_is_read_a_block_of_columns_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every column once for each entry, and the same matrix as one pass gives."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *ENTRIES, n_rows=4, n_ids=2)
    result = ds.run_feature(PairWideFeature())
    whole = {p: load_run_matrix(ds, result, perspective=p) for p in (None, 0, 1)}
    file_rows = 8
    cell_bytes = run_matrix_module.DECODE_FACTOR * 8 * file_rows
    monkeypatch.setattr(run_matrix_module, "BLOCK_BYTES", 3 * cell_bytes)
    monkeypatch.setattr(run_matrix_module, "BATCH_ROWS", 3)

    for perspective, expected in whole.items():
        reads = watch_reads(monkeypatch)
        blocked = load_run_matrix(ds, result, perspective=perspective)

        np.testing.assert_array_equal(blocked.values, expected.values)
        pd.testing.assert_frame_equal(blocked.index, expected.index)
        values = {"cluster", *(f"w{k}" for k in range(6))}
        passes = [
            names
            for names, _ in reads
            if names != ("perspective",) or perspective is None
        ]
        assert len(passes) == 3 * len(ENTRIES)
        for first in range(0, len(passes), 3):
            entry = passes[first : first + 3]
            assert sorted(
                name for names in entry for name in names if name in values
            ) == (sorted(values))
            assert all(len([n for n in names if n in values]) <= 3 for names in entry)
            assert "frame" in entry[0] and "frame" not in entry[1] + entry[2]


def test_the_columns_that_pyarrow_decodes_at_once_follow_the_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch size does not bound what pyarrow holds: it decodes a row group whole."""
    ds, result = very_wide_run(tmp_path)
    whole_reads = watch_reads(monkeypatch)
    baseline = arrow_held_bytes()
    whole = load_run_matrix(ds, result)
    file_rows = len(whole.values) // len(whole.entries)
    cell_bytes = run_matrix_module.DECODE_FACTOR * 8 * file_rows
    monkeypatch.setattr(run_matrix_module, "BLOCK_BYTES", 20 * cell_bytes)
    block_reads = watch_reads(monkeypatch)

    blocked = load_run_matrix(ds, result)

    np.testing.assert_array_equal(blocked.values, whole.values)
    held_whole = max(held for _, held in whole_reads) - baseline
    held_blocked = max(held for _, held in block_reads) - baseline
    assert len(block_reads) > len(whole_reads)
    assert held_whole > 8 * file_rows * len(whole.columns)
    assert held_blocked < held_whole / 3


def test_the_refusal_counts_what_pyarrow_decodes_of_the_largest_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same file fits when it is read in blocks, and does not when it is read whole."""
    ds, result = very_wide_run(tmp_path)
    whole = load_run_matrix(ds, result)
    n_rows, n_columns = whole.values.shape
    file_rows = n_rows // len(whole.entries)
    cell_bytes = run_matrix_module.DECODE_FACTOR * 8 * file_rows
    array_bytes = n_rows * n_columns * whole.values.itemsize
    available = int(1.5 * array_bytes / (1.0 - run_matrix_module.MEMORY_HEADROOM))
    monkeypatch.setattr(
        run_matrix_module, "usable_memory_bytes", lambda: (2 * available, available)
    )

    with pytest.raises(MemoryError):
        _ = load_run_matrix(ds, result)
    monkeypatch.setattr(run_matrix_module, "BLOCK_BYTES", 10 * cell_bytes)

    blocked = load_run_matrix(ds, result)

    np.testing.assert_array_equal(blocked.values, whole.values)


# --- the plan and the read -------------------------------------------------


def assert_same_matrix(actual: RunMatrix, expected: RunMatrix) -> None:
    """Every field of *actual* equals that field of *expected*."""
    assert (actual.feature, actual.run_id) == (expected.feature, expected.run_id)
    assert actual.entries == expected.entries
    assert actual.columns == expected.columns
    assert actual.missing == expected.missing
    assert actual.finished == expected.finished
    assert actual.values.dtype == expected.values.dtype
    np.testing.assert_array_equal(actual.values, expected.values)
    pd.testing.assert_frame_equal(actual.index, expected.index)


@pytest.mark.parametrize("make_run", [wide_run, pair_run], ids=["wide", "pair"])
def test_reading_the_plan_is_loading_the_run(
    tmp_path: Path, make_run: Callable[[Path], tuple[Dataset, Result]]
) -> None:
    ds, result = make_run(tmp_path)

    read = plan_run_read(ds, result)

    assert isinstance(read, RunRead)
    assert_same_matrix(read_run(read), load_run_matrix(ds, result))
    assert_same_matrix(
        read_run(plan_run_read(ds, result, perspective=0), dtype=np.float64),
        load_run_matrix(ds, result, perspective=0, dtype=np.float64),
    )


LABELED_ENTRIES = (("g", "a"), ("g", "b"))


def labeled_table(
    entry: tuple[str, str], n_rows: int = 3, **columns: object
) -> pd.DataFrame:
    """Rows of one entry: ``frame``, ``id``, a ``value`` and the *columns* given."""
    offset = 10.0 * LABELED_ENTRIES.index(entry)
    return pd.DataFrame(
        {
            "frame": np.arange(n_rows, dtype=np.int64),
            "id": np.zeros(n_rows, dtype=np.int64),
            "value": np.arange(n_rows, dtype=np.float64) + offset,
            **columns,
        }
    )


def strain_and_batch_run(tmp_path: Path) -> tuple[Dataset, Result]:
    """A run whose entries write a text key ``strain`` and an integer key ``batch``."""
    return labeled_run(
        tmp_path,
        {
            ("g", "a"): labeled_table(
                ("g", "a"), strain=["A", "A", "A"], batch=np.array([1, 1, 1])
            ),
            ("g", "b"): labeled_table(
                ("g", "b"), strain=["B", "B", "B"], batch=np.array([2, 2, 2])
            ),
        },
    )


@pytest.mark.parametrize(
    "columns", [None, "*", ["value"]], ids=["default", "glob", "name"]
)
def test_keys_follow_the_identity_in_the_index_and_are_not_values(
    tmp_path: Path, columns: str | list[str] | None
) -> None:
    ds, result = strain_and_batch_run(tmp_path)

    matrix = read_run(
        plan_run_read(ds, result, columns=columns, keys=["strain", "batch"])
    )

    assert list(matrix.index.columns) == [*KEY, "strain", "batch"]
    assert matrix.columns == ("value",)
    assert matrix.index["strain"].tolist() == ["A"] * 3 + ["B"] * 3
    assert matrix.index["batch"].tolist() == [1] * 3 + [2] * 3
    np.testing.assert_array_equal(
        matrix.values[:, 0], [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]
    )


def test_a_read_without_keys_takes_an_integer_label_as_a_value(
    tmp_path: Path,
) -> None:
    ds, result = strain_and_batch_run(tmp_path)

    matrix = read_run(plan_run_read(ds, result))

    assert matrix.columns == ("value", "batch")
    assert list(matrix.index.columns) == KEY


def test_a_key_is_read_beside_the_perspective_it_is_filtered_by(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *ENTRIES, n_rows=4, n_ids=2)
    result = ds.run_feature(PairWideFeature())

    matrix = read_run(plan_run_read(ds, result, perspective=1, keys=["cluster"]))

    assert set(matrix.index["perspective"]) == {1}
    assert matrix.index["cluster"].tolist() == (matrix.index["frame"] % 2).tolist()
    assert matrix.columns == tuple(f"w{k}" for k in range(6))


def test_each_entry_records_the_kind_of_each_key(tmp_path: Path) -> None:
    """An all-null label is Arrow ``null``, and an integer label with a null is floating."""
    ds, result = labeled_run(
        tmp_path,
        {
            ("g", "a"): labeled_table(
                ("g", "a"),
                strain=[None, None, None],
                batch=np.array([1.0, np.nan, 1.0]),
            ),
            ("g", "b"): labeled_table(
                ("g", "b"), strain=["B", "B", "B"], batch=np.array([2, 2, 2])
            ),
        },
    )

    read = plan_run_read(ds, result, keys=["strain", "batch"])

    assert read.keys == ("strain", "batch")
    assert read.key_kinds == (
        (("strain", "null"), ("batch", "floating")),
        (("strain", "string"), ("batch", "integer")),
    )
    assert dict(read.footer_columns)["value"] == "floating"
    matrix = read_run(read)
    strain = matrix.index["strain"]
    assert strain.isna().tolist() == [True] * 3 + [False] * 3
    assert strain.iloc[3:].tolist() == ["B"] * 3


@pytest.mark.parametrize(
    "keys",
    [
        ["strain", "strain"],
        ["group", "strain"],
        ["sequence", "strain"],
        ["id", "strain"],
        ["strain", "group", "sequence", "id", "strain"],
    ],
    ids=["twice", "group", "sequence", "id", "all"],
)
def test_a_key_already_in_the_index_is_read_once(
    tmp_path: Path, keys: list[str]
) -> None:
    """A key named twice, or naming a column of the identity the run writes."""
    ds, result = strain_and_batch_run(tmp_path)

    read = plan_run_read(ds, result, keys=keys)
    matrix = read_run(read)

    assert read.keys == ("strain",)
    assert read.key_kinds == ((("strain", "string"),),) * 2
    assert list(matrix.index.columns) == [*KEY, "strain"]
    assert matrix.index["strain"].tolist() == ["A"] * 3 + ["B"] * 3


def test_an_identity_column_the_run_does_not_write_is_a_missing_key(
    tmp_path: Path,
) -> None:
    ds, result = strain_and_batch_run(tmp_path)

    with pytest.raises(ValueError, match="'id1'") as raised:
        _ = plan_run_read(ds, result, keys=["id1"])

    assert isinstance(raised.value, ClassifiedError)
    assert raised.value.error_kind == "invalid"
    detail = raised.value.error_detail()
    assert detail is not None
    columns = detail["columns"]
    assert isinstance(columns, list)
    assert {"id", "strain", "value"} <= set(columns)


def test_a_key_missing_from_every_entry_lists_the_columns(tmp_path: Path) -> None:
    ds, result = strain_and_batch_run(tmp_path)

    with pytest.raises(ClassifiedError) as raised:
        _ = plan_run_read(ds, result, keys=["sex"])

    assert raised.value.error_kind == "invalid"
    assert "'sex'" in str(raised.value) and "strain" in str(raised.value)
    detail = raised.value.error_detail()
    assert detail is not None
    columns = detail["columns"]
    assert isinstance(columns, list)
    assert {"strain", "batch", "value"} <= set(columns)


@pytest.mark.parametrize("lacking", LABELED_ENTRIES, ids=["first", "second"])
def test_a_key_missing_from_one_entry_names_the_entries(
    tmp_path: Path, lacking: tuple[str, str]
) -> None:
    tables = {
        entry: labeled_table(entry, strain=["S"] * 3) for entry in LABELED_ENTRIES
    }
    tables[lacking] = labeled_table(lacking)
    ds, result = labeled_run(tmp_path, tables)

    with pytest.raises(ClassifiedError) as raised:
        _ = plan_run_read(ds, result, keys=["strain"])

    assert raised.value.error_kind == "conflict"
    detail = raised.value.error_detail()
    assert detail is not None
    assert detail["entries"] == ["g/a", "g/b"]
    assert detail["columns"] == ["strain"]


def estimated_bytes(read: RunRead, monkeypatch: pytest.MonkeyPatch) -> int:
    """The memory that :func:`read_run` counts for *read*, found by where it refuses.

    With no headroom a read goes ahead exactly when its estimate is at most the
    memory available, so the smallest amount that lets it go ahead is the estimate.
    The read stops before its first value once the check passes.
    """

    class Admitted(Exception):
        pass

    def admitted(*_: object) -> Generator[dict[str, npt.NDArray[Any]]]:
        raise Admitted
        yield {}

    monkeypatch.setattr(run_matrix_module, "MEMORY_HEADROOM", 0.0)
    monkeypatch.setattr(run_matrix_module, "iter_column_batches", admitted)
    low, high = 0, 2**40
    while low < high:
        middle = (low + high) // 2
        monkeypatch.setattr(
            run_matrix_module, "usable_memory_bytes", lambda: (middle, middle)
        )
        try:
            _ = read_run(read)
        except RunTooLarge:
            low = middle + 1
        except Admitted:
            high = middle
    return low


def test_each_key_counts_eight_bytes_a_row_as_an_identity_column_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frame holds a copy of each cell, and pyarrow decodes it beside the values."""
    ds, result = strain_and_batch_run(tmp_path)
    plain = plan_run_read(ds, result, columns="value")
    keyed = plan_run_read(ds, result, columns="value", keys=["strain", "batch"])
    n_rows = sum(plain.layout.entry_rows)
    longest = max(plain.layout.file_rows)

    grown = estimated_bytes(keyed, monkeypatch) - estimated_bytes(plain, monkeypatch)

    per_key = 2 * 8 * n_rows + run_matrix_module.DECODE_FACTOR * 8 * longest
    assert grown == 2 * per_key


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


# --- classified refusals ----------------------------------------------------


class KeysOnlyFeature(PlainFeature):
    """A table whose only numeric columns are identity and bookkeeping."""

    name = "prov-keys-only"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": df["frame"].to_numpy(),
                "id": df["id"].to_numpy(),
                "group": df["group"].to_numpy(),
                "sequence": df["sequence"].to_numpy(),
                "fps": np.full(len(df), 30.0),
            }
        )


@dataclass(frozen=True, slots=True)
class Refusal:
    """One way the reader refuses, and what a caller can tell about it."""

    provoke: Callable[[Path, pytest.MonkeyPatch], object]
    base: type[BaseException]
    kind: ErrorKind
    name: str


def _no_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, _ = wide_run(tmp_path)
    return load_run_matrix(ds, Result(feature="wide"))


def _pinned_run_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, result = wide_run(tmp_path)
    return load_run_matrix(ds, Result(feature=result.feature, run_id="0.1-0000000000"))


def _index_lists_no_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, result = wide_run(tmp_path)
    rewrite_index(ds, result, lambda frame: frame.iloc[0:0])
    return load_run_matrix(ds, Result(feature=result.feature))


def _two_leaves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("g", "a"), n_rows=6)
    feature = "speed-angvel__from__tracks"
    first, second, _ = (run_speed(ds, step) for step in (1, 2, 3))
    params = feature_run_root(ds, feature, second) / "params.json"
    record = json.loads(params.read_text())
    record["_resolved"] = [
        *record.get("_resolved", []),
        {"where": "inputs[0]", "feature": feature, "run_id": first},
    ]
    _ = params.write_text(json.dumps(record))
    return load_run_matrix(ds, Result(feature=feature))


def _scope_selects_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, result = wide_run(tmp_path)
    return load_run_matrix(ds, result, scope=Scope(groups=["absent"]))


def _only_absent_entries(allow_partial: bool) -> Callable[..., object]:
    def provoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        ds, result = wide_run(tmp_path)
        return load_run_matrix(
            ds,
            result,
            scope=Scope(entries=[("g9", "never")]),
            allow_partial=allow_partial,
        )

    return provoke


def _another_absent_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, result = wide_run(tmp_path)
    scope = Scope(entries=[("g1", "s1"), ("g9", "never")])
    return load_run_matrix(ds, result, scope=scope)


def _unfinished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, result = wide_run(tmp_path)
    rewrite_index(ds, result, lambda frame: frame.assign(finished_at=""))
    return load_run_matrix(ds, result)


def _some_outputs_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, result = wide_run(tmp_path)
    output_of(ds, result, "g1", "s2").unlink()
    return load_run_matrix(ds, result)


def _all_outputs_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, result = wide_run(tmp_path)
    for group, sequence in ENTRIES:
        output_of(ds, result, group, sequence).unlink()
    return load_run_matrix(ds, result, allow_partial=True)


def _pattern_matches_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, result = wide_run(tmp_path)
    return load_run_matrix(ds, result, columns="nope*")


def _no_numeric_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *ENTRIES, n_rows=4, n_ids=2)
    return load_run_matrix(ds, ds.run_feature(KeysOnlyFeature()))


def _entries_write_different_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> object:
    ds, result = wide_run(tmp_path)
    path = output_of(ds, result, "g1", "s2")
    pd.read_parquet(path).drop(columns=["w5"]).to_parquet(path, index=False)
    return load_run_matrix(ds, result)


def _dtype_cannot_hold_nan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, result = wide_run(tmp_path)
    return load_run_matrix(ds, result, dtype=np.int64)


def _read_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    ds, result = wide_run(tmp_path)
    monkeypatch.setattr(run_matrix_module, "usable_memory_bytes", lambda: (4096, 1024))
    return load_run_matrix(ds, result)


READER_REFUSALS = [
    pytest.param(
        Refusal(_no_runs, FileNotFoundError, "not_found", "RunNotFound"),
        id="the-feature-has-no-runs",
    ),
    pytest.param(
        Refusal(_pinned_run_absent, FileNotFoundError, "not_found", "RunNotFound"),
        id="a-pinned-run-is-not-in-the-index",
    ),
    pytest.param(
        Refusal(_index_lists_no_run, ValueError, "not_found", "RunIndexEmpty"),
        id="an-unpinned-read-of-an-index-listing-no-run",
    ),
    pytest.param(
        Refusal(_two_leaves, AmbiguousTrackLeaf, "conflict", "AmbiguousRunLeaf"),
        id="two-track-shaped-leaves",
    ),
    pytest.param(
        Refusal(_scope_selects_nothing, ValueError, "invalid", "RunReadInvalid"),
        id="the-scope-selects-no-output",
    ),
    pytest.param(
        Refusal(_only_absent_entries(False), ValueError, "conflict", "RunIncomplete"),
        id="the-scope-names-only-absent-entries",
    ),
    pytest.param(
        Refusal(_only_absent_entries(True), ValueError, "conflict", "RunIncomplete"),
        id="the-scope-names-only-absent-entries-with-allow-partial",
    ),
    pytest.param(
        Refusal(_another_absent_entry, ValueError, "conflict", "RunIncomplete"),
        id="the-scope-names-another-absent-entry",
    ),
    pytest.param(
        Refusal(_unfinished, ValueError, "conflict", "RunIncomplete"),
        id="an-unfinished-run",
    ),
    pytest.param(
        Refusal(_some_outputs_gone, FileNotFoundError, "conflict", "RunOutputsGone"),
        id="some-outputs-are-gone",
    ),
    pytest.param(
        Refusal(_all_outputs_gone, FileNotFoundError, "conflict", "RunOutputsGone"),
        id="all-outputs-are-gone",
    ),
    pytest.param(
        Refusal(_pattern_matches_nothing, ValueError, "invalid", "RunReadInvalid"),
        id="a-pattern-matches-no-numeric-column",
    ),
    pytest.param(
        Refusal(_no_numeric_payload, ValueError, "invalid", "RunReadInvalid"),
        id="a-run-with-no-numeric-payload-column",
    ),
    pytest.param(
        Refusal(
            _entries_write_different_columns,
            ValueError,
            "conflict",
            "RunColumnsConflict",
        ),
        id="entries-write-different-columns",
    ),
    pytest.param(
        Refusal(_dtype_cannot_hold_nan, ValueError, "invalid", "RunReadInvalid"),
        id="a-non-floating-dtype",
    ),
    pytest.param(
        Refusal(_read_too_large, MemoryError, "unavailable", "RunTooLarge"),
        id="a-read-too-large",
    ),
]


@pytest.mark.parametrize("refusal", READER_REFUSALS)
def test_every_refusal_of_the_reader_is_classified_and_keeps_its_base_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal: Refusal
) -> None:
    with pytest.raises(refusal.base) as raised:
        _ = refusal.provoke(tmp_path, monkeypatch)

    error = raised.value
    assert isinstance(error, ClassifiedError)
    assert error.error_kind == refusal.kind
    assert type(error).__name__ == refusal.name
    assert refusal.name in run_matrix_module.__all__
    detail = error.error_detail()
    assert detail is not None
    assert isinstance(detail["feature"], str) and detail["feature"]
    assert "run_id" in detail


def test_the_unpinned_ambiguity_is_caught_as_the_track_universes_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(AmbiguousTrackLeaf) as raised:
        _ = _two_leaves(tmp_path, monkeypatch)

    assert isinstance(raised.value, RuntimeError)
    assert isinstance(raised.value, ClassifiedError)
    assert "Name one by passing its run_id" in str(raised.value)


def test_the_refusal_of_a_partial_run_names_the_entries_and_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ClassifiedError) as raised:
        _ = _another_absent_entry(tmp_path, monkeypatch)

    detail = raised.value.error_detail()
    assert detail is not None
    assert detail["entries"] == ["g9/never"]
    assert isinstance(detail["run_id"], str)


def test_the_refusal_of_entries_with_different_columns_names_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ClassifiedError) as raised:
        _ = _entries_write_different_columns(tmp_path, monkeypatch)

    detail = raised.value.error_detail()
    assert detail is not None
    assert detail["entries"] == ["/solo", "g1/s2"]
    assert detail["columns"] == ["w5"]
