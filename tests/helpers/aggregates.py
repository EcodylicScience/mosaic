"""A run of an aggregate over tracks whose label varies inside one entry.

``keep_columns`` carries a label to the output when the rows reduced together
share it. The run here has two entries of two individuals: ``STEADY`` labels
every row ``a``, and ``MIXED`` labels a row ``a`` or ``b`` so that the label
varies within one frame, within one individual and within the entry. Each
aggregate then loses ``MIXED`` whatever it groups by, and writes ``STEADY``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.api.extensions import ExtensionArray

from mosaic.core.entry import Entry
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.types import COLUMNS as C, Feature, Result
from mosaic.runlog import EntryErrorRecord, read_entry_errors, run_log_dir
from tests.helpers.datasets import make_dataset
from tests.helpers.spies import observed
from tests.helpers.tracks import add_track_sequences

__all__ = [
    "MIXED",
    "ROI_VALUES",
    "STEADY",
    "TIME_SENTENCE",
    "LabeledTracksRun",
    "record_applied_sequences",
    "run_over_edited_tracks",
    "run_over_labeled_tracks",
    "stepped_individual",
    "constant_time_in_mixed",
    "without_time",
    "without_time_in_mixed",
]

STEADY: Entry = ("g1", "steady")
MIXED: Entry = ("g1", "mixed")

TIME_SENTENCE = (
    "duration and share weigh each row by its seconds, read from the time column: "
    "give the step an input with a time column, such as tracks."
)
"""The end of the error that ``duration`` and ``share`` raise without a ``time``."""

ROI_VALUES = ["center", "none", None, "center", "feeder"]
"""Five rows at 0.1 s: a null among four values of three kinds."""

_EXECUTION_ID = "LABELEDRUN1"
_LABEL = "strain"


@dataclass(frozen=True)
class LabeledTracksRun:
    """What a feature run over the labeled tracks wrote and lost.

    Attributes:
        result: The run's result.
        errors: One record per ``entry_error`` event of the run-log.
        written: The entry keys whose output file the run holds, sorted.
    """

    result: Result[str]
    errors: list[EntryErrorRecord]
    written: list[str]


def run_over_labeled_tracks(tmp_path: Path, feature: Feature) -> LabeledTracksRun:
    """Run *feature* over the two labeled entries.

    Args:
        tmp_path: The directory the dataset is made under.
        feature: A feature that reads ``tracks`` and has a ``name``.

    Returns:
        The result, the run-log's entry errors and the entries written.
    """
    return run_over_edited_tracks(tmp_path, feature, _label_entry)


def run_over_edited_tracks(
    tmp_path: Path,
    feature: Feature,
    edit: Callable[[Entry, pd.DataFrame], pd.DataFrame],
) -> LabeledTracksRun:
    """Run *feature* over the two entries, each tracks table rewritten by *edit*.

    Args:
        tmp_path: The directory the dataset is made under.
        feature: A feature that reads ``tracks`` and has a ``name``.
        edit: Returns the table to write for an entry, given the entry and its
            table of six frames of two individuals.

    Returns:
        The result, the run-log's entry errors and the entries written.
    """
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, STEADY, MIXED, n_rows=6, n_ids=2)
    for entry in (STEADY, MIXED):
        path = ds.get_root("tracks") / f"{make_entry_key(*entry)}.parquet"
        edit(entry, pd.read_parquet(path)).to_parquet(path)
    result = ds.run_feature(feature, execution_id=_EXECUTION_ID)
    assert result.run_id is not None
    run_root = ds.get_root("features") / result.feature / result.run_id
    written = sorted(path.stem for path in run_root.glob("*.parquet"))
    errors = read_entry_errors(run_log_dir(ds.base_dir), _EXECUTION_ID)["items"]
    return LabeledTracksRun(result=result, errors=errors, written=written)


def record_applied_sequences(
    monkeypatch: pytest.MonkeyPatch, feature_type: type[Feature]
) -> list[str]:
    """Record the sequence of every frame that *feature_type* applies.

    The list fills as the test runs the feature, until *monkeypatch* restores
    ``apply`` at the test's end.
    """
    applied: list[str] = []

    def note(self: Feature, df: pd.DataFrame) -> None:
        del self
        applied.append(str(df[C.seq_col].iloc[0]))

    monkeypatch.setattr(feature_type, "apply", observed(feature_type.apply, note))
    return applied


def _label_entry(entry: Entry, table: pd.DataFrame) -> pd.DataFrame:
    """Add the ``strain`` label to an entry's tracks table."""
    flipped = (table["frame"] + table["id"]) % 2 == 1
    return table.assign(**{_LABEL: np.where(flipped & (entry == MIXED), "b", "a")})


def constant_time_in_mixed(entry: Entry, table: pd.DataFrame) -> pd.DataFrame:
    """Give the ``MIXED`` entry a ``time`` that never advances."""
    return table.assign(time=0.0) if entry == MIXED else table


def without_time_in_mixed(entry: Entry, table: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``time`` column from the ``MIXED`` entry."""
    return table.drop(columns="time") if entry == MIXED else table


def without_time(entry: Entry, table: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``time`` column from every entry."""
    del entry
    return table.drop(columns="time")


def stepped_individual(
    columns: Mapping[str, Sequence[object] | pd.Series | ExtensionArray],
    *,
    step: float = 0.1,
) -> pd.DataFrame:
    """The rows of individual 0 of entry ``G``/``S1`` at a steady *step* in seconds.

    Args:
        columns: Each column's values, one per row. Every column has as many
            values as there are rows.
        step: The seconds from one frame to the next.

    Returns:
        A table with ``frame`` counting from 0, ``time``, ``id``, ``group``,
        ``sequence`` and *columns*.
    """
    length = len(next(iter(columns.values())))
    return pd.DataFrame(
        {
            "frame": np.arange(length),
            "time": np.arange(length) * step,
            "id": 0,
            "group": "G",
            "sequence": "S1",
            **columns,
        }
    )
