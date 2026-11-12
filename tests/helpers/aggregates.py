"""A run of an aggregate over tracks whose label varies inside one entry.

``keep_columns`` carries a label to the output when the rows reduced together
share it. The run here has two entries of two individuals: ``STEADY`` labels
every row ``a``, and ``MIXED`` labels a row ``a`` or ``b`` so that the label
varies within one frame, within one individual and within the entry. Each
aggregate then loses ``MIXED`` whatever it groups by, and writes ``STEADY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.types import Feature, Result
from mosaic.runlog import EntryErrorRecord, read_entry_errors, run_log_dir
from tests.helpers.datasets import make_dataset
from tests.helpers.tracks import add_track_sequences

__all__ = ["MIXED", "STEADY", "LabeledTracksRun", "run_over_labeled_tracks"]

STEADY: Entry = ("g1", "steady")
MIXED: Entry = ("g1", "mixed")

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
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, STEADY, MIXED, n_rows=6, n_ids=2)
    _label_tracks(ds)
    result = ds.run_feature(feature, execution_id=_EXECUTION_ID)
    assert result.run_id is not None
    run_root = ds.get_root("features") / result.feature / result.run_id
    written = sorted(path.stem for path in run_root.glob("*.parquet"))
    errors = read_entry_errors(run_log_dir(ds.base_dir), _EXECUTION_ID)["items"]
    return LabeledTracksRun(result=result, errors=errors, written=written)


def _label_tracks(ds: Dataset) -> None:
    """Add the ``strain`` label to each entry's tracks table."""
    for entry in (STEADY, MIXED):
        path = ds.get_root("tracks") / f"{make_entry_key(*entry)}.parquet"
        table = pd.read_parquet(path)
        flipped = (table["frame"] + table["id"]) % 2 == 1
        table[_LABEL] = np.where(flipped & (entry == MIXED), "b", "a")
        table.to_parquet(path)
