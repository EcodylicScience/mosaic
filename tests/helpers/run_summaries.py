"""Feature runs whose output tables a test writes cell by cell.

A test of a key's type across entries needs each entry's file to store the key
under the Arrow type it chose: a label that is null for a whole entry as Arrow
``null``, an integer label with a null as ``double``. :class:`LabeledFeature`
returns, for each entry, the table the test passes for it, so the run writes
that table as it is. Its runs stand in for an aggregate's runs where a test needs
only the shape of the table.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.index import feature_index_path
from mosaic.core.pipeline.types import Result
from tests.helpers.datasets import make_dataset
from tests.helpers.stand_in_features import PlainFeature
from tests.helpers.tracks import add_track_sequences


class LabeledFeature(PlainFeature):
    """A feature whose output for each entry is the table a test passes for it.

    ``group`` and ``sequence`` are filled in when the table lacks them. A table
    with no rows is written with its columns.
    """

    name = "prov-labeled"

    def __init__(self, rows_by_entry: Mapping[Entry, pd.DataFrame]) -> None:
        super().__init__()
        self.rows_by_entry: dict[Entry, pd.DataFrame] = dict(rows_by_entry)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        entry = (str(df["group"].iloc[0]), str(df["sequence"].iloc[0]))
        table = self.rows_by_entry[entry].reset_index(drop=True)
        named = {"group": entry[0], "sequence": entry[1]}
        missing = {name: value for name, value in named.items() if name not in table}
        return table.assign(**missing) if missing else table.copy()


def labeled_run(
    tmp_path: Path, rows_by_entry: Mapping[Entry, pd.DataFrame]
) -> tuple[Dataset, Result[str]]:
    """A dataset holding the entries of *rows_by_entry*, and a run that wrote them.

    Args:
        tmp_path: The directory the dataset is made under.
        rows_by_entry: The table each entry's output holds.

    Returns:
        The dataset and the run of :class:`LabeledFeature` over every entry.
    """
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *rows_by_entry, n_rows=4)
    return ds, ds.run_feature(LabeledFeature(rows_by_entry))


def rewrite_index(
    ds: Dataset, result: Result[str], edit: Callable[[pd.DataFrame], pd.DataFrame]
) -> None:
    """Rewrite the feature index of *result* through *edit*, a frame function."""
    path = feature_index_path(ds, result.feature)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    edit(frame).to_csv(path, index=False)
