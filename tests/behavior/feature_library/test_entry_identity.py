"""The group and sequence of a table's first row name its entry."""

from __future__ import annotations

import pandas as pd

from mosaic.behavior.feature_library.helpers import entry_identity


def test_the_entry_is_the_first_row_as_text() -> None:
    rows = pd.DataFrame({"group": [3, 4], "sequence": [10, 11], "frame": [0, 1]})

    assert entry_identity(rows) == ("3", "10")


def test_a_table_without_a_group_column_has_an_empty_group() -> None:
    rows = pd.DataFrame({"sequence": ["s1", "s2"], "frame": [0, 1]})

    assert entry_identity(rows) == ("", "s1")


def test_a_table_without_rows_has_an_empty_entry() -> None:
    rows = pd.DataFrame({"group": [], "sequence": [], "frame": []})

    assert entry_identity(rows) == ("", "")
