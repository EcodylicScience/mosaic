"""The check of a feature's output against the columns that its declaration predicts."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd

from mosaic.core.pipeline.graph import ColumnSet


def assert_writes_declared_columns(
    feature: str,
    predicted: ColumnSet,
    output: pd.DataFrame,
    passed: Collection[str] = (),
    *,
    declares_empty: bool = False,
) -> None:
    """Assert that the columns of *output* match the declared columns, *predicted*.

    *output* contains every guaranteed column and a column of every guaranteed
    family, and each of its columns is one that *predicted* admits.

    Args:
        feature: The feature's name, which each failure names.
        predicted: The columns of the feature's step, as ``resolved_columns``
            resolves them from the declarations.
        output: The table that the feature wrote.
        passed: The input columns that the feature passes through. Each is
            admitted beside the declared ones.
        declares_empty: Whether the feature declares an empty output
            (``Provides.is_empty``). *output* is then empty.
    """
    written = {str(column) for column in output.columns}
    if declares_empty:
        assert output.empty, f"{feature} declares an empty output and wrote rows"
    else:
        assert not output.empty, f"{feature} wrote an empty frame"
    missing = predicted.guaranteed - written
    assert not missing, f"{feature} declares {sorted(missing)} and did not write them"
    empty = sorted(
        prefix
        for prefix in predicted.guaranteed_prefixes
        if not any(column.startswith(prefix) for column in written)
    )
    assert not empty, f"{feature} declares columns of {empty} and wrote none"
    assert not predicted.unbounded, f"{feature}'s declared output is unbounded"
    undeclared = sorted(
        column for column in written - set(passed) if not predicted.admits(column)
    )
    assert not undeclared, f"{feature} wrote {undeclared}, which it does not declare"
