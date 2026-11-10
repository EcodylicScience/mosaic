"""Typed reads of a numeric CSV through pyarrow, as plain numpy columns.

pyarrow does not ship type stubs, and its compiled modules leave every attribute
unknown to the type checker. This module holds the one read of a CSV whose
columns are typed by name, so a caller that reads a large export type-checks in
full. The file-level directives below cover the unknown types of pyarrow alone.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.csv as pacsv


def read_numeric_csv(
    path: Path,
    columns: Collection[str],
    *,
    integer_columns: Collection[str] = (),
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]]:
    """Read the named columns of the CSV at *path*, each with a stated type.

    No type is inferred. A column of *integer_columns* is int64 and every other
    named column is float64, where an empty field is NaN. The file is read by
    pyarrow's column-typed reader, so a large export is not read row by row.

    Args:
        path: The CSV, whose first line names its columns.
        columns: The columns to read. A column of the file that is not named
            here is not read.
        integer_columns: The columns read as int64. An empty field in one is
            refused.

    Returns:
        Each column of *columns*, by name.

    Raises:
        ValueError: A named column is absent from the file, or an integer column
            has an empty field. The message names *path*.
    """
    integers = set(integer_columns)
    types = {name: pa.int64() if name in integers else pa.float64() for name in columns}
    try:
        table = pacsv.read_csv(
            path,
            convert_options=pacsv.ConvertOptions(
                column_types=types, include_columns=list(types)
            ),
        )
    except pa.ArrowInvalid as error:
        msg = f"{path}: {error}"
        raise ValueError(msg) from error
    result: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]] = {}
    for name in types:
        column = table.column(name)
        if name in integers:
            if column.null_count:
                msg = f"{path}: {name} has an empty field"
                raise ValueError(msg)
            result[name] = np.asarray(column.to_numpy(), dtype=np.int64)
        else:
            result[name] = np.asarray(column.to_numpy(), dtype=np.float64)
    return result
