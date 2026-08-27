"""Leaf module: load-spec models and load_from_spec() dispatcher.

No internal pipeline imports -- both types.py and loading.py import from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.params import Declared
from mosaic.core.strict_model import StrictModel

_KIND_DESCRIPTION = (
    "Fixed tag identifying this load specification's format. Selects the "
    "matching spec when a LoadSpec value is parsed."
)

_NPZ_KEY_DESCRIPTION = (
    "Array key read from the .npz archive. Raises when the archive has no such key."
)

_NPZ_TRANSPOSE_DESCRIPTION = "Transpose the loaded array after it is read."

_PARQUET_TRANSPOSE_DESCRIPTION = "Transpose the loaded table after column filtering."

_PARQUET_COLUMNS_DESCRIPTION = (
    "Column names read from the file. Unset reads every column, and "
    "numeric_only then filters the result."
)

_PARQUET_DROP_COLUMNS_DESCRIPTION = (
    "Column names dropped after loading. A name absent from the file is ignored."
)

_PARQUET_NUMERIC_ONLY_DESCRIPTION = (
    "Keep only numeric-dtype columns. Ignored when columns is set."
)

_PARQUET_FRAME_COLUMN_DESCRIPTION = "Column meant to be extracted as frame indices."

_PARQUET_FRAME_COLUMN_UNWIRED = (
    "no code path reads this field -- load_from_spec's ParquetLoadSpec case "
    "does not destructure it, and nothing else reads spec.frame_column"
)

_JOBLIB_KEY_DESCRIPTION = (
    "Dict key extracted from the loaded object. Unset returns the object as loaded."
)


class NpzLoadSpec(StrictModel):
    """Load spec for numpy .npz archives.

    Attributes:
        kind: Fixed tag identifying this load specification's format.
            Selects the matching spec when a LoadSpec value is parsed.
        key: Array key read from the .npz archive. Raises when the archive
            has no such key.
        transpose: Transpose the loaded array after it is read.
    """

    kind: Annotated[Literal["npz"], Declared(_KIND_DESCRIPTION)] = "npz"
    key: Annotated[str, Declared(_NPZ_KEY_DESCRIPTION)]
    transpose: Annotated[bool, Declared(_NPZ_TRANSPOSE_DESCRIPTION)] = False


class ParquetLoadSpec(StrictModel):
    """Load spec for parquet files.

    Attributes:
        kind: Fixed tag identifying this load specification's format.
            Selects the matching spec when a LoadSpec value is parsed.
        transpose: Transpose the loaded table after column filtering.
        columns: Column names read from the file. Unset reads every column,
            and numeric_only then filters the result.
        drop_columns: Column names dropped after loading. A name absent
            from the file is ignored.
        numeric_only: Keep only numeric-dtype columns. Ignored when columns
            is set.
        frame_column: Column meant to be extracted as frame indices. Unwired
            -- no code path reads this field.
    """

    kind: Annotated[Literal["parquet"], Declared(_KIND_DESCRIPTION)] = "parquet"
    transpose: Annotated[bool, Declared(_PARQUET_TRANSPOSE_DESCRIPTION)] = False
    columns: Annotated[list[str] | None, Declared(_PARQUET_COLUMNS_DESCRIPTION)] = None
    drop_columns: Annotated[
        list[str] | None, Declared(_PARQUET_DROP_COLUMNS_DESCRIPTION)
    ] = None
    numeric_only: Annotated[bool, Declared(_PARQUET_NUMERIC_ONLY_DESCRIPTION)] = True
    frame_column: Annotated[
        str | None,
        Declared(
            _PARQUET_FRAME_COLUMN_DESCRIPTION, unwired=_PARQUET_FRAME_COLUMN_UNWIRED
        ),
    ] = None


class JoblibLoadSpec(StrictModel):
    """Load spec for joblib-serialized objects.

    Attributes:
        kind: Fixed tag identifying this load specification's format.
            Selects the matching spec when a LoadSpec value is parsed.
        key: Dict key extracted from the loaded object. Unset returns the
            object as loaded.
    """

    kind: Annotated[Literal["joblib"], Declared(_KIND_DESCRIPTION)] = "joblib"
    key: Annotated[str | None, Declared(_JOBLIB_KEY_DESCRIPTION)] = None


LoadSpec = Annotated[
    NpzLoadSpec | ParquetLoadSpec | JoblibLoadSpec,
    Field(discriminator="kind"),
]


def load_from_spec(
    path: Path, spec: NpzLoadSpec | ParquetLoadSpec | JoblibLoadSpec
) -> object:
    """Load artifact from a file path using a typed load specification.

    Parameters
    ----------
    path
        Resolved path to the artifact file.
    spec
        Typed load specification (NpzLoadSpec, ParquetLoadSpec, or JoblibLoadSpec).

    Returns
    -------
    object
        Loaded data: np.ndarray (npz), pd.DataFrame (parquet), or arbitrary object (joblib).
    """
    match spec:
        case NpzLoadSpec(key=key, transpose=transpose):
            data = np.load(path, allow_pickle=True)
            if key not in data.files:
                msg = f"Key {key!r} not found in {path}"
                raise FileNotFoundError(msg)
            arr = np.asarray(data[key])
            if arr.ndim == 1:
                arr = arr[None, :]
            if transpose:
                arr = arr.T
            return arr.astype(np.float32, copy=False)

        case JoblibLoadSpec(key=key):
            obj: object = joblib.load(path)
            if key is not None:
                return obj[key]  # pyright: ignore[reportIndexIssue,reportUnknownVariableType]
            return obj

        case ParquetLoadSpec(
            columns=columns,
            drop_columns=drop_columns,
            numeric_only=numeric_only,
            transpose=transpose,
        ):
            df = pd.read_parquet(path, columns=columns)
            if drop_columns:
                df = df.drop(columns=set(drop_columns) & set(df.columns))
            if columns is None and numeric_only:
                df = df.select_dtypes(include=[np.number])
            if transpose:
                df = df.T
            return df
