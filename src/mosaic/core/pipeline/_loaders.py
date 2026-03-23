"""Leaf module: load-spec models and load_from_spec() dispatcher.

No internal pipeline imports -- both types.py and loading.py import from here.
"""

from __future__ import annotations

from collections.abc import KeysView
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DictModel(BaseModel):
    """BaseModel with dict-like access for backward compatibility.

    Provides __getitem__, get, __contains__, and keys() so that existing
    code using dict-style access (spec["key"], spec.get("key")) works
    transparently with typed models.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    def __getitem__(self, key: str) -> object:
        try:
            val: object = getattr(self, key)  # pyright: ignore[reportAny]
            return val
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default: object = None) -> object:
        try:
            val: object = getattr(self, key)  # pyright: ignore[reportAny]
            return val
        except AttributeError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in self.__class__.model_fields

    def keys(self) -> KeysView[str]:
        """Support {**params} dict spread and dict(params) conversion."""
        return self.__class__.model_fields.keys()


class NpzLoadSpec(DictModel):
    """Load spec for numpy .npz archives.

    Attributes:
        kind: Discriminator literal "npz".
        key: Array key to extract from the .npz file. Required.
        transpose: Transpose the loaded array. Default False.
    """

    kind: Literal["npz"] = "npz"
    key: str
    transpose: bool = False


class ParquetLoadSpec(DictModel):
    """Load spec for parquet files.

    Attributes:
        kind: Discriminator literal "parquet".
        transpose: Transpose the loaded array. Default False.
        columns: Explicit column list. None uses numeric_only filter.
        drop_columns: Columns to drop before loading.
        numeric_only: Keep only numeric columns. Default True.
        frame_column: Column to extract as frame indices.
    """

    kind: Literal["parquet"] = "parquet"
    transpose: bool = False
    columns: list[str] | None = None
    drop_columns: list[str] | None = None
    numeric_only: bool = True
    frame_column: str | None = None


class JoblibLoadSpec(DictModel):
    """Load spec for joblib-serialized objects.

    Attributes:
        kind: Discriminator literal "joblib".
        key: Dict key to extract from loaded object. None loads raw.
    """

    kind: Literal["joblib"] = "joblib"
    key: str | None = None


LoadSpec = Annotated[
    NpzLoadSpec | ParquetLoadSpec | JoblibLoadSpec,
    Field(discriminator="kind"),
]


def load_from_spec(path: Path, spec: NpzLoadSpec | ParquetLoadSpec | JoblibLoadSpec) -> object:
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
