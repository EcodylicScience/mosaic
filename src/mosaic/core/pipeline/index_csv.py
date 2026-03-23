from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd

from ._utils import now_iso

RowT = TypeVar("RowT")


@dataclasses.dataclass(frozen=True, slots=True)
class IndexRowBase:
    """Base for all index row dataclasses. Provides run-tracking fields."""

    run_id: str
    abs_path: Path
    started_at: str = dataclasses.field(init=False, default_factory=now_iso)
    finished_at: str = dataclasses.field(init=False, default="")

    def __post_init__(self) -> None:
        if isinstance(self.abs_path, str):
            object.__setattr__(self, "abs_path", Path(self.abs_path))
        if not self.abs_path.exists():
            msg = f"{type(self).__name__}.abs_path does not exist: {self.abs_path}"
            raise FileNotFoundError(msg)


_TYPE_TO_DTYPE: dict[type, str] = {
    str: "string",
    int: "Int64",
    float: "float64",
    bool: "boolean",
    Path: "string",
}


def _infer_schema(row_cls: type) -> dict[str, str]:
    """Infer a {column: pandas_dtype} schema from a dataclass's type hints."""
    hints = typing.get_type_hints(row_cls)
    schema: dict[str, str] = {}
    for field in dataclasses.fields(row_cls):
        py_type = hints[field.name]
        dtype = _TYPE_TO_DTYPE.get(py_type)
        if dtype is None:
            raise TypeError(
                f"No pandas dtype mapping for type {py_type!r} "
                f"on field {row_cls.__name__}.{field.name}"
            )
        schema[field.name] = dtype
    return schema


class IndexCSV(Generic[RowT]):
    """Generic CSV index: ensure, append-with-dedup, read.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    row_cls : type[RowT]
        Dataclass whose fields define the CSV columns. Type hints are mapped
        to pandas dtypes (str -> "string", int -> "Int64", float -> "float64",
        bool -> "boolean").
    dedup_keys : list[str] | None
        If set, existing rows matching ALL these columns are removed
        before appending new rows.
    """

    def __init__(
        self,
        path: Path,
        row_cls: type[RowT],
        dedup_keys: list[str] | None = None,
    ):
        self.path: Path = path
        self.row_cls: type[RowT] = row_cls
        self.schema: dict[str, str] = _infer_schema(row_cls)
        self.dedup_keys: list[str] | None = dedup_keys

    def ensure(self) -> None:
        """Create the CSV with column headers if it doesn't exist."""
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in self.schema.items()}
        )
        df.to_csv(self.path, index=False)

    def read(self) -> pd.DataFrame:
        """Read the CSV, validating that all abs_path entries still exist."""
        if not self.path.exists():
            raise FileNotFoundError(f"Index not found: {self.path}")
        df = pd.read_csv(self.path, keep_default_na=False)
        missing = [p for p in df["abs_path"] if not Path(p).exists()]
        if missing:
            msg = (
                f"Stale index {self.path}: "
                f"{len(missing)} path(s) no longer exist, "
                f"first: {missing[0]}"
            )
            raise FileNotFoundError(msg)
        return df

    def append(self, rows: list[RowT]) -> None:
        """Append rows, deduplicating by dedup_keys if configured.

        Rows are dataclass instances. pandas handles them
        natively in pd.DataFrame().
        """
        self.ensure()
        df = pd.read_csv(self.path, keep_default_na=False)
        df_new = pd.DataFrame(rows)

        if self.dedup_keys:
            for _, new_row in df_new.iterrows():
                mask = pd.Series(True, index=df.index)
                for key in self.dedup_keys:
                    mask &= df[key] == new_row[key]
                df = df[~mask]

        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(self.path, index=False)

    def list_runs(self) -> pd.DataFrame:
        """Return all rows sorted: finished (newest first), then unfinished (newest first)."""
        df = self.read()
        if df.empty:
            return df
        mask = df["finished_at"] != ""
        finished = df[mask].sort_values("finished_at", ascending=False, kind="stable")
        unfinished = df[~mask].sort_values("started_at", ascending=False, kind="stable")
        return pd.concat([finished, unfinished], ignore_index=True)

    def latest_run_id(self) -> str:
        """Return the most recent run_id. Prefers finished over in-progress."""
        df = self.list_runs()
        if df.empty:
            raise ValueError(f"No runs found in {self.path}")
        return str(df.iloc[0]["run_id"])

    def mark_finished(self, run_id: str) -> None:
        """Set finished_at to now on all rows matching run_id where it is empty."""
        df = pd.read_csv(self.path, keep_default_na=False)
        sel = (df["run_id"] == run_id) & (df["finished_at"] == "")
        if sel.any():
            df.loc[sel, "finished_at"] = now_iso()
            df.to_csv(self.path, index=False)
