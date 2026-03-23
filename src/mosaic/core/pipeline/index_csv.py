from __future__ import annotations

import dataclasses
import typing
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd

from ._utils import now_iso

RowT = TypeVar("RowT", bound="IndexRowBase")


@dataclasses.dataclass(frozen=True, slots=True)
class IndexRowBase:
    """Minimal index row -- just a validated abs_path."""

    abs_path: Path

    def __post_init__(self) -> None:
        raw = self.abs_path
        if isinstance(raw, str):
            if not raw:
                msg = f"{type(self).__name__}.abs_path cannot be empty"
                raise ValueError(msg)
            object.__setattr__(self, "abs_path", Path(raw))
        if not self.abs_path.exists():
            msg = f"{type(self).__name__}.abs_path does not exist: {self.abs_path}"
            raise FileNotFoundError(msg)


@dataclasses.dataclass(frozen=True, slots=True)
class RunIndexRowBase(IndexRowBase):
    """Index row with run-tracking fields."""

    run_id: str
    started_at: str = dataclasses.field(init=False, default_factory=now_iso)
    finished_at: str = dataclasses.field(init=False, default="")


@dataclasses.dataclass(frozen=True, slots=True)
class TracksIndexRow(IndexRowBase):
    """Index row for tracks (no run-tracking)."""

    group: str
    sequence: str


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

    def _assert_run_index(self) -> None:
        if not issubclass(self.row_cls, RunIndexRowBase):
            msg = f"{self.row_cls.__name__} is not a run index row type"
            raise TypeError(msg)

    def ensure(self) -> None:
        """Create the CSV with column headers if it doesn't exist."""
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in self.schema.items()}
        )
        df.to_csv(self.path, index=False)

    def read(
        self,
        run_id: str | None = None,
        filter_ext: str | None = None,
        groups: Iterable[str] | None = None,
        sequences: Iterable[str] | None = None,
        entries: Iterable[tuple[str, str]] | None = None,
        validate_paths: bool = True,
    ) -> pd.DataFrame:
        """Read the CSV with optional filtering and validation.

        Parameters
        ----------
        run_id : str | None
            If set, only rows matching this run_id are returned.
            Raises FileNotFoundError if the index contains no entries
            for the requested run_id.
        filter_ext : str | None
            If set (e.g. ``".parquet"``), only rows whose ``abs_path``
            ends with this suffix are returned.
        groups : Iterable[str] | None
            If set, only rows whose ``group`` column is in this set.
        sequences : Iterable[str] | None
            If set, only rows whose ``sequence`` column is in this set.
        entries : Iterable[tuple[str, str]] | None
            If set, only rows whose ``(group, sequence)`` pair is in
            this set.
        validate_paths : bool
            If True (default), raise FileNotFoundError when abs_path
            entries point to missing files. Set to False for discovery
            operations like list_runs().
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Index not found: {self.path}")
        df = pd.read_csv(self.path, keep_default_na=False)
        if run_id is not None:
            self._assert_run_index()
            df = df[df["run_id"] == run_id].reset_index(drop=True)
            if df.empty:
                msg = f"No entries for run_id '{run_id}' in {self.path}"
                raise FileNotFoundError(msg)
        if filter_ext is not None:
            df = df[df["abs_path"].str.endswith(filter_ext)].reset_index(drop=True)
        if groups is not None:
            df = df[df["group"].isin(set(groups))].reset_index(drop=True)
        if sequences is not None:
            df = df[df["sequence"].isin(set(sequences))].reset_index(drop=True)
        if entries is not None:
            entry_set = set(entries)
            mask = [
                (row["group"], row["sequence"]) in entry_set for _, row in df.iterrows()
            ]
            df = df[mask].reset_index(drop=True)
        if not validate_paths:
            return df
        # Resolve relative paths against the dataset root (grandparent of
        # the index file: features/<name>/index.csv -> features -> dataset_root)
        base = self.path.parent.parent.parent
        missing = []
        for p in df["abs_path"]:
            pp = Path(p)
            if not pp.is_absolute():
                pp = base / pp
            if not pp.exists():
                missing.append(p)
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

    def ordered_entries(
        self,
        run_id: str | None = None,
        filter_ext: str | None = None,
    ) -> list[tuple[str, str]]:
        """Return all (group, sequence) pairs in sorted order."""
        df = self.read(run_id=run_id, filter_ext=filter_ext)
        df = df.sort_values(["group", "sequence"])
        return list(zip(df["group"], df["sequence"]))

    def list_runs(self) -> pd.DataFrame:
        """Return all rows sorted: finished (newest first), then unfinished (newest first)."""
        self._assert_run_index()
        df = self.read(validate_paths=False)
        if df.empty:
            return df
        mask = df["finished_at"] != ""
        finished = df[mask].sort_values("finished_at", ascending=False, kind="stable")
        unfinished = df[~mask].sort_values("started_at", ascending=False, kind="stable")
        return pd.concat([finished, unfinished], ignore_index=True)

    def latest_run_id(self) -> str:
        """Return the most recent run_id. Prefers finished over in-progress."""
        self._assert_run_index()
        df = self.list_runs()
        if df.empty:
            raise ValueError(f"No runs found in {self.path}")
        return str(df.iloc[0]["run_id"])

    def mark_finished(self, run_id: str) -> None:
        """Set finished_at to now on all rows matching run_id where it is empty."""
        self._assert_run_index()
        df = pd.read_csv(self.path, keep_default_na=False)
        sel = (df["run_id"] == run_id) & (df["finished_at"] == "")
        if sel.any():
            df.loc[sel, "finished_at"] = now_iso()
            df.to_csv(self.path, index=False)
