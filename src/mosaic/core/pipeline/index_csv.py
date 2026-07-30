from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Generic, TypeVar

import pandas as pd

from ._utils import atomic_write, now_iso
from .index_lock import index_lock

RowT = TypeVar("RowT", bound="SchemaRowBase")


@dataclasses.dataclass(frozen=True, slots=True)
class SchemaRowBase:
    """The column contract alone: a dataclass whose fields are the CSV columns.

    Carries no field of its own. It exists so :class:`IndexCSV` can serve an
    index whose rows do not name a file -- a per-sequence composition row is a
    property of a *sequence*, and a sequence is not a file.

    Giving such a row a directory to satisfy a base class would be worse than it
    looks. :meth:`IndexRowBase.__post_init__` existence-checks an absolute path,
    so a sequence whose media sits on an unmounted share would crash *row
    construction*; :meth:`IndexCSV.prune_missing` would then delete composition
    rows whenever a directory moved, silently destroying the drift baseline rule
    P3 says must be stored rather than recomputed; and ``read(filter_ext=...)``
    would be testing a directory against a file suffix.

    The naming reads backwards on purpose. ``IndexRowBase`` keeps the plain name
    because it is what six of the seven row types in the toolkit actually extend;
    this one is the narrower contract underneath it.
    """


@dataclasses.dataclass(frozen=True, slots=True)
class IndexRowBase(SchemaRowBase):
    """Minimal index row -- just a validated abs_path."""

    abs_path: Path

    def __post_init__(self) -> None:
        raw = self.abs_path
        if isinstance(raw, str):
            if not raw:
                msg = f"{type(self).__name__}.abs_path cannot be empty"
                raise ValueError(msg)
            object.__setattr__(self, "abs_path", Path(raw))
        # Only absolute paths can be existence-checked here. Relative paths are
        # dataset-root-relative (the portable storage form) and carry no dataset
        # context in this deliberately Dataset-agnostic class; they are resolved
        # and validated later by the dataset-aware layer (``Dataset.resolve_path``).
        if self.abs_path.is_absolute() and not self.abs_path.exists():
            msg = f"{type(self).__name__}.abs_path does not exist: {self.abs_path}"
            raise FileNotFoundError(msg)


@dataclasses.dataclass(frozen=True, slots=True)
class RunIndexRowBase(IndexRowBase):
    """Index row with run-tracking fields."""

    run_id: str
    started_at: str = dataclasses.field(init=False, default_factory=now_iso)
    finished_at: str = dataclasses.field(init=False, default="")


_TYPE_TO_DTYPE: dict[type, str] = {
    str: "string",
    int: "Int64",
    float: "float64",
    bool: "boolean",
    Path: "string",
}


def project_to_schema(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Rebuild *df* as exactly *columns*: missing added empty, NaN as ``""``.

    The projection every typed index's ``adopt`` hook needs, in one place. An
    index read off disk may predate a column, carry a real NaN in a widened one
    (a previous hand-written writer concatenated onto a frame read with default NA
    handling), or hold an off-schema column an old writer emitted. This adds the
    missing columns empty, coerces NaN to ``""``, drops the off-schema ones, and
    builds every column with an explicit ``object`` dtype -- so a later
    ``pd.concat`` with a real row cannot widen an integer column to a float
    (``40`` reaching disk as ``40.0``, the same trap that made ``identity_scheme``
    a ``str``). Idempotent: a frame already in schema is projected onto itself.

    Built column by column into a fresh frame rather than mutated in place: that
    gives the projection and the column order for free and never widens a dtype by
    assigning into an existing column.
    """
    out = pd.DataFrame(index=df.index)
    for column in columns:
        if column in df.columns:
            cells = ["" if pd.isna(cell) else cell for cell in df[column]]
        else:
            cells = [""] * len(df)
        out[column] = pd.Series(cells, index=df.index, dtype="object")
    return out.reset_index(drop=True)


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
    adopt : Callable[[pd.DataFrame], pd.DataFrame] | None
        If set, called on the existing frame inside the write lock before rows
        are appended, to bring an older on-disk schema up to the current one.
        See :meth:`_append_locked` for the contract it must satisfy.
    """

    def __init__(
        self,
        path: Path,
        row_cls: type[RowT],
        dedup_keys: list[str] | None = None,
        adopt: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ):
        self.path: Path = path
        self.row_cls: type[RowT] = row_cls
        self.schema: dict[str, str] = _infer_schema(row_cls)
        self.dedup_keys: list[str] | None = dedup_keys
        self.adopt: Callable[[pd.DataFrame], pd.DataFrame] | None = adopt
        # Read every column the row class declares a string as a string. Left to
        # inference, pandas reads an all-digit cell as int64 and a dotted one as
        # float64 -- and because every write path here round-trips the whole file
        # (read, concat, rewrite), the re-inferred value is what lands back on
        # disk. See _read_frame for what that corrupts.
        self._str_dtypes: dict[str, type[str]] = {
            column: str for column, dtype in self.schema.items() if dtype == "string"
        }

    def _assert_run_index(self) -> None:
        if not issubclass(self.row_cls, RunIndexRowBase):
            msg = f"{self.row_cls.__name__} is not a run index row type"
            raise TypeError(msg)

    def _assert_path_index(self) -> None:
        """Refuse a path question an index whose rows have no path cannot answer.

        Both callers below index ``df["abs_path"]`` directly, so without this the
        failure is a pandas ``KeyError`` naming a column rather than a message
        naming what was asked for.
        """
        if not issubclass(self.row_cls, IndexRowBase):
            msg = (
                f"{self.row_cls.__name__} has no abs_path, so this index cannot "
                f"filter or validate by path"
            )
            raise TypeError(msg)

    def _empty_frame(self) -> pd.DataFrame:
        """A zero-row frame carrying exactly this index's schema and dtypes."""
        return pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in self.schema.items()}
        )

    def _read_frame(self) -> pd.DataFrame:
        """Read the CSV with this index's declared string columns kept as strings.

        The single read path, because every one of the four callers either
        rewrites the file afterwards or hands the frame to a caller that compares
        cells against Python strings, and both break under inference:

        - **On disk.** ``_append_locked``, ``prune_missing`` and ``mark_finished``
          each read the whole file and write it back. Inferred, a feature row's
          ``version`` ``"0.10"`` is rewritten as ``0.1`` and its ``params_hash``
          ``"0123456789"`` as ``123456789`` -- while the ``run_id`` that contains
          both keeps the original spelling. An index that contradicts its own
          identifiers is exactly what this milestone exists to prevent.
        - **In dedup.** ``_append_locked`` compares ``df[key] == new_row[key]``.
          With a numeric ``group``/``sequence``/``camera`` that is ``int64 1 ==
          "1"`` -- always False -- so an identical re-run appends instead of
          replacing and the index grows without bound. Numeric sequence names are
          the CalMS21 and MABe convention, so this is reachable, not theoretical.

        Only ``string`` columns are pinned. The Int64/float ones are already read
        correctly by inference, and a nullable cast would fail on the blank cell a
        defaulted numeric column legitimately holds. ``keep_default_na=False``
        stays: without it an empty ``group`` or ``finished_at`` becomes NaN and
        the ``!= ""`` masks in :meth:`list_runs` stop working. Columns absent from
        the file are ignored by pandas rather than raising, so an index written
        before a defaulted column existed still reads.
        """
        return pd.read_csv(self.path, keep_default_na=False, dtype=self._str_dtypes)

    def ensure(self) -> None:
        """Create the CSV with column headers if it doesn't exist."""
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = self._empty_frame()
        atomic_write(self.path, lambda p: df.to_csv(p, index=False))

    def read(
        self,
        run_id: str | None = None,
        filter_ext: str | None = None,
        groups: Iterable[str] | None = None,
        sequences: Iterable[str] | None = None,
        entries: Iterable[tuple[str, str]] | None = None,
        validate_paths: bool = False,
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
            If True, raise FileNotFoundError when abs_path entries point to
            missing files. Defaults to False. This check is deliberately
            naive: it stats the *raw* stored path (relative paths resolved
            only against the index's dataset root) and does NOT apply
            ``Dataset.resolve_path`` remapping. Dataset-aware callers must
            therefore resolve each ``abs_path`` via ``Dataset.resolve_path``
            and check existence themselves (see ``manifest._resolve_feature``
            and ``run._build_result_lookup``) so that relative/relocated
            paths on a moved or synced dataset resolve correctly instead of
            false-failing here.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Index not found: {self.path}")
        df = self._read_frame()
        if run_id is not None:
            self._assert_run_index()
            df = df[df["run_id"] == run_id].reset_index(drop=True)
            if df.empty:
                msg = f"No entries for run_id '{run_id}' in {self.path}"
                raise FileNotFoundError(msg)
        if filter_ext is not None:
            self._assert_path_index()
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
        self._assert_path_index()
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
        # ensure() runs *before* the lock, not inside it. Acquiring the lock
        # opens the index with O_CREAT, so on a first write it materializes a
        # zero-byte file -- after which ensure()'s "already exists" early return
        # would leave it headerless and the read below would raise
        # EmptyDataError. ensure() is itself atomic and idempotent, so two
        # writers racing here both write the same header harmlessly.
        self.ensure()
        with index_lock(self.path):
            self._append_locked(rows)

    def _append_locked(self, rows: list[RowT]) -> None:
        """Body of :meth:`append`, with the index lock already held.

        An ``adopt`` callable, when configured, runs here: on the frame just
        read, before the dedup backfill, and inside the same lock. That placement
        is forced rather than tidy.

        - **In memory, not a second write.** ``atomic_write`` renames a new inode
          over the path while the lock is held on the old one, so a locked block
          that writes twice loses its grip after the first write and a concurrent
          writer interleaves (see ``index_lock``). An ``adopt`` that rewrote the
          file would be exactly that block.
        - **Inside the lock**, because it decides what the merged frame contains;
          run before acquiring it, another writer's rows could land in between and
          be adopted away.
        - **Before the dedup backfill**, so the backfill below sees a frame that
          already carries every schema column and never has to stamp ``""`` over
          a column adoption was about to fill honestly.

        The callable must return a frame carrying **every** column in
        ``self.schema``. A partial adoption is worse than none: filling only the
        column a caller cared about still leaves ``list_runs`` and
        ``latest_run_id`` raising ``KeyError`` on ``finished_at``.
        """
        df = self._read_frame()
        if self.adopt is not None:
            df = self.adopt(df)
        df_new = pd.DataFrame(rows)

        if self.dedup_keys:
            # A dedup key added after some rows were written is absent from the
            # existing CSV; treat those older rows as carrying its empty default
            # so the comparison neither KeyErrors nor spuriously matches.
            for key in self.dedup_keys:
                if key not in df.columns:
                    df[key] = ""
            for _, new_row in df_new.iterrows():
                mask = pd.Series(True, index=df.index)
                for key in self.dedup_keys:
                    mask &= df[key] == new_row[key]
                df = df[~mask]

        merged: pd.DataFrame = pd.concat([df, df_new], ignore_index=True)

        def _write(p: Path) -> None:
            merged.to_csv(p, index=False)

        atomic_write(self.path, _write)

    def replace(self, rows: list[RowT]) -> None:
        """Rewrite the index to exactly *rows*: one locked block, one atomic write.

        The operation three of the toolkit's writers already perform by hand --
        ``write_media_index_rows``, ``write_tracks_raw_index_rows`` and the
        labels-index appender each rebuild a whole file -- expressed once, with
        the lock the hand-rolled versions did not take.

        Wanted rather than expressible as :meth:`append` because a projection is
        not an accumulation: a per-sequence composition index says "these are the
        sequences this root has, and this is what each is made of", so a sequence
        that has gone away must leave, and dedup keys can only ever add. A caller
        that wants to add one row still calls ``append``.

        ``ensure()`` runs before the lock for the reason ``append`` gives: the
        lock's ``O_CREAT`` would otherwise materialize a zero-byte file that
        ``ensure`` then declines to header. The frame is projected onto the
        schema, so column order is fixed and an extra key is dropped rather than
        widening the file.

        **What the lock here does and does not buy**, because the difference
        matters and is easy to assume away. It serializes the writes, so the file
        is never torn and every row in it belongs to one writer's set. It does
        **not** merge two writers: ``rows`` is computed by the caller before the
        lock is taken, so two processes each declaring a different whole set will
        have the last one win, entire. That is correct for a projection -- a
        caller saying "this is everything" cannot also be saying "keep what you
        had" -- and it is why :meth:`append`, whose merge happens inside the lock,
        stays the primitive for accumulating rows.

        For the per-sequence composition index that consequence is bounded and
        was accepted deliberately: the losing writer's projection described an
        index state that has already been superseded, so the result is stale
        rather than wrong, it over-reports rather than under-reports on the next
        comparison, and the next write heals it. Anything needing a merge under
        contention wants ``append``.
        """
        self.ensure()
        with index_lock(self.path):
            frame = pd.DataFrame(rows) if rows else self._empty_frame()
            projected = frame[list(self.schema)]
            atomic_write(self.path, lambda p: projected.to_csv(p, index=False))

    def prune_missing(
        self,
        resolver: Callable[[str], Path],
        *,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """Drop rows whose ``abs_path`` resolves to a non-existent file.

        Reconciles the index against disk, keeping only rows whose ``abs_path``
        -- resolved through *resolver* -- exists. The resolver decouples this
        deliberately Dataset-agnostic class from ``Dataset.resolve_path`` (pass
        ``ds.resolve_path``), so relocated-but-present paths are preserved and
        only genuinely-missing rows are dropped.

        Args:
            resolver: Maps a stored ``abs_path`` string to an absolute
                filesystem path (e.g. ``ds.resolve_path``).
            dry_run: If True, report what would be dropped without rewriting.

        Returns:
            The dropped rows as a DataFrame (empty if none). The index file is
            rewritten (unless *dry_run*) only when at least one row is dropped.
        """
        self._assert_path_index()
        if not self.path.exists():
            return pd.DataFrame()
        # Locked for the whole read-decide-write, not just the write: this is a
        # DELETE racing concurrent appends, so an append landing between the
        # read and the rewrite would be erased by a keep-set computed before it
        # existed. A dry run holds it too -- it reports what a real run would
        # drop, and that answer is only meaningful against a stable file.
        with index_lock(self.path):
            df = self._read_frame()
            if df.empty or "abs_path" not in df.columns:
                return df.iloc[0:0]
            keep_mask = [resolver(str(p)).exists() for p in df["abs_path"]]
            keep = df[keep_mask].reset_index(drop=True)
            dropped = df[[not m for m in keep_mask]].reset_index(drop=True)
            if len(dropped) > 0 and not dry_run:
                atomic_write(self.path, lambda p: keep.to_csv(p, index=False))
            return dropped

    def drop_runs(
        self,
        run_ids: Iterable[str],
        *,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """Drop every row belonging to one of *run_ids*.

        The index half of deleting a run: the caller removes the directory, this
        removes the rows that named it. A run absent from the file contributes
        nothing, so a caller may pass identifiers it is unsure about.

        Locked for the whole read-decide-write, and for ``prune_missing``'s
        reason: this is a DELETE racing concurrent appends, so a row landing
        between the read and the rewrite would be erased by a keep set computed
        before it existed. A dry run holds the lock too -- it reports what a real
        run would drop, and that answer means nothing against a moving file.

        Args:
            run_ids: The run identifiers whose rows are to go.
            dry_run: Report what would be dropped without rewriting.

        Returns:
            The dropped rows (empty if none). The file is rewritten only when at
            least one row is dropped, so a call that removes nothing leaves it
            byte-identical.
        """
        self._assert_run_index()
        # A sorted list rather than a set: ``isin`` takes a sequence, and the
        # order makes a dropped-row report read the same way twice.
        wanted = sorted({str(run_id) for run_id in run_ids})
        if not wanted:
            return self._empty_frame()
        return self._drop_where(lambda df: df["run_id"].isin(wanted), dry_run=dry_run)

    def drop_entries(
        self,
        entries: Iterable[tuple[str, str]],
        *,
        run_id: str | None = None,
        dry_run: bool = False,
    ) -> pd.DataFrame:
        """Drop rows for the given ``(group, sequence)`` entries.

        The per-entry sibling of :meth:`drop_runs`, for a change that invalidates
        some of a run's outputs and not others. *run_id* narrows it to one run;
        ``None`` drops the entries from every run in the file.

        Same lock, same rewrite-only-if-something-drops rule. A caller deleting
        files as well must unlink them *after* this returns, so a crash between
        the two leaves rows naming files that are gone -- which a reconcile
        removes -- rather than files nothing names, which nothing finds.
        """
        wanted = {(str(group), str(sequence)) for group, sequence in entries}
        if not wanted:
            return self._empty_frame()
        if run_id is not None:
            self._assert_run_index()

        def mask_of(df: pd.DataFrame) -> "pd.Series[bool]":
            paired = [
                (str(group), str(sequence)) in wanted
                for group, sequence in zip(df["group"], df["sequence"], strict=True)
            ]
            selected = pd.Series(paired, index=df.index)
            if run_id is None:
                return selected
            return selected & (df["run_id"] == run_id)

        return self._drop_where(mask_of, dry_run=dry_run)

    def _drop_where(
        self,
        mask_of: Callable[[pd.DataFrame], "pd.Series[bool]"],
        *,
        dry_run: bool,
    ) -> pd.DataFrame:
        """Drop the rows *mask_of* selects, under the lock, atomically.

        The shared body of :meth:`drop_runs` and :meth:`drop_entries`, and the
        lock is why it is shared rather than written twice: both are a DELETE
        racing concurrent appends, so a row landing between the read and the
        rewrite would be erased by a keep set computed before it existed. A dry
        run holds it too -- it reports what a real run would drop, and that answer
        means nothing against a moving file.
        """
        if not self.path.exists():
            return self._empty_frame()
        with index_lock(self.path):
            df = self._read_frame()
            if df.empty:
                return df.iloc[0:0]
            drop_mask = mask_of(df)
            keep = df[~drop_mask].reset_index(drop=True)
            dropped = df[drop_mask].reset_index(drop=True)
            if len(dropped) > 0 and not dry_run:
                atomic_write(self.path, lambda p: keep.to_csv(p, index=False))
            return dropped

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
        # Checked before the lock, which would otherwise create the file (see
        # the note in append) and turn a missing index into an EmptyDataError.
        if not self.path.exists():
            return
        with index_lock(self.path):
            df = self._read_frame()
            sel = (df["run_id"] == run_id) & (df["finished_at"] == "")
            if sel.any():
                df.loc[sel, "finished_at"] = now_iso()
                atomic_write(self.path, lambda p: df.to_csv(p, index=False))
