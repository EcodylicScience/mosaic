"""Data loading, artifact resolution, and key-data helpers.

Moved from ``mosaic.behavior.feature_library.helpers`` to live closer to the
pipeline infrastructure they depend on.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from ..helpers import load_labels_for_feature_frames
from ..scope import Scope
from ._loaders import JoblibLoadSpec, LoadSpec, ParquetLoadSpec
from .index import (
    feature_index,
    feature_index_path,
    feature_run_root,
    latest_feature_run_root,
    resolve_artifact_file,
)
from .types import ArtifactSpec, NNResult
from .types.data_config import ALIGN_COLS, COLUMNS, ID_COLS, NON_PAYLOAD_COLS

if TYPE_CHECKING:
    from ..dataset import Dataset

_ALIGN_COLS = ALIGN_COLS

# Features whose inputs legitimately align on frame alone. A closed set rather than
# a per-feature flag: ``Feature`` is a Protocol with 44 standalone implementations,
# so a required attribute is 44 edits, and it cannot live on ``Inputs`` whose dump
# feeds every downstream identifier. ``interaction-crop-pipeline`` merges tracks
# against a pair-level filter and groups afterwards, so its fan-out costs memory
# rather than correctness.
CROSS_JOIN_FEATURES: frozenset[str] = frozenset({"interaction-crop-pipeline"})


class MultiInputAlignmentError(ValueError):
    """Two inputs cannot be aligned without inventing or dropping rows."""


@dataclass(frozen=True, slots=True)
class AlignmentVerdict:
    """Whether two column sets can be joined, and on what.

    Exported so a caller composing a chain before running it uses the same rule the
    merge enforces, rather than learning at execution.
    """

    keys: frozenset[str]
    levels: tuple[str, ...]
    compatible: bool
    reason: str


def entity_level_of(columns: Iterable[str]) -> str:
    """``"pair"`` / ``"individual"`` / ``"global"`` from column names alone.

    Name-based, so it answers for a parquet schema without reading data -- which is
    what lets a submit-time check resolve a run's level cheaply.

    **There is one pair spelling, and that is the point.** It used to accept
    ``id_a``/``id_b`` and ``id_A``/``id_B`` beside ``id1``/``id2``, and still
    missed the ``focal_id``-with-a-target pair that ``pair-facing`` and
    ``attention-target`` wrote -- so both read as ``"global"``, no identity, and a
    join against an individual-level input was permitted on ``frame`` alone. A
    list of names that grows every time a feature invents one is a list somebody
    forgets to extend; every producer now writes ``id1`` (the focal) and ``id2``
    (the other), and a feature that does not is caught by
    ``tests/behavior/feature_library/test_pair_identity_convention.py`` rather
    than silently read as unidentified here.
    """
    present = set(columns)
    if "id1" in present and "id2" in present:
        return "pair"
    return "individual" if COLUMNS.id_col in present else "global"


def alignment_keys(column_sets: Sequence[Iterable[str]]) -> frozenset[str]:
    """The columns that a merge of these inputs joins on: the shared ``ALIGN_COLS``."""
    sets = [set(columns) for columns in column_sets]
    return frozenset(_ALIGN_COLS.intersection(*sets)) if sets else frozenset()


def alignment_verdict(column_sets: Sequence[Iterable[str]]) -> AlignmentVerdict:
    """Can these inputs be joined? The rule behind :class:`MultiInputAlignmentError`.

    Incompatible when two inputs carry identity at different levels and the keys
    they share carry no identity at all: joining an individual frame to a pair frame
    on ``frame`` alone is a per-frame cartesian product, not an alignment.

    "Carries identity" is ``ID_COLS``, not "a key that is not ``frame`` or ``time``".
    The two were the same set until ``perspective`` joined ``ALIGN_COLS``, and they
    mean different things: ``perspective`` separates two rows of one pair but names
    no individual, so an individual-level frame that happens to carry one must not
    read as sharing identity with a pair-level frame.
    """
    sets = [set(columns) for columns in column_sets]
    keys = alignment_keys(sets)
    levels = tuple(entity_level_of(columns) for columns in sets)
    identity = keys & ID_COLS
    concrete = {level for level in levels if level != "global"}
    if len(concrete) > 1 and not identity:
        return AlignmentVerdict(
            keys,
            levels,
            False,
            f"inputs are at different entity levels ({', '.join(levels)}) and share "
            f"no identity column, so joining on {sorted(keys) or 'nothing'} would "
            f"pair every row of one with every row of the other",
        )
    if not keys:
        return AlignmentVerdict(keys, levels, False, "no shared alignment columns")
    return AlignmentVerdict(keys, levels, True, "")


@dataclass(frozen=True, slots=True)
class MergeNaming:
    """What merging one more input does to the column names.

    Attributes:
        keys: The columns joined on, sorted.
        dropped: The incoming identity and bookkeeping columns that the merged
            side already contains, dropped from the incoming input.
        renamed: Each incoming payload column that the merged side already
            contains, mapped to its numbered name ``<column>__<position>``.
        columns: Every column of the merged frame afterwards.
    """

    keys: tuple[str, ...]
    dropped: tuple[str, ...]
    renamed: Mapping[str, str]
    columns: frozenset[str]


def merge_naming(
    merged: Iterable[str], incoming: Iterable[str], position: int
) -> MergeNaming:
    """The column names after merging *incoming*, the input at *position*.

    The naming rule of a multi-input merge, apart from the data. The merge
    applies it, and pipeline validation applies it to declared column sets. The
    two then agree on the columns of a step's merged input.

    A collided ``NON_PAYLOAD_COLS`` column is **dropped** from the incoming
    input, and not numbered. Identity is either joined on or constant within an
    entry, and the two copies state the same value. Of two bookkeeping copies
    the merged frame keeps the first input's, and a later ``nn_id`` that names
    a different neighbor is lost. A numbered copy such as ``perspective__1`` or
    ``nn_id__1`` matches no exclusion list, and ``perspective__1`` was once read
    downstream as a measurement and fitted as data. Only payload keeps the
    ``__<position>`` suffix.

    Args:
        merged: The columns merged so far.
        incoming: The next input's columns.
        position: The next input's declared position among the step's inputs.
    """
    left = frozenset(merged)
    right = frozenset(incoming)
    keys = alignment_keys([left, right])
    collided = sorted((right - keys) & left)
    dropped = tuple(column for column in collided if column in NON_PAYLOAD_COLS)
    renamed = {
        column: f"{column}__{position}"
        for column in collided
        if column not in NON_PAYLOAD_COLS
    }
    kept = (right - frozenset(dropped) - frozenset(renamed)) | frozenset(
        renamed.values()
    )
    return MergeNaming(
        keys=tuple(sorted(keys)),
        dropped=dropped,
        renamed=renamed,
        columns=left | kept,
    )


def load_parquet_dataframe(
    path: Path,
    load_spec: LoadSpec,
    df_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    """Load a parquet file as a full DataFrame, applying filter if given.

    Returns None for non-parquet specs or if the result is empty after
    filtering.
    """
    if not isinstance(load_spec, ParquetLoadSpec):
        return None
    df = pd.read_parquet(path)
    if df_filter is not None:
        df = df_filter(df)
    if df.empty:
        return None
    return df


def _column_names(df: pd.DataFrame) -> list[str]:
    """*df*'s column labels as strings, the only labels a parquet file stores."""
    labels: list[object] = list(df.columns)
    return [str(label) for label in labels]


def _merge_parquet_inputs(
    dfs: Iterable[tuple[int, pd.DataFrame]], *, cross_join: bool = False
) -> pd.DataFrame | None:
    """Merge parquet inputs on the identity columns they share.

    Each item carries the input's **declared** position, not its position among the
    ones that happened to load: the suffix on a collided column used to count
    survivors, so with an empty middle input a later input's column was renamed to
    an earlier input's index and read back as the wrong column entirely.

    Refuses rather than inventing rows -- see :func:`alignment_verdict`. Two escapes,
    both narrow: *cross_join* for a feature that declares a frame-only merge, and
    the multiplicity check below, which allows a one-to-many join (a per-frame table
    against a per-frame-per-id one) while refusing many-to-many.

    Collided columns are named by :func:`merge_naming`.
    """
    it = iter(dfs)
    first = next(it, None)
    if first is None:
        return None
    merged = first[1]

    for declared, df_next in it:
        merged_columns = _column_names(merged)
        next_columns = _column_names(df_next)
        verdict = alignment_verdict([merged_columns, next_columns])
        if not verdict.compatible and not cross_join:
            raise MultiInputAlignmentError(
                f"cannot merge input {declared}: {verdict.reason}"
            )
        naming = merge_naming(merged_columns, next_columns, declared)
        on_cols = list(naming.keys)
        if not on_cols:
            raise MultiInputAlignmentError(
                f"cannot merge input {declared}: no shared alignment columns"
            )
        if not cross_join and (
            merged.duplicated(on_cols).any() and df_next.duplicated(on_cols).any()
        ):
            raise MultiInputAlignmentError(
                f"cannot merge input {declared}: {on_cols} is not unique on either "
                "side, so the join would multiply rows"
            )
        if naming.dropped:
            df_next = df_next.drop(columns=list(naming.dropped))
        if naming.renamed:
            df_next = df_next.rename(columns=dict(naming.renamed))
        merged = merged.merge(df_next, how="inner", on=on_cols)

    if merged.empty:
        return None
    return merged


def load_entry_data(
    file_specs: list[tuple[Path, LoadSpec]],
    filters: Iterable[Callable[[pd.DataFrame], pd.DataFrame]] = (),
    *,
    cross_join: bool = False,
) -> pd.DataFrame | None:
    """Load and merge data for a single manifest entry.

    All manifest entries are parquet. Multiple inputs are merged via
    inner join on shared alignment columns. DataFrames are loaded
    lazily and merged incrementally to minimize peak memory.
    """

    def _load_dfs() -> Iterator[tuple[int, pd.DataFrame]]:
        for declared, (path, load_spec) in enumerate(file_specs):
            df = load_parquet_dataframe(path, load_spec)
            if df is not None:
                yield declared, df

    merged = _merge_parquet_inputs(_load_dfs(), cross_join=cross_join)
    if merged is None or merged.empty:
        return None
    for fn in filters:
        merged = fn(merged)
        if merged.empty:
            return None
    return merged


def build_nn_lookup(
    ds: Dataset,
    group: str,
    sequence: str,
    pair_filter_spec: NNResult,
) -> dict[tuple[int, int], int]:
    """
    Build a per-frame nearest-neighbor lookup for a given sequence.

    Returns ``{(frame, individual_id): nn_id}`` from the nearest-neighbor
    feature parquet.  Returns an empty dict when the NN feature has not been
    computed for this sequence (fail-open).

    ``NNResult`` is a ``Result``, so on the ``run_feature`` path
    :func:`~mosaic.core.pipeline.resolve.resolve_references` has already pinned
    ``pair_filter_spec.run_id`` and the fallback below does not fire. It used to:
    a pair filter resolved to "latest" here, at load time, while the identifier
    that named the run knew only ``None`` -- the same defect as an unpinned
    ``templates``, in a fourth place.
    """

    feature_name = pair_filter_spec.feature
    idx = feature_index(feature_index_path(ds, feature_name))

    run_id = pair_filter_spec.run_id
    if run_id is None:
        run_id = idx.latest_run_id()

    idx_df = idx.read(
        run_id=run_id,
        filter_ext=".parquet",
        scope=Scope(entries=[(group, sequence)]),
    )
    if idx_df.empty:
        return {}

    nn_path = ds.resolve_path(idx_df.iloc[0]["abs_path"])
    df_nn = pd.read_parquet(nn_path)

    # below is probably too defensive, NNResult (as a per sequence result) has a known parquet schema
    frame_col = (
        "frame"
        if "frame" in df_nn.columns
        else ("time" if "time" in df_nn.columns else None)
    )
    if frame_col is None or "id" not in df_nn.columns or "nn_id" not in df_nn.columns:
        return {}

    frames = df_nn[frame_col].to_numpy()
    ids = df_nn["id"].to_numpy()
    nn_ids = df_nn["nn_id"].to_numpy()

    lookup: dict[tuple[int, int], int] = {}
    for f, ind, nn in zip(frames, ids, nn_ids):
        if not np.isnan(nn):
            lookup[(int(f), int(ind))] = int(nn)

    return lookup


def nn_pair_mask(
    df: pd.DataFrame,
    nn_lookup: dict[tuple[int, int], int],
) -> np.ndarray:
    """
    Return a boolean mask for rows in a pair-feature DataFrame where at
    least one individual in the pair considers the other its nearest neighbor.

    For pair (id1, id2) at frame F the row is kept when:
      - ``nn_lookup[(F, id1)] == id2``  (id1's NN is id2), **or**
      - ``nn_lookup[(F, id2)] == id1``  (id2's NN is id1).

    Returns all-True when the DataFrame lacks the required columns or
    when *nn_lookup* is empty (fail-open).
    """
    frame_col = (
        "frame" if "frame" in df.columns else ("time" if "time" in df.columns else None)
    )
    if (
        not nn_lookup
        or frame_col is None
        or "id1" not in df.columns
        or "id2" not in df.columns
    ):
        return np.ones(len(df), dtype=bool)

    frames = df[frame_col].to_numpy(dtype=int)
    id1s = df["id1"].to_numpy(dtype=int)
    id2s = df["id2"].to_numpy(dtype=int)

    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        f = int(frames[i])
        a = int(id1s[i])
        b = int(id2s[i])
        if nn_lookup.get((f, a)) == b or nn_lookup.get((f, b)) == a:
            mask[i] = True

    return mask


def row_individual_pairs(df: pd.DataFrame) -> np.ndarray | None:
    """Each row's ``(id1, id2)``, for aligning labels to rows rather than to frames.

    A pair row gives its ``id1`` (the focal) and ``id2``. An individual row gives
    ``(id, -1)``, so an individual label and every pair label focal on that
    individual match it. A missing id is ``-1``, a wildcard.

    Returns:
        An ``(len(df), 2)`` int64 array, or ``None`` when *df* carries no id
        column, or carries ids that are not integers (names, say). ``None``
        means the frame alone aligns a row to its label, as it did before rows
        carried individuals.
    """
    if "id1" in df.columns and "id2" in df.columns:
        columns = [df["id1"], df["id2"]]
    elif COLUMNS.id_col in df.columns:
        columns = [df[COLUMNS.id_col], pd.Series(-1, index=df.index)]
    else:
        return None
    pairs = np.empty((len(df), 2), dtype=np.int64)
    for position, column in enumerate(columns):
        numeric = pd.to_numeric(column, errors="coerce")
        if bool((numeric.isna() & column.notna()).any()):
            return None
        filled = numeric.fillna(-1).to_numpy(dtype=np.float64)
        if not np.all(np.mod(filled, 1) == 0):
            return None
        pairs[:, position] = filled.astype(np.int64)
    return pairs


def labels_for_rows(
    path: Path, df: pd.DataFrame, *, default_label: int | None = None
) -> np.ndarray:
    """One label per row of *df*, from the label file at *path*.

    The one per-row alignment: extract-labeled-templates, ``load_values`` and the
    scoring helpers all call it, so a row is labeled one way everywhere. A row is
    matched by its frame and its own individuals (:func:`row_individual_pairs`),
    so a pair row takes the label of its own direction, and a row no event
    matches takes the file's background class -- :data:`~mosaic.core.helpers.UNLABELED`
    (``-1``) for a file with none.

    Args:
        path: The ``.npz`` label file of the row's entry.
        df: Rows with a ``frame`` column.
        default_label: The label for an unmatched row. ``None`` takes the
            file's background class.

    Returns:
        An int64 array of ``len(df)`` labels.

    Raises:
        ValueError: *df* has no ``frame`` column.
    """
    if COLUMNS.frame_col not in df.columns:
        raise ValueError(
            f"labels align to rows by their {COLUMNS.frame_col!r} column, which "
            f"these rows lack"
        )
    return load_labels_for_feature_frames(
        path,
        df[COLUMNS.frame_col].to_numpy(),
        default_label=default_label,
        individual_filter=row_individual_pairs(df),
    )


def resolve_sequence_identity(
    entry_key: str,
    entry_map: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    """Map an entry key to (group, sequence).

    Looks up in entry_map, falls back to ("", entry_key).
    """
    if entry_key in entry_map:
        return entry_map[entry_key]
    return "", entry_key


def get_feature_run_root(
    ds: Dataset,
    feature_name: str,
    run_id: str | None = None,
) -> tuple[str, Path]:
    """
    Resolve (run_id, run_root_path) for a feature.

    If run_id is None, picks the latest finished run.
    """
    if run_id is None:
        run_id, run_root = latest_feature_run_root(ds, feature_name)
    else:
        run_root = feature_run_root(ds, feature_name, run_id)
    return str(run_id), run_root


def load_joblib_artifact(ds: Dataset, artifact: ArtifactSpec) -> object:
    """
    Load a joblib artifact from a feature run root.

    Parameters
    ----------
    ds : Dataset
        Dataset instance
    artifact : ArtifactSpec
        Typed artifact specification with feature, run_id, pattern, and load spec.

    Returns
    -------
    object
        The loaded object, or obj[key] if load spec has a key
    """
    if not isinstance(artifact.load, JoblibLoadSpec):
        raise ValueError(
            f"load_joblib_artifact requires JoblibLoadSpec, "
            f"got {type(artifact.load).__name__}"
        )
    _, run_root = get_feature_run_root(ds, artifact.feature, artifact.run_id)
    resolved = resolve_artifact_file(
        "artifact", artifact.feature, run_root, artifact.pattern
    )
    if resolved is None:
        raise FileNotFoundError(f"No files matching '{artifact.pattern}' in {run_root}")
    return artifact.from_path(resolved)
