"""Data loading, artifact resolution, and key-data helpers.

Moved from ``mosaic.behavior.feature_library.helpers`` to live closer to the
pipeline infrastructure they depend on.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from ._loaders import JoblibLoadSpec, LoadSpec, ParquetLoadSpec
from .index import (
    feature_index,
    feature_index_path,
    feature_run_root,
    latest_feature_run_root,
    resolve_artifact_file,
)
from .types import ArtifactSpec, NNResult
from .types.data_config import ALIGN_COLS, COLUMNS

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
    what lets a submit-time check resolve a run's level cheaply. The value-sniffing
    twin, :func:`normalize_identity_columns`, additionally distinguishes a pair
    frame whose second id is all-null; this does not, and does not need to.

    **The pair spellings below are not every pair spelling in the library, and
    the fix is not to add one.** ``pair-facing`` and ``attention-target`` write
    ``focal_id`` with a target column, which none of these match, so both read
    as ``"global"`` -- no identity -- and a join against an individual-level
    input is then permitted on ``frame`` alone. Both declare ``emits = "pair"``,
    so a chain checked before it runs refuses that edge; this predicate, which
    runs at the merge itself, still does not.

    Those two features are to emit ``id1`` / ``id2`` like every other pair-level
    feature, with ``id1`` the focal individual and ``id2`` the target. A fourth
    spelling here would make this a list of names that grows every time a
    feature invents one, and each addition is a chance to forget; one spelling is
    what makes it a rule. The rename changes what those features *write*, so it
    is a change of its own.
    """
    present = set(columns)
    for a, b in (("id1", "id2"), ("id_a", "id_b"), ("id_A", "id_B")):
        if a in present and b in present:
            return "pair"
    return "individual" if COLUMNS.id_col in present else "global"


def alignment_verdict(column_sets: Sequence[Iterable[str]]) -> AlignmentVerdict:
    """Can these inputs be joined? The rule behind :class:`MultiInputAlignmentError`.

    Incompatible when two inputs carry identity at different levels and the keys
    they share carry no identity at all: joining an individual frame to a pair frame
    on ``frame`` alone is a per-frame cartesian product, not an alignment.
    """
    sets = [set(columns) for columns in column_sets]
    keys = frozenset(_ALIGN_COLS.intersection(*sets)) if sets else frozenset()
    levels = tuple(entity_level_of(columns) for columns in sets)
    identity = keys - {COLUMNS.frame_col, COLUMNS.time_col}
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


def _keypoint_sort_key(suffix: str) -> tuple[int, int, str]:
    """Order a pose suffix numerically where it is a number, lexically otherwise.

    Keypoint identity is positional: every caller that indexes into the returned
    list -- ``heading``'s ``front_idx`` / ``rear_idx``, the overlay's skeleton
    lines -- means "the Nth keypoint the converter emitted". A lexicographic sort
    breaks that silently from ten keypoints on, ordering ``poseX10`` between
    ``poseX1`` and ``poseX2``, so a 21-point midline is drawn and measured
    scrambled with nothing in the output to say so.

    Numeric suffixes sort first and among themselves by value; anything else
    keeps a stable lexicographic order after them, so a named keypoint set
    (``poseXhead``) is still ordered deterministically rather than raising.
    """
    if suffix.isdigit():
        return (0, int(suffix), "")
    return (1, 0, suffix)


# TODO: this shares logic with feature_library.params.PoseConfig
# so pose config, columns etc should also move to pipeline/core
def pose_column_pairs(columns: Iterable[str]) -> list[tuple[str, str]]:
    """Extract (poseX*, poseY*) column pairs, ordered by keypoint index.

    Args:
        columns: Column names to scan.

    Returns:
        The ``(poseX<k>, poseY<k>)`` pairs whose X and Y are both present, in
        keypoint order -- numerically for numeric suffixes. A ``poseX`` without
        its ``poseY`` is skipped rather than half-reported.
    """
    column_names = list(columns)
    present = set(column_names)
    suffixes = [c[len("poseX") :] for c in column_names if c.startswith("poseX")]
    return [
        (f"poseX{suffix}", f"poseY{suffix}")
        for suffix in sorted(suffixes, key=_keypoint_sort_key)
        if f"poseY{suffix}" in present
    ]


# is this really a valid approach? We may rather want to raise if not numeric but exists
# also, features decide whether they are "pair", "individual" or "global" internally, why do we need this here?
# just to write the __global__ marker row to the index?
def normalize_identity_columns(
    df: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None, str]:
    """
    Extract canonical identity columns from a frame-aligned DataFrame.

    Returns
    -------
    tuple
        (id1_series_or_None, id2_series_or_None, entity_level)
        where entity_level is one of {"global", "individual", "pair"}.
    """
    if "id1" in df.columns and "id2" in df.columns:
        id1 = pd.to_numeric(df["id1"], errors="coerce")
        id2 = pd.to_numeric(df["id2"], errors="coerce")
        if id1.notna().any() and id2.notna().any():
            return id1, id2, "pair"
        if id1.notna().any():
            return id1, pd.Series(np.nan, index=df.index), "individual"
        return None, None, "global"

    if "id" in df.columns:
        id1 = pd.to_numeric(df["id"], errors="coerce")
        if id1.notna().any():
            return id1, pd.Series(np.nan, index=df.index), "individual"
        return None, None, "global"

    # Backward-compatible aliases for older pair outputs
    for a_col, b_col in (("id_a", "id_b"), ("id_A", "id_B")):
        if a_col in df.columns and b_col in df.columns:
            id1 = pd.to_numeric(df[a_col], errors="coerce")
            id2 = pd.to_numeric(df[b_col], errors="coerce")
            if id1.notna().any() and id2.notna().any():
                return id1, id2, "pair"
            if id1.notna().any():
                return id1, pd.Series(np.nan, index=df.index), "individual"
            return None, None, "global"

    return None, None, "global"


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
    """
    it = iter(dfs)
    first = next(it, None)
    if first is None:
        return None
    merged = first[1]

    for declared, df_next in it:
        verdict = alignment_verdict([merged.columns, df_next.columns])
        if not verdict.compatible and not cross_join:
            raise MultiInputAlignmentError(
                f"cannot merge input {declared}: {verdict.reason}"
            )
        on_cols = sorted(verdict.keys)
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
        rename_map = {
            c: f"{c}__{declared}"
            for c in df_next.columns
            if c not in on_cols and c in merged.columns
        }
        if rename_map:
            df_next = df_next.rename(columns=rename_map)
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

    idx_df = idx.read(run_id=run_id, filter_ext=".parquet", entries=[(group, sequence)])
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
