from __future__ import annotations

import gc
import importlib
import json
import multiprocessing as mp
import sys
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mosaic.core.helpers import (
    filter_time_range,
    load_labels_for_feature_frames,
    make_entry_key,
    resolve_frame_range,
)

from ._utils import (
    FeatureMeta,
    Scope,
    derive_storage_name,
    hash_params,
    json_ready,
)
from .index import (
    FeatureIndexRow,
    feature_index,
    feature_index_path,
    feature_run_root,
)
from .loading import build_nn_lookup, nn_pair_mask, resolve_sequence_identity
from .manifest import FilterFactory, Manifest, build_manifest, iter_manifest
from .types import (
    COLUMNS,
    ArtifactSpec,
    DependencyLookup,
    Feature,
    InputStream,
    Inputs,
    LabelsSource,
    NNResult,
    Params,
    Result,
    ResultColumn,
    TrackInput,
    TracksColumn,
)
from .registry import FeatureRegistry
from .writers import FeatureOutput, trim_feature_output, write_output

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def build_output_path(group: str, sequence: str, run_root: Path) -> Path:
    """Build the parquet output path for a (group, sequence) entry."""
    return run_root / f"{make_entry_key(group, sequence)}.parquet"


def build_feature_meta(group: str, sequence: str, run_root: Path) -> FeatureMeta:
    """Build FeatureMeta for a (group, sequence) pair."""
    return FeatureMeta(
        group=group,
        sequence=sequence,
        out_path=build_output_path(group, sequence, run_root),
    )


# --- Dependency resolution ---


def _resolve_dependencies(
    ds: Dataset, params: Params
) -> tuple[dict[str, Path], dict[str, DependencyLookup]]:
    """Introspect params fields to resolve upstream dependencies.

    Returns (artifact_paths, dependency_lookups) where:
    - artifact_paths maps field name to resolved file/directory Path
    - dependency_lookups maps field name to (group, sequence) -> Path dict
    """
    from .index import latest_feature_run_root as _latest_feature_run_root

    artifact_paths: dict[str, Path] = {}
    dependency_lookups: dict[str, DependencyLookup] = {}

    for field_name in type(params).model_fields:
        value = getattr(params, field_name)

        match value:
            case NNResult():
                # Pair filter -- resolved separately by _resolve_pair_filter
                continue

            case ArtifactSpec(
                feature=str(feature_name),
                run_id=_run_id,
                pattern=str(pattern),
            ):
                if not feature_name:
                    continue
                if _run_id is None:
                    _run_id, dep_root = _latest_feature_run_root(ds, feature_name)
                else:
                    dep_root = feature_run_root(ds, feature_name, _run_id)
                files = sorted(dep_root.glob(pattern))
                if files:
                    artifact_paths[field_name] = files[0]

            case Result(feature=str(feature_name), run_id=_run_id):
                if not feature_name:
                    continue
                if feature_name == "nearest-neighbor":
                    msg = (
                        f"Field '{field_name}' references the nearest-neighbor "
                        f"feature but is typed as Result, not NNResult"
                    )
                    raise TypeError(msg)
                dep_index = feature_index(feature_index_path(ds, feature_name))
                if _run_id is None:
                    _run_id = dep_index.latest_run_id()
                index_df = dep_index.read(
                    run_id=_run_id, filter_ext=".parquet"
                )
                lookup: DependencyLookup = {}
                for _, row in index_df.iterrows():
                    group = str(row.get("group", ""))
                    sequence = str(row.get("sequence", ""))
                    abs_path = str(row.get("abs_path", ""))
                    if abs_path:
                        lookup[(group, sequence)] = ds.resolve_path(abs_path)
                dependency_lookups[field_name] = lookup

            case LabelsSource(kind=str(kind)):
                if not kind:
                    continue
                lookup = _build_labels_lookup(ds, kind)
                if not lookup:
                    msg = (
                        f"Labels lookup is empty for kind={kind!r}, "
                        f"required by field '{field_name}'"
                    )
                    raise FileNotFoundError(msg)
                dependency_lookups[field_name] = lookup

            case _:
                pass

    return artifact_paths, dependency_lookups


def _resolve_pair_filter(params: Params) -> NNResult | None:
    """Scan params for a pair_filter field containing a populated NNResult."""
    value = getattr(params, "pair_filter", None)
    if isinstance(value, NNResult) and value.feature:
        return value
    return None


def _build_nn_filter(
    ds: Dataset, group: str, sequence: str, pair_filter_spec: NNResult
) -> Callable[[pd.DataFrame], pd.DataFrame] | None:
    """Build a DataFrame filter that keeps only nearest-neighbor pairs."""
    nn_lookup = build_nn_lookup(ds, group, sequence, pair_filter_spec)
    if not nn_lookup:
        return None

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        mask = nn_pair_mask(df, nn_lookup)
        return df.loc[mask].reset_index(drop=True)

    return _filter


# --- Process worker ---


def _process_apply_worker(
    module_name: str,
    class_name: str,
    inputs_dump: dict[str, object],
    params_dump: dict[str, object],
    run_root_str: str,
    artifact_paths_str: dict[str, str],
    dependency_lookups: dict[str, DependencyLookup],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct a feature in a worker process and run apply."""
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    inputs_obj = Inputs.model_validate(inputs_dump)
    feature = cls(inputs_obj, params_dump)
    feature.load_state(
        Path(run_root_str),
        {k: Path(v) for k, v in artifact_paths_str.items()},
        dependency_lookups,
    )
    return feature.apply(df)


# --- Filter factory ---


def _make_filter_factory(
    ds: Dataset,
    scope: Scope,
    pair_filter_spec: NNResult | None,
    frame_start: int | None,
    frame_end: int | None,
) -> FilterFactory | None:
    """Build a filter factory for iter_manifest.

    Returns None when no filtering is needed (no pair filter, no frame range).
    """
    has_frame_filter = frame_start is not None or frame_end is not None
    if pair_filter_spec is None and not has_frame_filter:
        return None

    def factory(entry_key: str) -> list[Callable[[pd.DataFrame], pd.DataFrame]]:
        filters: list[Callable[[pd.DataFrame], pd.DataFrame]] = []
        if pair_filter_spec is not None:
            group, sequence = resolve_sequence_identity(entry_key, scope.entry_map)
            nn_filter = _build_nn_filter(ds, group, sequence, pair_filter_spec)
            if nn_filter is not None:
                filters.append(nn_filter)
        if has_frame_filter:

            def frame_filter(df: pd.DataFrame) -> pd.DataFrame:
                return filter_time_range(
                    df, filter_start_frame=frame_start, filter_end_frame=frame_end
                )

            filters.append(frame_filter)
        return filters

    return factory


# --- Main entry point ---


def run_feature(
    ds: Dataset,
    feature: Feature,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    overwrite: bool = False,
    parallel_workers: int | None = None,
    parallel_mode: str | None = "thread",
    overlap_frames: int = 0,
    filter_start_frame: int | None = None,
    filter_end_frame: int | None = None,
    filter_start_time: float | None = None,
    filter_end_time: float | None = None,
    registry: FeatureRegistry | None = None,
) -> Result:
    """Apply a Feature over a chosen scope (default: whole dataset).

    Input routing is determined by ``feature.inputs``: tracks (default),
    a single upstream feature, or a multi-input set.

    Parameters
    ----------
    feature : Feature
        The feature object implementing the Feature protocol.  Its ``inputs``
        attribute controls where data is read from.
    groups, sequences : optional iterables
        Scope filter (applies to whichever input source is used).
    overwrite : bool
        Overwrite existing outputs for this run_id.
    parallel_workers : int | None
        When >1 and the feature declares itself parallelizable, run the apply
        phase in parallel. Defaults to sequential execution.
    parallel_mode : {'thread','process'}
        Execution backend when parallel_workers > 1. 'thread' (default) uses
        ThreadPoolExecutor; 'process' uses ProcessPoolExecutor.
    overlap_frames : int, default 0
        Load this many frames from adjacent segments to handle edge effects.
        Mutually exclusive with frame/time filters.
    filter_start_frame : int | None
        If set, only include frames >= this value.
    filter_end_frame : int | None
        If set, only include frames < this value.
    filter_start_time : float | None
        If set, converted to start frame via fps_default from dataset metadata.
    filter_end_time : float | None
        If set, converted to end frame via fps_default from dataset metadata.
    """
    # Storage name derivation
    storage_feature_name = derive_storage_name(
        feature.name, feature.inputs.storage_suffix()
    )

    # Frame range + mutual exclusivity with overlap
    frame_start, frame_end = resolve_frame_range(
        ds.meta.get("fps_default"),
        filter_start_frame,
        filter_end_frame,
        filter_start_time,
        filter_end_time,
    )
    has_frame_filter = frame_start is not None or frame_end is not None
    if has_frame_filter and overlap_frames > 0:
        raise ValueError("Frame/time filters and overlap_frames are mutually exclusive")

    # Scope sets
    groups_set = {str(g) for g in groups} if groups is not None else None
    sequences_set = {str(s) for s in sequences} if sequences is not None else None

    # Build manifest
    if feature.inputs.is_empty:
        manifest: Manifest = {}
        scope = Scope()
    else:
        manifest, scope = build_manifest(ds, feature.inputs, groups_set, sequences_set)

    # Run ID hash
    hashable: dict[str, object] = {
        "_params": feature.params.model_dump(),
        "_inputs": feature.inputs.model_dump(),
        "_frame_range": [frame_start, frame_end],
    }
    if feature.scope_dependent:
        hashable["_scope_entries"] = sorted(scope.entries)
    params_hash = hash_params(hashable)
    run_id = f"{feature.version}-{params_hash}"

    # Run root + params.json
    run_root = feature_run_root(ds, storage_feature_name, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    params_path = run_root / "params.json"
    try:
        save_payload: dict[str, object] = {
            "_params": json_ready(feature.params),
            "_inputs": feature.inputs.model_dump(),
            "_frame_range": [frame_start, frame_end],
        }
        params_path.write_text(json.dumps(save_payload, indent=2))
    except Exception as exc:
        print(
            f"[feature:{feature.name}] failed to save params.json: {exc}",
            file=sys.stderr,
        )

    # Index CSV setup
    idx = feature_index(feature_index_path(ds, storage_feature_name))
    idx.ensure()

    # Registry: record run start
    if registry is not None:
        registry.record_run_start(
            feature=storage_feature_name,
            run_id=run_id,
            version=feature.version,
            params_hash=params_hash,
            params_json=params_path.read_text() if params_path.exists() else None,
            inputs_json=json.dumps(json_ready(feature.inputs.model_dump())),
        )

    # Bind dataset (for features that need media paths, etc.)
    if hasattr(feature, "bind_dataset"):
        feature.bind_dataset(ds)

    # Resolve dependencies
    artifact_paths, dependency_lookups = _resolve_dependencies(ds, feature.params)

    # Resolve pair filter
    pair_filter_spec = _resolve_pair_filter(feature.params)

    # Load state
    state_ready = feature.load_state(run_root, artifact_paths, dependency_lookups)

    # Build filter factory (shared by fit and apply phases)
    filter_factory = _make_filter_factory(
        ds, scope, pair_filter_spec, frame_start, frame_end
    )

    # Fit phase (if not state_ready)
    if not state_ready:

        def input_factory() -> Iterator[tuple[str, pd.DataFrame]]:
            return iter_manifest(manifest, filter_factory=filter_factory)

        feature.fit(InputStream(input_factory, n_entries=len(manifest)))
        feature.save_state(run_root)

    # Apply phase — index rows are flushed periodically for interrupt recovery
    _IDX_FLUSH_EVERY = 10
    _pending_idx_rows: list[FeatureIndexRow] = []
    _total_written = 0

    def _flush_idx() -> None:
        nonlocal _total_written
        if _pending_idx_rows:
            idx.append(list(_pending_idx_rows))
            _total_written += len(_pending_idx_rows)
            _pending_idx_rows.clear()

    def _record_row(row: FeatureIndexRow) -> None:
        _pending_idx_rows.append(row)
        if registry is not None:
            registry.record_entry(
                feature=row.feature,
                run_id=row.run_id,
                group=row.group,
                sequence=row.sequence,
                abs_path=row.abs_path,
                n_rows=row.n_rows,
            )
        if len(_pending_idx_rows) >= _IDX_FLUSH_EVERY:
            _flush_idx()

    max_workers = (
        parallel_workers if parallel_workers is not None and parallel_workers > 1 else 1
    )
    parallel_mode_str = (parallel_mode or "thread").lower()
    if parallel_mode_str not in {"thread", "process"}:
        parallel_mode_str = "thread"
    if max_workers > 1 and not feature.parallelizable:
        print(
            f"[feature:{feature.name}] parallel_workers requested but feature is not parallelizable; running sequentially.",
            file=sys.stderr,
        )
        max_workers = 1
    apply_overlap: int | None = overlap_frames if overlap_frames > 0 else None

    executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
    if max_workers > 1:
        if parallel_mode_str == "process":
            executor = ProcessPoolExecutor(
                max_workers=max_workers, mp_context=mp.get_context("spawn")
            )
        else:
            executor = ThreadPoolExecutor(max_workers=max_workers)

    pending: dict[Future[pd.DataFrame], tuple[FeatureMeta, int, int]] = {}

    def _drain_completed() -> None:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            meta, core_start, core_end = pending.pop(future)
            try:
                result_df: FeatureOutput = future.result()
            except Exception as exc:
                print(
                    f"[feature:{feature.name}] apply failed for ({meta.group},{meta.sequence}): {exc}",
                    file=sys.stderr,
                )
                continue
            if apply_overlap is not None and apply_overlap > 0:
                result_df = trim_feature_output(result_df, core_start, core_end)
            n_rows = write_output(meta, result_df)
            _record_row(
                FeatureIndexRow(
                    run_id=run_id,
                    feature=storage_feature_name,
                    version=feature.version,
                    group=meta.group,
                    sequence=meta.sequence,
                    abs_path=meta.out_path,
                    n_rows=n_rows,
                    params_hash=params_hash,
                )
            )
            del result_df
            gc.collect()

    def _process_entry(
        entry_key: str,
        df: pd.DataFrame,
        core_start: int,
        core_end: int,
    ) -> None:
        group, sequence = resolve_sequence_identity(entry_key, scope.entry_map)
        meta = build_feature_meta(group, sequence, run_root)

        # Skip existing outputs when state was loaded from cache
        if state_ready and not overwrite and meta.out_path.exists():
            n_rows = int(pq.read_metadata(meta.out_path).num_rows)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            _record_row(
                FeatureIndexRow(
                    run_id=run_id,
                    feature=storage_feature_name,
                    version=feature.version,
                    group=meta.group,
                    sequence=meta.sequence,
                    abs_path=meta.out_path,
                    n_rows=n_rows,
                    params_hash=params_hash,
                )
            )
            return

        if executor is not None:
            while len(pending) >= max_workers:
                _drain_completed()
            if parallel_mode_str == "process":
                artifact_paths_str = {k: str(v) for k, v in artifact_paths.items()}
                pending[
                    executor.submit(
                        _process_apply_worker,
                        feature.__module__,
                        type(feature).__name__,
                        feature.inputs.model_dump(),
                        feature.params.model_dump(),
                        str(run_root),
                        artifact_paths_str,
                        dependency_lookups,
                        df,
                    )
                ] = (meta, core_start, core_end)
            else:
                pending[executor.submit(feature.apply, df)] = (
                    meta,
                    core_start,
                    core_end,
                )
        else:
            try:
                result_df: FeatureOutput = feature.apply(df)
            except Exception as exc:
                print(
                    f"[feature:{feature.name}] apply failed for ({group},{sequence}): {exc}",
                    file=sys.stderr,
                )
                return
            if apply_overlap is not None and apply_overlap > 0:
                result_df = trim_feature_output(result_df, core_start, core_end)
            n_rows = write_output(meta, result_df)
            _record_row(
                FeatureIndexRow(
                    run_id=run_id,
                    feature=storage_feature_name,
                    version=feature.version,
                    group=group,
                    sequence=sequence,
                    abs_path=meta.out_path,
                    n_rows=n_rows,
                    params_hash=params_hash,
                )
            )
            del result_df
            gc.collect()

    # Iterate manifest entries
    if apply_overlap is not None:
        for entry_key, df, core_start, core_end in iter_manifest(
            manifest,
            filter_factory=filter_factory,
            overlap_frames=apply_overlap,
            progress_label=storage_feature_name,
        ):
            _process_entry(entry_key, df, core_start, core_end)
    else:
        for entry_key, df in iter_manifest(
            manifest,
            filter_factory=filter_factory,
            progress_label=storage_feature_name,
        ):
            _process_entry(entry_key, df, 0, len(df))

    # Drain remaining futures
    if executor is not None:
        while pending:
            _drain_completed()
        executor.shutdown(wait=True)

    # Global marker (for empty-input features)
    if _total_written == 0 and not _pending_idx_rows and not manifest:
        _pending_idx_rows.append(
            FeatureIndexRow(
                run_id=run_id,
                feature=storage_feature_name,
                version=feature.version,
                group="",
                sequence="__global__",
                abs_path=run_root,
                n_rows=0,
                params_hash=params_hash,
            )
        )

    # Finalize — flush any remaining rows
    _flush_idx()
    idx.mark_finished(run_id)
    if registry is not None:
        registry.mark_finished(storage_feature_name, run_id)
    print(f"[feature:{storage_feature_name}] completed run_id={run_id} -> {run_root}")
    return Result(feature=storage_feature_name, run_id=run_id)


# --- load_values ---


ValueSource = ResultColumn | TracksColumn | LabelsSource


def _source_column_name(source: ValueSource) -> str:
    """Derive output column name for a source."""
    if isinstance(source, (TracksColumn, ResultColumn)):
        return source.column
    return f"labels-{source.kind}"


def _deduplicate_column_names(names: list[str]) -> list[str]:
    """First occurrence bare, subsequent get -1, -2, ..."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        result.append(name if count == 0 else f"{name}-{count}")
    return result


def _build_labels_lookup(
    ds: Dataset, kind: str
) -> dict[tuple[str, str], Path]:
    try:
        labels_root = Path(ds.get_root("labels")) / kind
    except KeyError:
        return {}
    idx_path = labels_root / "index.csv"
    if not idx_path.exists():
        return {}
    df = pd.read_csv(idx_path, keep_default_na=False)
    lookup: dict[tuple[str, str], Path] = {}
    for _, row in df.iterrows():
        group = str(row.get("group", ""))
        sequence = str(row.get("sequence", ""))
        abs_path_raw = str(row.get("abs_path", ""))
        if abs_path_raw:
            path = ds.resolve_path(abs_path_raw)
            if path.exists():
                lookup[(group, sequence)] = path
    return lookup


def _find_merged_column(
    column: str, input_index: int, df: pd.DataFrame
) -> str | None:
    if input_index == 0:
        return column if column in df.columns else None
    suffixed = f"{column}__{input_index}"
    if suffixed in df.columns:
        return suffixed
    return column if column in df.columns else None


def load_values(
    ds: Dataset,
    sources: Iterable[ValueSource],
    *,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    filter_start_frame: int | None = None,
    filter_end_frame: int | None = None,
    filter_start_time: float | None = None,
    filter_end_time: float | None = None,
    pair_filter: NNResult | None = None,
) -> pd.DataFrame:
    """Load and align value columns from tracks, features, and labels.

    Sources can reference tracks columns, feature output columns, or
    ground-truth labels. All are aligned by frame/id via a single
    manifest pass.
    """
    source_list = list(sources)
    if not source_list:
        return pd.DataFrame()

    tracks_columns = [s for s in source_list if isinstance(s, TracksColumn)]
    result_columns = [s for s in source_list if isinstance(s, ResultColumn)]
    label_sources = [s for s in source_list if isinstance(s, LabelsSource)]

    if not tracks_columns and not result_columns:
        msg = "load_values requires at least one TracksColumn or ResultColumn"
        raise ValueError(msg)

    # Column naming
    raw_names = [_source_column_name(s) for s in source_list]
    column_names = _deduplicate_column_names(raw_names)

    # Build synthetic Inputs from unique sources
    unique_inputs: dict[str | tuple[str, str | None], int] = {}
    input_items: list[TrackInput | Result] = []

    if tracks_columns:
        unique_inputs["tracks"] = len(unique_inputs)
        input_items.append("tracks")

    for rc in result_columns:
        key = (rc.feature, rc.run_id)
        if key not in unique_inputs:
            unique_inputs[key] = len(unique_inputs)
            input_items.append(Result(feature=rc.feature, run_id=rc.run_id))

    # Map each source index to its input index for merged column resolution
    column_input_indices: dict[int, int] = {}
    for source_idx, source in enumerate(source_list):
        if isinstance(source, TracksColumn):
            column_input_indices[source_idx] = unique_inputs["tracks"]
        elif isinstance(source, ResultColumn):
            column_input_indices[source_idx] = unique_inputs[
                (source.feature, source.run_id)
            ]

    synthetic_inputs = Inputs(tuple(input_items))

    # Scope
    groups_set = {str(g) for g in groups} if groups is not None else None
    sequences_set = {str(s) for s in sequences} if sequences is not None else None

    manifest, scope = build_manifest(ds, synthetic_inputs, groups_set, sequences_set)

    # Frame range
    frame_start, frame_end = resolve_frame_range(
        ds.meta.get("fps_default"),
        filter_start_frame,
        filter_end_frame,
        filter_start_time,
        filter_end_time,
    )

    filter_factory = _make_filter_factory(
        ds, scope, pair_filter, frame_start, frame_end
    )

    # Pre-load labels lookups
    labels_lookups: dict[str, dict[tuple[str, str], Path]] = {}
    for ls in label_sources:
        if ls.kind not in labels_lookups:
            labels_lookups[ls.kind] = _build_labels_lookup(ds, ls.kind)

    meta_cols = COLUMNS.meta_set() | {"id1", "id2"}

    all_parts: list[pd.DataFrame] = []

    for entry_key, entry_df in iter_manifest(manifest, filter_factory=filter_factory):
        group, sequence = resolve_sequence_identity(entry_key, scope.entry_map)

        entry_data: dict[str, object] = {}

        for col in meta_cols:
            if col in entry_df.columns:
                entry_data[col] = entry_df[col].values
        if "group" not in entry_data:
            entry_data["group"] = [group] * len(entry_df)
        if "sequence" not in entry_data:
            entry_data["sequence"] = [sequence] * len(entry_df)

        for source_idx, source in enumerate(source_list):
            col_name = column_names[source_idx]

            if isinstance(source, (TracksColumn, ResultColumn)):
                input_idx = column_input_indices[source_idx]
                resolved = _find_merged_column(source.column, input_idx, entry_df)
                if resolved is not None:
                    entry_data[col_name] = entry_df[resolved].values

            else:
                label_lookup = labels_lookups.get(source.kind, {})
                label_path = label_lookup.get((group, sequence))
                if label_path is not None and "frame" in entry_df.columns:
                    feature_frames = entry_df["frame"].to_numpy()
                    entry_data[col_name] = load_labels_for_feature_frames(
                        label_path, feature_frames, default_label=0
                    )
                else:
                    entry_data[col_name] = np.zeros(len(entry_df), dtype=np.int64)

        if entry_data:
            all_parts.append(pd.DataFrame(entry_data))

    if not all_parts:
        return pd.DataFrame(columns=sorted(meta_cols) + column_names)

    return pd.concat(all_parts, ignore_index=True)
