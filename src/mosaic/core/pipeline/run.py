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

import pandas as pd
import pyarrow.parquet as pq

from mosaic.core.helpers import (
    filter_time_range,
    make_entry_key,
    resolve_frame_range,
)

from ._utils import (
    FeatureMeta,
    Scope,
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
from .types import ArtifactSpec, Feature, LabelsSource, NNResult, Params, Result
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
) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Introspect params fields to resolve upstream dependencies.

    Returns (artifact_paths, dependency_indices) where:
    - artifact_paths maps field name to resolved file/directory Path
    - dependency_indices maps field name to upstream index DataFrame
    """
    from .index import latest_feature_run_root as _latest_feature_run_root

    artifact_paths: dict[str, Path] = {}
    dependency_indices: dict[str, pd.DataFrame] = {}

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
                    _run_id, dep_root = _latest_feature_run_root(
                        ds, feature_name
                    )
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
                dependency_indices[field_name] = dep_index.read(
                    run_id=_run_id, filter_ext=".parquet"
                )

            case LabelsSource(kind=str(kind)):
                if not kind:
                    continue
                try:
                    labels_base = Path(ds.get_root("labels"))
                except KeyError:
                    msg = (
                        f"Dataset has no 'labels' root configured, "
                        f"required by field '{field_name}'"
                    )
                    raise FileNotFoundError(msg) from None
                labels_root = labels_base / kind
                if not labels_root.exists():
                    msg = f"Labels directory not found: {labels_root}"
                    raise FileNotFoundError(msg)
                artifact_paths[field_name] = labels_root

            case _:
                pass

    return artifact_paths, dependency_indices


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
    dependency_indices: dict[str, pd.DataFrame],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct a feature in a worker process and run apply."""
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    inputs_obj = cls.Inputs.model_validate(inputs_dump)
    feature = cls(inputs_obj, params_dump)
    feature.load_state(
        Path(run_root_str),
        {k: Path(v) for k, v in artifact_paths_str.items()},
        dependency_indices,
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
    suffix = feature.inputs.storage_suffix()
    if suffix is not None:
        storage_feature_name = f"{feature.name}__from__{suffix}"
    else:
        storage_feature_name = feature.name

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

    # Resolve dependencies
    artifact_paths, dependency_indices = _resolve_dependencies(ds, feature.params)

    # Resolve pair filter
    pair_filter_spec = _resolve_pair_filter(feature.params)

    # Load state
    state_ready = feature.load_state(run_root, artifact_paths, dependency_indices)

    # Build filter factory (shared by fit and apply phases)
    filter_factory = _make_filter_factory(
        ds, scope, pair_filter_spec, frame_start, frame_end
    )

    # Fit phase (if not state_ready)
    if not state_ready:

        def fit_factory() -> Iterator[tuple[str, pd.DataFrame]]:
            return iter_manifest(manifest, filter_factory=filter_factory)

        feature.fit(fit_factory)
        feature.save_state(run_root)

    # Apply phase
    out_rows: list[FeatureIndexRow] = []

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
            out_rows.append(
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
            out_rows.append(
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
                        dependency_indices,
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
            out_rows.append(
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
    if not out_rows and not manifest:
        out_rows.append(
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

    # Finalize
    idx.append(out_rows)
    idx.mark_finished(run_id)
    print(f"[feature:{storage_feature_name}] completed run_id={run_id} -> {run_root}")
    return Result(feature=storage_feature_name, run_id=run_id)
