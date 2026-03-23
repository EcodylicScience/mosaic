from __future__ import annotations

import dataclasses
import gc
import importlib
import json
import multiprocessing as mp
import sys
from collections.abc import Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.helpers import filter_time_range, resolve_frame_range, to_safe_name

from ._utils import (
    FeatureMeta,
    InputScope,
    build_scope_key,
    hash_params,
    json_ready,
)
from .index import (
    FeatureIndexRow,
    feature_index,
    feature_index_path,
    feature_run_root,
)
from .iteration import (
    inputset_from_inputs,
    resolve_feature_pairs,
    resolve_tracks_pairs,
    yield_feature_frames,
    yield_inputset_frames,
    yield_sequences,
    yield_sequences_with_overlap,
)
from .writers import FeatureOutput, trim_feature_output, write_output

if TYPE_CHECKING:
    from mosaic.behavior.feature_library.helpers import PartialIndexRow
    from mosaic.behavior.feature_library.spec import Feature
    from mosaic.core.dataset import Dataset


def build_output_path(group: str, sequence: str, run_root: Path) -> Path:
    """Build the parquet output path for a (group, sequence) pair."""
    safe_group = to_safe_name(group)
    safe_seq = to_safe_name(sequence)
    out_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
    return run_root / out_name


def build_feature_meta(group: str, sequence: str, run_root: Path) -> FeatureMeta:
    """Build FeatureMeta for a (group, sequence) pair."""
    return FeatureMeta(
        group=group,
        sequence=sequence,
        out_path=build_output_path(group, sequence, run_root),
    )


def process_transform_worker(payload):
    """
    Helper for process-based feature transforms.
    payload: (module, cls_name, params, df, extra_attrs, model_path)
    """
    module, cls_name, params, df, extra_attrs, model_path = payload
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    feat = cls(params)
    for name, val in (extra_attrs or {}).items():
        try:
            setattr(feat, name, val)
        except Exception:
            pass
    # Load fitted model if available
    if model_path and hasattr(feat, "load_model"):
        try:
            feat.load_model(Path(model_path))
        except Exception:
            pass
    return feat.transform(df)


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
):
    """
    Apply a Feature over a chosen scope (default: whole dataset).

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
        When >1 and the feature declares itself parallelizable, run the transform phase in parallel
        across per-sequence chunks using this many threads. Defaults to sequential execution.
    parallel_mode : {'thread','process'}
        Execution backend when parallel_workers > 1. 'thread' (default) uses ThreadPoolExecutor;
        'process' uses ProcessPoolExecutor and requires picklable params/feature/inputs.
    overlap_frames : int, default 0
        For continuous datasets, load this many frames from adjacent segments to handle
        edge effects in rolling windows, smoothing, etc. Only applies when inputs are tracks.
        When > 0, the transform receives data with overlap but output is trimmed to original bounds.
    filter_start_frame : int | None
        If set, only include frames >= this value.
    filter_end_frame : int | None
        If set, only include frames < this value.
    filter_start_time : float | None
        If set, converted to start frame via fps_default from dataset metadata.
    filter_end_time : float | None
        If set, converted to end frame via fps_default from dataset metadata.

    Behavior
    --------
    - Fits globally if feature.needs_fit().
      If feature.supports_partial_fit(), streams tables and calls partial_fit(); otherwise collects in memory.
    - Transforms per (group,sequence) table and writes:
          features/<feature>/<run_id>/<group_safe__sequence_safe>.parquet
    - Saves model (if any) under:
          features/<feature>/<run_id>/model.joblib
    - Returns run_id.
    """
    # Determine on-disk storage name (may encode upstream)
    storage_feature_name = feature.storage_feature_name
    use_input_suffix = feature.storage_use_input_suffix
    suffix = feature.inputs.storage_suffix()
    if suffix and use_input_suffix:
        storage_feature_name = f"{storage_feature_name}__from__{suffix}"

    # Prepare run id & root
    # Include scope in hash for features whose fit result depends on input scope
    scope_key = None
    _nf = (
        feature.needs_fit() if callable(getattr(feature, "needs_fit", None)) else False
    )
    _lo = feature.loads_own_data()
    if _nf and _lo:
        scope_key = build_scope_key(groups, sequences)

    frame_start, frame_end = resolve_frame_range(
        ds.meta.get("fps_default"),
        filter_start_frame,
        filter_end_frame,
        filter_start_time,
        filter_end_time,
    )
    _has_frame_filter = frame_start is not None or frame_end is not None

    hashable_params: dict[str, object] = {
        "_params": feature.params.model_dump(),
        "_inputs": feature.inputs.model_dump(),
    }
    if scope_key:
        hashable_params["_scope"] = scope_key
    if _has_frame_filter:
        hashable_params["_frame_range"] = [frame_start, frame_end]
    params_hash = hash_params(hashable_params)
    run_id = f"{feature.version}-{params_hash}"
    run_root = feature_run_root(ds, storage_feature_name, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    # Persist params (with scope) for discoverability
    params_path = run_root / "params.json"
    try:
        save_payload = {
            "_params": json_ready(feature.params),
            "_inputs": feature.inputs.model_dump(),
        }
        if scope_key is not None:
            save_payload["_scope"] = scope_key
        if _has_frame_filter:
            save_payload["_frame_range"] = [frame_start, frame_end]
        params_path.write_text(json.dumps(save_payload, indent=2))
    except Exception as exc:
        print(
            f"[feature:{feature.name}] failed to save params.json: {exc}",
            file=sys.stderr,
        )

    idx_path = feature_index_path(ds, storage_feature_name)
    idx = feature_index(idx_path)
    idx.ensure()

    max_workers = (
        int(parallel_workers) if parallel_workers and int(parallel_workers) > 1 else 1
    )
    parallel_mode = (parallel_mode or "thread").lower()
    if parallel_mode not in {"thread", "process"}:
        parallel_mode = "thread"
    parallel_allowed = feature.parallelizable
    if max_workers > 1 and not parallel_allowed:
        print(
            f"[feature:{feature.name}] parallel_workers requested but feature is not parallelizable; running sequentially.",
            file=sys.stderr,
        )
        max_workers = 1

    # Pre-pass: resolve candidate pairs, skip existing outputs before loading data
    input_scope: InputScope | None = None
    pairs_all: set[tuple[str, str]] | None = None
    pairs_to_compute: set[tuple[str, str]] | None = None
    preexisting_rows: list[FeatureIndexRow] = []
    resolved_input_run_id: str | None = None

    groups_set = {str(g) for g in groups} if groups is not None else None
    seq_set = {str(s) for s in sequences} if sequences is not None else None

    inputs = feature.inputs
    if inputs.is_single_tracks:
        try:
            pairs_all = resolve_tracks_pairs(ds, groups_set, seq_set)
        except FileNotFoundError:
            pairs_all = set()
    elif inputs.is_single_feature:
        fi = inputs.feature_inputs[0]
        pairs_all, resolved_input_ref = resolve_feature_pairs(
            ds, fi.feature, fi.run_id, groups_set, seq_set
        )
        resolved_input_run_id = resolved_input_ref.run_id
    elif inputs.is_multi:
        input_scope = inputset_from_inputs(ds, inputs, groups, sequences)
        pairs_all = input_scope.pairs
    elif feature.loads_own_data():
        if groups is not None or sequences is not None:
            raise ValueError(
                f"Feature '{feature.name}' has empty inputs and loads its "
                f"own data; groups/sequences filters cannot be applied."
            )
        pairs_all = set()
    else:
        raise ValueError("Feature.inputs is empty")

    if pairs_all is not None:
        pairs_to_compute = set()
        for pair in pairs_all:
            out_path = build_output_path(pair[0], pair[1], run_root)
            if out_path.exists() and not overwrite:
                if getattr(feature, "skip_existing_outputs", False):
                    continue  # neither compute nor index append
                g, s = pair
                g = "" if g is None else str(g)
                s = "" if s is None else str(s)
                preexisting_rows.append(
                    FeatureIndexRow(
                        run_id=run_id,
                        feature=storage_feature_name,
                        version=feature.version,
                        group=g,
                        sequence=s,
                        abs_path=out_path,
                        n_rows=0,
                        params_hash=params_hash,
                    )
                )
            else:
                pairs_to_compute.add(pair)

    # Choose input iterator (filtered to pairs_to_compute when known)
    use_overlap = overlap_frames > 0 and inputs.is_single_tracks

    if inputs.is_single_feature:
        _iter_fn = partial(
            yield_feature_frames,
            ds,
            inputs.feature_inputs[0].feature,
            resolved_input_run_id,
            groups,
            sequences,
            allowed_pairs=pairs_to_compute,
        )
    elif inputs.is_multi:
        if input_scope is None:
            raise ValueError("Feature.inputs is multi but no scope was resolved.")
        scope_for_iter = input_scope
        if pairs_to_compute is not None:
            scope_for_iter = dataclasses.replace(input_scope, pairs=pairs_to_compute)
        _wants_full = getattr(feature, "wants_full_inputset_data", None)
        _meta_only = callable(_wants_full) and not _wants_full()
        if _meta_only and _has_frame_filter:
            raise RuntimeError(
                f"[feature:{feature.name}] Time/frame filters are set but "
                f"this feature uses metadata-only iteration and cannot "
                f"apply them."
            )
        _iter_fn = partial(
            yield_inputset_frames,
            ds,
            groups,
            sequences,
            scope=scope_for_iter,
            metadata_only=_meta_only,
        )
        input_scope = scope_for_iter
    elif use_overlap:
        _iter_fn = partial(
            yield_sequences_with_overlap,
            ds,
            groups,
            sequences,
            allowed_pairs=pairs_to_compute,
            overlap_frames=overlap_frames,
        )
    else:
        _iter_fn = partial(
            yield_sequences, ds, groups, sequences, allowed_pairs=pairs_to_compute
        )

    def iter_inputs():
        for item in _iter_fn():
            if _has_frame_filter:
                g, s, df = item[0], item[1], item[2]
                df = filter_time_range(
                    df, filter_start_frame=frame_start, filter_end_frame=frame_end
                )
                if df.empty:
                    continue
                if len(item) > 3:
                    yield (g, s, df, *item[3:])
                else:
                    yield (g, s, df)
            else:
                yield item

    # ===== FIT PHASE =====
    feature.bind_dataset(ds)

    scope_constraints: dict[str, object] = {}
    if input_scope is not None:
        if input_scope.groups:
            scope_constraints["groups"] = input_scope.groups
        if input_scope.sequences:
            scope_constraints["sequences"] = input_scope.sequences
        if input_scope.safe_sequences:
            scope_constraints["safe_sequences"] = input_scope.safe_sequences
        if input_scope.pairs:
            scope_constraints["pairs"] = input_scope.pairs
    if groups is not None:
        norm_groups = sorted({str(g) for g in groups})
        if norm_groups:
            scope_constraints["groups"] = norm_groups
            scope_constraints["safe_groups"] = sorted(
                {to_safe_name(g) for g in norm_groups}
            )
    if sequences is not None:
        norm_sequences = sorted({str(s) for s in sequences})
        if norm_sequences:
            scope_constraints["sequences"] = norm_sequences
            if not scope_constraints.get("safe_sequences"):
                scope_constraints["safe_sequences"] = sorted(
                    {to_safe_name(s) for s in norm_sequences}
                )
    if _has_frame_filter:
        scope_constraints["frame_start"] = frame_start
        scope_constraints["frame_end"] = frame_end
    if scope_constraints:
        setattr(feature, "_scope_constraints", scope_constraints)
        _set_sc = getattr(feature, "set_scope_constraints", None)
        if callable(_set_sc):
            try:
                _set_sc(scope_constraints)
            except Exception as e:
                print(
                    f"[feature:{feature.name}] set_scope_constraints failed: {e}",
                    file=sys.stderr,
                )

    if input_scope is not None:
        feature.set_scope_filter(dataclasses.asdict(input_scope))

    # Validate time/frame filters against ds-loading features
    if _has_frame_filter and feature.loads_own_data():
        raise RuntimeError(
            f"[feature:{feature.name}] Time/frame filters are set but "
            f"this feature loads its own data and cannot apply them. "
            f"Remove the filters or use a feature that processes data "
            f"from the sequence iterator."
        )

    if feature.needs_fit():
        # Pass run_root to feature if it supports it (for streaming writes during fit)
        _set_rr = getattr(feature, "set_run_root", None)
        if callable(_set_rr):
            try:
                _set_rr(run_root)
            except Exception as e:
                print(
                    f"[feature:{feature.name}] set_run_root failed: {e}",
                    file=sys.stderr,
                )

        # Check if fit phase can be skipped (for global features with existing outputs)
        loads_own = feature.loads_own_data()
        model_path = run_root / "model.joblib"
        # Also check for global-specific artifacts (e.g., global_opentsne_embedding.joblib)
        embedding_path = run_root / "global_opentsne_embedding.joblib"
        fit_complete = model_path.exists() or embedding_path.exists()

        skip_fit = not overwrite and loads_own and fit_complete
        if skip_fit:
            print(
                f"[feature:{feature.name}] fit phase skipped (overwrite=False, outputs exist)",
                file=sys.stderr,
            )
        elif feature.supports_partial_fit():
            for item in iter_inputs():
                df = item[2]
                try:
                    feature.partial_fit(df)
                except Exception as e:
                    print(
                        f"[feature:{feature.name}] partial_fit failed: {e}",
                        file=sys.stderr,
                    )
            try:
                feature.finalize_fit()
            except Exception:
                pass
        else:
            # Check if feature loads its own data (e.g., GlobalTSNE) - avoid pre-loading
            if loads_own:
                # Feature will load data itself; pass empty iterator to satisfy protocol
                all_dfs = []
            else:
                all_dfs = []
                for item in iter_inputs():
                    df = item[2]
                    all_dfs.append(df)
            # Always call fit, even if no streamed inputs were found.
            # Many "global/artifact" features load their own matrices from disk.
            try:
                feature.fit(all_dfs)
            except TypeError:
                # Backward-compat: some features define fit() with no args.
                try:
                    getattr(feature, "fit")()
                except Exception as e:
                    print(
                        f"[feature:{feature.name}] fit() failed: {e}", file=sys.stderr
                    )

        # Save model state if any (only if fit was actually run)
        if not skip_fit:
            try:
                feature.save_model(model_path)
            except NotImplementedError:
                # Feature doesn't implement save_model() - this is optional, so just skip
                pass

    # ===== TRANSFORM PHASE =====
    out_rows: list[FeatureIndexRow] = list(preexisting_rows) if preexisting_rows else []
    had_transform_inputs = False

    def _append_row(meta: FeatureMeta, n_rows: int, abs_path: Path | None = None):
        out_rows.append(
            FeatureIndexRow(
                run_id=run_id,
                feature=storage_feature_name,
                version=feature.version,
                group=meta.group,
                sequence=meta.sequence,
                abs_path=abs_path or meta.out_path,
                n_rows=n_rows,
                params_hash=params_hash,
            )
        )

    def _append_external_row(row: PartialIndexRow):
        group = row.group
        sequence = row.sequence
        if not sequence:
            return
        abs_path: Path
        if row.abs_path:
            abs_path = Path(row.abs_path)
        else:
            safe_group = to_safe_name(group)
            safe_seq = to_safe_name(sequence)
            out_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
            abs_path = run_root / out_name
        n_rows = row.n_rows
        out_rows[:] = [
            r
            for r in out_rows
            if not (
                r.run_id == str(run_id) and r.group == group and r.sequence == sequence
            )
        ]
        out_rows.append(
            FeatureIndexRow(
                run_id=run_id,
                feature=storage_feature_name,
                version=feature.version,
                group=group,
                sequence=sequence,
                abs_path=abs_path,
                n_rows=n_rows,
                params_hash=params_hash,
            )
        )

    skip_transform_phase = bool(getattr(feature, "skip_transform_phase", False))
    if not skip_transform_phase:
        executor = None
        if max_workers > 1:
            if parallel_mode == "process":
                executor = ProcessPoolExecutor(
                    max_workers=max_workers, mp_context=mp.get_context("spawn")
                )
            else:
                executor = ThreadPoolExecutor(max_workers=max_workers)
        extra_attrs = {}
        for attr in ("_scope_filter", "_scope_constraints"):
            if hasattr(feature, attr):
                extra_attrs[attr] = getattr(feature, attr)

        def _collect_completed(
            pending: dict[Future[FeatureOutput], tuple[FeatureMeta, int, int]],
        ):
            """Wait for at least one future to finish; write results and free memory."""
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                meta, core_start, core_end = pending.pop(future)
                try:
                    df_feat: FeatureOutput = future.result()
                except Exception as e:
                    print(
                        f"[feature:{feature.name}] transform failed for ({meta.group},{meta.sequence}): {e}",
                        file=sys.stderr,
                    )
                    continue
                if use_overlap:
                    df_feat = trim_feature_output(df_feat, core_start, core_end)
                n_rows = write_output(meta, df_feat)
                _append_row(meta, n_rows)
                del df_feat
                gc.collect()

        # Transform loop - handle both 3-tuple and 5-tuple from iterators
        pending: dict[Future[FeatureOutput], tuple[FeatureMeta, int, int]] = {}
        for item in iter_inputs():
            g = str(item[0])
            s = str(item[1])
            df: pd.DataFrame = item[2]
            if use_overlap and len(item) >= 5:
                core_start, core_end = int(item[3]), int(item[4])
            else:
                core_start, core_end = 0, len(df)
            had_transform_inputs = True
            meta = build_feature_meta(g, s, run_root)
            out_path = Path(meta.out_path)
            if out_path.exists() and not overwrite:
                if getattr(feature, "skip_existing_outputs", False):
                    continue
                try:
                    n_rows = int(pd.read_parquet(out_path).shape[0])
                except Exception:
                    n_rows = 0
                _append_row(meta, n_rows, abs_path=out_path)
                continue

            if executor:
                # Bounded submission: drain completed futures before submitting
                # more, keeping at most max_workers tasks in flight.
                while len(pending) >= max_workers:
                    _collect_completed(pending)

                if parallel_mode == "process":
                    model_path = run_root / "model.joblib"
                    model_path_str = str(model_path) if model_path.exists() else None
                    payload = (
                        feature.__module__,
                        feature.__class__.__name__,
                        getattr(feature, "params", {}),
                        df,
                        extra_attrs,
                        model_path_str,
                    )
                    pending[executor.submit(process_transform_worker, payload)] = (
                        meta,
                        core_start,
                        core_end,
                    )
                else:
                    pending[executor.submit(feature.transform, df)] = (
                        meta,
                        core_start,
                        core_end,
                    )
            else:
                try:
                    df_feat: FeatureOutput = feature.transform(df)
                except Exception as e:
                    print(
                        f"[feature:{feature.name}] transform failed for ({g},{s}): {e}",
                        file=sys.stderr,
                    )
                    continue
                if use_overlap:
                    df_feat = trim_feature_output(df_feat, core_start, core_end)
                n_rows = write_output(meta, df_feat)
                _append_row(meta, n_rows)
                del df_feat
                gc.collect()

        # Drain remaining in-flight futures
        if executor:
            while pending:
                _collect_completed(pending)
            executor.shutdown(wait=True)

    # Features may emit outputs during fit()/save_model() and provide explicit index rows.
    _get_extra = getattr(feature, "get_additional_index_rows", None)
    if callable(_get_extra):
        extra_rows: list[PartialIndexRow] = []
        try:
            result = _get_extra()
            if isinstance(result, list):
                extra_rows = result
        except Exception as e:
            print(
                f"[feature:{feature.name}] get_additional_index_rows failed: {e}",
                file=sys.stderr,
            )
        for row in extra_rows:
            _append_external_row(row)

    # Some global-only features (e.g., clustering over standalone artifacts) never
    # receive streamed tables. In that case we still want to record the run in the
    # index so downstream tooling can discover the run_id.
    if not out_rows and not had_transform_inputs:
        marker_seq = "__global__"
        safe_marker_seq = to_safe_name(marker_seq)
        marker_path = run_root / f"{safe_marker_seq}.parquet"
        marker_df = pd.DataFrame({"run_marker": [True]})
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_df.to_parquet(marker_path, index=False)
        out_rows.append(
            FeatureIndexRow(
                run_id=run_id,
                feature=storage_feature_name,
                version=feature.version,
                group="",
                sequence=marker_seq,
                abs_path=marker_path,
                n_rows=int(len(marker_df)),
                params_hash=params_hash,
            )
        )

    idx.append(out_rows)
    idx.mark_finished(run_id)
    print(f"[feature:{storage_feature_name}] completed run_id={run_id} -> {run_root}")
    from mosaic.behavior.feature_library.spec import Result

    return Result(feature=storage_feature_name, run_id=run_id)
