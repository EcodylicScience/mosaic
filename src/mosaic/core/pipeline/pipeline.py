"""Declarative feature pipeline orchestrator.

Wraps ``dataset.run_feature()`` with a declarative graph of named steps.
Caching is automatic: same params + inputs → same run_id → skip.

Example
-------
>>> pipe = Pipeline(default_run_kwargs={"filter_start_time": 10.0})
>>> pipe.add(FeatureStep("smooth", TrajectorySmooth, {"window": 5}))
>>> pipe.add(FeatureStep("speed", SpeedAngvel, {}, ["smooth"]))
>>> pipe.status(dataset)   # show cached / stale / pending
>>> pipe.run(dataset)      # execute, skipping cached steps
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from mosaic.core.helpers import resolve_frame_range

from ._utils import derive_storage_name, hash_params
from .index import feature_index_path, feature_run_root, list_feature_runs
from .registry import open_registry
from .types import Inputs, Result

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


# ---------------------------------------------------------------------------
# Step dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FeatureStep:
    """One node in the pipeline graph.

    Parameters
    ----------
    name : str
        Unique identifier for this step (used in ``input_names`` of
        downstream steps).
    feature_cls : type
        Feature class to instantiate.  Must follow the Feature protocol.
    params : dict | None
        Parameter overrides passed to the feature constructor.
    input_names : list[str]
        Names of upstream FeatureSteps whose ``Result`` objects are wired
        into this feature's ``Inputs``.  Empty means the feature reads
        directly from tracks.
    run_kwargs : dict
        Extra keyword arguments forwarded to ``dataset.run_feature()``
        (e.g. ``parallel_workers``, ``filter_start_time``).
    """

    name: str
    feature_cls: type
    params: dict | None = None
    input_names: list[str] = field(default_factory=list)
    run_kwargs: dict = field(default_factory=dict)


@dataclass
class CallbackStep:
    """A custom function that runs between feature layers.

    Parameters
    ----------
    name : str
        Unique identifier for this step.
    fn : Callable
        Called as ``fn(dataset, results_so_far)``.
    depends_on : list[str]
        Upstream step names (used for staleness tracking).
    """

    name: str
    fn: Callable[["Dataset", dict[str, Result]], None]
    depends_on: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _newest_mtime(run_root: "Path") -> str | None:
    """Return the newest parquet mtime in a run directory as a short timestamp."""
    from datetime import datetime, timezone
    from pathlib import Path

    try:
        parquets = list(run_root.glob("*.parquet"))
        if not parquets:
            return None
        newest = max(f.stat().st_mtime for f in parquets)
        return datetime.fromtimestamp(newest, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return None


def storage_name(feature: object) -> str:
    """Compute on-disk storage name, mirroring ``run_feature`` logic.

    If the feature reads from upstream features, the directory includes
    a ``__from__`` suffix (e.g. ``speed__from__tracks``).
    """
    return derive_storage_name(feature.name, feature.inputs.storage_suffix())  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Lightweight feature pipeline orchestrator.

    Wraps ``dataset.run_feature()`` with a declarative graph of named
    steps.  Caching is automatic (same params → same run_id → skip).
    """

    def __init__(self, default_run_kwargs: dict | None = None) -> None:
        self.steps: list[FeatureStep | CallbackStep] = []
        self.results: dict[str, Result] = {}
        self.default_run_kwargs: dict = default_run_kwargs or {}

    # -- step registration --------------------------------------------------

    def add(self, step: FeatureStep | CallbackStep) -> Pipeline:
        """Register a step and return self for chaining."""
        existing = {s.name for s in self.steps}
        if step.name in existing:
            msg = f"Duplicate step name: {step.name!r}"
            raise ValueError(msg)

        # Validate that referenced inputs exist
        refs = (
            step.input_names
            if isinstance(step, FeatureStep)
            else step.depends_on
        )
        for ref in refs:
            if ref not in existing:
                msg = (
                    f"Step {step.name!r} references unknown upstream "
                    f"{ref!r}. Available: {sorted(existing)}"
                )
                raise ValueError(msg)

        self.steps.append(step)
        return self

    # -- cache resolution ---------------------------------------------------

    def _resolve_step_cache(self, dataset: Dataset) -> list[dict]:
        """Compute expected run_ids and check cache state for all steps."""
        mock_results: dict[str, Result] = {}
        stale_steps: set[str] = set()
        resolved: list[dict] = []

        # Inverted DAG: {parent -> [children]} for staleness propagation
        adj = self.dag()
        children_of: dict[str, list[str]] = {name: [] for name in adj}
        for child, parents in adj.items():
            for parent in parents:
                if parent in children_of:
                    children_of[parent].append(child)

        for step in self.steps:
            step_is_stale = step.name in stale_steps

            if isinstance(step, CallbackStep):
                # Mark stale if any dependency is not cached
                if not step_is_stale:
                    for dep in step.depends_on:
                        dep_info = next(
                            (r for r in resolved if r["step"].name == dep), None
                        )
                        if dep_info and not dep_info["cached"]:
                            step_is_stale = True
                            break
                # Propagate staleness to children if this callback is stale
                if step_is_stale:
                    for child_name in children_of.get(step.name, []):
                        stale_steps.add(child_name)
                resolved.append({
                    "step": step,
                    "storage_name": None,
                    "expected_run_id": None,
                    "cached": None,
                    "stale": step_is_stale,
                    "mock_result": None,
                })
                continue

            try:
                # Build inputs from upstream results
                if step.input_names:
                    input_items = tuple(
                        mock_results[n] for n in step.input_names
                    )
                else:
                    input_items = ("tracks",)

                feature = step.feature_cls(
                    inputs=Inputs(input_items), params=step.params
                )
                feat_storage_name = storage_name(feature)

                # Resolve frame range from merged kwargs
                kwargs = {**self.default_run_kwargs, **step.run_kwargs}
                frame_start, frame_end = resolve_frame_range(
                    dataset.meta.get("fps_default"),
                    kwargs.get("filter_start_frame"),
                    kwargs.get("filter_end_frame"),
                    kwargs.get("filter_start_time"),
                    kwargs.get("filter_end_time"),
                )

                # Compute expected run_id (same logic as run_feature)
                hashable: dict[str, object] = {
                    "_params": feature.params.model_dump(),
                    "_inputs": feature.inputs.model_dump(),
                    "_frame_range": [frame_start, frame_end],
                }
                expected_run_id = f"{feature.version}-{hash_params(hashable)}"

                # Check cache on disk
                run_root = feature_run_root(
                    dataset, feat_storage_name, expected_run_id
                )
                cached = run_root.exists() and any(run_root.glob("*.parquet"))

                # Staleness: if this step was marked stale by an upstream,
                # its cache cannot be trusted
                if step_is_stale:
                    cached = False

                result = Result(
                    feature=feat_storage_name,
                    run_id=expected_run_id if cached else None,
                )
                mock_results[step.name] = result

                resolved.append({
                    "step": step,
                    "storage_name": feat_storage_name,
                    "expected_run_id": expected_run_id,
                    "cached": cached,
                    "stale": step_is_stale,
                    "mock_result": result,
                    "feature_short": feature.name,
                })

            except Exception as e:
                mock_results[step.name] = Result(
                    feature=step.feature_cls.name, run_id=None
                )
                resolved.append({
                    "step": step,
                    "storage_name": step.feature_cls.name,
                    "expected_run_id": None,
                    "cached": False,
                    "stale": step_is_stale,
                    "mock_result": None,
                    "feature_short": step.feature_cls.name,
                    "error": str(e)[:60],
                })

            # If this step is not cached, mark its direct children stale
            # (transitive propagation happens naturally via topological order)
            if not resolved[-1]["cached"]:
                for child_name in children_of.get(step.name, []):
                    stale_steps.add(child_name)

        return resolved

    # -- public API ---------------------------------------------------------

    def status(self, dataset: Dataset) -> pd.DataFrame:
        """Show pipeline status: which steps are cached, their run_ids."""
        resolved = self._resolve_step_cache(dataset)
        rows = []

        for info in resolved:
            step = info["step"]

            if isinstance(step, CallbackStep):
                rows.append({
                    "step": step.name,
                    "feature": "(callback)",
                    "run_id": "-",
                    "n_seq": "-",
                    "runs": "-",
                    "cached": "-" if not info["stale"] else "stale",
                    "modified": "-",
                })
                continue

            try:
                runs_df = list_feature_runs(dataset, info["storage_name"])
                n_runs = len(runs_df["run_id"].unique())
                if info["cached"]:
                    matched = runs_df[
                        runs_df["run_id"] == info["expected_run_id"]
                    ]
                    n_seq = (
                        int(matched.iloc[0].get("n_entries", 0))
                        if not matched.empty
                        else 0
                    )
                else:
                    n_seq = 0
            except (FileNotFoundError, ValueError):
                n_runs = 0
                n_seq = 0

            cached_display = info["cached"]
            if info["stale"] and not info["cached"]:
                run_root = feature_run_root(
                    dataset,
                    info["storage_name"],
                    info["expected_run_id"] or "",
                )
                if info["expected_run_id"] and run_root.exists():
                    cached_display = "stale"

            # Modification timestamp (show even for stale runs if files exist)
            modified = None
            if info["expected_run_id"]:
                rr = feature_run_root(
                    dataset, info["storage_name"], info["expected_run_id"]
                )
                if rr.exists():
                    modified = _newest_mtime(rr)

            row: dict[str, object] = {
                "step": step.name,
                "feature": info.get("feature_short", info["storage_name"]),
                "run_id": (
                    info["expected_run_id"] if info["cached"] else "\u2014"
                ),
                "n_seq": n_seq,
                "runs": n_runs,
                "cached": cached_display,
                "modified": modified,
            }
            if "error" in info:
                row["error"] = info["error"]
            rows.append(row)

        return pd.DataFrame(rows)

    def load(self, dataset: Dataset) -> dict[str, Result]:
        """Populate ``self.results`` from cached runs on disk (no execution)."""
        resolved = self._resolve_step_cache(dataset)
        self.results = {}
        loaded: list[str] = []
        missing: list[tuple[str, str]] = []

        for info in resolved:
            step = info["step"]
            if isinstance(step, CallbackStep):
                continue

            if info["cached"]:
                self.results[step.name] = Result(
                    feature=info["storage_name"],
                    run_id=info["expected_run_id"],
                )
                loaded.append(step.name)
            else:
                reason = (
                    "stale (upstream changed)"
                    if info["stale"]
                    else "not cached"
                )
                missing.append((step.name, reason))

        total = len(loaded) + len(missing)
        print(f"Pipeline.load: {len(loaded)}/{total} steps loaded")
        if loaded:
            print(f"  Loaded: {', '.join(loaded)}")
        if missing:
            print(
                f"  Missing: {', '.join(f'{n} ({r})' for n, r in missing)}"
            )
        return self.results

    def run(
        self,
        dataset: Dataset,
        force_from: str | None = None,
    ) -> dict[str, Result]:
        """Execute the pipeline, skipping steps with cached results.

        Parameters
        ----------
        force_from : str, optional
            Step name from which to force overwrite (and all downstream).
        """
        resolved = self._resolve_step_cache(dataset)
        cached_map: dict[str, dict] = {
            info["step"].name: info
            for info in resolved
            if not isinstance(info["step"], CallbackStep)
        }

        # Open registry for this dataset so run_feature() can write to it
        try:
            features_root = dataset.get_root("features")
            registry = open_registry(features_root, migrate_csv=True)
        except Exception:
            registry = None

        if force_from:
            known = {s.name for s in self.steps}
            if force_from not in known:
                msg = f"force_from={force_from!r} is not a known step. Available: {sorted(known)}"
                raise ValueError(msg)

        self.results = {}
        force_set: set[str] = self._downstream_of(force_from) if force_from else set()

        try:
            for step in self.steps:
                if isinstance(step, CallbackStep):
                    print(f"  [{step.name}] running callback...")
                    step.fn(dataset, self.results)
                    continue

                # Check if cached
                info = cached_map.get(step.name, {})
                if step.name not in force_set and info.get("cached"):
                    run_id = info["expected_run_id"]
                    self.results[step.name] = Result(
                        feature=info["storage_name"],
                        run_id=run_id,
                    )
                    print(
                        f"  [{step.name}] {step.feature_cls.__name__}"
                        f" -> {run_id} (cached)"
                    )
                    continue

                # Not cached — execute
                if step.input_names:
                    input_items = tuple(
                        self.results[n] for n in step.input_names
                    )
                else:
                    input_items = ("tracks",)

                feature = step.feature_cls(
                    inputs=Inputs(input_items), params=step.params
                )
                kwargs = {**self.default_run_kwargs, **step.run_kwargs}
                if step.name in force_set:
                    kwargs["overwrite"] = True
                if registry is not None:
                    kwargs["registry"] = registry

                print(
                    f"  [{step.name}] {step.feature_cls.__name__} ...",
                    end=" ",
                )
                result = dataset.run_feature(feature, **kwargs)
                self.results[step.name] = result
                print(f"-> {result.run_id}")

        finally:
            if registry is not None:
                registry.close()

        return self.results

    def clean(
        self,
        dataset: Dataset,
        *,
        dry_run: bool = True,
    ) -> pd.DataFrame:
        """Remove orphaned run directories that no longer match the pipeline.

        For each FeatureStep, compares the expected run_id (from current
        params/inputs) against all run_ids found on disk.  Runs that don't
        match are deleted.

        Parameters
        ----------
        dry_run : bool
            If True (default), only report what would be deleted.
            If False, actually delete directories and clean index files.

        Returns
        -------
        pd.DataFrame
            Summary with columns ``[step, feature, run_id, status, n_files, size_mb, modified]``.
        """
        import shutil
        from datetime import datetime, timezone

        resolved = self._resolve_step_cache(dataset)
        rows: list[dict] = []

        for info in resolved:
            step = info["step"]
            if isinstance(step, CallbackStep):
                continue

            storage = info.get("storage_name")
            expected_rid = info.get("expected_run_id")
            if not storage:
                continue

            # List all run_ids on disk for this feature
            try:
                runs_df = list_feature_runs(dataset, storage)
            except (FileNotFoundError, ValueError):
                continue

            all_rids = runs_df["run_id"].unique().tolist() if not runs_df.empty else []

            for rid in all_rids:
                run_dir = feature_run_root(dataset, storage, rid)
                if run_dir.exists():
                    files = list(run_dir.glob("*"))
                    n_files = len(files)
                    stats = [f.stat() for f in files if f.is_file()]
                    size_bytes = sum(s.st_size for s in stats)
                    newest_mtime = max((s.st_mtime for s in stats), default=0)
                else:
                    n_files = 0
                    size_bytes = 0
                    newest_mtime = 0

                is_current = rid == expected_rid
                if is_current:
                    status = "current"
                elif dry_run:
                    status = "would remove"
                else:
                    # Delete the directory
                    if run_dir.exists():
                        shutil.rmtree(run_dir)
                    status = "removed"

                rows.append({
                    "step": step.name,
                    "feature": storage,
                    "run_id": rid,
                    "status": status,
                    "n_files": n_files,
                    "size_mb": round(size_bytes / 1_048_576, 1),
                    "modified": (
                        datetime.fromtimestamp(newest_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                        if newest_mtime > 0
                        else None
                    ),
                })

            # Clean index.csv if not dry_run
            if not dry_run:
                idx_path = feature_index_path(dataset, storage)
                if idx_path.exists():
                    idx_df = pd.read_csv(idx_path, keep_default_na=False)
                    before = len(idx_df)
                    idx_df = idx_df[
                        (idx_df["run_id"] == expected_rid)
                        | ~idx_df["run_id"].isin(all_rids)
                    ]
                    if len(idx_df) < before:
                        idx_df.to_csv(idx_path, index=False)

        summary = pd.DataFrame(rows)
        if summary.empty:
            print("Pipeline.clean: no feature runs found on disk.")
            return summary

        orphaned = summary[summary["status"] != "current"]
        if orphaned.empty:
            print("Pipeline.clean: no orphaned runs found — everything is current.")
        else:
            action = "Would remove" if dry_run else "Removed"
            total_mb = orphaned["size_mb"].sum()
            n_runs = len(orphaned)
            print(
                f"Pipeline.clean: {action} {n_runs} orphaned run(s) "
                f"({total_mb:.1f} MB)."
            )
            if dry_run:
                print("  Pass dry_run=False to delete.")

        return summary

    # -- visualization (delegates to mosaic.core.pipeline.viz) -------------

    def show(self, **kwargs):
        """Render the pipeline as a slide-friendly diagram.

        See :func:`mosaic.core.pipeline.viz.show_pipeline_diagram` for
        keyword arguments (``highlight``, ``stop_at``, ``show_feature_class``,
        ``show_inputs``, ``save_path``, etc.).
        """
        from .viz import show_pipeline_diagram
        return show_pipeline_diagram(self, **kwargs)

    def show_text(self, **kwargs):
        """Print the pipeline as an ASCII tree.

        See :func:`mosaic.core.pipeline.viz.show_pipeline_tree` for
        keyword arguments (``highlight``, ``stop_at``, ``show_extras``,
        ``show_feature_class``, ``return_string``, etc.).
        """
        from .viz import show_pipeline_tree
        return show_pipeline_tree(self, **kwargs)

    def dag(self) -> dict[str, list[str]]:
        """Return adjacency dict: ``{step_name: [upstream_names]}``."""
        adj: dict[str, list[str]] = {}
        for step in self.steps:
            if isinstance(step, FeatureStep):
                adj[step.name] = list(step.input_names)
            else:
                adj[step.name] = list(step.depends_on)
        return adj

    def _downstream_of(self, step_name: str) -> set[str]:
        """Return all step names transitively downstream of *step_name* (inclusive)."""
        adj = self.dag()
        # Invert: {parent -> [children]}
        children: dict[str, list[str]] = {name: [] for name in adj}
        for child, parents in adj.items():
            for parent in parents:
                if parent in children:
                    children[parent].append(child)
        # BFS
        visited: set[str] = set()
        queue = [step_name]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(children.get(node, []))
        return visited
