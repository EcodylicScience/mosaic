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

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pandas as pd

from mosaic.core.helpers import make_entry_key, resolve_frame_range

from ._utils import Scope, derive_storage_name
from .index import (
    drifted_entries,
    feature_index,
    feature_index_path,
    feature_run_root,
    list_feature_runs,
)
from .manifest import build_manifest
from .resolve import resolve_references
from .run import compute_run_id
from .tracks_index import read_tracks_index, select_variant_rows
from .types import Feature, Inputs, Result, TrackLike

if TYPE_CHECKING:
    from pathlib import Path

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


def build_step_feature(
    step: FeatureStep, input_items: tuple[TrackLike, ...]
) -> Feature:
    """Instantiate *step*'s feature over *input_items*, on private params.

    The overrides are deep-copied because pydantic accepts an already-built
    model instance by reference, so a ``FeatureStep`` whose params dict holds a
    ``ParquetArtifact`` hands the *same object* to every feature built from it.
    Since resolution pins references in place (item 1.1), sharing would let a
    ``status()`` call freeze an upstream choice into a ``run()`` that happens
    much later -- and let a second ``run()`` reuse the first one's pin instead
    of re-resolving. A step describes what to build; it must not accumulate
    state from having been previewed.
    """
    return step.feature_cls(inputs=Inputs(input_items), params=deepcopy(step.params))


def _newest_mtime(run_root: "Path") -> str | None:
    """Return the newest parquet mtime in a run directory as a short timestamp."""
    from datetime import datetime, timezone

    try:
        parquets = list(run_root.glob("*.parquet"))
        if not parquets:
            return None
        newest = max(f.stat().st_mtime for f in parquets)
        return datetime.fromtimestamp(newest, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
    except OSError:
        return None


def storage_name(feature: object) -> str:
    """Compute on-disk storage name, mirroring ``run_feature`` logic.

    If the feature reads from upstream features, the directory includes
    a ``__from__`` suffix (e.g. ``speed__from__tracks``).
    """
    return derive_storage_name(feature.name, feature.inputs.storage_suffix())  # type: ignore[union-attr]


def _read_track_universe(
    ds: "Dataset", tracks_run_id: str | None = None
) -> frozenset[tuple[str, str]]:
    """All ``(group, sequence)`` pairs the dataset can process, from tracks.

    This is the *intended* sequence universe — the pipeline's target scope
    before any per-step ``groups``/``sequences``/``entries`` narrowing — read
    once and reused for every step's cache check. Mirrors ``_resolve_tracks``
    (``manifest.py``): only rows whose track file actually exists are kept.
    Returns an empty set if the tracks index is absent (e.g. a fresh dataset),
    which makes the cache check fall back to the legacy "any parquet" glob.

    ``tracks_run_id`` must be the *same* selector the steps resolve with, and it
    is why this takes one at all. A step pinned to a variant covering part of the
    index would otherwise be measured for completeness against the whole index,
    read as permanently incomplete, and mark every downstream step stale on every
    invocation.

    Reads through the shared typed reader and variant selector rather than its
    own ``read_csv``, so "which rows count" has one answer across the resolver,
    ``load_tracks`` and this.
    """
    df = select_variant_rows(read_tracks_index(ds), tracks_run_id)
    out: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        abs_path = str(cast(object, row["abs_path"]))
        if ds.resolve_path(abs_path).exists():
            out.add(
                (str(cast(object, row["group"])), str(cast(object, row["sequence"])))
            )
    return frozenset(out)


def _as_set(x: object) -> set[str] | None:
    """Coerce a ``groups``/``sequences`` run_kwarg to ``set[str] | None``."""
    return {str(v) for v in cast(Iterable[object], x)} if x else None


def _as_entry_set(x: object) -> set[tuple[str, str]] | None:
    """Coerce an ``entries`` run_kwarg to ``set[(group, sequence)] | None``."""
    if not x:
        return None
    return {(str(g), str(s)) for g, s in cast(Iterable[tuple[object, object]], x)}


def _as_run_id(x: object) -> str | None:
    """Coerce a ``tracks_run_id`` run_kwarg to ``str | None``.

    Unlike ``_as_set``/``_as_entry_set`` an empty value is *not* "no
    restriction": ``""`` is a legal selector naming the unlabelled tables written
    before tracks carried an identity. Only an absent key means "unrestricted".
    """
    return None if x is None else str(x)


def _narrow_target(
    universe: frozenset[tuple[str, str]],
    groups: object = None,
    sequences: object = None,
    entries: object = None,
) -> set[tuple[str, str]]:
    """Narrow the track universe by a step's scope restriction.

    Applies the same intersecting ``groups`` ∧ ``sequences`` ∧ ``entries``
    filter as ``build_manifest``/``_resolve_tracks`` (``manifest.py:98-106``),
    so the target matches what ``run_feature`` would actually process. Values
    may arrive as lists or sets from ``run_kwargs``; ``None``/empty means "no
    restriction on this axis".
    """
    g = _as_set(groups)
    s = _as_set(sequences)
    e = _as_entry_set(entries)
    result: set[tuple[str, str]] = set()
    for grp, seq in universe:
        if g is not None and grp not in g:
            continue
        if s is not None and seq not in s:
            continue
        if e is not None and (grp, seq) not in e:
            continue
        result.add((grp, seq))
    return result


def _run_is_complete(run_root: "Path", target: set[tuple[str, str]]) -> bool:
    """Query-relative completeness test for a feature run.

    A run counts as cached only when *every* ``(group, sequence)`` in ``target``
    has its output parquet on disk — not merely when the run dir is non-empty.
    This is what makes a partial/interrupted run (e.g. 30 of 90 sequences) read
    as *not* cached, so ``pipe.run`` resumes it. Checking files directly (rather
    than the ``index.csv`` ``finished_at`` flag) is deliberate: ``finished_at``
    is *scope-relative* to whatever manifest an invocation saw — a downstream
    step whose partial upstream truncated its manifest can be honestly
    "finished" at 30/90 — and a file check also catches outputs deleted after
    the fact.

    Falls back to the legacy "any parquet" glob for the empty-input/global
    marker (a single ``__global__`` artifact, see ``run.py`` global marker) and
    when no target is known.

    Known limitation: a feature that *legitimately* produces fewer outputs than
    inputs — when an input filter empties a sequence and ``iter_manifest`` drops
    it (``manifest.py``/``loading.py``) — will read as not-cached and recompute
    each run. Distinguishing "not yet processed" from "processed and empty"
    needs a processed-entry ledger and is deferred.
    """
    if not run_root.exists():
        return False
    present = {
        p.stem for p in run_root.glob("*.parquet")
    }  # stem == make_entry_key(g, s)
    if "__global__" in present:
        return True
    if not target:
        return bool(present)
    return all(make_entry_key(g, s) in present for g, s in target)


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
        refs = step.input_names if isinstance(step, FeatureStep) else step.depends_on
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
        """Compute expected run_ids and check cache state for all steps.

        **Known residual: a cold ``scope_dependent`` step still predicts an
        identifier it will not execute.** Its scope comes from
        ``build_manifest`` over the upstream's index, and on a cold dataset that
        index does not exist, so ``_scope_entries`` predicts empty while
        execution hashes the real entries. Item 1.1 does not reach this: the
        divergence is in the *scope* term, not ``_inputs``. Closing it means
        predicting the entry set of a run that has not happened -- propagating
        each step's ``target`` down the DAG as a predicted scope -- which is a
        new prediction mechanism with its own drift surface and belongs with the
        chain-runner work (item 9.6).

        It cannot cause a wrong execution. An uncached upstream marks this step
        stale, ``cached`` is forced False, and ``run()`` takes the execute
        branch, where inputs come from real ``Result``s. The predicted
        identifier gates a skip only when the upstream *is* cached, and then the
        index it was resolved from exists.
        """
        mock_results: dict[str, Result] = {}
        stale_steps: set[str] = set()
        resolved: list[dict] = []

        # The intended sequence universe, per tracks selector. Each step's
        # target scope is this universe narrowed by that step's restriction; a
        # step is "cached" only when every target sequence is present on disk.
        #
        # Cached per distinct selector rather than read once per pipeline: a step
        # pinned to a variant covering part of the index has to be measured
        # against that variant, or it reads as permanently incomplete and marks
        # every step below it stale on every invocation. Steps almost always
        # share one selector, so this is still one read in practice.
        universes: dict[str | None, frozenset[tuple[str, str]]] = {}

        def track_universe_for(pin: str | None) -> frozenset[tuple[str, str]]:
            if pin not in universes:
                universes[pin] = _read_track_universe(dataset, pin)
            return universes[pin]

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
                resolved.append(
                    {
                        "step": step,
                        "storage_name": None,
                        "expected_run_id": None,
                        "cached": None,
                        "stale": step_is_stale,
                        "mock_result": None,
                    }
                )
                continue

            # Only *construction* is guarded. A mis-specified step, or one whose
            # upstream could not be constructed, is reported and skipped so the
            # rest of the preview stays useful. Everything after this point --
            # scope resolution and identity -- is deliberately unguarded: a
            # failure there must raise rather than silently yield a value that
            # is a valid identifier under some other set of assumptions.
            try:
                # Build inputs from upstream results. A KeyError here means an
                # upstream step failed to construct, so this one cannot be
                # resolved either -- the correct cascade.
                if step.input_names:
                    input_items = tuple(mock_results[n] for n in step.input_names)
                else:
                    input_items = ("tracks",)

                feature = build_step_feature(step, input_items)
                feat_storage_name = storage_name(feature)
            except Exception as e:
                # Deliberately no mock_results entry. The previous code inserted
                # Result(feature=step.feature_cls.name) here -- the *short* name
                # rather than the storage name -- which fed the next step's
                # Inputs and silently moved every downstream predicted
                # identifier. Leaving it absent makes dependents fail to
                # construct too, which is honest.
                resolved.append(
                    {
                        "step": step,
                        "storage_name": step.feature_cls.name,
                        "expected_run_id": None,
                        "cached": False,
                        "stale": step_is_stale,
                        "mock_result": None,
                        "feature_short": step.feature_cls.name,
                        "error": str(e)[:60],
                    }
                )
                for child_name in children_of.get(step.name, []):
                    stale_steps.add(child_name)
                continue

            # Resolve frame range from merged kwargs
            kwargs = {**self.default_run_kwargs, **step.run_kwargs}
            frame_start, frame_end = resolve_frame_range(
                dataset.meta_float("fps_default"),
                kwargs.get("filter_start_frame"),
                kwargs.get("filter_end_frame"),
                kwargs.get("filter_start_time"),
                kwargs.get("filter_end_time"),
            )

            # Target scope for this step: the intended sequence universe
            # narrowed by any groups/sequences/entries restriction. Used
            # both for the completeness check and (for scope_dependent
            # features) the run_id hash.
            target = _narrow_target(
                track_universe_for(_as_run_id(kwargs.get("tracks_run_id"))),
                kwargs.get("groups"),
                kwargs.get("sequences"),
                kwargs.get("entries"),
            )

            # Identity comes from compute_run_id -- the same function
            # run_feature calls, over a Scope resolved the same way (P2e).
            # This prediction is load-bearing for *execution*, not display:
            # when the predicted directory reads complete, run() skips
            # run_feature entirely and pins the predicted identifier into
            # the next step's inputs. A second implementation that drifted
            # would report "cached" over stale results.
            #
            # The build_manifest call is not optional and is not narrowed to
            # scope_dependent features. compute_run_id takes a *resolved*
            # Scope and cannot resolve one itself, and deciding here which
            # features need a scope would put the rule back in two places.
            # It is cheap to be faithful: compute_run_id ignores the scope
            # for a scope-free feature, and build_manifest returns an empty
            # scope for a not-yet-computed upstream rather than raising.
            #
            # Inputs are pinned here for the same reason run_feature pins them:
            # a params-level reference (a global fitter's `templates`) that
            # identity saw as None would predict one identifier and execute
            # another. Chain inputs arrive already pinned from mock_results
            # below, so this pass covers the params half. `on_missing_run` is
            # "empty" because a prediction legitimately names runs that do not
            # exist yet -- after a params change the upstream index holds only
            # the previous run, and reading that as a failure would take
            # `status()` down on an ordinary dataset.
            _ = resolve_references(dataset, feature)
            if feature.inputs.is_empty:
                scope = Scope()
            else:
                _, scope = build_manifest(
                    dataset,
                    feature.inputs,
                    _as_set(kwargs.get("groups")),
                    _as_set(kwargs.get("sequences")),
                    _as_entry_set(kwargs.get("entries")),
                    tracks_run_id=_as_run_id(kwargs.get("tracks_run_id")),
                    on_missing_run="empty",
                )
            expected_run_id, _ = compute_run_id(feature, frame_start, frame_end, scope)

            # Check cache on disk: cached only when the run is *complete*
            # for this step's target scope, not merely non-empty.
            run_root = feature_run_root(dataset, feat_storage_name, expected_run_id)
            cached = _run_is_complete(run_root, target)
            present: set[str] = (
                {p.stem for p in run_root.glob("*.parquet")}
                if run_root.exists()
                else set()
            )
            target_keys = {make_entry_key(g, s) for g, s in target}
            n_present = len(present & target_keys) if target else len(present)
            n_target = len(target)

            # Staleness: if this step was marked stale by an upstream,
            # its cache cannot be trusted
            if step_is_stale:
                cached = False

            # Pinned unconditionally, cached or not. Passing None for an
            # uncached step made the *downstream* step hash a different inputs
            # payload than the one execution would build -- `Pipeline.run`
            # always feeds a concrete `Result` from `run_feature` -- so a cold
            # chain predicted identifiers it then did not produce. The
            # prediction is deterministic, so pinning it is exactly what
            # execution will do.
            result = Result(feature=feat_storage_name, run_id=expected_run_id)
            mock_results[step.name] = result

            resolved.append(
                {
                    "step": step,
                    "storage_name": feat_storage_name,
                    "expected_run_id": expected_run_id,
                    "cached": cached,
                    "stale": step_is_stale,
                    "mock_result": result,
                    "feature_short": feature.name,
                    "n_present": n_present,
                    "n_target": n_target,
                }
            )

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
                rows.append(
                    {
                        "step": step.name,
                        "feature": "(callback)",
                        "run_id": "-",
                        "n_seq": "-",
                        "runs": "-",
                        "cached": "-" if not info["stale"] else "stale",
                        "drift": "-",
                        "modified": "-",
                    }
                )
                continue

            try:
                runs_df = list_feature_runs(dataset, info["storage_name"])
                n_runs = len(runs_df["run_id"].unique())
            except (FileNotFoundError, ValueError):
                n_runs = 0

            # Show how many of the target sequences are present (e.g. "30/90"),
            # so a partial run is visible instead of masked as 0.
            n_present = cast(int, info.get("n_present", 0))
            n_target = cast(int, info.get("n_target", 0))
            n_seq: object = f"{n_present}/{n_target}" if n_target else n_present

            # cached (complete) / partial (some files, not complete) / stale
            # (upstream invalidated) / False (nothing on disk).
            cached_display = info["cached"]
            if not info["cached"] and info["expected_run_id"]:
                run_root = feature_run_root(
                    dataset, info["storage_name"], info["expected_run_id"]
                )
                if run_root.exists() and any(run_root.glob("*.parquet")):
                    cached_display = "stale" if info["stale"] else "partial"

            # Modification timestamp (show even for stale runs if files exist)
            modified = None
            if info["expected_run_id"]:
                rr = feature_run_root(
                    dataset, info["storage_name"], info["expected_run_id"]
                )
                if rr.exists():
                    modified = _newest_mtime(rr)

            # Item 5.2's chain-runner half, as a sibling column rather than a
            # fifth value in `cached`. `cached` is a *cache* verdict -- is the
            # output on disk and complete -- and drift is a *source* verdict.
            # Folding them together would make `cached == "drift"` mean "I no
            # longer know whether it is cached", which is not what was measured:
            # a drifted run is complete and loadable, and superseded is not
            # invalid (rule P4). Deciding what to do about it is item 6.2's.
            drift = (
                drifted_entries(dataset, info["storage_name"], expected)
                if (expected := str(info["expected_run_id"]))
                else ()
            )
            row: dict[str, object] = {
                "step": step.name,
                "feature": info.get("feature_short", info["storage_name"]),
                "run_id": (info["expected_run_id"] if info["cached"] else "\u2014"),
                "n_seq": n_seq,
                "runs": n_runs,
                "cached": cached_display,
                "drift": len(drift) or "",
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
        drifted: list[tuple[str, int]] = []

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
                # Loaded anyway, and reported. A drifted run was correctly
                # derived from a state its run_id still describes -- it is old,
                # not wrong (rule P4's "superseded is not invalid"). Refusing
                # here would destroy the ability to compare across source
                # revisions, which is a thing users want; deciding to delete is
                # item 6.2's gesture, behind an explicit force.
                moved = drifted_entries(
                    dataset, info["storage_name"], str(info["expected_run_id"])
                )
                if moved:
                    drifted.append((step.name, len(moved)))
            else:
                reason = "stale (upstream changed)" if info["stale"] else "not cached"
                missing.append((step.name, reason))

        total = len(loaded) + len(missing)
        print(f"Pipeline.load: {len(loaded)}/{total} steps loaded")
        if loaded:
            print(f"  Loaded: {', '.join(loaded)}")
        if missing:
            print(f"  Missing: {', '.join(f'{n} ({r})' for n, r in missing)}")
        if drifted:
            moved_by_step = ", ".join(
                f"{name} ({count} sequence(s))" for name, count in drifted
            )
            print(f"  Drift: source has moved under {moved_by_step}")
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

        if force_from:
            known = {s.name for s in self.steps}
            if force_from not in known:
                msg = f"force_from={force_from!r} is not a known step. Available: {sorted(known)}"
                raise ValueError(msg)

        self.results = {}
        force_set: set[str] = self._downstream_of(force_from) if force_from else set()

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
                    f"  [{step.name}] {step.feature_cls.__name__} -> {run_id} (cached)"
                )
                continue

            # Not cached — execute. Each step's run_feature() opens its own
            # append-only JSONL run-log (track=True); no shared handle to manage.
            if step.input_names:
                input_items = tuple(self.results[n] for n in step.input_names)
            else:
                input_items = ("tracks",)

            feature = build_step_feature(step, input_items)
            kwargs = {**self.default_run_kwargs, **step.run_kwargs}
            if step.name in force_set:
                kwargs["overwrite"] = True

            print(
                f"  [{step.name}] {step.feature_cls.__name__} ...",
                end=" ",
            )
            result = dataset.run_feature(feature, **kwargs)
            self.results[step.name] = result
            print(f"-> {result.run_id}")

        return self.results

    def _build_clean_keep_sets(
        self, resolved: list[dict]
    ) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
        """Group resolved steps by on-disk storage path.

        Multiple FeatureSteps can share one storage path (same feature class
        + same input chain, different params). The returned ``keep_by_storage``
        is the union of every step's expected_run_id per storage — clean()
        must preserve all of them. Without this grouping, clean() would
        delete a sibling step's current run.

        Factored into its own method so safeguard tests can patch it to
        simulate a broken keep-set (the pre-fix bug).
        """
        keep_by_storage: dict[str, set[str]] = {}
        steps_by_storage: dict[str, list[str]] = {}
        for info in resolved:
            step = info["step"]
            if isinstance(step, CallbackStep):
                continue
            storage = info.get("storage_name")
            if not storage:
                continue
            steps_by_storage.setdefault(storage, []).append(step.name)
            rid = info.get("expected_run_id")
            if rid:
                keep_by_storage.setdefault(storage, set()).add(rid)
        return keep_by_storage, steps_by_storage

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

        keep_by_storage, steps_by_storage = self._build_clean_keep_sets(resolved)

        # Ground-truth set of (storage, rid) pairs that any step in the
        # pipeline considers current. Built directly from `resolved` — NOT
        # derived from keep_by_storage — so safeguard #1 below can catch a
        # broken grouping (e.g. stale-kernel running pre-fix code).
        global_keepers: set[tuple[str, str]] = {
            (info["storage_name"], info["expected_run_id"])
            for info in resolved
            if not isinstance(info["step"], CallbackStep)
            and info.get("storage_name")
            and info.get("expected_run_id")
        }
        step_to_rid: dict[str, str | None] = {
            info["step"].name: info.get("expected_run_id")
            for info in resolved
            if not isinstance(info["step"], CallbackStep)
        }

        # Safeguard #2: surface shared-storage groupings so the union-keeping
        # logic is observable. A silently-bypassed clean() would print no
        # such line even when steps share storage — an easy visual check.
        for storage, names in steps_by_storage.items():
            if len(names) > 1:
                descs = ", ".join(
                    f"{n} (rid={step_to_rid.get(n) or 'none'})" for n in names
                )
                print(
                    f"Pipeline.clean: storage {storage!r} shared by "
                    f"{len(names)} steps: {descs} — keeping all."
                )

        # Safeguard #3 prep: snapshot which keepers exist BEFORE deletion.
        # After the per-storage loop completes, any keeper that vanished
        # from this set is a bug. Keepers that never existed on disk
        # (steps not yet computed) are excluded so they don't false-alarm.
        keepers_existed_before: set[tuple[str, str]] = {
            (s, r)
            for (s, r) in global_keepers
            if feature_run_root(dataset, s, r).exists()
        }

        for storage, step_names in steps_by_storage.items():
            # List all run_ids on disk for this feature
            try:
                runs_df = list_feature_runs(dataset, storage)
            except (FileNotFoundError, ValueError):
                continue

            all_rids = runs_df["run_id"].unique().tolist() if not runs_df.empty else []
            keep_set = keep_by_storage.get(storage, set())
            step_label = ", ".join(step_names)

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

                is_current = rid in keep_set
                # Safeguard #1: would we mark a pipeline-wide keeper as
                # removable? With the current grouping fix this branch is
                # unreachable; it fires only when the keep-set build was
                # bypassed (stale kernel, non-editable install, refactor
                # regression). Hard-raise even in dry_run — the row is
                # already wrong.
                if not is_current and (storage, rid) in global_keepers:
                    owners = [n for n, r in step_to_rid.items() if r == rid]
                    raise RuntimeError(
                        f"Pipeline.clean: refusing to mark "
                        f"(storage={storage!r}, run_id={rid!r}) as removable "
                        f"— step(s) {owners!r} in this pipeline expect it as "
                        f"current. The storage-grouping logic in clean() did "
                        f"not run as expected (possibly a stale Jupyter kernel "
                        f"or a non-editable mosaic install). Restart your "
                        f"kernel and verify `mosaic.core.pipeline.pipeline."
                        f"__file__` points at the current source tree."
                    )
                if is_current:
                    status = "current"
                elif dry_run:
                    status = "would remove"
                else:
                    # Delete the directory
                    if run_dir.exists():
                        shutil.rmtree(run_dir)
                    status = "removed"

                rows.append(
                    {
                        "step": step_label,
                        "feature": storage,
                        "run_id": rid,
                        "status": status,
                        "n_files": n_files,
                        "size_mb": round(size_bytes / 1_048_576, 1),
                        "modified": (
                            datetime.fromtimestamp(
                                newest_mtime, tz=timezone.utc
                            ).strftime("%Y-%m-%d %H:%M")
                            if newest_mtime > 0
                            else None
                        ),
                    }
                )

            # Drop the rows that named the directories just removed. A run on
            # disk but not in the keep set is exactly what was deleted above; a
            # run the index knows and disk does not is left alone, which is what
            # the old expression's second clause said and this says by omission.
            #
            # Through IndexCSV rather than a bare read-modify-write: that took no
            # lock, so a worker appending between its read and its write lost the
            # row, and it re-parsed every surviving cell, which rewrites a
            # `version` of "0.10" as 0.1 and a params_hash of "0123456789" as
            # 123456789 -- an index contradicting the identifiers it holds.
            if not dry_run:
                _ = feature_index(feature_index_path(dataset, storage)).drop_runs(
                    set(all_rids) - keep_set
                )

        # Safeguard #3: last line of defense. If any keeper directory that
        # existed before the loop is now gone, something destroyed data it
        # shouldn't have. Only meaningful on real deletion runs.
        if not dry_run:
            destroyed = [
                (s, r)
                for (s, r) in keepers_existed_before
                if not feature_run_root(dataset, s, r).exists()
            ]
            if destroyed:
                details = "\n".join(
                    f"  - storage={s!r} run_id={r!r}" for s, r in destroyed
                )
                raise RuntimeError(
                    f"Pipeline.clean: destroyed {len(destroyed)} run "
                    f"directory(ies) that should have been kept:\n{details}"
                )

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

    def _upstream_of(self, step_name: str) -> set[str]:
        """Return all step names transitively upstream of *step_name* (inclusive)."""
        adj = self.dag()
        visited: set[str] = set()
        queue = [step_name]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj.get(node, []))
        return visited
