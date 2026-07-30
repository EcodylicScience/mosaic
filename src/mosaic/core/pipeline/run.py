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
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from mosaic.core.helpers import (
    filter_time_range,
    load_labels_for_feature_frames,
    make_entry_key,
    resolve_frame_range,
)

from ._utils import (
    FeatureMeta,
    Scope,
    atomic_write,
    derive_storage_name,
    hash_params,
    json_ready,
)
from .identity_scheme import (
    FEATURE_IDENTITY_SCHEME,
    write_identity_scheme,
)
from .index import (
    FeatureIndexRow,
    feature_index,
    feature_index_path,
    feature_run_root,
    missing_outputs_error,
    recorded_consumption,
)
from .loading import build_nn_lookup, nn_pair_mask, resolve_sequence_identity
from .manifest import FilterFactory, Manifest, build_manifest, iter_manifest
from .fit_scope import write_fit_scope
from .labels_index import read_labels_index, select_label_variant_rows
from .resolve import resolution_payload, resolve_references
from .sequence_index import encode_entry_composition
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
from .job import CancelToken, Cancelled, JobContext, job_context
from .progress import ProgressCallback
from .writers import (
    FeatureOutput,
    default_check_output,
    output_n_rows,
    trim_feature_output,
    write_output,
)

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


def _build_result_lookup(
    ds: Dataset, feature_name: str, run_id: str | None
) -> DependencyLookup:
    """Build a (group, sequence) -> Path lookup for an upstream feature's run.

    Reads the upstream feature's index and resolves each entry's parquet path.
    Shared by the params-field and ``feature.inputs`` dependency-resolution
    paths in :func:`_resolve_dependencies`.

    The ``run_id is None`` fallback below is unreachable from ``run_feature``:
    :func:`~mosaic.core.pipeline.resolve.resolve_references` pins every
    reference before the identifier is computed, so by the time this runs the
    caller's reference carries a concrete run. It is kept because this function
    is also reached from paths that mint no identity, and because resolving here
    instead of raising would be the defect item 1.1 fixed -- silently reading a
    run the identifier does not name.
    """
    dep_index = feature_index(feature_index_path(ds, feature_name))
    if run_id is None:
        run_id = dep_index.latest_run_id()
    index_df = dep_index.read(
        run_id=run_id, filter_ext=".parquet", validate_paths=False
    )
    lookup: DependencyLookup = {}
    missing: list[Path] = []
    for _, row in index_df.iterrows():
        group = str(row.get("group", ""))
        sequence = str(row.get("sequence", ""))
        abs_path = str(row.get("abs_path", ""))
        if not abs_path:
            continue
        # Resolve via the dataset (handles relative/relocated paths); check
        # existence here rather than trusting the raw stored string.
        resolved = ds.resolve_path(abs_path)
        if not resolved.exists():
            missing.append(resolved)
            continue
        lookup[(group, sequence)] = resolved
    if missing and not lookup:
        # Every resolved output is missing: dataset moved / non-portable paths.
        raise missing_outputs_error(feature_name, run_id, missing, len(index_df))
    if missing:
        print(
            f"[deps] feature {feature_name!r} run {run_id!r}: {len(missing)} of "
            f"{len(index_df)} output(s) missing; skipping.",
            file=sys.stderr,
        )
    return lookup


def _resolve_dependencies(
    ds: Dataset, feature: Feature, labels_run_id: str | None = None
) -> tuple[dict[str, Path], dict[str, DependencyLookup]]:
    """Resolve upstream dependencies from params fields and feature inputs.

    Returns (artifact_paths, dependency_lookups) where:
    - artifact_paths maps field name to resolved file/directory Path
    - dependency_lookups maps field name (or ``_input_<i>`` for Results
      referenced via ``feature.inputs``) to a (group, sequence) -> Path dict
    """
    from .index import latest_feature_run_root as _latest_feature_run_root

    params = feature.params
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
                    # Pinned by resolve_references before the identifier was
                    # computed, so this fires only for a caller that skipped
                    # that pass. See _build_result_lookup for why it survives.
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
                dependency_lookups[field_name] = _build_result_lookup(
                    ds, feature_name, _run_id
                )

            case LabelsSource(kind=str(kind)):
                if not kind:
                    continue
                lookup = _build_labels_lookup(ds, kind, labels_run_id)
                if not lookup:
                    msg = (
                        f"Labels lookup is empty for kind={kind!r}, "
                        f"required by field '{field_name}'"
                    )
                    raise FileNotFoundError(msg)
                dependency_lookups[field_name] = lookup

            case _:
                pass

    # Also resolve dependencies referenced via feature.inputs (typed Result
    # items), not just params fields. Features like FeralFeature carry their
    # upstream output location here rather than in a params field. Appended
    # after the params loop so params-derived lookups keep priority for
    # consumers that select the first non-empty lookup.
    for idx, result in enumerate(feature.inputs.feature_inputs):
        feature_name = result.feature
        if not feature_name or feature_name == "nearest-neighbor":
            continue
        lookup = _build_result_lookup(ds, feature_name, result.run_id)
        if lookup:  # skip empty so first-non-empty consumers aren't misled
            dependency_lookups[f"_input_{idx}"] = lookup

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
    ds_manifest_path: str | None = None,
) -> pd.DataFrame:
    """Reconstruct a feature in a worker process and run apply.

    If the feature declares ``bind_dataset`` (e.g. InteractionCropPipeline,
    EgocentricCropPipeline, FeralFeature), reconstruct the Dataset from its
    manifest in this worker and bind it. Without this, features that read
    media via ``self._ds.resolve_media(...)`` crash with
    ``'NoneType' object has no attribute 'resolve_media'`` because the
    main-process bind at ``run_feature`` doesn't propagate across process
    workers. Features without ``bind_dataset`` are unaffected.
    """
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    inputs_obj = Inputs.model_validate(inputs_dump)
    feature = cls(inputs_obj, params_dump)
    if ds_manifest_path is not None and hasattr(feature, "bind_dataset"):
        from mosaic.core.dataset import Dataset

        ds_local = Dataset(manifest_path=Path(ds_manifest_path)).load()
        feature.bind_dataset(ds_local)
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


class MissingScopeDeclaration(AttributeError):
    """A feature reached identity computation without declaring ``scope_dependent``.

    Subclasses ``AttributeError`` because that is what the bare attribute read
    below used to raise, so any caller already handling it keeps working -- but
    it names the feature, which a bare ``AttributeError: 'TimelinePlot' object
    has no attribute 'scope_dependent'`` does only by accident of the repr.

    Deliberately *not* solved by a ``ClassVar`` default on the protocol. A
    default would let the next feature ship with no declaration at all and
    silently inherit ``False``, which is the wrong answer for any feature that
    fits from its stream -- exactly the defect item 1.4 exists to fix. The
    declaration is a decision each feature owes, so its absence is an error.
    """


class MissingConsumedRootsDeclaration(AttributeError):
    """A feature reached identity computation without declaring ``consumed_roots``.

    The same argument as :class:`MissingScopeDeclaration`, one field over. A
    default of ``()`` would let the next feature that opens video ship
    undeclared and silently claim to read nothing -- and item 6.2's per-entry
    delete set reads this declaration, so an empty one is a stale crop nothing
    deletes.
    """


def encode_consumed_roots(roots: Iterable[str]) -> str:
    """Encode a feature's declared source roots into one index cell.

    Sorted and deduplicated, so one set of roots has one spelling however the
    feature listed them -- the same reason ``encode_source_roots`` sorts on the
    tracks row, and the same reason identity payloads sort their collections.
    """
    return ",".join(sorted({root for root in roots if root}))


def entry_composition(feature: Feature, scope: Scope, entry: tuple[str, str]) -> str:
    """What this entry was made of, under the roots *feature* declares.

    Recorded on the index row, never hashed for a per-frame feature -- see
    ``FeatureIndexRow``. One root is the whole of today's reality (two features
    declare ``media_raw``, forty declare nothing), so the common answers are a
    bare digest and ``""``; several roots join as ``root=digest`` pairs so the
    cell stays readable and says which value came from where.

    Empty means "nothing recorded", which covers a feature that declares no root
    and a root that has recorded no composition for this entry. Both mean the
    same thing to a reader: draw no conclusion.

    The encoding lives in :func:`encode_entry_composition` rather than here,
    because item 5.1's tracks half writes the same cell on the tracks row and two
    spellings of one answer would be two answers to item 6.2's walk.
    """
    roots = sorted({root for root in feature.consumed_roots if root})
    recorded = {root: scope.composition_of(entry, root) for root in roots}
    return encode_entry_composition(recorded, roots)


CacheDisposition = Literal["serve", "recompute", "undetectable"]
"""What may be done with a cached entry, given what it recorded consuming.

Item 6.2's refusal, and its three answers rather than two. ``serve`` and
``recompute`` are the obvious pair; ``undetectable`` is the case that has to be
named separately or the rule defeats the cache permanently.
"""


def cached_entry_disposition(recorded: str, current: str) -> CacheDisposition:
    """Whether a cached entry may be served, must be recomputed, or cannot be told.

    Fail closed means *recompute* here, not prompt: a library has no one to ask,
    and the cost of being wrong is CPU rather than data. But it has to converge,
    and that is what splits ``unknown`` in two.

    - **Both sides known and equal** -- serve. The ordinary answer.
    - **Both known and different** -- recompute. The source moved under an output
      still on disk.
    - **Exactly one side known** -- recompute. Either the entry predates item 5.1
      and nothing says what it was built from, or the projection that would say
      has gone. Recomputing *resolves* it: the row then records what is true now,
      and the next run serves. One wrong cache miss, by construction -- the
      migration shape M1 established.
    - **Neither side known** -- ``undetectable``. Recomputing would resolve
      nothing, because the row it wrote would record the same empty and the next
      run would recompute again, forever. A dataset whose media was indexed
      before item 4.4 is exactly this state, so failing closed here would mean an
      unbounded recompute on every run of the two features that declare a root.
      Serve it, and say out loud that drift cannot be detected until the
      projection exists.

    The distinction is the honest-empty rule applied to a decision rather than to
    a value: an empty needs a companion saying which kind of empty it is, and
    here the companion is the other side.
    """
    if recorded and current:
        return "serve" if recorded == current else "recompute"
    if recorded or current:
        return "recompute"
    return "undetectable"


def _disposition_reason(recorded: str, current: str) -> str:
    """Why a cached entry was refused, in the words a user can act on."""
    if recorded and current:
        return "its source moved"
    if recorded:
        return "the composition it was built from is no longer recorded"
    return "nothing records what it was built from"


def _scope_term(feature: Feature, scope: Scope) -> list[list[object]]:
    """The ``_scope_entries`` term: a sorted list of ``(group, sequence[, comp])``.

    **The third element is omitted, never empty**, and that is the whole
    mechanism. ``json.dumps(sort_keys=True)`` digests ``["", "a"]`` differently
    from ``["", "a", []]``, so a two-element entry is byte-identical to what
    ``sorted(scope.entries)`` produced before this term existed -- which is what
    keeps every identifier still on a dataset that has recorded no compositions,
    and what makes the golden diff for this change *zero moved lines*. Same
    omit-an-absent-term rule ``_tracks`` and ``tracks_identity``'s ``upstream``
    already state.

    A **list**, never a set of bare digests: two sequences sharing one
    composition are still two entries, and a set would collapse them, so
    cardinality and distinctness would not survive the hash.

    The third element is itself a sorted list of ``[root, digest]`` pairs rather
    than one combined digest, so ``params.json`` and the golden corpus stay
    readable and no second minter is introduced with a second scheme to keep
    honest. Every feature that exists today declares at most one root.

    Mixed entries within one scope are correct, not a wart: a sequence whose
    composition is recorded and one whose is not are genuinely different states,
    and a fit over both must move when the first changes and must not move
    because the second is unknown.
    """
    roots = sorted({root for root in feature.consumed_roots if root})
    term: list[list[object]] = []
    for group, sequence in sorted(scope.entries):
        pairs = [
            [root, digest]
            for root in roots
            if (digest := scope.composition_of((group, sequence), root))
        ]
        term.append([group, sequence, pairs] if pairs else [group, sequence])
    return term


def compute_run_id(
    feature: Feature,
    frame_start: int | None,
    frame_end: int | None,
    scope: Scope,
) -> tuple[str, str]:
    """Compute the content-addressed ``(run_id, params_hash)`` for a run.

    Pure and side-effect-free -- it does *not* execute anything -- so callers
    (e.g. an API dedup check) can learn the ``run_id`` a submission *would*
    produce and consult the feature's ``index.csv`` before spawning any work.

    ``identity_dump()`` drops ``HASH_EXCLUDE``-marked params (throughput knobs
    like ``infer_batch_size``) so retuning them doesn't bust the cache.

    ``_tracks`` names the tracks recipes behind the tables a run reads, and is
    **added only when there are any** -- the same omit-an-absent-term rule that
    lets ``_scope_entries`` exist for scope-dependent features without disturbing
    the rest, since ``json.dumps(sort_keys=True)`` digests an absent key
    differently from an empty one. Rows written before tracks carried an identity
    have an empty ``run_id`` and contribute nothing, so every dataset converted
    before Stage 3 keeps the identifiers it already has.

    It is a term of its own rather than a substitution inside ``_inputs`` because
    ``_inputs`` is also the wire form: ``run_feature`` ships
    ``feature.inputs.model_dump()`` to a process worker that rebuilds it with
    ``Inputs.model_validate``, where anything but the bare ``"tracks"`` literal
    fails validation. ``_labels`` is the same story: a ``GroundTruthLabelsSource``
    resolves to a dataset-wide root under a selector, exactly as ``"tracks"`` does,
    so pinning its resolved value into ``_inputs`` would move a scope-free
    consumer's identifier whenever a sequence was added -- it is a Scope term
    instead.

    Raises:
        MissingScopeDeclaration: if *feature* declares no ``scope_dependent``.
    """
    if not hasattr(feature, "scope_dependent"):
        name = getattr(feature, "name", type(feature).__name__)
        raise MissingScopeDeclaration(
            f"feature {name!r} ({type(feature).__name__}) declares no "
            f"'scope_dependent', so its identity cannot be computed. Declare it: "
            f"True if fit() derives anything from the set of sequences in scope, "
            f"False if each entry is computed from itself alone."
        )
    if not hasattr(feature, "consumed_roots"):
        name = getattr(feature, "name", type(feature).__name__)
        raise MissingConsumedRootsDeclaration(
            f"feature {name!r} ({type(feature).__name__}) declares no "
            f"'consumed_roots', so what it reads outside its inputs is unknown. "
            f"Declare it: the source roots this feature opens directly -- "
            f"('media_raw',) if it reads video, () for a feature that only reads "
            f"the tables its inputs hand it, which is almost all of them."
        )
    hashable: dict[str, object] = {
        "_params": feature.params.identity_dump(),
        "_inputs": feature.inputs.model_dump(),
        "_frame_range": [frame_start, frame_end],
    }
    if feature.scope_dependent:
        hashable["_scope_entries"] = _scope_term(feature, scope)
    if scope.tracks_variants:
        # Sorted here as well as in the resolver. Which recipes a run read is a
        # *set*, and `_ready` preserves list order on purpose, so hashing the
        # tuple as given would make two spellings of one answer two identifiers
        # for any caller that built a Scope by hand.
        hashable["_tracks"] = sorted(scope.tracks_variants)
    if scope.labels_variants:
        # The label analog of `_tracks` (item 9.3): which label recipes produced
        # the labels a run read. Same omit-when-absent rule, so a feature that
        # reads no labels -- almost all of them -- digests exactly as before and
        # the golden corpus moves only for the labels-consuming features.
        hashable["_labels"] = sorted(scope.labels_variants)
    params_hash = hash_params(hashable)
    return f"{feature.version}-{params_hash}", params_hash


def build_run_params_payload(
    feature: Feature,
    frame_start: int | None,
    frame_end: int | None,
    scope: Scope,
    feature_resolutions: list[dict[str, str | None]],
) -> dict[str, object]:
    """The ``params.json`` save payload -- provenance, deliberately not the digest.

    The single site that builds the ``{_params, _inputs, _frame_range, _scope,
    _resolved}`` document, so no caller reconstructs it: ``run_feature`` writes it,
    and ``Dataset.reconcile`` reuses it to restamp a re-addressed run's provenance
    with the upstream ids it now resolves to. Deliberately ``json_ready``, not
    ``identity_dump`` -- it keeps HASH_EXCLUDE fields and is never hashed.

    ``_scope`` is the scope of *this invocation*, not the fit scope (item 5.3):
    written unconditionally, so for a scope-free feature it is "whichever ran
    last"; ``fit_scope.json`` answers "what was this state trained on". The values
    are load-bearing, not garnish -- two runs both marked one scheme can differ
    because one ran before a ``sequences.csv`` existed and one after, honest only
    if the values are written down. Compositions are recorded for *every* root, so
    a drift check and a blast walk can explain a run without re-deriving it.

    ``_resolved`` is which concrete upstream each reference pinned to. The digest
    already covers these (they were pinned before it was computed), so this is the
    readable copy, feeding the reverse-dependency walk. *feature_resolutions* is the
    feature-to-feature and params half (from ``resolution_payload``); the tracks and
    labels variants are appended here from the scope -- they ride in ``_resolved``
    rather than ``_inputs``, which must stay the literal the process worker
    revalidates, and match what the digest's ``_tracks``/``_labels`` terms cover.
    """
    return {
        "_params": json_ready(feature.params),
        "_inputs": feature.inputs.model_dump(),
        "_frame_range": [frame_start, frame_end],
        "_scope": {
            "scope_dependent": feature.scope_dependent,
            "consumed_roots": sorted({r for r in feature.consumed_roots if r}),
            "entries": [list(entry) for entry in sorted(scope.entries)],
            "compositions": {
                make_entry_key(group, sequence): dict(sorted(per_root.items()))
                for (group, sequence), per_root in sorted(scope.compositions.items())
            },
        },
        "_resolved": list(feature_resolutions)
        + [
            {"where": "inputs[tracks]", "feature": "tracks", "run_id": variant}
            for variant in scope.tracks_variants
        ]
        + [
            {"where": "inputs[labels]", "feature": "labels", "run_id": variant}
            for variant in scope.labels_variants
        ],
    }


def run_feature(
    ds: Dataset,
    feature: Feature,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    entries: Iterable[tuple[str, str]] | None = None,
    overwrite: bool = False,
    parallel_workers: int | None = None,
    parallel_mode: str | None = "thread",
    overlap_frames: int = 0,
    filter_start_frame: int | None = None,
    filter_end_frame: int | None = None,
    filter_start_time: float | None = None,
    filter_end_time: float | None = None,
    check_output: bool = False,
    *,
    tracks_run_id: str | None = None,
    labels_run_id: str | None = None,
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: ProgressCallback | None = None,
    cancel_token: CancelToken | None = None,
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
        Scope filter (applies to whichever input source is used). These combine
        as a cross-product filter (any matching group AND any matching sequence).
    entries : optional iterable of (group, sequence)
        Restrict the run to exactly these ``(group, sequence)`` pairs. Unlike
        ``groups``/``sequences`` this selects an arbitrary subset and is
        unambiguous when sequence names repeat across groups -- e.g. running a
        feature over a tag-resolved set of sequences. Intersects with
        ``groups``/``sequences`` when those are also given.
    tracks_run_id : str | None
        Which tracks *variant* the ``"tracks"`` input resolves to. ``None``, the
        default, lets each entry resolve to whichever variant it has -- which is
        every dataset except one holding two genuinely different recipes for the
        same sequence, where resolution refuses rather than guesses and this is
        the way to answer it. ``""`` names the unlabelled tables written before
        tracks carried an identity.
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
    check_output : bool, default False
        When False, a cache hit only requires the output parquet to exist
        (footer metadata is read for n_rows). When True, each candidate cache
        hit is deeply validated before being skipped: the default validator
        fully reads the parquet (pq.read_table); a feature may override
        validation by defining ``check_output(self, meta, run_root) -> bool``
        (e.g. to verify side-car media). If validation fails the entry is
        recomputed. No effect when overwrite=True or when state is not cached.
        The deep read costs roughly an input-load per entry, so leave it off
        for routine runs.
    execution_id : str | None
        Reuse an externally minted ULID attempt id (how a Layer-2 subprocess
        inherits its identity); otherwise a fresh one is generated.
    owner : str
        Free-form attribution recorded on the attempt's run-log.
    track : bool, default True
        Record this attempt (status/progress/heartbeat) into an append-only
        JSONL run-log under ``<dataset_root>/.mosaic/runs/``. Set False to run
        without leaving any attempt record.
    progress_callback : ProgressCallback | None
        Receives per-entry (and, for trainers, per-epoch) progress. Defaults to
        the JSONL run-log keyed by ``execution_id`` when tracking is on.
    cancel_token : CancelToken | None
        Cooperative cancellation, polled between entries. The default token is
        inert.

    Returns
    -------
    Result
        Carries ``feature`` and ``run_id`` (the content id), plus ``execution_id``
        and ``cache_hit`` (excluded from serialization, so they never perturb a
        downstream feature's ``run_id``).
    """
    storage_feature_name = derive_storage_name(
        feature.name, feature.inputs.storage_suffix()
    )
    with job_context(
        ds,
        kind="feature",
        target=storage_feature_name,
        execution_id=execution_id,
        owner=owner,
        track=track,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    ) as ctx:
        return _run_feature_impl(
            ds,
            feature,
            ctx,
            storage_feature_name,
            groups=groups,
            sequences=sequences,
            entries=entries,
            tracks_run_id=tracks_run_id,
            labels_run_id=labels_run_id,
            overwrite=overwrite,
            parallel_workers=parallel_workers,
            parallel_mode=parallel_mode,
            overlap_frames=overlap_frames,
            filter_start_frame=filter_start_frame,
            filter_end_frame=filter_end_frame,
            filter_start_time=filter_start_time,
            filter_end_time=filter_end_time,
            check_output=check_output,
        )


def _run_feature_impl(
    ds: Dataset,
    feature: Feature,
    ctx: JobContext,
    storage_feature_name: str,
    *,
    groups: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
    entries: Iterable[tuple[str, str]] | None = None,
    tracks_run_id: str | None = None,
    labels_run_id: str | None = None,
    overwrite: bool = False,
    parallel_workers: int | None = None,
    parallel_mode: str | None = "thread",
    overlap_frames: int = 0,
    filter_start_frame: int | None = None,
    filter_end_frame: int | None = None,
    filter_start_time: float | None = None,
    filter_end_time: float | None = None,
    check_output: bool = False,
) -> Result:
    """Body of :func:`run_feature`, executed inside a :func:`job_context`.

    ``ctx`` owns the attempt lifecycle (status/progress/cancel via its append-only
    JSONL run-log). What-ran and cache state live in ``index.csv`` + the parquet
    outputs on disk -- the permanent source of truth.
    """
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
    entries_set = (
        {(str(g), str(s)) for g, s in entries} if entries is not None else None
    )

    # An explicitly empty selector is ambiguous, and currently answers two ways
    # within one call: on a tracks input the manifest tests truthiness and
    # yields the *full* scope, while on a feature input IndexCSV.read tests
    # `is not None` and yields the *empty* one. Neither is what a caller passing
    # [] meant. Reject it rather than pick a side. Checked after materializing,
    # so a generator argument is not consumed by the check. Nothing in src/ or
    # tests/ passes an empty collection, so this raises for no existing caller.
    for name, selector in (
        ("groups", groups_set),
        ("sequences", sequences_set),
        ("entries", entries_set),
    ):
        if selector is not None and not selector:
            raise ValueError(
                f"{name}=[] selects nothing, but is read as 'everything' on a "
                f"tracks input and as 'nothing' on a feature input. Pass None "
                f"(or omit it) to mean every sequence."
            )

    # Pin every unpinned upstream *before* anything hashes or reads it. An
    # unpinned reference used to be resolved after the digest, into a local that
    # was discarded, so a run that consumed the latest `extract-templates` and a
    # run that consumed a later one shared one identifier and one directory.
    # Resolution reads the filesystem; compute_run_id below stays pure.
    resolutions = resolve_references(ds, feature)

    # Build manifest
    if feature.inputs.is_empty:
        manifest: Manifest = {}
        scope = Scope()
    else:
        manifest, scope = build_manifest(
            ds,
            feature.inputs,
            groups_set,
            sequences_set,
            entries_set,
            tracks_run_id=tracks_run_id,
        )

    # Labels are a params dependency, not a manifest input, so they are resolved
    # here rather than inside build_manifest -- and *before* compute_run_id, so a
    # run over different label content gets a different identifier. The same
    # labels_run_id is threaded into _resolve_dependencies below, so the variant
    # the identity is built from is the variant the fit actually reads.
    scope.labels_variants = resolve_labels_variants(ds, feature, labels_run_id)

    # Run ID: content hash of params+inputs+frames (+scope). Attempt-level
    # identity (execution_id, progress, cancel) is deliberately NOT part of it.
    run_id, params_hash = compute_run_id(feature, frame_start, frame_end, scope)
    ctx.set_run_id(run_id)
    ctx.set_total(len(manifest))

    # Run root + params.json
    run_root = feature_run_root(ds, storage_feature_name, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    params_path = run_root / "params.json"
    try:
        # Built by the one payload site (build_run_params_payload) so no caller
        # reconstructs the document -- the same single-site rule the identity
        # payload follows. Deliberately provenance, not the hash payload.
        save_payload = build_run_params_payload(
            feature, frame_start, frame_end, scope, resolution_payload(resolutions)
        )
        atomic_write(
            params_path, lambda p: p.write_text(json.dumps(save_payload, indent=2))
        )
    except Exception as exc:
        print(
            f"[feature:{feature.name}] failed to save params.json: {exc}",
            file=sys.stderr,
        )

    # The identity-scheme marker. Written atomically and NOT best-effort: it is
    # what makes a half-migrated dataset detectable, so a silently skipped write
    # is a wrong answer rather than a cosmetic loss. run_id carries the
    # *feature's* version, never the version of the hashing contract, and
    # retrofitting this onto identifiers already on disk needs provenance that
    # does not exist -- so it has to be written from the start.
    #
    # It enters no hash and no path. Folding it into compute_run_id would make
    # the marker itself move every identifier, so the detector would cause the
    # event it exists to detect.
    write_identity_scheme(run_root, FEATURE_IDENTITY_SCHEME)

    # Index CSV setup -- the permanent record of what-ran (run_id, version,
    # params_hash, started_at per entry); complemented by params.json in run_root.
    idx = feature_index(feature_index_path(ds, storage_feature_name))
    idx.ensure()

    # Bind dataset (for features that need media paths, etc.)
    if hasattr(feature, "bind_dataset"):
        feature.bind_dataset(ds)

    # Resolve dependencies
    artifact_paths, dependency_lookups = _resolve_dependencies(
        ds, feature, labels_run_id
    )

    # Resolve pair filter
    pair_filter_spec = _resolve_pair_filter(feature.params)

    # Load state
    state_ready = feature.load_state(run_root, artifact_paths, dependency_lookups)

    # Build filter factory (shared by fit and apply phases)
    filter_factory = _make_filter_factory(
        ds, scope, pair_filter_spec, frame_start, frame_end
    )

    # Wire the Job Contract into the feature: trainers (FERAL/kpms/...) that
    # expose a ``progress_callback``/``cancel_token`` attribute get epoch-level
    # progress and cooperative cancel routed to this attempt.
    if hasattr(feature, "progress_callback"):
        setattr(feature, "progress_callback", ctx.progress)
    if hasattr(feature, "cancel_token"):
        setattr(feature, "cancel_token", ctx.cancel_token)

    # Fit phase (if not state_ready)
    if not state_ready:
        ctx.check_cancel()

        def input_factory() -> Iterator[tuple[str, pd.DataFrame]]:
            return iter_manifest(manifest, filter_factory=filter_factory)

        feature.fit(InputStream(input_factory, n_entries=len(manifest)))
        feature.save_state(run_root)
        # Item 5.3, and the placement is the item. Written here rather than
        # beside params.json because only this branch means a fit actually ran:
        # params.json is written on every invocation, before load_state, so its
        # `_scope` is the scope of whichever run came last. For a params-level
        # fitter -- scope-free, so every apply scope shares one run root -- those
        # two are different answers, and this is the one that stays true.
        write_fit_scope(run_root, scope, scope_dependent=feature.scope_dependent)

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
        # Single per-entry choke point (fires on cache-hit, inline, and
        # parallel-drain paths): report progress + refresh the liveness heartbeat.
        done = _total_written + len(_pending_idx_rows)
        ctx.progress.on_entry_end(
            done, ctx.total, make_entry_key(row.group, row.sequence)
        )
        ctx.heartbeat(done)
        if len(_pending_idx_rows) >= _IDX_FLUSH_EVERY:
            _flush_idx()

    # --- Cache-hit pre-pass --------------------------------------------------
    # When state is cached and we are not overwriting, resolve which entries
    # already have valid outputs WITHOUT loading their input parquet (the apply
    # loop would otherwise deserialize each input only to discard it). Validated
    # hits are recorded immediately; their keys are excluded from the compute
    # manifest so iter_manifest never loads them.
    # NOTE: ``check_output`` here is the run_feature *parameter* (a bool); the
    # feature's optional validator is the *attribute* feature.check_output.
    # Resolved once and reused by the cache-hit pre-pass.
    custom_check = getattr(feature, "check_output", None) if check_output else None
    if custom_check is not None and not callable(custom_check):
        custom_check = None  # not a validator method; use the default

    # What each entry recorded consuming, before this run touches the index. A
    # skipped entry carries its own value forward; re-deriving it here would
    # assert that an output not recomputed was built from whatever is true now,
    # and `drifted_entries` compares exactly this cell -- so the evidence of a
    # source having moved would erase itself on the first ordinary re-run.
    prior_compositions: dict[tuple[str, str], str] = (
        {
            entry: digest
            for entry, (_roots, digest) in recorded_consumption(
                ds, storage_feature_name, run_id
            ).items()
        }
        if feature.consumed_roots
        else {}
    )

    skip_keys: set[str] = set()
    # Entries whose provenance neither side can establish. Collected rather than
    # reported per entry: on a dataset with no projection this is every entry,
    # and one line naming the repair is worth more than two hundred naming the
    # symptom.
    undetectable: set[tuple[str, str]] = set()
    if state_ready and not overwrite:
        for entry_key in manifest:
            group, sequence = resolve_sequence_identity(entry_key, scope.entry_map)
            meta = build_feature_meta(group, sequence, run_root)
            if not meta.out_path.exists():
                continue
            if check_output:
                # A buggy/edge-case validator must not abort the whole run; treat
                # any failure as "invalid output" so the entry is recomputed.
                try:
                    ok = (
                        custom_check(meta, run_root)
                        if custom_check is not None
                        else default_check_output(meta, run_root)
                    )
                except Exception as exc:
                    print(
                        f"[feature:{feature.name}] check_output raised for "
                        f"({group},{sequence}): {exc}; recomputing.",
                        file=sys.stderr,
                    )
                    ok = False
                if not ok:
                    continue  # corrupt/incomplete -> fall through to recompute
            # Reading the footer can fail on a truncated/half-written parquet
            # (e.g. an OOM-killed prior run). Recompute rather than crash.
            try:
                n_rows = output_n_rows(meta.out_path)
            except Exception as exc:
                print(
                    f"[feature:{feature.name}] unreadable cached output for "
                    f"({group},{sequence}): {exc}; recomputing.",
                    file=sys.stderr,
                )
                continue
            entry = (meta.group, meta.sequence)
            recorded = prior_compositions.get(entry)
            was_made_from = recorded if recorded is not None else ""
            now_made_of = entry_composition(feature, scope, entry)
            disposition = cached_entry_disposition(was_made_from, now_made_of)
            if disposition == "recompute":
                # Serving this would be the wrong answer this milestone exists to
                # prevent, and recording it would restamp the only cell that says
                # so. Costly, loud, and never destructive.
                print(
                    f"[feature:{feature.name}] cannot serve cached output for "
                    f"({group},{sequence}): {_disposition_reason(was_made_from, now_made_of)}"
                    f"; recomputing.",
                    file=sys.stderr,
                )
                continue
            if disposition == "undetectable":
                undetectable.add(entry)
            _record_row(
                FeatureIndexRow(
                    run_id=run_id,
                    feature=storage_feature_name,
                    version=feature.version,
                    group=meta.group,
                    sequence=meta.sequence,
                    abs_path=Path(ds.relative_to_root(meta.out_path)),
                    consumed_roots=encode_consumed_roots(feature.consumed_roots),
                    # Carried, never re-derived. An entry with no prior row is an
                    # output nothing describes, so its provenance is unknown and
                    # says so, rather than claiming the present.
                    consumed_composition=was_made_from,
                    n_rows=n_rows,
                    params_hash=params_hash,
                    identity_scheme=FEATURE_IDENTITY_SCHEME,
                )
            )
            skip_keys.add(entry_key)

    if undetectable:
        roots = ", ".join(sorted({root for root in feature.consumed_roots if root}))
        print(
            f"[feature:{feature.name}] served {len(undetectable)} cached "
            f"entry(ies) whose source cannot be checked: neither the run nor "
            f"{roots} records a composition for them, so a change under "
            f"{roots} would go unnoticed. Run `mosaic reindex` (or "
            f"`mosaic reprobe-media --apply` for media) to write the projection.",
            file=sys.stderr,
        )

    compute_manifest = (
        {k: v for k, v in manifest.items() if k not in skip_keys}
        if skip_keys
        else manifest
    )

    # A full cache hit: state needed no fit and every manifest entry was already
    # valid on disk. Surfaced on Result (does not alter the run's flow).
    cache_hit = (
        state_ready
        and not overwrite
        and len(manifest) > 0
        and len(skip_keys) == len(manifest)
    )

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
                    abs_path=Path(ds.relative_to_root(meta.out_path)),
                    consumed_roots=encode_consumed_roots(feature.consumed_roots),
                    consumed_composition=entry_composition(
                        feature, scope, (meta.group, meta.sequence)
                    ),
                    n_rows=n_rows,
                    params_hash=params_hash,
                    identity_scheme=FEATURE_IDENTITY_SCHEME,
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
        # Cooperative cancel checkpoint: covers both apply loops and both the
        # executor and inline branches. Completed entries are already durable.
        ctx.check_cancel()
        group, sequence = resolve_sequence_identity(entry_key, scope.entry_map)
        meta = build_feature_meta(group, sequence, run_root)

        # Cache hits are resolved up-front in the pre-pass; any entry reaching
        # here needs (re)computation.
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
                        str(ds.manifest_path),
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
                    abs_path=Path(ds.relative_to_root(meta.out_path)),
                    consumed_roots=encode_consumed_roots(feature.consumed_roots),
                    consumed_composition=entry_composition(
                        feature, scope, (group, sequence)
                    ),
                    n_rows=n_rows,
                    params_hash=params_hash,
                    identity_scheme=FEATURE_IDENTITY_SCHEME,
                )
            )
            del result_df
            gc.collect()

    # Iterate manifest entries. A cooperative cancel (raised at the top of
    # _process_entry) stops dispatch; completed entries stay recorded in
    # index.csv + on disk, so a later identical run resumes from the entries whose
    # output parquet is already present (the cache-hit pre-pass above).
    try:
        if apply_overlap is not None:
            for entry_key, df, core_start, core_end in iter_manifest(
                compute_manifest,
                filter_factory=filter_factory,
                overlap_frames=apply_overlap,
                progress_label=storage_feature_name,
            ):
                _process_entry(entry_key, df, core_start, core_end)
        else:
            for entry_key, df in iter_manifest(
                compute_manifest,
                filter_factory=filter_factory,
                progress_label=storage_feature_name,
            ):
                _process_entry(entry_key, df, 0, len(df))

        # Drain remaining futures
        if executor is not None:
            while pending:
                _drain_completed()
            executor.shutdown(wait=True)
    except Cancelled:
        _flush_idx()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        raise

    # Global marker (for empty-input features).
    #
    # ``feature.inputs.is_empty`` is load-bearing, not belt-and-braces. An empty
    # manifest has two very different causes: a global feature that declares no
    # pipeline inputs and legitimately writes one artifact for the whole run, and
    # a *per-frame* feature whose inputs resolved to nothing -- an empty tracks
    # index, a scope narrowing that intersected to nothing, a selector matching no
    # variant. On ``not manifest`` alone the second case gets a ``__global__`` row
    # and, since ``all([])`` is True below, a finished marker: a run that computed
    # nothing, recorded as having completed something. That is reached from
    # ``Pipeline.run`` and ``.clean``, not only ``.status``, so it is a deletion
    # hazard rather than a display defect. ``is_empty`` is the same discriminator
    # ``run_feature`` already uses to decide whether to build a manifest at all.
    if (
        _total_written == 0
        and not _pending_idx_rows
        and not manifest
        and feature.inputs.is_empty
    ):
        _pending_idx_rows.append(
            FeatureIndexRow(
                run_id=run_id,
                feature=storage_feature_name,
                version=feature.version,
                group="",
                sequence="__global__",
                abs_path=Path(ds.relative_to_root(run_root)),
                n_rows=0,
                params_hash=params_hash,
                identity_scheme=FEATURE_IDENTITY_SCHEME,
            )
        )

    # Finalize — flush any remaining rows
    _flush_idx()
    # Mark the run finished only when every manifest entry's output parquet is on
    # disk. The filesystem is the source of truth (same disk check the Pipeline
    # cache gate uses): under concurrency the last finisher sees all files and
    # marks it once; a run that skipped an entry still owned by a live (or
    # crashed-within-window) peer sees a missing file and stays unfinished, so it
    # is resumable rather than falsely finished. Empty manifest / global-marker
    # runs have no entries → all([]) is True → finished, as before. A per-frame
    # feature that resolved to nothing now writes no row above, so this marks
    # nothing: ``mark_finished`` is a no-op when no row carries the run_id.
    all_entries = {
        resolve_sequence_identity(entry_key, scope.entry_map) for entry_key in manifest
    }
    complete = all(
        build_output_path(group, sequence, run_root).exists()
        for group, sequence in all_entries
    )
    if complete:
        idx.mark_finished(run_id)
    print(f"[feature:{storage_feature_name}] completed run_id={run_id} -> {run_root}")
    return Result(
        feature=storage_feature_name,
        run_id=run_id,
        execution_id=ctx.execution_id,
        cache_hit=cache_hit,
    )


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
    ds: Dataset, kind: str, labels_run_id: str | None = None
) -> dict[tuple[str, str], Path]:
    """The per-entry ``.npz`` paths for a label *kind*, one row per sequence.

    Resolves through the typed index and :func:`select_label_variant_rows`, so a
    consumer reads the same variant its identity was built from -- pass the same
    ``labels_run_id`` here as to :func:`resolve_labels_variants`. A labelled
    variant supersedes an unlabelled row; two genuine recipes raise.
    """
    df = select_label_variant_rows(read_labels_index(ds, kind), labels_run_id)
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


def resolve_labels_variants(
    ds: Dataset, feature: Feature, labels_run_id: str | None = None
) -> tuple[str, ...]:
    """Which label recipes this feature's params resolve to, before the digest.

    The label analog of ``_resolve_tracks`` -> ``scope.tracks_variants``, driven
    from the params fields rather than a manifest input: labels are a
    ``GroundTruthLabelsSource`` on ``feature.params``, not a ``"tracks"`` input.
    Each such field's kind is resolved through :func:`select_label_variant_rows`
    (labelled supersedes unlabelled, two genuine recipes raise), and every
    non-empty variant it names is collected. Authored kinds (empty ``run_id``)
    contribute nothing, exactly as unlabelled tracks do, so a dataset with no
    label recipe keeps the identifiers it already has.
    """
    params = getattr(feature, "params", None)
    if params is None:
        return ()
    variants: set[str] = set()
    for field_name in type(params).model_fields:
        value = getattr(params, field_name, None)
        if isinstance(value, LabelsSource) and value.kind:
            resolved = select_label_variant_rows(
                read_labels_index(ds, value.kind), labels_run_id
            )
            variants.update(str(r) for r in resolved["run_id"] if str(r))
    return tuple(sorted(variants))


def _find_merged_column(column: str, input_index: int, df: pd.DataFrame) -> str | None:
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
    entries: Iterable[tuple[str, str]] | None = None,
    filter_start_frame: int | None = None,
    filter_end_frame: int | None = None,
    filter_start_time: float | None = None,
    filter_end_time: float | None = None,
    pair_filter: NNResult | None = None,
    tracks_run_id: str | None = None,
) -> pd.DataFrame:
    """Load and align value columns from tracks, features, and labels.

    Sources can reference tracks columns, feature output columns, or
    ground-truth labels. All are aligned by frame/id via a single
    manifest pass.

    ``tracks_run_id`` names one tracks variant, and is only consulted when a
    ``TracksColumn`` is among the sources -- that is what puts the ``"tracks"``
    literal into the synthetic inputs below. Without it a notebook reading a
    column from a dataset holding two recipes for one sequence would meet the
    resolver's refusal with no keyword able to answer it.
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
    entries_set = (
        {(str(g), str(s)) for g, s in entries} if entries is not None else None
    )

    manifest, scope = build_manifest(
        ds,
        synthetic_inputs,
        groups_set,
        sequences_set,
        entries_set,
        tracks_run_id=tracks_run_id,
    )

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
