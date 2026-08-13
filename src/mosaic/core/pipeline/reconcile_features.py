"""The forward pass for feature runs.

A feature run is a directory ``features/<storage>/<run_id>/`` holding one parquet
per entry plus ``params.json`` and the ``.identity_scheme`` marker, with one index
row per entry all sharing ``run_id``. Re-addressing one means renaming the
directory to the new ``run_id``, refreshing the marker, and restamping every row's
``run_id`` and ``abs_path`` -- a metadata move, no recompute.

The run_id is recomputed by the one identity site, :func:`compute_run_id`, fed a
``Feature`` rebuilt from the recorded ``params.json`` and a ``Scope`` rebuilt from
the same file's ``_scope`` and ``_resolved`` blocks. Feature-to-feature upstreams
are substituted through the shared remap first, so a chain re-addresses in one
bottom-up pass; the runs here are ordered so an upstream lands before the
downstream that reads it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline._utils import Scope, atomic_write
from mosaic.core.pipeline.dataset_indexes import feature_storages
from mosaic.core.pipeline.identity_scheme import (
    FEATURE_IDENTITY_SCHEME,
    read_identity_scheme,
    write_identity_scheme,
)
from mosaic.core.pipeline.index import (
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.reconcile import (
    PassBuilder,
    ReconcileFinding,
    ReconcileKey,
    ReconcilePass,
    ReconcileState,
    Verdict,
    backup_index,
    register_identity_reconciler,
)
from mosaic.core.pipeline.run import build_run_params_payload, compute_run_id
from mosaic.core.pipeline.types import Result

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.types import Feature


_FROM = "__from__"


_HASH_LEN = 10
_HEX = frozenset("0123456789abcdef")


def _slug_of(storage_name: str) -> str:
    """The feature slug behind a storage directory name.

    ``derive_storage_name`` writes ``<slug>`` or ``<slug>__from__<suffix>``, so the
    slug is everything before the first ``__from__``. Reversed here so the class
    behind a run can be looked up in the registry.
    """
    return storage_name.split(_FROM, 1)[0]


def _split_feature_run_id(run_id: str) -> tuple[str, str] | None:
    """Split ``<version>-<10-hex-digest>`` into ``(version, digest)``, or ``None``.

    ``compute_run_id`` builds ``f"{feature.version}-{params_hash}"`` with a
    10-hex-char digest, so the version is everything before the final ``-`` and the
    digest is the last ten characters. Keeping the recorded version is what makes a
    version bump a new recipe rather than a re-address of old bytes.
    """
    if len(run_id) < _HASH_LEN + 2 or run_id[-(_HASH_LEN + 1)] != "-":
        return None
    version, digest = run_id[: -(_HASH_LEN + 1)], run_id[-_HASH_LEN:]
    if not version or any(char not in _HEX for char in digest):
        return None
    return version, digest


def _feature_class(slug: str) -> "type[Feature] | None":
    """The feature class registered under *slug*, or ``None`` if none is.

    The registry lookup ``cli._features.feature_class_for_slug`` performs, but
    returning ``None`` rather than exiting: a run whose feature has been removed
    from the registry is one this pass reports as unresolvable and steps past, not
    one that aborts the sweep.
    """
    from mosaic.behavior.feature_library import FEATURES
    from pydantic import BaseModel

    for cls in FEATURES.values():
        if getattr(cls, "name", None) != slug:
            continue
        inputs_cls = getattr(cls, "Inputs", None)
        if isinstance(inputs_cls, type) and issubclass(inputs_cls, BaseModel):
            return cls
        return None
    return None


def _is_result(value: object) -> TypeGuard["Result[str]"]:
    """Narrow *value* to ``Result[str]``.

    A bare ``isinstance(value, Result)`` leaves the type argument unknown -- a
    plain ``object`` gives the checker nothing to infer it from -- so the checker
    reports every ``.feature``/``.run_id`` access on it as unknown. ``Result``'s
    TypeVar is bound to ``str`` and defaults to it, so ``Result[str]`` is the only
    argument it can carry; stated once here, the same guard ``resolve.py`` uses.
    """
    return isinstance(value, Result)


class _ResolvedRef(BaseModel):
    """One ``_resolved`` entry from ``params.json`` -- provenance, read here."""

    model_config = ConfigDict(extra="ignore")

    where: str = ""
    feature: str = ""
    run_id: str | None = None


class _ScopeBlock(BaseModel):
    """The ``_scope`` block: the invocation's resolved scope and compositions."""

    model_config = ConfigDict(extra="ignore")

    scope_dependent: bool = False
    consumed_roots: list[str] = Field(default_factory=list)
    entries: list[list[str]] = Field(default_factory=list)
    compositions: dict[str, dict[str, str]] = Field(default_factory=dict)


class _ParamsFile(BaseModel):
    """The recorded ``params.json``, typed.

    Reading it through a model rather than raw ``json.loads`` navigation keeps this
    module free of the ``Any``/``Unknown`` that untyped JSON access spreads, and
    turns a malformed or partial file into a ``ValidationError`` the caller catches
    -- an unresolvable run, not a crash. ``extra="ignore"`` so a file written by a
    later scheme with new keys still reads.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    params: dict[str, object] = Field(default_factory=dict, alias="_params")
    inputs: list[object] = Field(default_factory=list, alias="_inputs")
    frame_range: list[int | None] = Field(default_factory=list, alias="_frame_range")
    overlap_frames: int = Field(0, alias="_overlap_frames")
    scope: _ScopeBlock = Field(default_factory=_ScopeBlock, alias="_scope")
    resolved: list[_ResolvedRef] = Field(default_factory=list, alias="_resolved")

    @property
    def records_resolutions(self) -> bool:
        """Whether the file carried a ``_resolved`` block at all.

        ``run_feature`` writes the key unconditionally, empty list included, so an
        absent one dates the file to before the block existed rather than saying
        the run resolved nothing. The distinction is load-bearing and the default
        erases it: both spellings arrive here as ``[]``.
        """
        return "resolved" in self.model_fields_set


def _frame_range(params: _ParamsFile | None) -> tuple[int | None, int | None]:
    if params is None or len(params.frame_range) != 2:
        return None, None
    return params.frame_range[0], params.frame_range[1]


def _overlap_frames(params: _ParamsFile | None) -> int:
    """The overlap a run recorded, defaulting to none.

    A file written before the key existed reads as 0, which is the right answer:
    the argument existed then, but the digest did not cover it, so a run that
    used it was addressed as though it had not. Such a run keeps the address it
    has, and a fresh identical invocation mints a new one -- the one-wrong-cache-
    miss migration this repository has taken before.
    """
    return 0 if params is None else int(params.overlap_frames)


def _entries(params: _ParamsFile | None) -> set[tuple[str, str]]:
    if params is None:
        return set()
    return {(pair[0], pair[1]) for pair in params.scope.entries if len(pair) == 2}


def _compositions(
    params: _ParamsFile | None, entries: set[tuple[str, str]]
) -> dict[tuple[str, str], dict[str, str]]:
    """Rebuild ``Scope.compositions`` from ``_scope.compositions``.

    The block is keyed by ``make_entry_key(group, sequence)``; the entries set
    gives the inverse map back to the ``(group, sequence)`` tuple ``Scope`` wants,
    so no key parsing is needed.
    """
    if params is None:
        return {}
    by_key = {make_entry_key(group, seq): (group, seq) for (group, seq) in entries}
    out: dict[tuple[str, str], dict[str, str]] = {}
    for entry_key, per_root in params.scope.compositions.items():
        entry = by_key.get(entry_key)
        if entry is not None:
            out[entry] = dict(per_root)
    return out


def _resolved_variants(
    params: _ParamsFile | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The recorded tracks and labels variant ids, from ``_resolved``.

    ``compute_run_id`` reads these off ``Scope``; ``params.json`` records them as
    synthetic ``inputs[tracks]`` / ``inputs[labels]`` entries in ``_resolved``
    (they are Scope terms, not ``_inputs`` fields).
    """
    if params is None:
        return (), ()
    tracks = [
        r.run_id for r in params.resolved if r.where == "inputs[tracks]" and r.run_id
    ]
    labels = [
        r.run_id for r in params.resolved if r.where == "inputs[labels]" and r.run_id
    ]
    return tuple(tracks), tuple(labels)


@dataclass
class _RunRead:
    """Everything read off disk about one feature run, before it is recomputed."""

    storage_name: str
    slug: str
    run_id: str
    run_root: Path
    index_path: Path
    scheme: str
    feature: "Feature | None"
    build_error: str
    frame_start: int | None
    frame_end: int | None
    overlap_frames: int
    entries: set[tuple[str, str]]
    compositions: dict[tuple[str, str], dict[str, str]]
    tracks_old: tuple[str, ...]
    labels_old: tuple[str, ...]
    records_resolutions: bool
    upstream_feature_runs: tuple[str, ...]
    # The scope rebuilt with upstreams substituted, filled by ``_classify`` and
    # read by ``_relocate`` to rewrite the run's recorded provenance so the next
    # pass reproduces its new identifier.
    scope: "Scope | None" = None


class FeatureReconciler:
    """Recompute and re-address feature runs. Registered under ``"features"``."""

    key: ReconcileKey = "features"
    # Tracks and labels are read by features, so they must be reconciled first --
    # a moved tracks variant changes the ``_tracks`` term of every feature that
    # read it. The feature-to-feature ordering is handled inside this reconciler.
    depends_on: tuple[ReconcileKey, ...] = ("tracks", "labels")

    def __init__(self, ds: Dataset) -> None:
        self._ds = ds

    # -- enumeration -------------------------------------------------------

    def _run_ids(self, storage_name: str) -> list[str]:
        index_path = feature_index_path(self._ds, storage_name)
        if not index_path.exists():
            return []
        frame = feature_index(index_path).read()
        return sorted({str(run_id) for run_id in frame["run_id"]})

    # -- reading -----------------------------------------------------------

    def _read_run(self, storage_name: str, run_id: str) -> _RunRead:
        ds = self._ds
        run_root = feature_run_root(ds, storage_name, run_id)
        index_path = feature_index_path(ds, storage_name)
        slug = _slug_of(storage_name)
        scheme = read_identity_scheme(run_root)

        params = self._load_params(run_root)
        frame_start, frame_end = _frame_range(params)
        overlap_frames = _overlap_frames(params)
        entries = _entries(params)
        compositions = _compositions(params, entries)
        tracks_old, labels_old = _resolved_variants(params)

        feature, build_error, upstream = self._build(slug, params)

        return _RunRead(
            storage_name=storage_name,
            slug=slug,
            run_id=run_id,
            run_root=run_root,
            index_path=index_path,
            scheme=scheme,
            feature=feature,
            build_error=build_error,
            frame_start=frame_start,
            frame_end=frame_end,
            overlap_frames=overlap_frames,
            entries=entries,
            compositions=compositions,
            tracks_old=tracks_old,
            labels_old=labels_old,
            records_resolutions=params is not None and params.records_resolutions,
            upstream_feature_runs=upstream,
        )

    def _load_params(self, run_root: Path) -> _ParamsFile | None:
        path = run_root / "params.json"
        if not path.exists():
            return None
        try:
            return _ParamsFile.model_validate_json(path.read_text())
        except (OSError, ValidationError):
            return None

    def _build(
        self, slug: str, params: _ParamsFile | None
    ) -> tuple["Feature | None", str, tuple[str, ...]]:
        """Rebuild the ``Feature`` and collect its upstream feature run ids.

        Returns ``(feature, error, upstream_run_ids)``. On any failure the feature
        is ``None`` and the error is a one-line reason -- the run is then
        unresolvable, and the sweep continues.
        """
        if params is None:
            return None, "params.json missing or unreadable", ()
        cls = _feature_class(slug)
        if cls is None:
            return None, f"feature {slug!r} is not in the registry", ()

        inputs_cls = getattr(cls, "Inputs", None)
        if inputs_cls is None:
            return None, f"feature {slug!r} has no Inputs model", ()
        try:
            inputs_obj = inputs_cls.model_validate(params.inputs)
            params_dict = params.params
            feature = cls(inputs_obj, params_dict)  # pyright: ignore[reportCallIssue]
        except (ValidationError, ValueError, TypeError) as exc:
            return None, f"could not rebuild feature: {exc}", ()
        return feature, "", _upstream_feature_runs(feature)

    # -- recompute + classify + apply -------------------------------------

    def reconcile(
        self, state: ReconcileState, *, apply: bool, force: bool
    ) -> ReconcilePass:
        del force  # features never delete; the destructive path is a later pass
        builder = PassBuilder()
        reads: list[_RunRead] = []
        for storage_name in feature_storages(self._ds):
            for run_id in self._run_ids(storage_name):
                try:
                    reads.append(self._read_run(storage_name, run_id))
                except Exception as exc:  # noqa: BLE001 - a bad run must not abort the sweep
                    builder.error(f"features/{storage_name}/{run_id}: {exc}")
        for read in _topo_order(reads):
            self._process(read, state, builder, apply=apply)
        return builder.build()

    def _process(
        self,
        read: _RunRead,
        state: ReconcileState,
        builder: PassBuilder,
        *,
        apply: bool,
    ) -> None:
        verdict, new_run_id, reason = self._classify(read, state)
        finding = ReconcileFinding(
            key="features",
            verdict=verdict,
            old_address=read.run_id,
            new_address=new_run_id,
            index_path=read.index_path,
            run_root=str(read.run_root),
            reason=reason,
        )
        # State first, so a downstream sees this run's outcome regardless of
        # whether files are being written this pass -- a dry run's report must show
        # the same cascade an applied run would.
        if verdict == "identity_shift_relocatable":
            state.record_move("features", read.run_id, new_run_id)
        elif verdict in ("identity_shift_recompute", "unresolvable_pre_provenance"):
            state.record_blocked("features", read.run_id)

        if not apply:
            builder.add(finding)
            return
        builder.add(self._apply(read, finding, builder))

    def _classify(
        self, read: _RunRead, state: ReconcileState
    ) -> tuple[Verdict, str, str]:
        """Return ``(verdict, new_run_id, reason)`` for one run.

        The recomputed identifier keeps the run's *recorded* version and swaps in
        the freshly recomputed digest: ``run_id = <version>-<params_hash>``, and a
        version bump is a new recipe (a fresh run), never a re-address of old bytes
        under a new version. Only the digest -- driven by the identity machinery
        and the resolved inputs -- decides a move here.

        ``new_run_id`` is the recorded id when nothing moved or nothing could be
        computed, so a finding always carries a usable address.
        """
        if read.feature is None:
            return "unresolvable_pre_provenance", read.run_id, read.build_error
        split = _split_feature_run_id(read.run_id)
        if split is None:
            return (
                "unresolvable_pre_provenance",
                read.run_id,
                f"run id {read.run_id!r} is not <version>-<digest>",
            )
        recorded_version, recorded_hash = split

        # Substitute every upstream's new address (features, tracks, labels) into
        # this run's inputs before recomputing. Mutates the rebuilt feature in
        # place, the same idiom `resolve_references` uses.
        _remap_feature_refs(read.feature, state)
        tracks_new = tuple(
            sorted({state.resolved("tracks", v) for v in read.tracks_old})
        )
        labels_new = tuple(
            sorted({state.resolved("labels", v) for v in read.labels_old})
        )
        scope = Scope(
            entries=set(read.entries),
            frame_start=read.frame_start,
            frame_end=read.frame_end,
            tracks_variants=tracks_new,
            labels_variants=labels_new,
            compositions=read.compositions,
        )
        read.scope = scope
        try:
            _, new_hash = compute_run_id(
                read.feature,
                read.frame_start,
                read.frame_end,
                scope,
                overlap_frames=read.overlap_frames,
            )
        except Exception as exc:  # noqa: BLE001 - a raising feature is unresolvable, not fatal
            return (
                "unresolvable_pre_provenance",
                read.run_id,
                f"recompute failed: {exc}",
            )
        new_run_id = f"{recorded_version}-{new_hash}"

        blocked = self._upstream_blocked(read, state)
        if blocked:
            # An input is now at an unknown address, so this run cannot be a pure
            # re-address whether or not its own digest happens to have moved.
            return "identity_shift_recompute", new_run_id, blocked

        scheme_current = read.scheme == FEATURE_IDENTITY_SCHEME
        if new_hash == recorded_hash:
            if scheme_current:
                return "ok", read.run_id, "identifier and scheme current"
            if not read.records_resolutions:
                # An equal digest is not evidence of sameness here. The recompute
                # reads the run's upstream variants out of ``_resolved``, and a
                # file predating that block yields none -- so the payload omits
                # the ``_tracks``/``_labels`` terms exactly as the original mint
                # did, and the digests agree by construction. A live run resolves
                # the indexes instead, finds whatever variant they now name, and
                # mints a different identifier. Refreshing the marker on that
                # would assert the run is current under a scheme whose identity
                # for its inputs is a different digest, stranding it at an address
                # nothing will address again while claiming the opposite.
                return (
                    "identity_shift_recompute",
                    read.run_id,
                    "run predates the recorded-resolution block, so its upstream "
                    "variants are unknown and an unchanged digest cannot confirm "
                    "the recipe; recompute rather than refresh",
                )
            return (
                "scheme_stale",
                read.run_id,
                f"identifier unchanged; marker {read.scheme or 'absent'!r} "
                f"-> {FEATURE_IDENTITY_SCHEME!r}",
            )

        # The digest moved. That is honest only when the identity machinery moved
        # (an older scheme) or an upstream was re-addressed this pass. A digest that
        # moved under the *current* scheme with no upstream move is an unexplained
        # divergence -- a recorded recipe that no longer hashes to its own id --
        # and re-addressing on it would be a guess, so decline.
        if scheme_current and not self._upstream_moved(read, state):
            return (
                "unresolvable_pre_provenance",
                new_run_id,
                "digest changed under the current scheme with no upstream move; "
                "recorded recipe does not reproduce its own identifier",
            )
        unconfirmed = self._unconfirmed_inputs(read)
        if unconfirmed:
            return "identity_shift_recompute", new_run_id, unconfirmed
        return (
            "identity_shift_relocatable",
            new_run_id,
            f"identity machinery moved ({read.run_id} -> {new_run_id}); "
            f"recipe confirmed unchanged",
        )

    def _upstream_moved(self, read: _RunRead, state: ReconcileState) -> bool:
        """True if any upstream this run reads was re-addressed this pass."""
        if any(("features", old) in state.remap for old in read.upstream_feature_runs):
            return True
        if any(("tracks", old) in state.remap for old in read.tracks_old):
            return True
        return any(("labels", old) in state.remap for old in read.labels_old)

    def _upstream_blocked(self, read: _RunRead, state: ReconcileState) -> str:
        """A reason string if any upstream could not be re-addressed, else ``""``."""
        for old in read.upstream_feature_runs:
            if state.is_blocked("features", old):
                return f"upstream feature run {old} could not be re-addressed"
        for old in read.tracks_old:
            if state.is_blocked("tracks", old):
                return f"upstream tracks variant {old} could not be re-addressed"
        for old in read.labels_old:
            if state.is_blocked("labels", old):
                return f"upstream labels variant {old} could not be re-addressed"
        return ""

    def _unconfirmed_inputs(self, read: _RunRead) -> str:
        """A reason a moved id cannot be trusted as a pure re-address, else ``""``.

        The confirmation predicate: every upstream reference must be pinned to a
        concrete run (an unpinned ``run_id`` means "latest, whichever that was",
        which cannot be re-derived), and a run written before either the scheme
        marker or the recorded-resolution block predates the provenance a
        re-address relies on.
        """
        if read.feature is None:
            return "feature could not be rebuilt"
        if read.scheme == "":
            return "run predates the identity-scheme marker; recompute rather than move"
        if not read.records_resolutions:
            return (
                "run predates the recorded-resolution block, so its upstream "
                "variants are unknown; recompute rather than move"
            )
        for item in read.feature.inputs.root:
            if isinstance(item, Result) and item.run_id is None:
                return f"upstream {item.feature!r} was never pinned to a run"
        for name in type(read.feature.params).model_fields:
            value: object = getattr(read.feature.params, name)
            if _is_result(value) and value.feature and value.run_id is None:
                return f"params.{name} reference {value.feature!r} was never pinned"
        return ""

    def _apply(
        self, read: _RunRead, finding: ReconcileFinding, builder: PassBuilder
    ) -> ReconcileFinding:
        if finding.verdict == "scheme_stale":
            write_identity_scheme(read.run_root, FEATURE_IDENTITY_SCHEME)
            return finding.with_action("marker_refreshed")
        if finding.verdict != "identity_shift_relocatable":
            return finding.with_action("reported")
        return self._relocate(read, finding, builder)

    def _relocate(
        self, read: _RunRead, finding: ReconcileFinding, builder: PassBuilder
    ) -> ReconcileFinding:
        ds = self._ds
        new_run_root = feature_run_root(ds, read.storage_name, finding.new_address)
        if new_run_root.exists():
            builder.error(
                f"features/{read.storage_name}: cannot re-address {read.run_id} "
                f"-> {finding.new_address}; target directory already exists"
            )
            return finding.with_action("none")
        builder.backed_up(backup_index(read.index_path))
        # Directory move first, index rewrite second. The move is deliberately
        # not under the index lock: it is filesystem work of unbounded duration
        # and the lock's timeout is tuned for a CSV rewrite.
        _ = shutil.move(str(read.run_root), str(new_run_root))
        write_identity_scheme(new_run_root, FEATURE_IDENTITY_SCHEME)
        self._rewrite_provenance(read, new_run_root)

        def _rewrite(stored: str) -> str:
            return ds.relative_to_root(new_run_root / Path(stored).name)

        _ = feature_index(read.index_path).remap_run_id(
            read.run_id, finding.new_address, path_rewrite=_rewrite
        )
        return finding.with_action("relocated")

    def _rewrite_provenance(self, read: _RunRead, new_run_root: Path) -> None:
        """Rewrite the moved run's ``params.json`` to record its new upstream ids.

        When a run relocates because an upstream moved, its recorded ``_inputs`` and
        ``_resolved`` still name the *old* upstream, so the next pass could not
        reproduce the new identifier. This restamps them from the substituted
        feature (whose ``Result`` ids ``_classify`` already pinned to their new
        values) and the resolved tracks/labels variants, preserving ``_scope`` and
        ``_frame_range`` exactly. A relocation with no upstream move rewrites the
        same content -- harmless.
        """
        if read.feature is None or read.scope is None:
            return
        # The feature-to-feature and params half of ``_resolved``, from the
        # substituted feature (whose Result ids ``_classify`` pinned to their new
        # values); ``build_run_params_payload`` appends the tracks/labels variants
        # from the scope. Same order and skip rule as ``resolve_references``:
        # inputs then params in declaration order, empty-feature references dropped.
        resolutions: list[dict[str, str | None]] = []
        for position, item in enumerate(read.feature.inputs.root):
            if isinstance(item, Result) and item.feature:
                resolutions.append(
                    {
                        "where": f"inputs[{position}]",
                        "feature": item.feature,
                        "run_id": item.run_id,
                    }
                )
        for name in type(read.feature.params).model_fields:
            value: object = getattr(read.feature.params, name)
            if _is_result(value) and value.feature:
                resolutions.append(
                    {
                        "where": f"params.{name}",
                        "feature": value.feature,
                        "run_id": value.run_id,
                    }
                )
        payload = build_run_params_payload(
            read.feature,
            read.frame_start,
            read.frame_end,
            read.scope,
            resolutions,
            overlap_frames=read.overlap_frames,
        )
        atomic_write(
            new_run_root / "params.json",
            lambda p: p.write_text(json.dumps(payload, indent=2)),
        )


def _upstream_feature_runs(feature: "Feature") -> tuple[str, ...]:
    """Every concrete upstream feature run id this feature reads.

    Read from the ``Result``-shaped references in ``inputs`` and in the top-level
    ``params`` fields -- the same places ``resolve_references`` pins. Used to order
    the runs bottom-up and to detect a blocked upstream.
    """
    runs: list[str] = []
    for item in feature.inputs.root:
        if isinstance(item, Result) and item.run_id is not None:
            runs.append(item.run_id)
    for name in type(feature.params).model_fields:
        value: object = getattr(feature.params, name)
        if _is_result(value) and value.feature and value.run_id is not None:
            runs.append(value.run_id)
    return tuple(runs)


def _remap_feature_refs(feature: "Feature", state: ReconcileState) -> None:
    """Point every upstream feature reference at its new address, in place.

    Mutates ``run_id`` on each ``Result`` in ``inputs`` and top-level ``params``,
    the same in-place idiom ``resolve_references`` uses -- ``Result`` is a mutable
    ``StrictModel``. An upstream that did not move resolves to its own id.
    """
    for item in feature.inputs.root:
        if isinstance(item, Result) and item.run_id is not None:
            item.run_id = state.resolved("features", item.run_id)
    for name in type(feature.params).model_fields:
        value: object = getattr(feature.params, name)
        if _is_result(value) and value.run_id is not None:
            value.run_id = state.resolved("features", value.run_id)


def _topo_order(reads: list[_RunRead]) -> list[_RunRead]:
    """Order runs so every upstream feature run precedes the runs that read it.

    Keyed on ``run_id`` -- an identifier is a hash over params, inputs and range,
    so a collision across two features is not a practical concern, and a
    feature-to-feature edge names its upstream by run id. A reference to a run not
    in this set (an upstream directory deleted by hand) is simply not an edge; that
    run resolves to its own id and the reader recomputes over it unchanged.
    """
    by_run: dict[str, _RunRead] = {read.run_id: read for read in reads}
    ordered: list[_RunRead] = []
    placed: set[str] = set()

    def place(read: _RunRead) -> None:
        if read.run_id in placed:
            return
        placed.add(read.run_id)  # mark first so a cycle terminates
        for upstream in read.upstream_feature_runs:
            parent = by_run.get(upstream)
            if parent is not None:
                place(parent)
        ordered.append(read)

    for read in reads:
        place(read)
    return ordered


def _build(ds: Dataset) -> FeatureReconciler:
    return FeatureReconciler(ds)


register_identity_reconciler("features", _build)
