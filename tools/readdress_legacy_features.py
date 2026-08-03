#!/usr/bin/env python
"""Re-address legacy feature runs to the identifiers the current code mints.

A migration aid for datasets whose feature runs predate the identity-scheme
marker, where recomputing them is expensive and provably unnecessary. Not part
of the toolkit's supported surface: it encodes an assertion the pipeline itself
declines to make, and the person running it owns that assertion.

Why ``mosaic reconcile`` will not do this
-----------------------------------------
Reconcile refuses to re-address a run written before the ``.identity_scheme``
marker existed. It cannot confirm from recorded provenance that the inputs are
unchanged -- legacy ``params.json`` has no ``_scope``/``_resolved`` blocks -- and
a loud cache miss beats a silent wrong hit. That refusal is right, and it stays
right: this script is not a ``--force`` for it, because reconcile derives the new
address from *recorded* provenance, and for a legacy run that provenance is empty.
The address it would compute is one no live run will ever look for.

So this derives the address a live ``run_feature`` would use instead -- pin, build
the manifest, resolve labels, hash, in that order -- and moves the artifact there.

What you are asserting by running it
------------------------------------
That the stored outputs are what the current code would produce for the same
params over the same inputs. That is decidable, and it is on you to decide it
before applying, typically by diffing the feature library across the upgrade:

  1. the source tables are equivalent (the converter's output semantics did not
     change, or changed only in ways the features normalise away), and
  2. the feature's *compute* did not change -- only declarations, type
     annotations and formatting did.

A version bump is the codebase telling you (2) is false, and that case is
refused automatically. Everything else is your reading of the diff.

What it refuses
---------------
* Any run whose feature ``version`` differs from the version recorded in its
  ``run_id``. A version bump is a *declared* recipe change, and the whole point
  of a declared version is that nothing re-addresses across one.
* Any run with an **unpinned upstream** (``"run_id": null`` in ``_inputs`` or
  ``_params``). That records "whichever run was latest at the time", which is not
  re-derivable; pinning it to today's latest would move the artifact under a
  lineage it never had, and a live run would resolve the same wrong way, so even
  the ``cache_hit`` check below would confirm it. Legacy runs are precisely the
  population that carries these.
* Anything downstream of a run it refused, because that downstream's identity
  pins an upstream id that is about to move under it.
* Any ``scope_dependent`` feature. Those hash the set of sequences in scope,
  which legacy ``params.json`` does not record, so the address would be a guess.

The dependency graph is keyed on ``(storage_name, run_id)``, never the digest
alone: ``compute_run_id``'s payload carries no feature-name term, so two features
can mint the same identifier, and a graph keyed on it would order a downstream
before its real upstream.

Everything it does is a directory rename plus a metadata rewrite: ``params.json``,
the identity-scheme marker, and the run's index rows, with each index backed up
first. No output is recomputed and nothing is deleted -- a refused run stays
exactly where it is and gets recomputed by an ordinary run.

Note that the dry run is *optimistic*: it resolves upstreams with
``on_missing_run="empty"``, since their index rows have not moved yet, so a run
can report ``readdress`` and then be skipped under ``--apply`` when the real
manifest check runs. Apply is still safe -- a skipped run is simply left -- but
the apply report is the authoritative one.

Usage
-----
    python tools/readdress_legacy_features.py --manifest /path/to/dataset.yaml
    python tools/readdress_legacy_features.py --manifest /path/to/dataset.yaml --apply

Verify afterwards by re-running the pipeline that produced those runs: every
re-addressed run must report ``cache_hit``. Keep the old inputs until it has.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard, cast

from pydantic import BaseModel

from mosaic.behavior.feature_library import FEATURES
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import Scope, atomic_write
from mosaic.core.pipeline.identity_scheme import (
    FEATURE_IDENTITY_SCHEME,
    write_identity_scheme,
)
from mosaic.core.pipeline.index import (
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.manifest import build_manifest
from mosaic.core.pipeline.reconcile import backup_index
from mosaic.core.pipeline.resolve import resolve_references
from mosaic.core.pipeline.run import (
    build_run_params_payload,
    compute_run_id,
    resolve_labels_variants,
)
from mosaic.core.pipeline.types import Result

if TYPE_CHECKING:
    from mosaic.core.pipeline.types import Feature

_FROM = "__from__"
_HASH_LEN = 10


@dataclass
class Run:
    """One feature run directory, as read off disk."""

    storage_name: str
    run_id: str
    run_root: Path
    doc: dict[str, object]
    # (feature_name, run_id) pairs; run_id is None for an unpinned reference,
    # which is recorded rather than dropped -- see _upstream_refs.
    upstream: tuple[tuple[str, str | None], ...]
    # filled in during processing
    verdict: str = "pending"
    new_run_id: str = ""
    reason: str = ""
    blocked_by: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return self.storage_name.split(_FROM, 1)[0]

    @property
    def frame_range(self) -> tuple[int | None, int | None]:
        raw = self.doc.get("_frame_range")
        pair: list[object] = (
            list(cast("list[object]", raw)) if isinstance(raw, list) else []
        )
        while len(pair) < 2:
            pair.append(None)
        start, end = pair[0], pair[1]
        return (
            start if isinstance(start, int) else None,
            end if isinstance(end, int) else None,
        )

    @property
    def recorded_version(self) -> str:
        """The version segment of ``<version>-<10 hex>``, or "" if malformed."""
        if len(self.run_id) < _HASH_LEN + 2 or self.run_id[-(_HASH_LEN + 1)] != "-":
            return ""
        return self.run_id[: -(_HASH_LEN + 1)]


def read_runs(ds: Dataset) -> list[Run]:
    """Every ``features/<storage>/<run_id>/`` holding a params.json."""
    root = ds.get_root("features")
    runs: list[Run] = []
    if not root.exists():
        return runs
    for storage_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in storage_dir.iterdir() if p.is_dir()):
            params_path = run_dir / "params.json"
            if not params_path.exists():
                continue
            try:
                doc = json.loads(params_path.read_text())
            except (OSError, ValueError) as exc:
                print(f"  ! unreadable {params_path}: {exc}", file=sys.stderr)
                continue
            runs.append(
                Run(
                    storage_name=storage_dir.name,
                    run_id=run_dir.name,
                    run_root=run_dir,
                    doc=doc,
                    upstream=tuple(_upstream_refs(doc)),
                )
            )
    return runs


def _upstream_refs(doc: dict[str, object]) -> list[tuple[str, str | None]]:
    """Every ``(feature, run_id)`` reference reachable from _inputs and _params.

    An **unpinned** reference -- ``"run_id": null`` -- is recorded with ``None``
    rather than dropped, and that is the whole point. Dropping it made the run
    look dependency-free: the upstream gate could not fire, ``_substitute`` had
    nothing to rewrite, and ``resolve_references`` then silently re-pinned it to
    today's ``latest_run_id()``, moving the artifact under a lineage it never
    had. Legacy runs are exactly the population that carries these, because they
    predate the scheme that made ``_inputs`` record a concrete upstream id.
    """
    found: list[tuple[str, str | None]] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            mapping = cast("dict[str, object]", node)
            feature = mapping.get("feature")
            if isinstance(feature, str) and feature and "run_id" in mapping:
                run_id = mapping.get("run_id")
                found.append((feature, run_id if isinstance(run_id, str) else None))
            for value in mapping.values():
                walk(value)
        elif isinstance(node, list):
            for value in cast("list[object]", node):
                walk(value)

    walk(doc.get("_inputs"))
    walk(doc.get("_params"))
    return found


def _substitute(node: object, remap: dict[tuple[str, str], str]) -> object:
    """Rewrite every pinned reference through *remap*, structurally."""
    if isinstance(node, dict):
        mapping = cast("dict[str, object]", node)
        out: dict[str, object] = {
            key: _substitute(value, remap) for key, value in mapping.items()
        }
        feature, run_id = out.get("feature"), out.get("run_id")
        if isinstance(feature, str) and isinstance(run_id, str):
            new = remap.get((feature, run_id))
            if new is not None:
                out["run_id"] = new
        return out
    if isinstance(node, list):
        return [_substitute(value, remap) for value in cast("list[object]", node)]
    return node


def _node(run: "Run") -> tuple[str, str]:
    """This run's graph key: ``(storage_name, run_id)``.

    Not the bare ``run_id``. ``compute_run_id``'s payload carries no feature-name
    term, so two different features can and do mint the same identifier -- and a
    graph keyed on the digest alone then places a downstream before its real
    upstream and hashes it against an empty remap.
    """
    return (run.storage_name, run.run_id)


def _resolve_ref(
    feature: str, run_id: str, index: dict[tuple[str, str], "Run"]
) -> tuple[str, str] | None:
    """The graph key a recorded reference names, or ``None`` if it names nothing here.

    A ``Result``'s ``feature`` is the *storage* name (``speed-angvel__from__tracks``),
    which is the key directly. A reference written with the bare slug is matched by
    slug as a fallback, and an ambiguous slug match is treated as unresolved rather
    than guessed at.
    """
    if (feature, run_id) in index:
        return (feature, run_id)
    by_slug = [
        key
        for key in index
        if key[1] == run_id and key[0].split(_FROM, 1)[0] == feature
    ]
    return by_slug[0] if len(by_slug) == 1 else None


def _topo_order(runs: list[Run]) -> list[Run]:
    """Upstream before downstream. Ties and cycles keep their read order."""
    index = {_node(run): run for run in runs}
    ordered: list[Run] = []
    placed: set[tuple[str, str]] = set()
    remaining = list(runs)
    while remaining:
        progressed = False
        deferred: list[Run] = []
        for run in remaining:
            pending = [
                key
                for feature, up_id in run.upstream
                if up_id
                and (key := _resolve_ref(feature, up_id, index)) is not None
                and key not in placed
                and key != _node(run)
            ]
            if pending:
                deferred.append(run)
                continue
            ordered.append(run)
            placed.add(_node(run))
            progressed = True
        if not progressed:  # a cycle: emit the rest in read order
            ordered.extend(deferred)
            break
        remaining = deferred
    return ordered


def _is_result(value: object) -> "TypeGuard[Result[str]]":
    """Narrow *value* to ``Result[str]``.

    A bare ``isinstance(value, Result)`` leaves the type argument unknown, so
    every ``.feature``/``.run_id`` access on it reads as unknown. ``Result``'s
    TypeVar is bound to ``str`` and defaults to it, so ``Result[str]`` is the
    only argument it can carry -- the same guard ``reconcile_features`` uses.
    """
    return isinstance(value, Result)


def _feature_class(slug: str) -> "type[Feature] | None":
    """The registered feature class behind a storage directory's slug."""
    for cls in FEATURES.values():
        if getattr(cls, "name", None) == slug:
            return cls
    return None


def _resolutions_payload(feature: "Feature") -> list[dict[str, str | None]]:
    """The ``_resolved`` list, in the order ``resolve_references`` produces it."""
    out: list[dict[str, str | None]] = []
    for position, item in enumerate(feature.inputs.root):
        if isinstance(item, Result) and item.feature:
            out.append(
                {
                    "where": f"inputs[{position}]",
                    "feature": item.feature,
                    "run_id": item.run_id,
                }
            )
    for name in type(feature.params).model_fields:
        value: object = getattr(feature.params, name)
        if _is_result(value) and value.feature:
            out.append(
                {
                    "where": f"params.{name}",
                    "feature": value.feature,
                    "run_id": value.run_id,
                }
            )
    return out


def process(ds: Dataset, apply: bool) -> tuple[list[Run], dict[Path, Path]]:
    all_runs = read_runs(ds)
    index = {_node(r): r for r in all_runs}
    runs = _topo_order(all_runs)
    remap: dict[tuple[str, str], str] = {}
    blocked: set[tuple[str, str]] = set()
    backed_up: dict[Path, Path] = {}

    for run in runs:
        # -- pin gate: recorded provenance must actually name its upstreams ----
        # An unpinned reference means "whichever run was latest when this ran",
        # which is not re-derivable. resolve_references would happily pin it to
        # today's latest and the move would look clean -- including under the
        # usual cache_hit check, since a live run resolves the same wrong way.
        unpinned = sorted({f for f, up_id in run.upstream if up_id is None})
        if unpinned:
            run.verdict, run.new_run_id = "unpinned-upstream", run.run_id
            run.reason = (
                f"recorded reference(s) {unpinned} carry no run_id; the provenance "
                f"does not say which upstream produced this. Recompute."
            )
            blocked.add(_node(run))
            continue

        # -- upstream gate: never move on top of an upstream that did not move --
        upstream_blocked = [
            key
            for feature, up_id in run.upstream
            if up_id
            and (key := _resolve_ref(feature, up_id, index)) is not None
            and key in blocked
        ]
        if upstream_blocked:
            run.verdict, run.new_run_id = "blocked", run.run_id
            _st, _rid = upstream_blocked[0]
            run.reason = f"upstream {_st}/{_rid} must be recomputed"
            blocked.add(_node(run))
            continue

        cls = _feature_class(run.slug)
        if cls is None:
            run.verdict, run.new_run_id = "skipped", run.run_id
            run.reason = f"feature {run.slug!r} is not in the registry"
            blocked.add(_node(run))
            continue

        # -- version gate: a declared recipe change is never a re-address --
        if run.recorded_version != getattr(cls, "version", ""):
            run.verdict, run.new_run_id = "version-bump", run.run_id
            run.reason = (
                f"recorded version {run.recorded_version!r} != "
                f"current {getattr(cls, 'version', '')!r}; recompute"
            )
            blocked.add(_node(run))
            continue

        # -- scope gate: a scope-dependent hash needs provenance legacy lacks --
        if getattr(cls, "scope_dependent", False):
            run.verdict, run.new_run_id = "scope-dependent", run.run_id
            run.reason = "hashes the sequence set, which legacy params.json omits"
            blocked.add(_node(run))
            continue

        # -- rebuild the feature with upstreams substituted ------------------
        doc_inputs = _substitute(run.doc.get("_inputs"), remap)
        doc_params = _substitute(run.doc.get("_params"), remap)
        inputs_cls = getattr(cls, "Inputs", None)
        if not (isinstance(inputs_cls, type) and issubclass(inputs_cls, BaseModel)):
            run.verdict, run.new_run_id = "skipped", run.run_id
            run.reason = f"feature {run.slug!r} has no Inputs model"
            blocked.add(_node(run))
            continue
        try:
            inputs = inputs_cls.model_validate(doc_inputs)
            feature = cls(inputs, doc_params)  # pyright: ignore[reportCallIssue]
        except Exception as exc:  # noqa: BLE001 - a bad run is skipped, not fatal
            run.verdict, run.new_run_id = "skipped", run.run_id
            run.reason = f"could not rebuild: {exc}"
            blocked.add(_node(run))
            continue

        # -- derive the address a live run_feature() would use ---------------
        # Same order as run_feature: pin, build the manifest, resolve labels,
        # hash. `on_missing_run="empty"` so a dry run does not raise on an
        # upstream whose index rows have not been restamped yet; it can only
        # affect `scope.entries`, which the scope gate above has excluded from
        # this hash.
        try:
            resolve_references(ds, feature)
            if feature.inputs.is_empty:
                scope = Scope()
            else:
                _, scope = build_manifest(
                    ds, feature.inputs, None, None, None, on_missing_run="empty"
                )
            scope.labels_variants = resolve_labels_variants(ds, feature, None)
            new_run_id, _ = compute_run_id(feature, *run.frame_range, scope)
        except Exception as exc:  # noqa: BLE001
            run.verdict, run.new_run_id = "skipped", run.run_id
            run.reason = f"could not derive new address: {exc}"
            blocked.add(_node(run))
            continue

        run.new_run_id = new_run_id
        if new_run_id == run.run_id:
            run.verdict = "current"
            run.reason = "already at its current address"
            if apply:
                write_identity_scheme(run.run_root, FEATURE_IDENTITY_SCHEME)
            continue

        new_root = feature_run_root(ds, run.storage_name, new_run_id)
        if new_root.exists():
            run.verdict = "skipped"
            run.reason = f"target {new_run_id} already exists"
            blocked.add(_node(run))
            continue

        run.verdict = "readdress"
        run.reason = f"{run.run_id} -> {new_run_id}"
        remap[(run.storage_name, run.run_id)] = new_run_id
        # A reference written with the bare slug resolves too, but only when that
        # slug is unambiguous at this digest -- otherwise the collision R2 guards
        # against would come back through the substitution instead of the graph.
        if (
            sum(
                1
                for st, rid in index
                if rid == run.run_id and st.split(_FROM, 1)[0] == run.slug
            )
            == 1
        ):
            remap[(run.slug, run.run_id)] = new_run_id

        if apply:
            index_path = feature_index_path(ds, run.storage_name)
            # Once per index, before the first row rewrite touches it -- the same
            # backup reconcile takes, and the only way back if the assertion this
            # script encodes turns out to be wrong.
            if index_path not in backed_up:
                backed_up[index_path] = backup_index(index_path)
            old_root = run.run_root
            shutil.move(str(run.run_root), str(new_root))
            write_identity_scheme(new_root, FEATURE_IDENTITY_SCHEME)
            payload = build_run_params_payload(
                feature, *run.frame_range, scope, _resolutions_payload(feature)
            )
            atomic_write(
                new_root / "params.json",
                lambda p, _payload=payload: p.write_text(
                    json.dumps(_payload, indent=2)
                ),
            )

            def _rewrite(
                stored: str, _old: Path = old_root, _new: Path = new_root
            ) -> str:
                """Re-anchor *stored* under the new run root, preserving its shape.

                Not ``_new / Path(stored).name``: a row may record the run
                *directory* itself rather than a file inside it -- ``run_feature``
                writes that for an Inputs-empty feature, whose single row is keyed
                ``__global__`` -- and keeping the basename would turn it into
                ``<new_root>/<old_run_id>``, a path that exists nowhere. Nothing
                catches that afterwards, because index reads do not validate paths.
                """
                try:
                    absolute = ds.resolve_path(stored)
                except Exception:  # noqa: BLE001 - a bad cell keeps the legacy shape
                    absolute = Path(stored)
                try:
                    rel = absolute.relative_to(_old)
                except ValueError:
                    rel = Path(absolute.name)
                if str(rel) in (".", ""):
                    return ds.relative_to_root(_new)
                return ds.relative_to_root(_new / rel)

            feature_index(index_path).remap_run_id(
                run.run_id, new_run_id, path_rewrite=_rewrite
            )
            run.run_root = new_root

    return runs, backed_up


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to dataset.yaml")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the renames. Default is a dry-run report.",
    )
    args = parser.parse_args()

    ds = Dataset(manifest_path=Path(args.manifest)).load()
    runs, backed_up = process(ds, apply=args.apply)

    order = [
        "readdress",
        "current",
        "version-bump",
        "unpinned-upstream",
        "scope-dependent",
        "blocked",
        "skipped",
    ]
    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'} — {len(runs)} feature run(s)\n")
    for verdict in order:
        group = [run for run in runs if run.verdict == verdict]
        if not group:
            continue
        print(f"{verdict}  ({len(group)})")
        for run in group:
            print(f"  {run.storage_name}/{run.run_id}")
            print(f"      {run.reason}")
        print()

    movable = sum(1 for run in runs if run.verdict == "readdress")
    recompute = sum(
        1
        for run in runs
        if run.verdict
        in {
            "version-bump",
            "unpinned-upstream",
            "scope-dependent",
            "blocked",
            "skipped",
        }
    )
    for original, backup in backed_up.items():
        print(f"index backed up: {original}  ->  {backup}")
    print(f"re-addressable: {movable}    must recompute: {recompute}")
    if not args.apply and movable:
        print("\nRe-run with --apply to perform the renames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
