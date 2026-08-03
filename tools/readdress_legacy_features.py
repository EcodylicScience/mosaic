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
* Anything downstream of a run it refused, because that downstream's identity
  pins an upstream id that is about to move under it.
* Any ``scope_dependent`` feature. Those hash the set of sequences in scope,
  which legacy ``params.json`` does not record, so the address would be a guess.

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
    upstream: tuple[tuple[str, str], ...]  # (feature_name, run_id) pairs
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


def _upstream_refs(doc: dict[str, object]) -> list[tuple[str, str]]:
    """Every pinned ``(feature, run_id)`` reachable from _inputs and _params."""
    found: list[tuple[str, str]] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            mapping = cast("dict[str, object]", node)
            feature, run_id = mapping.get("feature"), mapping.get("run_id")
            if isinstance(feature, str) and isinstance(run_id, str) and feature:
                found.append((feature, run_id))
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


def _topo_order(runs: list[Run]) -> list[Run]:
    """Upstream before downstream. Ties and cycles keep their read order."""
    by_id = {run.run_id: run for run in runs}
    ordered: list[Run] = []
    placed: set[str] = set()
    remaining = list(runs)
    while remaining:
        progressed = False
        deferred: list[Run] = []
        for run in remaining:
            pending = [
                up_id
                for _, up_id in run.upstream
                if up_id in by_id and up_id not in placed and up_id != run.run_id
            ]
            if pending:
                deferred.append(run)
                continue
            ordered.append(run)
            placed.add(run.run_id)
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
    runs = _topo_order(read_runs(ds))
    remap: dict[tuple[str, str], str] = {}
    blocked: set[str] = set()
    backed_up: dict[Path, Path] = {}

    for run in runs:
        # -- upstream gate: never move on top of an upstream that did not move --
        upstream_blocked = [up_id for _, up_id in run.upstream if up_id in blocked]
        if upstream_blocked:
            run.verdict, run.new_run_id = "blocked", run.run_id
            run.reason = f"upstream {upstream_blocked[0]} must be recomputed"
            blocked.add(run.run_id)
            continue

        cls = _feature_class(run.slug)
        if cls is None:
            run.verdict, run.new_run_id = "skipped", run.run_id
            run.reason = f"feature {run.slug!r} is not in the registry"
            blocked.add(run.run_id)
            continue

        # -- version gate: a declared recipe change is never a re-address --
        if run.recorded_version != getattr(cls, "version", ""):
            run.verdict, run.new_run_id = "version-bump", run.run_id
            run.reason = (
                f"recorded version {run.recorded_version!r} != "
                f"current {getattr(cls, 'version', '')!r}; recompute"
            )
            blocked.add(run.run_id)
            continue

        # -- scope gate: a scope-dependent hash needs provenance legacy lacks --
        if getattr(cls, "scope_dependent", False):
            run.verdict, run.new_run_id = "scope-dependent", run.run_id
            run.reason = "hashes the sequence set, which legacy params.json omits"
            blocked.add(run.run_id)
            continue

        # -- rebuild the feature with upstreams substituted ------------------
        doc_inputs = _substitute(run.doc.get("_inputs"), remap)
        doc_params = _substitute(run.doc.get("_params"), remap)
        inputs_cls = getattr(cls, "Inputs", None)
        if not (isinstance(inputs_cls, type) and issubclass(inputs_cls, BaseModel)):
            run.verdict, run.new_run_id = "skipped", run.run_id
            run.reason = f"feature {run.slug!r} has no Inputs model"
            blocked.add(run.run_id)
            continue
        try:
            inputs = inputs_cls.model_validate(doc_inputs)
            feature = cls(inputs, doc_params)  # pyright: ignore[reportCallIssue]
        except Exception as exc:  # noqa: BLE001 - a bad run is skipped, not fatal
            run.verdict, run.new_run_id = "skipped", run.run_id
            run.reason = f"could not rebuild: {exc}"
            blocked.add(run.run_id)
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
            blocked.add(run.run_id)
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
            blocked.add(run.run_id)
            continue

        run.verdict = "readdress"
        run.reason = f"{run.run_id} -> {new_run_id}"
        remap[(run.slug, run.run_id)] = new_run_id
        remap[(run.storage_name, run.run_id)] = new_run_id

        if apply:
            index_path = feature_index_path(ds, run.storage_name)
            # Once per index, before the first row rewrite touches it -- the same
            # backup reconcile takes, and the only way back if the assertion this
            # script encodes turns out to be wrong.
            if index_path not in backed_up:
                backed_up[index_path] = backup_index(index_path)
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
            feature_index(index_path).remap_run_id(
                run.run_id,
                new_run_id,
                path_rewrite=lambda stored: ds.relative_to_root(
                    new_root / Path(stored).name
                ),
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
        if run.verdict in {"version-bump", "scope-dependent", "blocked", "skipped"}
    )
    for original, backup in backed_up.items():
        print(f"index backed up: {original}  ->  {backup}")
    print(f"re-addressable: {movable}    must recompute: {recompute}")
    if not args.apply and movable:
        print("\nRe-run with --apply to perform the renames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
