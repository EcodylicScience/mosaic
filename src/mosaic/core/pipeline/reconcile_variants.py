"""The forward pass for op-addressed variants: tracks and labels.

A tracks variant is ``tracks/<op>.<version>-<digest>/`` and a label variant is
``labels/<kind>/<op>.<version>-<digest>/`` -- structurally the same thing: a
directory holding a sidecar ``params.json`` (the recipe) beside the tables it
names, with one index row per ``(group, sequence)`` all sharing the variant id.
Both are minted by :func:`op_run_id`, so both recompute the same way and share the
:class:`_VariantReconciler` base here; only enumeration, the per-op payload, and
which writer refreshes the sidecar differ.

The recompute keeps the recorded *version* and swaps in the freshly recomputed
*digest*, exactly as the feature pass does: a version bump is a new recipe, never a
re-address of old bytes. A digest that moves under an older scheme is the machinery
having changed -- a pure re-address. A digest that moves under the *current* scheme
is a recorded recipe not reproducing its own identifier, which is declined rather
than guessed at.

No current producer passes the ``upstream`` term (the reserved seam for a chained
or derived producer), so a variant's identity has no reconciled upstream and these
reconcilers declare none. When a derived producer lands, its sidecar must record
its upstream and this module must read it -- until then, reconstructing the payload
without one is correct for every variant on disk.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mosaic.core.pipeline.labels_identity import (
    LABELS_IDENTITY_SCHEME,
    label_convert_variant_payload,
    write_labels_variant,
)
from mosaic.core.pipeline.labels_index import (
    labels_index,
    labels_index_path,
    read_labels_index,
)
from mosaic.core.pipeline.op_identity import op_run_id, parse_op_run_id
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
from mosaic.core.pipeline.tracks_identity import (
    TRACKS_IDENTITY_SCHEME,
    convert_variant_payload,
    write_tracks_variant,
)
from mosaic.core.pipeline.tracks_index import (
    read_tracks_index,
    tracks_index,
    tracks_index_path,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


class _Remappable(Protocol):
    """The one index method a re-address needs, shared by the two typed indexes."""

    def remap_run_id(
        self,
        old_run_id: str,
        new_run_id: str,
        *,
        path_rewrite: Callable[[str], str] | None = None,
        dry_run: bool = False,
    ) -> int: ...


class _VariantSidecar(BaseModel):
    """A variant's ``params.json``, typed. ``kind`` is populated for labels only."""

    model_config = ConfigDict(extra="ignore")

    identity_scheme: str = ""
    op: str = ""
    version: str = ""
    kind: str = ""
    params: dict[str, object] = Field(default_factory=dict)
    observed: dict[str, str] = Field(default_factory=dict)


def _read_sidecar(path: Path) -> _VariantSidecar | None:
    if not path.exists():
        return None
    try:
        return _VariantSidecar.model_validate_json(path.read_text())
    except (OSError, ValidationError):
        return None


@dataclass(frozen=True)
class _VariantSite:
    """One variant on disk, and where its row and metadata live."""

    run_id: str
    variant_root: Path
    index_path: Path
    parent: Path  # the tracks root, or ``labels/<kind>`` -- where the variant sits
    kind: str = ""


def _classify_variant(
    recorded_run_id: str,
    new_run_id: str,
    recorded_scheme: str,
    current_scheme: str,
) -> tuple[Verdict, str, str]:
    """Classify a variant by comparing recorded and recomputed digests.

    The digest is machinery-governed; the version is a visible bump. A moved digest
    is honest only when the scheme moved (the machinery changed) -- there is no
    upstream to explain it otherwise, since no variant carries an ``upstream``
    term today -- so a digest that moved under the current scheme is declined.
    """
    old = parse_op_run_id(recorded_run_id)
    new = parse_op_run_id(new_run_id)
    if old is None or new is None:
        return (
            "unresolvable_pre_provenance",
            recorded_run_id,
            "identifier is not <op>.<version>-<digest>",
        )
    scheme_current = recorded_scheme == current_scheme
    if new.digest == old.digest:
        if scheme_current:
            return "ok", recorded_run_id, "identifier and scheme current"
        return (
            "scheme_stale",
            recorded_run_id,
            f"identifier unchanged; scheme {recorded_scheme or 'absent'!r} "
            f"-> {current_scheme!r}",
        )
    if scheme_current:
        return (
            "unresolvable_pre_provenance",
            new_run_id,
            "digest changed under the current scheme; recorded recipe does not "
            "reproduce its own identifier",
        )
    return (
        "identity_shift_relocatable",
        new_run_id,
        f"identity machinery moved ({recorded_run_id} -> {new_run_id}); "
        f"recipe confirmed unchanged",
    )


class _VariantReconciler:
    """Shared forward pass for tracks and label variants."""

    key: ReconcileKey
    depends_on: tuple[ReconcileKey, ...] = ()
    current_scheme: str

    def __init__(self, ds: Dataset) -> None:
        self._ds = ds

    # -- per-kind hooks ----------------------------------------------------

    def _sites(self) -> Iterable[_VariantSite]:
        raise NotImplementedError

    def _terms(self, sidecar: _VariantSidecar) -> dict[str, object]:
        raise NotImplementedError

    def _index(self, index_path: Path) -> _Remappable:
        raise NotImplementedError

    def _write_variant(
        self, site: _VariantSite, target_run_id: str, sidecar: _VariantSidecar
    ) -> None:
        raise NotImplementedError

    # -- shared flow -------------------------------------------------------

    def reconcile(
        self, state: ReconcileState, *, apply: bool, force: bool
    ) -> ReconcilePass:
        del force  # variants never delete; the destructive path is a later pass
        builder = PassBuilder()
        for site in self._sites():
            try:
                self._process(site, state, builder, apply=apply)
            except Exception as exc:  # noqa: BLE001 - a bad variant must not abort the sweep
                builder.error(f"{self.key}/{site.run_id}: {exc}")
        return builder.build()

    def _process(
        self,
        site: _VariantSite,
        state: ReconcileState,
        builder: PassBuilder,
        *,
        apply: bool,
    ) -> None:
        sidecar = _read_sidecar(site.variant_root / "params.json")
        if sidecar is None:
            state.record_blocked(self.key, site.run_id)
            builder.add(
                self._finding(
                    site,
                    "unresolvable_pre_provenance",
                    site.run_id,
                    "variant sidecar params.json missing or unreadable",
                    apply=apply,
                    action="reported",
                )
            )
            return
        try:
            new_run_id = op_run_id(sidecar.op, sidecar.version, self._terms(sidecar))
        except Exception as exc:  # noqa: BLE001 - a raising recompute is unresolvable, not fatal
            state.record_blocked(self.key, site.run_id)
            builder.add(
                self._finding(
                    site,
                    "unresolvable_pre_provenance",
                    site.run_id,
                    f"recompute failed: {exc}",
                    apply=apply,
                    action="reported",
                )
            )
            return

        verdict, address, reason = _classify_variant(
            site.run_id, new_run_id, sidecar.identity_scheme, self.current_scheme
        )
        if verdict == "identity_shift_relocatable":
            state.record_move(self.key, site.run_id, address)
        elif verdict in ("identity_shift_recompute", "unresolvable_pre_provenance"):
            state.record_blocked(self.key, site.run_id)

        finding = ReconcileFinding(
            key=self.key,
            verdict=verdict,
            old_address=site.run_id,
            new_address=address,
            index_path=site.index_path,
            run_root=str(site.variant_root),
            reason=reason,
        )
        if not apply:
            builder.add(finding)
            return
        builder.add(self._apply(site, sidecar, finding, builder))

    def _finding(
        self,
        site: _VariantSite,
        verdict: Verdict,
        address: str,
        reason: str,
        *,
        apply: bool,
        action: str,
    ) -> ReconcileFinding:
        finding = ReconcileFinding(
            key=self.key,
            verdict=verdict,
            old_address=site.run_id,
            new_address=address,
            index_path=site.index_path,
            run_root=str(site.variant_root),
            reason=reason,
        )
        return (
            finding.with_action("reported")
            if apply and action == "reported"
            else finding
        )

    def _apply(
        self,
        site: _VariantSite,
        sidecar: _VariantSidecar,
        finding: ReconcileFinding,
        builder: PassBuilder,
    ) -> ReconcileFinding:
        if finding.verdict == "scheme_stale":
            self._write_variant(site, site.run_id, sidecar)
            return finding.with_action("marker_refreshed")
        if finding.verdict != "identity_shift_relocatable":
            return finding.with_action("reported")
        return self._relocate(site, sidecar, finding, builder)

    def _relocate(
        self,
        site: _VariantSite,
        sidecar: _VariantSidecar,
        finding: ReconcileFinding,
        builder: PassBuilder,
    ) -> ReconcileFinding:
        ds = self._ds
        new_root = site.parent / finding.new_address
        if new_root.exists():
            builder.error(
                f"{self.key}: cannot re-address {site.run_id} -> "
                f"{finding.new_address}; target directory already exists"
            )
            return finding.with_action("none")
        builder.backed_up(backup_index(site.index_path))
        # Directory move first, index rewrite second: an atomic index write renames
        # a new inode over the path, so the row rewrite must be the only write held
        # under the lock. The sidecar rides inside the moved directory; rewriting it
        # refreshes the scheme marker to the current one.
        _ = shutil.move(str(site.variant_root), str(new_root))
        self._write_variant(site, finding.new_address, sidecar)

        def _rewrite(stored: str) -> str:
            return ds.relative_to_root(new_root / Path(stored).name)

        _ = self._index(site.index_path).remap_run_id(
            site.run_id, finding.new_address, path_rewrite=_rewrite
        )
        return finding.with_action("relocated")


def _tracks_terms(sidecar: _VariantSidecar) -> dict[str, object]:
    """The hashed payload behind a tracks variant, rebuilt from its sidecar.

    A converter records the inner params and wraps them at mint time, so it is
    re-wrapped here; every other producer (trex/sleap/litpose, inference) records
    the payload already assembled, so the sidecar params are it. See the mint sites
    in ``tracks_identity``'s callers.
    """
    if sidecar.op.startswith("convert-"):
        return convert_variant_payload(sidecar.params)
    return dict(sidecar.params)


def _labels_terms(sidecar: _VariantSidecar) -> dict[str, object]:
    """The hashed payload behind a label variant, rebuilt from its sidecar.

    A label converter records the inner params and the kind separately and wraps
    them at mint time, so both are re-wrapped here. No derived-label producer
    exists yet; when one does, its sidecar shape decides this branch.
    """
    return label_convert_variant_payload(sidecar.kind, sidecar.params)


class TracksReconciler(_VariantReconciler):
    """Recompute and re-address tracks variants. Registered under ``"tracks"``."""

    key: ReconcileKey = "tracks"
    current_scheme = TRACKS_IDENTITY_SCHEME

    def _sites(self) -> Iterable[_VariantSite]:
        if not self._ds.has_root("tracks"):
            return
        root = self._ds.get_root("tracks")
        if not root.exists():
            return
        index_path = tracks_index_path(self._ds)
        frame = read_tracks_index(self._ds)
        for run_id in sorted({str(value) for value in frame["run_id"] if str(value)}):
            yield _VariantSite(run_id, root / run_id, index_path, root)

    def _terms(self, sidecar: _VariantSidecar) -> dict[str, object]:
        return _tracks_terms(sidecar)

    def _index(self, index_path: Path) -> _Remappable:
        return tracks_index(index_path)

    def _write_variant(
        self, site: _VariantSite, target_run_id: str, sidecar: _VariantSidecar
    ) -> None:
        _ = write_tracks_variant(
            site.parent,
            target_run_id,
            sidecar.op,
            sidecar.version,
            sidecar.params,
            sidecar.observed,
        )


class LabelsReconciler(_VariantReconciler):
    """Recompute and re-address label variants. Registered under ``"labels"``."""

    key: ReconcileKey = "labels"
    current_scheme = LABELS_IDENTITY_SCHEME

    def _sites(self) -> Iterable[_VariantSite]:
        if not self._ds.has_root("labels"):
            return
        labels_root = self._ds.get_root("labels")
        if not labels_root.exists():
            return
        for kind_dir in sorted(labels_root.iterdir()):
            if not kind_dir.is_dir():
                continue
            index_path = labels_index_path(self._ds, kind_dir.name)
            if not index_path.exists():
                continue
            # read_labels_index projects onto the current schema, so a legacy flat
            # index with no run_id column reads as rows with an empty run_id (which
            # are not variants) rather than raising -- the generic IndexCSV.read
            # does not adopt, so it must not be used here.
            frame = read_labels_index(self._ds, kind_dir.name)
            if "run_id" not in frame.columns:
                continue
            for run_id in sorted({str(v) for v in frame["run_id"] if str(v)}):
                yield _VariantSite(
                    run_id, kind_dir / run_id, index_path, kind_dir, kind_dir.name
                )

    def _terms(self, sidecar: _VariantSidecar) -> dict[str, object]:
        return _labels_terms(sidecar)

    def _index(self, index_path: Path) -> _Remappable:
        return labels_index(index_path)

    def _write_variant(
        self, site: _VariantSite, target_run_id: str, sidecar: _VariantSidecar
    ) -> None:
        _ = write_labels_variant(
            site.parent,
            target_run_id,
            sidecar.op,
            sidecar.version,
            sidecar.kind or site.kind,
            sidecar.params,
            sidecar.observed,
        )


register_identity_reconciler("tracks", TracksReconciler)
register_identity_reconciler("labels", LabelsReconciler)
