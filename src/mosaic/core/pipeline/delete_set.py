"""The scoped delete set -- item 6.4, and what it refuses to touch.

P4's override: a source change blocks while derivatives exist, and forcing it
deletes the ones that became *wrong*. This computes that set from item 6.1's
walk and applies it, dry-run first.

**Not built on ``Pipeline.clean``.** The item calls that "the right primitive",
and against the code it is not: ``clean`` enumerates the steps of one ``Pipeline``
object, so an artifact produced from a notebook, the CLI or a queue job is
invisible to it -- exactly the set P4's placement clause exists to cover -- and it
deletes whole run directories where this needs per ``(run_id, group, sequence)``.
What is worth keeping is its *safeguards*, and those are here: nothing outside
the dataset's roots is ever unlinked, and every removal is checked against the
set that was previewed.

**Three states, not two.** A reached artifact is a candidate, or it is
*declined* with a reason. Declines are reported rather than silently omitted,
because "would delete 0" reads as an invitation to pass ``apply`` when the truth
may be "would delete 0, having refused 4".

**What can never appear, and why it is structural.** Extracted frames and
converted labels are P4 carve-outs. Neither is filtered out here: item 6.1's walk
does not enumerate them at all, so there is no branch to forget. A frame set
carrying annotations teaches a model what a subject looks like, which stays true
across a rearrangement; converted labels are remapped rather than deleted, and
until that remap exists a reorder over them is blocked at the gesture instead.

**A scope-dependent run is deleted whole or not at all.** Its state was fitted
over a set of entries, so removing one output leaves the rest describing a fit
that included it -- a directory whose contents come from two different fits, with
nothing on disk saying so. Detected through the fit scope item 5.3 records, and
declined rather than half-applied when the reached entries are only part of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .fit_scope import read_fit_scope
from .index import feature_index, feature_index_path
from .index import feature_run_root
from .provenance import index_records, reached_by
from .tracks_index import tracks_index, tracks_index_path

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "DeleteCandidate",
    "DeleteSetReport",
    "Declined",
    "delete_set",
]


@dataclass(frozen=True, slots=True)
class DeleteCandidate:
    """One artifact a source change made wrong."""

    kind: str
    name: str
    run_id: str
    group: str
    sequence: str
    abs_path: str
    via: str

    @property
    def entry(self) -> tuple[str, str]:
        return (self.group, self.sequence)


@dataclass(frozen=True, slots=True)
class Declined:
    """A reached artifact this refuses to delete, and why."""

    kind: str
    name: str
    run_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class DeleteSetReport:
    """What would be deleted, what was refused, and what actually went."""

    applied: bool
    candidates: tuple[DeleteCandidate, ...]
    declined: tuple[Declined, ...]
    removed: tuple[str, ...] = ()

    @property
    def considered(self) -> int:
        """How many reached artifacts were classified at all."""
        return len(self.candidates) + len(self.declined)


def delete_set(
    ds: Dataset,
    changed: Iterable[tuple[str, str]],
    root: str,
    *,
    apply: bool = False,
) -> DeleteSetReport:
    """The artifacts a change to *root* under *changed* made wrong.

    Dry-run unless *apply*. Deletes per ``(run_id, group, sequence)``: the output
    file, then the index row that named it.

    Only ``drifted`` is a candidate. A reached artifact whose recorded and current
    compositions still agree is current; one whose verdict is ``unknown`` is
    declined, because an absent record is not evidence of change and deleting on
    it would be deleting on a guess.
    """
    reached = reached_by(ds, changed, root)
    if reached.empty:
        return DeleteSetReport(applied=False, candidates=(), declined=())

    candidates: list[DeleteCandidate] = []
    declined: list[Declined] = []
    for record in index_records(reached):
        verdict = str(record.get("verdict", ""))
        kind = str(record.get("kind", ""))
        name = str(record.get("name", ""))
        run_id = str(record.get("run_id", ""))
        if verdict == "current":
            continue
        if verdict == "unknown":
            declined.append(
                Declined(
                    kind=kind,
                    name=name,
                    run_id=run_id,
                    reason=(
                        "nothing records what it was built from, so a change "
                        "cannot be shown to have reached it"
                    ),
                )
            )
            continue
        candidates.append(
            DeleteCandidate(
                kind=kind,
                name=name,
                run_id=run_id,
                group=str(record.get("group", "")),
                sequence=str(record.get("sequence", "")),
                abs_path=str(record.get("abs_path", "")),
                via=str(record.get("via", "")),
            )
        )

    candidates, partial = _decline_partial_fits(ds, candidates)
    declined.extend(partial)

    if not apply or not candidates:
        return DeleteSetReport(
            applied=False,
            candidates=tuple(candidates),
            declined=tuple(declined),
        )

    removed = _apply(ds, candidates)
    return DeleteSetReport(
        applied=True,
        candidates=tuple(candidates),
        declined=tuple(declined),
        removed=tuple(removed),
    )


def _decline_partial_fits(
    ds: Dataset, candidates: Sequence[DeleteCandidate]
) -> tuple[list[DeleteCandidate], list[Declined]]:
    """Refuse a scope-dependent run whose reached entries are only part of its fit.

    Deleting some of them would leave the rest describing a fit that included what
    is gone, and nothing on disk would say so. Whole or not at all.
    """
    kept: list[DeleteCandidate] = []
    declined: list[Declined] = []
    by_run: dict[tuple[str, str], list[DeleteCandidate]] = {}
    for candidate in candidates:
        if candidate.kind != "features":
            kept.append(candidate)
            continue
        by_run.setdefault((candidate.name, candidate.run_id), []).append(candidate)

    for (name, run_id), group in by_run.items():
        scope = read_fit_scope(feature_run_root(ds, name, run_id))
        if scope is None or not scope.scope_dependent:
            kept.extend(group)
            continue
        fitted = set(scope.entries)
        reached = {candidate.entry for candidate in group}
        if fitted <= reached:
            kept.extend(group)
            continue
        declined.append(
            Declined(
                kind="features",
                name=name,
                run_id=run_id,
                reason=(
                    f"fitted over {len(fitted)} entries and only {len(reached)} "
                    f"are reached; deleting part of a scope-dependent run leaves "
                    f"the rest describing a fit that included what went"
                ),
            )
        )
    return kept, declined


def _apply(ds: Dataset, candidates: Sequence[DeleteCandidate]) -> list[str]:
    """Unlink each candidate and drop the row that named it.

    Rows first, then files, so a crash between the two leaves rows naming files
    that are gone -- which a reconcile removes -- rather than files nothing names,
    which nothing finds.
    """
    _assert_inside_roots(ds, candidates)

    for (kind, name, run_id), group in _grouped(candidates).items():
        entries = [candidate.entry for candidate in group]
        if kind == "features":
            index = feature_index(feature_index_path(ds, name))
        else:
            index = tracks_index(tracks_index_path(ds))
        _ = index.drop_entries(entries, run_id=run_id)

    removed: list[str] = []
    for candidate in candidates:
        if not candidate.abs_path:
            continue
        path = ds.resolve_path(candidate.abs_path)
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def _grouped(
    candidates: Sequence[DeleteCandidate],
) -> dict[tuple[str, str, str], list[DeleteCandidate]]:
    grouped: dict[tuple[str, str, str], list[DeleteCandidate]] = {}
    for candidate in candidates:
        key = (candidate.kind, candidate.name, candidate.run_id)
        grouped.setdefault(key, []).append(candidate)
    return grouped


def _assert_inside_roots(ds: Dataset, candidates: Sequence[DeleteCandidate]) -> None:
    """Refuse to unlink anything outside the roots this may delete from.

    ``Pipeline.clean``'s safeguard, kept: it raises rather than skipping, because
    a candidate pointing outside is evidence the set was computed wrongly and the
    rest of it cannot be trusted either.
    """
    allowed: list[Path] = []
    for key in ("features", "tracks"):
        try:
            allowed.append(ds.get_root(key).resolve())
        except (KeyError, OSError):
            continue
    for candidate in candidates:
        if not candidate.abs_path:
            continue
        resolved = ds.resolve_path(candidate.abs_path).resolve()
        if not any(root == resolved or root in resolved.parents for root in allowed):
            message = (
                f"refusing to delete {resolved}: it is outside this dataset's "
                f"features and tracks roots, so the delete set was computed "
                f"wrongly and none of it can be trusted"
            )
            raise RuntimeError(message)


def to_frame(report: DeleteSetReport) -> pd.DataFrame:
    """The report's candidates as a frame, for display."""
    return pd.DataFrame(
        [
            {
                "kind": candidate.kind,
                "name": candidate.name,
                "run_id": candidate.run_id,
                "group": candidate.group,
                "sequence": candidate.sequence,
                "via": candidate.via,
            }
            for candidate in report.candidates
        ],
        columns=pd.Index(["kind", "name", "run_id", "group", "sequence", "via"]),
    )
