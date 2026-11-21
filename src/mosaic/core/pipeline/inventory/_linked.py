"""Join the records of one artifact that a dataset and the datasets it links hold.

A builder runs on each dataset of a lookup with its own signature, so each record
it returns is measured over that dataset's files and target. The join keeps every
dataset's evidence, measures it over the reading dataset's target, and derives
the status again through :func:`~.model.reclassify`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .model import AnyRecord, Coverage, merged_coverage, reclassify

if TYPE_CHECKING:
    from mosaic.core.entry import Entry

    from .model import ArtifactRef
    from .params import RunParamsRead

__all__ = [
    "LinkedPart",
    "Remeasure",
    "grouped_by_ref",
    "merged_record",
    "nearest_rows_present",
]


type LinkedPart[RecordT: AnyRecord] = tuple[str, RecordT]
"""One dataset's record of an artifact, with the link it was read through.

The link id is ``""`` for the reading dataset's own record.
"""


@dataclass(frozen=True, slots=True)
class Remeasure:
    """What the reading dataset measures again for an artifact, in place of a part's.

    Attributes:
        target: The keys the reading dataset wants of it, or ``None`` for the
            union of the parts' targets, which a kind whose target is its own
            rows takes.
        drift: Its drifted entries over the reading dataset, or ``None`` for the
            union of the parts'.
        params: Its run parameters as the reading dataset reads them, or
            ``None`` for the parameters of the record kept.
    """

    target: frozenset[object] | None = None
    drift: tuple[Entry, ...] | None = None
    params: RunParamsRead | None = None


def grouped_by_ref[RecordT: AnyRecord](
    datasets: Sequence[tuple[str, Sequence[RecordT]]],
) -> list[list[LinkedPart[RecordT]]]:
    """Group the records of each dataset by artifact, in the order first met.

    Args:
        datasets: One ``(link_id, records)`` per dataset in lookup order, the
            reading dataset first with ``""``.

    Returns:
        One list per artifact, holding each dataset's record of it in lookup
        order.
    """
    groups: dict[ArtifactRef, list[LinkedPart[RecordT]]] = {}
    for link_id, records in datasets:
        for record in records:
            groups.setdefault(record.ref, []).append((link_id, record))
    return list(groups.values())


def nearest_rows_present(
    parts: Sequence[LinkedPart[AnyRecord]],
) -> list[LinkedPart[AnyRecord]]:
    """Each part with the present keys that a nearer part holds a row of removed.

    A key then counts as present when the nearest dataset with a row of it has
    its file, the dataset whose row a read of the key selects.

    Args:
        parts: Each dataset's record of the artifact in lookup order.

    Returns:
        The parts in the same order, each with its coverage's present keys
        narrowed.
    """
    nearer: set[object] = set()
    narrowed: list[LinkedPart[AnyRecord]] = []
    for link_id, record in parts:
        present = frozenset(key for key in record.coverage.present if key not in nearer)
        narrowed.append(
            (
                link_id,
                replace(record, coverage=replace(record.coverage, present=present)),
            )
        )
        nearer.update(record.rows)
    return narrowed


def merged_record(
    parts: Sequence[LinkedPart[AnyRecord]],
    *,
    fills_linked_only: bool,
    remeasure: Remeasure | None = None,
) -> AnyRecord:
    """Join the records of one artifact into the record the reading dataset lists.

    The own record is kept when its run directory exists, or when its kind has
    none. Otherwise the first linked record and its link id are kept. The kept
    record gives the joined record its ``run_root``, index path, identity scheme
    and, unless remeasured, its parameters. The rows, the present keys, the
    orphans and the drift are joined across the parts: a row is an orphan when
    no dataset holds its key, and a file when no dataset has a row of it. A key
    is unfinished when no dataset finished it.

    Args:
        parts: Each dataset's record of the artifact in lookup order, at least
            one.
        fills_linked_only: Whether the record names the keys only a linked
            dataset holds, as ``linked_only``.
        remeasure: What the reading dataset measures again, or ``None`` for
            nothing.

    Returns:
        The joined record, its status derived over the reading target.

    Raises:
        ValueError: *parts* is empty.
    """
    if not parts:
        raise ValueError("merged_record needs at least one record")
    measured = remeasure if remeasure is not None else Remeasure()
    own = next((record for link_id, record in parts if not link_id), None)
    linked = [(link_id, record) for link_id, record in parts if link_id]
    if own is not None and (
        own.run_root is None or own.run_root.exists() or not linked
    ):
        link_id, base = "", own
    else:
        link_id, base = linked[0]

    records = [record for _, record in parts]
    rows: set[object] = set()
    present: set[object] = set()
    targets: set[object] = set()
    orphan_rows: set[object] = set()
    orphan_files: set[object] = set()
    finished_keys: set[object] = set()
    drift: set[Entry] = set()
    extra: dict[str, set[str]] = {}
    for record in records:
        rows |= record.rows
        present |= record.coverage.present
        targets |= record.coverage.target
        orphan_rows |= record.orphan_rows
        orphan_files |= record.orphan_files
        finished_keys |= record.rows - record.unfinished_rows
        drift |= set(record.drift)
        for key, values in record.extra.items():
            extra.setdefault(key, set()).update(values)
    target = frozenset(targets) if measured.target is None else measured.target
    coverage = merged_coverage(
        (
            Coverage[object](
                target=frozenset(record.coverage.target),
                present=frozenset(record.coverage.present),
                covers_all=record.coverage.covers_all,
            )
            for record in records
        ),
        target,
    )
    own_present: frozenset[object] = (
        frozenset() if own is None else frozenset(own.coverage.present)
    )
    joined = replace(
        base,
        coverage=coverage,
        link_id=link_id,
        rows=frozenset(rows),
        orphan_rows=frozenset(orphan_rows - present),
        orphan_files=frozenset(orphan_files - rows),
        unfinished_rows=frozenset(rows - finished_keys),
        drift=tuple(sorted(drift)) if measured.drift is None else measured.drift,
        started_at=next(
            (record.started_at for record in records if record.started_at), ""
        ),
        finished_at=next(
            (record.finished_at for record in records if record.finished_at), ""
        ),
        extra={key: frozenset(values) for key, values in extra.items()},
        linked_only=frozenset(coverage.present - own_present)
        if fills_linked_only
        else frozenset(),
    )
    read = measured.params
    if read is not None:
        joined = replace(
            joined,
            params=read.params,
            params_state=read.state,
            upstreams=tuple(sorted(read.params.consumed_run_ids()))
            if read.params is not None
            else (),
        )
    return replace(joined, status=reclassify(joined, coverage))
