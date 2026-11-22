"""What the writers of the tracks and labels indexes share.

Both indexes take their rows in batches of drafts, and both can index the files of a
variant that is already on disk. This module holds the steps the two do alike: naming
the entry of each draft, keeping the last row of each key, and the refusals that
precede an in-place indexing call. The tracks and labels modules import it and it
imports neither, so the two stay apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from mosaic.core.entry import (
    Entry,
    make_entry_key,
    parse_entry_key,
    validate_entry_name,
)
from mosaic.core.helpers import text_cell
from mosaic.core.pipeline.op_identity import OpRunId, parse_op_run_id
from mosaic.core.pipeline.tracks_identity import VariantSidecar

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "KeyedRow",
    "check_variant_sidecar",
    "keep_last_of_each_key",
    "parse_variant_run_id",
    "refuse_rows_with_inputs",
    "refuse_unnamed_entry",
    "validated_entries",
]


class KeyedRow(Protocol):
    """A row of an index keyed by ``(run_id, group, sequence)``."""

    @property
    def run_id(self) -> str: ...

    @property
    def group(self) -> str: ...

    @property
    def sequence(self) -> str: ...


def validated_entries(pairs: Iterable[tuple[object, object]]) -> list[Entry]:
    """The validated ``(group, sequence)`` of each of *pairs*.

    Validated before anything else, because the composition is looked up by the
    same key the row is written under: a lookup on the raw value would miss for the
    caller that reads its group off a pandas Series as a numpy scalar.
    """
    return [
        (
            validate_entry_name(text_cell(group), "group"),
            validate_entry_name(text_cell(sequence), "sequence"),
        )
        for group, sequence in pairs
    ]


def keep_last_of_each_key[RowT: KeyedRow](rows: Sequence[RowT]) -> list[RowT]:
    """The rows that no later row of the same ``(run_id, group, sequence)`` replaces.

    The result is the index that appending *rows* one after another gives: the last
    row of a key survives, at its place in the order.
    """
    last_of = {
        (row.run_id, row.group, row.sequence): position
        for position, row in enumerate(rows)
    }
    survivors = set(last_of.values())
    return [row for position, row in enumerate(rows) if position in survivors]


def parse_variant_run_id(run_id: str) -> OpRunId:
    """Split *run_id* into the op that produced it, the kind of its producer.

    Raises:
        ValueError: If *run_id* is not an op run id.
    """
    parsed = parse_op_run_id(run_id)
    if parsed is None:
        raise ValueError(
            f"{run_id!r} is not an op run id, so the kind of its producer is unknown."
        )
    return parsed


def check_variant_sidecar(
    root: Path, parsed: OpRunId, sidecar: VariantSidecar | None
) -> VariantSidecar:
    """*sidecar*, the record of the variant at *root*, checked against its run id.

    Raises:
        ValueError: If there is no readable sidecar, or its op is not the kind of
            the run id *parsed* split.
    """
    if sidecar is None:
        raise ValueError(f"{root} has no readable params.json.")
    if sidecar.op != parsed.kind:
        raise ValueError(
            f"The sidecar of {root} records op {sidecar.op!r}, "
            f"and the run id names {parsed.kind!r}."
        )
    return sidecar


def refuse_rows_with_inputs(
    held: pd.DataFrame, run_id: str, cells: Sequence[str], index_name: str
) -> None:
    """Refuse to replace the rows of *run_id* that record where they came from.

    Such a row comes from a conversion, and replacing it would drop the inputs it
    records.

    Args:
        held: The rows of the index that name *run_id*.
        run_id: The variant to index.
        cells: The columns that only a conversion fills.
        index_name: How a refusal names the index, such as ``"tracks index"``.

    Raises:
        ValueError: If a row of *held* has a filled cell among *cells*.
    """
    if any(held[cell].ne("").any() for cell in cells):
        raise ValueError(
            f"The {index_name} holds rows of {run_id!r} that record a source or a "
            "consumed composition. Indexing the files again would replace them "
            "and drop the inputs that they record."
        )


def refuse_unnamed_entry(path: Path, group: str, sequence: str) -> None:
    """Refuse a file that holds another entry than its name gives.

    Raises:
        ValueError: If ``make_entry_key(group, sequence)`` is not the stem of *path*.
    """
    if make_entry_key(group, sequence) != path.stem:
        group_named, sequence_named = parse_entry_key(path.stem)
        raise ValueError(
            f"{path} holds the entry ({group!r}, {sequence!r}), and its name "
            f"gives ({group_named!r}, {sequence_named!r})."
        )
