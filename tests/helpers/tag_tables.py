"""Tag tables in the shape the ``tags`` series saves them.

The tests of the tag readers, the import and the summaries each need a dataset
holding a revision of known sequence tags. One builder keeps the table's shape,
and the write that saves it, in one place.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from mosaic.core.annotations.projection import write_tag_table_revision
from mosaic.core.annotations.tag_table import (
    SequenceTags,
    TagDefinition,
    TagDefinitions,
    TagTable,
)
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.label_series import TAGS_SERIES
from mosaic.core.pipeline.label_series_index import SeriesRevision, series_index_path
from mosaic.core.typed_attribute import TypedAttributeType, TypedAttributeValue

__all__ = ["index_a_second_origin", "tag_table", "write_tags"]


def tag_table(
    sequence_tags: Mapping[Entry, Mapping[str, TypedAttributeValue]],
    types: Mapping[str, TypedAttributeType],
) -> TagTable:
    """A table of sequence tags alone, with no individuals and no arenas.

    Args:
        sequence_tags: The tags of each entry, keyed by ``(group, sequence)``.
        types: The type of each sequence tag. Every tag that an entry carries
            is named here.
    """
    definitions = TagDefinitions(
        sequence=tuple(
            TagDefinition(name=name, type=types[name]) for name in sorted(types)
        ),
        individual=(),
    )
    sequences = tuple(
        SequenceTags(
            group=group,
            sequence=sequence,
            tags=dict(tags),
            individuals={},
            arenas=(),
        )
        for (group, sequence), tags in sorted(sequence_tags.items())
    )
    return TagTable(definitions=definitions, sequences=sequences)


def write_tags(
    ds: Dataset,
    table: TagTable,
    origin: Mapping[str, JsonValue] | None = None,
) -> SeriesRevision:
    """Save *table* as the next revision of the dataset's ``tags`` series."""
    return write_tag_table_revision(ds, table=table, origin=origin or {})


def index_a_second_origin(ds: Dataset) -> str:
    """Index every revision of the ``tags`` series again as saved by another dataset.

    Returns:
        The ``origin_uuid`` of the added rows.
    """
    path = series_index_path(ds, TAGS_SERIES)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    other = rows.copy()
    other["origin_uuid"] = "other-origin"
    pd.concat([rows, other]).to_csv(path, index=False)
    return "other-origin"
