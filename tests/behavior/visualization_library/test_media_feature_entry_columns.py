"""The media features find an entry's video or tracks by its sequence.

An input without ``sequence`` is refused before apply, naming the column. The
group is read where present, and a dataset without groups names its entries
with an empty one.
"""

from __future__ import annotations

import pytest

from mosaic.core.pipeline.column_contract import (
    MissingInputColumns,
    check_input_columns,
)
from mosaic.core.pipeline.graph import build_feature

_SEGMENT: tuple[str, ...] = (
    "id1",
    "id2",
    "perspective",
    "interaction_id",
    "interaction_start",
    "interaction_end",
)

COLUMNS: dict[str, tuple[str, ...]] = {
    "egocentric-crop": ("frame", "id", "X", "Y"),
    "interaction-crop-pipeline": ("frame", "id", "X", "Y", *_SEGMENT),
    "overlay": ("frame", "id"),
}
"""Each feature's other required columns, with ``group`` and ``sequence`` left out."""


@pytest.mark.parametrize("feature", sorted(COLUMNS))
def test_an_input_without_its_sequence_is_refused_before_apply(feature: str) -> None:
    built = build_feature(feature, ["tracks"], {})

    check_input_columns(built, [*COLUMNS[feature], "sequence"])
    with pytest.raises(MissingInputColumns, match="lacks 'sequence'"):
        check_input_columns(built, COLUMNS[feature])
