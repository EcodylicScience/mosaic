"""The summary keys: the columns that name a bin or an arena."""

from __future__ import annotations

from mosaic.core.pipeline.types.data_config import (
    ALIGN_COLS,
    BIN_END,
    BIN_START,
    NON_PAYLOAD_COLS,
    SUMMARY_KEYS,
)
from mosaic.core.schema import ARENA_COLUMN


def test_the_summary_keys_are_the_bin_edges_and_the_arena() -> None:
    assert (BIN_START, BIN_END) == ("bin_start", "bin_end")
    assert SUMMARY_KEYS == frozenset({BIN_START, BIN_END, ARENA_COLUMN})


def test_a_summary_key_stays_payload_and_out_of_the_alignment_columns() -> None:
    assert not SUMMARY_KEYS & ALIGN_COLS
    assert not SUMMARY_KEYS & NON_PAYLOAD_COLS
