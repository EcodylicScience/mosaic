"""Byte counts as the decimal sizes the CLI prints."""

from __future__ import annotations

import pytest

from mosaic.byte_sizes import format_size


@pytest.mark.parametrize(
    ("size", "text"),
    [
        (0, "0 B"),
        (999, "999 B"),
        (1_000, "1.0 KB"),
        (1_500, "1.5 KB"),
        (12_613_112, "12.6 MB"),
        (999_000_000, "999.0 MB"),
        (1_000_000_000, "1.0 GB"),
        (1_900_000_000, "1.9 GB"),
        (5_000_000_000_000, "5000.0 GB"),
    ],
)
def test_a_size_is_in_the_largest_decimal_unit_below_a_thousand(
    size: int, text: str
) -> None:
    assert format_size(size) == text
