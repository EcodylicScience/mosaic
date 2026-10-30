"""The leading release numbers of a tool's version string."""

from __future__ import annotations

import pytest

from mosaic.tracking.common.release import release_numbers


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.3.1", (2, 3, 1)),
        ("0.3.5rc1", (0, 3, 5)),
        ("2.3.1.post1", (2, 3, 1)),
        ("3.0", (3, 0)),
        ("", ()),
        ("dev", ()),
    ],
)
def test_release_numbers_reads_the_leading_numbers(
    version: str, expected: tuple[int, ...]
) -> None:
    assert release_numbers(version) == expected
